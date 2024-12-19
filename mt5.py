import torch
from transformers import MT5Tokenizer, MT5ForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
from collections import Counter
import argparse
import os
import time
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTHONUNBUFFERED"] = "1"

def format_input(item, format_type='language_first'):
    components = {
        'language': f"language: {item['language']}",
        'site': f"site: {item['site']}",
        'claim': f"claim: {item['claim']}",
        'evidence': "",
        'claimant': f"claimant: {item.get('claimant', '')}",
        'claimDate': f"claimDate: {item.get('claimDate', '')}",
        'reviewDate': f"reviewDate: {item.get('reviewDate', '')}"
    }
    
    filtered_components = {k: v for k, v in components.items() 
                         if v and not v.endswith(": ")}
    
    for i in range(1, 6):
        ev_key = f'evidence_{i}'
        if ev_key in item and item[ev_key]:
            components['evidence'] += f"evidence_{i}: {item[ev_key]} "
    
    if format_type == 'language_first':
        text = f"{filtered_components['language']} {filtered_components['site']} "
        if 'claimant' in filtered_components:
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components:
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components:
            text += f"{filtered_components['reviewDate']} "
        text += f"{filtered_components['claim']} {components['evidence']}"
    elif format_type == 'claim_first':
        text = f"{filtered_components['claim']} "
        if 'claimant' in filtered_components:
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components:
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components:
            text += f"{filtered_components['reviewDate']} "
        text += f"{filtered_components['language']} {filtered_components['site']} {components['evidence']}"
    else:  # evidence_first
        text = f"{components['evidence']}"
        text += f"{filtered_components['language']} {filtered_components['site']} "
        if 'claimant' in filtered_components:
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components:
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components:
            text += f"{filtered_components['reviewDate']} "
        text += filtered_components['claim']
    
    return text.strip()

class XFACTDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.input_format = config.input_format
        
        self.label_map = {
            'false': 0,
            'mostly_false': 1,
            'partly_true': 2,
            'mostly_true': 3,
            'true': 4,
            'unverifiable': 5,
            'other': 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = format_input(item, self.input_format)

        # MT5 TOKENIZATION!
        inputs = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        label = self.label_map.get(item['label'].lower(), 6)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label),
            'languages': item['language']
        }
 

def calculate_metrics(all_labels, all_preds):
    """Calculate both macro and micro F1 scores"""
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    return macro_f1, micro_f1

def calculate_language_metrics(labels, preds, languages):
    """Calculate F1 scores for each language"""
    language_metrics = {}
    unique_languages = set(languages)
    
    for lang in unique_languages:
        lang_mask = [i for i, l in enumerate(languages) if l == lang]
        if lang_mask:
            lang_labels = [labels[i] for i in lang_mask]
            lang_preds = [preds[i] for i in lang_mask]
            
            macro_f1 = f1_score(lang_labels, lang_preds, average='macro')
            micro_f1 = f1_score(lang_labels, lang_preds, average='micro')
            
            language_metrics[f'{lang}_macro_f1'] = macro_f1
            language_metrics[f'{lang}_micro_f1'] = micro_f1
            language_metrics[f'{lang}_sample_count'] = len(lang_mask)
            
    return language_metrics

def get_scheduler(optimizer, num_training_steps, config):
    """Get scheduler based on config"""
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    if config.scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif config.scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:  # polynomial
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

def train_epoch(model, train_loader, optimizer, scheduler, device, config, scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []
    
    accumulated_loss = 0
    optimizer.zero_grad()  # Zero gradients once at start
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['languages']
        
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            loss = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)(outputs.logits, labels)
            loss = loss / config.accumulation_steps
        
        scaler.scale(loss).backward()
        
        accumulated_loss += loss.item() * config.accumulation_steps

        if (batch_idx + 1) % config.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Add to total loss and reset accumulation
            total_loss += accumulated_loss
            accumulated_loss = 0

        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)
    
    # Don't step scheduler here anymore
    
    macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
    language_metrics = calculate_language_metrics(all_labels, all_preds, all_languages)
    
    return total_loss / len(train_loader), macro_f1, micro_f1, language_metrics

def evaluate(model, eval_loader, device, split_name="eval"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []
    
    criterion = torch.nn.CrossEntropyLoss()  # No label smoothing during evaluation

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Evaluating {split_name}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            languages = batch['languages']

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)

    macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
    language_metrics = calculate_language_metrics(all_labels, all_preds, all_languages)
    
    prefixed_language_metrics = {
        f'{split_name}_{k}': v for k, v in language_metrics.items()
    }
    
    return total_loss / len(eval_loader), macro_f1, micro_f1, prefixed_language_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--sweep_id', type=str, required=True, help='W&B sweep ID')
    parser.add_argument('--offline', action='store_true', help='Run W&B in offline mode')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_CONSOLE"] = "off"

    WANDB_PROJECT = "mt5-search"
    WANDB_ENTITY = "aniezka"
    
    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'name': 'dev_macro_f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-6,
                'max': 5e-5
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-3,
                'max': 5e-2
            },
            'batch_size': {
                'values': [4, 8, 12]
            },
            'adam_beta2': {
                'values': [0.98, 0.99, 0.995, 0.999]
            },
            'warmup_ratio': {
                'values': [0.1, 0.15, 0.2]
            },
            'adam_beta1': {
                'values': [0.9, 0.92, 0.95]
            },
            'max_length': {
                'values': [256, 384, 512]
            },
            'patience': {
                'values': [3, 5, 7]
            },
            'scheduler_type': {
                'values': ['linear', 'cosine', 'polynomial']
            },
            'gradient_clip_val': {
                'values': [0.5, 1.0, 2.0]
            },
            'label_smoothing': {
                'values': [0.0, 0.1, 0.2]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'input_format': {
                'values': ['language_first', 'claim_first', 'evidence_first']
            },
            'accumulation_steps': {
                'values': [4, 6, 8]
            },
            'adam_epsilon': {'value': 1e-8},
            'epochs': {
                'values': [10, 15, 20]
            }
        }
    }

    if args.gpu_id == 0:
        sweep_id = wandb.sweep(
            sweep_configuration,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY
        )
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id

    def train():
        max_retries = 3
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                run = wandb.init(
                    settings=wandb.Settings(start_method="thread"),
                    reinit=True,
                    resume="allow"
                )
                if run is not None:
                    break
                print(f"W&B initialization attempt {attempt + 1} returned None")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                print("All W&B initialization attempts failed")
                return
            except Exception as e:
                print(f"W&B initialization attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                print("All W&B initialization attempts failed")
                return
        
        try:
            config = wandb.config
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
    
            dataset = load_dataset("utahnlp/x-fact", "all_languages")
            
            train_languages = [item['language'] for item in dataset['train']]
            language_counts = Counter(train_languages)
            try:
                wandb.run.summary['dataset_language_distribution'] = language_counts
            except Exception as e:
                print(f"Failed to log language distribution to W&B: {e}")
    
            model_name = "google/mt5-base"
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
            model = MT5ForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=7,
                        dropout_rate=config.dropout
                    ).to(device)
    
            dataloaders = {}
            for split_name, split_data in dataset.items():
                dataset_obj = XFACTDataset(split_data, tokenizer, config)
                batch_size = config.batch_size
                shuffle = (split_name == 'train')
                dataloaders[split_name] = DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle)
    
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
    
            num_training_steps = len(dataloaders['train']) * config.epochs
            scheduler = get_scheduler(optimizer, num_training_steps, config)
    
            best_dev_macro_f1 = 0
            best_metrics = {}
            patience_counter = 0
    
            use_amp = torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
            # Create checkpoints directory
            os.makedirs('checkpoints', exist_ok=True)
    
            for epoch in range(config.epochs):
                print(f'\nEpoch {epoch + 1}/{config.epochs}')
                
                try:
                    # Training
                    train_loss, train_macro_f1, train_micro_f1, train_lang_metrics = train_epoch(
                        model, dataloaders['train'], optimizer, scheduler, device, config, scaler
                    )
                    scheduler.step()
                    
                    # Evaluation
                    metrics = {'epoch': epoch + 1}
                    
                    metrics.update({
                        'train_loss': train_loss,
                        'train_macro_f1': train_macro_f1,
                        'train_micro_f1': train_micro_f1
                    })
                    metrics.update({f'train_{k}': v for k, v in train_lang_metrics.items()})
                    
                    # Evaluate all splits
                    for split in ['dev', 'test', 'ood', 'zeroshot']:
                        loss, macro_f1, micro_f1, lang_metrics = evaluate(
                            model, dataloaders[split], device, split
                        )
                        metrics.update({
                            f'{split}_loss': loss,
                            f'{split}_macro_f1': macro_f1,
                            f'{split}_micro_f1': micro_f1
                        })
                        metrics.update(lang_metrics)
    
                    try:
                        wandb.log(metrics)
                    except Exception as e:
                        print(f"Failed to log metrics to W&B for epoch {epoch + 1}: {e}")
    
                    # Model saving and early stopping
                    if metrics['dev_macro_f1'] > best_dev_macro_f1:
                        best_dev_macro_f1 = metrics['dev_macro_f1']
                        best_metrics = {f'best_{k}': v for k, v in metrics.items()}
                        patience_counter = 0
                        
                        # Save model
                        model_path = os.path.join('checkpoints', f'best_model_epoch_{epoch + 1}.pt')
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_dev_macro_f1': best_dev_macro_f1,
                                'config': config,
                            }, model_path)
                            
                            try:
                                wandb.save(model_path)
                            except Exception as e:
                                print(f"Failed to sync model to W&B: {e}")
                        except Exception as e:
                            print(f"Failed to save model checkpoint: {e}")
                            # Try to save with minimal state
                            try:
                                backup_path = os.path.join('checkpoints', f'backup_model_epoch_{epoch + 1}.pt')
                                torch.save(model.state_dict(), backup_path)
                                print(f"Saved backup model to {backup_path}")
                            except Exception as e2:
                                print(f"Failed to save backup model: {e2}")
                    else:
                        patience_counter += 1
                        if patience_counter >= config.patience:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break
    
                except Exception as e:
                    print(f"Error during epoch {epoch + 1}: {e}")
                    # Try to save emergency backup
                    try:
                        emergency_path = os.path.join('checkpoints', f'emergency_model_epoch_{epoch + 1}.pt')
                        torch.save(model.state_dict(), emergency_path)
                        print(f"Saved emergency backup to {emergency_path}")
                    except Exception as e2:
                        print(f"Failed to save emergency backup: {e2}")
                    continue
    
            # Log final best metrics
            try:
                wandb.log(best_metrics)
            except Exception as e:
                print(f"Failed to log final best metrics to W&B: {e}")
                print("Best metrics:", best_metrics)
            
            # Save final model
            try:
                final_path = os.path.join('checkpoints', 'final_model.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_dev_macro_f1': best_dev_macro_f1,
                    'best_metrics': best_metrics,
                    'config': config,
                }, final_path)
                try:
                    wandb.save(final_path)
                except Exception as e:
                    print(f"Failed to sync final model to W&B: {e}")
            except Exception as e:
                print(f"Failed to save final model: {e}")
    
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise
        finally:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Failed to properly finish W&B run: {e}")    

if __name__ == "__main__":
    main()
