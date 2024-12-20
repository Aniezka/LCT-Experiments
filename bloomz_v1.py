import torch
from transformers import MT5TokenizerFast, MT5ForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
from multiprocessing import Process, cpu_count
import argparse
import os
from collections import Counter
import gc
from torch.multiprocessing import Process, set_start_method
import signal
import sys

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

def format_input(item, format_type='language_first'):
    """Format input text according to specified template"""
    components = {
        'language': f"language: {item['language']}",
        'site': f"site: {item['site']}",
        'claim': f"claim: {item['claim']}",
        'evidence': "",
        'claimant': f"claimant: {item.get('claimant', '')}",
        'claimDate': f"claimDate: {item.get('claimDate', '')}",
        'reviewDate': f"reviewDate: {item.get('reviewDate', '')}"
    }
    
    # Filter out empty components
    filtered_components = {k: v for k, v in components.items() 
                         if v and not v.endswith(": ")}
    
    for i in range(1, 6):
        ev_key = f'evidence_{i}'
        if ev_key in item and item[ev_key]:
            components['evidence'] += f"evidence_{i}: {item[ev_key]} "
    
    # Different ordering based on format_type
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

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = self.label_map.get(item['label'].lower(), 6)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label),
            'languages': item['language']
        }

def evaluate(model, eval_loader, device, split_name="eval"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []

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
                    labels=labels
                )

                total_loss += outputs.loss.item()
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

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []
    optimizer.zero_grad(set_to_none=True)  # More efficient zero_grad

    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['languages']
        
        # Clear gradients at the start of each batch
        if batch_idx % config.accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        try:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / config.accumulation_steps
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected in batch {batch_idx}")
                    continue
                
                scaler.scale(loss).backward()

            # Collect predictions only if loss is valid
            with torch.no_grad():
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_languages.extend(languages)

            total_loss += loss.item() * config.accumulation_steps

            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Clip gradients before optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
                # Step optimizer and scaler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Zero gradients after optimizer step
                optimizer.zero_grad(set_to_none=True)
                
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    if len(all_labels) > 0:  # Only calculate metrics if we have valid predictions
        macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
        language_metrics = calculate_language_metrics(all_labels, all_preds, all_languages)
        return total_loss / len(train_loader), macro_f1, micro_f1, language_metrics
    else:
        return float('nan'), 0, 0, {}

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

def train():
    run = wandb.init()
    config = wandb.config

    # Initialize mixed precision scaler with more conservative growth
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**10,  # Start with a smaller scale
        growth_factor=1.5,  # More conservative growth
        backoff_factor=0.5,
        growth_interval=100  # Increase scale less frequently
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model with float32 initialization for better stability
    model_name = "bigscience/mt0-small"
    tokenizer = MT5TokenizerFast.from_pretrained(model_name)
    model = MT5ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=7,
        dropout_rate=config.dropout,
        use_cache=False,
        torch_dtype=torch.float32  # Initialize in full precision
    ).to(device)
    model.gradient_checkpointing_enable()

    dataloaders = {}
    for split_name, split_data in Dataset.items():
        dataset_obj = XFACTDataset(split_data, tokenizer, config)
        batch_size = config.batch_size
        shuffle = (split_name == 'train')
        dataloaders[split_name] = DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle)

    if config.frozen_layers > 0:
        for i in range(config.frozen_layers):
            for param in model.encoder.block[i].parameters():
                param.requires_grad = False

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

    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        
        train_loss, train_macro_f1, train_micro_f1, train_lang_metrics = train_epoch(
            model, dataloaders['train'], optimizer, scheduler, scaler, device, config
        )
        
        metrics = {'epoch': epoch + 1}
        
        metrics.update({
            'train_loss': train_loss,
            'train_macro_f1': train_macro_f1,
            'train_micro_f1': train_micro_f1
        })
        metrics.update({f'train_{k}': v for k, v in train_lang_metrics.items()})
        
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

        wandb.log(metrics)

        # Save best model based on dev macro F1
        if metrics['dev_macro_f1'] > best_dev_macro_f1:
            best_dev_macro_f1 = metrics['dev_macro_f1']
            best_metrics = {f'best_{k}': v for k, v in metrics.items()}
            
            model_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            wandb.save('best_model.pt')

    wandb.log(best_metrics)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, required=True, help='W&B sweep ID')
    args = parser.parse_args()

    WANDB_PROJECT = "mt0-search" 
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
            'min': 5e-6,  # Reduced minimum learning rate
            'max': 5e-5   # Reduced maximum learning rate
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'batch_size': {
            'values': [4, 8]  # Reduced batch sizes
        },
        'adam_beta2': {
            'values': [0.999]  # Simplified to most stable value
        },
        'warmup_ratio': {
            'values': [0.1]  # Simplified to single stable value
        },
        'adam_beta1': {
            'value': 0.9
        },
        'max_length': {
            'value': 256
        },
        'scheduler_type': {
            'value': 'linear'  # Simplified to most stable scheduler
        },
        'gradient_clip_val': {
            'value': 0.5  # Reduced from 1.0
        },
        'label_smoothing': {
            'value': 0.1  # Set to help with training stability
        },
        'dropout': {
            'value': 0.1
        },
        'frozen_layers': {
            'value': 0
        },
        'input_format': {
            'value': 'language_first'
        },
        'accumulation_steps': {
            'value': 4  # Fixed value to reduce complexity
        },
        'adam_epsilon': {
            'value': 1e-8
        },
        'epochs': {
            'value': 10  # Fixed to reduce complexity during debugging
        }
    }
}
   

    if args.sweep_id == "none":
        sweep_id = wandb.sweep(
            sweep_configuration,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY
        )
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id

    # Each agent will run 20 trials
    wandb.agent(
        sweep_id,
        function=train,
        count=20,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )

if __name__ == "__main__":
    main()
