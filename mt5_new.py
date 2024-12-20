import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    MT5ForSequenceClassification,
    MT5TokenizerFast
)
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
from typing import Dict, List
import logging
from collections import Counter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP = {
    'false': 0,
    'mostly false': 1,
    'mostly_false': 1,
    'partly true': 2,
    'partly_true': 2,
    'partly true/misleading': 2,
    'mostly true': 3,
    'mostly_true': 3,
    'true': 4,
    'unverifiable': 5,
    'complicated/hard to categorise': 6,
    'other': 6
}

def format_input(item, format_type='language_first'):
    """Format input text according to specified template"""
    components = {
        'language': f"language: {item['language']}",
        'site': f"site: {item['site']}",  # Added site
        'claim': f"claim: {item['claim']}",
        'evidence': "",
        'claimant': f"claimant: {item.get('claimant', '')}",
        'claimDate': f"claimDate: {item.get('claimDate', '')}",
        'reviewDate': f"reviewDate: {item.get('reviewDate', '')}"
    }
    
    # Filter out empty components
    filtered_components = {k: v for k, v in components.items() 
                         if v and not v.endswith(": ")}
    
    # Add evidence texts
    for i in range(1, 6):  # evidence_1 to evidence_5
        ev_key = f'evidence_{i}'
        if ev_key in item and item[ev_key]:
            components['evidence'] += f"evidence_{i}: {item[ev_key]} "
    
    # Different ordering based on format_type
    if format_type == 'language_first':
        text = f"{filtered_components['language']} {filtered_components['site']} "  # Added site
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
        text += f"{filtered_components['language']} {filtered_components['site']} {components['evidence']}"  # Added site
    else:  # evidence_first
        text = f"{components['evidence']}"
        text += f"{filtered_components['language']} {filtered_components['site']} "  # Added site
        if 'claimant' in filtered_components:
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components:
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components:
            text += f"{filtered_components['reviewDate']} "
        text += filtered_components['claim']
    
    return text.strip()

class XFACTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.input_format = config.input_format 

    def __len__(self):
        return len(self.data)

   def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the format_input function instead of manual formatting
        text = format_input(item, self.input_format)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Handle labels more robustly
        label = item['label'].lower()
        if label not in LABEL_MAP:
            logger.warning(f"Unknown label encountered: {label}, defaulting to 'other'")
            label = 'other'
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(LABEL_MAP[label]),
            'language': item['language']
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

def evaluate(model, eval_loader, device, split_name="eval"):
    """Evaluate the model on the given dataloader"""
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
            languages = batch['language']
            
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
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['language']
        
        # Clear gradients at the start of each batch if it's the first of an accumulation cycle
        if batch_idx % config.accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        try:
            # Forward pass with mixed precision
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
                
                # Backward pass with scaling
                scaler.scale(loss).backward()

            # Collect predictions
            with torch.no_grad():
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_languages.extend(languages)

            total_loss += loss.item() * config.accumulation_steps

            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            continue

    # Calculate metrics if we have valid predictions
    if len(all_labels) > 0:
        macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
        language_metrics = calculate_language_metrics(all_labels, all_preds, all_languages)
        return total_loss / len(train_loader), macro_f1, micro_f1, language_metrics
    else:
        return float('nan'), 0, 0, {}

def get_scheduler(optimizer, num_training_steps, config):
    """Get scheduler based on config"""
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    return get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    

def train():
    run = wandb.init()
    config = wandb.config

    # Initialize mixed precision scaler with more conservative growth
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**10,
        growth_factor=1.5,
        backoff_factor=0.5,
        growth_interval=100
    )
    
    # Get process ID and set device
    process_id = int(os.environ.get("PROCESS_ID", 0))
    device = torch.device(f'cuda:{process_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model with float32 initialization for better stability
    model = MT5ForSequenceClassification.from_pretrained(
        'google/mt5-base',
        num_labels=7,
        torch_dtype=torch.float32
    ).to(device)
    model.gradient_checkpointing_enable()
    
    tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-base')

    # Load dataset and create dataloaders
    dataset = load_dataset("utahnlp/x-fact", "all_languages")
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
    parser.add_argument('--process_id', type=int, default=0, help='Process ID for cluster')
    args = parser.parse_args()
    
    WANDB_PROJECT = "mt5-search"
    WANDB_ENTITY = "aniezka"
    
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'name': 'macro_f1', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': 1e-5,
                'max': 1e-3
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'epochs': {
                'values': [3, 5, 7]
            },
            'weight_decay': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            'warmup_ratio': {
                'values': [0.1, 0.15, 0.2]
            },
            'gradient_clip_val': {
                'values': [0.1, 0.5, 1.0]
            },
            'accumulation_steps': {
                'values': [2, 4, 8]
            },
            'max_length': {
                'value': 512
            },
            'input_format': {
                'value': 'language_first'
            },
            'adam_beta1': {
                'value': 0.9
            },
            'adam_beta2': {
                'value': 0.999
            },
            'adam_epsilon': {
                'value': 1e-8
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
