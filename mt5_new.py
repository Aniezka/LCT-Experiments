import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    MT5ForSequenceClassification,
    MT5TokenizerFast,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import f1_score
import wandb
from typing import Dict, List
import logging
from collections import Counter

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

class XFACTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine claim and evidence
        text = f"claim: {item['claim']} "
        for i in range(1, 6):  # evidence_1 to evidence_5
            evidence_key = f'evidence_{i}'
            if evidence_key in item and item[evidence_key]:
                text += f"evidence_{i}: {item[evidence_key]} "
        
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

def compute_metrics(pred: EvalPrediction, all_languages: List[str]) -> Dict:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Overall metrics
    metrics = {
        'micro_f1': f1_score(labels, preds, average='micro'),
        'macro_f1': f1_score(labels, preds, average='macro')
    }
    
    # Per-language metrics
    for lang in all_languages:
        lang_mask = (pred.inputs['language'] == lang)
        if sum(lang_mask) > 0:
            lang_labels = labels[lang_mask]
            lang_preds = preds[lang_mask]
            metrics[f'{lang}_micro_f1'] = f1_score(lang_labels, lang_preds, average='micro')
            metrics[f'{lang}_macro_f1'] = f1_score(lang_labels, lang_preds, average='macro')
    
    return metrics

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
