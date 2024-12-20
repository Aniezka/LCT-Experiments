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

def train(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Load tokenizer and model
        tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-base')
        model = MT5ForSequenceClassification.from_pretrained(
            'google/mt5-base',
            num_labels=len(LABEL_MAP)
        )
        
        # Enable gradient checkpointing after model initialization
        model.gradient_checkpointing_enable()
        
        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        # Load dataset
        dataset = load_dataset("utahnlp/x-fact", "all_languages")
        
        # Track language distribution
        train_languages = [item['language'] for item in dataset['train']]
        language_counts = Counter(train_languages)
        wandb.run.summary['dataset_language_distribution'] = dict(language_counts)
        
        train_dataset = XFACTDataset(dataset['train'], tokenizer)
        dev_dataset = XFACTDataset(dataset['dev'], tokenizer)
        test_dataset = XFACTDataset(dataset['test'], tokenizer)
        ood_dataset = XFACTDataset(dataset['ood'], tokenizer)
        zeroshot_dataset = XFACTDataset(dataset['zeroshot'], tokenizer)
        
        # Get unique languages
        all_languages = list(set(item['language'] for item in dataset['train']))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{run.id}",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            weight_decay=config.weight_decay,
            logging_dir=f"./logs/{run.id}",
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            report_to="wandb",
            gradient_accumulation_steps=config.accumulation_steps,
            max_grad_norm=config.gradient_clip_val,
            warmup_ratio=config.warmup_ratio,
            fp16=True,
            save_total_limit=3
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=lambda pred: compute_metrics(pred, all_languages)
        )
        
        # Train and evaluate
        trainer.train()
        
        # Evaluate on all splits
        dev_metrics = trainer.evaluate(dev_dataset)
        test_metrics = trainer.evaluate(test_dataset)
        ood_metrics = trainer.evaluate(ood_dataset)
        zeroshot_metrics = trainer.evaluate(zeroshot_dataset)
        
        # Prefix metrics for each split
        dev_metrics = {f"dev_{k}": v for k, v in dev_metrics.items()}
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        ood_metrics = {f"ood_{k}": v for k, v in ood_metrics.items()}
        zeroshot_metrics = {f"zeroshot_{k}": v for k, v in zeroshot_metrics.items()}
        
        # Combine all metrics
        all_metrics = {
            **dev_metrics,
            **test_metrics,
            **ood_metrics,
            **zeroshot_metrics
        }
        
        # Log metrics
        wandb.log(all_metrics)
        
        # Save best model
        trainer.save_model(f"./best_model/{run.id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, required=True, help='W&B sweep ID')
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
