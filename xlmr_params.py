import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
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
import torch.nn as nn
import gc
from torch.multiprocessing import Process, set_start_method
import signal
import sys
from collections import Counter

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

def calculate_metrics(all_labels, all_preds):
    """Calculate both macro and micro F1 scores"""
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    return macro_f1, micro_f1

def calculate_warmup_steps(batch_size, dataset_size, epochs):
    """Calculate 10% of total steps for warmup"""
    total_steps = (dataset_size // batch_size) * epochs
    return total_steps // 10


class XFACTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        
        text = f"language: {item['language']} site: {item['site']} claim: {item['claim']} "
        for i in range(1, 6): 
            ev_key = f'evidence_{i}'
            if ev_key in item and item[ev_key]:
                text += f"evidence_{i}: {item[ev_key]} "

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
  

def train_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps=16):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []  # Track languages
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        languages = batch['languages']  # Get languages from batch
        
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)  # Store languages

        total_loss += loss.item() * gradient_accumulation_steps

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Calculate overall metrics
    macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
    
    # Calculate language-specific metrics
    language_metrics = calculate_language_metrics(all_labels, all_preds, all_languages)
    
    return total_loss / len(train_loader), macro_f1, micro_f1, language_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--sweep_id', type=str, required=True, help='W&B sweep ID')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    WANDB_PROJECT = "upd-hyper-xlmr" 
    WANDB_ENTITY = "aniezka"       
    
    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'name': 'dev_macro_f1',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            's': 2,
            'eta': 3
        },
        'parameters': {
            'batch_size': {'values': [6, 8, 12]},
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-6,
                'max': 5e-5
            },
            'epochs': {'value': 10},
            'max_length': {'value': 512},
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'adam_beta1': {'values': [0.9, 0.95]},
            'adam_beta2': {'values': [0.98, 0.99, 0.999]},
            'patience': {'value': 5},
            'adam_epsilon': {'value': 1e-8}
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
        run = wandb.init()
        config = wandb.config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        dataset = load_dataset("utahnlp/x-fact", "all_languages")
        
        # Log dataset language distribution
        train_languages = [item['language'] for item in dataset['train']]
        language_counts = Counter(train_languages)
        wandb.run.summary['dataset_language_distribution'] = language_counts
        
        num_training_examples = len(dataset['train'])
        config.warmup_steps = calculate_warmup_steps(
            config.batch_size, 
            num_training_examples, 
            config.epochs
        )

        model_name = "xlm-roberta-base"
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        model = XLMRobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=7
            ).to(device)

        # Data loading
        dataloaders = {}
        for split_name, split_data in dataset.items():
            dataset_obj = XFACTDataset(split_data, tokenizer, config.max_length)
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

        best_dev_macro_f1 = 0
        best_metrics = {}

        for epoch in range(config.epochs):
            print(f'\nEpoch {epoch + 1}/{config.epochs}')
            
            # Training
            train_loss, train_macro_f1, train_micro_f1, train_lang_metrics = train_epoch(
                model, dataloaders['train'], optimizer, scheduler, device
            )
            
            # Evaluation
            metrics = {'epoch': epoch + 1}
            
            # Add training metrics
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

            wandb.log(metrics)

            # Update best metrics if dev performance improves
            if metrics['dev_macro_f1'] > best_dev_macro_f1:
                best_dev_macro_f1 = metrics['dev_macro_f1']
                best_metrics = {f'best_{k}': v for k, v in metrics.items()}
                
                model_path = os.path.join(wandb.run.dir, 'best_model.pt')
                torch.save(model.state_dict(), model_path)
                wandb.save('best_model.pt')

        wandb.log(best_metrics)

    wandb.agent(
        sweep_id,
        function=train,
        count=6,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )

if __name__ == "__main__":
    main()
