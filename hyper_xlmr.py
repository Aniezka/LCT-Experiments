import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
import argparse
import os
import torch.nn as nn
import gc

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
        for i in range(1,6):
            ev_key = f'evidence_{i}'
            if ev_key in item and item[ev_key]:
                text += f"evidence: {item[ev_key]} "

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
            'language': item['language']
        }

def evaluate(model, eval_loader, device, data_split="eval"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    predictions_by_language = {}

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Evaluating {data_split}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            languages = batch['language']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for pred, label, lang in zip(preds_np, labels_np, languages):
                if lang not in predictions_by_language:
                    predictions_by_language[lang] = {'preds': [], 'labels': []}
                predictions_by_language[lang]['preds'].append(pred)
                predictions_by_language[lang]['labels'].append(label)

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

    overall_loss = total_loss / len(eval_loader)
    overall_macro_f1 = f1_score(all_labels, all_preds, average='macro')
    overall_micro_f1 = f1_score(all_labels, all_preds, average='micro')

    language_metrics = {}
    for lang, data in predictions_by_language.items():
        lang_macro_f1 = f1_score(data['labels'], data['preds'], average='macro')
        lang_micro_f1 = f1_score(data['labels'], data['preds'], average='micro')
        language_metrics[lang] = {
            'macro_f1': lang_macro_f1,
            'micro_f1': lang_micro_f1
        }

    return overall_loss, overall_macro_f1, overall_micro_f1, language_metrics

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(train_loader),
        f1_score(all_labels, all_preds, average='macro'),
        f1_score(all_labels, all_preds, average='micro')
    )

def main():
    try:
        with wandb.init() as run:
            config = wandb.config
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            dataset = load_dataset("utahnlp/x-fact", "all_languages")
            model_name = "xlm-roberta-base"
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            
            model = XLMRobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=7,
                problem_type="single_label_classification",
                hidden_dropout_prob=config.dropout_prob,
                attention_probs_dropout_prob=config.dropout_prob
            ).to(device)

            # Create datasets
            train_data = XFACTDataset(dataset['train'], tokenizer, config.max_length)
            dev_data = XFACTDataset(dataset['dev'], tokenizer, config.max_length)
            test_data = XFACTDataset(dataset['test'], tokenizer, config.max_length)
            ood_data = XFACTDataset(dataset['ood'], tokenizer, config.max_length)
            zeroshot_data = XFACTDataset(dataset['zeroshot'], tokenizer, config.max_length)

            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            dev_loader = DataLoader(dev_data, batch_size=config.batch_size)
            test_loader = DataLoader(test_data, batch_size=config.batch_size)
            ood_loader = DataLoader(ood_data, batch_size=config.batch_size)
            zeroshot_loader = DataLoader(zeroshot_data, batch_size=config.batch_size)


            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=config.adam_epsilon
            )

            num_training_steps = len(train_loader) * config.epochs
            num_warmup_steps = int(num_training_steps * config.warmup_ratio)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

            scaler = torch.cuda.amp.GradScaler()

            best_dev_macro_f1 = 0
            best_metrics = {}
            early_stopping_counter = 0

            for epoch in range(config.epochs):
                print(f'\nEpoch {epoch + 1}/{config.epochs}')

                train_loss, train_macro_f1, train_micro_f1 = train_epoch(
                    model, train_loader, optimizer, scheduler, device, scaler
                )

                dev_loss, dev_macro_f1, dev_micro_f1, dev_lang_metrics = evaluate(
                    model, dev_loader, device, "dev"
                )

                if dev_macro_f1 > best_dev_macro_f1:
                    best_dev_macro_f1 = dev_macro_f1
                    early_stopping_counter = 0

                    test_loss, test_macro_f1, test_micro_f1, test_lang_metrics = evaluate(
                        model, test_loader, device, "test"
                    )
                    ood_loss, ood_macro_f1, ood_micro_f1, ood_lang_metrics = evaluate(
                        model, ood_loader, device, "ood"
                    )
                    zeroshot_loss, zeroshot_macro_f1, zeroshot_micro_f1, zeroshot_lang_metrics = evaluate(
                        model, zeroshot_loader, device, "zeroshot"
                    )

                    best_metrics = {
                        'dev_macro_f1': dev_macro_f1,
                        'dev_micro_f1': dev_micro_f1,
                        'test_macro_f1': test_macro_f1,
                        'test_micro_f1': test_micro_f1,
                        'ood_macro_f1': ood_macro_f1,
                        'ood_micro_f1': ood_micro_f1,
                        'zeroshot_macro_f1': zeroshot_macro_f1,
                        'zeroshot_micro_f1': zeroshot_micro_f1,
                    }

                else:
                    early_stopping_counter += 1

                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_macro_f1': train_macro_f1,
                    'train_micro_f1': train_micro_f1,
                    'dev_loss': dev_loss,
                    'dev_macro_f1': dev_macro_f1,
                    'dev_micro_f1': dev_micro_f1,
                    'current_learning_rate': scheduler.get_last_lr()[0]
                })

                print(f'Train - Loss: {train_loss:.4f}, Macro F1: {train_macro_f1:.4f}, Micro F1: {train_micro_f1:.4f}')
                print(f'Dev   - Loss: {dev_loss:.4f}, Macro F1: {dev_macro_f1:.4f}, Micro F1: {dev_micro_f1:.4f}')

                if early_stopping_counter >= config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            wandb.log(best_metrics)

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

sweep_configuration = {
    'method': 'bayes',
    'metric': {'name': 'dev_macro_f1', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [8, 12, 16]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 3e-4,
            'max': 7e-4
        },
        'epochs': {'value': 10},
        'max_length': {'value': 512},
        'dropout_prob': {
            'distribution': 'uniform',
            'min': 0.30,
            'max': 0.35
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 5e-4,
            'max': 1e-3
        },
        'adam_epsilon': {'value': 1e-8},
        'warmup_ratio': {
            'distribution': 'uniform',
            'min': 0.03,
            'max': 0.05
        },
        'patience': {'value': 3}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_configuration, project="experiments")
    wandb.agent(sweep_id, function=main, count=20)
