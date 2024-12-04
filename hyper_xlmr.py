import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
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


class XLMRClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels, dropout_prob=0.1):
        super().__init__()

        self.roberta = XLMRobertaModel.from_pretrained(pretrained_model_name)
        
        # Remove the parameter freezing - we want to train the entire model
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:,0,:]  # Get CLS token output

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return torch.nn.functional.softmax(logits, dim=1), loss


class XFACTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, model_type="claim_only"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

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

        if self.model_type == "claim_only":
            text = f"language: {item['language']} site: {item['site']} claim: {item['claim']}"
        else:
            text = f"language: {item['language']} site: {item['site']} claim: {item['claim']} "
            for i in range(1,4):
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

            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

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


def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

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
            
            # Use the new model class
            model = XLMRClassifier(
                model_name,
                num_labels=7,
                dropout_prob=config.dropout_prob
            ).to(device)

            # Create datasets and dataloaders
            train_data = XFACTDataset(dataset['train'], tokenizer, config.max_length, config.model_type)
            dev_data = XFACTDataset(dataset['dev'], tokenizer, config.max_length, config.model_type)
            test_data = XFACTDataset(dataset['test'], tokenizer, config.max_length, config.model_type)
            ood_data = XFACTDataset(dataset['ood'], tokenizer, config.max_length, config.model_type)
            zeroshot_data = XFACTDataset(dataset['zeroshot'], tokenizer, config.max_length, config.model_type)

            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            dev_loader = DataLoader(dev_data, batch_size=config.batch_size)
            test_loader = DataLoader(test_data, batch_size=config.batch_size)
            ood_loader = DataLoader(ood_data, batch_size=config.batch_size)
            zeroshot_loader = DataLoader(zeroshot_data, batch_size=config.batch_size)

            # Modified optimizer to include all parameters
            optimizer = AdamW(
                model.parameters(),  # Now includes all parameters
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

            best_dev_macro_f1 = 0
            best_metrics = {}
            early_stopping_counter = 0

            for epoch in range(config.epochs):
                print(f'\nEpoch {epoch + 1}/{config.epochs}')

                train_loss, train_macro_f1, train_micro_f1 = train_epoch(
                    model, train_loader, optimizer, scheduler, device
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

                    save_path = os.path.join(run.dir, 'best_model.pt')
                    torch.save(model.state_dict(), save_path)
                    wandb.save('best_model.pt')

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

            wandb.log({
                'best_dev_macro_f1': best_metrics['dev_macro_f1'],
                'best_dev_micro_f1': best_metrics['dev_micro_f1'],
                'best_test_macro_f1': best_metrics['test_macro_f1'],
                'best_test_micro_f1': best_metrics['test_micro_f1'],
                'best_ood_macro_f1': best_metrics['ood_macro_f1'],
                'best_ood_micro_f1': best_metrics['ood_micro_f1'],
                'best_zeroshot_macro_f1': best_metrics['zeroshot_macro_f1'],
                'best_zeroshot_micro_f1': best_metrics['zeroshot_micro_f1'],
            })

            wandb.finish()

    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'name': 'dev_macro_f1', 'goal': 'maximize'},
        'parameters': {
            'model_type': {'values': ['claim_only', 'attn_ea']},
            'batch_size': {'values': [4, 8, 12, 16, 24, 32]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 5e-5},
            'epochs': {'value': 10},
            'max_length': {'values': [360, 512]},
            'dropout_prob': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-2},
            'adam_epsilon': {'value': 1e-8},
            'warmup_ratio': {'distribution': 'uniform', 'min': 0.0, 'max': 0.2},
            'patience': {'value': 3}
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project="experiments")
    wandb.agent(sweep_id, function=main, count=20)
