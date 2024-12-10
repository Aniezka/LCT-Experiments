import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
import os
import gc
from transformers import AutoConfig
import logging                     
from typing import Dict, List 
import math
import argparse

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
        for i in range(1, 6):  # Support up to 5 pieces of evidence
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
            'labels': torch.tensor(label)
        }

def calculate_metrics(all_labels, all_preds):
    """Calculate both macro and micro F1 scores"""
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    return macro_f1, micro_f1

def calculate_warmup_steps(batch_size, dataset_size, epochs):
    """Calculate 10% of total steps for warmup"""
    total_steps = (dataset_size // batch_size) * epochs
    return total_steps // 10

def train_epoch(model, train_loader, optimizer, scheduler, device, gradient_accumulation_steps=8):  # increased from 4
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
        try:

            torch.cuda.empty_cache()
            gc.collect()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)


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

            total_loss += loss.item() * gradient_accumulation_steps

            # Update weights every gradient_accumulation_steps batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            del outputs, loss, input_ids, attention_mask, labels
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: out of memory on batch {batch_idx}. Skipping batch")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    gc.collect()
                optimizer.zero_grad()

                # Try to free some memory
                if 'outputs' in locals():
                    del outputs
                if 'loss' in locals():
                    del loss
                if 'input_ids' in locals():
                    del input_ids
                if 'attention_mask' in locals():
                    del attention_mask
                if 'labels' in locals():
                    del labels
                continue
            else:
                raise e

    macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
    return total_loss / len(train_loader), macro_f1, micro_f1

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc='Evaluating')):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: out of memory on evaluation batch {batch_idx}. Skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    macro_f1, micro_f1 = calculate_metrics(all_labels, all_preds)
    return total_loss / len(eval_loader), macro_f1, micro_f1

def train():
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use only 90% of available memory
        
    run = wandb.init()
    config = wandb.config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = load_dataset("utahnlp/x-fact", "all_languages")
    
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

    train_data = XFACTDataset(dataset['train'], tokenizer, config.max_length)
    dev_data = XFACTDataset(dataset['dev'], tokenizer, config.max_length)
    test_data = XFACTDataset(dataset['test'], tokenizer, config.max_length)
    ood_data = XFACTDataset(dataset['ood'], tokenizer, config.max_length)
    zeroshot_data = XFACTDataset(dataset['zeroshot'], tokenizer, config.max_length)

    eval_batch_size = max(4, config.batch_size // 2)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=eval_batch_size)
    test_loader = DataLoader(test_data, batch_size=eval_batch_size)
    ood_loader = DataLoader(ood_data, batch_size=eval_batch_size)
    zeroshot_loader = DataLoader(zeroshot_data, batch_size=eval_batch_size)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )

    best_dev_macro_f1 = 0
    best_metrics = {}

    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        
        train_loss, train_macro_f1, train_micro_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, dev_macro_f1, dev_micro_f1 = evaluate(model, dev_loader, device)
        test_loss, test_macro_f1, test_micro_f1 = evaluate(model, test_loader, device)
        ood_loss, ood_macro_f1, ood_micro_f1 = evaluate(model, ood_loader, device)
        zeroshot_loss, zeroshot_macro_f1, zeroshot_micro_f1 = evaluate(model, zeroshot_loader, device)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_macro_f1': train_macro_f1,
            'train_micro_f1': train_micro_f1,
            'dev_loss': dev_loss,
            'dev_macro_f1': dev_macro_f1,
            'dev_micro_f1': dev_micro_f1,
            'test_loss': test_loss,
            'test_macro_f1': test_macro_f1,
            'test_micro_f1': test_micro_f1,
            'ood_loss': ood_loss,
            'ood_macro_f1': ood_macro_f1,
            'ood_micro_f1': ood_micro_f1,
            'zeroshot_loss': zeroshot_loss,
            'zeroshot_macro_f1': zeroshot_macro_f1,
            'zeroshot_micro_f1': zeroshot_micro_f1
        })

        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_metrics = {
                'dev_macro_f1': dev_macro_f1,
                'dev_micro_f1': dev_micro_f1,
                'test_macro_f1': test_macro_f1,
                'test_micro_f1': test_micro_f1,
                'ood_macro_f1': ood_macro_f1,
                'ood_micro_f1': ood_micro_f1,
                'zeroshot_macro_f1': zeroshot_macro_f1,
                'zeroshot_micro_f1': zeroshot_micro_f1
            }
            
            model_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            wandb.save('best_model.pt')

    wandb.log(best_metrics)



if __name__ == "__main__":
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
            'batch_size': {
                'values': [8, 12, 16]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 5e-6,
                'max': 5e-5
            },
            'epochs': {
                'value': 10
            },
            'max_length': {
                'value': 512
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'adam_beta1': {
                'values': [0.9, 0.95]
            },
            'adam_beta2': {
                'values': [0.98, 0.99, 0.999]
            },
            'adam_epsilon': {
                'value': 1e-8
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


    wandb.agent(
        sweep_id,
        function=train,
        count=5,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )
