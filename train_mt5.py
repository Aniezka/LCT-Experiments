import torch
from transformers import MT5Tokenizer, MT5ForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
import argparse
import os

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
            for i in range(1, 4):
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
            'labels': torch.tensor(label)
        }

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader), f1_score(all_labels, all_preds, average='macro')

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
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

    return total_loss / len(eval_loader), f1_score(all_labels, all_preds, average='macro')

def main(args):
    wandb.init(
        project="preliminary_experiments",
        name=f"mt5-{args.model_type}",
        config=args.__dict__
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = load_dataset("utahnlp/x-fact", "all_languages")

    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=7
    ).to(device)

    train_data = XFACTDataset(dataset['train'], tokenizer, args.max_length, args.model_type)
    dev_data = XFACTDataset(dataset['dev'], tokenizer, args.max_length, args.model_type)
    test_data = XFACTDataset(dataset['test'], tokenizer, args.max_length, args.model_type)
    ood_data = XFACTDataset(dataset['ood'], tokenizer, args.max_length, args.model_type)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    ood_loader = DataLoader(ood_data, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_dev_f1 = 0
    best_test_f1 = 0
    best_ood_f1 = 0

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, dev_f1 = evaluate(model, dev_loader, device)
        test_loss, test_f1 = evaluate(model, test_loader, device)
        ood_loss, ood_f1 = evaluate(model, ood_loader, device)

        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}')
        print(f'OOD Loss: {ood_loss:.4f}, OOD F1: {ood_f1:.4f}')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_f1,
            'dev_loss': dev_loss,
            'dev_f1': dev_f1,
            'test_loss': test_loss,
            'test_f1': test_f1,
            'ood_loss': ood_loss,
            'ood_f1': ood_f1
        })

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_ood_f1 = ood_f1

            save_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            wandb.save('best_model.pt')

    print(f'\nBest Results:')
    print(f'Dev F1: {best_dev_f1:.4f}')
    print(f'Test F1: {best_test_f1:.4f}')
    print(f'OOD F1: {best_ood_f1:.4f}')

    wandb.log({
        'best_dev_f1': best_dev_f1,
        'best_test_f1': best_test_f1,
        'best_ood_f1': best_ood_f1
    })

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='claim_only', choices=['claim_only', 'attn_ea'])
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=360)

    args = parser.parse_args()
    main(args)
  
