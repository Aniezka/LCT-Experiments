import torch
from transformers import MT5TokenizerFast,MT5ForSequenceClassification
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb
from datasets import load_dataset
import argparse
import os
from collections import Counter
import random
import numpy as np


def set_seed(seed) :
    """Set all random seeds and CUDA settings"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(all_labels,all_preds) :
    """Calculate both macro and micro F1 scores"""
    macro_f1 = f1_score(all_labels,all_preds,average='macro')
    micro_f1 = f1_score(all_labels,all_preds,average='micro')
    return macro_f1,micro_f1


def calculate_language_metrics(labels,preds,languages) :
    """Calculate F1 scores for each language"""
    language_metrics = {}
    unique_languages = set(languages)

    for lang in unique_languages :
        lang_mask = [i for i,l in enumerate(languages) if l == lang]
        if lang_mask :
            lang_labels = [labels[i] for i in lang_mask]
            lang_preds = [preds[i] for i in lang_mask]

            macro_f1 = f1_score(lang_labels,lang_preds,average='macro')
            micro_f1 = f1_score(lang_labels,lang_preds,average='micro')

            language_metrics[f'{lang}_macro_f1'] = macro_f1
            language_metrics[f'{lang}_micro_f1'] = micro_f1
            language_metrics[f'{lang}_sample_count'] = len(lang_mask)

    return language_metrics


def format_input(item,format_type='language_first') :
    """Format input text according to specified template"""
    components = {
        'language' : f"language: {item['language']}",
        'site' : f"site: {item['site']}",
        'claim' : f"claim: {item['claim']}",
        'evidence' : "",
        'claimant' : f"claimant: {item.get('claimant','')}",
        'claimDate' : f"claimDate: {item.get('claimDate','')}",
        'reviewDate' : f"reviewDate: {item.get('reviewDate','')}"
    }

    # Filter out empty components
    filtered_components = {k : v for k,v in components.items()
                           if v and not v.endswith(": ")}

    for i in range(1,6) :
        ev_key = f'evidence_{i}'
        if ev_key in item and item[ev_key] :
            components['evidence'] += f"evidence_{i}: {item[ev_key]} "

    # Different ordering based on format_type
    if format_type == 'language_first' :
        text = f"{filtered_components['language']} {filtered_components['site']} "
        if 'claimant' in filtered_components :
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components :
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components :
            text += f"{filtered_components['reviewDate']} "
        text += f"{filtered_components['claim']} {components['evidence']}"
    elif format_type == 'claim_first' :
        text = f"{filtered_components['claim']} "
        if 'claimant' in filtered_components :
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components :
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components :
            text += f"{filtered_components['reviewDate']} "
        text += f"{filtered_components['language']} {filtered_components['site']} {components['evidence']}"
    else :  # evidence_first
        text = f"{components['evidence']}"
        text += f"{filtered_components['language']} {filtered_components['site']} "
        if 'claimant' in filtered_components :
            text += f"{filtered_components['claimant']} "
        if 'claimDate' in filtered_components :
            text += f"{filtered_components['claimDate']} "
        if 'reviewDate' in filtered_components :
            text += f"{filtered_components['reviewDate']} "
        text += filtered_components['claim']

    return text.strip()


class XFACTDataset(Dataset) :
    def __init__(self,data,tokenizer,config) :
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.input_format = config.input_format

        self.label_map = {
            'false' : 0,
            'mostly_false' : 1,
            'partly_true' : 2,
            'mostly_true' : 3,
            'true' : 4,
            'unverifiable' : 5,
            'other' : 6
        }

    def __len__(self) :
        return len(self.data)

    def __getitem__(self,idx) :
        item = self.data[idx]
        text = format_input(item,self.input_format)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = self.label_map.get(item['label'].lower(),6)
        return {
            'input_ids' : encoding['input_ids'].squeeze(),
            'attention_mask' : encoding['attention_mask'].squeeze(),
            'labels' : torch.tensor(label),
            'languages' : item['language']
        }

def evaluate(model,eval_loader,device,split_name="eval") :
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []

    with torch.no_grad() :
        for batch in tqdm(eval_loader,desc=f'Evaluating {split_name}') :
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            languages = batch['languages']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits,dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_languages.extend(languages)

    macro_f1,micro_f1 = calculate_metrics(all_labels,all_preds)
    language_metrics = calculate_language_metrics(all_labels,all_preds,all_languages)

    prefixed_language_metrics = {
        f'{split_name}_{k}' : v for k,v in language_metrics.items()
    }

    return total_loss / len(eval_loader),macro_f1,micro_f1,prefixed_language_metrics


def check_grads(model,loss_value,batch_idx) :
    """Monitor gradient values and loss for debugging"""
    if loss_value is None or torch.isnan(loss_value) or torch.isinf(loss_value) :
        print(f"Batch {batch_idx}: Invalid loss value: {loss_value}")
        return False,0,0

    grad_norm = 0
    max_grad = 0
    for p in model.parameters() :
        if p.grad is not None :
            grad_norm += p.grad.data.norm(2).item() ** 2
            max_grad = max(max_grad,p.grad.data.abs().max().item())
    grad_norm = grad_norm ** 0.5

    if grad_norm > 100 or max_grad > 100 :  # Gradient explosion check
        print(f"Batch {batch_idx}: Large gradients - Grad norm: {grad_norm}, Max grad: {max_grad}")
        return False,grad_norm,max_grad

    return True,grad_norm,max_grad


def train_epoch(model,train_loader,optimizer,scheduler,scaler,device,config) :
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_languages = []
    optimizer.zero_grad(set_to_none=True)

    nan_counter = 0
    max_nan_tolerance = 3

    for batch_idx,batch in enumerate(tqdm(train_loader,desc='Training')) :
        try :
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            languages = batch['languages']

            if torch.isnan(input_ids).any() or torch.isnan(attention_mask).any() :
                print(f"NaN found in inputs at batch {batch_idx}")
                continue

            with torch.cuda.amp.autocast() :
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / config.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss) :
                print(f"NaN/Inf loss at batch {batch_idx}: {loss.item()}")
                nan_counter += 1
                if nan_counter >= max_nan_tolerance :
                    raise RuntimeError(f"Hit {max_nan_tolerance} NaN losses in a row")
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulation_steps == 0 :
                scaler.unscale_(optimizer)

                valid_grads,grad_norm,max_grad = check_grads(
                    model,loss,batch_idx
                )

                if not valid_grads :
                    optimizer.zero_grad(set_to_none=True)
                    continue

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_val
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if batch_idx % 50 == 0 :
                    print(f"Batch {batch_idx}")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"Grad norm: {grad_norm:.4f}")
                    print(f"Max grad: {max_grad:.4f}")
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            with torch.no_grad() :
                logits = outputs.logits.float()
                preds = torch.argmax(logits,dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_languages.extend(languages)

            total_loss += loss.item() * config.accumulation_steps

        except RuntimeError as e :
            if "out of memory" in str(e) :
                print(f"OOM error in batch {batch_idx}. Clearing memory...")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            raise e

    if len(all_preds) == 0 :
        raise RuntimeError("No valid predictions in epoch")

    macro_f1,micro_f1 = calculate_metrics(all_labels,all_preds)
    language_metrics = calculate_language_metrics(all_labels,all_preds,all_languages)

    return total_loss / len(train_loader),macro_f1,micro_f1,language_metrics


def get_scheduler(optimizer,num_training_steps,config) :
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)

    if config.scheduler_type == 'linear' :
        return get_linear_schedule_with_warmup(
            optimizer,num_warmup_steps,num_training_steps
        )
    elif config.scheduler_type == 'cosine' :
        return get_cosine_schedule_with_warmup(
            optimizer,num_warmup_steps,num_training_steps
        )
    else :  # polynomial
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,num_warmup_steps,num_training_steps
        )


def train() :
    # Initialize wandb with unique run_id and capture the process ID
    process_id = os.environ.get('PROCESS','0')
    run_id = wandb.util.generate_id()

    run = wandb.init(
        id=run_id,
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
        tags=[f"process_{process_id}"]  # Tag runs with their process ID
    )
    config = wandb.config

    # Generate a unique seed combining process_id and run_id
    base_seed = int(run_id.split('-')[0],16) % (2 ** 32 - 1)
    process_offset = int(process_id) * 10000  # Ensure different seed ranges for each process
    seed = (base_seed + process_offset) % (2 ** 32 - 1)
    set_seed(seed)

    # Use the GPU assigned by Condor via CUDA_VISIBLE_DEVICES
    if not torch.cuda.is_available() :
        raise RuntimeError("No CUDA device available. Check CUDA_VISIBLE_DEVICES setting.")

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # Log device information
    gpu_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_id)
    print(f"Process {process_id} using GPU {gpu_id}: {gpu_name}")
    wandb.run.summary.update({
        "process_id" : process_id,
        "gpu_id" : gpu_id,
        "gpu_name" : gpu_name,
        "seed" : seed
    })

    scaler = torch.cuda.amp.GradScaler()

    try :
        # Load dataset with the unique seed
        dataset = load_dataset(
            "utahnlp/x-fact",
            "all_languages",
            split_seed=seed
        )

        train_languages = [item['language'] for item in dataset['train']]
        language_counts = Counter(train_languages)
        wandb.run.summary['dataset_language_distribution'] = language_counts

        # Initialize model with deterministic settings
        model_name = "bigscience/mt0-small"
        tokenizer = MT5TokenizerFast.from_pretrained(model_name)
        model = MT5ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=7,
            dropout_rate=config.dropout,
            use_cache=False
        ).to(device)

        model = model.train()
        model.gradient_checkpointing_enable()

        # Initialize dataloaders with process-specific worker_init_fn
        def worker_init_fn(worker_id) :
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        dataloaders = {}
        for split_name,split_data in dataset.items() :
            dataset_obj = XFACTDataset(split_data,tokenizer,config)
            batch_size = config.batch_size
            shuffle = (split_name == 'train')
            dataloaders[split_name] = DataLoader(
                dataset_obj,
                batch_size=batch_size,
                shuffle=shuffle,
                worker_init_fn=worker_init_fn,
                num_workers=2,  # Reduced from 10 to avoid overwhelming the system
                pin_memory=True,
                persistent_workers=True
            )

        if config.frozen_layers > 0 :
            for i in range(config.frozen_layers) :
                for param in model.encoder.block[i].parameters() :
                    param.requires_grad = False

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1,config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )

        num_training_steps = len(dataloaders['train']) * config.epochs
        scheduler = get_scheduler(optimizer,num_training_steps,config)

        best_dev_macro_f1 = 0
        best_metrics = {}

        # Training loop
        for epoch in range(config.epochs) :
            print(f'\nProcess {process_id} - Epoch {epoch + 1}/{config.epochs}')

            train_loss,train_macro_f1,train_micro_f1,train_lang_metrics = train_epoch(
                model,dataloaders['train'],optimizer,scheduler,scaler,device,config
            )

            metrics = {
                'epoch' : epoch + 1,
                'train_loss' : train_loss,
                'train_macro_f1' : train_macro_f1,
                'train_micro_f1' : train_micro_f1,
                'process_id' : process_id
            }
            metrics.update({f'train_{k}' : v for k,v in train_lang_metrics.items()})

            for split in ['dev','test','ood','zeroshot'] :
                loss,macro_f1,micro_f1,lang_metrics = evaluate(
                    model,dataloaders[split],device,split
                )
                metrics.update({
                    f'{split}_loss' : loss,
                    f'{split}_macro_f1' : macro_f1,
                    f'{split}_micro_f1' : micro_f1
                })
                metrics.update(lang_metrics)

            wandb.log(metrics)

            # Save best model with process-specific name
            if metrics['dev_macro_f1'] > best_dev_macro_f1 :
                best_dev_macro_f1 = metrics['dev_macro_f1']
                best_metrics = {f'best_{k}' : v for k,v in metrics.items()}

                model_path = os.path.join(wandb.run.dir,f'best_model_process{process_id}.pt')
                torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'epoch' : epoch,
                    'best_dev_macro_f1' : best_dev_macro_f1,
                    'config' : config._items,
                    'seed' : seed,
                    'process_id' : process_id
                },model_path)
                wandb.save(f'best_model_process{process_id}.pt')

        wandb.log(best_metrics)

    except Exception as e :
        print(f"Error in process {process_id}: {str(e)}")
        raise e

    finally :
        # Cleanup
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, required=True)
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
                'min': 5e-7,
                'max': 1e-6
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2
            },
            'batch_size': {
                'value': 2
            },
            'gradient_clip_val': {
                'value': 0.1
            },
            'adam_beta2': {
                'value': 0.999
            },
            'warmup_ratio': {
                'value': 0.1
            },
            'accumulation_steps': {
                'value': 8
            },
            'adam_beta1': {
                'value': 0.9
            },
            'max_length': {
                'value': 128
            },
            'scheduler_type': {
                'value': 'linear'
            },
            'label_smoothing': {
                'value': 0.05
            },
            'dropout': {
                'value': 0.2
            },
            'frozen_layers': {
                'value': 2
            },
            'input_format': {
                'value': 'language_first'
            },
            'adam_epsilon': {
                'value': 1e-8
            },
            'epochs': {
                'value': 10
            }
        }
    }

    if args.sweep_id.lower() == "none":
        sweep_id = wandb.sweep(
            sweep_configuration,
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY
        )
        print(f"Created sweep with ID: {sweep_id}")
        
        sweep_id_file = "/scratch/hshcharbakova/LCT-Experiments/current_sweep_id.txt"
        with open(sweep_id_file, 'w') as f:
            f.write(sweep_id)
    else:
        sweep_id = args.sweep_id

    wandb.agent(
        sweep_id,
        function=train,
        count=1,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )


if __name__ == "__main__" :
    main()
