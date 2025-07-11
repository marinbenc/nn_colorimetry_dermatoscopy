import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import pandas as pd
import torchvision.models as models
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
import matplotlib.pyplot as plt
from fp_dataset import FPDataset, bin_FP_by_mel
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import json
from torchvision.models.vgg import VGG11_BN_Weights
from model_factory import get_model_from_string
from tqdm import tqdm
from sklearn.model_selection import KFold

def make_stratified_loader(dataset, batch_size, shuffle=True, stratify_by_dataset=True):
    # If this is a CombinedFPDataset and stratify_by_dataset is True, sample by both dataset and class
    if stratify_by_dataset and hasattr(dataset, 'file_dataset'):
        file_dataset = np.array(dataset.file_dataset)
        labels = np.array(dataset.fps)
        # Get all (dataset, class) pairs
        pairs = np.array([f"{d}|{c}" for d, c in zip(file_dataset, labels)])
        unique_pairs, counts = np.unique(pairs, return_counts=True)
        # Assign each sample a weight inversely proportional to its (dataset, class) population
        pair_weights = {pair: 1.0 / count for pair, count in zip(unique_pairs, counts)}
        samples_weight = np.array([pair_weights[p] for p in pairs])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        # Fallback to class-balanced stratified sampling for single datasets or if disabled
        labels = np.array(dataset.fps)
        class_sample_count = np.array([(labels == t).sum() for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[np.where(np.unique(labels) == t)[0][0]] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def train_model(model, train_dataset, valid_dataset, lr=0.001, batch_size=32, num_epochs=100, patience=5, num_classes=6, save_path='mel_model.pth', log_dir='runs/fp_regression'):
    train_loader = make_stratified_loader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = float('inf')
    counter = 0
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)
        for i, (x, y) in enumerate(train_iter):
            levels = levels_from_labelbatch(y, num_classes=num_classes)
            levels = levels.to(device)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = coral_loss(logits, levels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [valid]", leave=False)
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_iter):
                levels = levels_from_labelbatch(y, num_classes=num_classes)
                levels = levels.to(device)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = coral_loss(logits, levels)
                valid_loss += loss.item()
                valid_iter.set_postfix(loss=loss.item())
            valid_loss /= len(valid_loader)

        print(f'Epoch {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter == patience:
                break
    writer.close()
    return model

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python train_fp_regression.py <config_path>')
        exit(1)
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    os.makedirs(experiment_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, 'model.pth')
    config_save_path = os.path.join(experiment_dir, 'config.json')

    # Save config
    config['log_dir'] = log_dir
    config['checkpoint_path'] = checkpoint_path
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    # K-fold cross validation or standard split
    k = config.get('k_folds', 0)
    all_dataset = FPDataset(config['dataset_name'], files=None, blur_amount=config['blur_amount'])
    all_files = np.array(all_dataset.orig_files)
    # Deterministic shuffle and split using KFold on file list only
    if k and k > 1:
        kfold = KFold(n_splits=k, shuffle=True, random_state=0)
        folds = list(kfold.split(all_files))
        # Save split indices for reproducibility
        folds_csv = os.path.join(experiment_dir, 'kfold_indices.csv')
        pd.DataFrame({f'fold_{i}_train': folds[i][0], f'fold_{i}_val': folds[i][1]} for i in range(k)).to_csv(folds_csv, index=False)
        used_val_files = set()
        all_files_set = set(all_files)
        for fold, (train_idx, val_idx) in enumerate(folds):
            train_files = all_files[train_idx]
            val_files = all_files[val_idx]
            # Assertions
            assert len(set(train_files) & set(val_files)) == 0, f"Fold {fold}: Train and validation files overlap!"
            used_val_files.update(val_files)
            # Save splits
            fold_dir = os.path.join(experiment_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            pd.DataFrame(train_files).to_csv(os.path.join(fold_dir, 'train_files.csv'), index=False)
            pd.DataFrame(val_files).to_csv(os.path.join(fold_dir, 'valid_files.csv'), index=False)
            # Prepare datasets
            train_dataset = FPDataset(config['dataset_name'], train_files, blur_amount=config['blur_amount'])
            valid_dataset = FPDataset(config['dataset_name'], val_files, blur_amount=config['blur_amount'])
            # Model definition
            if 'model' not in config:
                raise ValueError('Model type must be specified in the config file ("model" key).')
            model = get_model_from_string(config['model'], num_classes=6)
            # Optionally load checkpoint for fine-tuning
            if config.get('checkpoint') is not None and os.path.exists(config['checkpoint']):
                print(f"Loading checkpoint from {config['checkpoint']} for fine-tuning...")
                model.load_state_dict(torch.load(config['checkpoint'], map_location=torch.device('cpu')))
            # Train and save model for this fold
            fold_checkpoint_path = os.path.join(fold_dir, 'model.pth')
            trained_model = train_model(
                model,
                train_dataset,
                valid_dataset,
                lr=config['learning_rate'],
                batch_size=config['batch_size'],
                num_epochs=config['num_epochs'],
                save_path=fold_checkpoint_path,
                log_dir=os.path.join(fold_dir, 'tensorboard')
            )
        # After all folds, check that all validation files are disjoint and cover all samples
        assert used_val_files == all_files_set, "Not all files are covered in validation sets across folds!"
    else:
        # Standard train/val/test split
        all_files = np.sort(np.array(all_files))
        np.random.shuffle(all_files)
        train_files = all_files[:int(0.8*len(all_files))]
        valid_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
        test_files = all_files[int(0.9*len(all_files)):]
        pd.DataFrame(train_files).to_csv(os.path.join(experiment_dir, 'train_files.csv'), index=False)
        pd.DataFrame(valid_files).to_csv(os.path.join(experiment_dir, 'valid_files.csv'), index=False)
        pd.DataFrame(test_files).to_csv(os.path.join(experiment_dir, 'test_files.csv'), index=False)
        train_dataset = FPDataset(config['dataset_name'], train_files, blur_amount=config['blur_amount'])
        valid_dataset = FPDataset(config['dataset_name'], valid_files, blur_amount=config['blur_amount'])
        if 'model' not in config:
            raise ValueError('Model type must be specified in the config file ("model" key).')
        model = get_model_from_string(config['model'], num_classes=6)
        if config.get('checkpoint') is not None and os.path.exists(config['checkpoint']):
            print(f"Loading checkpoint from {config['checkpoint']} for fine-tuning...")
            model.load_state_dict(torch.load(config['checkpoint'], map_location=torch.device('cpu')))
        trained_model = train_model(
            model,
            train_dataset,
            valid_dataset,
            lr=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            save_path=checkpoint_path,
            log_dir=log_dir
        )