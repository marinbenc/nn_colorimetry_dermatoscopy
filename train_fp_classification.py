import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from fp_dataset import FPDataset
from model_factory import get_efficientnet_b4_classification

def train_model(model, train_dataset, valid_dataset, lr=0.001, batch_size=32, num_epochs=100, patience=5, num_classes=6, save_path='mel_model.pth', log_dir='runs/fp_classification'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = float('inf')
    counter = 0
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [valid]", leave=False)
        with torch.no_grad():
            for x, y in valid_iter:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
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
        print('Usage: python train_fp_classification.py <config_path>')
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

    # train / valid / test split
    all_dataset = FPDataset(config['dataset_name'], files=None, blur_amount=config['blur_amount'])
    all_files = all_dataset.orig_files
    all_files = np.sort(np.array(all_files))
    np.random.seed(0)
    np.random.shuffle(all_files)
    train_files = all_files[:int(0.8*len(all_files))]
    valid_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
    test_files = all_files[int(0.9*len(all_files)):]

    # save the split to csv
    pd.DataFrame(train_files).to_csv(os.path.join(experiment_dir, 'train_files.csv'), index=False)
    pd.DataFrame(valid_files).to_csv(os.path.join(experiment_dir, 'valid_files.csv'), index=False)
    pd.DataFrame(test_files).to_csv(os.path.join(experiment_dir, 'test_files.csv'), index=False)

    train_dataset = FPDataset(config['dataset_name'], train_files, blur_amount=config['blur_amount'])
    valid_dataset = FPDataset(config['dataset_name'], valid_files, blur_amount=config['blur_amount'])

    # Model definition
    model = get_efficientnet_b4_classification(num_classes=6)

    # Optionally load checkpoint for fine-tuning
    if config.get('checkpoint') is not None and os.path.exists(config['checkpoint']):
        print(f"Loading checkpoint from {config['checkpoint']} for fine-tuning...")
        model.load_state_dict(torch.load(config['checkpoint'], map_location=torch.device('cpu')))

    # Call the training function
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
