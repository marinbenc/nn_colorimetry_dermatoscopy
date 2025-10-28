import torch
import pandas as pd
import numpy as np
from fp_dataset import FPDataset
from model_factory import get_vgg11_bn_coral, get_efficientnet_b4_classification
from train_fp_regression import train_model as train_regression_model
from train_fp_classification import train_model as train_classification_model
import json
import os
import sys

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained_experiment', type=str, help='Name of the experiment to use as pretrained model')
    parser.add_argument('new_config', type=str, help='Path to config file for fine-tuning')
    parser.add_argument('--mode', type=str, choices=['regression', 'classification'], default='regression', help='Which model mode to fine-tune')
    args = parser.parse_args()

    # Load pretrained experiment config and checkpoint
    pretrained_dir = f"experiments/{args.pretrained_experiment}"
    pretrained_ckpt = os.path.join(pretrained_dir, 'model.pth')
    if not os.path.exists(pretrained_ckpt):
        print(f"Pretrained checkpoint not found: {pretrained_ckpt}")
        sys.exit(1)

    # Load new config
    with open(args.new_config, 'r') as f:
        config = json.load(f)

    # Set up new experiment directories
    experiment_dir = f"experiments/{config['experiment_name']}"
    os.makedirs(experiment_dir, exist_ok=True)
    log_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, 'model.pth')
    config_save_path = os.path.join(experiment_dir, 'config.json')
    config['log_dir'] = log_dir
    config['checkpoint_path'] = checkpoint_path
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Prepare dataset split for the new dataset
    all_dataset = FPDataset(config['dataset_name'], files=None, blur_amount=config['blur_amount'])
    all_files = all_dataset.orig_files
    all_files = np.sort(np.array(all_files))
    np.random.seed(0)
    np.random.shuffle(all_files)
    train_files = all_files[:int(0.8*len(all_files))]
    valid_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
    test_files = all_files[int(0.9*len(all_files)):]
    pd.DataFrame(train_files).to_csv(os.path.join(experiment_dir, 'train_files.csv'), index=False)
    pd.DataFrame(valid_files).to_csv(os.path.join(experiment_dir, 'valid_files.csv'), index=False)
    pd.DataFrame(test_files).to_csv(os.path.join(experiment_dir, 'test_files.csv'), index=False)
    train_dataset = FPDataset(config['dataset_name'], train_files, blur_amount=config['blur_amount'])
    valid_dataset = FPDataset(config['dataset_name'], valid_files, blur_amount=config['blur_amount'])

    # Model definition and load pretrained weights depending on mode
    if args.mode == 'classification':
        model = get_efficientnet_b4_classification(num_classes=6)
        print(f"Loading pretrained classification weights from {pretrained_ckpt}")
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

        # Fine-tune classification model
        trained_model = train_classification_model(
            model,
            train_dataset,
            valid_dataset,
            lr=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            save_path=checkpoint_path,
            log_dir=log_dir
        )
    else:
        model = get_vgg11_bn_coral(num_classes=6)
        print(f"Loading pretrained regression weights from {pretrained_ckpt}")
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

        # Fine-tune regression model
        trained_model = train_regression_model(
            model,
            train_dataset,
            valid_dataset,
            lr=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            save_path=checkpoint_path,
            log_dir=log_dir
        )
