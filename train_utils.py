import os
import json
import numpy as np
import pandas as pd
import torch
from fp_dataset import FPDataset
from lab_dataset import LabDataset
from model_factory import get_model_from_string, get_efficientnet_b4_classification
from utils.data_split_utils import (
    generate_leave_one_class_out,
    generate_kfold,
    generate_standard_split,
    save_split_files,
)


def load_datasets(config, files, dataset_class=FPDataset, white_balance=False):
    """Load train and valid datasets from file lists"""
    train_dataset = dataset_class(
        config['dataset_name'],
        files[0],
        blur_amount=config['blur_amount'],
        white_balance=white_balance
    )
    valid_dataset = dataset_class(
        config['dataset_name'],
        files[1],
        blur_amount=config['blur_amount'],
        white_balance=white_balance
    )
    return train_dataset, valid_dataset


def load_model(config, model_loader_fn=None):
    """Load model and optionally load checkpoint for fine-tuning"""
    if model_loader_fn is None:
        model_loader_fn = lambda: get_model_from_string(config['model'], num_classes=6)
    
    model = model_loader_fn()
    
    if config.get('checkpoint') is not None and os.path.exists(config['checkpoint']):
        print(f"Loading checkpoint from {config['checkpoint']} for fine-tuning...")
        checkpoint = torch.load(config['checkpoint'], map_location=torch.device('cpu'))
        
        # Get model's state dict keys
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        # Find keys that exist in both
        common_keys = model_keys & checkpoint_keys
        
        # Filter checkpoint to only include common keys
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in common_keys}
        
        # Load with strict=False to allow missing keys (e.g., classifier head)
        missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
        
        if missing_keys:
            print(f"  Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys (will be ignored): {unexpected_keys}")
        
        print(f"  Loaded {len(common_keys)}/{len(checkpoint_keys)} keys from checkpoint.")
    
    return model


def generate_splits(config, all_files, all_labels, experiment_dir):
    """Generate train/valid splits based on fold type and return list of (fold_name, train_files, val_files) tuples"""
    fold_type = config.get('fold_type', 'kfold')
    k = config.get('k_folds', 0)
    splits = []
    
    if fold_type == 'leave-one-class-out':
        print('Using leave-one-class-out cross-validation.')
        splits = list(generate_leave_one_class_out(all_files, all_labels))
    elif k and k > 1:
        print(f'Using {k}-fold cross-validation.')
        fold_splits = list(generate_kfold(all_files, k))
        # Convert (fold_number, train_files, val_files) to (fold_name, train_files, val_files)
        splits = [(f'fold_{fold}', train_files, val_files) for fold, train_files, val_files in fold_splits]
    else:
        # Standard train/val/test split
        print('Using standard train/val/test split.')
        train_files, valid_files, test_files = generate_standard_split(
            all_files,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=0
        )
        # Save to main experiment directory
        pd.DataFrame(train_files).to_csv(os.path.join(experiment_dir, 'train_files.csv'), index=False)
        pd.DataFrame(valid_files).to_csv(os.path.join(experiment_dir, 'valid_files.csv'), index=False)
        pd.DataFrame(test_files).to_csv(os.path.join(experiment_dir, 'test_files.csv'), index=False)
        splits = [('standard', train_files, valid_files)]
    
    return splits


def run_training_pipeline(config, trainer_class, dataset_class=FPDataset, model_loader_fn=None, white_balance=False):
    """Run the complete training pipeline for all folds"""
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
    
    # Load all data to determine splits
    all_dataset = dataset_class(
        config['dataset_name'],
        files=None,
        blur_amount=config['blur_amount'],
        white_balance=white_balance
    )
    all_files = np.array(all_dataset.orig_files)
    all_labels = np.array(all_dataset.fps) if hasattr(all_dataset, 'fps') else np.zeros(len(all_files))
    
    # Generate splits
    splits = generate_splits(config, all_files, all_labels, experiment_dir)
    
    # Track validation files for assertion
    used_val_files = set()
    all_files_set = set(all_files)
    
    # Train on each fold
    for fold_name, train_files, val_files in splits:
        # Assertions
        assert len(set(train_files) & set(val_files)) == 0, f"{fold_name}: Train and validation files overlap!"
        used_val_files.update(val_files)
        
        # Create fold directory and save splits
        if fold_name != 'standard':  # standard split already saved above
            fold_dir = os.path.join(experiment_dir, fold_name)
        else:
            fold_dir = experiment_dir
        os.makedirs(fold_dir, exist_ok=True)
        
        if fold_name != 'standard':  # don't save twice for standard split
            save_split_files(fold_dir, train_files, val_files)
        
        # Load datasets
        train_dataset, valid_dataset = load_datasets(
            config,
            (train_files, val_files),
            dataset_class=dataset_class,
            white_balance=white_balance
        )
        
        # Load model
        model = load_model(config, model_loader_fn)
        
        # Train
        fold_checkpoint_path = os.path.join(fold_dir, 'model.pth')
        fold_log_dir = os.path.join(fold_dir, 'tensorboard')
        trainer = trainer_class(
            model,
            lr=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            patience=config.get('patience', 5),
            save_path=fold_checkpoint_path,
            log_dir=fold_log_dir
        )
        trained_model = trainer.train(train_dataset, valid_dataset)
    
    # After all folds, check that all validation files are disjoint and cover all samples
    if len(splits) > 1 or splits[0][0] != 'standard':  # only check for cross-validation
        assert used_val_files == all_files_set, "Not all files are covered in validation sets across folds!"
