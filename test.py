import torch
import pandas as pd
import numpy as np
from fp_dataset import FPDataset
from model_factory import get_model_from_string
from test_fp_regression import test_fp_regression
from sklearn.metrics import classification_report, cohen_kappa_score
from tqdm import tqdm
import json
import sys
import os

def run_test_on_split(model, test_dataset, device, batch_size):
    df = test_fp_regression(model, test_dataset, device=device, batch_size=batch_size, tqdm_cls=tqdm)
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--test_dataset_name', type=str, default=None, help='Override test dataset name (use full dataset)')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['batch_size']
    model_name = config.get('model', 'efficientnet_b4_coral')
    num_classes = 6

    # K-fold detection: look for fold_0, fold_1, ...
    fold_dirs = [d for d in os.listdir(experiment_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(experiment_dir, d))]
    fold_dirs = sorted(fold_dirs, key=lambda x: int(x.split('_')[1])) if fold_dirs else []

    if fold_dirs:
        print(f"Detected {len(fold_dirs)} folds. Running k-fold evaluation (aggregated over all validation sets)...")
        all_dfs = []
        all_kappas = []
        for fold, fold_dir in enumerate(fold_dirs):
            fold_path = os.path.join(experiment_dir, fold_dir)
            valid_files_path = os.path.join(fold_path, 'valid_files.csv')
            checkpoint_path = os.path.join(fold_path, 'model.pth')
            if not (os.path.exists(valid_files_path) and os.path.exists(checkpoint_path)):
                print(f"Skipping {fold_dir}: missing valid_files.csv or model.pth")
                continue
            valid_files = pd.read_csv(valid_files_path).iloc[:, 0].values
            if args.test_dataset_name is not None and args.test_dataset_name != config['dataset_name']:
                val_dataset = FPDataset(args.test_dataset_name, files=None, blur_amount=config['blur_amount'])
            else:
                val_dataset = FPDataset(config['dataset_name'], files=valid_files, blur_amount=config['blur_amount'])
            model = get_model_from_string(model_name, num_classes=num_classes)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            df = run_test_on_split(model, val_dataset, device, batch_size)
            df['fold'] = fold
            all_dfs.append(df)
            # Compute kappa for this fold
            kappa = cohen_kappa_score(df['fp'], df['pred'], weights='quadratic')
            all_kappas.append(kappa)
        # Concatenate all validation predictions
        all_df = pd.concat(all_dfs, ignore_index=True)
        print("\n=== Aggregated Results Across All Folds (Validation Sets) ===")
        print(classification_report(all_df['fp'], all_df['pred'], digits=3))
        print("Cohen's kappa:", cohen_kappa_score(all_df['fp'], all_df['pred'], weights='quadratic'))
        print(f"Mean fold kappa: {np.mean(all_kappas):.4f}")
        print(f"Std dev fold kappa: {np.std(all_kappas):.4f}")
        print(f"Min fold kappa: {np.min(all_kappas):.4f}")
        print(f"Max fold kappa: {np.max(all_kappas):.4f}")
    else:
        # Single split (no k-folds)
        if args.test_dataset_name is not None and args.test_dataset_name != config['dataset_name']:
            test_dataset_name = args.test_dataset_name
            print(f"Testing on different dataset: {test_dataset_name}")
            test_dataset = FPDataset(test_dataset_name, files=None, blur_amount=config['blur_amount'])
        else:
            test_files_path = os.path.join(experiment_dir, 'test_files.csv')
            test_files = pd.read_csv(test_files_path).iloc[:, 0].values
            test_dataset = FPDataset(config['dataset_name'], files=test_files, blur_amount=config['blur_amount'])
        model = get_model_from_string(model_name, num_classes=num_classes)
        if os.path.exists(os.path.join(experiment_dir, 'model.pth')):
            checkpoint_path = os.path.join(experiment_dir, 'model.pth')
        elif config.get('checkpoint') and os.path.exists(config['checkpoint']):
            checkpoint_path = config['checkpoint']
        else:
            raise FileNotFoundError('No valid checkpoint found for testing.')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        df = run_test_on_split(model, test_dataset, device, batch_size)
        print(classification_report(df['fp'], df['pred'], digits=3))
        print('Cohen\'s kappa:', cohen_kappa_score(df['fp'], df['pred'], weights='quadratic'))

