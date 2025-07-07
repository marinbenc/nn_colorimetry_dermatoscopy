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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--test_dataset_name', type=str, default=None, help='Override test dataset name (use full dataset)')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Optionally override dataset for testing
    if args.test_dataset_name is not None and args.test_dataset_name != config['dataset_name']:
        test_dataset_name = args.test_dataset_name
        print(f"Testing on different dataset: {test_dataset_name}")
        test_dataset = FPDataset(test_dataset_name, files=None, blur_amount=config['blur_amount'])
    else:
        # Load test files from the experiment directory
        experiment_dir = f"experiments/{config['experiment_name']}"
        test_files_path = os.path.join(experiment_dir, 'test_files.csv')
        test_files = pd.read_csv(test_files_path).iloc[:, 0].values
        test_dataset = FPDataset(config['dataset_name'], files=test_files, blur_amount=config['blur_amount'])

    # Create model instance and load weights
    model = get_model_from_string(config.get('model', 'efficientnet_b4_coral'), num_classes=6)
    # If using a fine-tune config, the checkpoint is always in the experiment dir
    if os.path.exists(os.path.join(f"experiments/{config['experiment_name']}", 'model.pth')):
        checkpoint_path = os.path.join(f"experiments/{config['experiment_name']}", 'model.pth')
    elif config.get('checkpoint') and os.path.exists(config['checkpoint']):
        checkpoint_path = config['checkpoint']
    else:
        raise FileNotFoundError('No valid checkpoint found for testing.')
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    # Run test
    df = test_fp_regression(model, test_dataset, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=config['batch_size'], tqdm_cls=tqdm)
    # Print classification report
    print(classification_report(df['fp'], df['pred'], digits=3))
    # Print Cohen's kappa
    print('Cohen\'s kappa:', cohen_kappa_score(df['fp'], df['pred'], weights='quadratic'))

