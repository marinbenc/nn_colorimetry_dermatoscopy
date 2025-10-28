import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from fp_dataset import FPDataset
from model_factory import get_efficientnet_b4_classification
from utils.test_utils import evaluate_experiment, run_test_loop

def test_fp_classification(model, test_dataset, device='cuda', batch_size=32, tqdm_cls=None):
    """Wrapper for classification tests to match the regression test API.

    Returns a DataFrame with columns ['fp','pred','file'].
    """
    # Use shared run_test_loop from utils to avoid duplicate code
    df = run_test_loop(model, test_dataset, device=device, batch_size=batch_size, predict_fn=None, tqdm_cls=tqdm_cls)
    return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python test_fp_classification.py <config_path> [--test_dataset_name NAME]')
        exit(1)
    config_path = sys.argv[1]
    # Optional override argument for testing on a different dataset name (full dataset)
    test_dataset_name = None
    if len(sys.argv) > 2:
        # allow '--test_dataset_name' value
        if sys.argv[2].startswith('--test_dataset_name') and len(sys.argv) > 3:
            test_dataset_name = sys.argv[3]

    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 32)
    model_name = config.get('model', 'efficientnet_b4_coral')
    num_classes = 6

    # Use shared evaluation helper to avoid duplicating fold handling & aggregation
    def model_builder(n):
        return get_efficientnet_b4_classification(num_classes=n)

    def dataset_builder(name, files, blur_amount):
        return FPDataset(name, files=files, blur_amount=blur_amount)

    # For classification, default predict_fn (argmax) is fine, so pass None
    evaluate_experiment(
        config=config,
        experiment_dir=experiment_dir,
        model_builder=model_builder,
        dataset_builder=dataset_builder,
        predict_fn=None,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
        test_dataset_name=test_dataset_name,
        tqdm_cls=tqdm,
    )
