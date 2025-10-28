import torch
import pandas as pd
import numpy as np
from fp_dataset import FPDataset
from model_factory import get_model_from_string, get_efficientnet_b4_classification
from utils.test_utils import evaluate_experiment
from coral_pytorch.dataset import corn_label_from_logits
from tqdm import tqdm
import json
import sys
import os

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--test_dataset_name', type=str, default=None, help='Override test dataset name (use full dataset)')
    parser.add_argument('--mode', type=str, choices=['regression', 'classification'], default='regression', help='Which model mode to test')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = config['batch_size']
    model_name = config.get('model', 'efficientnet_b4_coral')
    num_classes = 6

    # Choose model builder and predict_fn based on mode
    if args.mode == 'classification':
        def model_builder(n):
            return get_efficientnet_b4_classification(num_classes=n)

        predict_fn = None
    else:
        def model_builder(n):
            # use model string from config for regression
            return get_model_from_string(model_name, num_classes=n)

        def predict_fn(logits):
            # coral logits -> label (0-indexed)
            # NOTE: dataset labels (fp) are 0..5, so return 0-indexed preds to match them.
            return corn_label_from_logits(logits).cpu().numpy()

    def dataset_builder(name, files, blur_amount):
        return FPDataset(name, files=files, blur_amount=blur_amount)

    # Run the shared evaluator which handles folds and single-split cases
    aggregate_df = evaluate_experiment(
        config=config,
        experiment_dir=experiment_dir,
        model_builder=model_builder,
        dataset_builder=dataset_builder,
        predict_fn=predict_fn,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
        test_dataset_name=args.test_dataset_name,
        tqdm_cls=tqdm,
    )

    # evaluate_experiment already prints reports; but return or further processing can be done here

