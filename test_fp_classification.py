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

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python test_fp_classification.py <config_path>')
        exit(1)
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    checkpoint_path = os.path.join(experiment_dir, 'model.pth')
    test_files_path = os.path.join(experiment_dir, 'test_files.csv')

    # Load test files
    test_files = pd.read_csv(test_files_path).values.squeeze().tolist()
    if isinstance(test_files, str):
        test_files = [test_files]

    test_dataset = FPDataset(config['dataset_name'], test_files, blur_amount=config['blur_amount'])
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_efficientnet_b4_classification(num_classes=6)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Testing'):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print('Classification report:')
    print(classification_report(all_labels, all_preds, digits=3))
    print('Cohen kappa:', cohen_kappa_score(all_labels, all_preds))
    print('Confusion matrix:')
    print(confusion_matrix(all_labels, all_preds))
