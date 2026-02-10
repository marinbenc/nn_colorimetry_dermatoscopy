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
    df_save = df.copy()
    df_save['fp'] = df_save['fp'] + 1
    df_save['pred'] = df_save['pred'] + 1
    df_save.to_csv('fp_classification_test_results.csv', index=False)
    return df

