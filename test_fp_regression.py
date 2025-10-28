import torch
import numpy as np
import pandas as pd
from coral_pytorch.dataset import corn_label_from_logits
from model_factory import get_vgg11_bn_coral
from utils.test_utils import run_test_loop


def test_fp_regression(model, test_dataset, device='cuda', batch_size=1, tqdm_cls=None):
    """Wrapper around the shared run_test_loop for regression.

    It adapts the regression-specific prediction (corn_label_from_logits) and
    preserves the behavior of writing `fp_regression_test_results.csv`.
    """
    def predict_fn(logits_tensor):
        # logits_tensor -> coral logits; corn_label_from_logits returns tensor of labels (0-indexed)
        pred = corn_label_from_logits(logits_tensor)
        # Return predictions as 0-indexed labels (0..5) to match dataset encoding
        return pred.cpu().numpy()

    df = run_test_loop(model, test_dataset, device=device, batch_size=batch_size, predict_fn=predict_fn, tqdm_cls=tqdm_cls)
    # Save historical CSV in 1..6 label space for compatibility with previous outputs
    df_save = df.copy()
    df_save['fp'] = df_save['fp'] + 1
    df_save['pred'] = df_save['pred'] + 1
    df_save.to_csv('fp_regression_test_results.csv', index=False)
    return df
