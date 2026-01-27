import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from inference_dataset import InferenceLabDataset
from model_factory import get_model_from_string


def run_inference_fold(model, dataset, device='cuda', batch_size=32):
    """
    Run inference for a regression lab model. Expects dataset.__getitem__ to return (img, target)
    but only uses the image component.
    Returns array of shape (n_samples, 3) with predicted L,a,b values.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference', leave=False):
            # LabDataset returns (img, lab) tuples; support both formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            logits = model(images)
            preds = logits.detach().cpu().numpy()
            all_preds.append(preds)

    if not all_preds:
        return np.zeros((0, 3), dtype=float)
    return np.concatenate(all_preds, axis=0)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python lab_inference.py <inference_dataset_name> <config_path>')
        print('  inference_dataset_name: e.g. mskcc')
        exit(1)

    inference_dataset_name = sys.argv[1]
    config_path = sys.argv[2]

    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_name = config['experiment_name']
    model_name = config.get('model', 'efficientnet_b4_lab')
    k_folds = config.get('k_folds', 5)
    batch_size = config.get('batch_size', 512)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load inference dataset once (shared across folds)
    print(f"Loading inference dataset: {inference_dataset_name}")
    inf_dataset = InferenceLabDataset(inference_dataset_name,
                                      files=None,
                                      blur_amount=config.get('blur_amount', 0),
                                      white_balance=config.get('white_balance', False))
    n_samples = len(inf_dataset)
    print(f"Loaded {n_samples} samples")

    all_fold_predictions = []  # list of (fold_index, preds_array)

    for fold in range(k_folds):
        print(f"\n[Fold {fold + 1}/{k_folds}]")
        model_path = f"experiments/{experiment_name}/fold_{fold}/model.pth"
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping fold {fold}")
            continue

        model = get_model_from_string(model_name, num_classes=3)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)

        fold_preds = run_inference_fold(model, inf_dataset, device=device, batch_size=batch_size)
        # Ensure shape is (n_samples, 3)
        if fold_preds.ndim == 1:
            fold_preds = fold_preds.reshape(-1, 3)
        all_fold_predictions.append((fold, fold_preds))
        print(f"Fold {fold} predictions shape: {fold_preds.shape}")

    print(f"\nCollecting per-fold predictions across {len(all_fold_predictions)} folds")
    results_df = pd.DataFrame({'file': [inf_dataset.orig_files[i] for i in range(n_samples)]})
    for fold_idx, preds in all_fold_predictions:
        # If preds has fewer samples than dataset (due to skipped/missing), pad with NaNs
        if preds.shape[0] < n_samples:
            pad = np.full((n_samples - preds.shape[0], preds.shape[1]), np.nan, dtype=float)
            preds = np.vstack([preds, pad])
        results_df[f'L_fold_{fold_idx}'] = preds[:, 0]
        results_df[f'a_fold_{fold_idx}'] = preds[:, 1]
        results_df[f'b_fold_{fold_idx}'] = preds[:, 2]

    os.makedirs('inference_results', exist_ok=True)
    output_path = f"inference_results/infer={inference_dataset_name}_model={experiment_name}_lab.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(results_df.head())
