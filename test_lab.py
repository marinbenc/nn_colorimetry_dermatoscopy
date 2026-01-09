import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lab_dataset import LabDataset
from model_factory import get_model_from_string
import pingouin as pg


def delta_e_1976_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute per-sample CIE Delta E 1976 between predicted and target Lab values.
    pred/target shape: (N, 3) for [L*, a*, b*]. Returns (N,) array.
    """
    diff = pred - target
    return np.sqrt(np.sum(diff ** 2, axis=1))


def icc_2_1_and_3_1(Y: np.ndarray) -> (float, float):
    """Compute ICC(2,1) and ICC(3,1) using Pingouin from a matrix Y (n_subjects, k_raters).
    Expects k=2 (gt vs pred) but supports k>=2.
    Returns (ICC2, ICC3).
    """
    n, k = Y.shape
    if n < 2 or k < 2:
        return np.nan, np.nan
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(n), k),
        'rater': np.tile([f'r{i}' for i in range(k)], n),
        'score': Y.reshape(-1),
    })
    icc_tbl = pg.intraclass_corr(data=df, targets='subject', raters='rater', ratings='score')
    icc2_row = icc_tbl.loc[icc_tbl['Type'] == 'ICC2']
    icc3_row = icc_tbl.loc[icc_tbl['Type'] == 'ICC3']
    icc2 = float(icc2_row['ICC'].values[0]) if not icc2_row.empty else np.nan
    icc3 = float(icc3_row['ICC'].values[0]) if not icc3_row.empty else np.nan
    return icc2, icc3


def run_lab_test_loop(model: torch.nn.Module, dataset: LabDataset, device: str = 'cuda', batch_size: int = 32, tqdm_cls=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    iterator = loader
    if tqdm_cls is not None:
        iterator = tqdm_cls(loader, desc='Testing (Lab)', unit='batch')

    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            logits = model(x)
            preds = logits.detach().cpu().numpy()
            targets = y.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    files = dataset.orig_files
    return preds, targets, files


def eval_fold(checkpoint_path: str, files_csv: str, config: dict, device: str, tqdm_cls=None):
    # Build dataset
    blur_amount = config.get('blur_amount', 0)
    dataset_name = config['dataset_name']
    if os.path.exists(files_csv):
        files = pd.read_csv(files_csv).iloc[:, 0].values
        dataset = LabDataset(dataset_name, files=files, blur_amount=blur_amount)
    else:
        # Fallback to full dataset
        dataset = LabDataset(dataset_name, files=None, blur_amount=blur_amount)

    # Build model
    model = get_model_from_string(config.get('model', 'efficientnet_b4_lab'), num_classes=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Run inference
    preds, targets, files = run_lab_test_loop(model, dataset, device=device, batch_size=int(config.get('batch_size', 32)), tqdm_cls=tqdm_cls)

    # Metrics
    delta_e = delta_e_1976_np(preds, targets)
    results = {
        'delta_e_mean': float(np.mean(delta_e)),
        'delta_e_std': float(np.std(delta_e)),
    }

    # ICC per channel and averages
    channel_names = ['L', 'a', 'b']
    icc2_list = []
    icc3_list = []
    for c in range(3):
        Y = np.stack([targets[:, c], preds[:, c]], axis=1)  # shape (N,2): [gt, pred]
        icc2, icc3 = icc_2_1_and_3_1(Y)
        results[f'ICC2_1_{channel_names[c]}'] = float(icc2)
        results[f'ICC3_1_{channel_names[c]}'] = float(icc3)
        icc2_list.append(icc2)
        icc3_list.append(icc3)
    results['ICC2_1_mean'] = float(np.nanmean(icc2_list))
    results['ICC3_1_mean'] = float(np.nanmean(icc3_list))

    # ITA = arctan((L* - 50)/b*) * (180/pi). Use arctan2 for numerical stability.
    ita_gt = np.degrees(np.arctan2(targets[:, 0] - 50.0, targets[:, 2]))
    ita_pred = np.degrees(np.arctan2(preds[:, 0] - 50.0, preds[:, 2]))
    Y_ita = np.stack([ita_gt, ita_pred], axis=1)
    icc2_ita, icc3_ita = icc_2_1_and_3_1(Y_ita)
    results['ICC2_1_ITA'] = float(icc2_ita)
    results['ICC3_1_ITA'] = float(icc3_ita)

    # Build per-sample DataFrame if needed
    df = pd.DataFrame({
        'file': files,
        'L_gt': targets[:, 0], 'a_gt': targets[:, 1], 'b_gt': targets[:, 2],
        'L_pred': preds[:, 0], 'a_pred': preds[:, 1], 'b_pred': preds[:, 2],
        'delta_e': delta_e,
        'ita_gt': ita_gt,
        'ita_pred': ita_pred,
    })

    return results, df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = json.load(f)

    experiment_dir = f"experiments/{config['experiment_name']}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Detect folds
    fold_dirs = [d for d in os.listdir(experiment_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(experiment_dir, d))]
    fold_dirs = sorted(fold_dirs, key=lambda x: int(x.split('_')[1])) if fold_dirs else []

    if fold_dirs:
        print('Detected folds. Evaluating each fold...')
        all_results = []
        all_dfs = []
        for fold_dir in fold_dirs:
            fold_path = os.path.join(experiment_dir, fold_dir)
            valid_files_csv = os.path.join(fold_path, 'valid_files.csv')
            checkpoint_path = os.path.join(fold_path, 'model.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Skipping {fold_dir}: missing checkpoint {checkpoint_path}")
                continue
            results, df = eval_fold(checkpoint_path, valid_files_csv, config, device, tqdm)
            results['fold'] = int(fold_dir.split('_')[1])
            all_results.append(results)
            df['fold'] = int(fold_dir.split('_')[1])
            all_dfs.append(df)

        if not all_results:
            raise RuntimeError('No valid folds found to evaluate.')

        results_df = pd.DataFrame(all_results).sort_values('fold')
        print('\n=== Per-fold Metrics ===')
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print('\n=== Aggregated Metrics ===')
        print(f"Delta E mean: {results_df['delta_e_mean'].mean():.4f} ± {results_df['delta_e_mean'].std():.4f}")
        print(f"Delta E std (avg across folds): {results_df['delta_e_std'].mean():.4f}")
        print(f"ICC(2,1) mean across folds: {results_df['ICC2_1_mean'].mean():.4f}")
        print(f"ICC(3,1) mean across folds: {results_df['ICC3_1_mean'].mean():.4f}")
        print(f"ICC(2,1) ITA mean across folds: {results_df['ICC2_1_ITA'].mean():.4f} ± {results_df['ICC2_1_ITA'].std():.4f}")
        print(f"ICC(3,1) ITA mean across folds: {results_df['ICC3_1_ITA'].mean():.4f} ± {results_df['ICC3_1_ITA'].std():.4f}")

        # Optionally, save combined predictions
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df_path = os.path.join(experiment_dir, 'lab_eval_aggregated.csv')
        all_df.to_csv(all_df_path, index=False)
        print(f"Saved aggregated predictions to {all_df_path}")
    else:
        print('No folds detected. Evaluating single split...')
        # Prefer experiments/<exp>/model.pth, else config['checkpoint']
        if os.path.exists(os.path.join(experiment_dir, 'model.pth')):
            checkpoint_path = os.path.join(experiment_dir, 'model.pth')
        elif config.get('checkpoint') and os.path.exists(config['checkpoint']):
            checkpoint_path = config['checkpoint']
        else:
            raise FileNotFoundError('No valid checkpoint found for testing.')
        test_files_csv = os.path.join(experiment_dir, 'test_files.csv')
        results, df = eval_fold(checkpoint_path, test_files_csv, config, device, tqdm)
        print('\n=== Single Split Metrics ===')
        print(f"Delta E mean: {results['delta_e_mean']:.4f}")
        print(f"Delta E std:  {results['delta_e_std']:.4f}")
        print(f"ICC(2,1) L/a/b: {results['ICC2_1_L']:.4f}, {results['ICC2_1_a']:.4f}, {results['ICC2_1_b']:.4f} (mean {results['ICC2_1_mean']:.4f})")
        print(f"ICC(3,1) L/a/b: {results['ICC3_1_L']:.4f}, {results['ICC3_1_a']:.4f}, {results['ICC3_1_b']:.4f} (mean {results['ICC3_1_mean']:.4f})")
        out_csv = os.path.join(experiment_dir, 'lab_eval.csv')
        df.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")


if __name__ == '__main__':
    main()
