"""Shared utilities for running tests / evaluations for classification and regression.

Provides:
- run_test_loop: run a model on a dataset and return a DataFrame with fp/pred/file
- evaluate_experiment: detect folds under an experiments dir and run per-fold or single-split evaluation,
  aggregating results and printing common metrics.

This centralizes fold-handling and aggregation so tests are consistent between tasks.
"""
from typing import Callable, Optional
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, cohen_kappa_score


def run_test_loop(model: torch.nn.Module,
                  test_dataset,
                  device: str = 'cuda',
                  batch_size: int = 1,
                  predict_fn: Optional[Callable] = None,
                  tqdm_cls=None) -> pd.DataFrame:
    """Run model on test_dataset and return a DataFrame with columns ['fp','pred','file'].

    predict_fn: callable(logits_tensor) -> numpy array of integer predictions.
    If None, will use argmax over logits (classification behavior).
    """
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    fps = []
    preds = []
    iterator = loader
    if tqdm_cls is not None:
        iterator = tqdm_cls(loader, desc='Testing', unit='batch')
    with torch.no_grad():
        for batch in iterator:
            # Expect batch to be (x, y)
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            if predict_fn is not None:
                pred = predict_fn(logits)
            else:
                # default: classification
                pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(np.asarray(pred).reshape(-1).tolist())
            fps.extend(y.cpu().numpy().reshape(-1).tolist())

    fps = np.array(fps)
    preds = np.array(preds)
    df = pd.DataFrame({'fp': fps, 'pred': preds, 'file': test_dataset.orig_files})
    return df


def evaluate_experiment(config: dict,
                        experiment_dir: str,
                        model_builder: Callable[[int], torch.nn.Module],
                        dataset_builder: Callable[[str, Optional[np.ndarray], int], object],
                        predict_fn: Optional[Callable] = None,
                        num_classes: int = 6,
                        batch_size: Optional[int] = None,
                        device: Optional[str] = None,
                        test_dataset_name: Optional[str] = None,
                        tqdm_cls=None) -> pd.DataFrame:
    """Run evaluation for an experiment.

    - Detects fold directories `fold_*` under experiment_dir. If present, evaluates each fold's
      `valid_files.csv` and `model.pth` and aggregates results.
    - Otherwise, falls back to single split, reading `test_files.csv` and `model.pth`.

    model_builder: callable(num_classes) -> model
    dataset_builder: callable(dataset_name, files_array_or_None, blur_amount) -> dataset instance
    predict_fn: callable(logits_tensor) -> numpy preds (optional)
    Returns aggregated DataFrame (concatenated per-fold or single split)
    """
    import json
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if batch_size is None:
        batch_size = int(config.get('batch_size', 32))

    # find folds
    fold_dirs = [d for d in os.listdir(experiment_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(experiment_dir, d))]
    fold_dirs = sorted(fold_dirs, key=lambda x: int(x.split('_')[1])) if fold_dirs else []

    if fold_dirs:
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
            if test_dataset_name is not None and test_dataset_name != config['dataset_name']:
                val_dataset = dataset_builder(test_dataset_name, None, config.get('blur_amount', 0))
            else:
                val_dataset = dataset_builder(config['dataset_name'], valid_files, config.get('blur_amount', 0))
            model = model_builder(num_classes)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            df = run_test_loop(model, val_dataset, device=device, batch_size=batch_size, predict_fn=predict_fn, tqdm_cls=tqdm_cls)
            df['fold'] = fold
            all_dfs.append(df)
            kappa = cohen_kappa_score(df['fp'], df['pred'], weights='quadratic')
            all_kappas.append(kappa)

        if len(all_dfs) == 0:
            raise RuntimeError('No valid folds found with both valid_files.csv and model.pth')
        all_df = pd.concat(all_dfs, ignore_index=True)
        print("\n=== Aggregated Results Across All Folds (Validation Sets) ===")
        print(classification_report(all_df['fp'], all_df['pred'], digits=3))
        print("Cohen's kappa:", cohen_kappa_score(all_df['fp'], all_df['pred'], weights='quadratic'))
        print(f"Mean fold kappa: {np.mean(all_kappas):.4f}")
        print(f"Std dev fold kappa: {np.std(all_kappas):.4f}")
        print(f"Min fold kappa: {np.min(all_kappas):.4f}")
        print(f"Max fold kappa: {np.max(all_kappas):.4f}")
        return all_df
    else:
        # single split
        test_files_path = os.path.join(experiment_dir, 'test_files.csv')
        if test_dataset_name is not None and test_dataset_name != config['dataset_name']:
            test_dataset = dataset_builder(test_dataset_name, None, config.get('blur_amount', 0))
        else:
            test_files = pd.read_csv(test_files_path).iloc[:, 0].values
            test_dataset = dataset_builder(config['dataset_name'], test_files, config.get('blur_amount', 0))
        # load checkpoint (prefer experiments/model.pth then config.checkpoint)
        if os.path.exists(os.path.join(experiment_dir, 'model.pth')):
            checkpoint_path = os.path.join(experiment_dir, 'model.pth')
        elif config.get('checkpoint') and os.path.exists(config['checkpoint']):
            checkpoint_path = config['checkpoint']
        else:
            raise FileNotFoundError('No valid checkpoint found for testing.')
        model = model_builder(num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        df = run_test_loop(model, test_dataset, device=device, batch_size=batch_size, predict_fn=predict_fn, tqdm_cls=tqdm_cls)
        print(classification_report(df['fp'], df['pred'], digits=3))
        print("Cohen's kappa:", cohen_kappa_score(df['fp'], df['pred'], weights='quadratic'))
        return df
