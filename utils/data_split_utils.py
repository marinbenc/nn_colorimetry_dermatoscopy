"""Utilities for dataset splitting and stratified loaders.

Provides:
- make_stratified_loader: create a DataLoader using WeightedRandomSampler with options
  to stratify by (dataset, class) pairs when available.
- generate_leave_one_class_out: produce (train_files, val_files) for leaving each class out.
- generate_kfold: produce k folds of (train_files, val_files).
- generate_standard_split: produce train/val/test splits.
- save_split_files: helper to save CSVs for splits.

These are extracted from train_fp_regression.py so other training scripts can reuse them.
"""
from typing import List, Tuple, Optional
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler


def make_stratified_loader(dataset, batch_size, shuffle=True, stratify_by_dataset=True):
    """Create a DataLoader that samples in a stratified (class-balanced) way.

    If `dataset` has attribute `file_dataset` (per-sample dataset id) and
    `stratify_by_dataset` is True, sampling weights will be computed by
    (dataset, class) pair to avoid dominance by larger datasets.

    Returns a DataLoader with a WeightedRandomSampler.
    """
    # If this is a CombinedFPDataset and stratify_by_dataset is True, sample by both dataset and class
    if stratify_by_dataset and hasattr(dataset, 'file_dataset'):
        file_dataset = np.array(dataset.file_dataset)
        labels = np.array(dataset.fps)
        # Get all (dataset, class) pairs
        pairs = np.array([f"{d}|{c}" for d, c in zip(file_dataset, labels)])
        unique_pairs, counts = np.unique(pairs, return_counts=True)
        # Assign each sample a weight inversely proportional to its (dataset, class) population
        pair_weights = {pair: 1.0 / count for pair, count in zip(unique_pairs, counts)}
        samples_weight = np.array([pair_weights[p] for p in pairs])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        # Fallback to class-balanced stratified sampling for single datasets or if disabled
        labels = np.array(dataset.fps)
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_sample_count = counts
        weight = 1. / class_sample_count
        # build a mapping from label->weight
        label_to_weight = {lab: w for lab, w in zip(unique_labels, weight)}
        samples_weight = np.array([label_to_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def generate_leave_one_class_out(all_files: np.ndarray, all_labels: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Generate train/val splits for leave-one-class-out cross validation.

    Returns a list of tuples: (fold_name, train_files, val_files)
    where fold_name is e.g. 'fold_0' for class 1 left out (0-indexed).
    """
    splits = []
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)
    for fp_class in sorted(unique_classes):
        val_idx = np.where(all_labels == fp_class)[0]
        train_idx = np.where(all_labels != fp_class)[0]
        train_files = all_files[train_idx]
        val_files = all_files[val_idx]
        fold_name = f'fold_{fp_class-1}' if fp_class is not None else f'fold_{fp_class}'
        # fold_name intentionally mirrors previous convention where classes start at 1
        splits.append((fold_name, train_files, val_files))
    return splits


def generate_kfold(all_files: np.ndarray, k: int, random_state: int = 0, patient_ids: Optional[np.ndarray] = None):
    """Generate k-fold train/val splits returning a list of (fold_index, train_files, val_files).
    
    If patient_ids is provided, performs per-patient splitting (all images from same patient stay together).
    Otherwise, performs per-file splitting.
    """
    from sklearn.model_selection import KFold

    all_files = np.array(all_files)
    
    if patient_ids is not None:
        # Per-patient splitting: group files by patient, then split patients
        patient_ids = np.array(patient_ids)
        unique_patients, patient_indices = np.unique(patient_ids, return_inverse=True)
        
        # Split patients (not files)
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = []
        for fold, (train_patient_idx, val_patient_idx) in enumerate(kfold.split(unique_patients)):
            train_patients = unique_patients[train_patient_idx]
            val_patients = unique_patients[val_patient_idx]
            
            # Get all files belonging to train/val patients
            train_files_mask = np.isin(patient_ids, train_patients)
            val_files_mask = np.isin(patient_ids, val_patients)
            
            train_files = all_files[train_files_mask]
            val_files = all_files[val_files_mask]
            splits.append((fold, train_files, val_files))
        return splits
    else:
        # Per-file splitting (original behavior)
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_files)):
            train_files = all_files[train_idx]
            val_files = all_files[val_idx]
            splits.append((fold, train_files, val_files))
        return splits


def generate_standard_split(all_files: np.ndarray, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed: Optional[int] = None):
    """Create a standard train/val/test split.

    Returns (train_files, val_files, test_files).
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError('train_frac + val_frac + test_frac must equal 1.0')
    files = np.sort(np.array(all_files))
    rng = np.random.RandomState(seed) if seed is not None else np.random
    rng.shuffle(files)
    n = len(files)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    return train_files, val_files, test_files


def save_split_files(fold_dir: str, train_files: np.ndarray, val_files: np.ndarray, test_files: Optional[np.ndarray] = None):
    """Save filenames for a split into CSV files under fold_dir."""
    os.makedirs(fold_dir, exist_ok=True)
    pd.DataFrame(train_files).to_csv(os.path.join(fold_dir, 'train_files.csv'), index=False)
    pd.DataFrame(val_files).to_csv(os.path.join(fold_dir, 'valid_files.csv'), index=False)
    if test_files is not None:
        pd.DataFrame(test_files).to_csv(os.path.join(fold_dir, 'test_files.csv'), index=False)
