import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from inference_dataset import InferenceFPDataset
from model_factory import get_model_from_string
from coral_pytorch.dataset import corn_label_from_logits


def run_inference_fold(model, dataset, device='cuda', batch_size=32, is_coral=False):
    """
    Run inference on a single fold and return predictions.
    
    Args:
        model: The trained model
        dataset: Inference dataset
        device: Device to run on
        batch_size: Batch size
        is_coral: If True, use CORAL decoding; else use argmax
    
    Returns array of predictions (0-indexed labels).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference', leave=False):
            images = batch.to(device)
            logits = model(images)
            
            if is_coral:
                # CORAL: decode logits to labels using corn_label_from_logits
                preds = corn_label_from_logits(logits).cpu().numpy()
            else:
                # Classification: argmax
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.append(preds)
    
    return np.concatenate(all_preds, axis=0)


def majority_vote(predictions_list):
    """
    Apply majority voting across folds.
    
    predictions_list: list of arrays, each of shape (n_samples,)
    Returns: array of majority voted predictions
    """
    predictions_array = np.array(predictions_list)  # (n_folds, n_samples)
    # For each sample, find the most common prediction
    majority_preds = np.zeros(predictions_array.shape[1], dtype=int)
    for i in range(predictions_array.shape[1]):
        majority_preds[i] = np.bincount(predictions_array[:, i]).argmax()
    return majority_preds


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python fp_inference.py <inference_dataset_name> <config_path>')
        print('  inference_dataset_name: isic2020 or milk10k')
        exit(1)
    
    inference_dataset_name = sys.argv[1]
    config_path = sys.argv[2]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    experiment_name = config['experiment_name']
    model_name = config.get('model', 'efficientnet_b4_classification')
    k_folds = config.get('k_folds', 5)
    batch_size = 512#config.get('batch_size', 32)
    num_classes = 6
    
    # Check if model is CORAL-based
    is_coral = 'coral' in model_name.lower()
    print(f"Model: {model_name}, Using CORAL: {is_coral}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load inference dataset once (shared across all folds)
    print(f"Loading inference dataset: {inference_dataset_name}")
    inf_dataset = InferenceFPDataset(inference_dataset_name, 
                                     files=None, 
                                     blur_amount=config.get('blur_amount', 0), 
                                     white_balance=config.get('white_balance', False))
    n_samples = len(inf_dataset)
    print(f"Loaded {n_samples} samples")
    
    # Run inference on each fold and collect predictions
    all_fold_predictions = []  # list of (fold_index, preds)
    
    for fold in range(k_folds):
        print(f"\n[Fold {fold + 1}/{k_folds}]")
        
        # Load fold model
        model_path = f"experiments/{experiment_name}/fold_{fold}/model.pth"
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping fold {fold}")
            continue
        
        model = get_model_from_string(model_name, num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        
        # Run inference
        fold_preds = run_inference_fold(model, inf_dataset, device=device, batch_size=batch_size, is_coral=is_coral)
        all_fold_predictions.append((fold, fold_preds))
        print(f"Fold {fold} predictions shape: {fold_preds.shape}")
    
    # Create results dataframe with per-fold predictions
    print(f"\nCollecting per-fold predictions across {len(all_fold_predictions)} folds")
    results_df = pd.DataFrame({
        'file': [inf_dataset.orig_files[i] for i in range(n_samples)]
    })
    for fold_idx, preds in all_fold_predictions:
        results_df[f'fold_{fold_idx}'] = preds
    
    # Save results
    os.makedirs('inference_results', exist_ok=True)
    output_path = f"inference_results/infer={inference_dataset_name}_model={experiment_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(results_df.head())
