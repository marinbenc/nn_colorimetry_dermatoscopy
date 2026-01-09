import json
import sys
from fp_dataset import FPDataset
from trainer import ClassificationTrainer
from train_utils import run_training_pipeline
from model_factory import get_efficientnet_b4_classification

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python train_fp_classification.py <config_path>')
        exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Run training pipeline with ClassificationTrainer
    run_training_pipeline(
        config,
        trainer_class=ClassificationTrainer,
        dataset_class=FPDataset,
        model_loader_fn=lambda: get_efficientnet_b4_classification(num_classes=6),
        white_balance=False
    )
