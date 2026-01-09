import json
import sys
from lab_dataset import LabDataset
from trainer import LabRegressionTrainer
from train_utils import run_training_pipeline
from model_factory import get_model_from_string

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python train_lab_regression.py <config_path>')
        exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate that model is specified
    if 'model' not in config:
        raise ValueError('Model type must be specified in the config file ("model" key).')
    
    # Run training pipeline with LabRegressionTrainer
    run_training_pipeline(
        config,
        trainer_class=LabRegressionTrainer,
        dataset_class=LabDataset,
        white_balance=False
    )
