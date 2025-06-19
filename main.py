import argparse
import yaml
import numpy as np
import torch
from trainandtest.train_unet import train_unet
from trainandtest.evaluate import evaluate_model
from Data_Preparation.data_loader import prepare_data
from models.unet1d import Unet1D
from utils.helpers import seed_everything
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    args = parser.parse_args()
    
    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Set random seed for reproducibility
    seed_everything(config["seed"])
    
    # Load dataset
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Initialize the model
    unet = Unet1D(dim=64, dim_mults=(1, 2, 4), channels=1)
    
    # Train the model if enabled in config
    if config['train']['train_model']:
        train_unet(unet, train_loader, val_loader, config)
    
    # Evaluate the model if enabled in config
    if config['test']['evaluate_model']:
        evaluate_model(unet, test_loader, config)

if __name__ == "__main__":
    main()
