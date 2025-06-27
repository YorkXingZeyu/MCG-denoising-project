import argparse
import yaml
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import pandas as pd
import warnings

from Data_Preparation.data_loader import prepare_full_dataset
from models.unet1d import Unet1D
from trainandtest.train_unet import train_unet
from trainandtest.evaluate import evaluate_fold
from utils.helpers import seed_everything

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed_everything(config["seed"])

    # Ensure output dirs exist
    os.makedirs(config["output_folder"], exist_ok=True)
    os.makedirs(os.path.join(config["output_folder"], "model_params"), exist_ok=True)

    # Prepare full dataset for 5-fold CV
    full_ds = prepare_full_dataset(config)
    indices = list(range(len(full_ds)))
    kf = KFold(n_splits=5, shuffle=True, random_state=config["seed"])

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n=== Fold {fold}/5 ===")

        train_loader = DataLoader(
            Subset(full_ds, train_idx),
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            Subset(full_ds, val_idx),
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            drop_last=False
        )

        # Instantiate model
        unet = Unet1D(dim=64, dim_mults=(1, 2, 4), channels=1)

        # Train & save weights
        if config['train']['train_model']:
            train_unet(unet, train_loader, val_loader, config)
            save_path = os.path.join(
                config["output_folder"],
                "model_params",
                f"Unet1D_fold{fold}.pth"
            )
            torch.save(unet.state_dict(), save_path)

        # Evaluate this fold
        if config['test']['evaluate_model']:
            metrics = evaluate_fold(unet, val_loader, config, fold=fold)
            all_metrics.append(metrics)

    # Aggregate 5-fold results
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        mean_row = df.mean(axis=0)
        df.loc["average"] = mean_row
        summary_path = os.path.join(
            config["output_folder"],
            "evaluation_results_5fold_summary.xlsx"
        )
        df.to_excel(summary_path)
        print(f"\n5-fold summary saved to {summary_path}")

if __name__ == "__main__":
    main()
