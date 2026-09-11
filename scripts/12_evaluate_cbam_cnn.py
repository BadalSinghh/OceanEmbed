# -*- coding: utf-8 -*-
"""
scripts/12_evaluate_cbam_cnn.py
===============================
Evaluation script for CBAM-CNN Model.

Evaluates CBAM-CNN on the 109-day held-out test set (2023-09-14 -> 2023-12-31),
computes overall and per-depth RMSE, MAE, R^2 metrics, and updates evaluation summaries.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (
    CBAMCNN,
    compute_metrics,
    compute_per_depth_metrics,
)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


class OceanDataset(Dataset):
    def __init__(self, split_name: str):
        split_dir = ROOT / "data" / "processed" / split_name
        self.X = np.load(split_dir / f"X_{split_name}.npz")["data"].astype(np.float32)
        self.Y = np.load(split_dir / f"Y_{split_name}.npz")["data"].astype(np.float32)
        self.dates = np.load(split_dir / f"dates_{split_name}.npy")
        self.X = np.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx]), str(self.dates[idx])


def main():
    print("=" * 70)
    print("STAGE 4 — MODEL C (CBAM-CNN) EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics_dir = ROOT / "results" / "metrics"
    models_dir  = ROOT / "results" / "models"

    test_dataset = OceanDataset("test")
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_dataset)} days ({test_dataset.dates[0]} -> {test_dataset.dates[-1]})")

    # Load CBAM-CNN Checkpoint
    cbam_ckpt_path = models_dir / "cbam_cnn_best.pt"
    if not cbam_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {cbam_ckpt_path}")

    model_cbam = CBAMCNN(in_channels=7, num_depths=15).to(device)
    ckpt_cbam  = torch.load(cbam_ckpt_path, map_location=device)
    model_cbam.load_state_dict(ckpt_cbam["model_state_dict"])
    model_cbam.eval()

    param_cbam = sum(p.numel() for p in model_cbam.parameters() if p.requires_grad)

    print("\n1. Evaluating CBAM-CNN on Test Set...")
    cbam_preds = []
    targets    = []
    dates_list = []

    with torch.no_grad():
        for x_b, y_b, d_b in test_loader:
            x_b = x_b.to(device)
            p_cbam = model_cbam(x_b).cpu().numpy()

            cbam_preds.append(p_cbam)
            targets.append(y_b.numpy())
            dates_list.extend(d_b)

    cbam_preds = np.concatenate(cbam_preds, axis=0)  # [109, 15, 69, 81]
    targets    = np.concatenate(targets, axis=0)     # [109, 15, 69, 81]

    cbam_overall  = compute_metrics(cbam_preds, targets)
    cbam_perdepth = compute_per_depth_metrics(cbam_preds, targets, TARGET_DEPTHS)

    print(f"\n--- CBAM-CNN Test Results ---")
    print(f"Parameters   : {param_cbam:,}")
    print(f"Test RMSE    : {cbam_overall['rmse']:.4f} °C")
    print(f"Test MAE     : {cbam_overall['mae']:.4f} °C")
    print(f"Test R²      : {cbam_overall['r2']:.4f}")

    # Save CBAM test predictions
    np.savez_compressed(
        metrics_dir / "cbam_cnn_test_predictions.npz",
        cbam_preds=cbam_preds,
        targets=targets,
        dates=np.array(dates_list)
    )

    # Load existing history and evaluation summary
    history_cbam_path = metrics_dir / "cbam_cnn_history.json"
    hist_cbam = {}
    if history_cbam_path.exists():
        with open(history_cbam_path) as f:
            hist_cbam = json.load(f)

    eval_summary_path = metrics_dir / "evaluation_summary.json"
    summary_data = {}
    if eval_summary_path.exists():
        with open(eval_summary_path) as f:
            summary_data = json.load(f)

    # Update summary data with CBAM_CNN
    if "models" not in summary_data:
        summary_data["models"] = {}
    summary_data["models"]["CBAM_CNN"] = {
        "parameters": param_cbam,
        "training_time_sec": hist_cbam.get("total_training_time_sec", 0.0),
        "best_val_loss": hist_cbam.get("best_val_loss", 0.0),
        "test_rmse": cbam_overall["rmse"],
        "test_mae": cbam_overall["mae"],
        "test_r2": cbam_overall["r2"],
    }

    # Update per_depth_test in summary_data
    if "per_depth_test" in summary_data:
        for d in TARGET_DEPTHS:
            str_d = str(d)
            if str_d in summary_data["per_depth_test"]:
                summary_data["per_depth_test"][str_d]["cbam_rmse"] = cbam_perdepth[d]["rmse"]
                summary_data["per_depth_test"][str_d]["cbam_mae"]  = cbam_perdepth[d]["mae"]
                summary_data["per_depth_test"][str_d]["cbam_r2"]   = cbam_perdepth[d]["r2"]

    with open(eval_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nUpdated {eval_summary_path.name}")

    # Update per_depth_metrics.csv
    csv_file = metrics_dir / "per_depth_metrics.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        cbam_rmse_col = [cbam_perdepth[d]["rmse"] for d in df["depth_m"]]
        cbam_mae_col  = [cbam_perdepth[d]["mae"] for d in df["depth_m"]]
        cbam_r2_col   = [cbam_perdepth[d]["r2"] for d in df["depth_m"]]

        df["cbam_rmse"] = cbam_rmse_col
        df["cbam_mae"]  = cbam_mae_col
        df["cbam_r2"]   = cbam_r2_col

        df.to_csv(csv_file, index=False)
        print(f"Updated {csv_file.name} with CBAM-CNN metrics.")

    print("=" * 70)


if __name__ == "__main__":
    main()
