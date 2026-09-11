# -*- coding: utf-8 -*-
"""
scripts/09_evaluate.py
======================
Evaluation & Ocean Embedding Extraction Script.

1. Evaluates CNN Baseline and OceanEmbed on held-out test set (109 days)
2. Computes overall RMSE, MAE, R^2, and per-depth metrics
3. Extracts and saves learned OceanEmbed spatial representations (embeddings)
   for train, val, and test splits along with dates and spatial coordinates
4. Saves results to results/metrics/ and results/embeddings/
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (
    CNNBaseline,
    OceanEmbed,
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

        # Replace input NaNs with 0.0 (mean post-normalization)
        self.X = np.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx]), str(self.dates[idx])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    all_dates = []

    with torch.no_grad():
        for x_b, y_b, dates_b in loader:
            x_b = x_b.to(device)
            pred = model(x_b)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.numpy())
            all_dates.extend(dates_b)

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    overall = compute_metrics(preds_arr, targets_arr)
    per_depth = compute_per_depth_metrics(preds_arr, targets_arr, TARGET_DEPTHS)

    return overall, per_depth, preds_arr, targets_arr, all_dates


def extract_embeddings(model: OceanEmbed, loader: DataLoader, device: torch.device):
    model.eval()
    all_embeds = []
    all_dates = []

    with torch.no_grad():
        for x_b, _, dates_b in loader:
            x_b = x_b.to(device)
            emb = model.encode(x_b)
            all_embeds.append(emb.cpu().numpy())
            all_dates.extend(dates_b)

    embeds_arr = np.concatenate(all_embeds, axis=0)  # [N, 128, 17, 20]
    return embeds_arr, all_dates


def main():
    print("=" * 70)
    print("STAGE 4 — MODEL EVALUATION & EMBEDDING EXTRACTION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    metrics_dir = ROOT / "results" / "metrics"
    embeddings_dir = ROOT / "results" / "embeddings"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load sidecar coordinates
    coords_ds = xr.open_dataset(ROOT / "data" / "processed" / "coords.nc")
    lat_coords = coords_ds["lat"].values
    lon_coords = coords_ds["lon"].values

    # Load Test Dataset
    test_dataset = OceanDataset("test")
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_dataset)} days ({test_dataset.dates[0]} -> {test_dataset.dates[-1]})")

    # 1. Load CNN Baseline Model
    cnn_ckpt = ROOT / "results" / "models" / "cnn_baseline_best.pt"
    cnn_model = CNNBaseline(in_channels=7, num_depths=15).to(device)
    cnn_data = torch.load(cnn_ckpt, map_location=device)
    cnn_model.load_state_dict(cnn_data["model_state_dict"])
    param_cnn = count_parameters(cnn_model)

    with open(ROOT / "results" / "metrics" / "cnn_baseline_history.json") as f:
        hist_cnn = json.load(f)

    # 2. Load OceanEmbed Model
    oe_ckpt = ROOT / "results" / "models" / "oceanembed_best.pt"
    oe_model = OceanEmbed(in_channels=7, num_depths=15, embed_dim=128, fno_modes=8, fno_layers=4).to(device)
    oe_data = torch.load(oe_ckpt, map_location=device)
    oe_model.load_state_dict(oe_data["model_state_dict"])
    param_oe = count_parameters(oe_model)

    with open(ROOT / "results" / "metrics" / "oceanembed_history.json") as f:
        hist_oe = json.load(f)

    print("\n1. Evaluating Models on Test Set...")

    t0 = time.time()
    cnn_overall, cnn_per_depth, cnn_preds, targets, dates = evaluate_model(cnn_model, test_loader, device)
    cnn_eval_time = time.time() - t0

    t0 = time.time()
    oe_overall, oe_per_depth, oe_preds, _, _ = evaluate_model(oe_model, test_loader, device)
    oe_eval_time = time.time() - t0

    print("\n--- Test Set Evaluation Results ---")
    print(f"CNN Baseline : RMSE={cnn_overall['rmse']:.4f} °C | MAE={cnn_overall['mae']:.4f} °C | R²={cnn_overall['r2']:.4f}")
    print(f"OceanEmbed   : RMSE={oe_overall['rmse']:.4f} °C | MAE={oe_overall['mae']:.4f} °C | R²={oe_overall['r2']:.4f}")

    # Save predictions sidecar for visualization script
    np.savez_compressed(ROOT / "results" / "metrics" / "test_predictions.npz",
                        cnn_preds=cnn_preds, oe_preds=oe_preds, targets=targets, dates=np.array(dates))

    # 2. Extract Learned Spatial Ocean Embeddings
    print("\n2. Extracting Learned Spatial Ocean Embeddings...")
    for split_name in ["train", "val", "test"]:
        ds_split = OceanDataset(split_name)
        ld_split = DataLoader(ds_split, batch_size=16, shuffle=False)
        embeds_arr, dates_split = extract_embeddings(oe_model, ld_split, device)

        out_emb_path = embeddings_dir / f"ocean_embeddings_{split_name}.npz"
        np.savez_compressed(
            out_emb_path,
            embeddings=embeds_arr,  # [N, 128, 17, 20]
            dates=np.array(dates_split),
            lat=lat_coords,
            lon=lon_coords,
        )
        print(f"  Saved {split_name} embeddings: shape={embeds_arr.shape} -> {out_emb_path.name}")

    # 3. Save Summary Metrics JSON
    summary_metrics = {
        "evaluation_timestamp_utc": pd.Timestamp.now("UTC").isoformat()[:19],
        "models": {
            "CNN_Baseline": {
                "parameters": param_cnn,
                "training_time_sec": hist_cnn["total_training_time_sec"],
                "best_val_loss": hist_cnn["best_val_loss"],
                "test_rmse": cnn_overall["rmse"],
                "test_mae": cnn_overall["mae"],
                "test_r2": cnn_overall["r2"],
            },
            "OceanEmbed": {
                "parameters": param_oe,
                "training_time_sec": hist_oe["total_training_time_sec"],
                "best_val_loss": hist_oe["best_val_loss"],
                "test_rmse": oe_overall["rmse"],
                "test_mae": oe_overall["mae"],
                "test_r2": oe_overall["r2"],
            },
        },
        "per_depth_test": {
            str(d): {
                "cnn_rmse": cnn_per_depth[d]["rmse"],
                "cnn_mae": cnn_per_depth[d]["mae"],
                "oceanembed_rmse": oe_per_depth[d]["rmse"],
                "oceanembed_mae": oe_per_depth[d]["mae"],
                "oceanembed_r2": oe_per_depth[d]["r2"],
            } for d in TARGET_DEPTHS
        }
    }

    eval_json_path = metrics_dir / "evaluation_summary.json"
    with open(eval_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\nSaved evaluation summary: {eval_json_path}")

    # 4. Save Per-Depth Metrics CSV
    csv_path = metrics_dir / "per_depth_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_m", "cnn_rmse", "cnn_mae", "oceanembed_rmse", "oceanembed_mae", "oceanembed_r2"])
        for d in TARGET_DEPTHS:
            writer.writerow([
                d,
                cnn_per_depth[d]["rmse"],
                cnn_per_depth[d]["mae"],
                oe_per_depth[d]["rmse"],
                oe_per_depth[d]["mae"],
                oe_per_depth[d]["r2"],
            ])
    print(f"Saved per-depth metrics CSV: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
