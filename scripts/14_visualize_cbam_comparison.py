# -*- coding: utf-8 -*-
"""
scripts/14_visualize_cbam_comparison.py
========================================
Generates consolidated 3-model comparison figures:
- CNN Baseline vs CBAM-CNN vs OceanEmbed Framework

Generates:
1. Per-depth RMSE comparison plot (results/figures/cbam_vs_all_depth_rmse.png)
2. Per-depth MAE comparison plot (results/figures/cbam_vs_all_depth_mae.png)
3. 2D Reconstruction & Error comparison maps (results/figures/cbam_reconstruction_maps.png)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def plot_3model_depth_rmse():
    print("1. Plotting 3-Model Per-Depth RMSE...")
    csv_file = ROOT / "results" / "metrics" / "per_depth_metrics.csv"
    if not csv_file.exists():
        print(f"  [SKIP] {csv_file.name} not found.")
        return

    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 6.5))
    plt.plot(df["cnn_rmse"], df["depth_m"], "o--", color="gray", label="CNN Baseline", linewidth=2, markersize=6)
    if "cbam_rmse" in df.columns:
        plt.plot(df["cbam_rmse"], df["depth_m"], "^-", color="teal", label="CBAM-CNN (Attention)", linewidth=2.5, markersize=7)
    plt.plot(df["oceanembed_rmse"], df["depth_m"], "s-", color="navy", label="OceanEmbed (CNN + FNO2D)", linewidth=2.5, markersize=7)

    plt.gca().invert_yaxis()
    plt.yscale("symlog", linthresh=50)

    plt.title("GLORYS Test Set — 3-Model Per-Depth RMSE Comparison", fontsize=12, pad=12)
    plt.xlabel("Root Mean Squared Error (°C)", fontsize=11)
    plt.ylabel("Depth (m)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_img = FIGURES / "cbam_vs_all_depth_rmse.png"
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_img.name}")


def plot_3model_depth_mae():
    print("2. Plotting 3-Model Per-Depth MAE...")
    csv_file = ROOT / "results" / "metrics" / "per_depth_metrics.csv"
    if not csv_file.exists():
        print(f"  [SKIP] {csv_file.name} not found.")
        return

    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 6.5))
    plt.plot(df["cnn_mae"], df["depth_m"], "o--", color="gray", label="CNN Baseline", linewidth=2, markersize=6)
    if "cbam_mae" in df.columns:
        plt.plot(df["cbam_mae"], df["depth_m"], "^-", color="teal", label="CBAM-CNN (Attention)", linewidth=2.5, markersize=7)
    plt.plot(df["oceanembed_mae"], df["depth_m"], "s-", color="navy", label="OceanEmbed (CNN + FNO2D)", linewidth=2.5, markersize=7)

    plt.gca().invert_yaxis()
    plt.yscale("symlog", linthresh=50)

    plt.title("GLORYS Test Set — 3-Model Per-Depth MAE Comparison", fontsize=12, pad=12)
    plt.xlabel("Mean Absolute Error (°C)", fontsize=11)
    plt.ylabel("Depth (m)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_img = FIGURES / "cbam_vs_all_depth_mae.png"
    plt.savefig(out_img, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_img.name}")


def plot_cbam_reconstruction_maps():
    print("3. Plotting CBAM-CNN Reconstruction Maps...")
    preds_file = ROOT / "results" / "metrics" / "test_predictions.npz"
    cbam_file  = ROOT / "results" / "metrics" / "cbam_cnn_test_predictions.npz"

    if not preds_file.exists() or not cbam_file.exists():
        print("  [SKIP] Prediction files not found.")
        return

    d1 = np.load(preds_file)
    targets  = d1["targets"]    # [109, 15, 69, 81]
    cnn_preds = d1["cnn_preds"] # [109, 15, 69, 81]
    oe_preds  = d1["oe_preds"]  # [109, 15, 69, 81]
    dates     = d1["dates"]

    d2 = np.load(cbam_file)
    cbam_preds = d2["cbam_preds"]  # [109, 15, 69, 81]

    coords_ds = xr.open_dataset(ROOT / "data" / "processed" / "coords.nc")
    lats = coords_ds["lat"].values
    lons = coords_ds["lon"].values

    t_idx = len(dates) // 2
    test_date = dates[t_idx]

    sample_depths = [0, 100, 500, 1000]
    depth_indices = [TARGET_DEPTHS.index(d) for d in sample_depths]

    fig, axes = plt.subplots(len(sample_depths), 4, figsize=(18, 3.5 * len(sample_depths)))

    for i, (d_val, d_idx) in enumerate(zip(sample_depths, depth_indices)):
        t_map = targets[t_idx, d_idx, :, :]
        c_map = cnn_preds[t_idx, d_idx, :, :]
        b_map = cbam_preds[t_idx, d_idx, :, :]
        o_map = oe_preds[t_idx, d_idx, :, :]

        vmax = np.nanmax(t_map)
        vmin = np.nanmin(t_map)

        # Target Map
        ax0 = axes[i, 0]
        im0 = ax0.pcolormesh(lons, lats, t_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax0.set_title(f"Target ({d_val}m) — {test_date}")
        ax0.set_ylabel("Latitude (°N)")
        if i == len(sample_depths) - 1:
            ax0.set_xlabel("Longitude (°E)")
        fig.colorbar(im0, ax=ax0, label="°C")

        # CNN Baseline Map
        ax1 = axes[i, 1]
        im1 = ax1.pcolormesh(lons, lats, c_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"CNN Baseline ({d_val}m)")
        if i == len(sample_depths) - 1:
            ax1.set_xlabel("Longitude (°E)")
        fig.colorbar(im1, ax=ax1, label="°C")

        # CBAM-CNN Map
        ax2 = axes[i, 2]
        im2 = ax2.pcolormesh(lons, lats, b_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax2.set_title(f"CBAM-CNN ({d_val}m)")
        if i == len(sample_depths) - 1:
            ax2.set_xlabel("Longitude (°E)")
        fig.colorbar(im2, ax=ax2, label="°C")

        # OceanEmbed Map
        ax3 = axes[i, 3]
        im3 = ax3.pcolormesh(lons, lats, o_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax3.set_title(f"OceanEmbed ({d_val}m)")
        if i == len(sample_depths) - 1:
            ax3.set_xlabel("Longitude (°E)")
        fig.colorbar(im3, ax=ax3, label="°C")

    plt.tight_layout()
    out_map = FIGURES / "cbam_reconstruction_maps.png"
    plt.savefig(out_map, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_map.name}")


def main():
    print("=" * 70)
    print("STAGE 4 — 3-MODEL COMPARATIVE VISUALIZATION PIPELINE")
    print("=" * 70)

    plot_3model_depth_rmse()
    plot_3model_depth_mae()
    plot_cbam_reconstruction_maps()

    print("\nVisualization pipeline complete! Figures saved in results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
