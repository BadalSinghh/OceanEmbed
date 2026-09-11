# -*- coding: utf-8 -*-
"""
scripts/10_visualize_results.py
===============================
Visualization Script for Stage 4 OceanEmbed Results.

Generates:
1. Reconstruction Maps (Target vs Prediction vs Error) at 0m, 100m, 500m, 1000m
2. Per-Depth RMSE Comparison Plot (CNN Baseline vs OceanEmbed)
3. Exploratory PCA scatter plot of learned Spatial Ocean Embeddings
   (Explicitly marked as exploratory — not claiming physical proof)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def plot_reconstruction_maps():
    print("1. Generating Reconstruction Maps...")
    preds_file = ROOT / "results" / "metrics" / "test_predictions.npz"
    if not preds_file.exists():
        print(f"  [SKIP] {preds_file} not found.")
        return

    data = np.load(preds_file)
    oe_preds = data["oe_preds"]  # [109, 15, 69, 81]
    targets  = data["targets"]   # [109, 15, 69, 81]
    dates    = data["dates"]

    coords_ds = xr.open_dataset(ROOT / "data" / "processed" / "coords.nc")
    lats = coords_ds["lat"].values
    lons = coords_ds["lon"].values

    # Select mid-test date
    t_idx = len(dates) // 2
    test_date = dates[t_idx]

    sample_depths = [0, 100, 500, 1000]
    depth_indices = [TARGET_DEPTHS.index(d) for d in sample_depths]

    fig, axes = plt.subplots(len(sample_depths), 3, figsize=(14, 3.5 * len(sample_depths)))

    for i, (d_val, d_idx) in enumerate(zip(sample_depths, depth_indices)):
        t_map = targets[t_idx, d_idx, :, :]
        p_map = oe_preds[t_idx, d_idx, :, :]
        e_map = np.abs(p_map - t_map)

        # Min/Max for temperature colorbar
        vmax = np.nanmax(t_map)
        vmin = np.nanmin(t_map)

        # Target Map
        ax0 = axes[i, 0]
        im0 = ax0.pcolormesh(lons, lats, t_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax0.set_title(f"Target GLORYS ({d_val}m) — {test_date}")
        ax0.set_ylabel("Latitude (°N)")
        if i == len(sample_depths) - 1:
            ax0.set_xlabel("Longitude (°E)")
        fig.colorbar(im0, ax=ax0, label="Temp (°C)")

        # Prediction Map
        ax1 = axes[i, 1]
        im1 = ax1.pcolormesh(lons, lats, p_map, cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"OceanEmbed Prediction ({d_val}m)")
        if i == len(sample_depths) - 1:
            ax1.set_xlabel("Longitude (°E)")
        fig.colorbar(im1, ax=ax1, label="Temp (°C)")

        # Absolute Error Map
        ax2 = axes[i, 2]
        err_max = np.nanpercentile(e_map, 98) if np.any(np.isfinite(e_map)) else 2.0
        im2 = ax2.pcolormesh(lons, lats, e_map, cmap="magma", vmin=0, vmax=err_max)
        ax2.set_title(f"Absolute Error ({d_val}m)")
        if i == len(sample_depths) - 1:
            ax2.set_xlabel("Longitude (°E)")
        fig.colorbar(im2, ax=ax2, label="|Error| (°C)")

    plt.tight_layout()
    out_map = FIGURES / "reconstruction_maps.png"
    plt.savefig(out_map, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved reconstruction maps to {out_map.name}")


def plot_depth_rmse():
    print("2. Generating Per-Depth RMSE Plot...")
    csv_file = ROOT / "results" / "metrics" / "per_depth_metrics.csv"
    if not csv_file.exists():
        print(f"  [SKIP] {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    plt.figure(figsize=(7, 6))
    plt.plot(df["cnn_rmse"], df["depth_m"], "o--", color="gray", label="CNN Baseline", linewidth=2, markersize=6)
    plt.plot(df["oceanembed_rmse"], df["depth_m"], "s-", color="navy", label="OceanEmbed (CNN + FNO2D)", linewidth=2.5, markersize=7)

    plt.gca().invert_yaxis()  # Ocean depth 0 at top, 1000 at bottom
    plt.yscale("symlog", linthresh=50)

    plt.title("OceanEmbed vs CNN Baseline — Per-Depth RMSE Comparison", fontsize=12, pad=12)
    plt.xlabel("Root Mean Squared Error (°C)", fontsize=11)
    plt.ylabel("Depth (m)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")

    # Annotate key depths
    for _, row in df.iterrows():
        d = int(row["depth_m"])
        if d in [0, 50, 100, 300, 500, 1000]:
            plt.annotate(f"{row['oceanembed_rmse']:.2f}°C",
                         (row["oceanembed_rmse"], d),
                         textcoords="offset points", xytext=(8, -3), fontsize=8, color="navy")

    plt.tight_layout()
    out_rmse = FIGURES / "depth_wise_rmse.png"
    plt.savefig(out_rmse, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-depth RMSE plot to {out_rmse.name}")


def plot_embedding_pca():
    print("3. Generating Exploratory Ocean Embedding PCA Plot...")
    emb_file = ROOT / "results" / "embeddings" / "ocean_embeddings_test.npz"
    if not emb_file.exists():
        print(f"  [SKIP] {emb_file} not found.")
        return

    data = np.load(emb_file)
    embeddings = data["embeddings"]  # [109, 128, 17, 20]
    dates      = data["dates"]

    N, C, H, W = embeddings.shape
    flat_embeds = embeddings.reshape(N, -1)  # [109, 128*17*20]

    # Perform PCA to 2 components
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(flat_embeds)
    exp_var = pca.explained_variance_ratio_ * 100.0

    # Extract month for seasonal coloring
    months = np.array([int(d.split("-")[1]) for d in dates])

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=months, cmap="twilight_shifted", s=40, edgecolors="none", alpha=0.85)

    cbar = plt.colorbar(sc, ticks=range(1, 13))
    cbar.set_label("Test Month (2023-09 to 2023-12)", fontsize=10)

    plt.title("Exploratory PCA of Learned Spatial Ocean Embeddings (Test Set)", fontsize=11, pad=12)
    plt.xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% variance)", fontsize=10)
    plt.ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% variance)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Mandatory Scientific Disclaimer
    disclaimer = ("Exploratory visualization only — PCA on latent embeddings demonstrates "
                  "temporal trajectory\nand dimensional structure, but does NOT prove physical cause.")
    plt.figtext(0.5, -0.02, disclaimer, wrap=True, horizontalalignment="center", fontsize=8, style="italic", color="gray")

    plt.tight_layout()
    out_pca = FIGURES / "embedding_pca.png"
    plt.savefig(out_pca, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved embedding PCA plot to {out_pca.name}")


def main():
    print("=" * 70)
    print("STAGE 4 — RESULT VISUALIZATION PIPELINE")
    print("=" * 70)

    plot_reconstruction_maps()
    plot_depth_rmse()
    plot_embedding_pca()

    print("\nVisualization pipeline complete! Figures saved in results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
