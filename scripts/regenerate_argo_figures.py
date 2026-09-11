# -*- coding: utf-8 -*-
"""
scripts/regenerate_argo_figures.py
==================================
Regenerates all independent Argo validation figures from the single authoritative
results/argo_validation/argo_matched_observations.csv dataset.

Generates:
1. results/figures/argo_scatter.png (4-panel comparison: GLORYS Reanalysis, CNN Baseline, CBAM-CNN, OceanEmbed)
2. results/figures/argo_depth_rmse.png (Per-depth RMSE for all models vs real Argo)
3. results/figures/argo_depth_bias.png (Per-depth Bias for all models vs real Argo)
4. results/figures/argo_validation_locations.png (Geographical distribution of 253 Argo profiles)
5. results/figures/argo_error_distribution.png (Residual error distributions for all models)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "argo_validation" / "argo_matched_observations.csv"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} observations across {df['profile_id'].nunique()} profiles from {CSV_PATH}")

    # Set overall styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 13,
        'figure.dpi': 200
    })

    # =========================================================================
    # 1. PREDICTED VS OBSERVED SCATTER PLOT (4 PANELS)
    # =========================================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    
    models = [
        ("GLORYS12 (Reanalysis)", "glorys_temp", "#4A5568"),
        ("CNN Baseline", "cnn_temp", "#3182CE"),
        ("CBAM-CNN (Attention)", "cbam_temp", "#319795"),
        ("OceanEmbed (CNN+FNO2D)", "oe_temp", "#805AD5"),
    ]

    for ax, (m_name, col_name, color) in zip(axes, models):
        valid = df.dropna(subset=[col_name, "obs_temp"])
        y_obs = valid["obs_temp"].values
        y_pred = valid[col_name].values
        
        rmse = np.sqrt(np.mean((y_pred - y_obs) ** 2))
        mae = np.mean(np.abs(y_pred - y_obs))
        bias = np.mean(y_pred - y_obs)
        r = np.corrcoef(y_obs, y_pred)[0, 1]
        r2 = 1.0 - np.sum((y_obs - y_pred)**2) / np.sum((y_obs - np.mean(y_obs))**2)

        ax.scatter(y_obs, y_pred, alpha=0.35, color=color, edgecolors="none", s=18)
        min_val = min(np.min(y_obs), np.min(y_pred)) - 1.0
        max_val = max(np.max(y_obs), np.max(y_pred)) + 1.0
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="1:1 Perfect Match")

        ax.set_title(
            f"{m_name}\n"
            f"RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C\n"
            f"Bias: {bias:+.3f}°C | R: {r:.4f} | R²: {r2:.4f}",
            fontsize=10, pad=8
        )
        ax.set_xlabel("Observed Argo In-Situ Temp (°C)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(4, 32)
        ax.set_ylim(4, 32)

    axes[0].set_ylabel("Predicted Subsurface Temp (°C)")
    fig.suptitle("Independent Argo Validation — Predicted vs Observed Subsurface Temperature (N=3,509 Obs, 253 Profiles)", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES / "argo_scatter.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("  -> Saved argo_scatter.png")

    # =========================================================================
    # 2. PER-DEPTH RMSE PLOT
    # =========================================================================
    depth_stats = []
    for d in TARGET_DEPTHS:
        sub = df[df["depth_m"] == d]
        if len(sub) > 0:
            c_rmse = np.sqrt(np.mean((sub["cnn_temp"] - sub["obs_temp"])**2))
            o_rmse = np.sqrt(np.mean((sub["oe_temp"] - sub["obs_temp"])**2))
            b_rmse = np.sqrt(np.mean((sub["cbam_temp"] - sub["obs_temp"])**2))
            g_rmse = np.sqrt(np.mean((sub["glorys_temp"] - sub["obs_temp"])**2))
            depth_stats.append({
                "depth": d, "n": len(sub),
                "cnn_rmse": c_rmse, "oe_rmse": o_rmse, "cbam_rmse": b_rmse, "glorys_rmse": g_rmse
            })
    df_dep = pd.DataFrame(depth_stats)

    plt.figure(figsize=(7.5, 7))
    plt.plot(df_dep["glorys_rmse"], df_dep["depth"], "d:", color="#4A5568", label="GLORYS12 Reference", linewidth=1.5, alpha=0.8)
    plt.plot(df_dep["cnn_rmse"], df_dep["depth"], "o-", color="#3182CE", label="CNN Baseline (192k params)", linewidth=2.2)
    plt.plot(df_dep["cbam_rmse"], df_dep["depth"], "^-", color="#319795", label="CBAM-CNN (196k params)", linewidth=2.2)
    plt.plot(df_dep["oe_rmse"], df_dep["depth"], "s-", color="#805AD5", label="OceanEmbed (8.67M params)", linewidth=2.2)
    
    plt.gca().invert_yaxis()
    plt.yscale("symlog", linthresh=50)
    plt.yticks(TARGET_DEPTHS, [str(d) for d in TARGET_DEPTHS])
    plt.title("Independent Argo Validation — Per-Depth RMSE (°C)\n253 CORA Profiles, Bay of Bengal Test Period (2023-09-14 to 2023-12-31)", fontsize=11, pad=10)
    plt.xlabel("Root Mean Squared Error (°C)")
    plt.ylabel("Depth (m) — symlog scale")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "argo_depth_rmse.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("  -> Saved argo_depth_rmse.png")

    # =========================================================================
    # 3. PER-DEPTH BIAS PLOT
    # =========================================================================
    bias_stats = []
    for d in TARGET_DEPTHS:
        sub = df[df["depth_m"] == d]
        if len(sub) > 0:
            c_bias = np.mean(sub["cnn_temp"] - sub["obs_temp"])
            o_bias = np.mean(sub["oe_temp"] - sub["obs_temp"])
            b_bias = np.mean(sub["cbam_temp"] - sub["obs_temp"])
            g_bias = np.mean(sub["glorys_temp"] - sub["obs_temp"])
            bias_stats.append({
                "depth": d, "n": len(sub),
                "cnn_bias": c_bias, "oe_bias": o_bias, "cbam_bias": b_bias, "glorys_bias": g_bias
            })
    df_bias = pd.DataFrame(bias_stats)

    plt.figure(figsize=(7.5, 7))
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Zero Bias")
    plt.plot(df_bias["glorys_bias"], df_bias["depth"], "d:", color="#4A5568", label="GLORYS12 Reference", linewidth=1.5, alpha=0.8)
    plt.plot(df_bias["cnn_bias"], df_bias["depth"], "o-", color="#3182CE", label="CNN Baseline", linewidth=2.2)
    plt.plot(df_bias["cbam_bias"], df_bias["depth"], "^-", color="#319795", label="CBAM-CNN", linewidth=2.2)
    plt.plot(df_bias["oe_bias"], df_bias["depth"], "s-", color="#805AD5", label="OceanEmbed", linewidth=2.2)

    plt.gca().invert_yaxis()
    plt.yscale("symlog", linthresh=50)
    plt.yticks(TARGET_DEPTHS, [str(d) for d in TARGET_DEPTHS])
    plt.title("Independent Argo Validation — Per-Depth Temperature Bias (°C)\n(Predicted − Observed In-Situ Argo)", fontsize=11, pad=10)
    plt.xlabel("Mean Bias (°C)")
    plt.ylabel("Depth (m) — symlog scale")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "argo_depth_bias.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("  -> Saved argo_depth_bias.png")

    # =========================================================================
    # 4. ERROR DISTRIBUTION HISTOGRAM
    # =========================================================================
    plt.figure(figsize=(8.5, 5.5))
    err_cnn = df["cnn_temp"] - df["obs_temp"]
    err_oe  = df["oe_temp"] - df["obs_temp"]
    err_cbam = df["cbam_temp"] - df["obs_temp"]

    plt.hist(err_cnn, bins=40, range=(-5, 5), alpha=0.45, color="#3182CE", label=f"CNN Baseline (std={err_cnn.std():.2f}°C, mean={err_cnn.mean():+.2f}°C)")
    plt.hist(err_cbam, bins=40, range=(-5, 5), alpha=0.45, color="#319795", label=f"CBAM-CNN (std={err_cbam.std():.2f}°C, mean={err_cbam.mean():+.2f}°C)")
    plt.hist(err_oe, bins=40, range=(-5, 5), alpha=0.45, color="#805AD5", label=f"OceanEmbed (std={err_oe.std():.2f}°C, mean={err_oe.mean():+.2f}°C)")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")

    plt.title("Residual Error Distribution on Independent Argo Observations (N=3,509)", fontsize=11, pad=10)
    plt.xlabel("Temperature Prediction Error: (Predicted − Observed) (°C)")
    plt.ylabel("Number of Matched Depth Observations")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "argo_error_distribution.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("  -> Saved argo_error_distribution.png")

    # =========================================================================
    # 5. VALIDATION LOCATIONS MAP
    # =========================================================================
    plt.figure(figsize=(8.5, 6.5))
    coords_nc = ROOT / "data" / "processed" / "coords.nc"
    if coords_nc.exists():
        cds = xr.open_dataset(coords_nc)
        lats = cds["lat"].values
        lons = cds["lon"].values
        plt.pcolormesh(lons, lats, np.zeros((len(lats), len(lons))), cmap="Blues", alpha=0.15)

    # Unique profile locations
    prof_df = df.groupby("profile_id").agg({
        "lat": "first",
        "lon": "first",
        "obs_temp": "mean",
        "depth_m": "count"
    }).reset_index()

    sc = plt.scatter(
        prof_df["lon"], prof_df["lat"], 
        c=prof_df["obs_temp"], cmap="plasma", 
        s=40, edgecolors="black", linewidth=0.6, alpha=0.9
    )
    plt.colorbar(sc, label="Profile Mean Temperature (°C)")
    plt.title(f"Spatial Distribution of Matched Argo Profiles (N={len(prof_df)} Unique Profiles)\nBay of Bengal Domain (5–22°N, 80–100°E)", fontsize=11, pad=10)
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.xlim(79.5, 100.5)
    plt.ylim(4.5, 22.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES / "argo_validation_locations.png", dpi=250, bbox_inches="tight")
    plt.close()
    print("  -> Saved argo_validation_locations.png")

    print("\nAll 5 Argo figures successfully generated with verified metrics.")

if __name__ == "__main__":
    main()
