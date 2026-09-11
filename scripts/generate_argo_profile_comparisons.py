# -*- coding: utf-8 -*-
"""
scripts/generate_argo_profile_comparisons.py
============================================
Generates high-resolution vertical profile comparison plots for representative
real in-situ CORA Argo profiling floats vs CNN Baseline, CBAM-CNN, OceanEmbed,
and GLORYS12 reanalysis reference.

Output:
results/figures/argo_profile_comparisons.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "argo_validation" / "argo_matched_observations.csv"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH}")

    # Select 4 geographically and dynamically distinct profiles with all 15 depths
    selected_profiles = [
        {"id": "argo_0233", "zone": "Southern Bay of Bengal (Equatorial Band)"},
        {"id": "argo_0215", "zone": "Western Bay of Bengal (East India Coastal Current)"},
        {"id": "argo_0004", "zone": "Central Bay of Bengal (Open Ocean Basin)"},
        {"id": "argo_0225", "zone": "North-Central Bay of Bengal (River Plume / Barrier Layer)"},
    ]

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 250
    })

    fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)

    for ax, pinfo in zip(axes, selected_profiles):
        pid = pinfo["id"]
        zone_label = pinfo["zone"]
        sub = df[df["profile_id"] == pid].sort_values("depth_m")
        
        if len(sub) == 0:
            continue

        date_str = sub["date"].iloc[0]
        lat_val = sub["lat"].iloc[0]
        lon_val = sub["lon"].iloc[0]

        # Depths and temperatures
        depths = sub["depth_m"].values
        obs_t = sub["obs_temp"].values
        glorys_t = sub["glorys_temp"].values
        cnn_t = sub["cnn_temp"].values
        cbam_t = sub["cbam_temp"].values
        oe_t = sub["oe_temp"].values

        # Plot vertical profiles
        ax.plot(obs_t, depths, "ko-", linewidth=2.5, markersize=5, label="Observed Argo Float", zorder=5)
        ax.plot(glorys_t, depths, "d:", color="#4A5568", linewidth=1.8, markersize=4, label="GLORYS12 Reference", alpha=0.85)
        ax.plot(cnn_t, depths, "s--", color="#3182CE", linewidth=2.0, markersize=4, label="CNN Baseline")
        ax.plot(cbam_t, depths, "^-.", color="#319795", linewidth=2.0, markersize=4, label="CBAM-CNN")
        ax.plot(oe_t, depths, "v-", color="#805AD5", linewidth=2.0, markersize=4, label="OceanEmbed (FNO)")

        # Invert y-axis for depth
        ax.set_ylim(1050, -20)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_yticks([0, 10, 20, 50, 100, 200, 500, 1000])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_title(
            f"{zone_label}\n"
            f"ID: {pid} | {date_str}\n"
            f"Lat: {lat_val:.2f}°N, Lon: {lon_val:.2f}°E",
            fontsize=10.5, pad=8
        )
        ax.set_xlabel("Temperature (°C)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.set_xlim(4, 32)

    axes[0].set_ylabel("Depth (m) — symlog scale")
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "Representative In-Situ Argo Vertical Profiles vs Reconstructed Predictions (Bay of Bengal Test Period)",
        fontsize=13, y=1.02
    )

    plt.tight_layout()
    output_path = FIGURES / "argo_profile_comparison.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    main()
