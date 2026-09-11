# -*- coding: utf-8 -*-
"""
scripts/evaluate_climatology.py
===============================
Evaluates ClimatologyBaseline on:
1. Held-out GLORYS test set (109 days, 2023-09-14 to 2023-12-31)
2. Independent CORA delayed-mode Argo validation (253 profiles, 3,509 observations)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.climatology_baseline import ClimatologyBaseline

def main():
    print("=" * 70)
    print("CLIMATOLOGY BASELINE EVALUATION")
    print("=" * 70)

    # 1. Load Training and Test Targets
    y_train = np.load(ROOT / "data" / "processed" / "train" / "Y_train.npz")["data"] # (511, 15, 69, 81)
    y_test = np.load(ROOT / "data" / "processed" / "test" / "Y_test.npz")["data"]   # (109, 15, 69, 81)
    land_mask = np.load(ROOT / "data" / "processed" / "land_mask.npy")              # (69, 81), True=Land

    # 2. Fit Climatology
    clim = ClimatologyBaseline()
    clim.fit(y_train)
    clim_pred = clim.predict(num_samples=y_test.shape[0]) # (109, 15, 69, 81)

    # 3. Evaluate on GLORYS Test Set
    # Mask out land
    ocean_mask_4d = ~land_mask[None, None, :, :] & np.isfinite(y_test)
    y_true_ocean = y_test[ocean_mask_4d]
    y_pred_ocean = clim_pred[ocean_mask_4d]

    diff_glorys = y_pred_ocean - y_true_ocean
    rmse_glorys = np.sqrt(np.mean(diff_glorys**2))
    mae_glorys = np.mean(np.abs(diff_glorys))
    bias_glorys = np.mean(diff_glorys)
    r_glorys = np.corrcoef(y_pred_ocean, y_true_ocean)[0, 1]
    r2_glorys = 1.0 - np.sum(diff_glorys**2) / np.sum((y_true_ocean - np.mean(y_true_ocean))**2)

    print("\n--- 1. HELD-OUT GLORYS TEST SET (109 days, 3,982 ocean cells, 15 depths) ---")
    print(f"  RMSE     : {rmse_glorys:.4f} °C")
    print(f"  MAE      : {mae_glorys:.4f} °C")
    print(f"  Bias     : {bias_glorys:+.4f} °C")
    print(f"  Pearson R: {r_glorys:.4f}")
    print(f"  R²       : {r2_glorys:.4f}")

    # Per-depth GLORYS metrics
    print("\n  GLORYS Test Set Per-Depth RMSE:")
    target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    for d_idx, d_m in enumerate(target_depths):
        mask_d = (~land_mask) & np.isfinite(y_test[:, d_idx, :, :])
        yt_d = y_test[:, d_idx, :, :][mask_d]
        yp_d = clim_pred[:, d_idx, :, :][mask_d]
        rmse_d = np.sqrt(np.mean((yp_d - yt_d)**2))
        print(f"    {d_m:5d}m: {rmse_d:.4f} °C")

    # 4. Evaluate on Independent CORA Argo Observations
    df_argo = pd.read_csv(ROOT / "results" / "argo_validation" / "argo_matched_observations.csv")
    coords_ds = xr.open_dataset(ROOT / "data" / "processed" / "coords.nc")
    model_lats = coords_ds["lat"].values
    model_lons = coords_ds["lon"].values

    clim_mean_grid = clim.climatology_mean[0] # (15, 69, 81)
    
    argo_clim_preds = []
    for _, row in df_argo.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        d_val = row["depth_m"]
        
        i_lat = int(np.argmin(np.abs(model_lats - lat)))
        i_lon = int(np.argmin(np.abs(model_lons - lon)))
        d_idx = target_depths.index(d_val)
        
        pred_val = clim_mean_grid[d_idx, i_lat, i_lon]
        argo_clim_preds.append(pred_val)

    df_argo["clim_temp"] = argo_clim_preds
    
    diff_argo = df_argo["clim_temp"].values - df_argo["obs_temp"].values
    rmse_argo = np.sqrt(np.mean(diff_argo**2))
    mae_argo = np.mean(np.abs(diff_argo))
    bias_argo = np.mean(diff_argo)
    r_argo = np.corrcoef(df_argo["clim_temp"].values, df_argo["obs_temp"].values)[0, 1]
    ss_tot_argo = np.sum((df_argo["obs_temp"].values - np.mean(df_argo["obs_temp"].values))**2)
    r2_argo = 1.0 - np.sum(diff_argo**2) / ss_tot_argo

    print("\n--- 2. INDEPENDENT IN-SITU CORA ARGO (253 profiles, 3,509 observations) ---")
    print(f"  RMSE     : {rmse_argo:.4f} °C")
    print(f"  MAE      : {mae_argo:.4f} °C")
    print(f"  Bias     : {bias_argo:+.4f} °C")
    print(f"  Pearson R: {r_argo:.4f}")
    print(f"  R²       : {r2_argo:.4f}")

    print("\n  Argo Per-Depth RMSE:")
    for d_m in target_depths:
        sub = df_argo[df_argo["depth_m"] == d_m]
        d_diff = sub["clim_temp"].values - sub["obs_temp"].values
        rmse_d = np.sqrt(np.mean(d_diff**2))
        print(f"    {d_m:5d}m (N={len(sub):3d}): {rmse_d:.4f} °C")

if __name__ == "__main__":
    main()
