"""
metrics.py

Evaluation metrics calculation for OceanEmbed models:
Calculates RMSE, MAE, Bias, and Pearson Correlation overall and depth-wise across the 15 standard ocean depths.
"""

import numpy as np
import pandas as pd

def calculate_metrics_overall(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: ndarray of shape (N, N_depths, Lat, Lon) or flattened
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    t = y_true[mask]
    p = y_pred[mask]

    diff = p - t
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    # Pearson Correlation
    if len(t) > 1 and np.std(t) > 1e-8 and np.std(p) > 1e-8:
        corr = np.corrcoef(t, p)[0, 1]
    else:
        corr = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
        "Correlation": float(corr)
    }

def calculate_metrics_depthwise(y_true: np.ndarray, y_pred: np.ndarray, depths=None):
    """
    y_true, y_pred: ndarray of shape (N_samples, N_depths, Lat, Lon)
    depths: list of depth values in meters
    """
    if depths is None:
        depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    n_depths = y_true.shape[1]
    results = []

    for d_idx in range(n_depths):
        depth_val = depths[d_idx] if d_idx < len(depths) else d_idx
        t_slice = y_true[:, d_idx, :, :]
        p_slice = y_pred[:, d_idx, :, :]

        metrics = calculate_metrics_overall(t_slice, p_slice)
        metrics["Depth_m"] = depth_val
        results.append(metrics)

    df_results = pd.DataFrame(results)
    cols = ["Depth_m", "RMSE", "MAE", "Bias", "Correlation"]
    return df_results[cols]
