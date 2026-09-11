# -*- coding: utf-8 -*-
"""
src/models/losses.py
====================
Masked loss functions and evaluation metrics for OceanEmbed.

Computes loss and metrics strictly over valid (finite) target cells, ignoring land
and bathymetry-invalid ocean bottom locations.
"""

import numpy as np
import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss.
    Calculates MSE strictly over valid (finite) target locations using masked_select.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target)
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.masked_select(pred, mask)
        target_valid = torch.masked_select(target, mask)
        return torch.mean((pred_valid - target_valid) ** 2)


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error Loss.
    Calculates MAE strictly over valid (finite) target locations using masked_select.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target)
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = torch.masked_select(pred, mask)
        target_valid = torch.masked_select(target, mask)
        return torch.mean(torch.abs(pred_valid - target_valid))


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute overall evaluation metrics (MSE, RMSE, MAE, R^2) over all valid target cells.

    Args:
        pred: Predicted temperature numpy array [N, 15, H, W]
        target: Target GLORYS temperature numpy array [N, 15, H, W]

    Returns:
        metrics: Dictionary with mse, rmse, mae, r2
    """
    mask = np.isfinite(target)
    if not mask.any():
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    p = pred[mask]
    t = target[mask]

    mse = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(p - t)))

    var_t = float(np.var(t))
    r2 = float(1.0 - (mse / var_t)) if var_t > 1e-8 else 0.0

    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
    }


def compute_per_depth_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    depths: list = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
) -> dict:
    """
    Compute per-depth RMSE, MAE, R^2 metrics.

    Args:
        pred: Predicted temperature numpy array [N, 15, H, W]
        target: Target GLORYS temperature numpy array [N, 15, H, W]
        depths: List of target depth values in meters

    Returns:
        per_depth_metrics: Dict depth_m -> {rmse, mae, r2, valid_count}
    """
    per_depth = {}
    for idx, d in enumerate(depths):
        p_d = pred[:, idx, :, :]
        t_d = target[:, idx, :, :]
        m_d = np.isfinite(t_d)

        if not m_d.any():
            per_depth[d] = {"rmse": None, "mae": None, "r2": None, "valid_count": 0}
            continue

        p_vals = p_d[m_d]
        t_vals = t_d[m_d]

        mse_d = float(np.mean((p_vals - t_vals) ** 2))
        rmse_d = float(np.sqrt(mse_d))
        mae_d = float(np.mean(np.abs(p_vals - t_vals)))
        var_d = float(np.var(t_vals))
        r2_d = float(1.0 - (mse_d / var_d)) if var_d > 1e-8 else 0.0

        per_depth[d] = {
            "rmse": round(rmse_d, 6),
            "mae": round(mae_d, 6),
            "r2": round(r2_d, 6),
            "valid_count": int(m_d.sum()),
        }

    return per_depth
