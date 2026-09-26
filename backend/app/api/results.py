# -*- coding: utf-8 -*-
"""
backend/app/api/results.py
==========================
Endpoints for serving pre-computed result data:
  - Evaluation summary
  - Per-depth metrics
  - Training histories
  - Argo validation observations
  - Argo aggregate metrics
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["results"])

ROOT = Path(__file__).resolve().parent.parent.parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
ARGO_DIR = ROOT / "results" / "argo_validation"


# ---------------------------------------------------------------------------
# GET /api/results/evaluation
# ---------------------------------------------------------------------------

@router.get("/results/evaluation")
async def get_evaluation_summary():
    """Returns the full evaluation_summary.json content."""
    path = METRICS_DIR / "evaluation_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evaluation_summary.json not found")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GET /api/results/per-depth
# ---------------------------------------------------------------------------

class PerDepthRow(BaseModel):
    depth_m: int
    cnn_rmse: float
    cnn_mae: float
    oceanembed_rmse: float
    oceanembed_mae: float
    oceanembed_r2: float
    cbam_rmse: float
    cbam_mae: float
    cbam_r2: float


@router.get("/results/per-depth", response_model=List[PerDepthRow])
async def get_per_depth_metrics():
    """Returns per-depth RMSE/MAE/R² for all three models."""
    path = METRICS_DIR / "per_depth_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="per_depth_metrics.csv not found")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(PerDepthRow(
                depth_m=int(row["depth_m"]),
                cnn_rmse=float(row["cnn_rmse"]),
                cnn_mae=float(row["cnn_mae"]),
                oceanembed_rmse=float(row["oceanembed_rmse"]),
                oceanembed_mae=float(row["oceanembed_mae"]),
                oceanembed_r2=float(row["oceanembed_r2"]),
                cbam_rmse=float(row["cbam_rmse"]),
                cbam_mae=float(row["cbam_mae"]),
                cbam_r2=float(row["cbam_r2"]),
            ))
    return rows


# ---------------------------------------------------------------------------
# GET /api/results/training-history
# ---------------------------------------------------------------------------

@router.get("/results/training-history")
async def get_training_history():
    """Returns training/validation loss curves for all models."""
    histories: Dict[str, Any] = {}
    for model_id, filename in [
        ("cnn_baseline", "cnn_baseline_history.json"),
        ("cbam_cnn", "cbam_cnn_history.json"),
        ("ocean_embed", "oceanembed_history.json"),
    ]:
        path = METRICS_DIR / filename
        if path.exists():
            with open(path) as f:
                histories[model_id] = json.load(f)
    return histories


# ---------------------------------------------------------------------------
# GET /api/results/argo
# ---------------------------------------------------------------------------

class ArgoProfile(BaseModel):
    profile_id: str
    date: str
    lat: float
    lon: float
    depth_m: float
    obs_temp: float
    glorys_temp: Optional[float]
    cbam_temp: Optional[float]
    oe_temp: Optional[float]
    cnn_temp: Optional[float]


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val) if val not in ("", "nan", "None") else None
    except (ValueError, TypeError):
        return None


@router.get("/results/argo")
async def get_argo_observations(
    profile_id: Optional[str] = None,
    limit: int = 500,
):
    """
    Returns Argo float matched observations.
    Optionally filter by profile_id. Paginate with limit.
    """
    path = ARGO_DIR / "argo_matched_observations.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="argo_matched_observations.csv not found")

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if profile_id and row.get("profile_id") != profile_id:
                continue
            rows.append({
                "profile_id": row.get("profile_id", ""),
                "date": row.get("date", ""),
                "lat": _safe_float(row.get("lat", "")),
                "lon": _safe_float(row.get("lon", "")),
                "depth_m": _safe_float(row.get("depth_m", "")),
                "obs_temp": _safe_float(row.get("obs_temp", "")),
                "glorys_temp": _safe_float(row.get("glorys_temp", "")),
                "cbam_temp": _safe_float(row.get("cbam_temp", "")),
                "oe_temp": _safe_float(row.get("oe_temp", "")),
                "cnn_temp": _safe_float(row.get("cnn_temp", "")),
            })
            if len(rows) >= limit:
                break

    return {"data": rows, "count": len(rows)}


@router.get("/results/argo/profiles")
async def get_argo_profile_list():
    """Returns list of unique Argo profile IDs and their first occurrence metadata."""
    path = ARGO_DIR / "argo_matched_observations.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="argo_matched_observations.csv not found")

    seen: Dict[str, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("profile_id", "")
            if pid not in seen:
                seen[pid] = {
                    "profile_id": pid,
                    "date": row.get("date", ""),
                    "lat": _safe_float(row.get("lat", "")),
                    "lon": _safe_float(row.get("lon", "")),
                }
    return {"profiles": list(seen.values()), "count": len(seen)}


# ---------------------------------------------------------------------------
# GET /api/results/argo/aggregate
# ---------------------------------------------------------------------------

@router.get("/results/argo/aggregate")
async def get_argo_aggregate():
    """Returns aggregate Argo validation metrics from app.py source of truth."""
    return {
        "total_profiles": 253,
        "total_observations": 3509,
        "test_period": {"start": "2023-09-14", "end": "2023-12-31"},
        "dataset": "CORA v1.3 delayed-mode QC-passed",
        "metrics": {
            "GLORYS12": {"rmse": 0.8061, "mae": 0.4565, "pearson_r": 0.9958},
            "OceanEmbed_FNO": {"rmse": 1.3777, "mae": 0.7905, "pearson_r": 0.9874},
            "CBAM_CNN": {"rmse": 1.7113, "mae": 1.1496, "pearson_r": 0.9811},
            "Climatology": {"rmse": 2.0253, "mae": 1.2297, "pearson_r": 0.9766},
        },
    }
