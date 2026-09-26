# -*- coding: utf-8 -*-
"""
backend/app/api/predict.py
==========================
Inference endpoint. Uses backend/app/loaders.py — NO Streamlit dependency.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.loaders import (
    TARGET_DEPTHS,
    load_coordinates,
    load_land_mask,
    load_normalization_stats,
    load_test_metadata,
    load_test_sample,
    load_pytorch_model,
    predict_temperature_field,
    unnormalize_input_channels,
)

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    model_id: str = Field(
        "cbam_cnn",
        description="Model identifier: cbam_cnn | ocean_embed",
    )
    sample_index: int = Field(0, ge=0, description="Index into test dataset (0-based, max 108)")
    depth_index: Optional[int] = Field(
        None, ge=0, le=14,
        description="Return only this depth level (0-14). Null = all 15.",
    )


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    date: str
    sample_index: int
    lats: list
    lons: list
    depths: list
    prediction: list
    ground_truth: list
    error: list
    surface_inputs: list


@router.get("/predict/dates")
async def get_test_dates():
    dates, n = load_test_metadata()
    return {"dates": dates, "n_samples": n}


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    dates, n_samples = load_test_metadata()

    if req.sample_index >= n_samples:
        raise HTTPException(422, f"sample_index {req.sample_index} out of range (max {n_samples - 1})")

    try:
        x_norm, y_true = load_test_sample(req.sample_index)
    except Exception as e:
        raise HTTPException(500, f"Failed to load test sample: {e}")

    model = load_pytorch_model(req.model_id)
    if model is None:
        raise HTTPException(404, f"Model '{req.model_id}' not found or checkpoint missing.")

    land_mask = load_land_mask()
    lats, lons, _ = load_coordinates()
    stats = load_normalization_stats()

    try:
        pred_3d = predict_temperature_field(model, x_norm, land_mask)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    y_true_masked = y_true.copy().astype(float)
    y_true_masked[:, land_mask] = np.nan

    x_phys = unnormalize_input_channels(x_norm, stats)
    x_phys_masked = x_phys.copy().astype(float)
    x_phys_masked[:, land_mask] = np.nan

    if req.depth_index is not None:
        d = req.depth_index
        out_depths = [TARGET_DEPTHS[d]]
        prediction_list = _arr(pred_3d[d])
        gt_list = _arr(y_true_masked[d])
        err_list = _arr(pred_3d[d] - y_true_masked[d])
    else:
        out_depths = TARGET_DEPTHS
        prediction_list = [_arr(pred_3d[i]) for i in range(len(TARGET_DEPTHS))]
        gt_list = [_arr(y_true_masked[i]) for i in range(len(TARGET_DEPTHS))]
        err_list = [_arr((pred_3d - y_true_masked)[i]) for i in range(len(TARGET_DEPTHS))]

    name_map = {"cbam_cnn": "OceanEmbed (CNN+FNO2D)", "ocean_embed": "CBAM-CNN"}

    return PredictResponse(
        model_id=req.model_id,
        model_name=name_map.get(req.model_id, req.model_id),
        date=dates[req.sample_index],
        sample_index=req.sample_index,
        lats=lats.tolist(),
        lons=lons.tolist(),
        depths=out_depths,
        prediction=prediction_list,
        ground_truth=gt_list,
        error=err_list,
        surface_inputs=[_arr(x_phys_masked[c]) for c in range(7)],
    )


def _arr(a: np.ndarray) -> list:
    return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in a]
