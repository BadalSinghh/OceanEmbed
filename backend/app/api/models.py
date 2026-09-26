# -*- coding: utf-8 -*-
"""
backend/app/api/models.py
=========================
Endpoint returning available model metadata.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(tags=["models"])

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


class ModelInfo(BaseModel):
    id: str
    name: str
    parameters: int
    architecture: str
    description: str
    test_rmse: float
    test_mae: float
    test_r2: float
    argo_rmse: float
    argo_mae: float
    pearson_r: float


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    target_depths: List[int]
    domain: dict
    input_channels: List[dict]


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Returns metadata about all available trained models."""
    return ModelsResponse(
        models=[
            ModelInfo(
                id="ocean_embed",
                name="OceanEmbed (CNN+FNO2D)",
                parameters=8670241,
                architecture="OceanEmbed (CNN+FNO2D)",
                description=(
                    "Convolutional encoder-decoder with spectral integral Fourier operator. "
                    "Extracts multiscale spatial dynamics and projects to 15 vertical ocean depth levels."
                ),
                test_rmse=1.3777,
                test_mae=0.7905,
                test_r2=0.9720,
                argo_rmse=1.3777,
                argo_mae=0.7905,
                pearson_r=0.9874,
            ),
            ModelInfo(
                id="cbam_cnn",
                name="CBAM-CNN (Attention)",
                parameters=195705,
                architecture="CBAM-CNN Attention Baseline",
                description=(
                    "Dual-attention convolutional baseline with channel and spatial attention gates. "
                    "Dynamically weights surface variables and spatial regions."
                ),
                test_rmse=1.7113,
                test_mae=1.1496,
                test_r2=0.9568,
                argo_rmse=1.7113,
                argo_mae=1.1496,
                pearson_r=0.9811,
            ),
        ],
        target_depths=TARGET_DEPTHS,
        domain={
            "region": "Bay of Bengal",
            "lat_min": 5.0,
            "lat_max": 22.0,
            "lon_min": 80.0,
            "lon_max": 100.0,
            "resolution_deg": 0.25,
            "grid_shape": [69, 81],
            "depth_min_m": 0,
            "depth_max_m": 1000,
            "depth_levels": 15,
        },
        input_channels=[
            {"id": "SST", "name": "Sea Surface Temperature", "unit": "°C", "source": "OSTIA MetOffice L4"},
            {"id": "SSS", "name": "Sea Surface Salinity", "unit": "psu", "source": "Multi-platform L4"},
            {"id": "SLA", "name": "Sea Level Anomaly", "unit": "m", "source": "DUACS merged altimetry"},
            {"id": "CURRENT_U", "name": "Geostrophic Current U", "unit": "m/s", "source": "Derived from SLA"},
            {"id": "CURRENT_V", "name": "Geostrophic Current V", "unit": "m/s", "source": "Derived from SLA"},
            {"id": "WIND_U", "name": "Wind U (10m)", "unit": "m/s", "source": "L4 blended scatterometer"},
            {"id": "WIND_V", "name": "Wind V (10m)", "unit": "m/s", "source": "L4 blended scatterometer"},
        ],
    )
