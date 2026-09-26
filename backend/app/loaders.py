# -*- coding: utf-8 -*-
"""
backend/app/loaders.py
======================
Standalone data and model loaders for the FastAPI backend.

Functionally identical to src/ui_helpers.py but uses functools.lru_cache
instead of @st.cache_resource / @st.cache_data — Streamlit is NOT required
as a backend dependency. The existing src/ui_helpers.py is untouched.
"""

import functools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


# ---------------------------------------------------------------------------
# Coordinate / mask / metadata loaders — cached once per process
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import xarray as xr
    ds = xr.open_dataset(ROOT / "data" / "processed" / "coords.nc")
    return ds["lat"].values, ds["lon"].values, ds["depth"].values


@functools.lru_cache(maxsize=1)
def load_land_mask() -> np.ndarray:
    return np.load(ROOT / "data" / "processed" / "land_mask.npy")


@functools.lru_cache(maxsize=1)
def load_test_metadata() -> Tuple[List[str], int]:
    dates = np.load(ROOT / "data" / "processed" / "test" / "dates_test.npy").tolist()
    return dates, len(dates)


@functools.lru_cache(maxsize=1)
def load_normalization_stats() -> Dict:
    with open(ROOT / "results" / "normalization_stats.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sample loader — NOT cached (each index is a different sample)
# ---------------------------------------------------------------------------

def load_test_sample(idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load X_test[idx] [7,69,81] and Y_test[idx] [15,69,81]."""
    x = np.load(ROOT / "data" / "processed" / "test" / "X_test.npz")["data"][idx].astype(np.float32)
    y = np.load(ROOT / "data" / "processed" / "test" / "Y_test.npz")["data"][idx].astype(np.float32)
    return x, y


def unnormalize_input_channels(x_norm: np.ndarray, stats: Dict) -> np.ndarray:
    keys = ["SST", "SSS", "SLA", "CURRENT_U", "CURRENT_V", "WIND_U", "WIND_V"]
    x_phys = np.empty_like(x_norm)
    for c, k in enumerate(keys):
        x_phys[c] = (x_norm[c] * stats[k]["std"]) + stats[k]["mean"]
    return x_phys


# ---------------------------------------------------------------------------
# Model loader — cached per model_id
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=3)
def load_pytorch_model(model_id: str):
    """
    Load and cache a PyTorch checkpoint by API model_id.
    cbam_cnn   → cbam_cnn_best.pt   (OceanEmbed CNN+FNO2D)
    ocean_embed → oceanembed_best.pt (CBAM-CNN Attention)
    """
    import torch
    from src.models import CBAMCNN, OceanEmbed

    models_dir = ROOT / "results" / "models"
    device = torch.device("cpu")

    if model_id == "cbam_cnn":
        model = CBAMCNN(in_channels=7, num_depths=15)
        ckpt_path = models_dir / "cbam_cnn_best.pt"
    elif model_id == "ocean_embed":
        model = OceanEmbed(in_channels=7, num_depths=15)
        ckpt_path = models_dir / "oceanembed_best.pt"
    else:
        return None

    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_temperature_field(model, x_norm: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """CPU inference. Returns [15, 69, 81] with land = NaN."""
    import torch
    x = torch.from_numpy(np.nan_to_num(x_norm, nan=0.0)).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()
    pred[:, land_mask] = np.nan
    return pred
