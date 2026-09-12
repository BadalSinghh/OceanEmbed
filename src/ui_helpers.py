# -*- coding: utf-8 -*-
"""
src/ui_helpers.py
=================
Cached data loaders, model inference routines, and Plotly visualization
generators for the OceanEmbed Streamlit application.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent

# 15 Standard Depths from SIH Problem Statement
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# 7 Surface Channels
INPUT_CHANNELS = [
    "SST (°C)",
    "SSS (psu)",
    "SLA (m)",
    "Current U (m/s)",
    "Current V (m/s)",
    "Wind U (m/s)",
    "Wind V (m/s)",
]


# ---------------------------------------------------------------------------
# Data Loaders (Cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads spatial coordinates (lat, lon, depth) from coords.nc."""
    coords_path = ROOT / "data" / "processed" / "coords.nc"
    if not coords_path.exists():
        coords_path = ROOT / "required_files" / "coords.nc"
    ds = xr.open_dataset(coords_path)
    lats = ds["lat"].values
    lons = ds["lon"].values
    depths = ds["depth"].values
    return lats, lons, depths


@st.cache_data(show_spinner=False)
def load_land_mask() -> np.ndarray:
    """Loads boolean land mask array [69, 81] (True = land)."""
    mask_path = ROOT / "data" / "processed" / "land_mask.npy"
    if not mask_path.exists():
        mask_path = ROOT / "required_files" / "land_mask.npy"
    return np.load(mask_path)


@st.cache_data(show_spinner=False)
def load_test_metadata() -> Tuple[List[str], int]:
    """Loads test dates and total number of test samples."""
    dates_path = ROOT / "data" / "processed" / "test" / "dates_test.npy"
    if not dates_path.exists():
        dates_path = ROOT / "required_files" / "dates_test.npy"
    dates = np.load(dates_path).tolist()
    return dates, len(dates)


@st.cache_data(show_spinner=False)
def load_normalization_stats() -> Dict:
    """Loads channel normalization statistics."""
    stats_path = ROOT / "results" / "normalization_stats.json"
    with open(stats_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_sample(idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads normalized X_test[idx] and target Y_test[idx].
    X shape: [7, 69, 81]
    Y shape: [15, 69, 81]
    """
    x_path = ROOT / "data" / "processed" / "test" / "X_test.npz"
    y_path = ROOT / "data" / "processed" / "test" / "Y_test.npz"
    x_data = np.load(x_path)["data"][idx].astype(np.float32)
    y_data = np.load(y_path)["data"][idx].astype(np.float32)
    return x_data, y_data


@st.cache_data(show_spinner=False)
def unnormalize_input_channels(x_norm: np.ndarray, stats: Dict) -> np.ndarray:
    """Reverts normalized 7-channel array to physical units."""
    keys = ["SST", "SSS", "SLA", "CURRENT_U", "CURRENT_V", "WIND_U", "WIND_V"]
    x_phys = np.empty_like(x_norm)
    for c, k in enumerate(keys):
        mean = stats[k]["mean"]
        std = stats[k]["std"]
        x_phys[c] = (x_norm[c] * std) + mean
    return x_phys


@st.cache_data(show_spinner=False)
def load_argo_dataset() -> pd.DataFrame:
    """Loads matched CORA Argo validation observations."""
    csv_path = ROOT / "results" / "argo_validation" / "argo_matched_observations.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_evaluation_summary() -> Dict:
    """Loads authoritative evaluation summary JSON."""
    summary_path = ROOT / "results" / "metrics" / "evaluation_summary.json"
    with open(summary_path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_per_depth_metrics() -> pd.DataFrame:
    """Loads per-depth CSV metrics."""
    csv_path = ROOT / "results" / "metrics" / "per_depth_metrics.csv"
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_training_histories() -> Dict[str, Dict]:
    """Loads training history loss curves for all models."""
    metrics_dir = ROOT / "results" / "metrics"
    histories = {}
    for model_name, filename in [
        ("CNN Baseline", "cnn_baseline_history.json"),
        ("CBAM-CNN", "cbam_cnn_history.json"),
        ("OceanEmbed (FNO)", "oceanembed_history.json"),
    ]:
        p = metrics_dir / filename
        if p.exists():
            with open(p, "r") as f:
                histories[model_name] = json.load(f)
    return histories


# ---------------------------------------------------------------------------
# PyTorch Model Loaders & Inference (Cached Resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pytorch_model(model_name: str) -> Optional[torch.nn.Module]:
    """Loads a pre-trained PyTorch model checkpoint onto CPU."""
    from src.models import CBAMCNN, CNNBaseline, OceanEmbed

    models_dir = ROOT / "results" / "models"
    device = torch.device("cpu")

    if model_name == "CNN Baseline":
        model = CNNBaseline(in_channels=7, num_depths=15)
        ckpt_path = models_dir / "cnn_baseline_best.pt"
    elif model_name == "OceanEmbed (FNO)":
        model = CBAMCNN(in_channels=7, num_depths=15)
        ckpt_path = models_dir / "cbam_cnn_best.pt"
    elif model_name == "CBAM-CNN":
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


def predict_temperature_field(
    model: torch.nn.Module, x_norm: np.ndarray, land_mask: np.ndarray
) -> np.ndarray:
    """
    Runs CPU inference given single-day normalized surface input [7, 69, 81].
    Returns reconstructed 3D field [15, 69, 81] with land masked as NaN.
    """
    x_clean = np.nan_to_num(x_norm, nan=0.0)
    x_tensor = torch.from_numpy(x_clean).unsqueeze(0).float()  # [1, 7, 69, 81]

    with torch.no_grad():
        pred_tensor = model(x_tensor)  # [1, 15, 69, 81]

    pred = pred_tensor.squeeze(0).cpu().numpy()  # [15, 69, 81]
    # Apply land mask
    pred[:, land_mask] = np.nan
    return pred


# ---------------------------------------------------------------------------
# Interactive Plotly Visualization Helpers
# ---------------------------------------------------------------------------
def create_spatial_heatmap(
    data_2d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    unit: str = "°C",
    colorscale: str = "Thermal",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    diverging: bool = False,
) -> go.Figure:
    """Generates an interactive Plotly heatmap over the Bay of Bengal."""
    fig = go.Figure(
        data=go.Heatmap(
            z=data_2d,
            x=lons,
            y=lats,
            colorscale="RdBu_r" if diverging else colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=dict(text=unit, side="top", font=dict(size=12, color="#e0e6ed")),
                tickfont=dict(color="#e0e6ed"),
                thickness=15,
                len=0.85,
            ),
            hovertemplate="<b>Lon</b>: %{x:.2f}°E<br><b>Lat</b>: %{y:.2f}°N<br><b>Value</b>: %{z:.2f} "
            + unit
            + "<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e6ed")),
        xaxis=dict(
            title="Longitude (°E)",
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude (°N)",
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 25, 47, 0.7)",
        height=380,
    )
    return fig


def create_vertical_transect(
    volume_3d: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    slice_dim: str = "lat",
    slice_idx: int = 34,
    title: str = "Vertical Depth Section",
) -> go.Figure:
    """
    Renders a vertical cross section (thermocline profile) across depth (0-1000m).
    slice_dim='lat': fixed latitude slice, across longitude (X=lon, Y=depth).
    slice_dim='lon': fixed longitude slice, across latitude (X=lat, Y=depth).
    """
    if slice_dim == "lat":
        # volume_3d is [15, 69, 81]
        transect_data = volume_3d[:, slice_idx, :]  # [15, 81]
        x_axis = lons
        x_label = f"Longitude (°E) at {lats[slice_idx]:.2f}°N"
    else:
        transect_data = volume_3d[:, :, slice_idx]  # [15, 69]
        x_axis = lats
        x_label = f"Latitude (°N) at {lons[slice_idx]:.2f}°E"

    fig = go.Figure(
        data=go.Contour(
            z=transect_data,
            x=x_axis,
            y=depths,
            colorscale="Thermal",
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
            colorbar=dict(
                title=dict(text="°C", font=dict(size=12, color="#e0e6ed")),
                tickfont=dict(color="#e0e6ed"),
                thickness=15,
            ),
            hovertemplate="<b>Coord</b>: %{x:.2f}°<br><b>Depth</b>: %{y}m<br><b>Temp</b>: %{z:.2f}°C<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e0e6ed")),
        xaxis=dict(
            title=x_label,
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="Depth (meters)",
            autorange="reversed",  # Invert depth so 0m is at the top
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        margin=dict(l=50, r=40, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 25, 47, 0.7)",
        height=400,
    )
    return fig


def create_argo_location_map(
    df: pd.DataFrame, selected_profile_id: Optional[str] = None
) -> go.Figure:
    """Displays 253 CORA Argo float profiles geographically on the Bay of Bengal map."""
    profiles = (
        df.groupby("profile_id")
        .first()
        .reset_index()[["profile_id", "date", "lat", "lon"]]
    )
    profiles["date_str"] = profiles["date"].dt.strftime("%Y-%m-%d")

    fig = px.scatter(
        profiles,
        x="lon",
        y="lat",
        hover_name="profile_id",
        hover_data={"date_str": True, "lat": ":.2f", "lon": ":.2f"},
        color_discrete_sequence=["#00f2fe"],
        labels={"lon": "Longitude (°E)", "lat": "Latitude (°N)"},
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(width=1, color="#ffffff")),
        hovertemplate="<b>%{hovertext}</b><br>Date: %{customdata[0]}<br>Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<extra></extra>",
    )

    # Highlight selected profile if provided
    if selected_profile_id:
        sel_row = profiles[profiles["profile_id"] == selected_profile_id]
        if not sel_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel_row["lon"],
                    y=sel_row["lat"],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color="#ff007f",
                        symbol="star",
                        line=dict(width=2, color="#ffffff"),
                    ),
                    name=f"Selected: {selected_profile_id}",
                    text=[selected_profile_id],
                    textposition="top right",
                    textfont=dict(color="#ff007f", size=12),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"CORA Delayed-Mode Argo Floats in Bay of Bengal (N={len(profiles)})",
            font=dict(size=14, color="#e0e6ed"),
        ),
        xaxis=dict(
            title="Longitude (°E)",
            range=[79.5, 100.5],
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="Latitude (°N)",
            range=[4.5, 22.5],
            scaleanchor="x",
            scaleratio=1,
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 25, 47, 0.7)",
        height=450,
        showlegend=False,
    )
    return fig


def create_argo_profile_comparison(prof_df: pd.DataFrame, profile_id: str) -> go.Figure:
    """
    Renders vertical temperature profile (0-1000m) for a selected Argo float,
    comparing in-situ observation with CNN, CBAM, OceanEmbed, and GLORYS Reanalysis.
    """
    pdf = prof_df.sort_values("depth_m")

    fig = go.Figure()

    # Argo In-Situ Observation
    fig.add_trace(
        go.Scatter(
            x=pdf["obs_temp"],
            y=pdf["depth_m"],
            mode="lines+markers",
            name="Argo Float (Ground Truth)",
            line=dict(color="#ffffff", width=3.5),
            marker=dict(size=7, color="#ffffff", symbol="circle"),
        )
    )

    # CNN Baseline
    if "cnn_temp" in pdf.columns:
        fig.add_trace(
            go.Scatter(
                x=pdf["cnn_temp"],
                y=pdf["depth_m"],
                mode="lines+markers",
                name="CNN Baseline",
                line=dict(color="#00f2fe", width=2.5),
                marker=dict(size=6, color="#00f2fe", symbol="diamond"),
            )
        )

    # CBAM-CNN
    if "cbam_temp" in pdf.columns:
        fig.add_trace(
            go.Scatter(
                x=pdf["cbam_temp"],
                y=pdf["depth_m"],
                mode="lines+markers",
                name="OceanEmbed (FNO) (Attention)",
                line=dict(color="#ff9900", width=2.5, dash="dash"),
                marker=dict(size=6, color="#ff9900", symbol="triangle-up"),
            )
        )

    # OceanEmbed (FNO)
    if "oe_temp" in pdf.columns:
        fig.add_trace(
            go.Scatter(
                x=pdf["oe_temp"],
                y=pdf["depth_m"],
                mode="lines+markers",
                name="CBAM-CNN",
                line=dict(color="#b388ff", width=2, dash="dot"),
                marker=dict(size=5, color="#b388ff", symbol="cross"),
            )
        )

    # GLORYS12 Reanalysis Reference
    if "glorys_temp" in pdf.columns:
        fig.add_trace(
            go.Scatter(
                x=pdf["glorys_temp"],
                y=pdf["depth_m"],
                mode="lines",
                name="GLORYS12 Reanalysis",
                line=dict(color="#00e676", width=2, dash="longdash"),
            )
        )

    # Metadata
    date_str = pdf["date"].iloc[0].strftime("%Y-%m-%d")
    lat_val = pdf["lat"].iloc[0]
    lon_val = pdf["lon"].iloc[0]

    fig.update_layout(
        title=dict(
            text=f"Vertical Temperature Stratification — Float {profile_id} ({date_str} at {lat_val:.2f}°N, {lon_val:.2f}°E)",
            font=dict(size=14, color="#e0e6ed"),
        ),
        xaxis=dict(
            title="Temperature (°C)",
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Depth (meters)",
            autorange="reversed",  # 0m at surface
            color="#a0aec0",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        margin=dict(l=50, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 25, 47, 0.7)",
        legend=dict(
            font=dict(color="#e0e6ed", size=11),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            x=0.03,
            y=0.05,
        ),
        height=520,
    )
    return fig
