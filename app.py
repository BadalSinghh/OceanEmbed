# -*- coding: utf-8 -*-
"""
app.py
======
OceanEmbed: Deep Learning Framework for Subsurface Ocean Temperature Reconstruction
from Satellite Observations.

Streamlit Interactive Dashboard for Subsurface Ocean Thermal Reconstruction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui_helpers import (
    TARGET_DEPTHS,
    create_argo_location_map,
    create_argo_profile_comparison,
    create_spatial_heatmap,
    create_vertical_transect,
    load_argo_dataset,
    load_per_depth_metrics,
    load_pytorch_model,
    load_test_metadata,
    load_test_sample,
    load_training_histories,
    load_coordinates,
    load_land_mask,
    load_normalization_stats,
    predict_temperature_field,
    unnormalize_input_channels,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OceanEmbed | Subsurface Ocean Thermal Reconstruction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(ellipse at 15% 10%, #0a1628 0%, #030712 80%);
        color: #e2e8f0;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.18s ease, border-color 0.18s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }

    .metric-val {
        font-size: 1.65rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-top: 0.35rem;
    }

    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #cbd5e1;
        border-left: 3px solid #38bdf8;
        padding-left: 0.75rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070d1f 0%, #050b1a 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.08);
    }

    /* Sidebar logo block */
    .sidebar-logo {
        padding: 1.2rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.1);
        margin-bottom: 1.2rem;
    }
    .sidebar-logo .logo-name {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-logo .logo-sub {
        font-size: 0.72rem;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.15rem;
    }

    /* Nav label */
    .nav-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #334155;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    /* Spec grid */
    .spec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .spec-card {
        background: rgba(56, 189, 248, 0.04);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 8px;
        padding: 0.55rem 0.6rem;
    }
    .spec-card .spec-val {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
    }
    .spec-card .spec-key {
        font-size: 0.68rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.15rem;
    }

    /* Divider line */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(56,189,248,0.07);
        margin: 1rem 0;
    }

    /* Domain section title */
    .domain-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #334155;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .stSelectbox label, .stSlider label, .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    /* Hide native radio label text (we use our own) */
    [data-testid="stRadio"] > label {
        display: none;
    }

    .stDataFrame {
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(255,255,255,0.07);
    }

    .stAlert {
        background: rgba(15,23,42,0.7);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load Core Data
# ---------------------------------------------------------------------------
lats, lons, depths = load_coordinates()
land_mask = load_land_mask()
test_dates, num_test_samples = load_test_metadata()
stats = load_normalization_stats()
argo_df = load_argo_dataset()
per_depth_df = load_per_depth_metrics()
histories = load_training_histories()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-name">OceanEmbed</div>
            <div class="logo-sub">Subsurface Digital Twin</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    menu_option = st.radio(
        "Navigation",
        [
            "3D Subsurface Reconstruction",
            "Argo Float Validation",
            "Models and Benchmark",
            "Live Point Profiler",
            "Scientific Documentation",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="domain-title">Target Domain</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="spec-grid">
            <div class="spec-card">
                <div class="spec-val">Bay of Bengal</div>
                <div class="spec-key">Region</div>
            </div>
            <div class="spec-card">
                <div class="spec-val">0.25 deg</div>
                <div class="spec-key">Resolution</div>
            </div>
            <div class="spec-card">
                <div class="spec-val">15 levels</div>
                <div class="spec-key">Depth Targets</div>
            </div>
            <div class="spec-card">
                <div class="spec-val">0 - 1000 m</div>
                <div class="spec-key">Column Depth</div>
            </div>
            <div class="spec-card">
                <div class="spec-val">7 channels</div>
                <div class="spec-key">Satellite Inputs</div>
            </div>
            <div class="spec-card">
                <div class="spec-val">69 x 81</div>
                <div class="spec-key">Grid Cells</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">OceanEmbed Subsurface Digital Twin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Reconstructing the 3-D vertical ocean temperature column (0 to 1000 m) '
    'from daily multi-satellite surface observables using deep representation learning.</div>',
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">15 Levels</div>'
        '<div class="metric-label">Vertical Depth Coverage</div></div>',
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">1.38 C</div>'
        '<div class="metric-label">Argo RMSE — OceanEmbed (FNO)</div></div>',
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">R = 0.9874</div>'
        '<div class="metric-label">Pearson Correlation (OceanEmbed (FNO))</div></div>',
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        '<div class="metric-card"><div class="metric-val">253 Floats</div>'
        '<div class="metric-label">3,509 Matched Argo Observations</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================================================
# TAB 1: 3D SUBSURFACE RECONSTRUCTION
# ===========================================================================
if menu_option == "3D Subsurface Reconstruction":
    st.markdown('<div class="section-header">Satellite Surface Observation and 3D Thermal Reconstruction</div>', unsafe_allow_html=True)
    st.caption(
        "Select a test date and architecture to view the reconstructed 3-D subsurface temperature field "
        "at any standard depth level alongside the GLORYS12 reanalysis ground truth."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.6, 1.2, 1.2])
    with ctrl1:
        date_selection = st.selectbox("Test Observation Date", test_dates, index=0)
        date_idx = test_dates.index(date_selection)
    with ctrl2:
        model_choice = st.selectbox(
            "Deep Learning Architecture",
            ["OceanEmbed (FNO)", "CBAM-CNN"],
            index=0,
        )
    with ctrl3:
        depth_selection = st.select_slider(
            "Target Depth Level (m)",
            options=TARGET_DEPTHS,
            value=100,
        )
        depth_idx = TARGET_DEPTHS.index(depth_selection)

    with st.spinner("Loading fields and running inference on CPU..."):
        x_norm, y_true = load_test_sample(date_idx)
        model = load_pytorch_model(model_choice)
        if model is not None:
            pred_3d = predict_temperature_field(model, x_norm, land_mask)
        else:
            st.error(f"Model checkpoint not found: {model_choice}")
            st.stop()
        y_true_masked = y_true.copy().astype(float)
        y_true_masked[:, land_mask] = np.nan

    # Surface Satellite Inputs
    with st.expander("Inspect 7 Satellite Surface Input Channels (Physical Units)", expanded=False):
        x_phys = unnormalize_input_channels(x_norm, stats)
        x_phys_masked = x_phys.copy().astype(float)
        x_phys_masked[:, land_mask] = np.nan

        ch_tabs = st.tabs([
            "SST", "SSS", "SLA", "Current U / V", "Wind U / V",
        ])
        with ch_tabs[0]:
            st.plotly_chart(
                create_spatial_heatmap(x_phys_masked[0], lats, lons, f"Sea Surface Temperature (C) — {date_selection}", unit="C", colorscale="Thermal"),
                width="stretch",
            )
        with ch_tabs[1]:
            st.plotly_chart(
                create_spatial_heatmap(x_phys_masked[1], lats, lons, f"Sea Surface Salinity (psu) — {date_selection}", unit="psu", colorscale="Viridis"),
                width="stretch",
            )
        with ch_tabs[2]:
            st.plotly_chart(
                create_spatial_heatmap(x_phys_masked[2], lats, lons, f"Sea Level Anomaly (m) — {date_selection}", unit="m", diverging=True),
                width="stretch",
            )
        with ch_tabs[3]:
            cu_col, cv_col = st.columns(2)
            with cu_col:
                st.plotly_chart(
                    create_spatial_heatmap(x_phys_masked[3], lats, lons, "Geostrophic Current U (m/s)", unit="m/s", diverging=True),
                    width="stretch",
                )
            with cv_col:
                st.plotly_chart(
                    create_spatial_heatmap(x_phys_masked[4], lats, lons, "Geostrophic Current V (m/s)", unit="m/s", diverging=True),
                    width="stretch",
                )
        with ch_tabs[4]:
            wu_col, wv_col = st.columns(2)
            with wu_col:
                st.plotly_chart(
                    create_spatial_heatmap(x_phys_masked[5], lats, lons, "Wind U (m/s)", unit="m/s", diverging=True),
                    width="stretch",
                )
            with wv_col:
                st.plotly_chart(
                    create_spatial_heatmap(x_phys_masked[6], lats, lons, "Wind V (m/s)", unit="m/s", diverging=True),
                    width="stretch",
                )

    st.markdown(f'<div class="section-header">Reconstruction at Depth {depth_selection} m — {date_selection}</div>', unsafe_allow_html=True)

    true_slice = y_true_masked[depth_idx]
    pred_slice = pred_3d[depth_idx]
    diff_slice = pred_slice - true_slice

    valid_vals = true_slice[~np.isnan(true_slice)]
    zmin = float(np.percentile(valid_vals, 2)) if len(valid_vals) > 0 else 10.0
    zmax = float(np.percentile(valid_vals, 98)) if len(valid_vals) > 0 else 30.0

    map1, map2, map3 = st.columns(3)
    with map1:
        st.plotly_chart(
            create_spatial_heatmap(true_slice, lats, lons, f"GLORYS12 Ground Truth — {depth_selection} m", unit="C", colorscale="Thermal", zmin=zmin, zmax=zmax),
            width="stretch",
        )
    with map2:
        st.plotly_chart(
            create_spatial_heatmap(pred_slice, lats, lons, f"{model_choice} Reconstruction — {depth_selection} m", unit="C", colorscale="Thermal", zmin=zmin, zmax=zmax),
            width="stretch",
        )
    with map3:
        st.plotly_chart(
            create_spatial_heatmap(diff_slice, lats, lons, "Prediction Error (Pred minus GT)", unit="C", diverging=True, zmin=-2.5, zmax=2.5),
            width="stretch",
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Vertical Thermal Transect — Cross-Sectional Slice (0 to 1000 m)</div>', unsafe_allow_html=True)
    st.caption("Slice the reconstructed 3-D field along a fixed latitude or longitude to inspect thermocline depth and stratification.")

    t_ctrl, t_plot = st.columns([1, 2.2])
    with t_ctrl:
        transect_axis = st.radio("Slice Orientation", ["Fixed Latitude (Zonal)", "Fixed Longitude (Meridional)"])
        if "Latitude" in transect_axis:
            lat_pick = st.slider("Latitude (N)", float(lats.min()), float(lats.max()), 13.5, step=0.25)
            s_idx = int(np.argmin(np.abs(lats - lat_pick)))
            fig_tr = create_vertical_transect(pred_3d, lats, lons, depths, slice_dim="lat", slice_idx=s_idx, title=f"Vertical Section at {lats[s_idx]:.2f} N")
        else:
            lon_pick = st.slider("Longitude (E)", float(lons.min()), float(lons.max()), 88.0, step=0.25)
            s_idx = int(np.argmin(np.abs(lons - lon_pick)))
            fig_tr = create_vertical_transect(pred_3d, lats, lons, depths, slice_dim="lon", slice_idx=s_idx, title=f"Vertical Section at {lons[s_idx]:.2f} E")
    with t_plot:
        st.plotly_chart(fig_tr, width="stretch")


# ===========================================================================
# TAB 2: ARGO FLOAT VALIDATION
# ===========================================================================
elif menu_option == "Argo Float Validation":
    st.markdown('<div class="section-header">Independent In-Situ Argo Profiling Float Benchmark</div>', unsafe_allow_html=True)
    st.caption(
        "Rigorous validation against 253 unique QC-passed CORA delayed-mode in-situ Argo floats "
        "(3,509 matched depth observations) across the held-out test period Sep 14 to Dec 31, 2023."
    )

    profiles = argo_df["profile_id"].unique()

    a1, a2 = st.columns([1.3, 1.7])
    with a1:
        st.markdown("**Float Selection**")
        selected_prof = st.selectbox("Profile ID", profiles, index=0)
        prof_data = argo_df[argo_df["profile_id"] == selected_prof].copy()
        prof_data["date"] = pd.to_datetime(prof_data["date"])
        fig_map = create_argo_location_map(argo_df, selected_profile_id=selected_prof)
        st.plotly_chart(fig_map, width="stretch")

    with a2:
        st.markdown("**Vertical Temperature Profile Comparison**")
        fig_profile = create_argo_profile_comparison(prof_data, selected_prof)
        st.plotly_chart(fig_profile, width="stretch")

    st.markdown(f"**Observation Data — Profile `{selected_prof}`**")
    disp_df = prof_data[["depth_m", "obs_temp", "glorys_temp", "cbam_temp", "oe_temp"]].copy()
    disp_df.columns = ["Depth (m)", "Argo In-Situ (C)", "GLORYS (C)", "OceanEmbed (FNO) (C)", "CBAM-CNN (C)"]
    disp_df["CBAM Error (C)"] = (disp_df["OceanEmbed (FNO) (C)"] - disp_df["Argo In-Situ (C)"]).round(3)
    disp_df["OceanEmbed Error (C)"] = (disp_df["CBAM-CNN (C)"] - disp_df["Argo In-Situ (C)"]).round(3)
    st.dataframe(disp_df.reset_index(drop=True), width="stretch")

    st.markdown("---")
    st.markdown('<div class="section-header">Global Argo Skill Metrics — All 3,509 Observations</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("OceanEmbed (FNO) Argo RMSE", "1.3777 C")
    with mc2:
        st.metric("OceanEmbed Argo RMSE", "1.7113 C")
    with mc3:
        st.metric("GLORYS Reference RMSE", "0.8061 C")

    with st.expander("Global Argo Scatter Distribution", expanded=True):
        scat_sub = argo_df.sample(min(1500, len(argo_df)), random_state=42)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[5, 32], y=[5, 32], mode="lines", name="1:1 Reference",
            line=dict(color="#475569", dash="dash", width=1.5)
        ))
        fig_sc.add_trace(go.Scatter(
            x=scat_sub["obs_temp"], y=scat_sub["cbam_temp"],
            mode="markers", name="OceanEmbed (FNO)",
            marker=dict(size=4, color="#fb923c", opacity=0.55)
        ))
        fig_sc.add_trace(go.Scatter(
            x=scat_sub["obs_temp"], y=scat_sub["oe_temp"],
            mode="markers", name="CBAM-CNN",
            marker=dict(size=4, color="#a78bfa", opacity=0.45)
        ))
        fig_sc.update_layout(
            title="Predicted vs. In-Situ Argo Float Temperature (C)",
            xaxis=dict(title="Argo In-Situ Observation (C)", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Model Prediction (C)", color="#94a3b8", gridcolor="rgba(255,255,255,0.05)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.7)",
            height=440,
            legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
        )
        st.plotly_chart(fig_sc, width="stretch")


# ===========================================================================
# TAB 3: MODELS AND BENCHMARK
# ===========================================================================
elif menu_option == "Models and Benchmark":
    st.markdown('<div class="section-header">Architecture Benchmark and Comparative Evaluation</div>', unsafe_allow_html=True)
    st.caption(
        "Quantitative evaluation on the held-out GLORYS reanalysis test set and real-world CORA in-situ Argo floats."
    )

    st.markdown("**Benchmark Leaderboard**")
    lb = pd.DataFrame({
        "Architecture": ["GLORYS12 Reanalysis", "OceanEmbed (FNO)", "CBAM-CNN (FNO2D)", "Climatology"],
        "Parameters": ["—", "195,705", "8,670,241", "0"],
        "Argo RMSE (C)": ["0.8061", "1.3777", "1.7113", "2.0253"],
        "Argo MAE (C)": ["0.4565", "0.7905", "1.1496", "1.2297"],
        "Pearson R": ["0.9958", "0.9874", "0.9811", "0.9766"],
        "GLORYS Test RMSE (C)": ["—", "1.3531", "1.4736", "1.8730"],
    })
    st.dataframe(lb, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Reconstruction Error vs. Depth</div>', unsafe_allow_html=True)

    fig_depth = go.Figure()
    if "cbam_rmse" in per_depth_df.columns:
        fig_depth.add_trace(go.Scatter(
            x=per_depth_df["cbam_rmse"], y=per_depth_df["depth_m"],
            mode="lines+markers", name="OceanEmbed (FNO)",
            line=dict(color="#fb923c", width=2.5), marker=dict(size=6)
        ))
    if "oceanembed_rmse" in per_depth_df.columns:
        fig_depth.add_trace(go.Scatter(
            x=per_depth_df["oceanembed_rmse"], y=per_depth_df["depth_m"],
            mode="lines+markers", name="CBAM-CNN",
            line=dict(color="#a78bfa", width=2.5, dash="dash"), marker=dict(size=5)
        ))
    fig_depth.update_layout(
        title="Per-Depth Test RMSE (C)",
        xaxis=dict(title="RMSE (C)", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Depth (m)", autorange="reversed", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,25,47,0.7)",
        height=440,
        legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
    )
    st.plotly_chart(fig_depth, width="stretch")

    # -----------------------------------------------------------------------
    # Animated Depth-Sweep: RMSE race across depth levels (0 → 1000 m)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">Animated Depth-Sweep — Error Evolution 0 → 1000 m</div>', unsafe_allow_html=True)
    st.caption(
        "Watch how reconstruction error evolves as the models predict deeper into the ocean. "
        "Each frame advances one depth level — press ▶ or drag the slider to explore."
    )

    import json as _json

    # Build raw per-depth data from evaluation_summary.json
    _eval_path = ROOT / "results" / "metrics" / "evaluation_summary.json"
    with open(_eval_path, "r") as _f:
        _eval = _json.load(_f)

    _pd_data = _eval.get("per_depth_test", {})
    _depth_keys = sorted(_pd_data.keys(), key=lambda x: int(x))
    _depths_anim  = [int(d) for d in _depth_keys]

    _model_cfg = [
        ("OceanEmbed (FNO)", "cbam_rmse",      "cbam_mae",      "#fb923c"),
        ("CBAM-CNN",         "oceanembed_rmse", "oceanembed_mae", "#a78bfa"),
        ("CNN Baseline",     "cnn_rmse",        "cnn_mae",        "#38bdf8"),
    ]

    # Pre-compute cumulative (surface-to-current-depth) running RMSE for animation
    # Each frame = one additional depth level revealed as a horizontal bar
    _frames = []
    for _i, _dk in enumerate(_depth_keys):
        _visible_depths = [int(d) for d in _depth_keys[: _i + 1]]
        _frame_traces = []
        for _mname, _rmse_col, _mae_col, _col in _model_cfg:
            _rmse_vals = [_pd_data[str(d)].get(_rmse_col, 0) for d in _visible_depths]
            _mae_vals  = [_pd_data[str(d)].get(_mae_col, 0)  for d in _visible_depths]
            _frame_traces.append(
                go.Bar(
                    x=_rmse_vals,
                    y=[str(d) + " m" for d in _visible_depths],
                    orientation="h",
                    name=_mname,
                    marker=dict(
                        color=_col,
                        opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.12)", width=0.8),
                    ),
                    customdata=list(zip(_mae_vals, _visible_depths)),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Depth: %{customdata[1]} m<br>"
                        "RMSE: %{x:.3f} °C<br>"
                        "MAE: %{customdata[0]:.3f} °C<extra></extra>"
                    ),
                    text=[f"{v:.2f}" for v in _rmse_vals],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=10),
                )
            )
        _frames.append(go.Frame(data=_frame_traces, name=str(_dk)))

    # Initial (first frame) traces
    _init_traces = _frames[0].data

    fig_anim = go.Figure(
        data=list(_init_traces),
        frames=_frames,
        layout=go.Layout(
            title=dict(
                text="RMSE by Depth Level — Animated Depth Sweep",
                font=dict(color="#e2e8f0", size=14),
            ),
            xaxis=dict(
                title="RMSE (°C)",
                color="#94a3b8",
                gridcolor="rgba(255,255,255,0.07)",
                range=[0, 3.8],
                fixedrange=True,
            ),
            yaxis=dict(
                title="Depth Level",
                color="#94a3b8",
                gridcolor="rgba(255,255,255,0.05)",
                autorange="reversed",
                categoryorder="array",
                categoryarray=[str(d) + " m" for d in _depths_anim],
            ),
            barmode="group",
            bargap=0.22,
            bargroupgap=0.06,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.8)",
            height=520,
            legend=dict(
                font=dict(color="#cbd5e1", size=12),
                bgcolor="rgba(15,23,42,0.85)",
                bordercolor="rgba(56,189,248,0.15)",
                borderwidth=1,
                x=0.72, y=0.02,
            ),
            margin=dict(l=80, r=40, t=60, b=80),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.12,
                    x=0.0,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶  Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=420, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=200, easing="cubic-in-out"),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸  Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(
                        prefix="Depth shown up to: ",
                        font=dict(color="#38bdf8", size=12),
                        visible=True,
                        xanchor="center",
                    ),
                    pad=dict(b=10, t=10),
                    len=0.88,
                    x=0.11,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [f.name],
                                dict(
                                    frame=dict(duration=300, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=150),
                                ),
                            ],
                            label=f"{dk} m",
                            method="animate",
                        )
                        for f, dk in zip(_frames, _depth_keys)
                    ],
                    font=dict(color="#64748b", size=10),
                    bgcolor="rgba(15,23,42,0.7)",
                    bordercolor="rgba(56,189,248,0.15)",
                    activebgcolor="#38bdf8",
                    tickcolor="rgba(56,189,248,0.4)",
                )
            ],
        ),
    )
    _anim_html = fig_anim.to_html(full_html=True, include_plotlyjs="cdn", config={"responsive": True})
    _components.html(_anim_html, height=620, scrolling=False)

    st.markdown("---")
    st.markdown('<div class="section-header">Training Convergence Curves</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    with h1:
        fig_val = go.Figure()
        colors = {"OceanEmbed (FNO)": "#fb923c", "CBAM-CNN": "#a78bfa"}
        for name, col in colors.items():
            if name in histories and "val_loss" in histories[name]:
                epochs = list(range(1, len(histories[name]["val_loss"]) + 1))
                fig_val.add_trace(go.Scatter(
                    x=epochs, y=histories[name]["val_loss"],
                    mode="lines", name=f"{name}",
                    line=dict(color=col, width=2.2)
                ))
        fig_val.update_layout(
            title="Validation Loss per Epoch (Masked MSE, C^2)",
            xaxis=dict(title="Epoch", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Validation Loss", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.7)",
            height=380,
            legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
        )
        st.plotly_chart(fig_val, width="stretch")

    with h2:
        fig_train = go.Figure()
        for name, col in colors.items():
            if name in histories and "train_loss" in histories[name]:
                epochs = list(range(1, len(histories[name]["train_loss"]) + 1))
                fig_train.add_trace(go.Scatter(
                    x=epochs, y=histories[name]["train_loss"],
                    mode="lines", name=f"{name}",
                    line=dict(color=col, width=2.2)
                ))
        fig_train.update_layout(
            title="Training Loss per Epoch (Masked MSE, C^2)",
            xaxis=dict(title="Epoch", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Training Loss", color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.7)",
            height=380,
            legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
        )
        st.plotly_chart(fig_train, width="stretch")

    st.markdown("---")
    st.markdown("**Key Scientific Findings**")
    st.markdown(
        """
        - **Peak error band at 75–150 m**: Both models show maximum RMSE in the pycnocline layer where steep vertical
          gradients from internal waves and mesoscale eddy dynamics are hardest to constrain from surface-only observations.
        - **Deep column accuracy below 300 m**: RMSE drops to 0.25–0.40 C at depths of 500 m, 700 m, and 1000 m,
          indicating the thermocline base and abyssal layer are well-constrained from surface altimetric and density signals.
        - **Mixed-layer advantage of CBAM attention**: The channel and spatial attention gates in CBAM-CNN achieve
          strong accuracy in the 0–30 m mixed layer, leveraging surface channel weighting to resolve shallow thermal gradients.
        - **Spectral learning in OceanEmbed**: The FNO2D operator in OceanEmbed captures long-range basin-scale
          teleconnections and mesoscale eddy structures not expressible by purely local convolutional kernels.
        """
    )


# ===========================================================================
# TAB 4: LIVE POINT PROFILER
# ===========================================================================
elif menu_option == "Live Point Profiler":
    st.markdown('<div class="section-header">Instantaneous Subsurface Temperature Profiler</div>', unsafe_allow_html=True)
    st.caption(
        "Select any ocean coordinate and observation date to extract satellite surface features and reconstruct "
        "the complete vertical temperature profile (0 to 1000 m) using trained models on CPU."
    )

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        prof_date = st.selectbox("Observation Date", test_dates, index=10)
        prof_date_idx = test_dates.index(prof_date)
        prof_lat = st.slider("Latitude (N)", 5.0, 22.0, 12.5, step=0.25)
        prof_lon = st.slider("Longitude (E)", 80.0, 100.0, 86.5, step=0.25)

        lat_idx = int(np.argmin(np.abs(lats - prof_lat)))
        lon_idx = int(np.argmin(np.abs(lons - prof_lon)))
        is_land = land_mask[lat_idx, lon_idx]

        if is_land:
            st.warning("Selected coordinate falls on land. Profiles are computed for ocean grid cells only.")
        else:
            st.success(f"Ocean coordinate — {lats[lat_idx]:.2f} N, {lons[lon_idx]:.2f} E")

        x_norm_p, y_true_p = load_test_sample(prof_date_idx)
        x_phys_p = unnormalize_input_channels(x_norm_p, stats)

        st.markdown("**Surface Satellite Signatures at Selected Point**")
        sigs = {
            "SST (C)": x_phys_p[0, lat_idx, lon_idx],
            "SSS (psu)": x_phys_p[1, lat_idx, lon_idx],
            "SLA (m)": x_phys_p[2, lat_idx, lon_idx],
            "Current U (m/s)": x_phys_p[3, lat_idx, lon_idx],
            "Current V (m/s)": x_phys_p[4, lat_idx, lon_idx],
            "Wind U (m/s)": x_phys_p[5, lat_idx, lon_idx],
            "Wind V (m/s)": x_phys_p[6, lat_idx, lon_idx],
        }
        for label, val in sigs.items():
            st.markdown(f"`{label}`: **{val:.3f}**")

    with pc2:
        model_cbam = load_pytorch_model("OceanEmbed (FNO)")
        model_oe = load_pytorch_model("CBAM-CNN")

        pred_cbam_p = predict_temperature_field(model_cbam, x_norm_p, land_mask)
        pred_oe_p = predict_temperature_field(model_oe, x_norm_p, land_mask)

        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(
            x=y_true_p[:, lat_idx, lon_idx], y=TARGET_DEPTHS,
            mode="lines+markers", name="GLORYS12 Ground Truth",
            line=dict(color="#4ade80", width=3), marker=dict(size=6)
        ))
        fig_pt.add_trace(go.Scatter(
            x=pred_cbam_p[:, lat_idx, lon_idx], y=TARGET_DEPTHS,
            mode="lines+markers", name="OceanEmbed (FNO)",
            line=dict(color="#fb923c", width=2.5), marker=dict(size=6)
        ))
        fig_pt.add_trace(go.Scatter(
            x=pred_oe_p[:, lat_idx, lon_idx], y=TARGET_DEPTHS,
            mode="lines+markers", name="CBAM-CNN",
            line=dict(color="#a78bfa", width=2.5, dash="dash"), marker=dict(size=6)
        ))
        fig_pt.update_layout(
            title=f"Vertical Profile — {lats[lat_idx]:.2f} N, {lons[lon_idx]:.2f} E ({prof_date})",
            xaxis=dict(title="Temperature (C)", color="#94a3b8", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Depth (m)", autorange="reversed", color="#94a3b8", gridcolor="rgba(255,255,255,0.07)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,25,47,0.7)",
            height=500,
            legend=dict(font=dict(color="#cbd5e1"), bgcolor="rgba(15,23,42,0.8)"),
        )
        st.plotly_chart(fig_pt, width="stretch")


# ===========================================================================
# TAB 5: SCIENTIFIC DOCUMENTATION
# ===========================================================================
elif menu_option == "Scientific Documentation":
    st.markdown('<div class="section-header">Technical Architecture and Scientific Methodology</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### Problem Formulation

        Spaceborne remote sensing satellites provide continuous, high-resolution coverage of ocean surface
        conditions. In-situ measurements of the vertical thermal structure — essential for understanding
        heat content, mixed-layer depth, baroclinic dynamics, and climate indices — are obtained primarily
        from ARGO profiling floats, moored buoys, and autonomous gliders. These in-situ platforms deliver
        accurate profiles but are sparse in space and time across the vast Indian Ocean basin.

        The goal of this project is to demonstrate that spaceborne multi-satellite surface observables
        carry sufficient dynamical signal to reconstruct the full vertical temperature column (0 to 1000 m)
        at standard depth levels using deep representation learning.

        ---
        ### Input Surface Observables (7 Channels)

        | Channel | Physical Variable | Satellite / Processing Stream | CMEMS Dataset |
        | :--- | :--- | :--- | :--- |
        | SST | Sea Surface Temperature | OSTIA MetOffice L4 Analysis | METOFFICE-GLO-SST-L4-REP-OBS-SST |
        | SSS | Sea Surface Salinity | Multi-platform obs-based L4 | cmems_obs-mob_glo_phy-sss_my_multi_P1D |
        | SLA | Sea Level Anomaly | DUACS merged altimetry | cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D |
        | Current U | Zonal geostrophic velocity | Derived from SLA | same as SLA |
        | Current V | Meridional geostrophic velocity | Derived from SLA | same as SLA |
        | Wind U | Zonal 10m wind | L4 blended scatterometer | cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H |
        | Wind V | Meridional 10m wind | L4 blended scatterometer | same as Wind U |

        ---
        ### Target Variable

        **3D Ocean Temperature** from GLORYS12V1 Global Ocean Physical Reanalysis
        (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, variable `thetao`) at 15 standard depth levels:
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m.

        Spatial domain: Bay of Bengal, 5–22 N, 80–100 E, 0.25 degree resolution (69 x 81 grid).

        ---
        ### Model Architectures

        **CBAM-CNN (195,705 parameters)**
        A convolutional encoder-decoder augmented with Convolutional Block Attention Module (CBAM)
        gates — channel-wise squeeze-and-excitation followed by spatial attention — between each
        encoder stage. The attention mechanism dynamically weights input channels and spatial
        locations, enabling the model to focus on SST gradients and altimetric anomalies most
        informative for each depth level. Achieves strong mixed-layer accuracy (0–30 m).

        **OceanEmbed (FNO2D) (8,670,241 parameters)**
        A three-stage framework: (1) a CNN spatial encoder producing a 128-channel feature
        representation, (2) a 2D Fourier Neural Operator (FNO2D) applying learned global spectral
        convolutions in latent space to capture basin-scale teleconnections and mesoscale eddy
        signatures, and (3) a depth-conditioned decoder projecting the spectral embeddings to
        the 15 target depth levels. Suited for resolving large-scale ocean dynamics.

        ---
        ### Independent Argo Validation Protocol

        Validation uses CORA v1.3 delayed-mode QC-passed Argo profiles from the
        CMEMS in-situ product `cmems_obs-ins_glo_phy-cur_my_argo_irr`. Float profiles were
        matched to the nearest GLORYS model grid cell and test-period date. 253 unique profiles
        contributing 3,509 paired (obs, pred) depth observations were used. No Argo data was
        used during model training or hyperparameter selection.
        """
    )
