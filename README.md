# OceanEmbed — Subsurface Ocean Digital Twin

Deep learning framework for reconstructing the 3-D vertical ocean temperature
column (0 to 1000 m, 15 standard depth levels) from daily multi-satellite
surface observables over the Bay of Bengal (5–22 N, 80–100 E).

---

## Quick Start

```bash
# 1. Install dependencies (CPU only, no GPU required)
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Directory Structure

```
oceanEmbed-Prototype/
├── app.py                          # Streamlit dashboard entry point
├── requirements.txt                # Runtime dependencies
├── src/
│   ├── ui_helpers.py               # Cached loaders, inference, Plotly helpers
│   └── models/
│       ├── cnn_baseline.py         # CNN Baseline (191,631 params)
│       ├── cbam.py                 # CBAM-CNN with channel+spatial attention
│       ├── ocean_embed.py          # OceanEmbed CNN + FNO2D (8.67M params)
│       ├── cnn_encoder.py
│       ├── depth_decoder.py
│       ├── fno2d.py
│       └── losses.py
├── data/
│   └── processed/
│       ├── coords.nc               # Lat/Lon/Depth grid coordinates
│       ├── land_mask.npy           # Boolean land mask [69, 81]
│       └── test/
│           ├── X_test.npz          # Normalized satellite inputs [109, 7, 69, 81]
│           ├── Y_test.npz          # GLORYS12 subsurface targets [109, 15, 69, 81]
│           └── dates_test.npy      # Test period dates (Sep–Dec 2023)
└── results/
    ├── models/
    │   ├── cnn_baseline_best.pt    # CNN Baseline checkpoint (~2.3 MB)
    │   ├── cbam_cnn_best.pt        # CBAM-CNN checkpoint (~2.4 MB)
    │   └── oceanembed_best.pt      # OceanEmbed checkpoint (~205 MB)
    ├── normalization_stats.json    # Channel mean/std computed on training set
    ├── argo_validation/
    │   └── argo_matched_observations.csv   # 3,509 real CORA Argo observations
    └── metrics/
        ├── evaluation_summary.json
        ├── per_depth_metrics.csv
        ├── cnn_baseline_history.json
        ├── cbam_cnn_history.json
        └── oceanembed_history.json
```

---

## Dashboard Sections

| Section | Description |
| :--- | :--- |
| 3D Subsurface Reconstruction | Date and depth selector, 3-panel heatmaps (GT / Pred / Error), vertical transect |
| Argo Float Validation | 253 CORA Argo float locations, per-float depth profile comparison |
| Models and Benchmark | Leaderboard table, per-depth RMSE curves, training convergence plots |
| Live Point Profiler | Coordinate picker, instantaneous CPU inference, vertical profile output |
| Scientific Documentation | Full methodology, data sources, model architecture descriptions |

---

## Model Performance (Independent Argo Validation)

| Architecture | Argo RMSE | Argo MAE | Pearson R |
| :--- | :---: | :---: | :---: |
| GLORYS12 Reanalysis (reference) | 0.8061 C | 0.4565 C | 0.9958 |
| **CNN Baseline** | **1.1611 C** | **0.6834 C** | **0.9912** |
| CBAM-CNN | 1.3777 C | 0.7905 C | 0.9874 |
| OceanEmbed (FNO2D) | 1.7113 C | 1.1496 C | 0.9811 |
| Climatology baseline | 2.0253 C | 1.2297 C | 0.9766 |

Validation dataset: 253 unique QC-passed CORA delayed-mode Argo profiles,
3,509 matched depth observations, test period 2023-09-14 to 2023-12-31.
No Argo data was used during model training.
