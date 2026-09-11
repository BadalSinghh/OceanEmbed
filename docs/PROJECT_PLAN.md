# OceanEmbed Project Plan & Technical Implementation Roadmap

## Project Overview

**Problem Statement**: SIH26066 - OceanEmbed: Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.
**Goal**: Build a scientifically rigorous, reproducible 2-year proof-of-concept for the Bay of Bengal region (2022–2023) at 0.25° spatial resolution and 15 standard ocean depth levels.

---

## Technical Stages & Implementation Roadmap

```
Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5 ──► Stage 6 ──► Stage 7
  Setup      Data Fetch  Preprocess   Model Arch   Training    Argo Eval   Dashboard
```

---

### Stage 1: Environment Setup & Data Catalogue Verification (COMPLETE)
* [x] Python 3.12 virtual environment setup (`py -3.12 -m venv .venv`).
* [x] Dependency management (`requirements.txt`) including `torch`, `xarray`, `netcdf4`, `copernicusmarine`.
* [x] Formulate project repository structure.
* [x] Compile verified dataset catalogue in `docs/DATA_SOURCES.md`.

---

### Stage 2: Data Acquisition & Preprocessing Pipeline
* [ ] Implement Copernicus API data fetch script (`src/data/copernicus_downloader.py`).
* [ ] Fetch 7 daily observation variables (SST, SSS, SLA, Ugos, Vgos, Wind U, Wind V) for 2022-01-01 to 2023-12-31 over 5°N–22°N, 80°E–100°E.
* [ ] Fetch GLORYS12V1 target temperature profiles (`thetao`) at 15 requested depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m.
* [ ] Fetch Argo in-situ profile dataset for independent validation.
* [ ] Implement regridding, missing data masking, land/sea masking, and z-score normalization pipelines (`src/data/preprocessing.py`).

---

### Stage 3: Spatial Grid Standardizer & PyTorch Data Loaders
* [ ] Build uniform 0.25° grid representation (`src/utils/spatial_grid.py`).
* [ ] Implement PyTorch Dataset & DataLoader class (`OceanEmbedDataset`) supporting rolling window sequences and spatial patches.
* [ ] Set up train/validation/test splits:
  * **Train**: 2022-01-01 to 2023-06-30 (~1.5 years)
  * **Validation**: 2023-07-01 to 2023-09-30 (3 months)
  * **Test**: 2023-10-01 to 2023-12-31 (3 months)

---

### Stage 4: Model Architecture Engineering

The project implements and compares **three distinct models**:

1. **Model 1: Climatology / Depth Baseline**
   * Computes spatial-temporal mean temperature profiles per grid cell across depth.

2. **Model 2: CNN-Only Baseline**
   * 2D ResNet / U-Net convolutional encoder-decoder predicting 15 vertical temperature slices directly from 7 input surface channels.

3. **Model 3: OceanEmbed (CNN + FNO-2D)**
   * **7 Surface Channels** $\rightarrow$ **CNN Encoder** $\rightarrow$ **Compact Spatial Ocean Embedding** $\rightarrow$ **FNO-2D Spectral Conv Blocks** $\rightarrow$ **Depth Decoder** $\rightarrow$ **15 Temperature Maps**.
   * Integrates 2D Fourier Neural Operators in the latent spectral domain to capture multi-scale hydrodynamic non-local teleconnections.

---

### Stage 5: Training & Quantitative Evaluation Protocol
* [ ] Loss functions: Depth-weighted Mean Squared Error (MSE) + Gradient Cosine Similarity loss to preserve thermocline gradients.
* [ ] Optimizer: AdamW with Cosine Annealing learning rate schedule.
* [ ] Validation metrics computed per depth level and overall:
  * Root Mean Square Error (RMSE)
  * Mean Absolute Error (MAE)
  * Bias (Mean Error)
  * Pearson Correlation Coefficient ($r$)

---

### Stage 6: Held-Out Argo In-Situ Benchmark Validation
* [ ] Extract held-out Argo float profiles in the Bay of Bengal for 2022-2023.
* [ ] Interpolate float temperature observations to the 15 standard depths.
* [ ] Match spatial coordinates and timestamps to model predictions.
* [ ] Generate overall and depth-resolved comparative error matrices (Climatology vs. CNN Baseline vs. OceanEmbed).

---

### Stage 7: Interactive Streamlit Visualizer & Dashboard
* [ ] Interactive spatial slice map viewer (depth vs. surface variables).
* [ ] Thermocline depth profile plot comparison against Argo observations.
* [ ] Performance dashboard showcasing error heatmaps and metric comparisons.
