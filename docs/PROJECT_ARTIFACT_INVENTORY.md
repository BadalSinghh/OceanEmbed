# OceanEmbed Prototype (SIH26066) — Complete Project Artifact Inventory

This document provides a comprehensive inventory of all source code files, configurations, datasets, trained checkpoints, evaluation metrics, figures, latent embeddings, and technical documentation produced for the OceanEmbed Prototype.

---

## 1. Trained Model Checkpoints (`results/models/`)

| File Name | File Size | Description & Parameters | Status |
| :--- | :---: | :--- | :--- |
| [`cnn_baseline_best.pt`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/models/cnn_baseline_best.pt) | 2.33 MB | PyTorch checkpoint for Model A (CNN Baseline, 191,631 parameters). Saved at Epoch 49 with best validation loss 0.575807 $^{\circ}\text{C}^2$. | **VERIFIED & FROZEN** |
| [`cbam_cnn_best.pt`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/models/cbam_cnn_best.pt) | 2.40 MB | PyTorch checkpoint for Model B (CBAM-CNN, 195,705 parameters). Saved with best validation loss 0.582082 $^{\circ}\text{C}^2$. | **VERIFIED & FROZEN** |
| [`oceanembed_best.pt`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/models/oceanembed_best.pt) | 204.77 MB | PyTorch checkpoint for Model C (OceanEmbed Framework, 8,670,241 parameters). Saved at Epoch 34 with best validation loss 0.622848 $^{\circ}\text{C}^2$. | **VERIFIED & FROZEN** |

---

## 2. Evaluation Metrics & Test Predictions (`results/metrics/`)

| File Name | File Size | Description | Status |
| :--- | :---: | :--- | :--- |
| [`evaluation_summary.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/evaluation_summary.json) | 4.80 KB | Full test set evaluation metrics (RMSE, MAE, $R^2$) for CNN Baseline, CBAM-CNN, and OceanEmbed overall and per-depth. | **GENERATED** |
| [`per_depth_metrics.csv`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/per_depth_metrics.csv) | 1.25 KB | Tabular CSV recording per-depth RMSE and MAE across all 15 SIH target depths. | **GENERATED** |
| [`cnn_baseline_history.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/cnn_baseline_history.json) | 2.20 KB | Per-epoch training and validation loss history and timing for Model A. | **GENERATED** |
| [`cbam_cnn_history.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/cbam_cnn_history.json) | 2.30 KB | Per-epoch training and validation loss history and timing for Model B. | **GENERATED** |
| [`oceanembed_history.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/oceanembed_history.json) | 2.08 KB | Per-epoch training and validation loss history and timing for Model C. | **GENERATED** |

---

## 3. Extracted Spatial Ocean Embeddings (`results/embeddings/`)

| File Name | File Size | Shape & Contents | Status |
| :--- | :---: | :--- | :--- |
| [`ocean_embeddings_train.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_train.npz) | 82.22 MB | Extracted spatial ocean embeddings for 511 training days (`shape=[511, 128, 17, 20]`). | **EXTRACTED** |
| [`ocean_embeddings_val.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_val.npz) | 17.71 MB | Extracted spatial ocean embeddings for 110 validation days (`shape=[110, 128, 17, 20]`). | **EXTRACTED** |
| [`ocean_embeddings_test.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_test.npz) | 17.54 MB | Extracted spatial ocean embeddings for 109 test days (`shape=[109, 128, 17, 20]`). | **EXTRACTED** |

---

## 4. Visualization Figures (12 Figures in `results/figures/`)

| File Name | Figure Description | Evaluation Arena |
| :--- | :--- | :--- |
| [`argo_scatter.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_scatter.png) | 4-panel Predicted vs Observed scatter plot for GLORYS, CNN, CBAM, and OceanEmbed ($N=3,509$). | Independent Argo |
| [`argo_profile_comparison.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_profile_comparison.png) | Vertical profile comparisons across 4 representative float locations ($0\text{--}1000\text{ m}$). | Independent Argo |
| [`argo_depth_rmse.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_depth_rmse.png) | Per-depth RMSE profile comparing all models against in-situ Argo observations. | Independent Argo |
| [`argo_depth_bias.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_depth_bias.png) | Per-depth Mean Bias profile showing positive offset across models. | Independent Argo |
| [`argo_error_distribution.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_error_distribution.png) | Residual prediction error distributions for all three models. | Independent Argo |
| [`argo_validation_locations.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/argo_validation_locations.png) | Geographical map of the 253 matched float coordinates in the Bay of Bengal. | Independent Argo |
| [`reconstruction_maps.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/reconstruction_maps.png) | 2D Spatial temperature reconstruction maps (Target vs OceanEmbed vs Error) at 0m, 50m, 100m, 500m. | GLORYS Test Set |
| [`cbam_reconstruction_maps.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/cbam_reconstruction_maps.png) | 2D Spatial temperature reconstruction maps for CBAM-CNN at 0m, 50m, 100m, 500m. | GLORYS Test Set |
| [`cbam_vs_all_depth_rmse.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/cbam_vs_all_depth_rmse.png) | Per-depth RMSE comparison across all 3 models on the GLORYS test grid. | GLORYS Test Set |
| [`cbam_vs_all_depth_mae.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/cbam_vs_all_depth_mae.png) | Per-depth MAE comparison across all 3 models on the GLORYS test grid. | GLORYS Test Set |
| [`depth_wise_rmse.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/depth_wise_rmse.png) | Comparative log-scale depth profile plot of RMSE (°C) for CNN Baseline vs OceanEmbed. | GLORYS Test Set |
| [`embedding_pca.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/embedding_pca.png) | Exploratory 2D PCA scatter plot of learned spatial ocean embeddings colored by test set month. | Latent Analysis |

---

## 5. Processed Dataset Tensors & Sidecars (`data/processed/`)

| File Name / Directory | Description & Specs | Status |
| :--- | :--- | :--- |
| `data/processed/train/` | `X_train.npz` (`[511, 7, 69, 81]`), `Y_train.npz` (`[511, 15, 69, 81]`), `dates_train.npy` | **VALIDATED** |
| `data/processed/val/` | `X_val.npz` (`[110, 7, 69, 81]`), `Y_val.npz` (`[110, 15, 69, 81]`), `dates_val.npy` | **VALIDATED** |
| `data/processed/test/` | `X_test.npz` (`[109, 7, 69, 81]`), `Y_test.npz` (`[109, 15, 69, 81]`), `dates_test.npy` | **VALIDATED** |
| [`coords.nc`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/data/processed/coords.nc) | NetCDF sidecar storing latitude ($5-22^\circ\text{N}$), longitude ($80-100^\circ\text{E}$), and target depth metadata. | **VALIDATED** |
| `land_mask.npy` | Boolean array `[69, 81]` specifying static land grid cells (True = Land). | **VALIDATED** |
| [`normalization_stats.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/normalization_stats.json) | Mean, std, min, max per channel computed strictly from training days. | **VALIDATED** |

---

## 6. Technical Documentation & Reports (`docs/`)

| File Name | Description | Status |
| :--- | :--- | :--- |
| [`FINAL_END_TO_END_REPORT.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/FINAL_END_TO_END_REPORT.md) | The single canonical research-grade final technical report detailing the complete experiment and validation. | **CANONICAL** |
| [`PROJECT_ARTIFACT_INVENTORY.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/PROJECT_ARTIFACT_INVENTORY.md) | Complete file and artifact registry (this document). | **COMPLETE** |
| [`EXPERIMENT_TIMELINE.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/EXPERIMENT_TIMELINE.md) | Chronological development history detailing every stage of the project. | **COMPLETE** |
| [`DATA_SOURCES.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/DATA_SOURCES.md) | Copernicus product catalogue specification and scientific boundary rules. | **FROZEN** |
| [`PROJECT_PLAN.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/PROJECT_PLAN.md) | Original architectural requirements and project plan. | **FROZEN** |
