# OceanEmbed Prototype (SIH26066) — Complete Project Artifact Inventory

This document provides a comprehensive inventory of all source code files, configurations, datasets, trained checkpoints, evaluation metrics, figures, latent embeddings, and technical documentation produced for the OceanEmbed Prototype.

---

## 1. Trained Model Checkpoints (`results/models/`)

| File Name | File Size | Description & Parameters | Status |
| :--- | :---: | :--- | :--- |
| [`cnn_baseline_best.pt`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/models/cnn_baseline_best.pt) | 2.33 MB | PyTorch checkpoint for Model A (CNN Baseline, 191,631 parameters). Saved at Epoch 49 with best validation loss 0.575807 $^{\circ}\text{C}^2$. | **VERIFIED & FROZEN** |
| [`oceanembed_best.pt`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/models/oceanembed_best.pt) | 204.77 MB | PyTorch checkpoint for Model B (OceanEmbed Framework, 8,670,241 parameters). Saved at Epoch 34 with best validation loss 0.622848 $^{\circ}\text{C}^2$. | **VERIFIED & FROZEN** |

---

## 2. Evaluation Metrics & Test Predictions (`results/metrics/`)

| File Name | File Size | Description | Status |
| :--- | :---: | :--- | :--- |
| [`evaluation_summary.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/evaluation_summary.json) | 3.27 KB | Full test set evaluation metrics (RMSE, MAE, $R^2$) for CNN Baseline and OceanEmbed overall and per-depth. | **GENERATED** |
| [`per_depth_metrics.csv`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/per_depth_metrics.csv) | 814 B | Tabular CSV recording per-depth RMSE and MAE across all 15 SIH target depths. | **GENERATED** |
| [`cnn_baseline_history.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/cnn_baseline_history.json) | 2.20 KB | Per-epoch training and validation loss history and timing for Model A. | **GENERATED** |
| [`oceanembed_history.json`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/oceanembed_history.json) | 2.08 KB | Per-epoch training and validation loss history and timing for Model B. | **GENERATED** |
| [`test_predictions.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/metrics/test_predictions.npz) | 81.48 MB | Saved model predictions (`cnn_preds`, `oe_preds`) and targets over the 109 test days. | **GENERATED** |

---

## 3. Extracted Spatial Ocean Embeddings (`results/embeddings/`)

| File Name | File Size | Shape & Contents | Status |
| :--- | :---: | :--- | :--- |
| [`ocean_embeddings_train.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_train.npz) | 82.22 MB | Extracted spatial ocean embeddings for 511 training days (`shape=[511, 128, 17, 20]`). | **EXTRACTED** |
| [`ocean_embeddings_val.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_val.npz) | 17.71 MB | Extracted spatial ocean embeddings for 110 validation days (`shape=[110, 128, 17, 20]`). | **EXTRACTED** |
| [`ocean_embeddings_test.npz`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/embeddings/ocean_embeddings_test.npz) | 17.54 MB | Extracted spatial ocean embeddings for 109 test days (`shape=[109, 128, 17, 20]`). | **EXTRACTED** |

---

## 4. Visualization Figures (`results/figures/`)

| File Name | File Size | Figure Description | Status |
| :--- | :---: | :--- | :--- |
| [`reconstruction_maps.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/reconstruction_maps.png) | 600.18 KB | 2D Spatial temperature reconstruction maps (Target vs OceanEmbed vs Error) at 0m, 100m, 500m, 1000m. | **GENERATED** |
| [`depth_wise_rmse.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/depth_wise_rmse.png) | 113.83 KB | Comparative log-scale depth profile plot of RMSE (°C) for CNN Baseline vs OceanEmbed. | **GENERATED** |
| [`embedding_pca.png`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/results/figures/embedding_pca.png) | 149.13 KB | Exploratory 2D PCA scatter plot of learned spatial ocean embeddings colored by test set month. | **GENERATED** |

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
| [`FINAL_END_TO_END_PROJECT_REPORT.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/FINAL_END_TO_END_PROJECT_REPORT.md) | Comprehensive 20-part research-grade project report detailing the complete experiment from acquisition to results. | **COMPLETE** |
| [`PROJECT_ARTIFACT_INVENTORY.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/PROJECT_ARTIFACT_INVENTORY.md) | Complete file and artifact registry (this document). | **COMPLETE** |
| [`EXPERIMENT_TIMELINE.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/EXPERIMENT_TIMELINE.md) | Chronological development history detailing every stage of the project. | **COMPLETE** |
| [`DATA_SOURCES.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/DATA_SOURCES.md) | Copernicus product catalogue specification and scientific boundary rules. | **FROZEN** |
| [`STAGE3_DATASET_REPORT.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/STAGE3_DATASET_REPORT.md) | Preprocessing and dataset summary report. | **FROZEN** |
| [`STAGE3_FINAL_AUDIT.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/STAGE3_FINAL_AUDIT.md) | Data integrity audit log verifying tensor shapes and non-leakage. | **FROZEN** |
| [`STAGE4_MODEL_REPORT.md`](file:///c:/Users/arush/OneDrive/Desktop/oceanembedPrototype/docs/STAGE4_MODEL_REPORT.md) | Initial Stage 4 model architecture and training report. | **FROZEN** |
