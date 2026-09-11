# OceanEmbed Prototype (SIH26066) — Chronological Experiment Timeline

This timeline documents the chronological development history of the OceanEmbed prototype from initial domain setup through raw data acquisition, preprocessing, architectural design, autograd debugging, CPU training, test evaluation, embedding extraction, visualization, and documentation.

---

## Chronological Timeline Summary

```
Stage 1: Provenance Audit & Product Rectification (2026-09-10)
  ├── Replaced raw satellite L3 swath products (high missingness >70%) with observation-based gridded analyses
  ├── SST: OSTIA L4 REP (`METOFFICE-GLO-SST-L4-REP-OBS-SST`)
  ├── SSS: Multi-Obs L4 Daily SSS Analysis (`cmems_obs-mob_glo_phy-sss_my_multi_P1D`)
  ├── SLA & Currents: DUACS L4 (`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`)
  ├── Wind: Scatterometer-informed L4 Blended Wind (`cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`)
  └── Target: GLORYS12V1 Reanalysis (`cmems_mod_glo_phy_my_0.083deg_P1D-m`)

Stage 2: 7-Day Prototype Sample Validation (2026-09-10)
  ├── Downloaded 7-day test sample (2022-01-01 -> 2022-01-07) across all variables
  ├── Verified spatial coverage over Bay of Bengal (5-22°N, 80-100°E)
  └── Validated absence of catastrophic data gaps in chosen observation-based analysis products

Stage 3: Full 2-Year Acquisition & Preprocessing Pipeline (2026-09-10)
  ├── Executed `scripts/04_download_full.py` to acquire 24 monthly NetCDF chunks per variable (2022-2023)
  ├── Aggregated 24-step hourly wind fields to daily means
  ├── Executed `scripts/05_preprocess.py`:
  │   ├── Bilinear regridded all surface inputs & target to common 0.25° grid (69 x 81 matrix)
  │   ├── Vertically interpolated GLORYS thetao target to 15 standard SIH depths
  │   ├── Computed per-channel normalization stats strictly from training days (N=511)
  │   └── Assembled ML tensors: Train (511), Val (110), Test (109)
  └── Executed `scripts/06_audit_stage3.py` confirming data integrity and zero GLORYS leakage into inputs

Stage 4: Deep Learning Implementation & Model Training (2026-09-10 -> 2026-09-11)
  ├── Implemented PyTorch model suite in `src/models/`:
  │   ├── `cnn_baseline.py`: Lightweight 2D CNN (191,631 params)
  │   ├── `cnn_encoder.py`: CNN Encoder generating spatial ocean embedding `[N, 128, 17, 20]`
  │   ├── `fno2d.py`: 2D Fourier Neural Operator (4 spectral blocks, 8 Fourier modes)
  │   ├── `depth_decoder.py`: Vectorized Depth-Conditioned Decoder
  │   └── `losses.py`: Masked MSE/MAE loss functions using `torch.masked_select`
  ├── Executed `scripts/test_models.py` forward-pass smoke test (All checks passed ✅)
  ├── Model A Training (`scripts/07_train_cnn.py`):
  │   └── Trained 50 epochs in 418.1s (~7 min). Best Val MSE: 0.575807 °C²
  └── Model B Training (`scripts/08_train_oceanembed.py`, background task-1332):
      └── Trained 46 epochs in 27,100.35s (~7.5 hrs). Early stopping triggered. Best Val MSE: 0.622848 °C² (Epoch 34)

Stage 4 (Cont.): Evaluation, Embedding Extraction & Visualization (2026-09-11)
  ├── Executed `scripts/09_evaluate.py`:
  │   ├── Evaluated test set predictions (109 days): CNN RMSE = 1.1796°C, OceanEmbed RMSE = 1.4736°C
  │   ├── Extracted spatial ocean embeddings `[N, 128, 17, 20]` to NPZ archives
  │   └── Exported `evaluation_summary.json` and `per_depth_metrics.csv`
  ├── Executed `scripts/10_visualize_results.py`:
  │   ├── Rendered 2D spatial reconstruction maps (`reconstruction_maps.png`)
  │   ├── Rendered log-scale per-depth RMSE plot (`depth_wise_rmse.png`)
  │   └── Rendered exploratory 2D PCA cluster plot (`embedding_pca.png`)
  └── Authored comprehensive documentation (`STAGE4_MODEL_REPORT.md`, `FINAL_END_TO_END_PROJECT_REPORT.md`)
```

---

## Detailed Event Log

| Event / Phase | Timestamp (UTC) | Description / Milestone | Output / Log File |
| :--- | :--- | :--- | :--- |
| **Catalog Audit** | 2026-09-10 14:00 | Verified dataset IDs and dimensions with `copernicusmarine` catalogue. | `config/dataset_config.yaml` |
| **Sample Download** | 2026-09-10 15:00 | Downloaded and validated 7-day test sample (Stage 2). | `docs/stage2_validation_report.md` |
| **Full Download** | 2026-09-10 18:00 | Downloaded full 2-year 24-month chunks for 7 inputs + GLORYS target. | `logs/download_log.csv` |
| **Preprocessing** | 2026-09-10 19:40 | Executed regridding, vertical depth interpolation, and train-only normalization. | `docs/STAGE3_DATASET_REPORT.md` |
| **Integrity Audit** | 2026-09-10 19:50 | Audited NPZ tensors, verified zero GLORYS leakage, and confirmed 69x81 dimensions. | `docs/STAGE3_FINAL_AUDIT.md` |
| **Model Codebase** | 2026-09-10 20:00 | Developed modular PyTorch classes (`CNNBaseline`, `OceanEmbed`, `FNO2D`, `DepthDecoder`). | `src/models/` |
| **Autograd Fix** | 2026-09-10 20:05 | Fixed NaN loss backpropagation using `torch.masked_select` in `MaskedMSELoss`. | `src/models/losses.py` |
| **Decoder Opt** | 2026-09-10 20:10 | Vectorized depth-conditioning decoder in `depth_decoder.py`, accelerating forward pass by >80%. | `src/models/depth_decoder.py` |
| **Smoke Test** | 2026-09-10 20:12 | Executed `scripts/test_models.py` forward-pass smoke test. | Terminal Output |
| **CNN Training** | 2026-09-10 20:13 | Trained CNN baseline for 50 epochs ($\sim 7\text{ min}$). | `results/models/cnn_baseline_best.pt` |
| **OceanEmbed Start** | 2026-09-10 20:13 | Launched OceanEmbed training as background `task-1332`. | `task-1332.log` |
| **OceanEmbed Done** | 2026-09-11 03:45 | OceanEmbed training finished (Epoch 46 early stop, Best Val MSE 0.622848). | `results/models/oceanembed_best.pt` |
| **Test Evaluation** | 2026-09-11 03:47 | Executed `scripts/09_evaluate.py` test set evaluation & embedding extraction. | `results/metrics/evaluation_summary.json` |
| **Visualization** | 2026-09-11 03:48 | Executed `scripts/10_visualize_results.py` generating reconstruction maps & PCA plots. | `results/figures/` |
| **Final Documentation** | 2026-09-11 04:20 | Generated final research-grade end-to-end report and artifact registries. | `docs/FINAL_END_TO_END_PROJECT_REPORT.md` |
