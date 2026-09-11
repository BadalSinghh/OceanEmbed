# OceanEmbed Prototype (SIH26066)
## Deep Learning Framework for Subsurface Ocean Temperature Reconstruction from Satellite Observations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Validated%20Real%20Argo%20Data-brightgreen.svg)]()

---

### Canonical Technical Documentation

For the complete, end-to-end scientific methodology, architectural formulations, training details, reanalysis benchmark, and independent in-situ CORA delayed-mode Argo validation, see:

👉 **[docs/FINAL_END_TO_END_REPORT.md](docs/FINAL_END_TO_END_REPORT.md)** 👈

---

### Key Research Results at a Glance

Reconstructing 15 subsurface depth levels ($0\text{--}1000\text{ m}$) across the Bay of Bengal ($5\text{--}22^\circ\text{N}$, $80\text{--}100^\circ\text{E}$) from 7 surface observable channels (SST, SSS, SLA, Geostrophic Currents U/V, 10m Wind U/V).

| Model Architecture | Parameters | Independent Argo RMSE (°C) | Independent Argo MAE (°C) | Pearson $R$ | GLORYS Test RMSE (°C) |
|---|:---:|:---:|:---:|:---:|:---:|
| **GLORYS12 (Reanalysis Reference)** | — | 0.8061 | 0.4565 | 0.9958 | — |
| **CNN Baseline** | **191,631** | **1.1611** | **0.6834** | **0.9912** | **1.1796** |
| **CBAM-CNN (Attention)** | 195,705 | 1.3777 | 0.7905 | 0.9874 | 1.3531 |
| **OceanEmbed (CNN+FNO2D)** | 8,670,241 | 1.7113 | 1.1496 | 0.9811 | 1.4736 |

- **Validation Dataset**: 253 unique QC-passed CORA delayed-mode Argo profiles comprising 3,509 matched depth observations across the held-out test period (2023-09-14 to 2023-12-31).
- **Core Finding**: Spaceborne multi-satellite surface observables contain strong dynamical signal ($R > 0.98$) to reconstruct full vertical thermal stratification from 0 to 1000m. CNN Baseline delivers the best overall accuracy, while CBAM-CNN achieves the highest surface mixed layer precision ($0\text{--}30\text{ m}$, $0.610^\circ\text{C}$).

---

### Quickstart & Reproduction

```powershell
# 1. Activate Environment
.venv\Scripts\Activate.ps1

# 2. Run Preprocessing & Model Training
python scripts/05_preprocess.py
python scripts/07_train_cnn.py
python scripts/11_train_cbam_cnn.py
python scripts/08_train_oceanembed.py

# 3. Evaluate & Validate Against Real CORA Argo Profiles
python scripts/09_evaluate.py
python scripts/12_evaluate_cbam_cnn.py
python scripts/13_process_argo_and_validate.py
python scripts/regenerate_argo_figures.py
python scripts/generate_argo_profile_comparisons.py
```
