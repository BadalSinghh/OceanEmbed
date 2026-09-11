# -*- coding: utf-8 -*-
"""
scripts/test_models.py
======================
Synthetic Forward-Pass Smoke Test for Stage 4 Models.

Verifies:
1. CUDA availability and device assignment
2. NPZ tensor shapes & dtypes
3. Parameter counts for CNN baseline and OceanEmbed
4. Forward pass execution for CNNBaseline
5. Forward pass execution for OceanEmbed components:
   - CNNEncoder -> Learned Spatial Ocean Embedding [N, 128, 17, 20]
   - FNO2D -> Latent spatial representation [N, 128, 17, 20]
   - DepthConditionedDecoder -> Reconstructed temperature fields [N, 15, 69, 81]
6. Masked MSE loss calculation is finite
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CNNBaseline, MaskedMSELoss, OceanEmbed


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print("OCEAEMBED STAGE 4 — SYNTHETIC FORWARD-PASS SMOKE TEST")
    print("=" * 70)

    # 1. Environment & CUDA Check
    print("\n1. Environment & Hardware Probe:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  Active Device  : {device}")
    if torch.cuda.is_available():
        print(f"  Device Name    : {torch.cuda.get_device_name(0)}")

    # 2. Inspect Stage 3 Processed Data Specs
    print("\n2. Stage 3 Processed Data Specs Check:")
    x_sample = np.load(ROOT / "data" / "processed" / "train" / "X_train.npz")["data"]
    y_sample = np.load(ROOT / "data" / "processed" / "train" / "Y_train.npz")["data"]
    print(f"  X_train shape: {x_sample.shape} | dtype: {x_sample.dtype}")
    print(f"  Y_train shape: {y_sample.shape} | dtype: {y_sample.dtype}")

    # Memory Estimation
    batch_size = 4
    x_mem_mb = (batch_size * 7 * 69 * 81 * 4) / (1024 * 1024)
    y_mem_mb = (batch_size * 15 * 69 * 81 * 4) / (1024 * 1024)
    print(f"  Batch size {batch_size} memory: X = {x_mem_mb:.2f} MB | Y = {y_mem_mb:.2f} MB")

    # 3. Create Models
    print("\n3. Instantiating Models:")
    cnn_baseline = CNNBaseline(in_channels=7, num_depths=15).to(device)
    ocean_embed  = OceanEmbed(in_channels=7, num_depths=15, embed_dim=128, fno_modes=8, fno_layers=4).to(device)

    param_cnn = count_parameters(cnn_baseline)
    param_oe  = count_parameters(ocean_embed)

    print(f"  Model A (CNN Baseline) trainable parameters: {param_cnn:,}")
    print(f"  Model B (OceanEmbed)  trainable parameters: {param_oe:,}")

    # 4. Synthetic Input Batch
    print("\n4. Running Forward-Pass Smoke Test...")
    x_test = torch.randn(batch_size, 7, 69, 81, device=device)
    y_test = torch.randn(batch_size, 15, 69, 81, device=device)
    # Simulate NaNs in target for masked loss testing
    y_test[y_test < -1.0] = float("nan")

    # 4a. CNN Baseline Forward Pass
    out_cnn = cnn_baseline(x_test)
    print(f"  CNN Baseline Output Shape : {list(out_cnn.shape)}  (Expected: [4, 15, 69, 81])")
    assert out_cnn.shape == (batch_size, 15, 69, 81), "CNN output shape mismatch!"

    # 4b. OceanEmbed Step-by-Step Forward Pass & API Check
    embedding = ocean_embed.encode(x_test)
    print(f"  [EXPLICIT API] model.encode(x) -> Learned Spatial Ocean Embedding Shape: {list(embedding.shape)}  (Expected: [4, 128, 17, 20])")
    assert embedding.shape == (batch_size, 128, 17, 20), "Embedding shape mismatch!"

    fno_latent = ocean_embed.fno(embedding)
    print(f"  FNO2D Output Shape                             : {list(fno_latent.shape)}  (Expected: [4, 128, 17, 20])")
    assert fno_latent.shape == (batch_size, 128, 17, 20), "FNO latent shape mismatch!"

    out_oe = ocean_embed(x_test)
    print(f"  OceanEmbed Final Reconstructed Output Shape    : {list(out_oe.shape)}  (Expected: [4, 15, 69, 81])")
    assert out_oe.shape == (batch_size, 15, 69, 81), "OceanEmbed output shape mismatch!"

    # 5. Masked Loss Computation Test
    print("\n5. Testing Masked Loss Function...")
    criterion = MaskedMSELoss()
    loss_cnn = criterion(out_cnn, y_test)
    loss_oe  = criterion(out_oe, y_test)

    print(f"  CNN Baseline Synthetic Masked MSE Loss : {loss_cnn.item():.6f}")
    print(f"  OceanEmbed Synthetic Masked MSE Loss   : {loss_oe.item():.6f}")

    assert torch.isfinite(loss_cnn), "CNN Loss is not finite!"
    assert torch.isfinite(loss_oe), "OceanEmbed Loss is not finite!"

    print("\n" + "=" * 70)
    print("ALL FORWARD-PASS SMOKE TESTS PASSED SUCCESSFULLY! ✅")
    print("Architecture design is verified and ready for training script creation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
