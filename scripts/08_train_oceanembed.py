# -*- coding: utf-8 -*-
"""
scripts/08_train_oceanembed.py
==============================
Train Model B: OceanEmbed (CNN + FNO2D + Depth-Conditioned Decoder).

Input: [N, 7, 69, 81] (surface satellite observations)
Learned Spatial Embedding: [N, 128, 17, 20]
Output: [N, 15, 69, 81] (subsurface temperature fields at 15 depths)

Train: 511 days (2022-01-01 -> 2023-05-26)
Val  : 110 days (2023-05-27 -> 2023-09-13)
Test : 109 days (2023-09-14 -> 2023-12-31)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import MaskedMSELoss, OceanEmbed

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class OceanDataset(Dataset):
    def __init__(self, split_name: str):
        split_dir = ROOT / "data" / "processed" / split_name
        self.X = np.load(split_dir / f"X_{split_name}.npz")["data"].astype(np.float32)
        self.Y = np.load(split_dir / f"Y_{split_name}.npz")["data"].astype(np.float32)
        self.dates = np.load(split_dir / f"dates_{split_name}.npy")

        # Replace input NaNs with 0.0 (mean post-normalization)
        self.X = np.nan_to_num(self.X, nan=0.0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx]), str(self.dates[idx])


def main():
    print("=" * 70)
    print("STAGE 4 — TRAINING MODEL B: OCEANEMBED (CNN + FNO2D + DECODER)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output paths
    models_dir = ROOT / "results" / "models"
    metrics_dir = ROOT / "results" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / "oceanembed_best.pt"
    history_path = metrics_dir / "oceanembed_history.json"

    # Datasets and Loaders
    train_dataset = OceanDataset("train")
    val_dataset   = OceanDataset("val")

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model = OceanEmbed(
        in_channels=7,
        num_depths=15,
        embed_dim=128,
        fno_modes=8,
        fno_layers=4,
        target_shape=(69, 81),
    ).to(device)

    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    num_epochs = 50
    patience = 12
    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "epoch_time_sec": []}

    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_b, y_b, _ in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_b, y_b, _ in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                loss = criterion(pred, y_b)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        t_epoch = time.time() - t0

        scheduler.step(avg_val_loss)

        history["train_loss"].append(round(avg_train_loss, 6))
        history["val_loss"].append(round(avg_val_loss, 6))
        history["epoch_time_sec"].append(round(t_epoch, 2))

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {t_epoch:.2f}s")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"  [SAVED CHECKPOINT] New best val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_training_time = time.time() - start_total_time
    history["total_training_time_sec"] = round(total_training_time, 2)
    history["best_val_loss"] = round(best_val_loss, 6)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nOceanEmbed Training Complete! Total time: {total_training_time:.2f}s | Best Val Loss: {best_val_loss:.6f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
