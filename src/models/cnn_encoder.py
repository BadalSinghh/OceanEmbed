# -*- coding: utf-8 -*-
"""
src/models/cnn_encoder.py
=========================
CNN Encoder module for OceanEmbed.

Transforms 7 surface observation channels [N, 7, H, W] into a spatial
latent representation [N, 128, H_lat, W_lat], designated as the
"LEARNED SPATIAL OCEAN EMBEDDING".
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN Encoder for surface satellite observations.

    Input: [N, 7, 69, 81]
    Output: [N, 128, 17, 20] (Learned Spatial Ocean Embedding)
    """

    def __init__(self, in_channels: int = 7, embed_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Block 1: 7 -> 32 channels, 69x81 -> 34x40
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 -> 64 channels, 34x40 -> 17x20
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64 -> 128 channels, 17x20 -> 17x20
        self.block3 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the learned spatial ocean embedding.

        Args:
            x: Tensor of shape [N, 7, H, W]

        Returns:
            embedding: Spatial embedding tensor of shape [N, 128, H_lat, W_lat]
        """
        h1 = self.block1(x)
        h2 = self.block2(h1)
        embedding = self.block3(h2)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
