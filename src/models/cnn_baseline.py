# -*- coding: utf-8 -*-
"""
src/models/cnn_baseline.py
===========================
Lightweight CNN Baseline Model (Model A).

Conventional CNN encoder-decoder architecture without FNO and without depth conditioning.
Used to establish a quantitative baseline against OceanEmbed.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Lightweight conventional CNN baseline model.

    Input: [N, 7, 69, 81]
    Output: [N, 15, 69, 81]
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_depths: int = 15,
        target_shape: tuple = (69, 81),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths
        self.target_shape = target_shape

        # Encoder: 7 -> 32 -> 64 -> 128 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 69x81 -> 34x40

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 34x40 -> 17x20

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),  # [N, 128, 17, 20]
        )

        # Decoder: 128 -> 64 -> 32 -> 15 channels
        self.decoder = nn.Sequential(
            nn.Upsample(size=(34, 40), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Upsample(size=target_shape, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, num_depths, kernel_size=3, padding=1),  # [N, 15, 69, 81]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input surface observations tensor [N, 7, 69, 81]

        Returns:
            output: Reconstructed temperature fields [N, 15, 69, 81]
        """
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
