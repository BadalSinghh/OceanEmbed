# -*- coding: utf-8 -*-
"""
src/models/cbam.py
==================
Convolutional Block Attention Module (CBAM) and CBAM-CNN Architecture.

CBAM integrates Channel Attention and Spatial Attention modules to sequentially
refine intermediate feature representations.

Reference:
Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Aggregates spatial information using both Average Pooling and Max Pooling,
    passing both through a shared MLP to compute channel attention weights.
    """

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.GELU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Aggregates channel information using Average Pooling and Max Pooling along
    the channel axis, then applies a 7x7 convolution to compute spatial weights.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat_out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention followed by Spatial Attention.
    """

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class CBAMCNN(nn.Module):
    """
    CBAM-enhanced 2D CNN Model.

    Controlled modification of CNNBaseline with CBAM modules inserted at key feature
    representation layers in both encoder and decoder.

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

        # Encoder Stage 1: 7 -> 32 channels + CBAM + MaxPool (69x81 -> 34x40)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            CBAM(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Encoder Stage 2: 32 -> 64 channels + CBAM + MaxPool (34x40 -> 17x20)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            CBAM(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Encoder Stage 3: 64 -> 128 channels + CBAM (17x20)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            CBAM(128),
        )

        # Decoder Stage 1: Upsample -> 64 channels + CBAM (17x20 -> 34x40)
        self.dec1 = nn.Sequential(
            nn.Upsample(size=(34, 40), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            CBAM(64),
        )

        # Decoder Stage 2: Upsample -> 32 channels + CBAM (34x40 -> 69x81)
        self.dec2 = nn.Sequential(
            nn.Upsample(size=target_shape, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            CBAM(32),
        )

        # Final Projection Layer: 32 -> 15 depth channels
        self.proj = nn.Conv2d(32, num_depths, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input surface observations tensor [N, 7, 69, 81]

        Returns:
            output: Reconstructed temperature fields [N, 15, 69, 81]
        """
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        latent = self.enc3(h2)

        d1 = self.dec1(latent)
        d2 = self.dec2(d1)
        output = self.proj(d2)

        return output
