# -*- coding: utf-8 -*-
"""
src/models/depth_decoder.py
===========================
Depth-Conditioned Decoder module for OceanEmbed.

Reconstructs 3D subsurface ocean temperature across 15 standard depth levels
from the FNO spatial latent representation, conditioned on learned depth embeddings.
"""

import torch
import torch.nn as nn


class DepthConditionedDecoder(nn.Module):
    """
    Depth-conditioned spatial decoder.

    Input: [N, 128, 17, 20]
    Output: [N, 15, 69, 81] (Subsurface temperature at 15 depths)
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_depths: int = 15,
        depth_embed_dim: int = 32,
        target_shape: tuple = (69, 81),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths
        self.depth_embed_dim = depth_embed_dim
        self.target_shape = target_shape

        # Learnable depth embeddings for 15 standard depth levels
        self.depth_embedding = nn.Embedding(num_depths, depth_embed_dim)

        # Spatial upsampling decoder stages
        # Stage 1: 128 -> 64 channels, 17x20 -> 34x40
        self.up1 = nn.Sequential(
            nn.Upsample(size=(34, 40), mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Stage 2: 64 -> 32 channels, 34x40 -> 69x81
        self.up2 = nn.Sequential(
            nn.Upsample(size=target_shape, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Stage 3: Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # Depth-conditioning interaction:
        # Maps 32 spatial channels + 32 depth embedding channels -> 1 temperature field per depth level
        self.depth_projector = nn.Sequential(
            nn.Conv2d(32 + depth_embed_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, fno_latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass reconstructing 15 depth temperature fields.

        Args:
            fno_latent: Tensor of shape [N, 128, H_lat, W_lat]

        Returns:
            temp_fields: Reconstructed temperature tensor of shape [N, 15, 69, 81]
        """
        batch_size = fno_latent.shape[0]

        # Upsample spatial latent representation
        h1 = self.up1(fno_latent)  # [N, 64, 34, 40]
        h2 = self.up2(h1)          # [N, 32, 69, 81]
        feat = self.refine(h2)     # [N, 32, 69, 81]

        H, W = self.target_shape

        # Vectorized depth-conditioning interaction across all 15 depth levels
        feat_exp = feat.unsqueeze(1).expand(-1, self.num_depths, -1, -1, -1).reshape(batch_size * self.num_depths, 32, H, W)

        d_indices = torch.arange(self.num_depths, device=fno_latent.device)
        d_embeds = self.depth_embedding(d_indices)  # [15, 32]
        d_embeds_exp = d_embeds.view(1, self.num_depths, self.depth_embed_dim, 1, 1).expand(batch_size, -1, -1, H, W).reshape(batch_size * self.num_depths, self.depth_embed_dim, H, W)

        cat_feat = torch.cat([feat_exp, d_embeds_exp], dim=1)  # [N*15, 64, H, W]
        temp_flat = self.depth_projector(cat_feat)            # [N*15, 1, H, W]
        out = temp_flat.view(batch_size, self.num_depths, H, W) # [N, 15, H, W]

        return out
