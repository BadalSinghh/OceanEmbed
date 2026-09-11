# -*- coding: utf-8 -*-
"""
src/models/ocean_embed.py
=========================
OceanEmbed Model (Model B).

Architecture:
7 Surface Channels -> CNN Encoder -> LEARNED SPATIAL OCEAN EMBEDDING -> FNO2D -> Depth-Conditioned Decoder -> 15 Subsurface Temperatures
"""

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .depth_decoder import DepthConditionedDecoder
from .fno2d import FNO2D


class OceanEmbed(nn.Module):
    """
    OceanEmbed deep-learning architecture for subsurface ocean temperature reconstruction.

    Input: [N, 7, 69, 81] (SST, SSS, SLA, Current_U, Current_V, Wind_U, Wind_V)
    Output: [N, 15, 69, 81] (Subsurface temperature fields at 15 SIH depths)
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_depths: int = 15,
        embed_dim: int = 128,
        fno_modes: int = 8,
        fno_layers: int = 4,
        target_shape: tuple = (69, 81),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_depths = num_depths
        self.embed_dim = embed_dim

        # 1. CNN Encoder
        self.encoder = CNNEncoder(in_channels=in_channels, embed_dim=embed_dim)

        # 2. 2D Fourier Neural Operator (FNO2D)
        self.fno = FNO2D(
            in_channels=embed_dim,
            out_channels=embed_dim,
            modes1=fno_modes,
            modes2=fno_modes,
            num_layers=fno_layers,
        )

        # 3. Depth-Conditioned Decoder
        self.decoder = DepthConditionedDecoder(
            in_channels=embed_dim,
            num_depths=num_depths,
            depth_embed_dim=32,
            target_shape=target_shape,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the learned spatial ocean embedding.

        Args:
            x: Input surface observations tensor [N, 7, 69, 81]

        Returns:
            embedding: Learned spatial ocean embedding tensor [N, 128, 17, 20]
        """
        return self.encoder.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass reconstructing 15 subsurface temperature depth levels.

        Args:
            x: Input surface observations tensor [N, 7, 69, 81]

        Returns:
            output: Reconstructed temperature fields [N, 15, 69, 81]
        """
        # 1. CNN Encoder -> Spatial Ocean Embedding
        embedding = self.encode(x)  # [N, 128, 17, 20]

        # 2. FNO2D -> Global Spatial Relational Feature Representation
        fno_latent = self.fno(embedding)  # [N, 128, 17, 20]

        # 3. Depth-Conditioned Decoder -> 15 Subsurface Temperature Fields
        output = self.decoder(fno_latent)  # [N, 15, 69, 81]

        return output
