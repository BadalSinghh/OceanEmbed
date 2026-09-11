# -*- coding: utf-8 -*-
"""
src/models package
"""
from .cbam import CBAM, CBAMCNN, ChannelAttention, SpatialAttention
from .cnn_baseline import CNNBaseline
from .cnn_encoder import CNNEncoder
from .depth_decoder import DepthConditionedDecoder
from .fno2d import FNO2D, SpectralConv2d
from .losses import MaskedMAELoss, MaskedMSELoss, compute_metrics, compute_per_depth_metrics
from .ocean_embed import OceanEmbed

__all__ = [
    "CNNEncoder",
    "SpectralConv2d",
    "FNO2D",
    "DepthConditionedDecoder",
    "OceanEmbed",
    "CNNBaseline",
    "CBAM",
    "CBAMCNN",
    "ChannelAttention",
    "SpatialAttention",
    "MaskedMSELoss",
    "MaskedMAELoss",
    "compute_metrics",
    "compute_per_depth_metrics",
]
