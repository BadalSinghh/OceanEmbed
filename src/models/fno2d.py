# -*- coding: utf-8 -*-
"""
src/models/fno2d.py
===================
2D Fourier Neural Operator (FNO2D) module for OceanEmbed.

Operates on the 2D spatial latent representation (Ocean Embedding) using
spectral convolution layers in the Fourier domain to capture large-scale,
multi-scale spatial relationships.
"""

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Performs FFT, linear transform on low modes, and IFFT.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 8, modes2: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in latitude
        self.modes2 = modes2  # Number of Fourier modes to keep in longitude

        scale = 1.0 / (in_channels * out_channels)
        # Complex weights for low modes
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # input: [batch, in_channel, x, y]
        # weights: [in_channel, out_channel, x, y]
        # output: [batch, out_channel, x, y]
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        size_h, size_w = x.shape[2], x.shape[3]

        # Compute 2D Real FFT
        x_ft = torch.fft.rfft2(x)

        # Allocate output tensor in Fourier domain
        out_ft = torch.zeros(
            batchsize, self.out_channels, size_h, size_w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Truncate modes to dimensions available
        m1 = min(self.modes1, size_h // 2)
        m2 = min(self.modes2, size_w // 2 + 1)

        # Multiply low modes
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # Compute Inverse 2D Real FFT
        x_out = torch.fft.irfft2(out_ft, s=(size_h, size_w))
        return x_out


class FNO2D(nn.Module):
    """
    Lightweight 2D Fourier Neural Operator consisting of 4 Fourier layers.

    Input: [N, in_channels, H_lat, W_lat]
    Output: [N, out_channels, H_lat, W_lat]
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        modes1: int = 8,
        modes2: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.fourier_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.fourier_layers.append(
                SpectralConv2d(in_channels, out_channels, modes1=modes1, modes2=modes2)
            )
            self.w_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier layers.

        Args:
            x: Input spatial embedding [N, 128, H_lat, W_lat]

        Returns:
            FNO-processed spatial representation [N, 128, H_lat, W_lat]
        """
        h = x
        for i in range(self.num_layers):
            x1 = self.fourier_layers[i](h)
            x2 = self.w_layers[i](h)
            h = self.act(x1 + x2)
        return h
