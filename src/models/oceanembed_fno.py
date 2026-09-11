"""
oceanembed_fno.py

OceanEmbed Framework Architecture:
7 Surface Channels -> CNN Encoder -> Compact Spatial Ocean Embedding -> FNO-2D Blocks -> Depth Decoder -> 15 Temperature Maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Convolution layer.
    Performs Fourier transformation, filters high-frequency modes, multiplies with complex weights, and inverse transforms.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes along latitude
        self.modes2 = modes2  # Number of Fourier modes along longitude

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute 2D Real FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical spatial domain via Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2DBlock(nn.Module):
    """
    Single FNO-2D layer combining Fourier Spectral Convolution with spatial 1x1 bypass.
    """
    def __init__(self, width, modes1=12, modes2=12):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.w_conv = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.w_conv(x)
        return F.gelu(x1 + x2)


class OceanEmbedFNO2D(nn.Module):
    """
    OceanEmbed Proposed Architecture for SIH26066:
    7 Surface Satellite Channels -> CNN Encoder -> Compact Spatial Ocean Embedding -> FNO-2D Spectral Blocks -> Depth Decoder -> 15 Temperature Slices.
    """
    def __init__(self, in_channels=7, out_channels=15, width=64, modes1=12, modes2=12, num_fno_blocks=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width

        # 1. CNN Encoder: 7 surface channels -> spatial ocean embedding
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU()
        )

        # 2. Compact Spatial Ocean Embedding refinement
        self.embedding_refiner = nn.Conv2d(width, width, kernel_size=1)

        # 3. FNO-2D Spectral Operator Blocks
        self.fno_blocks = nn.ModuleList([
            FNO2DBlock(width, modes1, modes2) for _ in range(num_fno_blocks)
        ])

        # 4. Depth Decoder: Latent spatial embedding -> 15 requested temperature depth maps
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, 7, Lat, Lon) representing 7 surface satellite channels
        Returns: Tensor of shape (B, 15, Lat, Lon) representing 15 subsurface temperature levels
        """
        # CNN Encoder
        feat = self.cnn_encoder(x)
        
        # Spatial Ocean Embedding
        embedding = self.embedding_refiner(feat)

        # FNO-2D Blocks in Spectral Space
        out = embedding
        for block in self.fno_blocks:
            out = block(out)

        # Depth Decoder to 15 temperature maps
        predicted_depth_maps = self.depth_decoder(out)
        return predicted_depth_maps
