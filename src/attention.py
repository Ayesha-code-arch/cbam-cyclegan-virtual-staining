"""Convolutional Block Attention Module (Woo et al., ECCV 2018).

Applied inside the generator bottleneck to focus capacity on the sparse,
punctate chromogenic ISH signal rather than treating every spatial
position uniformly.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel-wise gate from pooled average and max descriptors."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc(self.avg(x)) + self.fc(self.max(x))
        return x * torch.sigmoid(gate)


class SpatialAttention(nn.Module):
    """Spatial gate from a 7x7 convolution over channel-pooled maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        gate = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * torch.sigmoid(gate)


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))
