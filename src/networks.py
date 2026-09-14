"""Generator and discriminator architectures.

The generator follows the 9-block ResNet design of Zhu et al. (CycleGAN,
ICCV 2017): a reflection-padded 7x7 input convolution, two strided
downsampling stages, nine residual blocks, two transposed-convolution
upsampling stages, and a 7x7 tanh output convolution. A CBAM module is
inserted after every third residual block (blocks 3, 6, 9), giving three
attention stages at 128x128, 256-channel resolution.

The discriminator is the standard 70x70 PatchGAN of Isola et al.
(pix2pix, CVPR 2017), which enforces local texture and color consistency
appropriate for cross-stain translation.
"""

import torch
import torch.nn as nn

from .attention import CBAM


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """CBAM-augmented ResNet generator for 512x512 patches."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 ngf: int = 64, n_blocks: int = 9, cbam_every: int = 3):
        super().__init__()

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]

        bottleneck = []
        for i in range(n_blocks):
            bottleneck.append(ResnetBlock(ngf * 4))
            if (i + 1) % cbam_every == 0:
                bottleneck.append(CBAM(ngf * 4))

        decoder = [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *bottleneck, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the `_orig_mod.` prefix added by torch.compile when saving."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
