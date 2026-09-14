"""Auxiliary structural losses applied to cycle reconstructions.

SSIM and Sobel-edge terms are computed only between an input image and
its own cycle reconstruction (same domain), so they remain valid despite
the ~20um physical offset between adjacent ISH/H&E sections.
"""

import torch
import torch.nn.functional as F


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, computed with torchmetrics when available."""
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        return 1.0 - ssim_fn(
            (x * 0.5 + 0.5).clamp(0, 1),
            (y * 0.5 + 0.5).clamp(0, 1),
            data_range=1.0,
        )
    except ImportError:
        return _ssim_loss_fallback(x, y)


def _ssim_loss_fallback(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    xp = F.pad((x * 0.5 + 0.5).clamp(0, 1), [pad] * 4, mode="reflect")
    yp = F.pad((y * 0.5 + 0.5).clamp(0, 1), [pad] * 4, mode="reflect")

    mu_x = F.avg_pool2d(xp, window, 1, 0)
    mu_y = F.avg_pool2d(yp, window, 1, 0)
    sigma_x = F.avg_pool2d(xp * xp, window, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(yp * yp, window, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(xp * yp, window, 1, 0) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - ssim_map.mean()


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude on the luminance channel."""
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def cycle_edge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_magnitude(x), sobel_magnitude(y))
