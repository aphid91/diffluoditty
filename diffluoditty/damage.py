"""Damage masks, in render (image) space. 1.0 = destroy, 0.0 = keep.

Masks are system-agnostic: each DynamicalSystem decides what "destroy the part
of the state under this mask" means.
"""

from __future__ import annotations

import torch


def _grid(h: int, w: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates in [-1, 1], shaped (1, H, W)."""
    y = torch.linspace(-1.0, 1.0, h, device=device).view(1, h, 1)
    x = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, w)
    return y, x


def circle_masks(
    n: int,
    h: int,
    w: int,
    device=None,
    center_range: float = 0.5,
    radius_range: tuple[float, float] = (0.1, 0.4),
) -> torch.Tensor:
    """Random circles, as in the paper: centre in [-0.5, 0.5]^2, radius in [0.1, 0.4]."""
    y, x = _grid(h, w, device)
    center = (torch.rand(n, 2, 1, 1, device=device) * 2 - 1) * center_range
    r_lo, r_hi = radius_range
    radius = torch.rand(n, 1, 1, device=device) * (r_hi - r_lo) + r_lo
    dist2 = (x - center[:, 0]) ** 2 + (y - center[:, 1]) ** 2
    return (dist2 < radius**2).float().unsqueeze(1)  # (n, 1, H, W)


def half_mask(n: int, h: int, w: int, side: str = "right", device=None) -> torch.Tensor:
    """Destroy one half of the image: side in {'left', 'right', 'top', 'bottom'}."""
    y, x = _grid(h, w, device)
    region = {"left": x < 0, "right": x > 0, "top": y < 0, "bottom": y > 0}[side]
    return region.expand(n, h, w).float().unsqueeze(1)
