"""Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020).

https://distill.pub/2020/growing-ca/

State: a (B, C, H, W) tensor. Channels 0-2 are premultiplied RGB, channel 3 is
alpha (the "alive" signal), and the remaining C-4 channels are hidden state the
network can use however it likes.

One step:
  1. perceive:  each cell looks at its 3x3 neighbourhood through fixed filters
                (identity, Sobel-x, Sobel-y) applied per channel  -> 3C features
  2. update:    a per-cell MLP (1x1 convs) maps those features to a delta  -> C
  3. stochastic: each cell applies its delta with probability ``fire_rate``
                (no global clock: cells update asynchronously)
  4. alive mask: a cell is alive if any cell in its 3x3 neighbourhood has
                alpha > 0.1, both before and after the update. Dead cells are
                zeroed entirely.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .system import DynamicalSystem


class NCA(DynamicalSystem):
    def __init__(
        self,
        size: tuple[int, int] = (72, 72),
        channels: int = 16,
        hidden: int | Sequence[int] = 128,  # width of the update MLP's hidden layer(s); 0 or () = linear
        fire_rate: float = 0.5,
        alive_threshold: float = 0.1,
        padding_mode: str = "zeros",  # "zeros" matches the paper; "circular" is a common variant
    ):
        super().__init__()
        self.size = tuple(size)
        self.channels = channels
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold
        self.padding_mode = padding_mode

        # Fixed perception filters: identity, Sobel-x, Sobel-y. Stored as a
        # buffer (not a parameter) so it moves with .to(device) but isn't trained.
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = sobel_x.T
        kernels = torch.stack([identity, sobel_x, sobel_y])  # (3, 3, 3)
        # Depthwise conv weight: (C*3, 1, 3, 3), ordered [c0:id, c0:dx, c0:dy, c1:id, ...]
        self.register_buffer("perception_kernel", kernels.repeat(channels, 1, 1).unsqueeze(1))

        widths = [hidden] if isinstance(hidden, int) else list(hidden)
        widths = [w for w in widths if w > 0]
        # Everything needed to rebuild this model: saved alongside checkpoints.
        self.config = dict(
            size=self.size, channels=channels, hidden=widths, fire_rate=fire_rate,
            alive_threshold=alive_threshold, padding_mode=padding_mode,
        )

        dims = [channels * 3, *widths, channels]
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                layers.append(nn.ReLU())
            layers.append(nn.Conv2d(d_in, d_out, kernel_size=1))
        self.update = nn.Sequential(*layers)
        # "Do nothing" initial behaviour: the last layer starts at zero, so the
        # untrained CA leaves the seed unchanged. Important for stable training.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "NCA":
        """Rebuild a trained NCA (shape and weights) from a ``Trainer.save`` file."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(**ckpt["system_config"]).to(device or "cpu")
        model.load_state_dict(ckpt["system"])
        return model

    # --- DynamicalSystem interface -------------------------------------------

    def seed(self, n: int) -> torch.Tensor:
        """A single live cell in the centre with all non-RGB channels set to 1."""
        h, w = self.size
        x = torch.zeros(n, self.channels, h, w, device=self.device)
        x[:, 3:, h // 2, w // 2] = 1.0
        return x

    def step(self, x: torch.Tensor) -> torch.Tensor:
        pre_alive = self.alive_mask(x)

        dx = self.update(self.perceive(x))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= self.fire_rate).float()
        x = x + dx * fire

        alive = pre_alive & self.alive_mask(x)
        return x * alive.float()

    def render(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :4]

    def damage(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - mask)

    # --- pieces of the step ----------------------------------------------------

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "circular":
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
            return F.conv2d(x, self.perception_kernel, groups=self.channels)
        return F.conv2d(x, self.perception_kernel, padding=1, groups=self.channels)

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.alive_threshold


def nca_param_count(channels: int, hidden: int | Sequence[int]) -> int:
    """Trainable parameters of ``NCA(channels=..., hidden=...)``, without building it.

    For one hidden layer of width H: (3C*H + H) + (H*C + C) = 4CH + H + C.
    """
    widths = [hidden] if isinstance(hidden, int) else list(hidden)
    dims = [channels * 3, *[w for w in widths if w > 0], channels]
    return sum(d_in * d_out + d_out for d_in, d_out in zip(dims[:-1], dims[1:]))


def shapes_for_budget(budget: int, channels=range(4, 33), tolerance: float = 0.15) -> list[tuple[int, int, int]]:
    """Single-hidden-layer (channels, hidden, n_params) shapes within ``tolerance`` of ``budget``."""
    out = []
    for c in channels:
        h = round((budget - c) / (4 * c + 1))
        n = nca_param_count(c, h)
        if h >= 1 and abs(n - budget) <= tolerance * budget:
            out.append((c, h, n))
    return out
