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

import torch
import torch.nn.functional as F
from torch import nn

from .system import DynamicalSystem


class NCA(DynamicalSystem):
    def __init__(
        self,
        size: tuple[int, int] = (72, 72),
        channels: int = 16,
        hidden: int = 128,
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

        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # "Do nothing" initial behaviour: the last layer starts at zero, so the
        # untrained CA leaves the seed unchanged. Important for stable training.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

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
