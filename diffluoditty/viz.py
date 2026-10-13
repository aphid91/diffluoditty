"""Matplotlib helpers for watching training and the trained system."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from .system import DynamicalSystem, State


def to_rgb(rgba: torch.Tensor) -> np.ndarray:
    """Premultiplied (B, 4, H, W) or (4, H, W) RGBA -> (..., H, W, 3) numpy RGB on white."""
    rgba = rgba.detach().float().cpu()
    rgb, a = rgba[..., :3, :, :], rgba[..., 3:4, :, :].clamp(0, 1)
    out = (1.0 - a + rgb).clamp(0, 1)
    return out.movedim(-3, -1).numpy()


def show_images(images: np.ndarray, titles=None, ncols: int | None = None, scale: float = 1.6):
    """Show a stack of (N, H, W, 3) images in a grid."""
    n = len(images)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(scale * ncols, scale * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_history(history: list[dict], ax=None):
    """log10(loss) per iteration, plus a smoothed line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    loss = np.log10([r["loss"] for r in history])
    ax.plot(loss, ".", alpha=0.15, markersize=2)
    if len(loss) >= 50:
        k = 50
        ax.plot(np.arange(k - 1, len(loss)), np.convolve(loss, np.ones(k) / k, mode="valid"))
    ax.set_xlabel("iteration")
    ax.set_ylabel("log10(loss)")
    ax.grid(alpha=0.3)
    return ax


def show_progress(trainer) -> None:
    """Training callback: loss curve, and the last batch before/after its rollout."""
    from IPython.display import clear_output

    clear_output(wait=True)
    system, hist = trainer.system, trainer.history
    n = trainer.config.batch_size

    fig = plt.figure(figsize=(max(8, 1.3 * n), 6))
    gs = fig.add_gridspec(3, n, height_ratios=[2, 1, 1])
    plot_history(hist, fig.add_subplot(gs[0, :]))
    rows = [("before", trainer.last_start), ("after", trainer.last_end)]
    for r, (label, state) in enumerate(rows, start=1):
        imgs = to_rgb(system.render(state))
        for i in range(n):
            ax = fig.add_subplot(gs[r, i])
            ax.imshow(imgs[i])
            ax.axis("off")
            if i == 0:
                ax.set_title(label, fontsize=8, loc="left")
    last = hist[-1]
    fig.suptitle(
        f"step {last['step']}  loss {last['loss']:.5f}  lr {last['lr']:.1e}  "
        f"{last['seconds']*1000:.0f} ms/it"
    )
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def record(
    system: DynamicalSystem,
    state: State,
    n_steps: int,
    every: int = 1,
    events: dict | None = None,
) -> list[np.ndarray]:
    """Run the system and collect rendered frames as (B, H, W, 3) arrays.

    ``events`` maps step number -> function(state) -> state, e.g.
    ``{200: lambda s: system.damage(s, mask)}`` to damage the pattern at step 200.
    """
    events = events or {}
    frames = [to_rgb(system.render(state))]
    for t in range(1, n_steps + 1):
        if t in events:
            state = events[t](state)
        state = system.step(state)
        if t % every == 0:
            frames.append(to_rgb(system.render(state)))
    return frames


def animate(frames: list[np.ndarray], fps: int = 30, scale: float = 2.0):
    """Turn frames into a matplotlib animation. Each frame may be (H, W, 3) or a
    batch (B, H, W, 3), which is tiled horizontally.

    In a notebook: ``from IPython.display import HTML; HTML(animate(frames).to_jshtml())``
    To save:       ``animate(frames).save("out.mp4")`` (or ``"out.gif"``)
    """
    tiles = [np.concatenate(list(f), axis=1) if f.ndim == 4 else f for f in frames]
    h, w = tiles[0].shape[:2]
    fig = plt.figure(figsize=(scale * w / 72, scale * h / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(tiles[0], interpolation="nearest")
    plt.close(fig)

    def update(i):
        im.set_data(tiles[i])
        return (im,)

    return animation.FuncAnimation(fig, update, frames=len(tiles), interval=1000 / fps, blit=True)
