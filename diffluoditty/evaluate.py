"""A fixed, seeded benchmark so models of different sizes can be compared.

Grows ``n`` copies from the seed, then measures:

  grow_loss     mean loss over steps [64, 96]  (the window training optimises)
  persist_loss  loss at ``persist_steps``      (does the shape hold long after training's horizon?)
  regen_loss    loss ``regen_steps`` after random circular damage at ``damage_at``
  died / exploded  fraction of copies whose alive area at ``persist_steps`` is
                <10% / >200% of the target's — the two classic failure modes

Only uses the DynamicalSystem interface, so it works for any system.
"""

from __future__ import annotations

import torch

from .damage import circle_masks
from .system import DynamicalSystem, State
from .train import per_sample_loss


def _alive_area(rgba: torch.Tensor) -> torch.Tensor:
    return (rgba[:, 3] > 0.1).float().sum(dim=(1, 2))


@torch.no_grad()
def evaluate(
    system: DynamicalSystem,
    target: torch.Tensor,
    n: int = 16,
    grow_window: tuple[int, int] = (64, 96),
    persist_steps: int = 1000,
    damage_at: int = 200,
    regen_steps: int = 200,
    seed: int = 0,
) -> dict:
    device = system.device
    target = target.to(device)
    h, w = target.shape[-2:]
    was_training = system.training
    system.eval()

    # Fork the RNG so evaluating mid-training doesn't perturb the training run.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        masks = circle_masks(n, h, w, device=device)

        x: State = system.seed(n)
        damaged: State | None = None
        curve, damaged_curve, grow_losses = [], [], []
        for t in range(1, persist_steps + 1):
            x = system.step(x)
            loss = per_sample_loss(system.render(x), target)
            curve.append(loss.mean().item())
            if grow_window[0] <= t <= grow_window[1]:
                grow_losses.append(loss.mean().item())

            if t == damage_at:
                damaged = system.damage(x, masks)
            elif damaged is not None and t <= damage_at + regen_steps:
                damaged = system.step(damaged)
                damaged_curve.append(per_sample_loss(system.render(damaged), target).mean().item())

        area = _alive_area(system.render(x)) / _alive_area(target[None])
        regen_loss = per_sample_loss(system.render(damaged), target).mean().item() if damaged is not None else float("nan")
        final_rgba = system.render(x)[:4].cpu()
        regen_rgba = system.render(damaged)[:4].cpu() if damaged is not None else None

    system.train(was_training)
    return dict(
        grow_loss=sum(grow_losses) / len(grow_losses) if grow_losses else float("nan"),
        persist_loss=curve[-1],
        regen_loss=regen_loss,
        died=(area < 0.1).float().mean().item(),
        exploded=(area > 2.0).float().mean().item(),
        curve=curve,  # mean loss per step, undamaged
        damaged_curve=damaged_curve,  # mean loss per step after damage
        final_rgba=final_rgba,  # a few samples at persist_steps, for thumbnails
        regen_rgba=regen_rgba,  # the same samples, regen_steps after damage
    )


def load_runs(pattern: str, target: torch.Tensor, device=None, **eval_kwargs) -> list[dict]:
    """Load every NCA checkpoint matching ``pattern`` (e.g. "checkpoints/*.pt")
    for side-by-side comparison. Uses the evaluation stored in the checkpoint
    when there is one, otherwise runs ``evaluate`` now."""
    import glob
    import os

    from .nca import NCA

    runs = []
    for path in sorted(glob.glob(pattern)):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = NCA.from_checkpoint(path, device)
        runs.append(dict(
            name=ckpt.get("run_name", os.path.splitext(os.path.basename(path))[0]),
            n_params=sum(p.numel() for p in model.parameters()),
            config=ckpt["system_config"],
            history=ckpt["history"],
            eval=ckpt.get("eval") or evaluate(model, target, **eval_kwargs),
        ))
    return sorted(runs, key=lambda r: r["n_params"])
