"""How far back through time can we usefully backpropagate?

For each rollout length T, ``gradient_horizon`` computes the loss gradient
from several independent batches and reports:

  grad_norm   size of the gradient. Exponential growth with T = exploding gradients.
  agreement   mean cosine similarity between the gradients of independent batches.
              Near 1: every batch agrees on which way to go — the gradient is signal.
              Near 0: batches disagree — the gradient is mostly chaos-driven noise,
              even if it isn't exploding.

Works for any DynamicalSystem.
"""

from __future__ import annotations

import itertools

import torch

from .system import DynamicalSystem, State, detach_state
from .train import per_sample_loss, rollout


def _flat_grad(system: DynamicalSystem) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in system.parameters() if p.grad is not None])


def gradient_horizon(
    system: DynamicalSystem,
    target: torch.Tensor,
    horizons=(8, 16, 32, 64, 128),
    batch_size: int = 4,
    n_batches: int = 3,
    start: State | None = None,
    warmup: int = 0,
    grad_checkpoint_every: int = 0,
    seed: int = 0,
) -> list[dict]:
    """``start``: a batch of states to begin from (default: fresh seeds, then
    ``warmup`` steps without gradient), e.g. states from a trained pool."""
    target = target.to(system.device)
    results = []
    with torch.random.fork_rng(devices=[system.device] if system.device.type == "cuda" else []):
        for T in horizons:
            grads, losses, per_param = [], [], {}
            for b in range(n_batches):
                torch.manual_seed(seed + b)
                if start is None:
                    x = system.seed(batch_size)
                    with torch.no_grad():
                        x = rollout(system, x, warmup)
                else:
                    x = start
                x = detach_state(x)
                system.zero_grad(set_to_none=True)
                loss = per_sample_loss(system.render(rollout(system, x, T, grad_checkpoint_every)), target).mean()
                loss.backward()
                grads.append(_flat_grad(system))
                losses.append(loss.item())
                for name, p in system.named_parameters():
                    if p.grad is not None:
                        per_param.setdefault(name, []).append(p.grad.norm().item())
            cos = [torch.nn.functional.cosine_similarity(a, b, dim=0).item() for a, b in itertools.combinations(grads, 2)]
            results.append(dict(
                horizon=T,
                loss=sum(losses) / len(losses),
                grad_norm=sum(g.norm().item() for g in grads) / len(grads),
                agreement=sum(cos) / len(cos) if cos else float("nan"),
                per_param={k: sum(v) / len(v) for k, v in per_param.items()},
            ))
    system.zero_grad(set_to_none=True)
    return results


def plot_gradient_horizon(results: list[dict]) -> None:
    import matplotlib.pyplot as plt

    T = [r["horizon"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    axes[0].loglog(T, [r["grad_norm"] for r in results], "o-")
    axes[0].set_title("gradient norm")
    axes[1].semilogx(T, [r["agreement"] for r in results], "o-")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylim(-0.2, 1.05)
    axes[1].set_title("agreement between independent batches (cosine)")
    for name in results[0]["per_param"]:
        axes[2].loglog(T, [r["per_param"][name] for r in results], "o-", label=name, markersize=3)
    axes[2].set_title("gradient norm per parameter")
    axes[2].legend(fontsize=6, ncol=2)
    for ax in axes:
        ax.set_xlabel("backprop horizon (steps)")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plt.show()
