"""Training a DynamicalSystem to grow (and maintain, and regenerate) a target image.

The recipe from the paper, in one place:

  1. Draw a batch of starting states: either fresh seeds, or (with the sample
     pool) states produced by earlier training iterations.
  2. Pool only: replace the worst sample with a fresh seed, so the system
     never forgets how to grow from scratch; optionally damage the best few,
     so it learns to repair.
  3. Run the system for a random number of steps (64-96), keeping the graph.
  4. Loss = MSE between the rendered final state and the target.
  5. Backprop through all of those steps; normalise each parameter's gradient
     to unit L2 norm (this is what keeps BPTT through ~100 steps stable);
     take an Adam step.
  6. Pool only: write the final states back into the pool.

The loop only touches states through the DynamicalSystem interface, so it is
equally happy with an NCA or with something stranger.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace

import torch
from torch.utils.checkpoint import checkpoint

from .damage import circle_masks
from .system import (
    DynamicalSystem,
    State,
    batch_size,
    cat_states,
    detach_state,
    index_state,
    tree_leaves,
)


@dataclass
class TrainConfig:
    batch_size: int = 8
    pool_size: int = 1024
    use_pool: bool = True  # False -> every batch starts from the seed (Experiment 1)
    damage_n: int = 3  # how many of the best samples in each batch get damaged (Experiment 3)
    min_steps: int = 64  # rollout length is uniform in [min_steps, max_steps]
    max_steps: int = 96
    lr: float = 2e-3
    lr_milestones: list[int] = field(default_factory=lambda: [2000])
    lr_gamma: float = 0.1  # 2e-3 -> 2e-4 at step 2000
    normalize_grads: bool = True
    grad_checkpoint_every: int = 0  # >0: recompute activations in chunks of this many steps to save memory
    log_every: int = 100


# The three experiments from the paper differ only in these settings.
EXPERIMENTS = {
    "growing": dict(use_pool=False, damage_n=0),
    "persistent": dict(use_pool=True, damage_n=0),
    "regenerating": dict(use_pool=True, damage_n=3),
}


def experiment_config(name: str, **overrides) -> TrainConfig:
    return replace(TrainConfig(), **{**EXPERIMENTS[name], **overrides})


def per_sample_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per batch element. rendered: (B, 4, H, W), target: (4, H, W)."""
    return ((rendered - target) ** 2).mean(dim=(1, 2, 3))


def rollout(system: DynamicalSystem, state: State, n_steps: int, checkpoint_every: int = 0) -> State:
    """Run ``n_steps`` of the system, optionally with gradient checkpointing."""
    if checkpoint_every <= 0:
        for _ in range(n_steps):
            state = system.step(state)
        return state

    def run_chunk(s, k):
        for _ in range(k):
            s = system.step(s)
        return s

    for start in range(0, n_steps, checkpoint_every):
        k = min(checkpoint_every, n_steps - start)
        # preserve_rng_state (the default) makes the recomputed stochastic
        # updates identical to the original forward pass.
        state = checkpoint(run_chunk, state, k, use_reentrant=False)
    return state


class SamplePool:
    """A fixed-size bank of states that persists across training iterations."""

    def __init__(self, seed_state: State):
        self.state = detach_state(seed_state)
        self.size = batch_size(seed_state)

    def sample(self, n: int) -> tuple[torch.Tensor, State]:
        device = tree_leaves(self.state)[0].device
        idx = torch.randperm(self.size, device=device)[:n]
        return idx, index_state(self.state, idx)

    @torch.no_grad()
    def commit(self, idx: torch.Tensor, state: State) -> None:
        for pool_leaf, new_leaf in zip(tree_leaves(self.state), tree_leaves(state)):
            pool_leaf[idx] = new_leaf.detach()


class Trainer:
    """Holds everything that changes during training, so you can run, stop,
    inspect, tweak, and resume from a notebook: ``trainer.train(1000)`` twice
    is the same as ``trainer.train(2000)`` once."""

    def __init__(self, system: DynamicalSystem, target: torch.Tensor, config: TrainConfig | None = None):
        self.system = system
        self.config = config = config or TrainConfig()
        self.target = target.to(system.device)
        self.optimizer = torch.optim.Adam(system.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
        )
        self.pool = SamplePool(system.seed(config.pool_size)) if config.use_pool else None
        self.step_count = 0
        self.history: list[dict] = []
        # Most recent batch, before and after the rollout (handy for plotting).
        self.last_start: State | None = None
        self.last_end: State | None = None

    # --- one iteration -----------------------------------------------------------

    def make_batch(self) -> tuple[torch.Tensor | None, State]:
        cfg, system = self.config, self.system
        if self.pool is None:
            return None, system.seed(cfg.batch_size)

        idx, x = self.pool.sample(cfg.batch_size)
        with torch.no_grad():
            # Sort worst-first, then: worst -> fresh seed, best few -> damaged.
            order = per_sample_loss(system.render(x), self.target).argsort(descending=True)
            idx, x = idx[order], index_state(x, order)
            n = cfg.batch_size
            parts = [system.seed(1), index_state(x, slice(1, n - cfg.damage_n))]
            if cfg.damage_n > 0:
                h, w = self.target.shape[-2:]
                masks = circle_masks(cfg.damage_n, h, w, device=system.device)
                parts.append(system.damage(index_state(x, slice(n - cfg.damage_n, n)), masks))
            x = cat_states(parts)
        return idx, x

    def train_step(self) -> dict:
        cfg, system = self.config, self.system
        t0 = time.time()

        idx, x0 = self.make_batch()
        n_steps = int(torch.randint(cfg.min_steps, cfg.max_steps + 1, ()).item())
        x = rollout(system, x0, n_steps, cfg.grad_checkpoint_every)
        loss = per_sample_loss(system.render(x), self.target).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for p in system.parameters() if p.grad is not None]
        grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in params)).item()
        if cfg.normalize_grads:
            for p in params:
                p.grad /= p.grad.norm() + 1e-8
        self.optimizer.step()
        self.scheduler.step()

        if self.pool is not None:
            self.pool.commit(idx, x)
        self.last_start, self.last_end = detach_state(x0), detach_state(x)

        self.step_count += 1
        record = dict(
            step=self.step_count,
            loss=loss.item(),
            grad_norm=grad_norm,
            lr=self.optimizer.param_groups[0]["lr"],
            rollout_steps=n_steps,
            seconds=time.time() - t0,
        )
        self.history.append(record)
        return record

    # --- the loop ---------------------------------------------------------------------

    def train(self, n_iters: int, callback=None, callback_every: int | None = None) -> None:
        """Run ``n_iters`` iterations. ``callback(trainer)`` is called every
        ``callback_every`` iterations (default: ``config.log_every``) — use it
        for live plots. Without a callback, progress is printed instead."""
        every = callback_every or self.config.log_every
        for _ in range(n_iters):
            rec = self.train_step()
            if rec["step"] % every == 0:
                if callback is not None:
                    callback(self)
                else:
                    recent = self.history[-every:]
                    avg_loss = sum(r["loss"] for r in recent) / len(recent)
                    avg_sec = sum(r["seconds"] for r in recent) / len(recent)
                    print(f"step {rec['step']:6d} | loss {avg_loss:.5f} | lr {rec['lr']:.1e} | {avg_sec*1000:.0f} ms/it")

    # --- checkpoints ----------------------------------------------------------------

    def save(self, path: str, include_pool: bool = False, **extra) -> None:
        """Save everything needed to resume. ``extra`` keyword args (e.g. eval
        results, a run name) are stored alongside."""
        ckpt = dict(
            system=self.system.state_dict(),
            # Constructor kwargs, if the system records them (NCA does), so
            # comparison code can rebuild the model without knowing its shape.
            system_config=getattr(self.system, "config", None),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            config=asdict(self.config),
            history=self.history,
            step_count=self.step_count,
            target=self.target.cpu(),
            **extra,
        )
        if include_pool and self.pool is not None:
            ckpt["pool"] = self.pool.state
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.system.device, weights_only=False)
        self.system.load_state_dict(ckpt["system"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.history = ckpt["history"]
        self.step_count = ckpt["step_count"]
        if "pool" in ckpt and self.pool is not None:
            self.pool.state = ckpt["pool"]
