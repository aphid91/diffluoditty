"""The interface between the training loop and the thing being trained.

The training loop never looks inside a state. It only needs to be able to:

  * make fresh seed states                    -> ``seed(n)``
  * advance a batch of states by one tick     -> ``step(state)``
  * turn a batch of states into an RGBA image -> ``render(state)``
  * erase part of a batch of states           -> ``damage(state, mask)``

The NCA in ``nca.py`` is one implementation (its state is a plain image-shaped
tensor). A differentiable particle system would be another: its state might be
a dict of per-particle positions / velocities / features, ``render`` would
splat the particles onto a grid, and ``damage`` would kill the particles that
fall inside the mask.

States are "pytrees": a tensor, or a dict / list / tuple of tensors (nested
arbitrarily). Every tensor leaf must have the batch dimension first. That is
all the sample pool needs to store, index, and overwrite states generically.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

State = Any  # a tensor, or a (nested) dict / list / tuple of tensors


class DynamicalSystem(nn.Module):
    """Base class for anything we train by backpropagating through time."""

    def seed(self, n: int) -> State:
        """Return ``n`` copies of the initial state, on the module's device."""
        raise NotImplementedError

    def step(self, state: State) -> State:
        """Advance every state in the batch by one tick. Must be differentiable."""
        raise NotImplementedError

    def render(self, state: State) -> torch.Tensor:
        """Return a (B, 4, H, W) premultiplied RGBA image compared against the target."""
        raise NotImplementedError

    def damage(self, state: State, mask: torch.Tensor) -> State:
        """Destroy the parts of each state that fall inside ``mask``.

        ``mask`` is (B, 1, H, W) in render space, 1.0 where damage should happen.
        """
        raise NotImplementedError

    def forward(self, state: State) -> State:
        return self.step(state)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# --- tiny pytree helpers ------------------------------------------------------
# (torch.utils._pytree does this too, but it is private API and this is 15 lines)

def tree_map(fn: Callable, *trees: State) -> State:
    """Apply ``fn`` leaf-wise across one or more identically-structured states."""
    first = trees[0]
    if isinstance(first, torch.Tensor):
        return fn(*trees)
    if isinstance(first, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(tree_map(fn, *items) for items in zip(*trees))
    raise TypeError(f"Unsupported state leaf type: {type(first)}")


def tree_leaves(tree: State) -> list[torch.Tensor]:
    if isinstance(tree, torch.Tensor):
        return [tree]
    if isinstance(tree, dict):
        return [leaf for v in tree.values() for leaf in tree_leaves(v)]
    if isinstance(tree, (list, tuple)):
        return [leaf for v in tree for leaf in tree_leaves(v)]
    raise TypeError(f"Unsupported state leaf type: {type(tree)}")


def batch_size(state: State) -> int:
    return tree_leaves(state)[0].shape[0]


def index_state(state: State, idx: torch.Tensor) -> State:
    return tree_map(lambda x: x[idx], state)


def cat_states(states: list[State]) -> State:
    return tree_map(lambda *xs: torch.cat(xs, dim=0), *states)


def detach_state(state: State) -> State:
    return tree_map(lambda x: x.detach(), state)
