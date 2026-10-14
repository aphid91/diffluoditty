"""Reading Fluoddity-Core config files into plain numbers.

This is a non-differentiable, run-once-at-load-time port of the bits of
entity_update.glsl that turn a config into the rule(s) particles actually use:

  * ``generate_random_centers(seed)`` — the random rule used when a config's rule is all zero
  * ``mutate_rule(rule, amount, seed)`` — the per-cohort mutation. Its random
    seed is a hash of the rule's own float bits, so it is chaotic in the rule
    parameters and can't be differentiated; we apply it once here and then
    train the mutated rules directly.

The hashes reproduce the GLSL float32/uint32 arithmetic, so imported rules
should match Core's (up to any GPU float quirks).

A rule is 10 Fourier centres, stored as two (10, 4) arrays: ``frequency`` and ``amplitude``.
"""

from __future__ import annotations

import json

import numpy as np

f32 = np.float32
_MASK = 0xFFFFFFFF


def _pcg_hash(seed: int) -> int:
    state = (seed * 747796405 + 2891336453) & _MASK
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & _MASK
    return ((word >> 22) ^ word) & _MASK


def _bits(x) -> int:
    return int(np.array(x, dtype=np.float32).view(np.uint32))


def hash2(x, y) -> np.float32:
    """GLSL ``hash(vec2)``: PCG on the float bit patterns, mapped to [0, 1]."""
    h = _pcg_hash(_bits(x) ^ _pcg_hash(_bits(y)))
    return f32(h) / f32(0xFFFFFFFF)


def hash4(x, y) -> np.ndarray:
    x, y = f32(x), f32(y)
    return np.array([
        hash2(x, y),
        hash2(x * f32(-1) + f32(5), y * f32(-1) + f32(5)),
        hash2(y - f32(100), x - f32(100)),
        hash2(y * f32(-1) + f32(25), x * f32(-1) + f32(25)),
    ], dtype=np.float32)


def generate_random_centers(seed: float) -> tuple[np.ndarray, np.ndarray]:
    seed = f32(seed)
    freq = np.zeros((10, 4), np.float32)
    amp = np.zeros((10, 4), np.float32)
    for i in range(10):
        freq_scale = f32(1) + f32(2) * hash2(seed, f32(i * 8)) ** 2
        for k in range(4):
            freq[i, k] = (hash2(seed, f32(i * 8 + k)) * f32(2) - f32(1)) * freq_scale
            amp[i, k] = hash2(seed, f32(i * 8 + 4 + k)) * f32(2) - f32(1)
    return freq, amp


def mutate_rule(freq: np.ndarray, amp: np.ndarray, amount: float, cohort_seed: float):
    """Core's ``mutate_rule``. ``cohort_seed`` is ``rule_seed + cohort_index``."""
    freq, amp, amount = freq.astype(np.float32).copy(), amp.astype(np.float32).copy(), f32(amount)
    key = freq[4, :2] + amp[7, [1, 0]] + freq[1, 2:]
    seed = hash2(key[0], key[1]) + f32(cohort_seed)
    for i in range(10):
        amp[i] += amount * (f32(-1) + f32(2) * hash4(f32(-0.5) + (f32(-i) + seed), f32(-0.5) + f32(i)))
        freq[i] *= f32(1) + amount * f32(0.5) * (hash2(seed, f32(i)) - f32(0.5))
    return freq, amp


# Physics values that each config (cohort) can have its own copy of.
PER_CONFIG_PHYSICS = [
    "sensor_gain", "sensor_angle", "sensor_distance", "global_force_mult",
    "drag", "strafe_power", "axial_force", "lateral_force",
]
# Values shared by the whole world (they act on the canvas, not on particles).
GLOBAL_PHYSICS = ["trail_persistence", "trail_diffusion"]


def load_core_config(path: str, n_configs: int | None = None, apply_mutation: bool = True) -> dict:
    """Load a Fluoddity-Core JSON config.

    Returns ``dict(freq=(K,10,4), amp=(K,10,4), physics={name: float}, cohorts=int)``
    with one mutated rule per config slot, exactly as Core derives one per cohort.
    ``n_configs`` defaults to the file's cohort count.
    """
    with open(path) as f:
        data = json.load(f)
    physics, settings = data["physics"], data["settings"]
    rule = np.array(data["rule"], dtype=np.float32).reshape(10, 2, 4)
    base_freq, base_amp = rule[:, 0], rule[:, 1]
    rule_seed = f32(settings["rule_seed"])
    n_configs = n_configs or int(settings.get("num_cohorts", 1))

    freqs, amps = [], []
    for c in range(n_configs):
        freq, amp = base_freq, base_amp
        if not freq[0].any() and not amp[5].any():  # Core's "no rule given" convention
            freq, amp = generate_random_centers(rule_seed + f32(c))
        if apply_mutation:
            freq, amp = mutate_rule(freq, amp, physics["mutation_scale"], rule_seed + f32(c))
        freqs.append(freq)
        amps.append(amp)

    return dict(
        freq=np.stack(freqs),
        amp=np.stack(amps),
        physics={k: float(physics[k]) for k in PER_CONFIG_PHYSICS + GLOBAL_PHYSICS},
        cohorts=n_configs,
    )
