"""A differentiable PyTorch port of the Fluoddity-Core particle system.

https://github.com/aphid91/Fluoddity-Core

State (a dict, batch dimension first):
  pos     (B, N, 2)  particle positions in world space, a torus over [-1, 1)^2
  vel     (B, N, 2)  particle velocities (world units per step)
  canvas  (B, 2, R, R)  the trail field: a smeared record of recent particle velocities

One step, following Core:
  1. sense:   each particle bilinearly samples the canvas at two sensors,
              placed ``sensor_distance`` ahead and rotated +-``sensor_angle`` from its heading
  2. decide:  the two readings, in the particle's own frame (forward / left),
              go through the rule — a 10-term Fourier feature function R^4 -> R^4,
              evaluated twice (normal and mirrored) and combined so the rule
              has no left/right bias — giving a force (-> velocity) and a strafe (-> position)
  3. move:    vel = vel * drag + force;  pos += vel + strafe * strafe_power  (wrapped)
  4. deposit: each particle splats its velocity onto the canvas (bilinear, scatter-add)
  5. trails:  canvas = persistence * blur(canvas) + (1 - persistence) * deposits

Changes from Core, all for differentiability:
  * bilinear splat instead of a tiny instanced Gaussian quad (same total deposit)
  * a smooth ``safenorm`` (no singular gradient at zero velocity)
  * no rule mutation inside the step (it hashes the rule's float bits); mutated
    rules are computed once at load time (see fluoddity_rules.py) and trained directly
  * no hazard resets, one fixed boundary condition (wrap)

Multiple configs: particles are split into ``n_configs`` equal, contiguous
index blocks; block k uses rule k and physics k. Seed positions are random, so
this is the same as assigning configs at random, but needs no per-particle
gather of parameters. Trail persistence and diffusion act on the shared canvas,
so they are global.

Units follow Core: ``sqrt_world_size = R / 1024`` keeps sensor distances,
speeds and deposit strengths constant *in pixels* whatever the canvas size.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .fluoddity_rules import GLOBAL_PHYSICS, PER_CONFIG_PHYSICS, generate_random_centers, load_core_config
from .system import DynamicalSystem

# How each physics value is stored as a trainable number. Parameters live in
# "slider units" (value / span), so one learning rate suits them all, and the
# few that must stay in a range are squashed into it.
#   name: (kind, span)   kind: "linear" -> value = span * p
#                              "tanh"   -> value = span * tanh(p)
#                              "sigmoid"-> value = lo + (hi - lo) * sigmoid(p), span = (lo, hi)
PHYSICS_PARAM = {
    "sensor_gain": ("linear", 10.0),
    "sensor_angle": ("linear", 1.0),
    "sensor_distance": ("linear", 3.0),
    "global_force_mult": ("linear", 2.0),
    "drag": ("tanh", 1.0),  # |drag| < 1, or velocities blow up
    "strafe_power": ("linear", 0.5),
    "axial_force": ("linear", 1.0),
    "lateral_force": ("linear", 1.0),
    "trail_persistence": ("sigmoid", (0.0, 0.999)),  # Core's clamp range
    "trail_diffusion": ("sigmoid", (0.001, 1.0)),
}

# Core's brush: each particle blends vel * k^2 over a ~1.5px Gaussian quad.
# Its total is the closed form from Fluoddity-Web's `deposit` (dot_px = 1.536).
_SIGMA, _RADIUS, _DOT_PX = 0.163, 0.5, 1.536
DEPOSIT_AMOUNT = (1 - math.exp(-_RADIUS**2 / _SIGMA**2)) / (4 * math.pi * _SIGMA**2) * _DOT_PX**2  # ~7.07


def _to_raw(name: str, value: float) -> float:
    kind, span = PHYSICS_PARAM[name]
    if kind == "linear":
        return value / span
    if kind == "tanh":
        return math.atanh(max(-0.98, min(0.98, value / span)))
    lo, hi = span
    frac = max(0.02, min(0.98, (value - lo) / (hi - lo)))  # keep off the flat ends of the sigmoid
    return math.log(frac / (1 - frac))


def _from_raw(name: str, raw: torch.Tensor) -> torch.Tensor:
    kind, span = PHYSICS_PARAM[name]
    if kind == "linear":
        return span * raw
    if kind == "tanh":
        return span * torch.tanh(raw)
    lo, hi = span
    return lo + (hi - lo) * torch.sigmoid(raw)


# --- grid helpers ---------------------------------------------------------------

def bilinear_taps(pos: torch.Tensor, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The 4 pixels around each position on a res x res torus, and their weights.

    pos (B, N, 2) in [-1, 1) (x -> column, y -> row). Returns flat pixel
    indices (B, N, 4) and bilinear weights (B, N, 4). Pixel centres sit at
    (i + 0.5) / res in uv, as in a GL texture.
    """
    p = (pos + 1.0) * (res / 2.0) - 0.5
    p0 = torch.floor(p)
    frac = p - p0  # gradient flows through here; floor has none
    i0 = p0.long()
    x0, y0 = i0[..., 0] % res, i0[..., 1] % res
    x1, y1 = (x0 + 1) % res, (y0 + 1) % res
    fx, fy = frac[..., 0], frac[..., 1]
    idx = torch.stack([y0 * res + x0, y0 * res + x1, y1 * res + x0, y1 * res + x1], dim=-1)
    w = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], dim=-1)
    return idx, w


def sample_bilinear(grid: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Read a (B, C, R, R) torus grid at positions (B, N, 2) -> (B, N, C)."""
    b, c, res, _ = grid.shape
    idx, w = bilinear_taps(pos, res)
    n = pos.shape[1]
    flat = grid.reshape(b, c, res * res)
    taps = flat.gather(2, idx.reshape(b, 1, n * 4).expand(b, c, n * 4)).reshape(b, c, n, 4)
    return (taps * w.unsqueeze(1)).sum(-1).transpose(1, 2)


def splat_bilinear(pos: torch.Tensor, values: torch.Tensor, res: int) -> torch.Tensor:
    """Add per-particle values (B, N, C) onto a (B, C, res, res) torus grid.

    This is the PyTorch form of an atomic-add splat: on CUDA, scatter_add is a
    float atomicAdd, and its backward pass is a gather.
    """
    b, n, c = values.shape
    idx, w = bilinear_taps(pos, res)
    src = (values.unsqueeze(2) * w.unsqueeze(-1)).permute(0, 3, 1, 2).reshape(b, c, n * 4)
    out = torch.zeros(b, c, res * res, device=values.device, dtype=values.dtype)
    out = out.scatter_add(2, idx.reshape(b, 1, n * 4).expand(b, c, n * 4), src)
    return out.reshape(b, c, res, res)


def safenorm(v: torch.Tensor, eps: float) -> torch.Tensor:
    """v / |v|, smoothed so the gradient stays finite near v = 0."""
    return v * torch.rsqrt((v * v).sum(-1, keepdim=True) + eps * eps)


def rotate(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Core's pR: rotate 2-vectors p by angle a (clockwise in x/y)."""
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * p[..., 0] + s * p[..., 1], c * p[..., 1] - s * p[..., 0]], dim=-1)


def y_reflect(p: torch.Tensor) -> torch.Tensor:
    return torch.stack([p[..., 0], -p[..., 1]], dim=-1)


# --- the system ------------------------------------------------------------------------------

class Fluoddity(DynamicalSystem):
    def __init__(
        self,
        n_particles: int = 20_000,
        canvas_size: int = 192,
        render_size: int = 72,
        n_configs: int = 1,
        freq=None,  # (K, 10, 4) initial rule frequencies; default: Core's random rule for rule_seed
        amp=None,  # (K, 10, 4) initial rule amplitudes
        physics: dict | None = None,  # initial physics values; default: 9LeafClovers'
        rule_seed: float = 0.5,
        seed_radius: float = 0.05,  # world units; the initial disc of particles
        seed_speed_px: float = 0.5,  # pixels per step, all particles heading the same way
        seed_heading: float = -math.pi / 2,  # radians; -pi/2 = up on screen (rows grow with +y)
        alpha_density: float | None = None,  # particles per render pixel giving alpha = 1 - 1/e
        trainable: tuple[str, ...] | None = None,  # parameter names to train; default all
        detach_sensing: bool = False,  # backward pass treats the particle position its sensors are placed from as constant
        detach_heading: bool = False,  # backward pass treats particle headings as constant
        detach_deposit: bool = False,  # backward pass treats WHERE particles deposit trails as constant
    ):
        super().__init__()
        assert n_particles % n_configs == 0, "n_particles must be divisible by n_configs"
        self.n_particles, self.canvas_size, self.render_size = n_particles, canvas_size, render_size
        self.n_configs = n_configs
        self.detach_sensing, self.detach_heading, self.detach_deposit = detach_sensing, detach_heading, detach_deposit
        self.seed_radius, self.seed_speed_px, self.seed_heading = seed_radius, seed_speed_px, seed_heading
        self.sqrt_world_size = canvas_size / 1024.0
        self.px = 2.0 / canvas_size  # one canvas pixel in world units
        # Default: a uniform spread over a quarter of the render area would give alpha ~0.63
        self.alpha_density = alpha_density or n_particles / (0.25 * render_size**2)

        if freq is None or amp is None:
            rules = [generate_random_centers(rule_seed + k) for k in range(n_configs)]
            freq = [r[0] for r in rules]
            amp = [r[1] for r in rules]
        dtype = torch.get_default_dtype()
        freq = torch.as_tensor(np.asarray(freq), dtype=dtype).reshape(n_configs, 10, 4)
        amp = torch.as_tensor(np.asarray(amp), dtype=dtype).reshape(n_configs, 10, 4)
        self.rule_freq = nn.Parameter(freq.clone())
        self.rule_amp = nn.Parameter(amp.clone())

        defaults = dict(  # 9LeafClovers
            sensor_gain=4.346, sensor_angle=-0.141, sensor_distance=1.013, global_force_mult=0.94,
            drag=0.125, strafe_power=0.169, axial_force=0.022, lateral_force=-0.186,
            trail_persistence=0.943, trail_diffusion=1.0,
        )
        physics = {**defaults, **(physics or {})}
        # One small tensor per physics value, so per-parameter gradient
        # normalisation treats each knob separately.
        self.physics = nn.ParameterDict()
        for name in PER_CONFIG_PHYSICS:
            v = torch.as_tensor(physics[name], dtype=dtype).expand(n_configs)
            self.physics[name] = nn.Parameter(torch.tensor([_to_raw(name, float(x)) for x in v]))
        for name in GLOBAL_PHYSICS:
            self.physics[name] = nn.Parameter(torch.tensor(_to_raw(name, float(physics[name]))))

        if trainable is not None:
            for name, p in self.named_parameters():
                p.requires_grad_(name.split(".")[-1] in trainable or name in trainable)

        self.config = dict(
            n_particles=n_particles, canvas_size=canvas_size, render_size=render_size, n_configs=n_configs,
            seed_radius=seed_radius, seed_speed_px=seed_speed_px, seed_heading=seed_heading,
            alpha_density=self.alpha_density, detach_sensing=detach_sensing, detach_heading=detach_heading,
            detach_deposit=detach_deposit,
        )

    @classmethod
    def from_core_config(cls, path: str, n_configs: int | None = None, apply_mutation: bool = True, **kwargs):
        """Start from a Fluoddity-Core JSON config: one (mutated) rule per cohort."""
        c = load_core_config(path, n_configs, apply_mutation)
        return cls(n_configs=c["cohorts"], freq=c["freq"], amp=c["amp"], physics=c["physics"], **kwargs)

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "Fluoddity":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = cls(**ckpt["system_config"]).to(device or "cpu")
        model.load_state_dict(ckpt["system"])
        return model

    def physics_values(self) -> dict[str, torch.Tensor]:
        """Physics in Core's units: (K,) per-config tensors, scalars for global ones."""
        return {name: _from_raw(name, raw) for name, raw in self.physics.items()}

    # --- DynamicalSystem interface --------------------------------------------------

    def seed(self, n: int) -> dict:
        """All particles in a small disc at the centre, all heading the same way."""
        dev, N = self.device, self.n_particles
        r = self.seed_radius * torch.sqrt(torch.rand(n, N, device=dev))
        theta = 2 * math.pi * torch.rand(n, N, device=dev)
        pos = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        heading = torch.tensor([math.cos(self.seed_heading), math.sin(self.seed_heading)], device=dev)
        vel = (self.seed_speed_px * self.px * heading).expand(n, N, 2).clone()
        canvas = torch.zeros(n, 2, self.canvas_size, self.canvas_size, device=dev)
        return dict(pos=pos, vel=vel, canvas=canvas)

    def step(self, state: dict) -> dict:
        pos, vel, canvas = state["pos"], state["vel"], state["canvas"]
        B, N, K = pos.shape[0], self.n_particles, self.n_configs
        M = N // K
        S = self.sqrt_world_size
        ph = self.physics_values()

        def per_config(v):  # (K,) -> broadcastable against (B, K, M, ...)
            return v.view(1, K, 1, 1)

        # 1. sense
        heading = safenorm(vel.detach() if self.detach_heading else vel, eps=1e-3 * self.px)  # (B, N, 2)
        h = heading.view(B, K, M, 2)
        sample_dist = per_config(ph["sensor_distance"]) * (0.005 / S)
        angle = ph["sensor_angle"].view(1, K, 1) * math.pi  # rotate() works on (B, K, M) components
        left_off = rotate(h * sample_dist, angle)
        right_off = rotate(h * sample_dist, -angle)
        p = pos.view(B, K, M, 2)
        # detach_sensing cuts the particle's own position out of the backward
        # pass (the chaotic feedback loop) but keeps the sensor offsets, so
        # sensor_angle / sensor_distance still get gradients.
        p_sense = p.detach() if self.detach_sensing else p
        sensor_pos = torch.cat([p_sense + left_off, p_sense + right_off], dim=2)  # (B, K, 2M, 2): [left | right]
        taps = sample_bilinear(canvas, sensor_pos.reshape(B, 2 * N, 2)).view(B, K, 2 * M, 2)
        taps = taps * per_config(ph["sensor_gain"]) * (S * 38.855)
        L, R = taps[:, :, :M], taps[:, :, M:]

        # 2. decide
        force, strafe = self.behaviour(L, R, h, ph)
        gfm = per_config(ph["global_force_mult"])
        force = force * gfm / (S * 400.0)
        strafe = strafe * gfm / (S * 20.0)

        # 3. move
        v = vel.view(B, K, M, 2) * per_config(ph["drag"]) + force
        p = p + v + strafe * per_config(ph["strafe_power"])
        p = torch.remainder(p + 1.0, 2.0) - 1.0
        vel, pos = v.reshape(B, N, 2), p.reshape(B, N, 2)

        # 4. deposit, 5. trails
        deposits = splat_bilinear(pos.detach() if self.detach_deposit else pos, vel * DEPOSIT_AMOUNT, self.canvas_size)
        P = ph["trail_persistence"]
        canvas = P * self.blur(canvas, ph["trail_diffusion"]) + (1 - P) * deposits
        return dict(pos=pos, vel=vel, canvas=canvas)

    def render(self, state: dict) -> torch.Tensor:
        """Particle density as a black silhouette: RGB = 0, alpha = 1 - exp(-density / alpha_density)."""
        pos = state["pos"]
        ones = torch.ones(*pos.shape[:2], 1, device=pos.device, dtype=pos.dtype)
        density = splat_bilinear(pos, ones, self.render_size)
        alpha = 1.0 - torch.exp(-density / self.alpha_density)
        return torch.cat([torch.zeros_like(alpha).expand(-1, 3, -1, -1), alpha], dim=1)

    @torch.no_grad()
    def damage(self, state: dict, mask: torch.Tensor) -> dict:
        """Erase trails under the mask, and move the particles under it onto
        randomly chosen surviving particles (particle count is conserved, so
        the pattern can regrow)."""
        pos, vel = state["pos"].clone(), state["vel"].clone()
        res = self.canvas_size
        canvas_mask = F.interpolate(mask, size=(res, res), mode="nearest")
        canvas = state["canvas"] * (1.0 - canvas_mask)

        hit = sample_bilinear(canvas_mask, pos)[..., 0] > 0.5  # (B, N)
        for b in range(pos.shape[0]):
            dead, alive = hit[b].nonzero().squeeze(1), (~hit[b]).nonzero().squeeze(1)
            if len(dead) == 0 or len(alive) == 0:
                continue
            donor = alive[torch.randint(len(alive), (len(dead),), device=pos.device)]
            jitter = (torch.rand(len(dead), 2, device=pos.device) - 0.5) * self.px
            pos[b, dead] = torch.remainder(pos[b, donor] + jitter + 1.0, 2.0) - 1.0
            vel[b, dead] = vel[b, donor]
        return dict(pos=pos, vel=vel, canvas=canvas)

    # --- pieces of the step ------------------------------------------------------------------

    def fourier(self, signals: torch.Tensor) -> torch.Tensor:
        """The rule: (B, K, M, 4) -> (B, K, M, 4), one rule per config."""
        freq, amp = self.rule_freq, self.rule_amp  # (K, 10, 4)
        K = freq.shape[0]
        i = torch.arange(10, device=freq.device, dtype=freq.dtype)
        phase = torch.einsum("bkmj,kcj->bkmc", signals, freq)  # (B, K, M, 10)
        offset = (2 * i * 0.6283 + amp[..., 3] * 3.14159).view(1, K, 1, 10)
        basis = torch.stack([
            torch.sin(phase + offset),
            torch.cos(phase + offset * 0.7),
            torch.sin(phase * 2.0 + offset * 1.3),
            torch.cos(phase * 2.0 + offset * 0.5),
        ], dim=-1)  # (B, K, M, 10, 4)
        return torch.einsum("bkmcj,kcj->bkmj", basis, amp)

    def behaviour(self, L, R, forward, ph):
        """Core's calculate_entity_behavior. L, R, forward: (B, K, M, 2)."""
        left = torch.stack([forward[..., 1], -forward[..., 0]], dim=-1)

        def local(v):
            return torch.stack([(v * forward).sum(-1), (v * left).sum(-1)], dim=-1)

        L, R = local(L), local(R)
        M = L.shape[2]
        # Normal and mirrored evaluations in one batch.
        signals = torch.cat([torch.cat([L, R], -1), torch.cat([y_reflect(R), y_reflect(L)], -1)], dim=2)
        out = self.fourier(signals)
        base, mirror = out[:, :, :M], out[:, :, M:]
        force = base[..., :2] + y_reflect(mirror[..., :2])
        strafe = base[..., 2:] + y_reflect(mirror[..., 2:])

        K = self.n_configs
        axial = ph["axial_force"].view(1, K, 1, 1)
        lateral = ph["lateral_force"].view(1, K, 1, 1)
        force = forward * force[..., :1] * axial + left * force[..., 1:] * lateral
        strafe = forward * strafe[..., :1] * axial + left * strafe[..., 1:] * lateral
        return force, strafe

    @staticmethod
    def blur(canvas: torch.Tensor, trail_diffusion: torch.Tensor) -> torch.Tensor:
        """Core's 4-neighbour diffusion on the torus: (K*c + n + s + e + w) / (4 + K)."""
        d = trail_diffusion**2
        k = 4.0 / (torch.pow(5.0, d) - 1.0)
        nbrs = (
            torch.roll(canvas, 1, dims=2) + torch.roll(canvas, -1, dims=2)
            + torch.roll(canvas, 1, dims=3) + torch.roll(canvas, -1, dims=3)
        )
        return (canvas * k + nbrs) / (4.0 + k)
