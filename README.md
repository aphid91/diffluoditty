# diffluoditty

Training dynamical systems by backpropagating through time. It contains two systems:
- **A PyTorch reproduction of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)**
  (Mordvintsev et al., Distill 2020), as a baseline.
- **A differentiable port of the [Fluoddity-Core](https://github.com/aphid91/Fluoddity-Core)
  particle system**, trained with the same recipe. See [docs/fluoddity.md](docs/fluoddity.md).

| notebook | |
|---|---|
| NCA gecko | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aphid91/diffluoditty/blob/main/notebooks/train_gecko.ipynb) |
| Fluoddity gecko silhouette | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aphid91/diffluoditty/blob/main/notebooks/train_fluoddity.ipynb) |

## Quick start

Open `notebooks/train_gecko.ipynb` in Colab, select a GPU runtime, and run all
cells. It clones this repo, trains the regenerating gecko (8000 iterations),
and shows growth, regeneration, and long-term stability.

Architecture and training settings are Colab forms at the top (presets for
the paper's 8.3k-parameter model, ~800, and ~100 parameters). Each run is
evaluated on a fixed benchmark and checkpointed; the "Compare runs" section at
the bottom plots every run in `checkpoints/` against parameter count.

## Layout

| file | what's in it |
|---|---|
| `diffluoditty/system.py` | `DynamicalSystem`: the interface the training loop uses (`seed`, `step`, `render`, `damage`), plus helpers for pytree-shaped states |
| `diffluoditty/nca.py` | The NCA from the paper: Sobel perception → 1×1-conv MLP → stochastic update → alive masking. Configurable channels / hidden widths, plus `nca_param_count` and `shapes_for_budget` |
| `diffluoditty/fluoddity.py` | Differentiable Fluoddity: bilinear sense / scatter-add splat, Fourier rule, per-particle configs, density render, mass-conserving damage |
| `diffluoditty/fluoddity_rules.py` | Load Fluoddity-Core JSON configs (port of Core's hashes, random rules, cohort mutation) |
| `diffluoditty/diagnostics.py` | `gradient_horizon`: gradient size and seed-to-seed agreement vs. backprop length |
| `configs/` | Fluoddity-Core physics configs, usable as starting points |
| `diffluoditty/train.py` | `TrainConfig`, the three paper experiments, sample pool, rollout (optional gradient checkpointing), `Trainer` |
| `diffluoditty/evaluate.py` | Fixed, seeded benchmark (grow / persist / regenerate loss, died / exploded rate) and `load_runs` for comparing checkpoints |
| `diffluoditty/damage.py` | Damage masks in image space (random circles, halves) |
| `diffluoditty/target.py` | Emoji / image loading as premultiplied RGBA |
| `diffluoditty/viz.py` | Matplotlib: live training plots, frame recording, animations |
| `tests/` | Fast CPU checks, including finite-difference gradient checks (`python -m pytest tests`) |

## The training loop is state-agnostic

`Trainer` never inspects the state directly. Anything that implements

```python
class MySystem(DynamicalSystem):
    def seed(self, n): ...            # n initial states
    def step(self, state): ...        # one differentiable tick
    def render(self, state): ...      # -> (B, 4, H, W) premultiplied RGBA
    def damage(self, state, mask): ...# mask: (B, 1, H, W), 1 = destroy
```

can be trained with the same pool / damage / BPTT recipe. A state can be a
tensor or any nested dict/list/tuple of tensors with the batch dimension first
(e.g. `{"pos": (B, N, 2), "vel": (B, N, 2), "feat": (B, N, K)}` for particles).

For systems that are more expensive per step, set
`TrainConfig.grad_checkpoint_every` to trade compute for memory.

## Paper defaults

16 channels (RGBA + 12 hidden), 128 hidden units, fire rate 0.5, alive
threshold 0.1, 40px target padded to 72×72, batch 8, pool 1024, 64–96 steps
per rollout, Adam lr 2e-3 → 2e-4 at step 2000, per-parameter L2 gradient
normalisation, 3 damaged samples per batch, 8000 iterations.

Note: the paper's emoji URL no longer resolves (the Noto repo moved its PNGs
under `2D/`); `target.py` uses the new location.
