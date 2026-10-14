# Differentiable Fluoddity: design notes

The plan for training the Fluoddity-Core particle system with backpropagation
through time. It reuses the NCA training stack unchanged: the same pool,
damage, and per-parameter gradient normalisation.

## Decisions

| question | decision |
|---|---|
| Fidelity to Core / exporting trained rules | Not a goal for now. Train well first. |
| Base implementation | Fluoddity-Core (no Web extras: gravity, walls, colour…) |
| What the loss sees | Particle density, rendered as a black silhouette, against the gecko's alpha |
| Symmetry breaking | Seed: a small dense disc at the centre, every particle heading the same way |
| Trainable parameters | Rule (80) + per-config physics (8) per config, + 2 global trail settings |
| Multiple configs | Particles split into K equal index blocks, block k uses config k (like Web's config slots). No cohort mutation in the loop. |
| Damage | Erase trails under the mask; move particles under it onto random surviving particles (count is conserved) |
| Scale | Configurable; default 20k particles on a 192² canvas (≈0.54 particles/px, Core runs at ≈0.57) |

## Port (`diffluoditty/fluoddity.py`)

State `{pos (B,N,2), vel (B,N,2), canvas (B,2,R,R)}`, all batch-first, so the
sample pool stores it like any other state.

Changes from Core, each for differentiability:

- **Splat:** bilinear `scatter_add`, replacing the instanced Gaussian quad. On
  CUDA this is a float atomicAdd, and its backward pass is a gather. The total
  deposit per particle matches the quad's (≈7.07 × vel).
- **Sensing:** a bilinear gather with wrap-around, differentiable with respect
  to both the canvas and the sensor positions.
- **`safenorm`:** smoothed, `v / sqrt(|v|² + ε²)`, so the gradient stays finite
  near zero velocity.
- **Mutation:** applied once, at config-load time (`fluoddity_rules.py`
  reproduces Core's PCG hashes). It can't be differentiated: its seed is a
  hash of the rule's own float bits.
- **Dropped:** hazard resets. The only boundary condition is wrap.
- **Order:** sense → move → deposit → trails. This is the order Core's comments
  call "proper"; Core itself splats first for legacy-config reasons.
- **Units:** `sqrt_world_size = R/1024`, so distances, speeds and deposits are
  constant in pixels at any canvas size.

Physics parameters are stored in "slider units" (value / slider span), so one
learning rate fits all of them. `drag` goes through tanh (|drag| < 1). Trail
persistence and diffusion go through sigmoids, using Core's clamp ranges.

## Checks

- Forward: 9LeafClovers and Cilia, started Core-style (uniform positions, tiny
  velocities), produce their recognisable structures (clovers with ~9 spokes;
  ciliated blobs).
- Gradients match central finite differences to ~1e-6 relative error (float64)
  for every parameter type, through 6 and 20 full steps (`tests/test_fluoddity.py`).
- Gradient checkpointing gives the same gradients.

## Finding: per-particle chaos, and the detached gradient

Measured with `diagnostics.gradient_horizon` (9LeafClovers, 20k particles, 192² canvas, CPU):

| backward pass | from seed | after 50–100 warm-up steps |
|---|---|---|
| exact | grows ~10–20× per step: \|grad\| ≈ 1e14 at T=16, NaN at T=64; batches stop agreeing by T≈3 | ~5× per step: \|grad\| ≈ 7e5 at T=12; agreement < 0 by T≈6 |
| detach sensing + heading | — | agreement 0.86–0.94 to T=8, explodes after ~16 |
| detach sensing + heading + deposit | — | \|grad\| 0.005 → 0.2 and agreement 0.93–0.99 out to T=64 |

The exact gradient follows individual particle paths. Those are chaotic: a
position error changes the trail value read, which changes the force and
strafe, which changes the position error, multiplied by ~5–20 per step. The
loss only sees density, a smooth average over thousands of particles, so its
expected gradient is meaningful even though each path's gradient isn't
(cf. Metz et al. 2021, "Gradients are not all you need").

`Fluoddity(detach_sensing=, detach_heading=, detach_deposit=)` cut those loops
in the backward pass only; the simulation is unchanged. `detach_sensing`
detaches only the particle position the sensors are placed from. The sensor
offsets stay differentiable, so `sensor_angle` and `sensor_distance` still
get gradients. The result is a
*biased* gradient. Gradient still flows through what particles read, the rule,
velocities, the deposits' values, and positions into the render. Truncated
BPTT (exact gradient over 3 steps, after 16 steps without gradient) did not
work: the loss dipped, then rose as the system learned to spread into a band.

Fallback if the biased gradient doesn't train well: evolution strategies. The
model has only ~90 parameters per config, and ES needs forward passes only,
so it is immune to the chaos. It would need per-world parameters (a population
dimension on the parameters).

## Phases

1. ✅ Forward port + import of Core configs.
2. ✅ Gradient correctness tests; `diagnostics.gradient_horizon` measures
   gradient norm and seed-to-seed gradient agreement against backprop horizon.
   The exact gradient is noise beyond ~4 steps; the detached gradient is stable to 64+ (above).
3. Self-recovery: perturb a Core config, then train it back to match the
   original's rendered output. A sanity test that the parameters are learnable.
4. Gecko silhouette, growing → persistent → regenerating
   (`notebooks/train_fluoddity.ipynb`).
5. Speed: `torch.compile`, then custom kernels only if profiling demands.

## Known risks

- **Chaos / long horizons.** Patterns take hundreds of steps to form, and
  particle trajectories are probably chaotic. Watch the horizon diagnostic;
  the pool lets each backprop window stay short.
- **Fourier frequencies** sit inside a sine, which makes them non-convex to
  optimise. If training is jumpy, lower their learning rate or freeze them
  (`trainable=`).
- **Particle conservation.** The silhouette needs density in the right places,
  and excess particles must go somewhere. The saturating alpha
  (`1 - exp(-density / alpha_density)`) makes over-dense regions cheap and
  stray particles expensive.
- **Bilinear sampling** has piecewise-constant gradients with respect to
  position. The trail blur smooths this.
