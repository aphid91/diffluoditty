"""CPU checks for the differentiable Fluoddity port. Run with: python -m pytest tests"""

import glob

import pytest
import torch

from diffluoditty import Trainer, experiment_config, rollout
from diffluoditty.damage import half_mask
from diffluoditty.fluoddity import Fluoddity, sample_bilinear, splat_bilinear
from diffluoditty.fluoddity_rules import load_core_config
from diffluoditty.system import detach_state


def small(**kw):
    args = dict(n_particles=96, canvas_size=24, render_size=12, n_configs=2, seed_radius=0.2, seed_speed_px=0.7)
    return Fluoddity(**{**args, **kw})


def test_splat_conserves_mass_and_is_adjoint_of_sample():
    torch.manual_seed(0)
    pos = torch.rand(2, 50, 2) * 2 - 1
    vals = torch.randn(2, 50, 3)
    grid = splat_bilinear(pos, vals, 16)
    assert torch.allclose(grid.sum(dim=(2, 3)), vals.sum(dim=1), atol=1e-5)  # wrap loses nothing
    g = torch.randn(2, 3, 16, 16)
    # <sample(g, pos), vals> == <g, splat(pos, vals)>: the splat is exactly the transpose of the read
    assert torch.allclose((sample_bilinear(g, pos) * vals).sum(), (g * grid).sum(), atol=1e-4)


def test_all_core_configs_load_and_run():
    for path in sorted(glob.glob("configs/*.json")):
        c = load_core_config(path)
        k = c["cohorts"]
        m = Fluoddity.from_core_config(path, n_particles=8 * k, canvas_size=16, render_size=8)
        s = rollout(m, m.seed(1), 3)
        assert all(torch.isfinite(v).all() for v in s.values()), path
        assert m.render(s).shape == (1, 4, 8, 8)


def test_physics_round_trip():
    m = Fluoddity.from_core_config("configs/9LeafClovers.json", n_particles=8, canvas_size=16)
    ph = m.physics_values()
    assert abs(ph["drag"].item() - 0.125) < 1e-5
    assert abs(ph["sensor_gain"].item() - 4.346) < 1e-4
    assert abs(ph["trail_persistence"].item() - 0.943) < 1e-4
    assert 0.9 < ph["trail_diffusion"].item() <= 1.0  # 1.0 is pulled off the sigmoid's flat end


def test_gradients_match_finite_differences():
    """Autograd vs central differences, in float64, through 6 full steps."""
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        m = small()
        torch.manual_seed(1)
        # Get some trails on the canvas first. Detached, so gradients cover only the 6 steps below.
        x0 = detach_state(rollout(m, m.seed(2), 5))
        target = torch.rand(4, 12, 12)

        def loss_fn():
            return ((m.render(rollout(m, x0, 6)) - target) ** 2).mean()

        loss_fn().backward()
        checks = [(m.rule_amp, (0, 3, 1)), (m.rule_freq, (1, 2, 0)), (m.physics["drag"], (0,)),
                  (m.physics["sensor_angle"], (1,)), (m.physics["trail_persistence"], ())]
        for p, i in checks:
            eps = 1e-6
            with torch.no_grad():
                old = p[i].item()
                p[i] = old + eps
                up = loss_fn().item()
                p[i] = old - eps
                down = loss_fn().item()
                p[i] = old
            fd = (up - down) / (2 * eps)
            assert abs(p.grad[i].item() - fd) <= 1e-4 * abs(fd) + 1e-9, (p.shape, i, p.grad[i].item(), fd)
    finally:
        torch.set_default_dtype(torch.float32)


def test_gradient_checkpointing_matches_on_dict_state():
    torch.manual_seed(0)
    m = small()
    x0 = m.seed(2)

    def grads(every):
        m.zero_grad()
        m.render(rollout(m, x0, 10, every))[:, 3].sum().backward()
        return [p.grad.clone() for p in m.parameters()]

    for a, b in zip(grads(0), grads(4)):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_damage_relocates_particles_and_clears_trails():
    torch.manual_seed(0)
    m = small(n_particles=400, seed_radius=0.9)
    s = rollout(m, m.seed(2), 5)
    mask = half_mask(2, 12, 12, "right")
    d = m.damage(s, mask)
    assert d["pos"].shape == s["pos"].shape  # particle count conserved
    assert (d["pos"][..., 0] > 0.05).sum() == 0  # nobody left in the right half (bar the boundary pixel)
    assert d["canvas"][..., m.canvas_size // 2 + 1:].abs().sum() == 0


def test_trainer_runs_on_fluoddity():
    torch.manual_seed(0)
    m = small()
    target = torch.zeros(4, 12, 12)
    target[3, 3:9, 3:9] = 1.0
    cfg = experiment_config("regenerating", pool_size=8, batch_size=4, damage_n=1,
                            min_steps=4, max_steps=6, grad_checkpoint_every=3)
    t = Trainer(m, target, cfg)
    before = {k: v.detach().clone() for k, v in m.named_parameters()}
    t.train(3, callback=lambda _: None)
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in t.history)
    assert any(not torch.equal(before[k], v) for k, v in m.named_parameters())


def test_trainable_subset():
    m = small(trainable=("rule_amp", "drag"))
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names == {"rule_amp", "physics.drag"}


def test_detach_options_change_only_the_backward_pass():
    flags = dict(detach_sensing=True, detach_heading=True, detach_deposit=True)
    torch.manual_seed(0)
    exact = small()
    torch.manual_seed(0)
    approx = small(**flags)
    x0 = detach_state(rollout(exact, exact.seed(2), 4))
    out = []
    for m in (exact, approx):
        y = rollout(m, x0, 8)
        m.render(y)[:, 3].sum().backward()
        out.append((y, m.rule_amp.grad.clone()))
    (y_exact, g_exact), (y_approx, g_approx) = out
    assert all(torch.equal(y_exact[k], y_approx[k]) for k in y_exact)  # identical simulation
    assert torch.isfinite(g_approx).all() and not torch.allclose(g_exact, g_approx)


def test_sensor_parameters_still_train_when_detached():
    torch.manual_seed(0)
    m = small(detach_sensing=True, detach_heading=True, detach_deposit=True)
    x0 = detach_state(rollout(m, m.seed(2), 4))
    m.render(rollout(m, x0, 6))[:, 3].sum().backward()
    for name in ["sensor_angle", "sensor_distance", "trail_persistence", "trail_diffusion", "drag"]:
        assert m.physics[name].grad.abs().sum() > 0, name
