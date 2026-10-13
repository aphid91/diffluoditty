"""Fast CPU checks. Run with: python -m pytest tests"""

import torch

from diffluoditty import NCA, Trainer, circle_masks, experiment_config, rollout
from diffluoditty.system import cat_states, index_state


def disc_target(size=24, pad=6):
    """A red disc: a stand-in for the emoji that doesn't need network access."""
    h = size + 2 * pad
    y, x = torch.meshgrid(torch.arange(h), torch.arange(h), indexing="ij")
    inside = (((y - h / 2 + 0.5) ** 2 + (x - h / 2 + 0.5) ** 2) < (size / 2) ** 2).float()
    return torch.stack([inside, 0 * inside, 0 * inside, inside])


def test_untrained_nca_preserves_seed():
    torch.manual_seed(0)
    nca = NCA(size=(16, 16))
    x = nca.seed(2)
    assert x.shape == (2, 16, 16, 16)
    assert torch.equal(rollout(nca, x, 10), x)  # zero-initialised last layer -> no-op


def test_dead_cells_stay_dead():
    torch.manual_seed(0)
    nca = NCA(size=(16, 16))
    torch.nn.init.normal_(nca.update[-1].weight, std=0.1)
    x = rollout(nca, nca.seed(1), 5)
    # Nothing can appear further than one cell per step from the seed.
    assert x[..., :2, :].abs().sum() == 0 and x[..., :, :2].abs().sum() == 0


def test_gradient_checkpointing_matches():
    torch.manual_seed(0)
    nca = NCA(size=(16, 16))
    torch.nn.init.normal_(nca.update[-1].weight, std=0.05)
    x0 = nca.seed(2)

    def grads(every):
        nca.zero_grad()
        torch.manual_seed(1)
        rollout(nca, x0, 12, every)[:, :4].square().mean().backward()
        return [p.grad.clone() for p in nca.parameters()]

    for a, b in zip(grads(0), grads(5)):
        assert torch.allclose(a, b, atol=1e-6)


def test_pytree_helpers_on_dict_state():
    s = {"pos": torch.arange(6.0).view(3, 2), "feat": (torch.arange(3.0),)}
    sub = index_state(s, torch.tensor([2, 0]))
    assert sub["pos"].tolist() == [[4, 5], [0, 1]] and sub["feat"][0].tolist() == [2, 0]
    both = cat_states([sub, sub])
    assert both["pos"].shape == (4, 2)


def test_circle_masks():
    torch.manual_seed(0)
    m = circle_masks(5, 32, 32)
    assert m.shape == (5, 1, 32, 32)
    assert ((m == 0) | (m == 1)).all() and (m.sum(dim=(1, 2, 3)) > 0).all()


def test_regenerating_training_reduces_loss():
    torch.manual_seed(0)
    target = disc_target()
    nca = NCA(size=tuple(target.shape[-2:]))
    cfg = experiment_config("regenerating", pool_size=32, min_steps=16, max_steps=24)
    trainer = Trainer(nca, target, cfg)
    trainer.train(100, callback=lambda t: None)
    first = sum(r["loss"] for r in trainer.history[:10]) / 10
    last = sum(r["loss"] for r in trainer.history[-10:]) / 10
    assert last < 0.75 * first, (first, last)
    # Pool got written back: some entries are no longer the bare seed.
    assert (trainer.pool.state[:, 3].sum(dim=(1, 2)) > 1).any()


def test_save_and_load(tmp_path):
    torch.manual_seed(0)
    target = disc_target()
    nca = NCA(size=tuple(target.shape[-2:]))
    cfg = experiment_config("growing", min_steps=4, max_steps=6)
    trainer = Trainer(nca, target, cfg)
    trainer.train(3, callback=lambda t: None)
    trainer.save(tmp_path / "ckpt.pt")

    other = Trainer(NCA(size=tuple(target.shape[-2:])), target, cfg)
    other.load(tmp_path / "ckpt.pt")
    assert other.step_count == 3
    for a, b in zip(nca.parameters(), other.system.parameters()):
        assert torch.equal(a, b)
