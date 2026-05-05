"""Tests for the Phase 6 algorithmic primitives: GAE, normalizer, KL ctrl, RND."""

from __future__ import annotations

import pytest
import torch

from htbrl.algo.gae import compute_gae, normalize_advantages
from htbrl.algo.kl_ctrl import AdaptiveKLController
from htbrl.algo.normalizer import RunningStats
from htbrl.algo.rnd import RND


# ---- GAE ---------------------------------------------------------------------


def test_gae_lam_zero_recovers_one_step_td():
    """With lambda=0, advantage_t = r_t + gamma*V(s_{t+1}) - V(s_t)."""
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [1.0], [1.5]])
    dones = torch.tensor([[0.0], [0.0], [0.0]])
    last_value = torch.tensor([2.0])
    gamma = 0.9
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=gamma, lam=0.0)
    # adv[0] = 1.0 + 0.9*1.0 - 0.5 = 1.4
    # adv[1] = 2.0 + 0.9*1.5 - 1.0 = 2.35
    # adv[2] = 3.0 + 0.9*2.0 - 1.5 = 3.3
    assert torch.allclose(adv, torch.tensor([[1.4], [2.35], [3.3]]), atol=1e-5)
    # returns = adv + values
    assert torch.allclose(ret, adv + values, atol=1e-5)


def test_gae_done_zeros_bootstrap():
    """When done_t = 1, we should not bootstrap V(s_{t+1})."""
    rewards = torch.tensor([[1.0], [10.0]])
    values = torch.tensor([[0.0], [0.0]])
    dones = torch.tensor([[1.0], [0.0]])  # episode ended at step 0
    last_value = torch.tensor([100.0])  # would dominate without dones mask
    adv, _ = compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)
    # Step 0: not_terminal=0 -> next_v contribution zeroed. delta = 1.0 - 0 = 1.0.
    # GAE_0 = delta + gamma*lam*not_terminal_0 * next_gae = 1.0 + 0 = 1.0
    assert torch.allclose(adv[0], torch.tensor([1.0]), atol=1e-5)


def test_gae_shape_mismatch_raises():
    rewards = torch.zeros(3, 2)
    values = torch.zeros(3, 2)
    dones = torch.zeros(2, 2)  # wrong T
    last_value = torch.zeros(2)
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_gae(rewards, values, dones, last_value)


def test_gae_last_value_shape_mismatch():
    rewards = torch.zeros(3, 2)
    values = torch.zeros(3, 2)
    dones = torch.zeros(3, 2)
    last_value = torch.zeros(3)  # wrong N
    with pytest.raises(ValueError, match="last_value shape"):
        compute_gae(rewards, values, dones, last_value)


def test_normalize_advantages_zero_mean_unit_std():
    a = torch.randn(100)
    a_norm = normalize_advantages(a)
    assert abs(a_norm.mean().item()) < 1e-6
    assert abs(a_norm.std(unbiased=False).item() - 1.0) < 1e-5


# ---- normalizer --------------------------------------------------------------


def test_running_stats_matches_torch_stats():
    rs = RunningStats()
    # Use fp64 input so the Welford fp64 path doesn't drift vs torch.std fp64.
    samples = torch.randn(1000, dtype=torch.float64)
    for chunk in samples.chunk(10):
        rs.update(chunk)
    assert abs(rs.mean.item() - samples.mean().item()) < 1e-12
    # std() with N-1 divisor (matches default torch.std)
    assert abs(rs.std.item() - samples.std(unbiased=True).item()) < 1e-10


def test_running_stats_normalize_zero_when_empty():
    rs = RunningStats()
    x = torch.tensor([1.0, 2.0, 3.0])
    out = rs.normalize(x)
    # No samples seen yet -> normalize returns a copy unchanged
    assert torch.allclose(out, x)


def test_running_stats_state_dict_round_trip():
    rs = RunningStats()
    rs.update(torch.randn(100))
    sd = rs.state_dict()
    rs2 = RunningStats()
    rs2.load_state_dict(sd)
    assert rs2.count == rs.count
    assert abs(rs2.mean.item() - rs.mean.item()) < 1e-12
    assert abs(rs2.m2.item() - rs.m2.item()) < 1e-9


# ---- KL controller -----------------------------------------------------------


def test_kl_controller_raises_on_high_kl():
    c = AdaptiveKLController(init_coef=0.2, target_kl=0.02)
    new = c.update(observed_kl=0.10)  # well above 1.5 * target = 0.03
    assert new > 0.2


def test_kl_controller_lowers_on_low_kl():
    c = AdaptiveKLController(init_coef=0.2, target_kl=0.02)
    new = c.update(observed_kl=0.001)  # below target/1.5 = 0.0133
    assert new < 0.2


def test_kl_controller_stable_in_window():
    c = AdaptiveKLController(init_coef=0.2, target_kl=0.02)
    new = c.update(observed_kl=0.02)  # exactly on target
    assert new == pytest.approx(0.2)


def test_kl_controller_clamps():
    c = AdaptiveKLController(init_coef=0.2, target_kl=0.02, max_coef=1.0)
    for _ in range(50):
        c.update(observed_kl=10.0)  # always too high
    assert c.coef <= 1.0


def test_kl_controller_invalid_args():
    with pytest.raises(ValueError):
        AdaptiveKLController(init_coef=0.0, target_kl=0.02)
    with pytest.raises(ValueError):
        AdaptiveKLController(init_coef=0.2, target_kl=0.0)
    with pytest.raises(ValueError):
        AdaptiveKLController(init_coef=0.2, target_kl=0.02, scale_up=1.0)
    with pytest.raises(ValueError):
        AdaptiveKLController(init_coef=100.0, target_kl=0.02, max_coef=1.0)


# ---- RND ---------------------------------------------------------------------


def test_rnd_target_is_frozen():
    rnd = RND(d_in=16, d_hidden=32, d_out=8)
    target_params_before = [p.detach().clone() for p in rnd.target.parameters()]
    optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
    for _ in range(5):
        emb = torch.randn(4, 16)
        loss = rnd.predictor_loss(emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Target params unchanged, predictor params changed.
    for before, after in zip(target_params_before, rnd.target.parameters()):
        assert torch.allclose(before, after)


def test_rnd_bonus_decreases_with_training():
    """Bonus should drop on inputs the predictor has been trained on."""
    torch.manual_seed(0)
    rnd = RND(d_in=16, d_hidden=32, d_out=8)
    train_emb = torch.randn(64, 16)
    initial_bonus = rnd.compute_bonus(train_emb, update_running_stats=False).mean().item()

    optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-2)
    for _ in range(50):
        loss = rnd.predictor_loss(train_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_bonus = rnd.compute_bonus(train_emb, update_running_stats=False).mean().item()
    assert final_bonus < initial_bonus * 0.5


def test_rnd_bonus_is_clipped():
    rnd = RND(d_in=8, d_hidden=16, d_out=4)
    emb = torch.randn(2, 8) * 1000  # extreme inputs to force big errors
    bonus = rnd.compute_bonus(emb, clip_to=2.0)
    assert (bonus <= 2.0).all()


def test_rnd_predictor_grad_flows():
    rnd = RND(d_in=8, d_hidden=16, d_out=4)
    emb = torch.randn(3, 8)
    loss = rnd.predictor_loss(emb)
    loss.backward()
    for p in rnd.predictor.parameters():
        assert p.grad is not None
    # Target gradients are None (never required grad).
    for p in rnd.target.parameters():
        assert p.grad is None
