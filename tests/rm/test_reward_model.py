"""Tests for the reward-model architecture and Bradley-Terry loss."""

from __future__ import annotations

import pytest
import torch

from htbrl.rm.model import (
    RewardModel,
    RewardModelConfig,
    bradley_terry_loss,
    pairwise_accuracy,
)


def _tiny_cfg() -> RewardModelConfig:
    return RewardModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


def test_reward_model_forward_shapes():
    rm = RewardModel(_tiny_cfg())
    rm.eval()
    ids = torch.randint(0, 64, (3, 16))
    score = rm(ids)
    assert score.shape == (3,)


def test_reward_model_attn_mask_picks_last_unmasked():
    rm = RewardModel(_tiny_cfg())
    rm.eval()
    ids = torch.randint(0, 64, (1, 16))
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :8] = True
    s_masked = rm(ids, mask)
    s_truncated = rm(ids[:, :8])
    assert torch.allclose(s_masked, s_truncated, atol=1e-4)


def test_reward_model_param_count_bounded():
    """Per PLAN.md, RM is meaningfully smaller than the policy (~12 M target)."""
    cfg = RewardModelConfig(
        vocab_size=32_768,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_len=1024,
    )
    rm = RewardModel(cfg)
    n = rm.n_parameters()
    # Allow generous envelope around the planned ~12M figure.
    assert 8_000_000 <= n <= 25_000_000


def test_reward_model_gradients_flow():
    rm = RewardModel(_tiny_cfg())
    rm.train()
    ids = torch.randint(0, 64, (4, 16))
    score = rm(ids)
    loss = score.mean()
    loss.backward()
    for name, p in rm.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad on {name}"
            assert torch.isfinite(p.grad).all()


# ---- Bradley-Terry loss ------------------------------------------------------


def test_bt_loss_shape_mismatch_raises():
    a = torch.zeros(4)
    b = torch.zeros(3)
    with pytest.raises(ValueError, match="shape mismatch"):
        bradley_terry_loss(a, b)


def test_bt_loss_zero_when_preferred_dominates():
    """If r(preferred) >> r(other), loss approaches 0."""
    score_p = torch.tensor([100.0, 100.0])
    score_o = torch.tensor([-100.0, -100.0])
    loss = bradley_terry_loss(score_p, score_o)
    assert loss.item() < 1e-30


def test_bt_loss_high_when_preferred_loses():
    score_p = torch.tensor([0.0, 0.0])
    score_o = torch.tensor([10.0, 10.0])
    loss = bradley_terry_loss(score_p, score_o)
    # -log sigmoid(-10) ≈ 10
    assert loss.item() > 5.0


def test_bt_loss_at_tie_is_log2():
    """When scores are equal, sigmoid(0) = 0.5, -log(0.5) = log(2)."""
    s = torch.zeros(8)
    loss = bradley_terry_loss(s, s)
    import math
    assert abs(loss.item() - math.log(2)) < 1e-6


def test_pairwise_accuracy():
    # 1.0>0.5 ✓, 2.0<2.5 ✗, 3.0>1.0 ✓, 4.0<5.0 ✗ -> 2/4 = 0.5
    p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    o = torch.tensor([0.5, 2.5, 1.0, 5.0])
    assert pairwise_accuracy(p, o) == pytest.approx(0.5)
    # And a clean 3/4 case for sanity:
    p2 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    o2 = torch.tensor([0.5, 1.5, 1.0, 5.0])
    assert pairwise_accuracy(p2, o2) == pytest.approx(0.75)


def test_bt_loss_trains_a_model_to_separate_pairs():
    """Sanity check: gradient descent on BT loss actually learns the preference."""
    torch.manual_seed(0)
    rm = RewardModel(_tiny_cfg())
    rm.train()
    optim = torch.optim.Adam(rm.parameters(), lr=1e-2)

    # Two distinct token patterns; we declare pattern A always preferred.
    pref_ids = torch.full((8, 8), 5)  # filled with token 5
    other_ids = torch.full((8, 8), 7)  # filled with token 7

    initial_acc = pairwise_accuracy(rm(pref_ids), rm(other_ids))
    for _ in range(50):
        loss = bradley_terry_loss(rm(pref_ids), rm(other_ids))
        optim.zero_grad()
        loss.backward()
        optim.step()
    final_acc = pairwise_accuracy(rm(pref_ids), rm(other_ids))
    assert final_acc > 0.99, f"BT loss failed to separate trivially-distinct pairs: {final_acc}"
