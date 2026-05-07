"""Tests for the RLHF composite reward."""

from __future__ import annotations

import pytest
import torch

from htbrl.algo.rlhf import (
    CompositeRewardConfig,
    alpha_schedule,
    beta_schedule,
    composite_reward,
    reduce_rm_ensemble,
)


def test_alpha_schedule_ramps():
    cfg = CompositeRewardConfig(rm_coef=0.5, rm_anneal_steps=1000)
    assert alpha_schedule(0, cfg) == 0.0
    assert alpha_schedule(500, cfg) == pytest.approx(0.25)
    assert alpha_schedule(1000, cfg) == pytest.approx(0.5)
    assert alpha_schedule(2000, cfg) == pytest.approx(0.5)  # clamped


def test_alpha_zero_anneal_steps_returns_full_coef():
    cfg = CompositeRewardConfig(rm_coef=0.5, rm_anneal_steps=0)
    assert alpha_schedule(0, cfg) == 0.5


def test_beta_schedule_anneals_down():
    cfg = CompositeRewardConfig(rnd_coef_init=0.5, rnd_coef_final=0.05, rnd_anneal_steps=1000)
    assert beta_schedule(0, cfg) == pytest.approx(0.5)
    assert beta_schedule(500, cfg) == pytest.approx(0.275)
    assert beta_schedule(1000, cfg) == pytest.approx(0.05)
    assert beta_schedule(2000, cfg) == pytest.approx(0.05)


def test_reduce_rm_ensemble_min_pessimism():
    a = torch.tensor([1.0, 5.0])
    b = torch.tensor([2.0, 3.0])
    result = reduce_rm_ensemble([a, b], use_min=True)
    assert torch.allclose(result, torch.tensor([1.0, 3.0]))


def test_reduce_rm_ensemble_mean():
    a = torch.tensor([1.0, 5.0])
    b = torch.tensor([3.0, 1.0])
    result = reduce_rm_ensemble([a, b], use_min=False)
    assert torch.allclose(result, torch.tensor([2.0, 3.0]))


def test_reduce_rm_ensemble_empty_raises():
    with pytest.raises(ValueError):
        reduce_rm_ensemble([])


def test_composite_reward_env_only_when_rm_disabled():
    env_r = torch.tensor([1.0, 2.0, 3.0])
    cfg = CompositeRewardConfig(rm_coef=0.5, rm_anneal_steps=1000)
    total, scalars = composite_reward(
        env_reward=env_r, rm_score=None, rnd_bonus=None,
        env_steps_seen=0, cfg=cfg,
    )
    assert torch.allclose(total, env_r)
    assert scalars["alpha"] == 0.0


def test_composite_reward_blends_rm_at_full_alpha():
    env_r = torch.tensor([1.0, 2.0])
    rm = torch.tensor([0.5, -0.5])
    cfg = CompositeRewardConfig(rm_coef=1.0, rm_anneal_steps=0)  # already at full
    total, scalars = composite_reward(
        env_reward=env_r, rm_score=rm, rnd_bonus=None,
        env_steps_seen=0, cfg=cfg,
    )
    expected = env_r + 1.0 * rm
    assert torch.allclose(total, expected)
    assert scalars["alpha"] == 1.0


def test_composite_reward_blends_rnd():
    env_r = torch.tensor([0.0, 0.0])
    rnd = torch.tensor([1.0, 2.0])
    cfg = CompositeRewardConfig(rnd_coef_init=0.5, rnd_coef_final=0.5, rnd_anneal_steps=0)
    total, scalars = composite_reward(
        env_reward=env_r, rm_score=None, rnd_bonus=rnd,
        env_steps_seen=0, cfg=cfg,
    )
    assert torch.allclose(total, 0.5 * rnd)
    assert "rnd_bonus_mean" in scalars
