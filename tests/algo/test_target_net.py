"""Tests for the Polyak target-network helper."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from htbrl.algo.target_net import TargetNetwork


def _net() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 1, bias=False))


def test_target_starts_equal_to_live():
    live = _net()
    tn = TargetNetwork(live, tau=0.01)
    for tp, lp in zip(tn.target.parameters(), live.parameters()):
        assert torch.allclose(tp, lp)


def test_target_params_have_no_grad():
    tn = TargetNetwork(_net(), tau=0.01)
    for p in tn.target.parameters():
        assert not p.requires_grad


def test_invalid_tau_rejected():
    with pytest.raises(ValueError):
        TargetNetwork(_net(), tau=0.0)
    with pytest.raises(ValueError):
        TargetNetwork(_net(), tau=2.0)


def test_update_does_polyak_avg():
    """target' = (1 - tau) * target + tau * live"""
    live = _net()
    # Force distinct parameter values.
    for p in live.parameters():
        nn.init.uniform_(p, -1.0, 1.0)
    tn = TargetNetwork(live, tau=0.5)

    target_before = [p.detach().clone() for p in tn.target.parameters()]
    # Modify live to a known value
    for p in live.parameters():
        p.data.add_(torch.full_like(p, 1.0))

    tn.update()

    for tp, tb_before, lp in zip(tn.target.parameters(), target_before, live.parameters()):
        expected = 0.5 * tb_before + 0.5 * lp
        assert torch.allclose(tp, expected, atol=1e-6)


def test_update_with_explicit_tau_arg():
    live = _net()
    tn = TargetNetwork(live, tau=0.001)
    target_before = [p.detach().clone() for p in tn.target.parameters()]
    # Modify live params
    for p in live.parameters():
        p.data.add_(torch.full_like(p, 10.0))
    tn.update(tau=1.0)  # full hard copy
    for tp, lp in zip(tn.target.parameters(), live.parameters()):
        assert torch.allclose(tp, lp)
    # target_before is irrelevant after hard copy
    _ = target_before


def test_hard_copy_resets_target():
    live = _net()
    tn = TargetNetwork(live, tau=0.001)
    for p in live.parameters():
        p.data.fill_(7.0)
    tn.hard_copy()
    for tp in tn.target.parameters():
        assert torch.allclose(tp, torch.full_like(tp, 7.0))


def test_state_dict_round_trip():
    live = _net()
    tn = TargetNetwork(live, tau=0.123)
    sd = tn.state_dict()
    tn2 = TargetNetwork(_net(), tau=0.001)
    tn2.load_state_dict(sd)
    assert tn2.tau == pytest.approx(0.123)
    for a, b in zip(tn.target.parameters(), tn2.target.parameters()):
        assert torch.allclose(a, b)


def test_target_eval_mode():
    """Target is always in eval mode (no dropout, no BN updates)."""
    live = _net()
    tn = TargetNetwork(live, tau=0.01)
    assert not tn.target.training
