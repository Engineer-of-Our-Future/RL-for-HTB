"""Tests for the behavioral-cloning loss."""

from __future__ import annotations

import math

import pytest
import torch

from htbrl.algo.bc import (
    bc_loss,
    bc_slot_loss,
    bc_tool_loss,
    cross_entropy_with_label_smoothing,
    tool_accuracy,
)


def test_tool_loss_basic():
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    targets = torch.tensor([0, 2])
    loss = bc_tool_loss(logits, targets, label_smoothing=0.0)
    # Each row picks the right class with logit gap 1.0.
    # CE = log(sum exp) - logit_target = log(e + 2) - 1 ~ 0.55
    assert 0.5 < loss.item() < 0.6


def test_tool_loss_with_label_smoothing_is_higher():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    plain = bc_tool_loss(logits, targets, label_smoothing=0.0).item()
    smoothed = bc_tool_loss(logits, targets, label_smoothing=0.2).item()
    assert smoothed > plain


def test_slot_loss_empty_returns_zero():
    out = bc_slot_loss([], [])
    assert out.item() == 0.0


def test_slot_loss_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        bc_slot_loss(
            [torch.zeros(2, 3)],
            [torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)],
        )


def test_slot_loss_handles_ignore_index():
    """ignore_index masks out absent slots; surviving slots still produce gradient."""
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # Mark one example as "ignore" for this slot.
    targets = torch.tensor([0, -100])
    loss = bc_slot_loss(
        [logits],
        [targets],
        label_smoothing=0.0,
        ignore_index=-100,
    )
    # With one example ignored, only the first contributes; CE ~ 0.55.
    assert 0.5 < loss.item() < 0.6


def test_combined_bc_loss_matches_components():
    tool_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_tool = torch.tensor([0, 1])
    slot_logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target_slot = torch.tensor([0, 1])
    total, metrics = bc_loss(
        tool_logits,
        target_tool,
        slot_logits_list=[slot_logits],
        target_slot_ids_list=[target_slot],
        slot_weight=1.0,
        label_smoothing=0.0,
    )
    expected = bc_tool_loss(tool_logits, target_tool, 0.0) + bc_slot_loss(
        [slot_logits], [target_slot], 0.0
    )
    assert torch.allclose(total, expected, atol=1e-6)
    assert "loss/total" in metrics and "loss/tool" in metrics and "loss/slot" in metrics


def test_combined_bc_loss_no_slots():
    tool_logits = torch.tensor([[1.0, 0.0]])
    target_tool = torch.tensor([0])
    total, metrics = bc_loss(tool_logits, target_tool)
    assert metrics["loss/slot"] == 0.0


def test_tool_accuracy():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([1, 0, 0])  # 2 correct, 1 wrong
    assert tool_accuracy(logits, targets) == pytest.approx(2 / 3)


def test_cross_entropy_wrapper_matches_torch():
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    a = cross_entropy_with_label_smoothing(logits, targets, label_smoothing=0.1).item()
    b = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.1).item()
    assert math.isclose(a, b, rel_tol=1e-7)
