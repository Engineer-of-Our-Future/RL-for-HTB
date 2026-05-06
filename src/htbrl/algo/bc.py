"""Behavioral cloning loss for the warm-start phase (PLAN.md Phase 5).

Two losses, summed:
- ``tool_loss``: cross-entropy on the tool ID given state
- ``slot_loss``: cross-entropy summed over the slot heads the chosen tool uses

A label-smoothing factor lifts ε mass uniformly over the vocab. We don't add a
ratio loss or KL term here - that's PPO's job in Phase 6.

The actual training loop (``scripts/train_bc.py``) handles batching, optimizer,
checkpointing. This module is just the math.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Numerically-stable CE with optional label smoothing and ignore_index.

    Wraps ``F.cross_entropy`` for clarity / consistent kwarg naming across the
    project. Provided so test code and downstream losses can patch one function
    if we ever need to swap in a custom CE.
    """
    return F.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )


def bc_tool_loss(
    tool_logits: torch.Tensor,
    target_tool_ids: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """CE over tool IDs.

    Args:
        tool_logits:     (B, n_tools)
        target_tool_ids: (B,) int64
    """
    return cross_entropy_with_label_smoothing(
        tool_logits, target_tool_ids, label_smoothing=label_smoothing
    )


def bc_slot_loss(
    slot_logits_list: list[torch.Tensor],
    target_slot_ids_list: list[torch.Tensor],
    label_smoothing: float = 0.05,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Sum of per-slot CEs.

    ``slot_logits_list[i]`` and ``target_slot_ids_list[i]`` correspond to the
    i-th slot of the chosen tool. Slots not present in a given example are
    masked via ``ignore_index`` in the target tensor, so missing-slot positions
    contribute zero loss.

    Returns 0 when both lists are empty.
    """
    if len(slot_logits_list) != len(target_slot_ids_list):
        raise ValueError(
            f"slot logits ({len(slot_logits_list)}) and targets "
            f"({len(target_slot_ids_list)}) length mismatch"
        )
    if not slot_logits_list:
        # Caller wants to skip slot loss entirely. Return a zero tensor that
        # is differentiable-friendly (no grad, but won't break .backward()).
        return torch.zeros((), dtype=torch.float32)
    losses = [
        cross_entropy_with_label_smoothing(
            logits, targets, label_smoothing=label_smoothing, ignore_index=ignore_index
        )
        for logits, targets in zip(slot_logits_list, target_slot_ids_list)
    ]
    return torch.stack(losses).mean()


def bc_loss(
    tool_logits: torch.Tensor,
    target_tool_ids: torch.Tensor,
    slot_logits_list: list[torch.Tensor] | None = None,
    target_slot_ids_list: list[torch.Tensor] | None = None,
    label_smoothing: float = 0.05,
    slot_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined BC loss = tool_CE + slot_weight * slot_CE_mean.

    Returns the scalar loss tensor and a metrics dict for logging.
    """
    tl = bc_tool_loss(tool_logits, target_tool_ids, label_smoothing=label_smoothing)
    if slot_logits_list and target_slot_ids_list:
        sl = bc_slot_loss(
            slot_logits_list, target_slot_ids_list, label_smoothing=label_smoothing
        )
        total = tl + slot_weight * sl
        metrics = {"loss/total": total.item(), "loss/tool": tl.item(), "loss/slot": sl.item()}
    else:
        total = tl
        metrics = {"loss/total": total.item(), "loss/tool": tl.item(), "loss/slot": 0.0}
    return total, metrics


def tool_accuracy(tool_logits: torch.Tensor, target_tool_ids: torch.Tensor) -> float:
    """Top-1 accuracy on tool prediction. Pure metric - no gradient."""
    pred = tool_logits.argmax(dim=-1)
    return (pred == target_tool_ids).float().mean().item()
