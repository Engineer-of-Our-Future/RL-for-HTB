"""Reward model architecture (PLAN.md Phase 8).

A small transformer (4 layers, d_model=256) with a scalar output head, trained
via Bradley-Terry pairwise loss on human preference labels collected through
the FastAPI feedback UI.

This is structurally similar to the policy trunk but:
- Smaller (~12 M params vs ~27 M policy)
- Has its own embedding tables; does NOT share weights with the policy
- Output is a single scalar per trajectory snippet
- Trained alone (no RL), then frozen and used inside RLHF (Phase 9)

The RM is *evaluated* per-trajectory-snippet rather than per-step: we feed it
the same turn-structured token sequence the policy reads, but we output
``r_theta(snippet) ∈ R`` from the final position. PPO sums this over each
rollout's snippets and adds it to the env reward (with annealed α).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from htbrl.model.init import gpt2_init
from htbrl.model.transformer import TransformerEncoder


@dataclass
class RewardModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1


class RewardModel(nn.Module):
    """Trajectory-snippet -> scalar reward."""

    def __init__(self, cfg: RewardModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.score_head = nn.Linear(cfg.d_model, 1)
        gpt2_init(self, n_residual_layers=cfg.n_layers)

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            token_ids: (B, T) int64
            attn_mask: optional (B, T) bool keep-mask
        Returns:
            (B,) scalar score per trajectory snippet
        """
        h = self.encoder(token_ids, attn_mask)  # (B, T, d_model)
        # Take the hidden state at the last unmasked position.
        if attn_mask is not None:
            B, T, _ = h.shape
            positions = torch.arange(T, device=h.device)
            masked_pos = torch.where(attn_mask, positions, torch.full_like(positions, -1))
            last_idx = masked_pos.max(dim=1).values.clamp(min=0)
            last_h = h[torch.arange(B, device=h.device), last_idx]
        else:
            last_h = h[:, -1, :]
        return self.score_head(last_h).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----- Bradley-Terry pairwise loss -------------------------------------------


def bradley_terry_loss(
    score_preferred: torch.Tensor,
    score_other: torch.Tensor,
) -> torch.Tensor:
    """Standard BT pairwise preference loss.

    Args:
        score_preferred: (B,) reward for the trajectory the human preferred
        score_other:     (B,) reward for the other trajectory in the pair

    Returns:
        scalar loss = -mean log sigmoid(score_preferred - score_other)
    """
    if score_preferred.shape != score_other.shape:
        raise ValueError(
            f"shape mismatch: preferred {score_preferred.shape}, other {score_other.shape}"
        )
    return -torch.nn.functional.logsigmoid(score_preferred - score_other).mean()


def pairwise_accuracy(
    score_preferred: torch.Tensor,
    score_other: torch.Tensor,
) -> float:
    """Fraction of pairs where the RM gives a higher score to the preferred snippet."""
    return (score_preferred > score_other).float().mean().item()
