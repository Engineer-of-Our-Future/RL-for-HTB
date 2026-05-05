"""Actor-critic heads on top of the transformer trunk.

Three heads:
- ``ToolHead``: matrix-aware categorical over tool IDs. Adds a learned per-matrix
  bias so the same trunk can serve Enterprise / Mobile / ICS without retraining.
- ``ValueHead``: scalar V(s) for the critic.
- ``SlotHead``: generic per-slot categorical, conditioned on the chosen tool's
  embedding. The policy holds one SlotHead per *distinct slot vocab size*, not
  one per (tool, slot) pair - the tool embedding is what specializes it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ToolHead(nn.Module):
    """Categorical over the tool vocabulary, biased by the matrix selector.

    output_logits[b, t] = W h[b] + matrix_bias[matrix_id[b], t]
    """

    def __init__(self, d_model: int, n_tools: int, n_matrices: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_tools)
        self.matrix_bias = nn.Embedding(n_matrices, n_tools)
        # Init the matrix bias to zero so untrained policies start matrix-agnostic.
        nn.init.zeros_(self.matrix_bias.weight)

    def forward(self, hidden: torch.Tensor, matrix_id: torch.Tensor) -> torch.Tensor:
        """
        hidden:    (B, d_model) trunk hidden state at the prediction position
        matrix_id: (B,) int64 in [0, n_matrices)
        returns:   (B, n_tools) logits
        """
        return self.proj(hidden) + self.matrix_bias(matrix_id)


class ValueHead(nn.Module):
    """Scalar V(s)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, d_model) -> (B,) scalar."""
        return self.proj(hidden).squeeze(-1)


class SlotHead(nn.Module):
    """Generic categorical slot head conditioned on the chosen tool.

    Concatenates the trunk hidden state with the tool embedding before
    projecting to ``slot_vocab_size`` logits. The policy reuses one SlotHead per
    distinct slot vocab size (e.g. one for ``port_list`` slots, one for the 4
    different ``wordlist_id`` enums) rather than one per (tool, slot) pair.
    """

    def __init__(self, d_model: int, slot_vocab_size: int) -> None:
        super().__init__()
        self.slot_vocab_size = slot_vocab_size
        self.proj = nn.Linear(d_model * 2, slot_vocab_size)

    def forward(self, hidden: torch.Tensor, tool_emb: torch.Tensor) -> torch.Tensor:
        """
        hidden:   (B, d_model)
        tool_emb: (B, d_model) - row from the policy's tool embedding table for the chosen tool
        returns:  (B, slot_vocab_size) logits
        """
        return self.proj(torch.cat([hidden, tool_emb], dim=-1))
