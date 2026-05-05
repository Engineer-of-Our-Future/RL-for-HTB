"""Compose the trunk + heads into a single ``ActorCriticPolicy``.

The policy is what PPO calls during rollouts and updates. It exposes:
- ``forward(token_ids, matrix_id, attn_mask)`` -> (tool_logits, value, hidden)
  Cheap during rollouts; PPO uses ``hidden`` to lazily query slot heads.
- ``encode(token_ids, attn_mask)`` -> trunk hidden states for inspection / RM.
- ``slot_logits(hidden, tool_id, slot_idx)`` -> per-slot logits.
- ``sample_tool(token_ids, matrix_id)`` -> (tool_id, log_prob, value, hidden)
  for rollout-time sampling. Slot sampling is deferred to the env+algo layer
  because slot schema depends on the tool registry (Phase 1 module).

Hyperparameters in this docstring match the budget in PLAN.md "Hardware
budget" - target ≈ 50-80 M params on the 3060.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import SlotHead, ToolHead, ValueHead
from .init import gpt2_init
from .transformer import TransformerEncoder


@dataclass
class PolicyConfig:
    vocab_size: int
    n_tools: int
    n_matrices: int = 3                 # enterprise / mobile / ics
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.1
    slot_vocab_sizes: tuple[int, ...] = ()  # one entry per registered SlotHead size


class ActorCriticPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
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
        self.tool_emb = nn.Embedding(cfg.n_tools, cfg.d_model)
        self.tool_head = ToolHead(cfg.d_model, cfg.n_tools, cfg.n_matrices)
        self.value_head = ValueHead(cfg.d_model)
        self.slot_heads = nn.ModuleList(
            [SlotHead(cfg.d_model, n) for n in cfg.slot_vocab_sizes]
        )
        gpt2_init(self, n_residual_layers=cfg.n_layers)

    # ----- core forward -------------------------------------------------------

    def encode(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return trunk hidden states (B, T, d_model)."""
        return self.encoder(token_ids, attn_mask)

    def forward(
        self,
        token_ids: torch.Tensor,
        matrix_id: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the trunk + tool head + value head.

        Args:
            token_ids: (B, T) int64.
            matrix_id: (B,) int64.
            attn_mask: optional (B, T) bool keep-mask.

        Returns:
            tool_logits: (B, n_tools)
            value:       (B,)
            hidden:      (B, d_model) at the prediction position (last non-pad)
        """
        h_full = self.encode(token_ids, attn_mask)  # (B, T, D)
        last_h = self._last_position(h_full, attn_mask)
        tool_logits = self.tool_head(last_h, matrix_id)
        value = self.value_head(last_h)
        return tool_logits, value, last_h

    # ----- slot logits --------------------------------------------------------

    def slot_logits(
        self,
        hidden: torch.Tensor,
        tool_id: torch.Tensor,
        slot_head_idx: int,
    ) -> torch.Tensor:
        """Compute slot logits for a given slot-head index conditioned on a tool.

        Args:
            hidden:        (B, d_model) - last_h from forward()
            tool_id:       (B,) int64
            slot_head_idx: which entry in self.slot_heads to use (set by the
                           registry-aware action sampler in algo/)

        Returns:
            (B, slot_vocab_size) logits
        """
        if not (0 <= slot_head_idx < len(self.slot_heads)):
            raise IndexError(f"slot_head_idx {slot_head_idx} out of range")
        tool_emb = self.tool_emb(tool_id)  # (B, d_model)
        return self.slot_heads[slot_head_idx](hidden, tool_emb)

    # ----- rollout-time helpers ----------------------------------------------

    @torch.no_grad()
    def sample_tool(
        self,
        token_ids: torch.Tensor,
        matrix_id: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a tool ID. Returns (tool_id, log_prob, value, hidden)."""
        tool_logits, value, hidden = self.forward(token_ids, matrix_id, attn_mask)
        if temperature != 1.0:
            tool_logits = tool_logits / max(temperature, 1e-6)
        probs = F.softmax(tool_logits, dim=-1)
        tool_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = (
            F.log_softmax(tool_logits, dim=-1)
            .gather(-1, tool_id.unsqueeze(-1))
            .squeeze(-1)
        )
        return tool_id, log_prob, value, hidden

    # ----- helpers ------------------------------------------------------------

    @staticmethod
    def _last_position(
        h_full: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the hidden state at the last non-pad position per batch row.

        If ``attn_mask`` is None, returns ``h_full[:, -1]`` (assumes no padding).
        """
        if attn_mask is None:
            return h_full[:, -1, :]
        # last_idx = max(i where attn_mask[b, i] == True)
        # Build a (B, T) tensor of [0..T-1] * mask, take argmax along T.
        B, T, _ = h_full.shape
        positions = torch.arange(T, device=h_full.device)
        masked_pos = torch.where(attn_mask, positions, torch.full_like(positions, -1))
        last_idx = masked_pos.max(dim=1).values  # (B,)
        # If a row is all-pad (last_idx == -1) fall back to position 0.
        last_idx = last_idx.clamp(min=0)
        return h_full[torch.arange(B, device=h_full.device), last_idx]

    # ----- diagnostic ---------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
