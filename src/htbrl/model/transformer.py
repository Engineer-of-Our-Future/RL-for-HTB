"""Pure-PyTorch transformer encoder with RoPE.

Spec (set in PLAN.md "Phase 3 - Neural architecture"):
- 8 layers, d_model=384, n_heads=8, d_ff=1536, max_seq_len=1024
- Pre-LayerNorm, GELU MLP
- Rotary positional embeddings (RoPE), applied to Q and K only
- Attention via ``F.scaled_dot_product_attention`` so we get FlashAttention-2
  on Ampere (3060 supports it) without depending on a third-party LLM library
- Bidirectional (no causal mask) - the policy reads a fully-known turn-history
  window; we don't need autoregressive generation at the trunk level

Why we wrote it from scratch: project rule "no pretrained models / no LLMs /
pure PyTorch from absolute ground up." Every parameter here is initialized
fresh by ``model/init.py``; nothing is downloaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- RoPE -------------------------------------------------------------------


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for rotary positional embeddings.

    Returns two tensors of shape ``(seq_len, head_dim // 2)`` each.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor of shape ``(B, H, T, D)``.

    The convention here splits the last dim into [first_half, second_half] and
    rotates them as a complex pair (a + ib) -> (a*cos - b*sin) + i(a*sin + b*cos).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # cos/sin: (T, half) -> broadcastable (1, 1, T, half)
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    rotated_first = x1 * cos_b - x2 * sin_b
    rotated_second = x1 * sin_b + x2 * cos_b
    return torch.cat([rotated_first, rotated_second], dim=-1)


# ----- attention --------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # F.scaled_dot_product_attention routes to FlashAttention-2 on Ampere
        # when (q, k, v) are bf16/fp16, contiguous, and head_dim is supported.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        # out: (B, H, T, head_dim) -> (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


# ----- MLP --------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ----- block ------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: x = x + attn(LN(x)); x = x + mlp(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ----- encoder ----------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks with a token embedding and final LayerNorm."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1536,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        # Precompute RoPE tables once. Stored as buffers so they move with .to()
        # and are part of the module's device but NOT saved into the state dict
        # (they're trivially recomputable on load).
        cos, sin = precompute_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) int64 token indices.
            attn_mask: optional (B, T) bool mask; True = keep, False = pad. Pads get
                -inf attention bias so they contribute nothing to other tokens'
                attention output.

        Returns:
            (B, T, d_model) hidden states after final LayerNorm.
        """
        B, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        x = self.emb_dropout(self.tok_emb(token_ids))
        cos = self.rope_cos[:T].to(dtype=x.dtype)
        sin = self.rope_sin[:T].to(dtype=x.dtype)

        sdpa_mask: torch.Tensor | None = None
        if attn_mask is not None:
            if attn_mask.shape != (B, T):
                raise ValueError(
                    f"attn_mask shape {attn_mask.shape} != (B={B}, T={T})"
                )
            # Convert (B, T) bool keep-mask to additive bias of shape (B, 1, 1, T).
            # F.scaled_dot_product_attention broadcasts the H and source-T dims.
            sdpa_mask = torch.zeros(B, 1, 1, T, dtype=x.dtype, device=x.device)
            sdpa_mask.masked_fill_(~attn_mask[:, None, None, :], float("-inf"))

        for block in self.blocks:
            x = block(x, cos, sin, sdpa_mask)
        return self.ln_final(x)
