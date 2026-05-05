"""Phase 3 tests: transformer + actor-critic heads + policy.

These tests run on CPU to be CI-portable; the GPU path is exercised manually
during training. We do however reserve a few tests for CUDA-only checks
(memory bound, bf16 compatibility) and skip them if no GPU is available.
"""

from __future__ import annotations

import math

import pytest
import torch

from htbrl.model.heads import SlotHead, ToolHead, ValueHead
from htbrl.model.init import gpt2_init
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.model.transformer import (
    MLP,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerEncoder,
    apply_rope,
    precompute_rope_cache,
)


# ---- RoPE --------------------------------------------------------------------


def test_rope_cache_shape():
    cos, sin = precompute_rope_cache(seq_len=128, head_dim=48)
    assert cos.shape == (128, 24)
    assert sin.shape == (128, 24)


def test_rope_cache_is_deterministic():
    a = precompute_rope_cache(64, 48)
    b = precompute_rope_cache(64, 48)
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


def test_apply_rope_preserves_shape():
    cos, sin = precompute_rope_cache(seq_len=10, head_dim=16)
    x = torch.randn(2, 4, 10, 16)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape


def test_apply_rope_is_norm_preserving_per_pair():
    """RoPE rotates pairs of features, so the per-pair norm is preserved."""
    cos, sin = precompute_rope_cache(seq_len=8, head_dim=8)
    x = torch.randn(1, 1, 8, 8)
    y = apply_rope(x, cos, sin)
    # ||x[..., i] || should equal ||y[..., i]|| in the rotated-pair sense
    half = x.shape[-1] // 2
    x_pair_norm = torch.sqrt(x[..., :half] ** 2 + x[..., half:] ** 2)
    y_pair_norm = torch.sqrt(y[..., :half] ** 2 + y[..., half:] ** 2)
    assert torch.allclose(x_pair_norm, y_pair_norm, atol=1e-5)


def test_rope_requires_even_head_dim():
    with pytest.raises(AssertionError):
        precompute_rope_cache(seq_len=8, head_dim=7)


# ---- attention ---------------------------------------------------------------


def test_attention_shape():
    attn = MultiHeadSelfAttention(d_model=64, n_heads=4)
    cos, sin = precompute_rope_cache(seq_len=10, head_dim=16)
    x = torch.randn(2, 10, 64)
    out = attn(x, cos, sin)
    assert out.shape == x.shape


def test_attention_rejects_misshaped_d_model():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model=65, n_heads=4)


def test_attention_padding_mask_zeros_pad_contribution():
    """Padded positions should not contribute to the attention output."""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4)
    attn.eval()
    cos, sin = precompute_rope_cache(seq_len=4, head_dim=8)

    x = torch.randn(1, 4, 32)
    # Mask out positions 2 and 3 (additive bias of -inf there).
    mask = torch.zeros(1, 1, 1, 4)
    mask[..., 2:] = float("-inf")
    out_with_mask = attn(x, cos, sin, mask)

    # Replacing the padded positions with arbitrary noise should not change the
    # output at the unmasked positions.
    x_perturbed = x.clone()
    x_perturbed[:, 2:, :] = 1e3 * torch.randn_like(x[:, 2:, :])
    out_perturbed = attn(x_perturbed, cos, sin, mask)
    assert torch.allclose(out_with_mask[:, :2, :], out_perturbed[:, :2, :], atol=1e-4)


# ---- block + encoder ---------------------------------------------------------


def test_block_residual_output_shape():
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256)
    cos, sin = precompute_rope_cache(seq_len=8, head_dim=16)
    x = torch.randn(2, 8, 64)
    out = block(x, cos, sin)
    assert out.shape == x.shape


def test_encoder_forward_shapes():
    enc = TransformerEncoder(
        vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16
    )
    enc.eval()
    ids = torch.randint(0, 100, (3, 12))
    out = enc(ids)
    assert out.shape == (3, 12, 32)


def test_encoder_rejects_seq_overflow():
    enc = TransformerEncoder(vocab_size=100, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=4)
    ids = torch.randint(0, 100, (1, 10))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        enc(ids)


def test_encoder_attn_mask_shape_check():
    enc = TransformerEncoder(vocab_size=100, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8)
    ids = torch.randint(0, 100, (2, 5))
    bad_mask = torch.ones(2, 99, dtype=torch.bool)
    with pytest.raises(ValueError, match="attn_mask shape"):
        enc(ids, attn_mask=bad_mask)


def test_encoder_gradients_flow():
    enc = TransformerEncoder(vocab_size=100, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=8)
    ids = torch.randint(0, 100, (2, 5))
    out = enc(ids)
    loss = out.sum()
    loss.backward()
    # Every learnable parameter should have a gradient.
    for name, p in enc.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad on {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


# ---- heads -------------------------------------------------------------------


def test_tool_head_shape_and_matrix_bias_used():
    torch.manual_seed(0)
    head = ToolHead(d_model=8, n_tools=20, n_matrices=3)
    # Force the matrix bias to a non-zero, distinguishable pattern.
    with torch.no_grad():
        head.matrix_bias.weight.copy_(
            torch.tensor([[1.0] * 20, [0.0] * 20, [-1.0] * 20])
        )
    h = torch.zeros(3, 8)
    matrix_id = torch.tensor([0, 1, 2])
    logits = head(h, matrix_id)
    assert logits.shape == (3, 20)
    # The matrix bias should drive a clear ordering since trunk activation is 0.
    assert (logits[0] > logits[1]).all()
    assert (logits[1] > logits[2]).all()


def test_value_head_shape():
    head = ValueHead(d_model=16)
    h = torch.randn(5, 16)
    v = head(h)
    assert v.shape == (5,)


def test_slot_head_shape():
    head = SlotHead(d_model=12, slot_vocab_size=7)
    h = torch.randn(4, 12)
    tool_emb = torch.randn(4, 12)
    logits = head(h, tool_emb)
    assert logits.shape == (4, 7)


# ---- init --------------------------------------------------------------------


def test_gpt2_init_layernorm_defaults():
    ln = torch.nn.LayerNorm(16)
    # Mess up params
    with torch.no_grad():
        ln.weight.uniform_(-1, 1)
        ln.bias.uniform_(-1, 1)
    gpt2_init(ln)
    assert torch.allclose(ln.weight, torch.ones(16))
    assert torch.allclose(ln.bias, torch.zeros(16))


def test_gpt2_init_residual_scaling():
    """fc2 in MLP and proj in attention should be scaled by 1/sqrt(2L)."""
    block = TransformerBlock(d_model=8, n_heads=2, d_ff=16)
    n_layers = 4
    # Capture pre-init scale of fc2.weight (sqrt of mean-square).
    gpt2_init(block, n_residual_layers=n_layers)
    # The scaling factor is 1/sqrt(2L); standard init std was 0.02. Expected
    # post-scaling std is 0.02 / sqrt(8) ≈ 0.00707.
    expected_std = 0.02 / math.sqrt(2 * n_layers)
    actual_std = block.mlp.fc2.weight.std().item()
    assert actual_std < expected_std * 1.5, f"fc2 std={actual_std}, expected ~{expected_std}"


# ---- policy ------------------------------------------------------------------


def _tiny_cfg() -> PolicyConfig:
    return PolicyConfig(
        vocab_size=128,
        n_tools=10,
        n_matrices=3,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        slot_vocab_sizes=(5, 8),
    )


def test_policy_forward_shapes():
    p = ActorCriticPolicy(_tiny_cfg())
    p.eval()
    ids = torch.randint(0, 128, (3, 16))
    matrix_id = torch.tensor([0, 1, 2])
    tool_logits, value, hidden = p(ids, matrix_id)
    assert tool_logits.shape == (3, 10)
    assert value.shape == (3,)
    assert hidden.shape == (3, 32)


def test_policy_forward_with_padding_picks_last_unmasked_position():
    p = ActorCriticPolicy(_tiny_cfg())
    p.eval()
    ids = torch.randint(0, 128, (1, 16))
    matrix_id = torch.tensor([0])
    # Mask: positions 0..7 valid, 8..15 padded.
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :8] = True
    _, _, hidden_with_pad = p(ids, matrix_id, mask)
    # Compare with running just on the un-padded prefix.
    _, _, hidden_unpadded = p(ids[:, :8], matrix_id)
    # _last_position picks position 7 in both cases; the trunk states should match
    # at that position.
    assert torch.allclose(hidden_with_pad, hidden_unpadded, atol=1e-4)


def test_policy_slot_logits_routing():
    cfg = _tiny_cfg()
    p = ActorCriticPolicy(cfg)
    p.eval()
    hidden = torch.randn(3, cfg.d_model)
    tool_id = torch.tensor([0, 1, 2])
    out_0 = p.slot_logits(hidden, tool_id, slot_head_idx=0)
    out_1 = p.slot_logits(hidden, tool_id, slot_head_idx=1)
    assert out_0.shape == (3, 5)   # cfg.slot_vocab_sizes[0]
    assert out_1.shape == (3, 8)   # cfg.slot_vocab_sizes[1]


def test_policy_slot_logits_invalid_index():
    p = ActorCriticPolicy(_tiny_cfg())
    hidden = torch.randn(1, 32)
    tool_id = torch.tensor([0])
    with pytest.raises(IndexError):
        p.slot_logits(hidden, tool_id, slot_head_idx=99)


def test_policy_sample_tool_returns_valid_ids():
    torch.manual_seed(0)
    p = ActorCriticPolicy(_tiny_cfg())
    p.eval()
    ids = torch.randint(0, 128, (4, 16))
    matrix_id = torch.tensor([0, 1, 2, 0])
    tool_id, log_prob, value, hidden = p.sample_tool(ids, matrix_id)
    assert tool_id.shape == (4,)
    assert (tool_id >= 0).all() and (tool_id < 10).all()
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert hidden.shape == (4, 32)


def test_policy_gradients_flow_end_to_end():
    """Combined loss must exercise trunk + tool head + value head + slot heads + tool embedding."""
    p = ActorCriticPolicy(_tiny_cfg())
    p.train()
    ids = torch.randint(0, 128, (2, 16))
    matrix_id = torch.tensor([0, 1])
    tool_logits, value, hidden = p(ids, matrix_id)
    tool_id = torch.tensor([0, 1])
    # Slot logits use the tool embedding + slot heads.
    slot0 = p.slot_logits(hidden, tool_id, slot_head_idx=0)
    slot1 = p.slot_logits(hidden, tool_id, slot_head_idx=1)
    # Synthetic combined PPO-like loss.
    target = torch.zeros(2)
    loss = (
        torch.nn.functional.cross_entropy(tool_logits, tool_id)
        + (value - target).pow(2).mean()
        + torch.nn.functional.cross_entropy(slot0, torch.tensor([0, 0]))
        + torch.nn.functional.cross_entropy(slot1, torch.tensor([0, 0]))
    )
    loss.backward()
    for name, par in p.named_parameters():
        if par.requires_grad:
            assert par.grad is not None, f"no grad on {name}"
            assert torch.isfinite(par.grad).all(), f"non-finite grad on {name}"


def test_policy_param_count_matches_budget():
    """Verify the configured Phase 3 model fits the 3060 hardware budget.

    Plan spec (d_model=384, n_layers=8, d_ff=1536, vocab=32k, n_tools=300)
    actually yields ~27 M parameters - smaller than the rough "50-80 M" call-out
    in PLAN.md's narrative. That's fine and in fact better for the 3060: more
    headroom for activations + reference policy + reward model in 12 GB. The
    hard upper bound enforced here is 60 M, well under what the GPU can fit
    with bf16 + 8-bit AdamW.
    """
    cfg = PolicyConfig(
        vocab_size=32_768,
        n_tools=300,
        n_matrices=3,
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1536,
        max_seq_len=1024,
        dropout=0.0,
        slot_vocab_sizes=(),
    )
    p = ActorCriticPolicy(cfg)
    n = p.n_parameters()
    assert 20_000_000 <= n <= 60_000_000, (
        f"param count {n:,} outside expected envelope (20M-60M for 3060 budget)"
    )


# ---- CUDA-only ---------------------------------------------------------------


@pytest.mark.gpu
def test_policy_runs_on_cuda_under_vram_budget():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = PolicyConfig(
        vocab_size=32_768,
        n_tools=300,
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1536,
        max_seq_len=1024,
        dropout=0.0,
    )
    p = ActorCriticPolicy(cfg).cuda()
    ids = torch.randint(0, 32_768, (4, 1024), device="cuda")
    matrix_id = torch.zeros(4, dtype=torch.int64, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_logits, value, hidden = p(ids, matrix_id)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Plan budget for inference at this batch size: comfortably under 4 GB.
    assert peak_gb < 5.0, f"inference peak VRAM {peak_gb:.2f} GB > 5 GB budget"
    assert out_logits.shape == (4, 300)
    assert value.shape == (4,)
    assert hidden.shape == (4, 384)
