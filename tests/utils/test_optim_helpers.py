"""Tests for the hardware-targeted training helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from htbrl.utils.optim_helpers import (
    apply_grad_checkpointing_to_blocks,
    autocast_context,
    build_optimizer,
    is_grad_checkpointed,
    log_vram_state,
    maybe_compile,
    reset_peak_vram,
)


# ---- maybe_compile ----------------------------------------------------------


def test_maybe_compile_disable_returns_original_module():
    m = nn.Linear(4, 4)
    out = maybe_compile(m, enable=False)
    assert out is m


def test_maybe_compile_runs_forward_after_compile():
    """Whether or not torch.compile actually compiles, the resulting module's
    forward should still produce the right shape on a simple input."""
    m = nn.Linear(4, 4)
    compiled = maybe_compile(m, enable=True, mode="default")
    out = compiled(torch.randn(3, 4))
    assert out.shape == (3, 4)


# ---- gradient checkpointing -------------------------------------------------


def test_grad_checkpointing_marker():
    block = nn.Linear(4, 4)
    assert not is_grad_checkpointed(block)
    apply_grad_checkpointing_to_blocks([block])
    assert is_grad_checkpointed(block)


def test_grad_checkpointing_preserves_forward_shape():
    blocks = [nn.Linear(4, 4) for _ in range(3)]
    apply_grad_checkpointing_to_blocks(blocks)

    x = torch.randn(2, 4, requires_grad=True)
    y = x
    for b in blocks:
        y = b(y)
    assert y.shape == (2, 4)
    # Gradient still flows
    y.sum().backward()
    assert x.grad is not None
    for b in blocks:
        for p in b.parameters():
            assert p.grad is not None


def test_grad_checkpointing_count():
    blocks = [nn.Linear(4, 4) for _ in range(5)]
    n = apply_grad_checkpointing_to_blocks(blocks)
    assert n == 5


# ---- build_optimizer --------------------------------------------------------


def test_build_optimizer_returns_adamw():
    params = nn.Linear(4, 4).parameters()
    opt = build_optimizer(params, lr=1e-3, weight_decay=0.01, fused=False)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_8bit_falls_back_when_bnb_missing():
    """If bitsandbytes isn't importable (e.g. native Windows), the helper
    must transparently return torch.optim.AdamW. We don't actually mock the
    import - on machines that DO have bnb installed, this just confirms the
    happy path returns *some* optimizer."""
    params = list(nn.Linear(4, 4).parameters())
    opt = build_optimizer(params, lr=1e-3, use_8bit=True, fused=False)
    # Either bnb was available -> AdamW8bit ;  or it wasn't -> AdamW
    assert hasattr(opt, "step")
    # Steps should not raise on a synthetic gradient
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()


def test_build_optimizer_step_does_not_raise():
    layer = nn.Linear(4, 4)
    opt = build_optimizer(layer.parameters(), lr=1e-3, fused=False)
    x = torch.randn(2, 4)
    loss = layer(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()


# ---- autocast --------------------------------------------------------------


def test_autocast_returns_nullcontext_on_cpu():
    cm = autocast_context(device_type="cpu", enabled=True)
    # Either nullcontext or autocast - just ensure it works as a CM
    with cm:
        x = torch.randn(2, 2) @ torch.randn(2, 2)
    assert x.shape == (2, 2)


def test_autocast_disabled_is_noop():
    cm = autocast_context(enabled=False)
    with cm:
        x = torch.randn(2, 2)
    assert x.dtype == torch.float32


# ---- VRAM diagnostics -------------------------------------------------------


def test_log_vram_state_returns_dict_on_cpu():
    info = log_vram_state(label="test")
    assert "current_mb" in info
    assert "peak_mb" in info
    assert info["label"] == "test"


def test_reset_peak_vram_no_op_on_cpu():
    # Just shouldn't raise
    reset_peak_vram()
