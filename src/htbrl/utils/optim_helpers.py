"""Hardware-targeted training helpers (PLAN.md Phase 10).

These wrap PyTorch's standard optimizations with graceful fallbacks so the
training scripts work the same on the Windows host (where bitsandbytes wheels
are unreliable, torch.compile sometimes regresses on small custom modules,
etc.) as inside WSL Linux. Each helper is a thin function with one job:

- ``maybe_compile``: ``torch.compile(model, mode=...)`` with try/except so a
  compile failure logs a warning and returns the un-compiled module.
- ``apply_grad_checkpointing_to_blocks``: re-wraps each block's forward in
  ``torch.utils.checkpoint.checkpoint`` (use_reentrant=False). Cuts activation
  memory ~60 % for ~25 % step-time hit.
- ``build_optimizer``: builds AdamW. With ``use_8bit=True``, tries
  ``bitsandbytes.optim.AdamW8bit``; falls back to torch's AdamW with
  ``fused=True`` on CUDA. Saves ≈ 1.5 GB optimizer state at 100 M params.
- ``autocast_context``: ``torch.autocast(...)`` with project-default dtype
  (bf16 on Ampere) and CPU/CUDA detection.
- ``log_vram_state``: cheap diagnostic that prints peak/current VRAM. Easy to
  call before/after a phase to verify we're under the 11 GB ceiling.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


log = logging.getLogger("htbrl.optim")


# ---- torch.compile ----------------------------------------------------------


def maybe_compile(
    module: nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    enable: bool = True,
) -> nn.Module:
    """Try to ``torch.compile`` ``module``. On any error, log + return the original.

    ``enable=False`` is a hard kill switch: returns the un-compiled module
    without trying. Useful in tests + CPU-only paths where compile gives no
    benefit and slows things down.
    """
    if not enable:
        return module
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    except Exception as exc:
        log.warning("torch.compile failed (%s); using un-compiled module", exc)
        return module


# ---- gradient checkpointing -------------------------------------------------


def apply_grad_checkpointing_to_blocks(
    blocks: Iterable[nn.Module],
    use_reentrant: bool = False,
) -> int:
    """Re-wrap each block's forward to call ``torch.utils.checkpoint`` instead.

    Operates in place on the iterable of modules (e.g. ``encoder.blocks``).
    Returns the number of blocks that got wrapped.

    ``use_reentrant=False`` is the modern default; reentrant=True is the
    legacy path that's slated for removal in PyTorch 2.x.
    """
    count = 0
    for block in blocks:
        original_forward = block.forward

        def make_ckpt_forward(orig):
            def fwd(*args, **kwargs):
                return checkpoint(orig, *args, use_reentrant=use_reentrant, **kwargs)
            return fwd

        block.forward = make_ckpt_forward(original_forward)
        # Mark so we can detect-and-skip a second application.
        block._htbrl_grad_checkpointed = True  # type: ignore[attr-defined]
        count += 1
    return count


def is_grad_checkpointed(block: nn.Module) -> bool:
    return getattr(block, "_htbrl_grad_checkpointed", False)


# ---- optimizer --------------------------------------------------------------


def build_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    use_8bit: bool = False,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    """Build AdamW with optional 8-bit optimizer state.

    Order of preference:
      1. If ``use_8bit=True`` and ``bitsandbytes`` imports cleanly -> ``AdamW8bit``.
         (Linux/WSL only; Windows wheels are historically unreliable.)
      2. Else ``torch.optim.AdamW``. Uses ``fused=True`` on CUDA when ``fused``
         is None or True; respects ``fused=False`` for explicit override.
    """
    if use_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore
            log.info("using bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(
                params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
            )
        except ImportError:
            log.warning(
                "use_8bit=True but bitsandbytes is not installed; "
                "falling back to torch.optim.AdamW"
            )

    if fused is None:
        fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
        fused=fused,
    )


# ---- autocast ---------------------------------------------------------------


def autocast_context(
    device_type: str | None = None,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
):
    """``torch.autocast`` wrapper picking sane defaults.

    On a CUDA device with bf16 support (Ampere+, the 3060 qualifies), uses
    bf16. Otherwise falls back to fp16 (older CUDA) or no-op (CPU).
    """
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if device_type == "cuda":
            cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            dtype = torch.bfloat16
    if not enabled or device_type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


# ---- diagnostics ------------------------------------------------------------


def log_vram_state(label: str = "") -> dict:
    """Log + return the current and peak VRAM (in MB).

    No-ops on CPU. Useful to sprinkle around training loops to find spikes:
    the 3060 is 12 GB total; we want peak <= 11 GB (PLAN.md hardware budget).
    """
    if not torch.cuda.is_available():
        return {"current_mb": 0.0, "peak_mb": 0.0, "label": label}
    cur = torch.cuda.memory_allocated() / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    log.info("[vram%s] current=%.0f MB  peak=%.0f MB",
             f" {label}" if label else "", cur, peak)
    return {"current_mb": cur, "peak_mb": peak, "label": label}


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
