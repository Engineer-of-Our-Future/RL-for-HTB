"""Weight initialization for from-scratch training.

GPT-2-style scaled init: linear weights ~ N(0, 0.02), embeddings ~ N(0, 0.02),
LayerNorm gamma=1, beta=0. Residual-projection layers get an extra 1/sqrt(2L)
scaling to keep activations bounded as depth grows.

This is the *formula* GPT-2 uses, applied to a freshly-randomized network. We
don't load any GPT-2 weights or any other pretrained checkpoint - the project
rule is "no pretrained models," and that includes pretrained tokenizer
embeddings, not just full LLM weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def gpt2_init(module: nn.Module, n_residual_layers: int | None = None) -> None:
    """Initialize every submodule of `module` in place.

    If `n_residual_layers` is given, the second linear of each residual
    projection (named ``proj`` on attention, ``fc2`` on MLP) is rescaled by
    ``1 / sqrt(2 * n_residual_layers)`` so deeper stacks don't blow up.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    if n_residual_layers is not None and n_residual_layers > 0:
        scale = 1.0 / math.sqrt(2.0 * n_residual_layers)
        for name, p in module.named_parameters():
            # Match the residual projection of attention (proj) and MLP (fc2).
            if name.endswith("attn.proj.weight") or name.endswith("mlp.fc2.weight"):
                with torch.no_grad():
                    p.mul_(scale)
