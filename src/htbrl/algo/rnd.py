"""Random Network Distillation (Burda et al. 2018).

Provides an exploration bonus = MSE between a frozen random network's output
and a trainable predictor's output, both fed the same observation embedding.
The predictor learns to match the target on visited states; the prediction
error stays high on novel states, which gives a "novelty bonus" without any
explicit count-based exploration.

Why this matters for our project: the user explicitly asked for "BC warm-start
plus cold-start RL spirit." RND is the cleanest way to inject cold-start
exploration into a policy that's already been BC-pretrained: the agent gets
extra reward for trying state-action combinations the BC distribution didn't
cover.

Architecture notes:
- Both nets are tiny MLPs (we don't need a transformer; the predictor only has
  to match the target's distribution, and the input is already the trunk's
  hidden state).
- Target network's parameters are frozen at init and never updated.
- Bonus is normalized by a running std of prediction errors so it stays on a
  reasonable scale across long training (Burda et al. recipe).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalizer import RunningStats


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RND(nn.Module):
    """Random Network Distillation.

    Call ``compute_bonus(emb)`` for the exploration reward (no grad).
    Call ``predictor_loss(emb)`` during training to update the predictor.
    """

    def __init__(self, d_in: int = 384, d_hidden: int = 256, d_out: int = 128) -> None:
        super().__init__()
        self.target = _MLP(d_in, d_hidden, d_out)
        self.predictor = _MLP(d_in, d_hidden, d_out)
        # Freeze the target. We use a separate ParameterList so torch knows
        # not to optimize it when an outer optimizer sees self.parameters().
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.bonus_stats = RunningStats()

    @torch.no_grad()
    def compute_bonus(
        self,
        emb: torch.Tensor,
        update_running_stats: bool = True,
        clip_to: float = 5.0,
    ) -> torch.Tensor:
        """Return the per-sample novelty bonus: ||target(emb) - predictor(emb)||^2 (mean over feature dim).

        emb: (B, d_in) -> bonus: (B,)
        """
        target_out = self.target(emb)
        predictor_out = self.predictor(emb)
        err = (target_out - predictor_out).pow(2).mean(dim=-1)  # (B,)
        if update_running_stats:
            self.bonus_stats.update(err)
        if self.bonus_stats.count >= 2:
            std = self.bonus_stats.std.to(err.device, err.dtype)
            err = err / (std + 1e-8)
        return err.clamp(max=clip_to)

    def predictor_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """MSE training loss for the predictor. Target is frozen; gradients flow through predictor only."""
        target_out = self.target(emb).detach()
        predictor_out = self.predictor(emb)
        return F.mse_loss(predictor_out, target_out)
