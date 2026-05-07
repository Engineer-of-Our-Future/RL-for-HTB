"""Polyak-averaged target networks (PLAN.md Phase 7 - Critic refinements).

The target net follows the live net via slow exponential moving average. PPO
uses it as the bootstrap value source for GAE: ``V_target(s_{T+1})`` rather
than ``V_live(s_{T+1})`` — this dampens the value-overestimation that hits
long-horizon episodes once the critic starts moving fast under the policy
loss.

This module is a generic helper, not PPO-specific:
- ``TargetNetwork.update`` does ``target = tau * live + (1 - tau) * target``
- ``TargetNetwork.value`` runs a forward through the frozen target without grad

Wiring into the PPO loop is in scripts/train_ppo.py: build a target policy
mirroring the live policy at startup, ``update`` it after each ppo_update, and
use it for the ``last_value`` bootstrap fed to compute_gae.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class TargetNetwork:
    """Polyak-averaged copy of a torch ``nn.Module``.

    Constructed with a *live* module reference; deep-copies it for the target.
    Subsequent ``update(tau=...)`` blends the live params into the target.
    """

    def __init__(self, live: nn.Module, tau: float = 0.005) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1]; got {tau}")
        self._live = live
        self._target = copy.deepcopy(live)
        for p in self._target.parameters():
            p.requires_grad_(False)
        self._target.eval()
        self.tau = tau

    @property
    def target(self) -> nn.Module:
        return self._target

    def to(self, device: torch.device | str) -> "TargetNetwork":
        self._target.to(device)
        return self

    @torch.no_grad()
    def update(self, tau: float | None = None) -> None:
        """Move target params toward live params by ``tau``."""
        t = self.tau if tau is None else tau
        for tp, lp in zip(self._target.parameters(), self._live.parameters()):
            tp.mul_(1.0 - t).add_(lp.detach(), alpha=t)
        # Buffers (e.g., LayerNorm running stats) — copy directly each update.
        for tb, lb in zip(self._target.buffers(), self._live.buffers()):
            tb.copy_(lb.detach())

    @torch.no_grad()
    def hard_copy(self) -> None:
        """Reset the target to the live params exactly (tau=1)."""
        self.update(tau=1.0)

    def state_dict(self) -> dict:
        return {"target": self._target.state_dict(), "tau": self.tau}

    def load_state_dict(self, sd: dict) -> None:
        self._target.load_state_dict(sd["target"])
        self.tau = float(sd["tau"])
