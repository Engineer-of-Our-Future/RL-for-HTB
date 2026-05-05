"""Welford-style online running mean / std.

Used for reward normalization (PLAN.md Phase 7 - Critic refinements). The
``RunningStats`` object accumulates (sum, sum_of_squares, n) so it can be saved
to a checkpoint and resumed exactly. Numerical stability comes from doing all
arithmetic in fp64 internally; callers can pass fp32 / bf16 tensors freely.
"""

from __future__ import annotations

import torch


class RunningStats:
    """Tracks the running mean and variance of a stream of scalars.

    Implements the parallel form of Welford's algorithm so that batches of
    samples can be incorporated with a single ``update(batch)`` call.

    Internal state:
    - ``count``  : number of samples seen so far (Python int)
    - ``mean``   : running mean (fp64 tensor scalar)
    - ``m2``     : running sum-of-squared-deviations from current mean (fp64)
    """

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: torch.Tensor = torch.zeros((), dtype=torch.float64)
        self.m2: torch.Tensor = torch.zeros((), dtype=torch.float64)

    @property
    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros((), dtype=torch.float64)
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> torch.Tensor:
        return self.variance.sqrt()

    def update(self, x: torch.Tensor) -> None:
        """Incorporate a batch of samples. ``x`` may be any shape; flattened first."""
        flat = x.detach().to(torch.float64).reshape(-1)
        n = flat.numel()
        if n == 0:
            return
        batch_mean = flat.mean()
        batch_m2 = ((flat - batch_mean) ** 2).sum()

        new_count = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (n / new_count)
        new_m2 = self.m2 + batch_m2 + (delta ** 2) * (self.count * n / new_count)

        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Return ``(x - mean) / (std + eps)`` in ``x``'s original dtype/device."""
        if self.count < 2:
            return x.clone()
        std = self.std.to(x.device, x.dtype)
        mean = self.mean.to(x.device, x.dtype)
        return (x - mean) / (std + eps)

    # ----- persistence --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean.item(),
            "m2": self.m2.item(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.count = int(sd["count"])
        self.mean = torch.tensor(sd["mean"], dtype=torch.float64)
        self.m2 = torch.tensor(sd["m2"], dtype=torch.float64)
