"""Generalized Advantage Estimation (Schulman et al. 2016).

Pure-function implementation, no module state. Operates on plain torch tensors
of shape ``(T, N)`` (timesteps along axis 0, parallel envs along axis 1).

Why this lives in ``algo/``: GAE is the bridge between rollout collection and
policy update. PPO calls ``compute_gae(rewards, values, dones, last_value)``
once per rollout to produce advantages + returns, then iterates over them in
minibatch updates.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.995,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards:    (T, N) reward at each step.
        values:     (T, N) critic prediction at each step (V(s_t)).
        dones:      (T, N) bool/float; 1 means the episode terminated AT step t
                    (so there is no bootstrap from t+1).
        last_value: (N,) bootstrap value for s_T (V(s_T)). Pass zeros for terminated rollouts.
        gamma:      discount.
        lam:        GAE lambda. lam=0 -> 1-step TD; lam=1 -> Monte-Carlo returns.

    Returns:
        advantages: (T, N)
        returns:    (T, N)  — equal to ``advantages + values``, the regression
                    target for the value head.

    Numerical notes:
    - Computed in fp32 regardless of input dtype, then cast back to ``rewards.dtype``.
    - Iterates backwards in Python: T is small (≤ 1024 typically) so the loop
      cost is negligible relative to the rest of an update step.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"shape mismatch: rewards {rewards.shape}, values {values.shape}, dones {dones.shape}"
        )
    if last_value.shape != rewards.shape[1:]:
        raise ValueError(
            f"last_value shape {last_value.shape} does not match rewards trailing shape {rewards.shape[1:]}"
        )

    T = rewards.shape[0]
    out_dtype = rewards.dtype
    rewards_f = rewards.to(torch.float32)
    values_f = values.to(torch.float32)
    dones_f = dones.to(torch.float32)
    last_v = last_value.to(torch.float32)

    advantages = torch.zeros_like(rewards_f)
    gae = torch.zeros_like(last_v)
    for t in reversed(range(T)):
        not_terminal = 1.0 - dones_f[t]
        next_v = last_v if t == T - 1 else values_f[t + 1]
        delta = rewards_f[t] + gamma * next_v * not_terminal - values_f[t]
        gae = delta + gamma * lam * not_terminal * gae
        advantages[t] = gae

    returns = advantages + values_f
    return advantages.to(out_dtype), returns.to(out_dtype)


def normalize_advantages(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-batch z-score normalization. Standard PPO trick to stabilize updates."""
    flat = advantages.reshape(-1)
    mean = flat.mean()
    std = flat.std(unbiased=False)
    return (advantages - mean) / (std + eps)
