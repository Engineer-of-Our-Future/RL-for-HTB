"""PPO update math (PLAN.md Phase 6).

This module is the *update step* — given a filled rollout buffer, compute the
PPO loss and step the optimizer. Rollout collection lives in the env layer
(Phase 4) which calls this once per rollout.

Composition:
- ``compute_ppo_loss`` — the loss math (clipped policy + value + entropy + KL).
- ``ppo_update`` — one update over a rollout: iterate epochs × minibatches,
  call backward, step optimizer, return aggregate metrics.

What's NOT here:
- Rollout collection (env-dependent; Phase 4 supplies it via a callback).
- The slot-action loss (slot heads need the tool registry to know which slot
  vocab to score; integrate in Phase 6 once the env wraps registry actions).
  This module currently scores only the tool-id logits, which is enough to
  get the trainer running end-to-end on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from htbrl.algo.gae import normalize_advantages
from htbrl.algo.kl_ctrl import AdaptiveKLController
from htbrl.data.rollout_buffer import RolloutBatch, RolloutBuffer


@dataclass
class PPOConfig:
    clip_range: float = 0.2          # PPO surrogate clip
    value_clip_range: float = 0.2    # value-loss clip (Mnih-style)
    value_coef: float = 0.5
    entropy_coef: float = 0.02       # high; preserves cold-start exploration
    n_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl_for_early_stop: float | None = 0.05  # if observed_kl exceeds, stop
    kl_penalty_coef: float = 0.0     # adaptive KL ctrl supplies this if used
    normalize_advantages: bool = True


@dataclass
class PPOMetrics:
    loss_total: float
    loss_policy: float
    loss_value: float
    loss_entropy: float
    loss_kl: float
    approx_kl: float                 # estimator for KL(pi_old || pi_new) per Schulman
    clip_fraction: float
    explained_variance: float
    n_epochs_run: int
    early_stopped: bool


def compute_ppo_loss(
    *,
    new_log_prob: torch.Tensor,    # (B,)
    old_log_prob: torch.Tensor,    # (B,)
    advantage: torch.Tensor,       # (B,)
    return_: torch.Tensor,         # (B,)
    new_value: torch.Tensor,       # (B,)
    old_value: torch.Tensor,       # (B,)
    entropy: torch.Tensor,         # (B,) per-sample policy entropy
    cfg: PPOConfig,
    ref_log_prob: torch.Tensor | None = None,  # (B,) for KL penalty
    kl_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-batch PPO loss (without backward / optimizer)."""
    if cfg.normalize_advantages:
        advantage = normalize_advantages(advantage)

    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()

    # Clipped policy loss
    pg_unclipped = -advantage * ratio
    pg_clipped = -advantage * ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    policy_loss = torch.maximum(pg_unclipped, pg_clipped).mean()

    # Clipped value loss (PPO2 style)
    v_clipped = old_value + (new_value - old_value).clamp(
        -cfg.value_clip_range, cfg.value_clip_range
    )
    v_loss_unclipped = (new_value - return_) ** 2
    v_loss_clipped = (v_clipped - return_) ** 2
    value_loss = 0.5 * torch.maximum(v_loss_unclipped, v_loss_clipped).mean()

    entropy_loss = -entropy.mean()  # we *maximize* entropy => negate

    kl_loss = torch.zeros((), device=new_log_prob.device, dtype=new_log_prob.dtype)
    if ref_log_prob is not None and kl_coef > 0:
        # Forward KL(pi || pi_ref) ≈ E_pi[log pi - log pi_ref]
        # We use Schulman's k3 estimator for low-variance unbiased KL.
        log_ratio_ref = new_log_prob - ref_log_prob
        kl_per_sample = (log_ratio_ref.exp() - 1) - log_ratio_ref
        kl_loss = kl_coef * kl_per_sample.mean()

    total = (
        policy_loss
        + cfg.value_coef * value_loss
        + cfg.entropy_coef * entropy_loss
        + kl_loss
    )

    # Diagnostics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()  # k3 estimator of KL(old||new)
        clip_fraction = (
            ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
        )
        var_returns = return_.var(unbiased=False)
        explained_variance = (
            torch.tensor(0.0, device=new_log_prob.device)
            if var_returns < 1e-8
            else 1.0 - ((return_ - new_value).var(unbiased=False) / var_returns)
        )

    diagnostics = {
        "loss_policy": policy_loss.detach(),
        "loss_value": value_loss.detach(),
        "loss_entropy": entropy_loss.detach(),
        "loss_kl": kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else torch.tensor(0.0),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clip_fraction.detach(),
        "explained_variance": explained_variance.detach(),
    }
    return total, diagnostics


def policy_logp_value_entropy(
    policy,
    batch: RolloutBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the policy on a rollout batch and return (new_logp, new_value, entropy).

    ``policy`` is duck-typed: must accept ``forward(token_ids, matrix_id, attn_mask)``
    returning ``(tool_logits, value, hidden)``. Slot heads are NOT scored here -
    see module docstring.
    """
    tool_logits, value, _ = policy(batch.obs_tokens, batch.matrix_id, batch.attn_mask)
    log_probs = F.log_softmax(tool_logits, dim=-1)
    new_logp = log_probs.gather(-1, batch.action_tool_id.unsqueeze(-1)).squeeze(-1)
    # Entropy = -sum_a p(a) log p(a)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return new_logp, value, entropy


def ppo_update(
    *,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBuffer,
    cfg: PPOConfig,
    kl_ctrl: AdaptiveKLController | None = None,
    ref_policy: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> PPOMetrics:
    """Run one PPO update over the (already-GAE'd) rollout and return aggregate metrics."""
    device = torch.device(device)
    policy.train()
    if ref_policy is not None:
        ref_policy.eval()

    # Aggregators
    accum = {
        "loss_total": 0.0, "loss_policy": 0.0, "loss_value": 0.0,
        "loss_entropy": 0.0, "loss_kl": 0.0, "approx_kl": 0.0,
        "clip_fraction": 0.0, "explained_variance": 0.0,
    }
    n_batches = 0
    early_stop = False
    epochs_run = 0
    kl_coef = float(kl_ctrl.coef) if kl_ctrl is not None else cfg.kl_penalty_coef

    for epoch in range(cfg.n_epochs):
        for batch in rollout.iter_minibatches(cfg.minibatch_size, shuffle=True, device=device):
            new_logp, new_val, entropy = policy_logp_value_entropy(policy, batch)

            ref_logp: torch.Tensor | None = None
            if ref_policy is not None:
                with torch.no_grad():
                    ref_logp_full, _, _ = policy_logp_value_entropy(ref_policy, batch)
                ref_logp = ref_logp_full

            loss, diag = compute_ppo_loss(
                new_log_prob=new_logp,
                old_log_prob=batch.log_prob_old,
                advantage=batch.advantage,
                return_=batch.return_,
                new_value=new_val,
                old_value=batch.value_old,
                entropy=entropy,
                cfg=cfg,
                ref_log_prob=ref_logp,
                kl_coef=kl_coef,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            for k in accum:
                if k == "loss_total":
                    accum[k] += float(loss.detach())
                else:
                    accum[k] += float(diag[k])
            n_batches += 1

        epochs_run = epoch + 1
        epoch_kl = accum["approx_kl"] / max(n_batches, 1)
        if cfg.target_kl_for_early_stop is not None and epoch_kl > cfg.target_kl_for_early_stop:
            early_stop = True
            break

    avg = {k: v / max(n_batches, 1) for k, v in accum.items()}

    # Update adaptive KL controller for next rollout, if attached.
    if kl_ctrl is not None:
        kl_ctrl.update(observed_kl=avg["approx_kl"])

    return PPOMetrics(
        loss_total=avg["loss_total"],
        loss_policy=avg["loss_policy"],
        loss_value=avg["loss_value"],
        loss_entropy=avg["loss_entropy"],
        loss_kl=avg["loss_kl"],
        approx_kl=avg["approx_kl"],
        clip_fraction=avg["clip_fraction"],
        explained_variance=avg["explained_variance"],
        n_epochs_run=epochs_run,
        early_stopped=early_stop,
    )
