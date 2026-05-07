"""RLHF composite reward (PLAN.md Phase 9).

Combines the env reward, the learned reward model's score, the RND novelty
bonus, and the KL penalty against the reference policy into one scalar
the PPO loss consumes:

    r_total = r_env + alpha * r_rm + beta * r_rnd - c * KL(pi || pi_ref)

The KL term is applied per-sample inside the PPO loss (compute_ppo_loss in
algo/ppo.py); this module composes only the additive r_total signal that goes
into GAE.

We keep this layer thin and side-effect-free so it stays trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CompositeRewardConfig:
    # alpha: env-reward + alpha * rm_reward
    rm_coef: float = 0.5
    rm_anneal_steps: int = 1_000_000  # alpha ramps 0 -> rm_coef over this budget
    rm_anneal_floor: float = 0.0      # final alpha after anneal complete (=rm_coef)

    # beta: RND novelty bonus
    rnd_coef_init: float = 0.5
    rnd_coef_final: float = 0.05
    rnd_anneal_steps: int = 5_000_000

    # Reward-model "min-of-ensemble" pessimism: when an ensemble is provided,
    # we use the minimum across heads to be conservative against high-variance
    # regions where one RM is overconfident.
    use_rm_ensemble_min: bool = True


def alpha_schedule(env_steps: int, cfg: CompositeRewardConfig) -> float:
    """Linear ramp of the RM weight from 0 to ``rm_coef`` over ``rm_anneal_steps``."""
    if cfg.rm_anneal_steps <= 0:
        return cfg.rm_coef
    progress = min(1.0, env_steps / cfg.rm_anneal_steps)
    return progress * cfg.rm_coef


def beta_schedule(env_steps: int, cfg: CompositeRewardConfig) -> float:
    """Linear anneal of RND weight from ``rnd_coef_init`` to ``rnd_coef_final``."""
    if cfg.rnd_anneal_steps <= 0:
        return cfg.rnd_coef_final
    progress = min(1.0, env_steps / cfg.rnd_anneal_steps)
    return cfg.rnd_coef_init + (cfg.rnd_coef_final - cfg.rnd_coef_init) * progress


def reduce_rm_ensemble(
    scores: list[torch.Tensor],
    use_min: bool = True,
) -> torch.Tensor:
    """Combine multiple RM head scores into one scalar per sample.

    ``scores`` is a list of (B,) tensors, one per RM in the ensemble.
    ``use_min=True`` -> per-sample minimum (Schulman pessimism).
    ``use_min=False`` -> mean.
    """
    if not scores:
        raise ValueError("reduce_rm_ensemble called with no scores")
    stacked = torch.stack(scores, dim=0)  # (n_rm, B)
    if use_min:
        return stacked.min(dim=0).values
    return stacked.mean(dim=0)


def composite_reward(
    *,
    env_reward: torch.Tensor,           # (B,) or (T, N)
    rm_score: torch.Tensor | None,
    rnd_bonus: torch.Tensor | None,
    env_steps_seen: int,
    cfg: CompositeRewardConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``(r_total, scalars)``. KL penalty is applied separately in PPO loss.

    All tensors must broadcast against ``env_reward``. ``rm_score`` and
    ``rnd_bonus`` may be None to disable that term.
    """
    alpha = alpha_schedule(env_steps_seen, cfg)
    beta = beta_schedule(env_steps_seen, cfg)
    total = env_reward.clone()
    scalars: dict[str, float] = {"alpha": alpha, "beta": beta}

    if rm_score is not None and alpha > 0:
        if rm_score.shape != env_reward.shape:
            # broadcast support: rm_score is sometimes (B,) when env_reward is (T,N).
            rm_score = rm_score.expand_as(env_reward)
        total = total + alpha * rm_score
        scalars["rm_score_mean"] = float(rm_score.detach().mean())

    if rnd_bonus is not None and beta > 0:
        if rnd_bonus.shape != env_reward.shape:
            rnd_bonus = rnd_bonus.expand_as(env_reward)
        total = total + beta * rnd_bonus
        scalars["rnd_bonus_mean"] = float(rnd_bonus.detach().mean())

    scalars["env_reward_mean"] = float(env_reward.detach().mean())
    scalars["composite_reward_mean"] = float(total.detach().mean())
    return total, scalars
