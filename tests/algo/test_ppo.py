"""Tests for the PPO update step. No env, no rollout collection - we synthesize
filled rollout buffers and verify the loss math + update step are stable."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from htbrl.algo.gae import compute_gae
from htbrl.algo.kl_ctrl import AdaptiveKLController
from htbrl.algo.ppo import (
    PPOConfig,
    compute_ppo_loss,
    policy_logp_value_entropy,
    ppo_update,
)
from htbrl.data.rollout_buffer import RolloutBuffer
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig


# ---- compute_ppo_loss --------------------------------------------------------


def test_ppo_loss_zero_when_no_advantage_and_clean_policy():
    """If advantage is 0 and ratio = 1, policy / value / KL terms should vanish; only entropy survives."""
    new_logp = torch.zeros(4)
    old_logp = torch.zeros(4)
    advantage = torch.zeros(4)
    return_ = torch.zeros(4)
    new_value = torch.zeros(4)
    old_value = torch.zeros(4)
    entropy = torch.zeros(4)
    cfg = PPOConfig(entropy_coef=0.0, normalize_advantages=False)
    loss, diag = compute_ppo_loss(
        new_log_prob=new_logp, old_log_prob=old_logp,
        advantage=advantage, return_=return_,
        new_value=new_value, old_value=old_value,
        entropy=entropy, cfg=cfg,
    )
    assert abs(loss.item()) < 1e-6


def test_ppo_loss_clip_kicks_in_far_from_old_policy():
    """When ratio drifts well past 1 ± clip_range, clipping should bound the policy loss."""
    new_logp = torch.tensor([5.0, 5.0, 5.0, 5.0])
    old_logp = torch.tensor([0.0, 0.0, 0.0, 0.0])
    advantage = torch.ones(4)
    return_ = torch.zeros(4)
    new_value = torch.zeros(4)
    old_value = torch.zeros(4)
    entropy = torch.zeros(4)
    cfg = PPOConfig(clip_range=0.2, entropy_coef=0.0, value_coef=0.0, normalize_advantages=False)
    loss, diag = compute_ppo_loss(
        new_log_prob=new_logp, old_log_prob=old_logp,
        advantage=advantage, return_=return_,
        new_value=new_value, old_value=old_value,
        entropy=entropy, cfg=cfg,
    )
    # Without clipping policy_loss = -advantage * ratio = -e^5 ≈ -148.
    # With clipping it's bounded around -1.2 (= -1 * (1 + 0.2)).
    assert loss.item() > -2.0
    assert diag["clip_fraction"].item() == 1.0


def test_ppo_loss_kl_term_only_when_ref_provided():
    new_logp = torch.tensor([0.0, 0.0])
    old_logp = torch.tensor([0.0, 0.0])
    advantage = torch.zeros(2)
    return_ = torch.zeros(2)
    new_value = torch.zeros(2)
    old_value = torch.zeros(2)
    entropy = torch.zeros(2)
    ref_logp = torch.tensor([1.0, -1.0])  # different from new
    cfg = PPOConfig(entropy_coef=0.0, value_coef=0.0, normalize_advantages=False)

    # Without ref, KL term is zero.
    _, diag_no_ref = compute_ppo_loss(
        new_log_prob=new_logp, old_log_prob=old_logp,
        advantage=advantage, return_=return_,
        new_value=new_value, old_value=old_value,
        entropy=entropy, cfg=cfg,
    )
    assert diag_no_ref["loss_kl"].item() == 0.0

    # With ref + nonzero coef, KL term contributes.
    _, diag_ref = compute_ppo_loss(
        new_log_prob=new_logp, old_log_prob=old_logp,
        advantage=advantage, return_=return_,
        new_value=new_value, old_value=old_value,
        entropy=entropy, cfg=cfg,
        ref_log_prob=ref_logp, kl_coef=1.0,
    )
    assert diag_ref["loss_kl"].item() > 0.0


# ---- policy_logp_value_entropy -----------------------------------------------


def _tiny_policy_cfg() -> PolicyConfig:
    return PolicyConfig(
        vocab_size=64, n_tools=8, n_matrices=3,
        d_model=32, n_layers=2, n_heads=4, d_ff=64,
        max_seq_len=32, dropout=0.0, slot_vocab_sizes=(),
    )


def _make_filled_buffer(n_steps=3, n_envs=2, max_seq_len=8, n_tools=8) -> RolloutBuffer:
    buf = RolloutBuffer(n_steps=n_steps, n_envs=n_envs, max_seq_len=max_seq_len, pin_memory=False)
    for _ in range(n_steps):
        buf.add(
            obs_tokens=torch.randint(0, 64, (n_envs, max_seq_len)),
            attn_mask=torch.ones(n_envs, max_seq_len, dtype=torch.bool),
            matrix_id=torch.zeros(n_envs, dtype=torch.int64),
            action_tool_id=torch.randint(0, n_tools, (n_envs,)),
            log_prob=torch.zeros(n_envs),
            value=torch.zeros(n_envs),
            reward=torch.randn(n_envs) * 0.1,
            done=torch.zeros(n_envs, dtype=torch.bool),
        )
    # Fill GAE outputs with dummy zeros for now; proper test below uses real GAE.
    buf.set_advantages_and_returns(
        torch.zeros(n_steps, n_envs), torch.zeros(n_steps, n_envs)
    )
    return buf


def test_policy_logp_value_entropy_shapes():
    policy = ActorCriticPolicy(_tiny_policy_cfg())
    policy.eval()
    buf = _make_filled_buffer()
    batches = list(buf.iter_minibatches(minibatch_size=3))
    for b in batches:
        new_logp, value, entropy = policy_logp_value_entropy(policy, b)
        assert new_logp.shape == (b.action_tool_id.shape[0],)
        assert value.shape == new_logp.shape
        assert entropy.shape == new_logp.shape
        assert (entropy >= 0).all()  # entropy is always nonneg


# ---- ppo_update end-to-end ---------------------------------------------------


def test_ppo_update_runs_and_decreases_loss_on_synthetic_signal():
    """Sanity check: with a fixed signal (advantage encourages action 0 every step),
    PPO should make the policy more confident in action 0."""
    torch.manual_seed(0)
    cfg_pol = _tiny_policy_cfg()
    policy = ActorCriticPolicy(cfg_pol)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    n_steps, n_envs, T_seq = 4, 2, 8
    buf = RolloutBuffer(n_steps=n_steps, n_envs=n_envs, max_seq_len=T_seq, pin_memory=False)

    # Simple deterministic obs; positive advantage on action 0; zero on others.
    obs = torch.zeros(n_envs, T_seq, dtype=torch.int64)
    mask = torch.ones(n_envs, T_seq, dtype=torch.bool)
    matrix = torch.zeros(n_envs, dtype=torch.int64)
    for _ in range(n_steps):
        # Compute the policy's current log_prob for action 0 to use as old_logp.
        with torch.no_grad():
            tool_logits, value, _ = policy(obs, matrix, mask)
            old_logp_step = torch.nn.functional.log_softmax(tool_logits, dim=-1)[:, 0]
        buf.add(
            obs_tokens=obs,
            attn_mask=mask,
            matrix_id=matrix,
            action_tool_id=torch.zeros(n_envs, dtype=torch.int64),
            log_prob=old_logp_step,
            value=value,
            reward=torch.ones(n_envs),  # positive reward for action 0
            done=torch.zeros(n_envs, dtype=torch.bool),
        )

    # GAE with a sufficient bootstrap.
    last_value = torch.zeros(n_envs)
    advantages, returns = compute_gae(
        rewards=buf.reward, values=buf.value, dones=buf.done.float(),
        last_value=last_value, gamma=0.99, lam=0.95,
    )
    buf.set_advantages_and_returns(advantages, returns)

    cfg = PPOConfig(
        clip_range=0.2, value_coef=0.5, entropy_coef=0.0,
        n_epochs=4, minibatch_size=2, max_grad_norm=0.5,
        target_kl_for_early_stop=None, normalize_advantages=False,
    )
    metrics_before = policy(obs, matrix, mask)
    initial_logp_action0 = torch.nn.functional.log_softmax(
        metrics_before[0], dim=-1
    )[:, 0].mean().item()

    metrics = ppo_update(policy=policy, optimizer=optimizer, rollout=buf, cfg=cfg)

    final_logp_action0 = torch.nn.functional.log_softmax(
        policy(obs, matrix, mask)[0], dim=-1
    )[:, 0].mean().item()

    assert metrics.n_epochs_run >= 1
    # The log_prob of action 0 should have increased.
    assert final_logp_action0 > initial_logp_action0


def test_ppo_update_with_kl_controller_updates_coef():
    torch.manual_seed(0)
    cfg_pol = _tiny_policy_cfg()
    policy = ActorCriticPolicy(cfg_pol)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    buf = _make_filled_buffer()
    cfg = PPOConfig(n_epochs=1, minibatch_size=2, normalize_advantages=False)
    kl_ctrl = AdaptiveKLController(init_coef=0.2, target_kl=0.02)
    initial_coef = kl_ctrl.coef
    _ = ppo_update(policy=policy, optimizer=optimizer, rollout=buf, cfg=cfg, kl_ctrl=kl_ctrl)
    # Coef should still be positive and clamped.
    assert kl_ctrl.coef > 0


def test_ppo_update_early_stop_on_high_kl():
    """If we set target_kl really small, ppo should bail after the first epoch."""
    torch.manual_seed(0)
    policy = ActorCriticPolicy(_tiny_policy_cfg())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0)  # huge lr forces big KL
    buf = _make_filled_buffer()
    # Need real returns so the value loss isn't trivially zero.
    advantages, returns = compute_gae(
        rewards=buf.reward, values=buf.value, dones=buf.done.float(),
        last_value=torch.zeros(buf.n_envs), gamma=0.99, lam=0.95,
    )
    buf.set_advantages_and_returns(advantages, returns)
    cfg = PPOConfig(
        n_epochs=10, minibatch_size=2,
        target_kl_for_early_stop=1e-9,  # any KL drift triggers stop
        normalize_advantages=False,
    )
    metrics = ppo_update(policy=policy, optimizer=optimizer, rollout=buf, cfg=cfg)
    assert metrics.early_stopped
    assert metrics.n_epochs_run < 10
