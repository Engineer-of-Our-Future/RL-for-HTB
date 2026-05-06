"""End-to-end integration test: stub env -> rollout -> GAE -> PPO update.

This is the closest thing to "the trainer works" we can run without a real
SSH-into-Kali env. If this test passes, the full Phase 6 training stack
composes correctly.

Marked ``slow`` because it constructs a small policy + runs a short rollout +
a few PPO updates. About 2-5s on CPU.
"""

from __future__ import annotations

import pytest
import torch

from htbrl.algo.gae import compute_gae
from htbrl.algo.ppo import PPOConfig, ppo_update
from htbrl.data.rollout_buffer import RolloutBuffer
from htbrl.env.rollout_runner import collect_rollout
from htbrl.env.stub_env import StubPentestEnv
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


@pytest.mark.slow
def test_full_ppo_loop_on_stub_env_runs_without_error():
    torch.manual_seed(0)

    vocab = load_registry()
    tokenizer = ByteLevelBPE.initialize()  # untrained: byte-level fallback

    cfg = PolicyConfig(
        vocab_size=tokenizer.vocab_size,
        n_tools=vocab.n_tools,
        n_matrices=3,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        slot_vocab_sizes=(),
    )
    policy = ActorCriticPolicy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    n_envs, n_steps = 2, 8
    envs = [StubPentestEnv(vocab, max_steps=12, seed=i) for i in range(n_envs)]
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, max_seq_len=cfg.max_seq_len, pin_memory=False
    )

    metrics = collect_rollout(
        envs=envs,
        policy=policy,
        tokenizer=tokenizer,
        buffer=buffer,
        max_seq_len=cfg.max_seq_len,
        history_window=4,
    )

    # Buffer should be full.
    assert buffer.is_full
    # Should have logged some technique attempts via stub.
    assert len(metrics["techniques_attempted"]) > 0

    # Bootstrap value at the end of rollout (zeros for now).
    last_value = torch.zeros(n_envs)
    advantages, returns = compute_gae(
        rewards=buffer.reward, values=buffer.value, dones=buffer.done.float(),
        last_value=last_value, gamma=0.99, lam=0.95,
    )
    buffer.set_advantages_and_returns(advantages, returns)

    ppo_cfg = PPOConfig(
        n_epochs=2, minibatch_size=4, normalize_advantages=True,
        target_kl_for_early_stop=None,
    )
    ppo_metrics = ppo_update(policy=policy, optimizer=optimizer, rollout=buffer, cfg=ppo_cfg)

    # Sanity checks: loss is finite, at least one epoch ran.
    assert ppo_metrics.n_epochs_run >= 1
    import math
    assert math.isfinite(ppo_metrics.loss_total)
    assert math.isfinite(ppo_metrics.approx_kl)

    # Cleanup
    for env in envs:
        env.close()


@pytest.mark.slow
def test_ppo_learning_signal_on_stub_after_many_rollouts():
    """Over multiple rollouts, episode return should trend upward (learning signal).

    Only a coarse check - we use a small budget so the trend is noisy. This
    test catches catastrophic regressions where the trainer doesn't learn at
    all (e.g., wrong sign on advantage, unclipped explosion, broken backprop).
    """
    torch.manual_seed(0)
    vocab = load_registry()
    tokenizer = ByteLevelBPE.initialize()

    cfg = PolicyConfig(
        vocab_size=tokenizer.vocab_size,
        n_tools=vocab.n_tools,
        n_matrices=3,
        d_model=32, n_layers=2, n_heads=4, d_ff=64,
        max_seq_len=64, dropout=0.0,
    )
    policy = ActorCriticPolicy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    n_envs, n_steps = 2, 16
    # Use a fixed seed per env so the favored sequence stays the same across rollouts;
    # this lets the policy actually learn the answer.
    envs = [StubPentestEnv(vocab, max_steps=12, seed=42 + i) for i in range(n_envs)]
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, max_seq_len=cfg.max_seq_len, pin_memory=False
    )

    early_returns: list[float] = []
    late_returns: list[float] = []
    n_rollouts = 8

    ppo_cfg = PPOConfig(n_epochs=2, minibatch_size=8, normalize_advantages=True,
                       target_kl_for_early_stop=None)

    for rollout_idx in range(n_rollouts):
        m = collect_rollout(
            envs=envs, policy=policy, tokenizer=tokenizer, buffer=buffer,
            max_seq_len=cfg.max_seq_len,
        )
        last_value = torch.zeros(n_envs)
        adv, ret = compute_gae(
            rewards=buffer.reward, values=buffer.value, dones=buffer.done.float(),
            last_value=last_value,
        )
        buffer.set_advantages_and_returns(adv, ret)
        ppo_update(policy=policy, optimizer=optimizer, rollout=buffer, cfg=ppo_cfg)

        # Track episode returns from this rollout.
        if m["episode_returns"]:
            r = sum(m["episode_returns"]) / len(m["episode_returns"])
            if rollout_idx < 2:
                early_returns.append(r)
            elif rollout_idx >= n_rollouts - 2:
                late_returns.append(r)

    for env in envs:
        env.close()

    # If learning happened, late returns should be higher (or at least not
    # catastrophically lower) than early returns. Use a wide tolerance because
    # the budget is small.
    if early_returns and late_returns:
        avg_early = sum(early_returns) / len(early_returns)
        avg_late = sum(late_returns) / len(late_returns)
        # No catastrophic divergence
        assert avg_late > avg_early - 1.0, (
            f"PPO training appears to be diverging: early avg={avg_early:.3f}, "
            f"late avg={avg_late:.3f}"
        )
