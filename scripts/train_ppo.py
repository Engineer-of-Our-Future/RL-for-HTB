"""PPO training driver - composes env / policy / rollout / GAE / update / checkpoint.

Drives a complete PPO training run end-to-end. Designed to work against
either the stub env (no Kali required, useful for trainer validation) or
the real HTBEnv (requires WSL Kali setup - see README).

Usage examples:

    # Smoke run on the stub env (fast, cpu-only):
    python scripts/train_ppo.py --env-type stub --total-rollouts 4 \\
        --d-model 32 --n-layers 2 --n-envs 2 --n-steps 8 --device cpu \\
        --output runs/smoke

    # Real run against WSL Kali:
    python scripts/train_ppo.py --env-type htb \\
        --kali-host htbrl@127.0.0.1:2222 \\
        --kali-key ~/.ssh/htbrl_kali \\
        --allowlist-cidr 127.0.0.0/8 \\
        --total-rollouts 100 --output runs/htb-v1

This is intentionally argparse-only (not Hydra) for the moment - the Hydra
configs in `configs/` are still consumed by training notebooks and the
upcoming experiment-launcher script. argparse keeps the CLI dependency-free
for ad-hoc use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from htbrl.algo.gae import compute_gae
from htbrl.algo.kl_ctrl import AdaptiveKLController
from htbrl.algo.normalizer import RunningStats
from htbrl.algo.ppo import PPOConfig, ppo_update
from htbrl.algo.rlhf import CompositeRewardConfig, composite_reward
from htbrl.data.rollout_buffer import RolloutBuffer
from htbrl.env.base import PentestEnv
from htbrl.env.rollout_runner import collect_rollout
from htbrl.env.stub_env import StubPentestEnv
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    # Env
    p.add_argument("--env-type", choices=["stub", "htb"], default="stub")
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=128, help="rollout horizon per env")
    p.add_argument("--max-episode-steps", type=int, default=20)
    # HTB-specific
    p.add_argument("--kali-host", default=None, help="user@host[:port] for HTBEnv")
    p.add_argument("--kali-key", default=None, help="path to SSH private key")
    p.add_argument("--allowlist-cidr", action="append", default=[],
                   help="repeat to allow multiple CIDRs (e.g. 127.0.0.0/8)")

    # Model
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)

    # Optimizer
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # PPO
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)

    # KL controller
    p.add_argument("--kl-target", type=float, default=0.02)
    p.add_argument("--kl-init-coef", type=float, default=0.0,
                   help="0 disables the KL penalty entirely (use ref policy with kl-init-coef>0 to enable)")

    # Outer loop
    p.add_argument("--total-rollouts", type=int, default=20)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=10)

    # IO
    p.add_argument("--output", type=Path, required=True, help="run dir for checkpoints + log.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    return p


# ---- env factory -------------------------------------------------------------


def _build_env_factory(args, vocab) -> list[PentestEnv]:
    if args.env_type == "stub":
        return [
            StubPentestEnv(vocab, max_steps=args.max_episode_steps, seed=args.seed + i)
            for i in range(args.n_envs)
        ]

    # htb
    from htbrl.env.htb_env import HTBEnv
    from htbrl.env.ssh_session import SSHCredentials

    if not args.kali_host:
        raise SystemExit("--kali-host is required for --env-type htb")
    user_at_host, _, port_str = args.kali_host.partition(":")
    user, _, host = user_at_host.partition("@")
    creds = SSHCredentials(
        host=host,
        port=int(port_str) if port_str else 22,
        user=user,
        identity_file=os.path.expanduser(args.kali_key) if args.kali_key else None,
        connect_timeout_seconds=10.0,
    )
    if not args.allowlist_cidr:
        raise SystemExit("--allowlist-cidr is required at least once for HTBEnv safety")
    return [
        HTBEnv(
            vocab=vocab,
            ssh_creds=creds,
            allowlist_cidrs=args.allowlist_cidr,
            max_steps=args.max_episode_steps,
            episode_target_id=f"htb-env-{i}",
        )
        for i in range(args.n_envs)
    ]


# ---- main loop ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    logfile = (args.output / "log.jsonl").open("w", encoding="utf-8")

    # Build everything
    vocab = load_registry()
    tokenizer = ByteLevelBPE.initialize()  # untrained; bytes pass-through fallback

    cfg = PolicyConfig(
        vocab_size=tokenizer.vocab_size,
        n_tools=vocab.n_tools,
        n_matrices=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        slot_vocab_sizes=(),
    )
    policy = ActorCriticPolicy(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    envs = _build_env_factory(args, vocab)
    buffer = RolloutBuffer(
        n_steps=args.n_steps, n_envs=args.n_envs, max_seq_len=args.max_seq_len,
        pin_memory=False,
    )

    ppo_cfg = PPOConfig(
        clip_range=args.clip_range, value_clip_range=args.clip_range,
        value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs, minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm, target_kl_for_early_stop=args.target_kl,
        normalize_advantages=True,
    )
    rlhf_cfg = CompositeRewardConfig()
    reward_norm = RunningStats()
    kl_ctrl = (
        AdaptiveKLController(init_coef=args.kl_init_coef, target_kl=args.kl_target)
        if args.kl_init_coef > 0 else None
    )

    print(f"[train] policy params = {policy.n_parameters():,}")
    print(f"[train] vocab tools   = {vocab.n_tools}")
    print(f"[train] env_type      = {args.env_type}")
    print(f"[train] device        = {args.device}")

    total_env_steps = 0

    for rollout_idx in range(args.total_rollouts):
        t0 = time.time()
        rollout_metrics = collect_rollout(
            envs=envs, policy=policy, tokenizer=tokenizer,
            buffer=buffer, max_seq_len=args.max_seq_len, device=args.device,
        )
        t_rollout = time.time() - t0

        # Reward normalization (PLAN.md Phase 7) - update running stats then scale.
        reward_norm.update(buffer.reward)
        if reward_norm.count >= 2:
            std = float(reward_norm.std)
            if std > 1e-6:
                buffer.reward.div_(std)

        # Composite reward path: env_reward only for now (RM disabled until
        # phase 8 trainer ships a real checkpoint). beta/alpha schedules logged
        # for telemetry parity with the RLHF target shape.
        composite, comp_scalars = composite_reward(
            env_reward=buffer.reward,
            rm_score=None, rnd_bonus=None,
            env_steps_seen=total_env_steps,
            cfg=rlhf_cfg,
        )
        buffer.reward.copy_(composite)

        # GAE
        last_value = torch.zeros(args.n_envs)
        adv, ret = compute_gae(
            rewards=buffer.reward, values=buffer.value, dones=buffer.done.float(),
            last_value=last_value, gamma=args.gamma, lam=args.lam,
        )
        buffer.set_advantages_and_returns(adv, ret)

        t1 = time.time()
        ppo_metrics = ppo_update(
            policy=policy, optimizer=optimizer, rollout=buffer,
            cfg=ppo_cfg, kl_ctrl=kl_ctrl, ref_policy=None, device=args.device,
        )
        t_update = time.time() - t1

        steps_in_rollout = args.n_steps * args.n_envs
        total_env_steps += steps_in_rollout

        # Log
        log_entry = {
            "rollout": rollout_idx,
            "env_steps": total_env_steps,
            "t_rollout_s": round(t_rollout, 2),
            "t_update_s": round(t_update, 2),
            "loss_total": ppo_metrics.loss_total,
            "loss_policy": ppo_metrics.loss_policy,
            "loss_value": ppo_metrics.loss_value,
            "approx_kl": ppo_metrics.approx_kl,
            "clip_fraction": ppo_metrics.clip_fraction,
            "explained_variance": ppo_metrics.explained_variance,
            "n_episodes": len(rollout_metrics["episode_returns"]),
            "avg_episode_return": (
                sum(rollout_metrics["episode_returns"]) / len(rollout_metrics["episode_returns"])
                if rollout_metrics["episode_returns"] else None
            ),
            "techniques_attempted": len(rollout_metrics["techniques_attempted"]),
            "techniques_succeeded": len(rollout_metrics["techniques_succeeded"]),
            **comp_scalars,
        }
        logfile.write(json.dumps(log_entry) + "\n")
        logfile.flush()

        if rollout_idx % args.log_every == 0:
            ep_ret = log_entry["avg_episode_return"]
            ep_str = f"{ep_ret:+.3f}" if ep_ret is not None else "n/a"
            print(
                f"[train] rollout {rollout_idx:>4}  "
                f"steps={total_env_steps:>7}  "
                f"loss={ppo_metrics.loss_total:+.3f}  "
                f"kl={ppo_metrics.approx_kl:.4f}  "
                f"ev={ppo_metrics.explained_variance:+.2f}  "
                f"ep_ret={ep_str}  "
                f"techs_succ={len(rollout_metrics['techniques_succeeded'])}  "
                f"({t_rollout:.1f}s+{t_update:.1f}s)"
            )

        if (rollout_idx + 1) % args.checkpoint_every == 0:
            ckpt = args.output / f"ckpt-{rollout_idx+1:05d}.pt"
            torch.save(
                {
                    "model": policy.state_dict(),
                    "config": asdict(cfg),
                    "rollout_idx": rollout_idx + 1,
                    "env_steps": total_env_steps,
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"[train] saved {ckpt.name}")

    # Final checkpoint
    final = args.output / "ckpt-final.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "config": asdict(cfg),
            "rollout_idx": args.total_rollouts,
            "env_steps": total_env_steps,
            "args": vars(args),
        },
        final,
    )
    print(f"[train] done. final = {final}")
    logfile.close()
    for env in envs:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
