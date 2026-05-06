"""Generic rollout collection: env x policy -> filled RolloutBuffer.

Used by both PPO (Phase 6) and the stub-env integration tests. Knows about:
- The PentestEnv contract (env/base.py)
- The ActorCriticPolicy (model/policy.py)
- The RolloutBuffer (data/rollout_buffer.py)
- The state encoder (data/encode_state.py)

But it's deliberately narrow: it just runs ``env.step`` in a loop and
populates the buffer with the policy's chosen action + log_prob + value, plus
the env's reward/done. It does NOT do GAE - that's done after the rollout
completes by the PPO trainer.
"""

from __future__ import annotations

from typing import Sequence

import torch

from htbrl.data.encode_state import Turn, encode_episode_window
from htbrl.data.rollout_buffer import RolloutBuffer
from htbrl.env.base import Action, Observation, PentestEnv
from htbrl.model.policy import ActorCriticPolicy
from htbrl.tokenizer.bpe import ByteLevelBPE


_MATRIX_ID = {"enterprise": 0, "mobile": 1, "ics": 2}


def collect_rollout(
    *,
    envs: Sequence[PentestEnv],
    policy: ActorCriticPolicy,
    tokenizer: ByteLevelBPE,
    buffer: RolloutBuffer,
    max_seq_len: int,
    history_window: int = 8,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
) -> dict:
    """Collect ``buffer.n_steps`` steps across ``len(envs)`` parallel envs.

    Each env contributes one column to the buffer. Returns a metrics dict.

    Side effects:
    - Each env may be reset() if it returned done=True. Reset is automatic so
      the buffer always fills.
    - Buffer is reset() at entry, written to ``n_steps`` times, then has its
      advantages/returns slot left empty (caller must run GAE before update).
    """
    if len(envs) != buffer.n_envs:
        raise ValueError(
            f"got {len(envs)} envs but buffer expects n_envs={buffer.n_envs}"
        )
    device = torch.device(device)
    policy.eval()
    buffer.reset()

    # Per-env rolling-window history of Turns.
    histories: list[list[Turn]] = [[] for _ in envs]
    last_obs: list[Observation] = [env.reset() for env in envs]

    metrics = {
        "episode_returns": [],
        "episode_lengths": [],
        "techniques_attempted": set(),
        "techniques_succeeded": set(),
    }
    pending_returns = [0.0] * len(envs)
    pending_lengths = [0] * len(envs)

    for step in range(buffer.n_steps):
        # Build the encoded state batch.
        token_batch = torch.zeros(len(envs), max_seq_len, dtype=torch.int64)
        mask_batch = torch.zeros(len(envs), max_seq_len, dtype=torch.bool)
        matrix_batch = torch.zeros(len(envs), dtype=torch.int64)
        for i, env in enumerate(envs):
            ids, mask = encode_episode_window(
                tokenizer, env.matrix, histories[i], max_seq_len=max_seq_len
            )
            token_batch[i] = ids
            mask_batch[i] = mask
            matrix_batch[i] = _MATRIX_ID[env.matrix]

        token_batch = token_batch.to(device)
        mask_batch = mask_batch.to(device)
        matrix_batch = matrix_batch.to(device)

        # Sample tool from policy.
        tool_id, log_prob, value, _ = policy.sample_tool(
            token_batch, matrix_batch, mask_batch, temperature=temperature
        )

        # Step every env with its sampled tool.
        rewards = torch.zeros(len(envs), dtype=torch.float32)
        dones = torch.zeros(len(envs), dtype=torch.bool)
        for i, env in enumerate(envs):
            action = Action(tool_id=int(tool_id[i].item()))
            next_obs, r, done, info = env.step(action)
            rewards[i] = r
            dones[i] = done
            metrics["techniques_attempted"].update(info.techniques_attempted)
            metrics["techniques_succeeded"].update(info.techniques_succeeded)
            pending_returns[i] += r
            pending_lengths[i] += 1

            # Append the just-completed turn to history for next step's encoding.
            tool_name = policy.cfg.n_tools  # fallback unused; we use vocab via env
            histories[i].append(
                Turn(
                    obs_text=last_obs[i].obs_text,
                    action_tool_name=str(action.tool_id),  # name lookup happens in env wrapper later
                    action_render=info.rendered_command,
                    reward=r,
                )
            )
            histories[i] = histories[i][-history_window:]
            last_obs[i] = next_obs

            if done:
                metrics["episode_returns"].append(pending_returns[i])
                metrics["episode_lengths"].append(pending_lengths[i])
                pending_returns[i] = 0.0
                pending_lengths[i] = 0
                last_obs[i] = env.reset()
                histories[i] = []

        # Write into the buffer.
        buffer.add(
            obs_tokens=token_batch.cpu(),
            attn_mask=mask_batch.cpu(),
            matrix_id=matrix_batch.cpu(),
            action_tool_id=tool_id.cpu(),
            log_prob=log_prob.cpu(),
            value=value.cpu(),
            reward=rewards,
            done=dones,
        )

    metrics["techniques_attempted"] = sorted(metrics["techniques_attempted"])
    metrics["techniques_succeeded"] = sorted(metrics["techniques_succeeded"])
    return metrics
