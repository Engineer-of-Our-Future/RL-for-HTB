"""Evaluation runner (PLAN.md Phase 11).

Drives a policy through N episodes per env, aggregates ATT&CK coverage and
outcome metrics into a SuiteResult.

Works on the stub env today (no Kali required) so we have a regression
fixture for the trainer; will work on the real Phase 4 env unchanged once
that lands.
"""

from __future__ import annotations

from typing import Callable

import torch

from htbrl.data.encode_state import Turn, encode_episode_window
from htbrl.env.base import Action, PentestEnv, StepInfo
from htbrl.eval.metrics import EpisodeResult, SuiteResult, killchain_depth_from_steps
from htbrl.model.policy import ActorCriticPolicy
from htbrl.tokenizer.bpe import ByteLevelBPE


_MATRIX_ID = {"enterprise": 0, "mobile": 1, "ics": 2}


def run_episode(
    env: PentestEnv,
    policy: ActorCriticPolicy,
    tokenizer: ByteLevelBPE,
    *,
    max_seq_len: int,
    history_window: int = 8,
    max_steps: int = 50,
    target_id: str = "unknown",
    deterministic: bool = True,
    device: torch.device | str = "cpu",
) -> EpisodeResult:
    """Roll one episode of `policy` against `env` and return per-episode metrics."""
    device = torch.device(device)
    policy.eval()

    obs = env.reset()
    history: list[Turn] = []
    result = EpisodeResult(
        matrix=env.matrix,
        target_id=target_id,
        total_reward=0.0,
        n_steps=0,
    )
    matrix_id_t = torch.tensor([_MATRIX_ID[env.matrix]], dtype=torch.int64, device=device)

    for _ in range(max_steps):
        ids, mask = encode_episode_window(
            tokenizer, env.matrix, history, max_seq_len=max_seq_len
        )
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            tool_logits, _, _ = policy(ids, matrix_id_t, mask)
        if deterministic:
            tool_id_int = int(tool_logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(tool_logits, dim=-1)
            tool_id_int = int(torch.multinomial(probs, num_samples=1).item())

        action = Action(tool_id=tool_id_int)
        next_obs, reward, done, info = env.step(action)
        result.step_infos.append(info)
        result.total_reward += reward
        result.n_steps += 1
        result.tools_used.add(tool_id_int)
        result.techniques_attempted.update(info.techniques_attempted)
        result.techniques_succeeded.update(info.techniques_succeeded)
        if info.techniques_succeeded:
            result.tactics_completed.update(info.tactic_ids)

        # Outcome flags - we infer from info.extras and parser-id heuristics on
        # the stub; the real env will set these directly via info.extras.
        extras = info.extras or {}
        if extras.get("foothold"):
            result.foothold = True
        if extras.get("user_flag"):
            result.user_flag = True
        if extras.get("root_flag"):
            result.root_flag = True

        history.append(
            Turn(
                obs_text=obs.obs_text,
                action_tool_name=str(tool_id_int),
                action_render=info.rendered_command,
                reward=reward,
            )
        )
        history = history[-history_window:]
        obs = next_obs

        if done:
            break

    result.killchain_depth = killchain_depth_from_steps(result.step_infos)
    return result


def run_suite(
    env_factories: list[Callable[[], PentestEnv]],
    policy: ActorCriticPolicy,
    tokenizer: ByteLevelBPE,
    *,
    max_seq_len: int,
    n_episodes_per_env: int = 5,
    max_steps_per_episode: int = 50,
    history_window: int = 8,
    deterministic: bool = True,
    device: torch.device | str = "cpu",
) -> SuiteResult:
    """Iterate over env factories x episodes, returning a flat SuiteResult."""
    suite = SuiteResult()
    for ef_idx, ef in enumerate(env_factories):
        for ep_idx in range(n_episodes_per_env):
            env = ef()
            try:
                target_id = f"env_{ef_idx}_ep_{ep_idx}"
                ep = run_episode(
                    env=env,
                    policy=policy,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    history_window=history_window,
                    max_steps=max_steps_per_episode,
                    target_id=target_id,
                    deterministic=deterministic,
                    device=device,
                )
                suite.episodes.append(ep)
            finally:
                env.close()
    return suite
