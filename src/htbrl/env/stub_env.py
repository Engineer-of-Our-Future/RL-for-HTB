"""Stub pentest env — synthetic, runs anywhere, no Kali needed.

Purpose: validate the *training* code (rollout collection + PPO update loop)
end-to-end without an SSH session into Kali. The stub mimics a tiny pentest
where the agent has to chain through a fixed sequence of "favored" tools
(picked uniformly at random per episode) to reach a synthetic flag.

Reward structure tries to mirror Phase 4's primitives so the trainer doesn't
encode any stub-specific logic:
- +0.1 for any "discovery"-tactic tool
- +0.5 for the user-shell milestone tool
- +1.0 for the user-flag tool
- +1.5 for the root-shell tool
- +2.0 for the root-flag tool
- -0.01 per command (efficiency penalty)

Useful for:
- Smoke-testing rollout_runner + PPO update loop without external services
- Performance benchmarking the trainer (deterministic, fast)
- A regression fixture: any change to the training stack must keep the agent
  learning to chain at least to the foothold milestone in a small budget
"""

from __future__ import annotations

import random
from typing import Any

from htbrl.env.base import Action, Observation, PentestEnv, StepInfo
from htbrl.tools.loader import ActionVocabulary


class StubPentestEnv(PentestEnv):
    """A 5-stage synthetic pentest. Always Enterprise matrix."""

    matrix = "enterprise"

    # Stage labels match the canonical pentest "kill chain" milestones.
    _STAGES = ["recon", "discovery", "user_shell", "user_flag", "root_shell", "root_flag"]
    _STAGE_REWARD = {
        "recon": 0.1,
        "discovery": 0.1,
        "user_shell": 0.5,
        "user_flag": 1.0,
        "root_shell": 1.5,
        "root_flag": 2.0,
    }
    # Bonus tactic associations per stage (used so info.tactic_ids reflects
    # what the agent achieved - same shape the real env will emit).
    _STAGE_TACTIC = {
        "recon": "TA0043",
        "discovery": "TA0007",
        "user_shell": "TA0001",
        "user_flag": "TA0009",
        "root_shell": "TA0004",
        "root_flag": "TA0009",
    }

    def __init__(
        self,
        vocab: ActionVocabulary,
        max_steps: int = 20,
        seed: int | None = None,
    ) -> None:
        self.vocab = vocab
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._stage = 0
        self._step = 0
        self._closed = False
        # Pick a "favored tool" per stage at episode start. The agent has to
        # *find* which tool (chosen uniformly from registry) advances each stage.
        self._favored: list[int] = []

    # ---- lifecycle ----------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._stage = 0
        self._step = 0
        self._favored = [self._rng.randrange(self.vocab.n_tools) for _ in self._STAGES]
        return Observation(
            obs_text="<stub>: episode reset; nothing scanned yet.",
            parsed_features={"stage": "recon", "favored_first_tool": self._favored[0]},
            last_reward=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, StepInfo]:
        if self._closed:
            raise RuntimeError("step() on closed env")
        self._step += 1

        # Validate tool id.
        if not (0 <= action.tool_id < self.vocab.n_tools):
            raise ValueError(f"tool_id {action.tool_id} out of range")

        tool_def = self.vocab.tools[action.tool_id]
        rendered = f"<stub-render>:{tool_def.name}:{action.slots!r}"

        # Determine reward: did the agent pick the favored tool for the current stage?
        reward = -0.01  # per-command penalty
        info = StepInfo(
            techniques_attempted=list(tool_def.attack.techniques),
            tactic_ids=list(tool_def.attack.tactics),
            rendered_command=rendered,
            parser_id=tool_def.output_parser_id,
        )

        if self._stage < len(self._STAGES) and action.tool_id == self._favored[self._stage]:
            stage_name = self._STAGES[self._stage]
            reward += self._STAGE_REWARD[stage_name]
            info.techniques_succeeded = list(tool_def.attack.techniques)
            info.tactic_ids = sorted(set(info.tactic_ids + [self._STAGE_TACTIC[stage_name]]))
            self._stage += 1

        # Done conditions
        done = self._stage >= len(self._STAGES) or self._step >= self.max_steps

        next_text = (
            f"<stub>: step={self._step} stage={self._stage}/{len(self._STAGES)} "
            f"reward={reward:+.3f}"
        )
        next_obs = Observation(
            obs_text=next_text,
            parsed_features={"stage_idx": self._stage, "step": self._step},
            last_reward=reward,
        )
        if not done and self._stage < len(self._STAGES):
            next_obs.parsed_features["favored_next_tool"] = self._favored[self._stage]
        return next_obs, reward, done, info

    def close(self) -> None:
        self._closed = True
