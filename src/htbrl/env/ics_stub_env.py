"""ICS-matrix stub env (PLAN.md Phase 16 prep).

Synthetic OT/ICS progression. Real ICSEnv (Phase 16: OpenPLC + ConPot +
GRFICSv2 lab) lands in v0.3; this stub lets the trainer exercise the
ICS-matrix path end-to-end.

Stages (loosely tracking ATT&CK-ICS tactics):
- scan      (TA0102 Discovery)            - plcscan / s7scan / enip-list-id
- enum      (TA0102 Discovery)            - read holding registers
- read      (TA0100 Collection)           - dump tags
- impair    (TA0106 Impair Process Control) - modbus_write_coil
- impact    (TA0105 Impact)               - persistent process disruption

Same "favored tool per stage" mechanic as the other stub envs. Stages here
deliberately model the OT kill-chain order the user described
(scan -> read -> impair) since process-control disruption is the canonical
ICS-engagement objective.
"""

from __future__ import annotations

import random

from htbrl.env.base import Action, Observation, PentestEnv, StepInfo
from htbrl.tools.loader import ActionVocabulary


class ICSStubEnv(PentestEnv):
    matrix = "ics"

    _STAGES = ["scan", "enum", "read", "impair", "impact"]
    _STAGE_REWARD = {
        "scan": 0.1,
        "enum": 0.2,
        "read": 0.5,
        "impair": 1.0,
        "impact": 2.0,
    }
    _STAGE_TACTIC = {
        "scan": "TA0102",
        "enum": "TA0102",
        "read": "TA0100",
        "impair": "TA0106",
        "impact": "TA0105",
    }

    def __init__(
        self,
        vocab: ActionVocabulary,
        max_steps: int = 25,
        seed: int | None = None,
    ) -> None:
        self.vocab = vocab
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._stage = 0
        self._step = 0
        self._closed = False
        self._favored: list[int] = []

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        self._stage = 0
        self._step = 0
        self._favored = [
            self._rng.randrange(self.vocab.n_tools) for _ in self._STAGES
        ]
        return Observation(
            obs_text="<ics-stub>: episode reset; PLC subnet idle.",
            parsed_features={"stage": "scan", "favored_first_tool": self._favored[0]},
            last_reward=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, StepInfo]:
        if self._closed:
            raise RuntimeError("step() on closed env")
        self._step += 1
        if not (0 <= action.tool_id < self.vocab.n_tools):
            raise ValueError(f"tool_id {action.tool_id} out of range")
        tool_def = self.vocab.tools[action.tool_id]
        rendered = f"<ics-stub-render>:{tool_def.name}:{action.slots!r}"
        reward = -0.01
        info = StepInfo(
            techniques_attempted=list(tool_def.attack.techniques),
            tactic_ids=list(tool_def.attack.tactics),
            rendered_command=rendered,
            parser_id=tool_def.output_parser_id,
        )
        if self._stage < len(self._STAGES) and action.tool_id == self._favored[self._stage]:
            stage = self._STAGES[self._stage]
            reward += self._STAGE_REWARD[stage]
            info.techniques_succeeded = list(tool_def.attack.techniques)
            info.tactic_ids = sorted(set(info.tactic_ids + [self._STAGE_TACTIC[stage]]))
            self._stage += 1

        done = self._stage >= len(self._STAGES) or self._step >= self.max_steps
        next_text = (
            f"<ics-stub>: step={self._step} stage={self._stage}/{len(self._STAGES)} "
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
