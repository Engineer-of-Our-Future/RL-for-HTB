"""Mobile-matrix stub env (PLAN.md Phase 13 prep).

Synthetic Android-like progression. Mirrors what a real
``MobileEnv`` would expose (Phase 13: Genymotion / AVD via ADB) so the trainer
+ rollout runner work end-to-end on the Mobile matrix without browser binaries
or emulator setup.

Stages (loosely tracking ATT&CK-Mobile tactics):
- recon       (TA0032 Discovery)         - adb shell pm list / mobsf static
- install     (TA0027 Initial Access)    - apk install / drozer launch
- execute     (TA0041 Execution)         - frida hook / objection startup
- harvest     (TA0031 Credential Access) - keystore dump / app-data dump
- exfil       (TA0035 Collection / TA0037 Exfiltration)

Just like StubPentestEnv, the agent has to find the "favored tool" per stage
(picked uniformly at random per episode) to advance. Reward magnitudes mirror
Phase 13's planned auto-rewards.
"""

from __future__ import annotations

import random

from htbrl.env.base import Action, Observation, PentestEnv, StepInfo
from htbrl.tools.loader import ActionVocabulary


class MobileStubEnv(PentestEnv):
    matrix = "mobile"

    _STAGES = ["recon", "install", "execute", "harvest", "exfil"]
    _STAGE_REWARD = {
        "recon": 0.1,
        "install": 0.5,    # foothold equivalent: app instrumented
        "execute": 0.7,
        "harvest": 1.0,    # user-flag analog
        "exfil": 1.5,      # root-flag analog
    }
    _STAGE_TACTIC = {
        "recon": "TA0032",
        "install": "TA0027",
        "execute": "TA0041",
        "harvest": "TA0031",
        "exfil": "TA0035",
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
            obs_text="<mobile-stub>: episode reset; emulator idle.",
            parsed_features={"stage": "recon", "favored_first_tool": self._favored[0]},
            last_reward=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, StepInfo]:
        if self._closed:
            raise RuntimeError("step() on closed env")
        self._step += 1
        if not (0 <= action.tool_id < self.vocab.n_tools):
            raise ValueError(f"tool_id {action.tool_id} out of range")
        tool_def = self.vocab.tools[action.tool_id]
        rendered = f"<mobile-stub-render>:{tool_def.name}:{action.slots!r}"
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
            f"<mobile-stub>: step={self._step} stage={self._stage}/{len(self._STAGES)} "
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
