"""Abstract base class for pentest environments.

Every concrete env (Phase 4 SSH-into-Kali, Phase 13 Android emulator, Phase 16
ICS lab) implements this same interface. The training code (PPO, BC, eval)
talks to this contract and never touches transport / tool details.

Why a custom base class instead of plain ``gymnasium.Env``: gymnasium's
observation space machinery is shaped for low-dimensional vector / image
observations, whereas our observations are *text* (parser output + raw
stdout). We keep gymnasium-style ``reset / step`` semantics but use a Python
dict for observation and emit ATT&CK technique tags via ``info["attack"]``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepInfo:
    """``info`` dict returned by ``step``. Plain dataclass for type safety."""

    techniques_attempted: list[str] = field(default_factory=list)
    techniques_succeeded: list[str] = field(default_factory=list)
    tactic_ids: list[str] = field(default_factory=list)
    timed_out: bool = False
    rendered_command: str = ""
    parser_id: str = "raw"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Structured action emitted by the policy. The env renders it before running."""

    tool_id: int
    slots: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Observation dict the env returns. Plain dataclass; the encoder turns it into tokens."""

    obs_text: str
    parsed_features: dict[str, Any] = field(default_factory=dict)
    last_reward: float = 0.0


class PentestEnv(abc.ABC):
    """Abstract base. Concrete envs override ``reset``, ``step``, ``close``."""

    matrix: str  # "enterprise" | "mobile" | "ics"

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Reset to a fresh episode and return the initial observation."""

    @abc.abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, StepInfo]:
        """Apply ``action`` and return ``(next_obs, reward, done, info)``.

        - ``next_obs`` is the observation the policy reads next.
        - ``reward`` is the scalar env reward (no learned RM here - that's
          composed in the PPO loop).
        - ``done`` indicates the episode terminated (terminal state, time-out,
          or detected reset condition).
        - ``info`` carries ATT&CK metadata + parser-specific extras.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any external resources (SSH session, emulator process, etc.)."""

    # Optional context-manager support so ``with PentestEnv() as e: ...`` works.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
