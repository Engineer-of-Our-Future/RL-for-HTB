"""Plain dataclasses describing HTB Academy structure.

We deliberately model only what we need for autonomous progression - module
listing, section content, questions, sandbox credentials, progress state.
Anything richer (badges, certificates, achievements) is intentionally out of
scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    """Question shapes we know how to handle."""

    TEXT = "text"                # short free-text answer
    MULTIPLE_CHOICE = "mc"       # one option from a list
    FLAG = "flag"                # answer obtained from sandbox (cmd output / file)
    UNSUPPORTED = "unsupported"  # question that requires LLM-style reasoning; we skip


@dataclass
class AcademyQuestion:
    id: str
    prompt: str
    type: QuestionType
    multiple_choice_options: list[str] = field(default_factory=list)
    # Hints from the page that help the answerer (e.g., a snippet of code or a
    # sandbox-cmd template the section authors expect the student to run).
    hints: list[str] = field(default_factory=list)
    points: int = 0
    # Reward metadata as shown on the academy question card (e.g. green cube
    # +5 / purple HP +20). Useful for the orchestrator to prioritize
    # high-value questions and for the demo writer to scale per-turn rewards.
    cubes_reward: int = 0
    hp_reward: int = 0

    def __post_init__(self) -> None:
        if self.type == QuestionType.MULTIPLE_CHOICE and not self.multiple_choice_options:
            raise ValueError(f"MC question {self.id!r} has no options")


@dataclass
class AcademySandbox:
    """Per-module sandbox VM, exposed via SSH inside the module page."""

    host: str
    user: str
    port: int = 22
    password: str | None = None
    key_text: str | None = None  # raw private key string (rare in academy)
    notes: str = ""              # free-text from the module about how the sandbox is used


@dataclass
class AcademySection:
    """One numbered subsection inside a module page (e.g. 'Section 11 / 22')."""

    id: str
    title: str
    body_text: str               # markdown / plaintext extracted from HTML
    questions: list[AcademyQuestion] = field(default_factory=list)
    sandbox: AcademySandbox | None = None
    # Fenced code blocks the section author included; used as command hints
    # by the answerer's "run-this-and-paste-output" heuristic.
    code_blocks: list[str] = field(default_factory=list)
    # Inline `code` spans extracted from the body. Often contain the literal
    # answer to text questions (e.g. "the {up-to-date} command" -> answer is
    # "up-to-date"). The HeuristicAnswerer prefers these over arbitrary tails.
    inline_code: list[str] = field(default_factory=list)
    # Bulleted lists found in the body, ordered as they appear. The answerer
    # uses these for "How many X..." / "Which X..." questions.
    bullet_lists: list[list[str]] = field(default_factory=list)
    # 1-based section index inside its module ("11 / 22").
    section_index: int = 0
    section_total: int = 0
    # +HP reward shown on the section's "Mark Complete & Next" button.
    hp_reward: int = 0


@dataclass
class AcademyModule:
    """One academy module (e.g., 'Linux Fundamentals', 'Network Enumeration')."""

    id: str
    title: str
    tier: int                    # 0..3 typical
    cubes_to_unlock: int = 0     # cubes spent at module-start
    cubes_reward: int = 0        # cubes returned on full completion
    description: str = ""
    sections: list[AcademySection] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    estimated_minutes: int = 0
    completed: bool = False
    # Curriculum metadata (PLAN.md Phase 5b - path-aware ordering).
    # category in {"general", "offensive", "defensive", "other"}.
    # path_ids lists the academy paths this module belongs to (a single
    # module can be in multiple paths e.g. "Pre-Employability" + "Junior
    # Penetration Tester").
    category: str = "other"
    path_ids: list[str] = field(default_factory=list)
    # Cheat sheet attached to the module (HTB Academy modules expose a
    # markdown table at ``/api/v2/modules/<id>`` -> ``data.cheatsheet``).
    # Each row is a structured dict, e.g. ``{"command": "ls",
    # "description": "lists files in a directory"}``. Used by the answerer
    # as a high-precision command -> description lookup AND surfaced in
    # demos as one ``academy_cheat_sheet`` turn so BC training sees the
    # canonical reference next to the theory text.
    cheat_sheet: list[dict[str, str]] = field(default_factory=list)
    # Prelude / conclusion / takeaways copy from the modules API; these
    # carry module-level intro and learning objectives respectively. They
    # often state the LITERAL answers to summary questions and are
    # therefore valuable as theory context for the answerer.
    prelude: str = ""
    conclusion: str = ""
    takeaways: str = ""

    @property
    def all_questions(self) -> list[AcademyQuestion]:
        out: list[AcademyQuestion] = []
        for s in self.sections:
            out.extend(s.questions)
        return out


@dataclass
class AcademyAnswer:
    """One attempt at one question by the agent."""

    question_id: str
    answer_text: str
    confidence: float                # 0..1; >= 0.7 is "submit-confident"
    method: str                      # "heuristic_text" | "mc_match" | "sandbox_cmd" | "manual" | "skipped"
    rationale: str = ""              # short human-readable explanation
    sandbox_command: str | None = None  # if produced by running a command, the command we ran
    sandbox_output: str | None = None   # truncated output from the command


@dataclass
class ProgressState:
    """User-level academy progress."""

    user_id: str
    cubes_balance: int = 0
    completed_module_ids: list[str] = field(default_factory=list)
    in_progress_module_id: str | None = None
    # Modules the user could start right now (have prereqs + enough cubes).
    next_eligible_module_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_completed(self, module_id: str) -> bool:
        return module_id in self.completed_module_ids
