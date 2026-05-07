"""HTB Academy session abstraction.

``AcademySession`` is the contract every transport layer implements. Right now
we ship:

- ``MockAcademySession``: in-memory, deterministic, no network. Used by tests
  and dry-runs.
- *(future)* ``PlaywrightAcademySession``: real browser automation. Lives in a
  separate module so adding it doesn't pull Playwright into the test path.

Credentials are passed once at session creation and never logged. The session
never persists them to disk - if you want a credential cache, build it
separately and pass the values in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    ProgressState,
)


@dataclass
class AcademyCredentials:
    """User-provided HTB Academy credentials. Read from env vars by the CLI.

    NEVER logs the password. NEVER persists to disk. NEVER commits to git.
    """

    username: str
    password: str
    totp_secret: str | None = None  # optional TOTP for 2FA-enabled accounts

    def redacted(self) -> str:
        u = self.username
        return f"AcademyCredentials(username={u!r}, password='***', totp={'set' if self.totp_secret else 'unset'})"

    def __repr__(self) -> str:
        # Force redaction even on accidental logging.
        return self.redacted()


class AcademySession(abc.ABC):
    """Abstract academy session. Concrete implementations override the
    transport methods. The orchestrator depends only on this interface."""

    # ---- lifecycle -----------------------------------------------------------

    @abc.abstractmethod
    def login(self, creds: AcademyCredentials) -> None:
        """Authenticate. Raises on failure. Idempotent."""

    @abc.abstractmethod
    def is_logged_in(self) -> bool:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release any browser/network resources."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- progress ------------------------------------------------------------

    @abc.abstractmethod
    def get_progress_state(self) -> ProgressState:
        """Fetch the user's current progress (cubes balance + completed modules)."""

    @abc.abstractmethod
    def list_modules(self) -> list[AcademyModule]:
        """List all modules the account can see (locked + unlocked + completed)."""

    @abc.abstractmethod
    def fetch_module(self, module_id: str) -> AcademyModule:
        """Fetch a module's full content (sections, questions, sandbox info)."""

    # ---- mutations -----------------------------------------------------------

    @abc.abstractmethod
    def start_module(self, module_id: str) -> None:
        """Spend cubes / mark the module as in-progress on the academy side."""

    @abc.abstractmethod
    def submit_answer(self, module_id: str, answer: AcademyAnswer) -> bool:
        """Submit an answer. Returns True on accepted/correct, False otherwise.

        Implementations MUST honor a parent ``study_only=True`` setting and
        refuse to actually POST when configured for dry-run. The default
        ``MockAcademySession`` implements both modes.
        """


# ---- mock implementation ----------------------------------------------------


class MockAcademySession(AcademySession):
    """In-memory academy session driven by a static fixture.

    Used in tests and as a no-network sanity environment for the orchestrator.
    Construct with a list of modules + an initial ProgressState. ``submit_answer``
    is a configurable predicate so tests can simulate correct / incorrect answers.

    >>> sess = MockAcademySession(modules=[mod], state=state)
    >>> sess.login(AcademyCredentials("u", "p"))
    >>> sess.list_modules()
    """

    def __init__(
        self,
        modules: list[AcademyModule],
        state: ProgressState,
        accept_predicate=None,
        study_only: bool = True,
    ) -> None:
        self._modules = {m.id: m for m in modules}
        self._state = state
        self._logged_in = False
        self._accept = accept_predicate or (lambda mod_id, ans: True)
        self.study_only = study_only
        # Track every submission attempted (whether actually POST'd or not),
        # for assertions in tests + audit logs in real runs.
        self.submission_log: list[tuple[str, AcademyAnswer, bool]] = []

    # ---- lifecycle -----------------------------------------------------------

    def login(self, creds: AcademyCredentials) -> None:
        if not creds.username or not creds.password:
            raise ValueError("empty credentials")
        self._logged_in = True

    def is_logged_in(self) -> bool:
        return self._logged_in

    def close(self) -> None:
        self._logged_in = False

    # ---- progress ------------------------------------------------------------

    def get_progress_state(self) -> ProgressState:
        return self._state

    def list_modules(self) -> list[AcademyModule]:
        return list(self._modules.values())

    def fetch_module(self, module_id: str) -> AcademyModule:
        if module_id not in self._modules:
            raise KeyError(module_id)
        return self._modules[module_id]

    def start_module(self, module_id: str) -> None:
        if module_id not in self._modules:
            raise KeyError(module_id)
        m = self._modules[module_id]
        if self._state.cubes_balance < m.cubes_to_unlock:
            raise RuntimeError(
                f"insufficient cubes for {module_id}: have {self._state.cubes_balance}, need {m.cubes_to_unlock}"
            )
        if not self.study_only:
            self._state.cubes_balance -= m.cubes_to_unlock
        self._state.in_progress_module_id = module_id

    def submit_answer(self, module_id: str, answer: AcademyAnswer) -> bool:
        if not self._logged_in:
            raise RuntimeError("not logged in")
        if module_id not in self._modules:
            raise KeyError(module_id)
        accepted = bool(self._accept(module_id, answer))
        # In study_only mode, we still RECORD the would-be submission so
        # demos capture the agent's intent, but we don't mutate cubes /
        # completion state.
        self.submission_log.append((module_id, answer, accepted))
        if not self.study_only and accepted:
            mod = self._modules[module_id]
            # If this was the last question of the module, mark complete.
            answered = {a.question_id for (_, a, ok) in self.submission_log if ok and _ == module_id}
            all_q = {q.id for q in mod.all_questions}
            if all_q.issubset(answered):
                mod.completed = True
                if module_id not in self._state.completed_module_ids:
                    self._state.completed_module_ids.append(module_id)
                self._state.cubes_balance += mod.cubes_reward
                if self._state.in_progress_module_id == module_id:
                    self._state.in_progress_module_id = None
        return accepted
