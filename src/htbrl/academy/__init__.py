"""HTB Academy auto-learner (PLAN.md Phase 5b).

The academy package gives the agent a curriculum: read modules in order, work
through their content + sandbox exercises, and log every interaction as a
``Demonstration`` for downstream BC pretraining. This *augments* the manual
demonstration collection (Phase 5) - it never replaces it.

The package is structured around clear contracts so the actual transport layer
(Playwright vs requests vs a future HTB API) can be swapped without touching
the orchestrator. The default ``MockSession`` lets tests + dry-runs work
without browser binaries or a real account.

**Important - HTB Academy Terms of Service**:
- Auto-submitting answers to academy questions to farm cubes/XP is a gray-zone
  use of an educational platform. Default mode is ``study_only=True`` which
  reads content + practices in the sandbox but does NOT submit answers.
- ``auto_submit=True`` requires an explicit acknowledgment flag in the config
  AND on the CLI. Use only on a research account you're prepared to lose.
- The orchestrator never solves a CAPTCHA, never bypasses 2FA, never alters
  account billing/cubes balance.
"""

from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySandbox,
    AcademySection,
    ProgressState,
    QuestionType,
)
from htbrl.academy.session import (
    AcademyCredentials,
    AcademySession,
    MockAcademySession,
)

__all__ = [
    "AcademyAnswer",
    "AcademyCredentials",
    "AcademyModule",
    "AcademyQuestion",
    "AcademySandbox",
    "AcademySection",
    "AcademySession",
    "MockAcademySession",
    "ProgressState",
    "QuestionType",
]
