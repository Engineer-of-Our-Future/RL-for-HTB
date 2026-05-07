"""HTB Academy auto-learner CLI.

Read the academy module list, work through the next eligible module, log
a Demonstration to ``data/auto_demos/``, and (optionally) submit answers.

ToS WARNING: see README. Default mode is ``study_only=True`` (no submission).
The ``--enable-auto-submit`` flag opts in to actual answer submission. Use
only on a research account.

Examples:

    # Study-only run on the mock fixture (no network, no real account):
    python scripts/htb_academy.py --transport mock --max-modules 2

    # Real run with study-only mode (reads modules + uses sandbox + logs demos
    # but never POSTs an answer):
    HTB_ACADEMY_USER=alice HTB_ACADEMY_PASS=hunter2 \\
        python scripts/htb_academy.py --transport playwright --max-modules 1

    # Real run with auto-submit (USE A RESEARCH ACCOUNT):
    HTB_ACADEMY_USER=alice HTB_ACADEMY_PASS=hunter2 \\
        python scripts/htb_academy.py --transport playwright --max-modules 1 \\
        --enable-auto-submit --i-accept-academy-tos-risk
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from htbrl.academy.answerer import HeuristicAnswerer
from htbrl.academy.orchestrator import AutoLearner, OrchestratorConfig
from htbrl.academy.session import AcademyCredentials


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transport",
        choices=["mock", "playwright"],
        default="mock",
        help="mock = in-memory fixture (no network); playwright = real browser",
    )
    p.add_argument("--auto-demo-dir", type=Path, default=Path("data/auto_demos"))
    p.add_argument("--max-modules", type=int, default=1, help="cap modules per run")
    p.add_argument("--max-questions-per-module", type=int, default=200)
    p.add_argument("--manual-review-threshold", type=float, default=0.3)
    p.add_argument("--submit-confidence-threshold", type=float, default=0.7)
    # Submission-mode gates
    p.add_argument(
        "--enable-auto-submit", action="store_true",
        help="ACTUALLY submit answers (default is study-only / dry-run).",
    )
    p.add_argument(
        "--i-accept-academy-tos-risk", action="store_true",
        help="Required alongside --enable-auto-submit. See README ToS warning.",
    )
    p.add_argument(
        "--preferred-modules", default=None,
        help="comma-separated module ids to prefer first",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _build_session(args, creds: AcademyCredentials):
    if args.transport == "mock":
        # Build a tiny demo fixture so the CLI can run end-to-end without
        # network access. Mirrors what tests use.
        from htbrl.academy.page_models import (
            AcademyModule,
            AcademySection,
            AcademyQuestion,
            QuestionType,
            ProgressState,
        )
        from htbrl.academy.session import MockAcademySession

        mod = AcademyModule(
            id="m1",
            title="Linux Fundamentals",
            tier=0,
            cubes_to_unlock=0,
            cubes_reward=10,
            sections=[
                AcademySection(
                    id="s1",
                    title="Basic commands",
                    body_text=(
                        "The pwd command prints the current directory. "
                        'The "ls" command lists files.'
                    ),
                    questions=[
                        AcademyQuestion(
                            id="q1",
                            prompt="Which command lists files in a directory?",
                            type=QuestionType.MULTIPLE_CHOICE,
                            multiple_choice_options=["ls", "pwd", "cat", "rm"],
                        ),
                    ],
                ),
            ],
        )
        state = ProgressState(user_id="mock", cubes_balance=0)
        return MockAcademySession(
            modules=[mod], state=state,
            study_only=not args.enable_auto_submit,
        )

    # playwright transport - lives in a separate optional module so the test
    # suite doesn't pull in browser binaries. We import lazily.
    try:
        from htbrl.academy.playwright_session import PlaywrightAcademySession  # type: ignore
    except ImportError:
        print(
            "ERROR: --transport playwright requires htbrl.academy.playwright_session "
            "which is not implemented in this build. Use --transport mock or "
            "implement PlaywrightAcademySession against the AcademySession ABC.",
            file=sys.stderr,
        )
        sys.exit(2)
    sess = PlaywrightAcademySession(study_only=not args.enable_auto_submit)
    sess.login(creds)
    return sess


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.enable_auto_submit and not args.i_accept_academy_tos_risk:
        print(
            "ERROR: --enable-auto-submit requires --i-accept-academy-tos-risk. "
            "See README for the ToS warning.",
            file=sys.stderr,
        )
        return 2

    creds_user = os.environ.get("HTB_ACADEMY_USER", "")
    creds_pass = os.environ.get("HTB_ACADEMY_PASS", "")
    creds_totp = os.environ.get("HTB_ACADEMY_TOTP_SECRET")
    if args.transport == "playwright" and not (creds_user and creds_pass):
        print(
            "ERROR: HTB_ACADEMY_USER and HTB_ACADEMY_PASS env vars are required "
            "for --transport playwright.",
            file=sys.stderr,
        )
        return 2
    creds = AcademyCredentials(
        username=creds_user or "mock-user",
        password=creds_pass or "mock-pass",
        totp_secret=creds_totp,
    )

    print(f"=== HTB Academy auto-learner ===")
    print(f"  transport      : {args.transport}")
    print(f"  study_only     : {not args.enable_auto_submit}")
    print(f"  auto_demo_dir  : {args.auto_demo_dir}")
    print(f"  credentials    : {creds!r}")  # __repr__ redacts password

    session = _build_session(args, creds)
    if not session.is_logged_in():
        session.login(creds)

    cfg = OrchestratorConfig(
        study_only=not args.enable_auto_submit,
        submit_confidence_threshold=args.submit_confidence_threshold,
        manual_review_threshold=args.manual_review_threshold,
        max_questions_per_module=args.max_questions_per_module,
        max_modules_per_run=args.max_modules,
        auto_demo_dir=args.auto_demo_dir,
    )
    learner = AutoLearner(session=session, cfg=cfg, answerer=HeuristicAnswerer())

    preferred = (
        args.preferred_modules.split(",") if args.preferred_modules else None
    )
    result = learner.run(preferred_order=preferred)

    print("--- run summary ---")
    print(f"  modules_attempted = {result.modules_attempted}")
    print(f"  modules_completed = {result.modules_completed}")
    print(f"  demos_written     = {[str(p) for p in result.demos_written]}")
    print(f"  manual_review     = {len(result.manual_review)} item(s)")
    if result.errors:
        print(f"  errors            = {result.errors}")

    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
