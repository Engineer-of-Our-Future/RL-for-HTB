"""Top-level academy auto-learner loop.

Drives one cycle of:
  1. Pick the next eligible module (curriculum.next_module)
  2. start_module(...)
  3. For each section -> open sandbox if any -> for each question:
       a. Answer via HeuristicAnswerer
       b. If study_only OR confidence < threshold: log + skip submit
       c. Else: submit_answer; record acceptance
  4. Compose Demonstration(s) and write to disk
  5. Refresh ProgressState and loop

The orchestrator is deliberately simple: it doesn't drive RL training. The
output is a stream of Demonstrations the BC trainer can consume.

If the heuristic answerer's confidence is below ``manual_review_threshold``,
the orchestrator pauses and records the question for manual review (no
submission, no auto-skip).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from htbrl.academy.answerer import HeuristicAnswerer
from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.curriculum import (
    UnlockGateResult,
    check_unlock_gate,
    next_module,
    progress_summary,
)
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    QuestionType,
)
from htbrl.academy.sandbox import SandboxRunner
from htbrl.academy.session import AcademySession
from htbrl.data.demo_dataset import Demonstration, save_demonstration


log = logging.getLogger("htbrl.academy")


@dataclass
class OrchestratorConfig:
    # If True, never POSTs an answer; just records what we would have submitted.
    study_only: bool = True
    # Even in auto-submit, only submit when confidence >= this.
    submit_confidence_threshold: float = 0.7
    # Below this, pause for manual review (orchestrator returns; you can resume later).
    manual_review_threshold: float = 0.3
    # Hard cap on questions per module so a buggy answerer doesn't loop forever.
    max_questions_per_module: int = 200
    # Where to write generated demos.
    auto_demo_dir: Path = field(default_factory=lambda: Path("data/auto_demos"))
    # Write demos compressed.
    compress_demos: bool = True
    # Hard cap on number of modules the orchestrator processes per ``run`` call.
    max_modules_per_run: int = 1
    # Unlock gate: don't open a NEW module until the previous one is fully
    # attempted (every question has a recorded submission) AND the cube balance
    # has actually changed (the academy's reward signal landed). Wires into
    # ``curriculum.check_unlock_gate``. The user's exact rule was:
    #   "open new module only if all questions are answered and cube balance
    #    are updated"
    # Both halves can be relaxed here for partial-walk scenarios (e.g. when
    # the sandbox is unavailable so flag questions can't be answered).
    require_all_answered_before_unlock: bool = True
    require_cube_refresh_before_unlock: bool = True


@dataclass
class ManualReviewItem:
    module_id: str
    question: AcademyQuestion
    answer: AcademyAnswer


@dataclass
class OrchestratorRunResult:
    modules_attempted: list[str] = field(default_factory=list)
    modules_completed: list[str] = field(default_factory=list)
    demos_written: list[Path] = field(default_factory=list)
    manual_review: list[ManualReviewItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # The unlock-gate decisions made during this run, in chronological order.
    # The orchestrator ALWAYS records the gate result before each module open
    # so the user can audit "why did we stop?" or "why did we open module X?".
    unlock_gates: list[UnlockGateResult] = field(default_factory=list)


# Type alias for a callback the caller can pass to handle a ManualReviewItem
# interactively. If None, the orchestrator just records it and moves on.
ManualReviewHandler = Callable[[ManualReviewItem], AcademyAnswer | None]


class AutoLearner:
    """Top-level orchestrator. Construct, then call ``run()`` to advance."""

    def __init__(
        self,
        session: AcademySession,
        cfg: OrchestratorConfig,
        answerer: HeuristicAnswerer | None = None,
        manual_review_handler: ManualReviewHandler | None = None,
    ) -> None:
        self.session = session
        self.cfg = cfg
        self.answerer = answerer
        self.manual_review_handler = manual_review_handler
        self.cfg.auto_demo_dir.mkdir(parents=True, exist_ok=True)

    # ---- main loop ----------------------------------------------------------

    def run(self, preferred_order: list[str] | None = None) -> OrchestratorRunResult:
        result = OrchestratorRunResult()
        if not self.session.is_logged_in():
            result.errors.append("session not logged in")
            return result

        # Unlock-gate carry-state. After the first module attempt we know:
        #   - which questions we submitted answers for (set of qids)
        #   - what the cube balance was BEFORE we started that module
        # Combined with the fresh-state cube balance read each iteration, this
        # is enough to enforce "open new module only if all questions are
        # answered and cube balance are updated".
        last_module: AcademyModule | None = None
        last_attempted_qids: set[str] = set()
        last_cubes_before: int = 0
        # Modules we've already attempted in THIS run. The session tracks
        # completion, but in study_only mode no module is ever marked complete,
        # so we'd re-pick the same module forever without this set.
        already_attempted_in_run: set[str] = set()

        for _ in range(self.cfg.max_modules_per_run):
            state = self.session.get_progress_state()
            modules = self.session.list_modules()
            # Hide modules we already attempted in THIS run from the curriculum
            # selector (it filters by completion, which study_only never sets).
            modules = [m for m in modules if m.id not in already_attempted_in_run]
            log.info("progress: %s", progress_summary(modules, state))

            gate = check_unlock_gate(
                current_module=last_module,
                answered_question_ids=last_attempted_qids,
                cube_balance_before=last_cubes_before,
                cube_balance_after=state.cubes_balance,
                require_all_answered=self.cfg.require_all_answered_before_unlock,
                require_cube_refresh=self.cfg.require_cube_refresh_before_unlock,
            )
            result.unlock_gates.append(gate)
            log.info("unlock gate: allowed=%s reason=%s", gate.allowed, gate.reason)
            if not gate.allowed:
                result.errors.append(f"unlock gate: {gate.reason}")
                break

            mod = next_module(
                modules, state,
                preferred_order=preferred_order,
                unlock_gate=gate,
            )
            if mod is None:
                log.info("no eligible module - all done or insufficient cubes")
                break

            cubes_before = state.cubes_balance
            attempted_qids: set[str] = set()
            try:
                attempted_qids = self._attempt_module(mod, result)
                result.modules_attempted.append(mod.id)
            except Exception as exc:
                log.exception("module %s failed", mod.id)
                result.errors.append(f"{mod.id}: {exc!r}")
                break

            # Carry state into the next iteration so the gate sees it,
            # and bar this module from being re-picked by next_module's
            # eligibility filter (study_only never marks completion, so
            # without this set we'd loop on the same module forever).
            already_attempted_in_run.add(mod.id)
            last_module = mod
            last_attempted_qids = attempted_qids
            last_cubes_before = cubes_before
        return result

    # ---- per-module ---------------------------------------------------------

    def _attempt_module(
        self, module_summary: AcademyModule, result: OrchestratorRunResult,
    ) -> set[str]:
        """Attempt every question in the module; return the set of qids attempted.

        Used by ``run`` to feed the unlock-gate state for the next iteration.
        A qid is "attempted" iff we recorded a submission tuple for it
        (correct, wrong, or manual-review-skipped); pure no-op skips do NOT
        count, because the unlock-gate's "all questions answered" check would
        otherwise be trivially true for any module with a manual_review_handler
        that always returns None.
        """
        log.info("starting module: %s (%s)", module_summary.id, module_summary.title)
        # Pull the FULL module (with sections + questions + sandbox).
        module = self.session.fetch_module(module_summary.id)
        self.session.start_module(module.id)

        all_submissions: list[tuple[str, AcademyAnswer, bool]] = []
        n_questions_seen = 0
        attempted_qids: set[str] = set()

        for section in module.sections:
            sandbox_runner: SandboxRunner | None = None
            if section.sandbox is not None:
                sandbox_runner = SandboxRunner(section.sandbox)
                try:
                    sandbox_runner.open()
                except Exception as exc:
                    log.warning("could not open sandbox for %s/%s: %s", module.id, section.id, exc)
                    sandbox_runner = None

            try:
                ans = self.answerer or HeuristicAnswerer(sandbox_runner=sandbox_runner)
                # If the user passed a single answerer, wire its sandbox runner
                # for this section temporarily.
                if self.answerer is not None and sandbox_runner is not None:
                    self.answerer.sandbox_runner = sandbox_runner

                for q in section.questions:
                    if n_questions_seen >= self.cfg.max_questions_per_module:
                        log.warning("max_questions_per_module reached on %s", module.id)
                        break
                    n_questions_seen += 1

                    answer = ans.answer(q, section)
                    log.info(
                        "  q=%s  method=%s  conf=%.2f  ans=%r",
                        q.id, answer.method, answer.confidence,
                        (answer.answer_text or "")[:80],
                    )

                    if answer.confidence < self.cfg.manual_review_threshold:
                        item = ManualReviewItem(module_id=module.id, question=q, answer=answer)
                        if self.manual_review_handler is not None:
                            override = self.manual_review_handler(item)
                            if override is not None:
                                answer = override
                            else:
                                result.manual_review.append(item)
                                all_submissions.append((module.id, answer, False))
                                attempted_qids.add(q.id)
                                continue
                        else:
                            result.manual_review.append(item)
                            all_submissions.append((module.id, answer, False))
                            attempted_qids.add(q.id)
                            continue

                    accepted = False
                    if (
                        not self.cfg.study_only
                        and answer.confidence >= self.cfg.submit_confidence_threshold
                    ):
                        try:
                            accepted = self.session.submit_answer(module.id, answer)
                        except Exception as exc:
                            log.warning("submit failed for %s/%s: %s", module.id, q.id, exc)
                    all_submissions.append((module.id, answer, accepted))
                    attempted_qids.add(q.id)
            finally:
                if sandbox_runner is not None:
                    sandbox_runner.close()
                # Don't leak the sandbox_runner on the shared answerer.
                if self.answerer is not None:
                    self.answerer.sandbox_runner = None

        # Build + persist a Demonstration regardless of outcome.
        demo: Demonstration = session_to_demonstration(
            module, all_submissions, study_only=self.cfg.study_only,
        )
        path = self._demo_path(module.id)
        save_demonstration(demo, path, compress=self.cfg.compress_demos)
        result.demos_written.append(path)
        # The session itself decides true completion (n correct == total).
        # We refresh state and check; "any accepted answer" alone isn't enough.
        state = self.session.get_progress_state()
        if state.has_completed(module.id):
            result.modules_completed.append(module.id)
        return attempted_qids

    # ---- internal -----------------------------------------------------------

    def _demo_path(self, module_id: str) -> Path:
        suffix = ".msgpack.gz" if self.cfg.compress_demos else ".msgpack"
        safe = module_id.replace("/", "_").replace(" ", "_")
        return self.cfg.auto_demo_dir / f"academy_{safe}{suffix}"
