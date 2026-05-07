"""Heuristic question answerer.

We do NOT use an LLM. Each question routes to a handler based on its
``QuestionType``:

- ``MULTIPLE_CHOICE`` -> match the prompt's keywords against the option strings,
  pick the closest.
- ``TEXT`` -> regex/keyword extraction from the section's body text + code blocks
  (the academy's text usually directly contains the answer).
- ``FLAG`` -> run the section's most-relevant code block in the sandbox SSH and
  parse the output for a flag-shaped string.
- ``UNSUPPORTED`` -> skip with a manual-fallback recommendation.

Every answer carries a ``confidence`` and a ``rationale`` so the orchestrator
can decide whether to actually submit. The default policy:

  - confidence >= 0.7 and study_only=False  ->  submit
  - else                                    ->  log to demo, do not submit

Answers are also passed verbatim into the demonstration trajectory so BC
pretraining can learn the agent's reasoning trace.
"""

from __future__ import annotations

import re
from typing import Callable

from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyQuestion,
    AcademySection,
    QuestionType,
)


# ---- helpers ----------------------------------------------------------------


_FLAG_LINE_RE = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$", re.MULTILINE)
_HTB_FLAG_RE = re.compile(r"HTB\{[^}]+\}")


def _tokenize(s: str) -> set[str]:
    """Lowercase alpha-num word set, useful for similarity scoring."""
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Sandbox runner type: takes a command, returns its stdout text.
SandboxRunner = Callable[[str], str]


# ---- answerer ---------------------------------------------------------------


class HeuristicAnswerer:
    """Stateless answerer; one call per question."""

    def __init__(self, sandbox_runner: SandboxRunner | None = None) -> None:
        self.sandbox_runner = sandbox_runner

    def answer(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        if question.type == QuestionType.MULTIPLE_CHOICE:
            return self._answer_mc(question, section)
        if question.type == QuestionType.TEXT:
            return self._answer_text(question, section)
        if question.type == QuestionType.FLAG:
            return self._answer_flag(question, section)
        return AcademyAnswer(
            question_id=question.id,
            answer_text="",
            confidence=0.0,
            method="skipped",
            rationale=f"question type {question.type.value} not supported by heuristic answerer",
        )

    # ---- per-type handlers ---------------------------------------------------

    def _answer_mc(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        prompt_tokens = _tokenize(question.prompt)
        body_tokens = _tokenize(section.body_text)
        scored: list[tuple[float, str]] = []
        for opt in question.multiple_choice_options:
            opt_tokens = _tokenize(opt)
            # Score = how much overlap the option has with the section content,
            # weighted by overlap with prompt to break ties.
            body_score = _jaccard(opt_tokens, body_tokens)
            prompt_score = _jaccard(opt_tokens, prompt_tokens)
            scored.append((body_score * 2.0 + prompt_score, opt))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_opt = scored[0]
        # Confidence is the gap between top and runner-up, scaled.
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        confidence = min(1.0, max(0.0, best_score - runner_up + 0.3))
        return AcademyAnswer(
            question_id=question.id,
            answer_text=best_opt,
            confidence=confidence if best_score > 0 else 0.1,
            method="mc_match",
            rationale=(
                f"top option overlap with section body ({best_score:.2f}); "
                f"runner-up ({runner_up:.2f})"
            ),
        )

    def _answer_text(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        # Strategy: scan the body text for sentences whose tokens overlap
        # heavily with the question, then return the most "answer-shaped"
        # short noun phrase from them. We pick the SHORTEST candidate so we
        # don't paste a paragraph as the answer.
        prompt_tokens = _tokenize(question.prompt)
        sentences = re.split(r"(?<=[.!?])\s+", section.body_text)
        candidates: list[tuple[float, str]] = []
        for s in sentences:
            s_tokens = _tokenize(s)
            score = _jaccard(prompt_tokens, s_tokens)
            if score > 0:
                candidates.append((score, s.strip()))

        if not candidates:
            return AcademyAnswer(
                question_id=question.id,
                answer_text="",
                confidence=0.0,
                method="skipped",
                rationale="no sentence in section overlapped with question tokens",
            )

        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        best_score, best_sentence = candidates[0]
        # Try to extract a short tail (last token / quoted span).
        match = re.search(r'"([^"]{2,80})"', best_sentence) or re.search(
            r"`([^`]{2,80})`", best_sentence
        )
        if match:
            answer = match.group(1)
        else:
            # Fall back to the last 1-3 words of the matched sentence.
            words = best_sentence.split()
            answer = " ".join(words[-3:]) if words else ""

        confidence = min(0.6, best_score)  # text answers are inherently uncertain
        return AcademyAnswer(
            question_id=question.id,
            answer_text=answer,
            confidence=confidence,
            method="heuristic_text",
            rationale=(
                f"matched sentence with jaccard={best_score:.2f}; "
                f"extracted tail/quoted span"
            ),
        )

    def _answer_flag(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        if self.sandbox_runner is None:
            return AcademyAnswer(
                question_id=question.id,
                answer_text="",
                confidence=0.0,
                method="skipped",
                rationale="flag question but no sandbox_runner provided",
            )
        if not section.code_blocks and not question.hints:
            return AcademyAnswer(
                question_id=question.id,
                answer_text="",
                confidence=0.0,
                method="skipped",
                rationale="flag question but no code block / hint to run",
            )
        # Pick the most flag-likely command: one that mentions cat/grep/curl
        # with a flag-y file/url, or just the longest hint as a fallback.
        candidate_cmds = list(question.hints) + list(section.code_blocks)
        scored = sorted(
            candidate_cmds,
            key=lambda c: (
                ("flag" in c.lower())
                + ("cat " in c.lower())
                + ("HTB" in c),
            ),
            reverse=True,
        )
        cmd = scored[0]
        try:
            out = self.sandbox_runner(cmd)
        except Exception as exc:
            return AcademyAnswer(
                question_id=question.id,
                answer_text="",
                confidence=0.0,
                method="sandbox_cmd",
                rationale=f"sandbox command raised: {exc}",
                sandbox_command=cmd,
            )
        # Look for a flag-shape in the output. Prefer HTB{...} pattern.
        m = _HTB_FLAG_RE.search(out)
        if m:
            return AcademyAnswer(
                question_id=question.id,
                answer_text=m.group(0),
                confidence=0.95,
                method="sandbox_cmd",
                rationale="found HTB{...} pattern in sandbox output",
                sandbox_command=cmd,
                sandbox_output=out[:4096],
            )
        # Fall back to a single-token-on-a-line shape.
        m2 = _FLAG_LINE_RE.search(out)
        if m2:
            return AcademyAnswer(
                question_id=question.id,
                answer_text=m2.group(0),
                confidence=0.55,
                method="sandbox_cmd",
                rationale="found token-only line in sandbox output",
                sandbox_command=cmd,
                sandbox_output=out[:4096],
            )
        return AcademyAnswer(
            question_id=question.id,
            answer_text="",
            confidence=0.0,
            method="sandbox_cmd",
            rationale="no flag-shaped output from sandbox",
            sandbox_command=cmd,
            sandbox_output=out[:4096],
        )
