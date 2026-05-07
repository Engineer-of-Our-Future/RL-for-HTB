"""Heuristic question answerer (theory-aware).

The user pointed out the right framing: each academy section gives the model
*theory* first, then asks questions whose answers are usually directly stated
in that theory. So the answerer's job is reading-comprehension on the section
body, not generic knowledge.

We do NOT use an LLM. Each question routes to a handler based on its
``QuestionType`` plus a small set of pattern-matched sub-handlers:

- **Acronym questions** (``"What does the acronym X stand for"``) - look for
  X expanded in the body in the canonical patterns ``"X (Y)"``, ``"X = Y"``,
  ``"X - Y"``, ``"X stands for Y"``, ``"Y (X)"``.
- **Inline-code preference** - when the answer is a single token, prefer the
  contents of any inline ``<code>...</code>`` span over guessing a tail of a
  sentence. The screenshot's "{up-to-date}" highlight is exactly this case.
- **MC questions** - Jaccard token overlap between option text and section body.
- **Number-counting questions** (``"How many X..."``) - count bullet-list
  items or pattern matches in the body.
- **FLAG questions** - run the section's most relevant code block in the
  sandbox SSH and parse the output for a flag-shaped string.

Below ``manual_review_threshold`` (set on the orchestrator) we hand off to the
operator rather than guess. Every answer carries ``confidence`` and a short
``rationale`` so the demo trail is auditable.
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
# Capture the full noun phrase between "the acronym" / "what does" and "stand for";
# we then extract the actual acronym (token with >=2 uppercase letters in a row)
# from inside it. Handles both "what does PAM stand for" and "what does the
# acronym Linux PAM stand for" - the screenshot's exact phrasing.
_ACRONYM_PROMPT_RE = re.compile(
    r"\bwhat\s+does\s+(?:the\s+acronym\s+)?(?P<phrase>[\w\s\-]{1,60}?)\s+stand\s+for\b",
    re.IGNORECASE,
)
_ACRONYM_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,15})\b")
_HOWMANY_PROMPT_RE = re.compile(r"\bhow\s+many\b", re.IGNORECASE)


def _tokenize(s: str) -> set[str]:
    """Lowercase alpha-num word set, useful for similarity scoring."""
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _expand_acronym(acronym: str, body: str) -> str | None:
    """Look for ``ACRO (Words ...)`` or ``Words ... (ACRO)`` etc. in body.

    Returns the expansion text if found, else None.
    """
    a = re.escape(acronym)
    # 1. "PAM (Pluggable Authentication Modules)"
    m = re.search(rf"\b{a}\s*\(([^)]{{3,80}}?)\)", body)
    if m:
        cand = m.group(1).strip()
        if _looks_like_expansion(cand, acronym):
            return cand
    # 2. "Pluggable Authentication Modules (PAM)"
    m = re.search(rf"([A-Z][A-Za-z\s\-]{{3,80}}?)\s*\(\s*{a}\s*\)", body)
    if m:
        cand = m.group(1).strip()
        if _looks_like_expansion(cand, acronym):
            return cand
    # 3. "PAM stands for Pluggable Authentication Modules"
    m = re.search(
        rf"\b{a}\s+(?:stands?\s+for|is\s+short\s+for)\s+([^.,;\n]{{3,80}})",
        body, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    # 4. "PAM = Pluggable Authentication Modules" / "PAM - Pluggable ..."
    m = re.search(rf"\b{a}\s*[-=]\s*([A-Z][^,.;\n]{{3,80}})", body)
    if m:
        return m.group(1).strip()
    return None


def _looks_like_expansion(candidate: str, acronym: str) -> bool:
    """Sanity check: candidate's leading initials must roughly match the acronym.

    This filters out spurious parentheses like "PAM (which is on Linux)".
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]{1,30}", candidate)
    if not words:
        return False
    # Minimum: at least half of the acronym's letters should match leading
    # letters of words in the candidate.
    initials = "".join(w[0] for w in words).upper()
    matched = sum(1 for c in acronym.upper() if c in initials)
    return matched >= max(1, len(acronym) // 2)


def _count_pattern_matches(prompt: str, section: AcademySection) -> int | None:
    """Return a count if the body has a list / pattern that the prompt hints at.

    Trivial heuristic: if the prompt has 'how many <noun>' and the body has
    bullet lists, the count is the longest bullet list's length.
    """
    if not section.bullet_lists:
        return None
    # Prefer the bullet list whose tokens overlap the prompt the most.
    p = _tokenize(prompt)
    scored = [
        (sum(_jaccard(p, _tokenize(item)) for item in lst), len(lst), lst)
        for lst in section.bullet_lists
    ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][1] if scored else None


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
        prompt = question.prompt

        # 1. Acronym pattern -> direct expansion lookup. We capture the noun
        #    phrase ("Linux PAM") and pull the actual acronym from inside it.
        m = _ACRONYM_PROMPT_RE.search(prompt)
        if m:
            phrase = m.group("phrase") or ""
            acro_m = _ACRONYM_TOKEN_RE.search(phrase)
            if acro_m:
                acro = acro_m.group(1).upper()
                expansion = _expand_acronym(acro, section.body_text)
                if expansion:
                    return AcademyAnswer(
                        question_id=question.id,
                        answer_text=expansion,
                        confidence=0.85,
                        method="acronym_expansion",
                        rationale=f"found expansion of {acro!r} (from prompt phrase {phrase!r}) in section body",
                    )

        # 2. "How many ..." -> count bullet-list items.
        if _HOWMANY_PROMPT_RE.search(prompt):
            count = _count_pattern_matches(prompt, section)
            if count is not None:
                return AcademyAnswer(
                    question_id=question.id,
                    answer_text=str(count),
                    confidence=0.6,
                    method="howmany_count",
                    rationale=f"counted {count} items in best-matching bullet list",
                )

        # 3. Body-text scan -> sentence overlap, with strong preference for
        #    inline-code spans inside the matched sentence (academy questions
        #    typically expect the literal token shown in `<code>`).
        prompt_tokens = _tokenize(prompt)
        sentences = re.split(r"(?<=[.!?])\s+", section.body_text)
        candidates: list[tuple[float, str]] = []
        for s in sentences:
            s_tokens = _tokenize(s)
            score = _jaccard(prompt_tokens, s_tokens)
            if score > 0:
                candidates.append((score, s.strip()))

        if candidates:
            candidates.sort(key=lambda x: (-x[0], len(x[1])))
            best_score, best_sentence = candidates[0]
            # Prefer inline code spans that happen to appear in the matched
            # sentence (or the section as a whole if the sentence has none).
            inline_in_sentence = [
                c for c in section.inline_code if c and c in best_sentence
            ]
            if inline_in_sentence:
                return AcademyAnswer(
                    question_id=question.id,
                    answer_text=inline_in_sentence[0],
                    confidence=min(0.8, best_score + 0.3),
                    method="inline_code_in_match",
                    rationale=(
                        f"best sentence (jaccard={best_score:.2f}) contains inline "
                        f"code span -> using it as the literal answer"
                    ),
                )
            # Fall back to a quoted/back-ticked span anywhere in the sentence,
            # else last 3 words.
            mq = re.search(r'"([^"]{2,80})"', best_sentence) or re.search(
                r"`([^`]{2,80})`", best_sentence
            )
            if mq:
                answer = mq.group(1)
            else:
                words = best_sentence.split()
                answer = " ".join(words[-3:]) if words else ""
            return AcademyAnswer(
                question_id=question.id,
                answer_text=answer,
                confidence=min(0.5, best_score),
                method="heuristic_text",
                rationale=(
                    f"matched sentence with jaccard={best_score:.2f}; "
                    f"extracted tail/quoted span"
                ),
            )

        # 4. As a last resort, if the section has only one inline code span,
        #    use it (the screenshot's "up-to-date" case).
        if len(section.inline_code) == 1:
            return AcademyAnswer(
                question_id=question.id,
                answer_text=section.inline_code[0],
                confidence=0.4,
                method="lone_inline_code",
                rationale="section had a single inline code span; used as answer fallback",
            )

        return AcademyAnswer(
            question_id=question.id,
            answer_text="",
            confidence=0.0,
            method="skipped",
            rationale="no overlap, no acronym, no list, no inline code to lean on",
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
