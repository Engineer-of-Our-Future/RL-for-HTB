"""Heuristic question answerer (theory-aware) + ranked-candidate proposer.

The user pointed out the right framing: each academy section gives the model
*theory* first, then asks questions whose answers are usually directly stated
in that theory. So the answerer's job is reading-comprehension on the section
body, not generic knowledge.

We do NOT use an LLM. Each question routes through a battery of pattern-matched
candidate generators (acronym, "how many", "what command", "which path",
"what port", inline-code-near-match, MC overlap, sandbox flag). Each generator
emits zero or more :class:`AcademyAnswer` candidates with a confidence score.
``propose()`` returns the ranked list (highest confidence first); ``answer()``
returns just the top candidate so the existing study-only flow keeps working.

The wizard uses ``propose()`` so the operator can pick from alternatives when
the top guess is wrong, and the auto-submit code path uses the top candidate's
confidence to decide whether to submit without prompting.

Lab FLAG questions remain a separate path: they require a sandbox runner and
NEVER auto-submit (per the project rule that flag submission is a user-only
action — auto-submitting would risk an account ban). The answerer can still
*propose* the parsed flag so the user can verify + submit by hand.
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


# ---- regexes ----------------------------------------------------------------


_FLAG_LINE_RE = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$", re.MULTILINE)
_HTB_FLAG_RE = re.compile(r"HTB\{[^}]+\}")
# Capture the full noun phrase between "the acronym" / "what does" and "stand for";
# we then extract the actual acronym (token with >=2 uppercase letters in a row)
# from inside it. Handles both "what does PAM stand for" and "what does the
# acronym Linux PAM stand for".
_ACRONYM_PROMPT_RE = re.compile(
    r"\bwhat\s+does\s+(?:the\s+acronym\s+)?(?P<phrase>[\w\s\-]{1,60}?)\s+stand\s+for\b",
    re.IGNORECASE,
)
_ACRONYM_TOKEN_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,15})\b")
_HOWMANY_PROMPT_RE = re.compile(r"\bhow\s+many\b", re.IGNORECASE)
# Question hints that strongly imply the answer is a single shell token
# (a command, flag, option, switch, file path, directory).
_COMMAND_HINT_RE = re.compile(
    r"\b(?:command|cmd|binary|tool|utility|program|option|flag|switch|argument|parameter|"
    r"file(?:name)?|path|directory|folder|service|daemon|process|user(?:name)?|group|"
    r"package|module|library|protocol|extension|prefix|suffix|character|symbol|operator|"
    r"keyword|variable|env(?:ironment)? variable|setting|attribute)\b",
    re.IGNORECASE,
)
_PORT_PROMPT_RE = re.compile(r"\b(?:port|tcp|udp)\b", re.IGNORECASE)
_VERSION_PROMPT_RE = re.compile(r"\bversion\b", re.IGNORECASE)
_PATH_PROMPT_RE = re.compile(r"\b(?:path|directory|folder|file)\b", re.IGNORECASE)
_NUMBER_PROMPT_RE = re.compile(
    r"\b(?:how many|number of|count of|size|in (?:bytes|kb|mb|gb)|year|"
    r"port number)\b",
    re.IGNORECASE,
)
# Numeric-looking spans we might pull from the body.
_PORT_NUM_RE = re.compile(r"\b(?:port\s+)?(\d{1,5})\b")
_VERSION_NUM_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?(?:[a-z\-][\w\-]*)?)\b")
_PATH_LIKE_RE = re.compile(r"(/[\w\-./]{2,80})")
_FILENAME_RE = re.compile(r"\b([\w\-.]{2,40}\.(?:txt|log|conf|cfg|sh|py|c|ini|bak|md|json|yaml|yml))\b")


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
    """Sanity check: candidate's leading initials must roughly match the acronym."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{1,30}", candidate)
    if not words:
        return False
    initials = "".join(w[0] for w in words).upper()
    matched = sum(1 for c in acronym.upper() if c in initials)
    return matched >= max(1, len(acronym) // 2)


def _count_pattern_matches(prompt: str, section: AcademySection) -> int | None:
    """Return a count if the body has a list / pattern that the prompt hints at."""
    if not section.bullet_lists:
        return None
    p = _tokenize(prompt)
    scored = [
        (sum(_jaccard(p, _tokenize(item)) for item in lst), len(lst), lst)
        for lst in section.bullet_lists
    ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][1] if scored else None


def _best_sentence(prompt: str, body: str) -> tuple[float, str] | None:
    """Return (jaccard, sentence) for the body sentence most overlapping the prompt."""
    p = _tokenize(prompt)
    if not p:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", body)
    best: tuple[float, str] | None = None
    for s in sentences:
        score = _jaccard(p, _tokenize(s))
        if score > 0 and (best is None or score > best[0]):
            best = (score, s.strip())
    return best


# Sandbox runner type: takes a command, returns its stdout text.
SandboxRunner = Callable[[str], str]


# ---- answerer ---------------------------------------------------------------


class HeuristicAnswerer:
    """Stateless answerer with single-best (``answer``) + ranked (``propose``) APIs."""

    def __init__(self, sandbox_runner: SandboxRunner | None = None) -> None:
        self.sandbox_runner = sandbox_runner

    # ---- public API ---------------------------------------------------------

    def answer(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        """Return the single best-confidence candidate (legacy single-best API).

        Always returns an ``AcademyAnswer`` - if every generator failed it's a
        ``method="skipped"`` placeholder so the orchestrator can still record
        the attempt.
        """
        candidates = self.propose(question, section)
        if candidates:
            return candidates[0]
        return AcademyAnswer(
            question_id=question.id, answer_text="", confidence=0.0,
            method="skipped",
            rationale="no heuristic generator produced a candidate",
        )

    def propose(
        self,
        question: AcademyQuestion,
        section: AcademySection,
        *,
        top_n: int = 5,
    ) -> list[AcademyAnswer]:
        """Return up to ``top_n`` ranked candidate answers, best first.

        Used by the wizard so the operator can pick from alternatives when the
        top guess is wrong, and by the auto-submit path which checks the top
        candidate's confidence against a threshold.
        """
        if question.type == QuestionType.MULTIPLE_CHOICE:
            return self._propose_mc(question, section)[:top_n]
        if question.type == QuestionType.FLAG:
            # Flag candidates come from the sandbox; only one slot meaningfully.
            return [self._answer_flag(question, section)][:top_n]
        if question.type == QuestionType.TEXT:
            cands = self._propose_text(question, section)
            return _dedupe_keep_best(cands)[:top_n]
        # Unsupported type - return one skipped marker so callers see a record.
        return [AcademyAnswer(
            question_id=question.id, answer_text="", confidence=0.0,
            method="skipped",
            rationale=f"question type {question.type.value} not supported",
        )]

    # ---- multiple choice ----------------------------------------------------

    def _propose_mc(
        self, question: AcademyQuestion, section: AcademySection,
    ) -> list[AcademyAnswer]:
        prompt_tokens = _tokenize(question.prompt)
        body_tokens = _tokenize(section.body_text)
        scored: list[tuple[float, str, float]] = []
        for opt in question.multiple_choice_options:
            opt_tokens = _tokenize(opt)
            body_score = _jaccard(opt_tokens, body_tokens)
            prompt_score = _jaccard(opt_tokens, prompt_tokens)
            scored.append((body_score * 2.0 + prompt_score, opt, body_score))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[AcademyAnswer] = []
        for rank, (combined, opt, body_score) in enumerate(scored):
            # Confidence: combined-score gap-based for #1, decaying for the rest.
            if rank == 0:
                runner = scored[1][0] if len(scored) > 1 else 0.0
                conf = min(1.0, max(0.0, combined - runner + 0.3))
                if combined == 0:
                    conf = 0.1
            else:
                conf = max(0.05, 0.4 - 0.1 * rank)
            out.append(AcademyAnswer(
                question_id=question.id, answer_text=opt, confidence=conf,
                method="mc_match",
                rationale=f"MC rank {rank+1}: combined={combined:.2f} body={body_score:.2f}",
            ))
        return out

    # ---- text questions: candidate generators ------------------------------

    def _propose_text(
        self, question: AcademyQuestion, section: AcademySection,
    ) -> list[AcademyAnswer]:
        prompt = question.prompt
        out: list[AcademyAnswer] = []

        # 1. Acronym expansion (highest confidence when matched).
        out.extend(self._gen_acronym(prompt, section, question.id))

        # 2. "How many ..." -> count bullet items / list lengths.
        out.extend(self._gen_howmany(prompt, section, question.id))

        # 3. Numeric extractions for port/version/size/year.
        out.extend(self._gen_numeric(prompt, section, question.id))

        # 4. Path / filename extractions.
        out.extend(self._gen_path(prompt, section, question.id))

        # 5. Inline-code spans, ranked by overlap with question keywords.
        out.extend(self._gen_inline_code_ranked(prompt, section, question.id))

        # 6. Best-sentence + inline-code-in-sentence (legacy generator, kept).
        out.extend(self._gen_sentence_anchor(prompt, section, question.id))

        # 7. Bullet items matching the prompt (for "Which of ..." type prompts).
        out.extend(self._gen_bullet_match(prompt, section, question.id))

        # 8. Quoted spans anywhere in the body that the prompt's keywords hit.
        out.extend(self._gen_quoted_spans(prompt, section, question.id))

        # 9. Lone inline-code fallback.
        out.extend(self._gen_lone_inline(section, question.id))

        return out

    # ---- generators ---------------------------------------------------------

    def _gen_acronym(self, prompt: str, section: AcademySection, qid: str) -> list[AcademyAnswer]:
        m = _ACRONYM_PROMPT_RE.search(prompt)
        if not m:
            return []
        phrase = m.group("phrase") or ""
        acro_m = _ACRONYM_TOKEN_RE.search(phrase)
        if not acro_m:
            return []
        acro = acro_m.group(1).upper()
        expansion = _expand_acronym(acro, section.body_text)
        if not expansion:
            return []
        return [AcademyAnswer(
            question_id=qid, answer_text=expansion, confidence=0.85,
            method="acronym_expansion",
            rationale=f"expansion of {acro!r} found in section body",
        )]

    def _gen_howmany(self, prompt: str, section: AcademySection, qid: str) -> list[AcademyAnswer]:
        if not _HOWMANY_PROMPT_RE.search(prompt):
            return []
        out: list[AcademyAnswer] = []
        # Top: best-overlapping bullet list length.
        count = _count_pattern_matches(prompt, section)
        if count is not None:
            out.append(AcademyAnswer(
                question_id=qid, answer_text=str(count), confidence=0.6,
                method="howmany_count",
                rationale=f"counted {count} items in best-matching bullet list",
            ))
        # Also propose every bullet-list length as a lower-confidence
        # alternative (the wizard can pick the right one if needed).
        for lst in section.bullet_lists:
            if not lst:
                continue
            n = len(lst)
            if not any(c.answer_text == str(n) for c in out):
                out.append(AcademyAnswer(
                    question_id=qid, answer_text=str(n), confidence=0.35,
                    method="howmany_bullets_alt",
                    rationale=f"bullet list of length {n}",
                ))
        return out

    def _gen_numeric(self, prompt: str, section: AcademySection, qid: str) -> list[AcademyAnswer]:
        if not _NUMBER_PROMPT_RE.search(prompt) and not _PORT_PROMPT_RE.search(prompt) \
                and not _VERSION_PROMPT_RE.search(prompt):
            return []
        out: list[AcademyAnswer] = []
        body = section.body_text or ""
        # Port-number questions: numbers in 1..65535, scored by sentence overlap.
        if _PORT_PROMPT_RE.search(prompt):
            best = _best_sentence(prompt, body)
            haystack = best[1] if best else body[:4000]
            for m in _PORT_NUM_RE.finditer(haystack):
                num = int(m.group(1))
                if 1 <= num <= 65535:
                    out.append(AcademyAnswer(
                        question_id=qid, answer_text=str(num),
                        confidence=0.55 if best else 0.35,
                        method="port_number",
                        rationale=f"port-shaped number {num} near best-overlap sentence",
                    ))
        # Version questions: dotted-decimal versions.
        if _VERSION_PROMPT_RE.search(prompt):
            for m in _VERSION_NUM_RE.finditer(body):
                v = m.group(1)
                out.append(AcademyAnswer(
                    question_id=qid, answer_text=v, confidence=0.4,
                    method="version_string",
                    rationale=f"dotted-version-shaped token {v!r}",
                ))
        # Generic "how many / number of" with a sentence anchor: fall back to
        # any standalone integer in the matched sentence.
        if _NUMBER_PROMPT_RE.search(prompt):
            best = _best_sentence(prompt, body)
            if best:
                for m in re.finditer(r"\b(\d+)\b", best[1]):
                    out.append(AcademyAnswer(
                        question_id=qid, answer_text=m.group(1),
                        confidence=0.35,
                        method="number_in_sentence",
                        rationale=f"integer in best-overlap sentence (score={best[0]:.2f})",
                    ))
        return out

    def _gen_path(self, prompt: str, section: AcademySection, qid: str) -> list[AcademyAnswer]:
        if not _PATH_PROMPT_RE.search(prompt):
            return []
        out: list[AcademyAnswer] = []
        body = section.body_text or ""
        # Prefer a path inside the most-relevant sentence, but also propose any
        # path in the section body as alternatives.
        best = _best_sentence(prompt, body)
        if best:
            for m in _PATH_LIKE_RE.finditer(best[1]):
                out.append(AcademyAnswer(
                    question_id=qid, answer_text=m.group(1),
                    confidence=0.65,
                    method="path_in_match",
                    rationale=f"path in best-overlap sentence (score={best[0]:.2f})",
                ))
        for m in _PATH_LIKE_RE.finditer(body):
            out.append(AcademyAnswer(
                question_id=qid, answer_text=m.group(1),
                confidence=0.4,
                method="path_in_body",
                rationale="path-shaped token in body",
            ))
        # Filenames (e.g. ``passwd.bak``, ``config.ini``).
        for m in _FILENAME_RE.finditer(body):
            out.append(AcademyAnswer(
                question_id=qid, answer_text=m.group(1),
                confidence=0.45,
                method="filename_match",
                rationale="filename-with-extension in body",
            ))
        return out

    def _gen_inline_code_ranked(
        self, prompt: str, section: AcademySection, qid: str,
    ) -> list[AcademyAnswer]:
        """Rank inline-code spans by token overlap with question keywords.

        This is the workhorse: most academy text questions ask about a single
        command/flag/option that the section already wrote in `<code>`. We score
        each inline code span by Jaccard with the prompt; a strong match earns
        a high-confidence candidate.
        """
        if not section.inline_code:
            return []
        out: list[AcademyAnswer] = []
        prompt_tokens = _tokenize(prompt)
        # Build per-span context: each inline-code span typically appears inside
        # a sentence that defines what it does. Find that sentence.
        body = section.body_text or ""
        sentences = re.split(r"(?<=[.!?])\s+", body)
        looks_like_command = bool(_COMMAND_HINT_RE.search(prompt))
        for code in section.inline_code:
            code = code.strip()
            if not code or len(code) > 80:
                continue
            ctx = next((s for s in sentences if code in s), "")
            ctx_score = _jaccard(prompt_tokens, _tokenize(ctx)) if ctx else 0.0
            base = 0.45 if looks_like_command else 0.30
            conf = min(0.85, base + ctx_score * 0.6)
            out.append(AcademyAnswer(
                question_id=qid, answer_text=code, confidence=conf,
                method="inline_code_ranked",
                rationale=(
                    f"inline-code span; surrounding-sentence overlap={ctx_score:.2f}; "
                    f"command-hint={'yes' if looks_like_command else 'no'}"
                ),
            ))
        return out

    def _gen_sentence_anchor(
        self, prompt: str, section: AcademySection, qid: str,
    ) -> list[AcademyAnswer]:
        body = section.body_text or ""
        best = _best_sentence(prompt, body)
        if not best:
            return []
        score, sentence = best
        out: list[AcademyAnswer] = []
        # Inline codes that show up specifically in this sentence are extra
        # likely to be the literal answer (legacy strong heuristic, kept).
        for code in section.inline_code:
            if code and code in sentence:
                out.append(AcademyAnswer(
                    question_id=qid, answer_text=code,
                    confidence=min(0.8, score + 0.3),
                    method="inline_code_in_match",
                    rationale=f"inline-code in best-overlap sentence (score={score:.2f})",
                ))
        # Tail of the sentence, last resort but useful when nothing else fires.
        words = sentence.split()
        if words:
            out.append(AcademyAnswer(
                question_id=qid, answer_text=" ".join(words[-3:]),
                confidence=min(0.4, score),
                method="heuristic_text",
                rationale=f"sentence tail (jaccard={score:.2f})",
            ))
        return out

    def _gen_bullet_match(
        self, prompt: str, section: AcademySection, qid: str,
    ) -> list[AcademyAnswer]:
        if not section.bullet_lists:
            return []
        out: list[AcademyAnswer] = []
        prompt_tokens = _tokenize(prompt)
        for lst in section.bullet_lists:
            scored = [
                (_jaccard(prompt_tokens, _tokenize(item)), item)
                for item in lst if item
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, item in scored[:2]:
                if score <= 0:
                    continue
                # First word/phrase of the bullet is often the answer
                # ("ls — list directory contents" → answer is "ls").
                head = item.split("—")[0].split("-")[0].split(":")[0].strip()
                if 1 <= len(head) <= 40:
                    out.append(AcademyAnswer(
                        question_id=qid, answer_text=head,
                        confidence=min(0.55, 0.25 + score * 0.5),
                        method="bullet_head_match",
                        rationale=f"head of bullet item with overlap={score:.2f}",
                    ))
        return out

    def _gen_quoted_spans(
        self, prompt: str, section: AcademySection, qid: str,
    ) -> list[AcademyAnswer]:
        body = section.body_text or ""
        best = _best_sentence(prompt, body)
        if not best:
            return []
        score, sentence = best
        out: list[AcademyAnswer] = []
        for m in re.finditer(r'"([^"]{2,80})"', sentence):
            out.append(AcademyAnswer(
                question_id=qid, answer_text=m.group(1),
                confidence=min(0.6, 0.3 + score),
                method="quoted_in_match",
                rationale=f"\"...\" span in best-overlap sentence",
            ))
        for m in re.finditer(r"`([^`]{2,80})`", sentence):
            out.append(AcademyAnswer(
                question_id=qid, answer_text=m.group(1),
                confidence=min(0.6, 0.3 + score),
                method="backtick_in_match",
                rationale=f"`...` span in best-overlap sentence",
            ))
        return out

    def _gen_lone_inline(
        self, section: AcademySection, qid: str,
    ) -> list[AcademyAnswer]:
        if len(section.inline_code) != 1:
            return []
        return [AcademyAnswer(
            question_id=qid, answer_text=section.inline_code[0],
            confidence=0.4,
            method="lone_inline_code",
            rationale="section has a single inline-code span",
        )]

    # ---- flag (sandbox) -----------------------------------------------------

    def _answer_flag(self, question: AcademyQuestion, section: AcademySection) -> AcademyAnswer:
        if self.sandbox_runner is None:
            return AcademyAnswer(
                question_id=question.id, answer_text="", confidence=0.0,
                method="skipped",
                rationale="flag question but no sandbox_runner provided",
            )
        if not section.code_blocks and not question.hints:
            return AcademyAnswer(
                question_id=question.id, answer_text="", confidence=0.0,
                method="skipped",
                rationale="flag question but no code block / hint to run",
            )
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
                question_id=question.id, answer_text="", confidence=0.0,
                method="sandbox_cmd",
                rationale=f"sandbox command raised: {exc}",
                sandbox_command=cmd,
            )
        m = _HTB_FLAG_RE.search(out)
        if m:
            return AcademyAnswer(
                question_id=question.id, answer_text=m.group(0),
                confidence=0.95,
                method="sandbox_cmd",
                rationale="found HTB{...} pattern in sandbox output",
                sandbox_command=cmd, sandbox_output=out[:4096],
            )
        m2 = _FLAG_LINE_RE.search(out)
        if m2:
            return AcademyAnswer(
                question_id=question.id, answer_text=m2.group(0),
                confidence=0.55,
                method="sandbox_cmd",
                rationale="found token-only line in sandbox output",
                sandbox_command=cmd, sandbox_output=out[:4096],
            )
        return AcademyAnswer(
            question_id=question.id, answer_text="", confidence=0.0,
            method="sandbox_cmd",
            rationale="no flag-shaped output from sandbox",
            sandbox_command=cmd, sandbox_output=out[:4096],
        )


def _dedupe_keep_best(candidates: list[AcademyAnswer]) -> list[AcademyAnswer]:
    """Keep the highest-confidence record for each unique answer_text.

    Sort the result descending by confidence, then by method (stable for tests).
    """
    by_text: dict[str, AcademyAnswer] = {}
    for c in candidates:
        if not c.answer_text:
            continue
        key = c.answer_text.strip().lower()
        if key not in by_text or c.confidence > by_text[key].confidence:
            by_text[key] = c
    return sorted(
        by_text.values(),
        key=lambda a: (-a.confidence, a.method, a.answer_text),
    )
