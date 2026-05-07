"""Module ordering / curriculum logic (path-aware) + unlock gate.

The user's directive (Phase 5b): the agent MUST work modules in real-world
learning order:

  1. **General / fundamentals** (Linux, Windows, networking, intros)
  2. **Offensive** (pentest, exploit, web attacks, AD attacks)
  3. **Defensive** (SOC, blue team, incident response, detection)
  4. **Other** (anything that doesn't classify above)

This module owns:
- ``classify_module`` - heuristic on title/description + tier-0 fast path.
  Result is cached on ``AcademyModule.category`` once the scraper / fixture
  loader assigns it.
- ``next_module`` - among eligible modules, sort by:
    (category_rank, tier, cubes_to_unlock, id). Optionally constrained by
    an ``unlock_gate`` so we never open a *new* module before the current
    in-progress one is fully attempted and the cube balance has refreshed.
- ``UnlockGate`` / ``check_unlock_gate`` - the gating policy itself.

``preferred_order=[id, ...]`` overrides the curriculum if any of those ids
are eligible AND the unlock gate is satisfied.

The unlock gate exists because the academy charges cubes to *open* a module.
Burning cubes by opening a second module before the first one is finished
wastes the research-account's budget; the gate enforces the rule:

  "open new module only if all questions are answered AND cube balance are
   updated" (operator's words)

Both halves matter:
  * "all questions answered" - even a wrong answer counts as an attempt.
    The point is that no question is left silently un-attempted, so the demo
    record is complete.
  * "cube balance updated" - the academy's reward signal must have actually
    landed before we move on. If we read state immediately after submitting
    and the balance is unchanged, the academy may not have committed the
    reward yet (e.g. still showing the in-progress submission); jumping to a
    new module risks overlapping cube-spend.
"""

from __future__ import annotations

from dataclasses import dataclass

from htbrl.academy.page_models import AcademyModule, ProgressState


# Lower rank = earlier in the curriculum.
CATEGORY_RANK = {
    "general":   0,
    "offensive": 1,
    "defensive": 2,
    "other":     3,
}


_GENERAL_KEYWORDS = (
    "fundamental", "basics", "introduction", "intro to", "getting started",
    "primer", "linux ", "windows ", "networking", "shell command",
    "command line", "programming", "system internals",
    "html ", "javascript", "python", "bash", "regular expression",
    "academy", "learning",
)
_OFFENSIVE_KEYWORDS = (
    "pentest", "penetration", "exploit", "exploitation", "attack", "offensive",
    "red team", "bug bounty", "bypass", "evasion", "shellcoding",
    "buffer overflow", "privilege escalation", "lateral movement", "kerberoast",
    "active directory", "web attack", "api attack", "phishing", "post-exploit",
)
_DEFENSIVE_KEYWORDS = (
    "soc", "blue team", "incident", "defense", "defensive", "detection",
    "forensic", "dfir", "siem", "threat hunting", "malware analysis",
    "intrusion detection", "intelligence", "monitoring", "yara", "sigma",
    "log analysis",
)


def classify_module(module: AcademyModule) -> str:
    """Return one of 'general' | 'offensive' | 'defensive' | 'other'.

    Tier 0 = always 'general' (HTB convention: tier 0 is foundational).
    For higher tiers we keyword-match the title + description.
    """
    if module.tier == 0:
        return "general"
    text = f"{module.title} {module.description}".lower()
    if any(kw in text for kw in _GENERAL_KEYWORDS):
        return "general"
    if any(kw in text for kw in _OFFENSIVE_KEYWORDS):
        return "offensive"
    if any(kw in text for kw in _DEFENSIVE_KEYWORDS):
        return "defensive"
    return "other"


def is_eligible(module: AcademyModule, state: ProgressState) -> bool:
    if state.has_completed(module.id):
        return False
    if state.cubes_balance < module.cubes_to_unlock:
        return False
    return all(state.has_completed(p) for p in module.prerequisites)


def eligible_modules(
    modules: list[AcademyModule], state: ProgressState
) -> list[AcademyModule]:
    return [m for m in modules if is_eligible(m, state)]


def _category_rank(module: AcademyModule) -> int:
    cat = module.category if module.category in CATEGORY_RANK else classify_module(module)
    return CATEGORY_RANK.get(cat, CATEGORY_RANK["other"])


def next_module(
    modules: list[AcademyModule],
    state: ProgressState,
    preferred_order: list[str] | None = None,
    *,
    unlock_gate: "UnlockGateResult | None" = None,
) -> AcademyModule | None:
    """Return the next module to attempt, or None if nothing's eligible.

    Sort key: (category_rank, tier, cubes_to_unlock, id). general(0) before
    offensive(1) before defensive(2) before other(3), matching the user's
    real-world learning order.

    If ``unlock_gate`` is supplied and ``unlock_gate.allowed`` is False,
    ``next_module`` returns None - the orchestrator must finish the current
    module (and let the cube balance refresh) before opening a new one.
    """
    if unlock_gate is not None and not unlock_gate.allowed:
        return None
    pool = eligible_modules(modules, state)
    if not pool:
        return None
    if preferred_order:
        for mid in preferred_order:
            for m in pool:
                if m.id == mid:
                    return m
    pool.sort(key=lambda m: (_category_rank(m), m.tier, m.cubes_to_unlock, m.id))
    return pool[0]


# ---- unlock gate ------------------------------------------------------------


@dataclass
class UnlockGateResult:
    """Outcome of :func:`check_unlock_gate`.

    Carries the verdict (``allowed``), a human-readable ``reason`` for logs,
    and the underlying counts so callers can decide whether to retry or warn.
    """

    allowed: bool
    reason: str
    questions_total: int = 0
    questions_unanswered: int = 0
    cube_balance_delta: int = 0


def check_unlock_gate(
    current_module: AcademyModule | None,
    answered_question_ids: set[str] | list[str] | tuple[str, ...] | None,
    cube_balance_before: int,
    cube_balance_after: int,
    *,
    require_all_answered: bool = True,
    require_cube_refresh: bool = True,
) -> UnlockGateResult:
    """Decide whether the orchestrator may open a NEW module.

    Implements the operator rule: "open new module only if all questions are
    answered and cube balance are updated". Both halves are independently
    toggleable via ``require_all_answered`` and ``require_cube_refresh`` so a
    partial walk (e.g. sandbox unavailable) can still proceed if the operator
    knows what they're doing.

    Args:
      current_module: the module currently being attempted (None means no
          gating - first module of a fresh run is always allowed).
      answered_question_ids: which question ids have a recorded submission
          (correct or wrong; empty/skipped don't count).
      cube_balance_before: cubes shown by the academy *before* the run started.
      cube_balance_after: cubes shown *after* answering attempts and a state
          refresh.
      require_all_answered: if True, every question in ``current_module``
          must appear in ``answered_question_ids``.
      require_cube_refresh: if True, ``cube_balance_after`` must differ from
          ``cube_balance_before`` when the module has at least one question
          (the cube reward signal must have landed). If the module has no
          questions (theory-only), this check is skipped automatically.
    """
    if current_module is None:
        return UnlockGateResult(allowed=True, reason="no current module - first run")

    answered = set(answered_question_ids or ())
    all_q = current_module.all_questions
    n_q = len(all_q)
    n_answered = sum(1 for q in all_q if q.id in answered)
    n_unanswered = n_q - n_answered
    delta = cube_balance_after - cube_balance_before

    if require_all_answered and n_unanswered > 0:
        return UnlockGateResult(
            allowed=False,
            reason=(
                f"gate closed: {n_unanswered}/{n_q} questions still unanswered "
                f"in module {current_module.id!r}"
            ),
            questions_total=n_q,
            questions_unanswered=n_unanswered,
            cube_balance_delta=delta,
        )

    # Cube refresh only matters when there's something to be rewarded for.
    if require_cube_refresh and n_q > 0 and delta == 0:
        return UnlockGateResult(
            allowed=False,
            reason=(
                f"gate closed: cube balance unchanged "
                f"({cube_balance_before}->{cube_balance_after}) after answering "
                f"{n_q} question(s) in {current_module.id!r}; refresh state and retry"
            ),
            questions_total=n_q,
            questions_unanswered=0,
            cube_balance_delta=delta,
        )

    return UnlockGateResult(
        allowed=True,
        reason=(
            f"gate open: {n_q} question(s) answered, cube delta {delta:+d} "
            f"({cube_balance_before}->{cube_balance_after})"
        ),
        questions_total=n_q,
        questions_unanswered=0,
        cube_balance_delta=delta,
    )


def progress_summary(
    modules: list[AcademyModule], state: ProgressState
) -> dict[str, object]:
    completed = [m for m in modules if state.has_completed(m.id)]
    eligible = eligible_modules(modules, state)
    by_cat: dict[str, dict[str, int]] = {
        "general": {"total": 0, "done": 0},
        "offensive": {"total": 0, "done": 0},
        "defensive": {"total": 0, "done": 0},
        "other": {"total": 0, "done": 0},
    }
    for m in modules:
        cat = m.category if m.category in by_cat else classify_module(m)
        by_cat.setdefault(cat, {"total": 0, "done": 0})["total"] += 1
        if state.has_completed(m.id):
            by_cat[cat]["done"] += 1
    return {
        "n_total": len(modules),
        "n_completed": len(completed),
        "n_eligible": len(eligible),
        "completion_pct": (
            len(completed) / len(modules) * 100.0 if modules else 0.0
        ),
        "cubes_balance": state.cubes_balance,
        "in_progress": state.in_progress_module_id,
        "by_category": by_cat,
    }
