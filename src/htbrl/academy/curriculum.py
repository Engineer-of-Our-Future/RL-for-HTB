"""Module ordering / curriculum logic (path-aware).

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
    (category_rank, tier, cubes_to_unlock, id)

``preferred_order=[id, ...]`` overrides the curriculum if any of those ids
are eligible.
"""

from __future__ import annotations

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
) -> AcademyModule | None:
    """Return the next module to attempt, or None if nothing's eligible.

    Sort key: (category_rank, tier, cubes_to_unlock, id). general(0) before
    offensive(1) before defensive(2) before other(3), matching the user's
    real-world learning order.
    """
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
