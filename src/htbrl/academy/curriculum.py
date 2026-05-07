"""Module ordering / curriculum logic.

We don't hard-code the academy's module list - we just provide the sorting
logic the orchestrator uses to pick the next module:

1. Modules with all prerequisites satisfied
2. Sorted by tier (lowest first), then by cubes_to_unlock (cheapest first)
3. Excluding already-completed modules
4. Affordable given the user's current cubes balance

If you want to override the order, supply a ``preferred_order`` list to
``next_module``: any module in that list goes first, in the given order, as
long as it's eligible.
"""

from __future__ import annotations

from htbrl.academy.page_models import AcademyModule, ProgressState


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


def next_module(
    modules: list[AcademyModule],
    state: ProgressState,
    preferred_order: list[str] | None = None,
) -> AcademyModule | None:
    """Return the next module to attempt, or None if nothing's eligible."""
    pool = eligible_modules(modules, state)
    if not pool:
        return None
    if preferred_order:
        for mid in preferred_order:
            for m in pool:
                if m.id == mid:
                    return m
    pool.sort(key=lambda m: (m.tier, m.cubes_to_unlock, m.id))
    return pool[0]


def progress_summary(
    modules: list[AcademyModule], state: ProgressState
) -> dict[str, object]:
    completed = [m for m in modules if state.has_completed(m.id)]
    eligible = eligible_modules(modules, state)
    return {
        "n_total": len(modules),
        "n_completed": len(completed),
        "n_eligible": len(eligible),
        "completion_pct": (
            len(completed) / len(modules) * 100.0 if modules else 0.0
        ),
        "cubes_balance": state.cubes_balance,
        "in_progress": state.in_progress_module_id,
    }
