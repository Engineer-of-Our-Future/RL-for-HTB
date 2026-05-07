"""Multi-module academy walker (study-only).

Walks **every** owned/in_progress academy module the user has access to,
sequentially, using the same single-module plumbing as
``scripts/htb_academy_run.py`` (the per-module helpers in
``htbrl.academy.cdp_walker``). The point is to bulk-build training demos
across the entire owned curriculum with a single CDP attach.

Hard rules:

  * **study_only=True is hard-coded.** This walker never clicks Submit on
    the page and therefore never moves the cube balance. The wizard
    (``htb_academy_wizard.py``) is the only path that auto-submits answers.
  * **Unlock gate enforced between modules.** Per the operator rule,
    "open new module only if all questions are answered and cube balance
    are updated", we run :func:`check_unlock_gate` after every module.
    Because study-only never spends/earns cubes, the cube-refresh half is
    OFF; the all-questions-answered half stays ON so we don't silently
    leave gaps in the demo record.
  * **One CDP attach reused across all modules.** Opening a websocket
    costs ~5s; doing that per module on a 50-module curriculum is wasted
    budget. We attach once, walk many modules, close once.
  * **Skip-existing-demos by default.** If a demo for module N already
    lives in ``--auto-demo-dir``, we skip the walk for module N (use
    ``--force`` to re-walk).

Hard requirements before running:

  1. ``scripts/start_chrome_for_htb.ps1`` is running and the operator is
     logged into HTB Academy in that Chrome window.
  2. The Chrome debug endpoint is reachable at ``$HTBRL_ACADEMY_CDP``
     (default ``http://127.0.0.1:9222``).

Use ``--dry-run`` to preview the walk plan without opening any module.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from htbrl.academy.answerer import HeuristicAnswerer
from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.cdp_walker import (
    CDPClient,
    build_section_from_scrape,
    click_next,
    enter_module,
    fetch_module_via_api,
    go_to_first_section,
    open_cdp,
    parse_cheatsheet_markdown,
    read_cube_balance,
    scrape_section,
)
from htbrl.academy.curriculum import check_unlock_gate
from htbrl.academy.mitre_mapping import techniques_for_module
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySection,
    QuestionType,
)
from htbrl.data.demo_dataset import save_demonstration


# Default include states for the academy modules listing. The user's owned
# curriculum is the union of these two: "owned" = paid for / in plan,
# "in_progress" = currently being worked on (a strict subset of owned, but
# the API returns it as a separate state).
DEFAULT_INCLUDE_STATES = "owned,in_progress"


# ---- module listing ---------------------------------------------------------


_MODULES_LIST_JS = r"""
(async () => {
    try {
        const r = await fetch('/api/v2/modules', {credentials: 'include'});
        if (!r.ok) return {__error: 'http ' + r.status};
        const j = await r.json();
        // /api/v2/modules returns either {data: [...]} or {data: {modules: [...]}}.
        // Normalize to a flat list.
        let mods = j.data;
        if (mods && !Array.isArray(mods) && Array.isArray(mods.modules)) {
            mods = mods.modules;
        }
        if (!Array.isArray(mods)) return {__error: 'unexpected shape', shape: typeof mods};
        return {modules: mods};
    } catch (e) {
        return {__error: String(e)};
    }
})()
"""


@dataclass
class ModuleListEntry:
    """One entry from /api/v2/modules, narrowed to fields we need."""

    id: int
    title: str
    tier: int
    cubes_to_unlock: int
    state: str  # "owned" | "in_progress" | "completed" | "locked" | ...

    @classmethod
    def from_api(cls, raw: dict) -> "ModuleListEntry | None":
        try:
            mid = int(raw.get("id"))
        except (TypeError, ValueError):
            return None
        # State is reported under several keys depending on the API
        # version; fall back gracefully.
        state = (
            raw.get("state")
            or raw.get("user_state")
            or raw.get("progress_state")
            or ""
        )
        if isinstance(state, dict):
            state = state.get("state", "") or state.get("name", "") or ""
        state = str(state).lower().strip()
        # Some payloads expose progress_percentage / completion_percentage
        # instead of explicit state. If we have neither, infer:
        #   completed = 100, in_progress = (0, 100), owned = 0.
        if not state:
            pct = raw.get("progress_percentage") or raw.get("completion_percentage") or 0
            try:
                pct = float(pct)
            except (TypeError, ValueError):
                pct = 0.0
            if pct >= 100.0:
                state = "completed"
            elif pct > 0.0:
                state = "in_progress"
            elif raw.get("owned") or raw.get("is_owned"):
                state = "owned"
            else:
                state = "locked"
        # HTB sometimes returns tier as a structured object
        # ``{"id": 1, "name": "Tier 1"}`` instead of a flat int. Normalize.
        raw_tier = raw.get("tier")
        if isinstance(raw_tier, dict):
            tier_val = raw_tier.get("id") or raw_tier.get("level") or 0
        else:
            tier_val = raw_tier or 0
        try:
            tier_int = int(tier_val)
        except (TypeError, ValueError):
            tier_int = 0
        return cls(
            id=mid,
            title=str(raw.get("name") or raw.get("title") or f"Module {mid}"),
            tier=tier_int,
            cubes_to_unlock=int(
                raw.get("cubes_to_unlock") or raw.get("cubes") or raw.get("price") or 0
            ),
            state=state,
        )


def fetch_modules_list(cdp: CDPClient) -> tuple[list[ModuleListEntry], str | None]:
    """Fetch ``/api/v2/modules`` from inside the authenticated page.

    Returns ``(entries, error)``. On success ``error`` is None and entries
    is a list of :class:`ModuleListEntry`. On failure (network / API
    shape) entries is ``[]`` and error carries the diagnostic.
    """
    try:
        v = cdp.evaluate(_MODULES_LIST_JS, await_promise=True)
    except Exception as exc:
        return [], f"cdp evaluate failed: {exc}"
    if not isinstance(v, dict):
        return [], f"unexpected return shape: {type(v).__name__}"
    if v.get("__error"):
        return [], str(v.get("__error"))
    raw_list = v.get("modules") or []
    out: list[ModuleListEntry] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        entry = ModuleListEntry.from_api(raw)
        if entry is not None:
            out.append(entry)
    return out, None


def filter_and_sort_modules(
    entries: Iterable[ModuleListEntry],
    include_states: set[str],
) -> list[ModuleListEntry]:
    """Filter modules to ``include_states`` and sort by (tier, cubes, id)."""
    pool = [e for e in entries if e.state in include_states]
    pool.sort(key=lambda e: (e.tier, e.cubes_to_unlock, e.id))
    return pool


# ---- single-module walk -----------------------------------------------------


@dataclass
class ModuleWalkResult:
    """Per-module summary of a single walk."""

    module_id: int
    title: str
    sections_walked: int
    n_questions: int
    n_attempted: int
    n_turns: int
    cheat_rows: int
    techniques: list[str] = field(default_factory=list)
    gate_allowed: bool = True
    gate_reason: str = ""
    demo_path: Path | None = None
    skipped_reason: str = ""  # set if we skipped (existing demo, API error, etc.)
    error: str = ""           # set on hard failure


def walk_one_module(
    cdp: CDPClient,
    module_id: int,
    *,
    auto_demo_dir: Path,
    max_sections: int,
    answerer: HeuristicAnswerer,
    cdp_endpoint: str,
    section_sleep_s: float = 2.0,
) -> ModuleWalkResult:
    """Walk one academy module via CDP and write a study-only demo.

    Mirrors the per-module flow in ``scripts/htb_academy_run.py``, but
    surfaces all the per-module counters into a :class:`ModuleWalkResult`
    so the multi-module driver can summarize the run without scraping log
    lines.

    The unlock-gate evaluation (the half that ALWAYS applies in study-only
    mode: "every question must have an attempt") is performed by the
    caller, not here, because it needs the cube balance read between
    modules.
    """
    sections: list[AcademySection] = []
    seen_ids: set[str] = set()
    cheat_sheet_rows: list[dict[str, str]] = []
    api_prelude = ""
    api_conclusion = ""
    api_takeaways = ""
    api_title = ""

    api = fetch_module_via_api(cdp, module_id)
    if api and isinstance(api, dict) and not api.get("__error"):
        cheat_md = api.get("cheatsheet") or ""
        cheat_sheet_rows = parse_cheatsheet_markdown(cheat_md)
        api_prelude = (api.get("prelude") or "").strip()
        api_conclusion = (api.get("conclusion") or "").strip()
        api_takeaways = (api.get("takeaways") or "").strip()
        api_title = (api.get("name") or "").strip()
        print(f"[walk-all]   API metadata: cheat_rows={len(cheat_sheet_rows)} "
              f"prelude={len(api_prelude)} title={api_title!r}")
    else:
        # Soft-fail: a missing API doesn't necessarily mean the module is
        # walkable - we still try to enter and scrape sections from the DOM.
        print(f"[walk-all]   API metadata unavailable: {api}")

    entered, url = enter_module(cdp, module_id)
    print(f"[walk-all]   in section view: {url}  (entered={entered})")
    if not entered:
        # Don't crash the whole driver - just record the skip and let the
        # caller decide whether to halt.
        return ModuleWalkResult(
            module_id=module_id,
            title=api_title or f"Module {module_id}",
            sections_walked=0, n_questions=0, n_attempted=0, n_turns=0,
            cheat_rows=len(cheat_sheet_rows),
            skipped_reason="enter_module returned False (module locked or unloaded)",
        )

    rewound = go_to_first_section(cdp)
    print(f"[walk-all]   rewound to first section: {rewound}")

    for hop in range(max_sections):
        scraped = scrape_section(cdp)
        section = build_section_from_scrape(scraped)
        if section.id in seen_ids:
            print(f"[walk-all]   section {section.id!r} already seen; stopping walk")
            break
        seen_ids.add(section.id)
        sections.append(section)
        print(
            f"[walk-all]   sec {section.section_index}/{section.section_total} "
            f"{section.title!r} body={len(section.body_text)} "
            f"questions={len(section.questions)}"
        )
        if section.section_total and section.section_index >= section.section_total:
            print("[walk-all]   reached final section")
            break
        if not click_next(cdp):
            print("[walk-all]   no Next button; stopping walk")
            break
        time.sleep(section_sleep_s)

    module = AcademyModule(
        id=str(module_id),
        title=api_title or f"Module {module_id}",
        tier=0,
        sections=sections,
        category="general",
        cheat_sheet=cheat_sheet_rows,
        prelude=api_prelude,
        conclusion=api_conclusion,
        takeaways=api_takeaways,
    )

    # Heuristic answerer over every question. study_only=True means
    # ``accepted=False`` for all; the demo carries the model's *proposal*
    # without ever submitting it to HTB.
    submissions: list[tuple[str, AcademyAnswer, bool]] = []
    n_questions = sum(len(s.questions) for s in sections)
    if n_questions == 0:
        print(f"[walk-all]   {len(sections)} section(s); module has no questions")
    else:
        print(f"[walk-all]   running answerer on {n_questions} question(s)")
    for s in sections:
        for q in s.questions:
            ans = answerer.answer(q, s, module=module)
            submissions.append((str(module_id), ans, False))

    techniques = list(techniques_for_module(module))
    demo = session_to_demonstration(
        module, submissions, study_only=True,
        extra_metadata={
            "cdp_endpoint": cdp_endpoint,
            "walk_all": True,
        },
    )
    auto_demo_dir.mkdir(parents=True, exist_ok=True)
    out = auto_demo_dir / f"academy_module_{module_id}.msgpack.gz"
    save_demonstration(demo, out, compress=True)

    return ModuleWalkResult(
        module_id=module_id,
        title=module.title,
        sections_walked=len(sections),
        n_questions=n_questions,
        n_attempted=len(submissions),
        n_turns=len(demo.turns),
        cheat_rows=len(cheat_sheet_rows),
        techniques=techniques,
        demo_path=out,
    )


# ---- main loop --------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cdp",
        default=os.environ.get("HTBRL_ACADEMY_CDP", "http://127.0.0.1:9222"),
        help="Chrome DevTools endpoint (default $HTBRL_ACADEMY_CDP or http://127.0.0.1:9222)",
    )
    p.add_argument(
        "--auto-demo-dir", type=Path, default=Path("data/auto_demos"),
        help="where to write per-module .msgpack.gz demos (default data/auto_demos)",
    )
    p.add_argument(
        "--max-modules", type=int, default=50,
        help="cap on the number of modules to walk in this run (default 50)",
    )
    p.add_argument(
        "--max-sections-per-module", type=int, default=50,
        help="hard cap on sections walked per module (default 50)",
    )
    p.add_argument(
        "--include-states", default=DEFAULT_INCLUDE_STATES,
        help=(
            "comma-separated module states to include "
            f"(default {DEFAULT_INCLUDE_STATES!r}). "
            "Common values: owned, in_progress, completed, locked."
        ),
    )
    p.add_argument(
        "--force", action="store_true",
        help="re-walk modules whose demo file already exists in --auto-demo-dir.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="list what we would walk without opening any module or writing demos.",
    )
    p.add_argument(
        "--module-sleep-s", type=float, default=5.0,
        help="seconds to sleep between modules to be respectful of the academy "
             "rate limit (default 5.0)",
    )
    p.add_argument(
        "--section-sleep-s", type=float, default=2.0,
        help="seconds to sleep between section-Next clicks (default 2.0)",
    )
    return p


def _existing_demo_for(auto_demo_dir: Path, module_id: int) -> Path | None:
    """Return the existing study-only demo path for ``module_id`` or None."""
    p = auto_demo_dir / f"academy_module_{module_id}.msgpack.gz"
    return p if p.exists() else None


def _print_plan(plan: list[ModuleListEntry], skipped: list[tuple[ModuleListEntry, str]]) -> None:
    print(f"[walk-all] PLAN: {len(plan)} module(s) to walk, "
          f"{len(skipped)} skipped (already-have-demo / out-of-state)")
    for e in plan:
        print(f"[walk-all]   WALK   id={e.id:<5d} tier={e.tier} cubes={e.cubes_to_unlock:<3d} "
              f"state={e.state!s:<14s} {e.title!r}")
    for e, reason in skipped:
        print(f"[walk-all]   SKIP   id={e.id:<5d} tier={e.tier} cubes={e.cubes_to_unlock:<3d} "
              f"state={e.state!s:<14s} {e.title!r}  ({reason})")


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    include_states = {s.strip().lower() for s in args.include_states.split(",") if s.strip()}
    print(f"[walk-all] cdp={args.cdp} include_states={sorted(include_states)} "
          f"auto_demo_dir={args.auto_demo_dir} max_modules={args.max_modules} "
          f"force={args.force} dry_run={args.dry_run}")

    try:
        cdp, ws, ws_url = open_cdp(args.cdp)
    except Exception as exc:
        print(f"[walk-all] failed to attach to {args.cdp}: {exc}")
        print("[walk-all] is start_chrome_for_htb.ps1 running?")
        return 2
    print(f"[walk-all] attached to {ws_url}")

    results: list[ModuleWalkResult] = []
    try:
        entries, list_err = fetch_modules_list(cdp)
        if list_err:
            print(f"[walk-all] failed to fetch /api/v2/modules: {list_err}")
            return 3
        print(f"[walk-all] fetched {len(entries)} module(s) from /api/v2/modules")

        plan = filter_and_sort_modules(entries, include_states)
        # Cap by max-modules BEFORE applying skip-existing so the user
        # always sees the full top-N candidate list in --dry-run.
        plan = plan[: args.max_modules]

        # Partition into walk-now / skip-existing.
        walks: list[ModuleListEntry] = []
        skipped: list[tuple[ModuleListEntry, str]] = []
        for e in plan:
            existing = _existing_demo_for(args.auto_demo_dir, e.id)
            if existing and not args.force:
                skipped.append((e, f"existing demo {existing.name}"))
            else:
                walks.append(e)
        _print_plan(walks, skipped)

        if args.dry_run:
            print("[walk-all] --dry-run set; not opening any module.")
            return 0

        answerer = HeuristicAnswerer()

        for i, entry in enumerate(walks):
            print(f"\n[walk-all] === MODULE {i+1}/{len(walks)}: "
                  f"id={entry.id} {entry.title!r} ===")

            cubes_before = read_cube_balance(cdp)
            print(f"[walk-all]   cubes_balance before: {cubes_before}")

            try:
                result = walk_one_module(
                    cdp, entry.id,
                    auto_demo_dir=args.auto_demo_dir,
                    max_sections=args.max_sections_per_module,
                    answerer=answerer,
                    cdp_endpoint=args.cdp,
                    section_sleep_s=args.section_sleep_s,
                )
            except Exception as exc:  # noqa: BLE001 - bubble up the failure as an error result
                print(f"[walk-all]   ERROR walking module {entry.id}: {exc!r}")
                result = ModuleWalkResult(
                    module_id=entry.id, title=entry.title,
                    sections_walked=0, n_questions=0, n_attempted=0, n_turns=0,
                    cheat_rows=0, error=repr(exc),
                )

            results.append(result)

            cubes_after = read_cube_balance(cdp)
            print(f"[walk-all]   cubes_balance after:  {cubes_after}")
            print(f"[walk-all]   summary id={result.module_id} | "
                  f"{result.title!r} | sections={result.sections_walked} | "
                  f"n_q={result.n_questions} | n_attempted={result.n_attempted} | "
                  f"turns={result.n_turns} | cheat_rows={result.cheat_rows}")

            # Unlock-gate check: study_only never moves cubes, so the
            # cube-refresh half is OFF. The all-questions-answered half
            # always applies (we want every question to have a heuristic
            # answer logged, even a wrong one).
            if result.error or result.skipped_reason:
                # Skip the gate for hard failures - we couldn't walk it.
                print(f"[walk-all]   gate skipped (module not walked): "
                      f"{result.error or result.skipped_reason}")
                # Still proceed to the next module; the user can re-run later.
                time.sleep(args.module_sleep_s)
                continue

            # walk_one_module returns counts, not question ids. Build a
            # stub module + question-id set that matches the gate's only
            # interest: "did n_attempted reach n_questions?". When yes, we
            # mark every stub q as answered; when no, the gate trips.
            stub_questions = [
                AcademyQuestion(id=f"q-stub-{j}", prompt="", type=QuestionType.TEXT)
                for j in range(result.n_questions)
            ]
            stub_section = AcademySection(
                id="sec-stub", title="", body_text="", questions=stub_questions,
            )
            stub_module = AcademyModule(
                id=str(result.module_id), title=result.title,
                tier=0, sections=[stub_section], category="general",
            )
            attempted_qids = (
                {q.id for q in stub_questions}
                if result.n_attempted >= result.n_questions
                else set()
            )
            gate = check_unlock_gate(
                current_module=stub_module,
                answered_question_ids=attempted_qids,
                cube_balance_before=cubes_before if cubes_before is not None else 0,
                cube_balance_after=cubes_after if cubes_after is not None else 0,
                require_all_answered=True,
                require_cube_refresh=False,
            )
            result.gate_allowed = gate.allowed
            result.gate_reason = gate.reason
            print(f"[walk-all]   gate: allowed={gate.allowed} - {gate.reason}")

            # Operator rule: if some questions exist and we attempted >0
            # but didn't reach all, STOP the loop. This lets the operator
            # investigate (e.g. a sandbox-required question that the
            # study-only walker can't address) before opening a new
            # module.
            if (
                not gate.allowed
                and result.n_questions > 0
                and 0 < result.n_attempted < result.n_questions
            ):
                print(f"[walk-all]   STOP: partial attempt on module {entry.id} "
                      f"({result.n_attempted}/{result.n_questions}); "
                      f"refusing to open further modules.")
                break

            # Polite delay between modules.
            time.sleep(args.module_sleep_s)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    _print_overall_summary(results)
    return 0


def _print_overall_summary(results: list[ModuleWalkResult]) -> None:
    total_modules = len(results)
    total_walked = sum(1 for r in results if r.demo_path is not None)
    total_turns = sum(r.n_turns for r in results)
    total_questions = sum(r.n_questions for r in results)
    total_attempts = sum(r.n_attempted for r in results)
    total_cheat = sum(r.cheat_rows for r in results)
    techs: set[str] = set()
    for r in results:
        techs.update(r.techniques)
    print()
    print("=" * 78)
    print("[walk-all] OVERALL SUMMARY")
    print(f"[walk-all]   modules_seen           : {total_modules}")
    print(f"[walk-all]   modules_walked         : {total_walked}")
    print(f"[walk-all]   total_turns            : {total_turns}")
    print(f"[walk-all]   total_questions        : {total_questions}")
    print(f"[walk-all]   total_attempts         : {total_attempts}")
    print(f"[walk-all]   total_cheat_sheet_rows : {total_cheat}")
    print(f"[walk-all]   distinct_mitre_techs   : {len(techs)} "
          f"({', '.join(sorted(techs)[:8])}{'...' if len(techs) > 8 else ''})")
    closed = [r for r in results if not r.gate_allowed]
    if closed:
        print(f"[walk-all]   gate_closed_modules    : {len(closed)}")
        for r in closed:
            print(f"[walk-all]     id={r.module_id} - {r.gate_reason}")
    errored = [r for r in results if r.error]
    if errored:
        print(f"[walk-all]   errored_modules        : {len(errored)}")
        for r in errored:
            print(f"[walk-all]     id={r.module_id} - {r.error}")
    skipped = [r for r in results if r.skipped_reason]
    if skipped:
        print(f"[walk-all]   skipped_modules        : {len(skipped)}")
        for r in skipped:
            print(f"[walk-all]     id={r.module_id} - {r.skipped_reason}")


if __name__ == "__main__":
    sys.exit(main())
