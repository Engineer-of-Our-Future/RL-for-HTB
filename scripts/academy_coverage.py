"""Print a coverage report for HTB Academy auto-learner demos.

Walks ``data/auto_demos/academy_module_*.msgpack.gz`` (override with
``--demos-dir``) and aggregates per-module + cross-module statistics so
the user can see at a glance what's been harvested and what's missing.

The report covers:

* Per-module table: sections / questions / attempts / accepted /
  theory size / distinct ATT&CK techniques touched.
* Cross-module method-tag totals (e.g. ``inline_code_in_match=42``,
  ``wizard_skip_low_conf=130``).
* Cross-module distinct ``techniques_attempted`` set.

Run:
    python scripts/academy_coverage.py
    python scripts/academy_coverage.py --json
    python scripts/academy_coverage.py --min-questions 10

Exit code is always 0 unless an unexpected error occurs - this is a
read-only inspection tool, not a CI gate. Demos that fail to load are
warned about on stderr and skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from htbrl.data.demo_dataset import Demonstration, load_demonstration


_DEFAULT_DEMOS_DIR = Path("data/auto_demos")
_DEMO_GLOB = "academy_module_*.msgpack.gz"
_TITLE_TRUNC = 30


def _module_title(demo: Demonstration) -> str:
    """Best-effort module title for a demo.

    The auto_demo_writer doesn't currently stash the module title in
    metadata, so we fall back to the first ``academy_section_read`` turn's
    ``title`` slot (which is the section's title - close enough for an
    at-a-glance table) or, failing that, the ``target_id``.
    """
    for t in demo.turns:
        if t.action_tool_name == "academy_section_read":
            title = t.action_slots.get("title") or ""
            if title:
                return str(title)
    return demo.target_id


def _n_sections(demo: Demonstration) -> int:
    """Number of distinct sections this demo covers.

    Each ``academy_section_read`` turn corresponds to one section. We also
    expose ``section_total`` from the slots, but that's the module-wide
    section count, not the count actually walked - the demo turn count is
    the right answer here.
    """
    return sum(1 for t in demo.turns if t.action_tool_name == "academy_section_read")


def _n_section_total(demo: Demonstration) -> int:
    """Module's total section count (academy view), if recorded."""
    for t in demo.turns:
        if t.action_tool_name == "academy_section_read":
            v = t.action_slots.get("section_total")
            if isinstance(v, int):
                return v
    return _n_sections(demo)


def _n_accepted(demo: Demonstration) -> int:
    """How many answers landed as accepted by the academy.

    Wizard demos record ``n_accepted`` in metadata directly. For older /
    non-wizard demos we approximate by counting answer turns whose
    ``cubes_reward > 0`` (the academy's positive-acceptance signal).
    """
    md_n = demo.metadata.get("n_accepted")
    if isinstance(md_n, int):
        return md_n
    n = 0
    for t in demo.turns:
        if t.action_tool_name != "academy_answer":
            continue
        if int(t.action_slots.get("cubes_reward", 0) or 0) > 0:
            n += 1
    return n


def _theory_bytes(demo: Demonstration) -> int:
    return sum(len(t.obs_text.encode("utf-8")) for t in demo.turns)


def _distinct_techs(
    demo: Demonstration, key: str = "techniques_attempted"
) -> set[str]:
    out: set[str] = set()
    for t in demo.turns:
        for x in getattr(t, key, []):
            out.add(x)
    return out


def _summarize_demo(demo: Demonstration, *, path: Path) -> dict[str, Any]:
    """Produce a flat dict of per-demo stats."""
    md = demo.metadata
    method_counts: Counter[str] = Counter()
    n_section_read = 0
    n_answer = 0
    for t in demo.turns:
        if t.action_tool_name == "academy_section_read":
            n_section_read += 1
        elif t.action_tool_name == "academy_answer":
            n_answer += 1
        m = t.action_slots.get("method") if isinstance(t.action_slots, dict) else None
        if m:
            method_counts[str(m)] += 1
    techs_attempted = _distinct_techs(demo, "techniques_attempted")
    techs_succeeded = _distinct_techs(demo, "techniques_succeeded")
    return {
        "path": str(path),
        "module_id": str(md.get("module_id", "")),
        "module_tier": md.get("module_tier"),
        "title": _module_title(demo),
        "target_id": demo.target_id,
        "n_sections": _n_sections(demo),
        "n_section_total": _n_section_total(demo),
        "n_questions": int(md.get("n_questions") or 0),
        "n_attempts": int(md.get("n_attempts") or 0),
        "n_accepted": _n_accepted(demo),
        "n_section_read_turns": n_section_read,
        "n_answer_turns": n_answer,
        "n_turns": demo.n_turns,
        "method_counts": dict(method_counts),
        "techniques_attempted": sorted(techs_attempted),
        "techniques_succeeded": sorted(techs_succeeded),
        "module_techniques": list(md.get("module_techniques") or []),
        "theory_bytes": _theory_bytes(demo),
        "total_reward": demo.total_reward,
        "outcome": {
            "foothold": demo.outcome.foothold,
            "user_flag": demo.outcome.user_flag,
            "root_flag": demo.outcome.root_flag,
        },
        "wizard": bool(md.get("wizard", False)),
    }


def _module_id_sort_key(s: str) -> tuple[int, str]:
    """Sort module ids numerically when possible, then lexicographically."""
    try:
        return (int(s), s)
    except (TypeError, ValueError):
        return (10**9, s)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    if n <= 1:
        return s[:n]
    return s[: n - 1] + "…"


def _format_table(rows: list[dict[str, Any]]) -> str:
    """Render the per-module rows as a fixed-width ASCII table."""
    headers = [
        "Module", "Title", "Sections", "Questions",
        "Attempts", "Accepted", "Theory KB", "Techniques",
    ]
    table_rows: list[list[str]] = []
    for r in rows:
        sec_str = f"{r['n_sections']}/{r['n_section_total']}"
        kb = r["theory_bytes"] / 1024.0
        techs = r["techniques_attempted"]
        tech_str = (
            ",".join(techs[:4]) + ("+" if len(techs) > 4 else "")
            if techs else "-"
        )
        table_rows.append([
            str(r["module_id"]),
            _truncate(r["title"] or "", _TITLE_TRUNC),
            sec_str,
            str(r["n_questions"]),
            str(r["n_attempts"]),
            str(r["n_accepted"]),
            f"{kb:.1f}",
            tech_str,
        ])

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [_fmt(headers), _fmt(["-" * w for w in widths])]
    for row in table_rows:
        lines.append(_fmt(row))
    return "\n".join(lines)


def _walk_demos(
    demos_dir: Path, *, glob: str = _DEMO_GLOB
) -> list[tuple[Path, Demonstration]]:
    """Load every academy demo under ``demos_dir``.

    Failures are logged to stderr and skipped (the rest of the report still
    lands - one corrupt file shouldn't blind the operator to the rest).
    """
    out: list[tuple[Path, Demonstration]] = []
    paths = sorted(demos_dir.glob(glob)) if demos_dir.is_dir() else []
    for p in paths:
        try:
            demo = load_demonstration(p)
        except Exception as e:
            print(
                f"warning: failed to load {p}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            continue
        out.append((p, demo))
    return out


def build_report(
    demos_dir: Path | str = _DEFAULT_DEMOS_DIR,
    *,
    min_questions: int = 0,
) -> dict[str, Any]:
    """Aggregate per-module + cross-module statistics into one dict."""
    demos_dir = Path(demos_dir)
    loaded = _walk_demos(demos_dir)
    rows: list[dict[str, Any]] = []
    for path, demo in loaded:
        summary = _summarize_demo(demo, path=path)
        if summary["n_questions"] < min_questions:
            continue
        rows.append(summary)
    rows.sort(key=lambda r: _module_id_sort_key(r["module_id"]))

    method_totals: Counter[str] = Counter()
    techs_attempted: set[str] = set()
    techs_succeeded: set[str] = set()
    module_techs: set[str] = set()
    for r in rows:
        for k, v in r["method_counts"].items():
            method_totals[k] += v
        techs_attempted.update(r["techniques_attempted"])
        techs_succeeded.update(r["techniques_succeeded"])
        module_techs.update(r["module_techniques"])

    return {
        "demos_dir": str(demos_dir),
        "n_demos": len(rows),
        "n_demos_skipped": len(loaded) - len(rows),
        "min_questions": min_questions,
        "modules": rows,
        "method_totals": dict(method_totals),
        "techniques_attempted": sorted(techs_attempted),
        "techniques_succeeded": sorted(techs_succeeded),
        "module_techniques": sorted(module_techs),
    }


def render_report(report: dict[str, Any]) -> str:
    """Pretty-print a coverage report (the table + cross-module summaries)."""
    lines: list[str] = []
    lines.append(f"=== HTB Academy demo coverage ({report['demos_dir']}) ===")
    if report["n_demos"] == 0:
        if report["min_questions"] > 0:
            lines.append(
                f"no demos found matching --min-questions={report['min_questions']} "
                f"(skipped {report['n_demos_skipped']})"
            )
        else:
            lines.append("no demos found.")
        return "\n".join(lines)

    lines.append(
        f"{report['n_demos']} demo(s)"
        + (
            f" (skipped {report['n_demos_skipped']} below "
            f"--min-questions={report['min_questions']})"
            if report["n_demos_skipped"]
            else ""
        )
    )
    lines.append("")
    lines.append(_format_table(report["modules"]))
    lines.append("")

    if report["method_totals"]:
        method_summary = ", ".join(
            f"{m}={c}" for m, c in sorted(
                report["method_totals"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        )
        lines.append(f"Method tags: {method_summary}")
    else:
        lines.append("Method tags: (none recorded)")

    techs = report["techniques_attempted"]
    lines.append(
        f"Distinct techniques_attempted across all demos: "
        + (", ".join(techs) if techs else "(none)")
        + f" ({len(techs)} total)"
    )

    succ = report["techniques_succeeded"]
    if succ:
        lines.append(
            f"Distinct techniques_succeeded across all demos: "
            f"{', '.join(succ)} ({len(succ)} total)"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--demos-dir",
        type=Path,
        default=_DEFAULT_DEMOS_DIR,
        help=f"Directory holding academy demos (default: {_DEFAULT_DEMOS_DIR}).",
    )
    p.add_argument(
        "--min-questions",
        type=int,
        default=0,
        help=(
            "Only include demos where metadata.n_questions >= N. "
            "Useful for focusing on modules with actual training value."
        ),
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the aggregated report as JSON instead of a table.",
    )
    args = p.parse_args(argv)

    report = build_report(args.demos_dir, min_questions=args.min_questions)

    if args.as_json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
