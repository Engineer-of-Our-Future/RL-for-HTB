"""List HTB Academy modules visible to the logged-in user.

Attaches to the operator's already-running Chrome (via the same CDP plumbing
the walkers use), fetches ``/api/v2/modules`` from inside the authenticated
page, and prints a compact table of every module with its tier, cube cost,
section count, state (``owned`` / ``in_progress`` / ``locked``), and title.

Useful for:
  - "What's next on my curriculum?" - filter to ``--state owned`` and sort by
    tier+cubes to see what to walk next.
  - "Where could I unlock a cheap module?" - filter to ``--state locked
    --max-cubes 10`` to find the lightest-touch unlocks.
  - "Have I walked everything I own?" - compare with ``data/auto_demos/``.

Hard requirements:

  1. Operator has run ``scripts/start_chrome_for_htb.ps1`` and is logged into
     HTB Academy in that Chrome window.
  2. The Chrome debug endpoint is reachable at ``$HTBRL_ACADEMY_CDP``
     (default ``http://127.0.0.1:9222``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from htbrl.academy.cdp_walker import open_cdp


# JS that fetches the modules list from inside the authenticated page and
# returns a slim projection (only the fields we render in the table). The
# full response is ~1 MB so a server-side projection keeps the round-trip
# small and the WebSocket happy.
_LIST_JS = r"""
(async () => {
    try {
        const r = await fetch('/api/v2/modules', {credentials: 'include'});
        if (!r.ok) return {__error: 'http ' + r.status};
        const j = await r.json();
        const data = j.data || [];
        return data.map(m => ({
            id: m.id,
            name: m.name || '',
            tier: typeof m.tier === 'object' && m.tier !== null
                ? (m.tier.id || m.tier.name || '?') : (m.tier ?? -1),
            cubes_to_unlock: m.cubes_to_unlock || 0,
            sections_count: m.sections_count || 0,
            state: m.state || '',
            status: m.status || '',
            difficulty: m.difficulty || '',
            estimated_minutes: m.estimated_time_of_completion_in_minutes || 0,
        }));
    } catch (e) { return {__error: String(e)}; }
})()
"""


def _fmt_table(rows: list[dict], known_demo_ids: set[int]) -> str:
    """Render rows as an aligned ASCII table, sorted by (tier, cubes, id)."""
    if not rows:
        return "(no modules)"

    def _sort_key(m: dict) -> tuple:
        tier = m.get("tier")
        if isinstance(tier, str):
            try:
                tier = int(tier)
            except ValueError:
                tier = 99
        return (tier or 0, m.get("cubes_to_unlock", 0), m.get("id", 0))

    rows = sorted(rows, key=_sort_key)
    headers = ["ID", "Tier", "Cubes", "Sec", "Min", "State", "Walked", "Title"]
    table_rows: list[list[str]] = []
    for m in rows:
        title = (m.get("name") or "").strip()
        if len(title) > 40:
            title = title[:39] + "…"
        walked = "yes" if m["id"] in known_demo_ids else ""
        table_rows.append([
            str(m["id"]),
            str(m.get("tier", "")),
            str(m.get("cubes_to_unlock", 0)),
            str(m.get("sections_count", 0)),
            str(m.get("estimated_minutes", 0)),
            (m.get("state") or "")[:12],
            walked,
            title,
        ])
    widths = [
        max(len(h), max((len(r[i]) for r in table_rows), default=0))
        for i, h in enumerate(headers)
    ]
    out: list[str] = []
    out.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    out.append("  ".join("-" * w for w in widths))
    for r in table_rows:
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


def _summary(rows: list[dict]) -> str:
    n = len(rows)
    by_state: dict[str, int] = {}
    for m in rows:
        by_state[m.get("state") or "?"] = by_state.get(m.get("state") or "?", 0) + 1
    parts = [f"total={n}"]
    for k in sorted(by_state):
        parts.append(f"{k}={by_state[k]}")
    return "  ".join(parts)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdp",
                   default=os.environ.get("HTBRL_ACADEMY_CDP", "http://127.0.0.1:9222"))
    p.add_argument("--auto-demo-dir", type=Path, default=Path("data/auto_demos"),
                   help="directory whose existing demos mark modules as 'walked'")
    p.add_argument("--state", default="owned,in_progress",
                   help="comma-separated list of states to include "
                        "(owned, in_progress, locked); default 'owned,in_progress'. "
                        "Use 'all' for every state.")
    p.add_argument("--max-cubes", type=int, default=None,
                   help="only show modules with cubes_to_unlock <= N "
                        "(useful with --state locked to find cheap unlocks)")
    p.add_argument("--unwalked-only", action="store_true",
                   help="only show modules that don't yet have a demo on disk")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return p


def _existing_demo_ids(auto_demo_dir: Path) -> set[int]:
    """Return the set of module ids that already have a demo on disk."""
    out: set[int] = set()
    if not auto_demo_dir.is_dir():
        return out
    for p in auto_demo_dir.glob("academy_module_*.msgpack.gz"):
        # filename: academy_module_<id>.msgpack.gz   or
        #           academy_module_<id>_wizard.msgpack.gz
        stem = p.name.removeprefix("academy_module_").split(".", 1)[0]
        stem = stem.split("_", 1)[0]
        try:
            out.add(int(stem))
        except ValueError:
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        cdp, ws, _ = open_cdp(args.cdp)
    except Exception as exc:
        print(f"[list] failed to attach to {args.cdp}: {exc}", file=sys.stderr)
        return 2

    try:
        rows = cdp.evaluate(_LIST_JS, await_promise=True) or []
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if isinstance(rows, dict) and rows.get("__error"):
        print(f"[list] API error: {rows['__error']}", file=sys.stderr)
        return 1
    if not isinstance(rows, list):
        print(f"[list] unexpected response shape: {type(rows).__name__}", file=sys.stderr)
        return 1

    # Filter by state.
    states = {s.strip() for s in (args.state or "").split(",") if s.strip()}
    if states and "all" not in states:
        rows = [m for m in rows if m.get("state") in states]
    if args.max_cubes is not None:
        rows = [m for m in rows if (m.get("cubes_to_unlock", 0) or 0) <= args.max_cubes]
    walked_ids = _existing_demo_ids(args.auto_demo_dir)
    if args.unwalked_only:
        rows = [m for m in rows if m["id"] not in walked_ids]

    if args.json:
        for m in rows:
            m["walked"] = m["id"] in walked_ids
        print(json.dumps({"modules": rows, "summary": _summary(rows)}, indent=2))
        return 0

    print(f"=== HTB Academy modules (filter state={args.state!r}) ===")
    print(_summary(rows))
    print()
    print(_fmt_table(rows, walked_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
