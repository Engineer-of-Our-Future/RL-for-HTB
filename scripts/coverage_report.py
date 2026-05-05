"""Print the registry's MITRE ATT&CK coverage as a per-matrix table.

Run:
    python scripts/coverage_report.py

Exit code is non-zero if any matrix has a tactic with fewer tools than the
configured `--min-per-tactic` threshold. Used by CI to gate merges.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from htbrl.tools.loader import load_registry


# Pretty-print labels for the Enterprise tactic IDs we expect to encounter most
# often. Unknown IDs are still printed by ID (no breakage on new MITRE tactics).
_TACTIC_LABELS = {
    "TA0001": "Initial Access",
    "TA0002": "Execution",
    "TA0003": "Persistence",
    "TA0004": "Privilege Escalation",
    "TA0005": "Defense Evasion",
    "TA0006": "Credential Access",
    "TA0007": "Discovery",
    "TA0008": "Lateral Movement",
    "TA0009": "Collection",
    "TA0010": "Exfiltration",
    "TA0011": "Command and Control",
    "TA0040": "Impact",
    "TA0042": "Resource Development",
    "TA0043": "Reconnaissance",
}


def _format_table(rows: Iterable[tuple[str, str, int]]) -> str:
    """Render `(tactic_id, label, count)` rows as a fixed-width table."""
    rows = list(rows)
    if not rows:
        return "  (no tactics covered)"
    id_w = max(len(r[0]) for r in rows)
    lbl_w = max(len(r[1]) for r in rows)
    out = []
    for tid, lbl, n in rows:
        bar = "#" * min(n, 30)
        out.append(f"  {tid:<{id_w}}  {lbl:<{lbl_w}}  {n:>3}  {bar}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-per-tactic",
        type=int,
        default=0,
        help="If > 0, exit non-zero when any tactic has fewer tools than this. "
        "Set per-matrix via repeated --min for-matrix flags (future).",
    )
    p.add_argument(
        "--matrix",
        choices=["enterprise", "mobile", "ics", "all"],
        default="all",
        help="Restrict the report to one matrix.",
    )
    args = p.parse_args(argv)

    vocab = load_registry()
    summary = vocab.coverage_summary()

    print(f"=== ATT&CK coverage of {vocab.n_tools} registered tools ===\n")

    failed_thresholds: list[str] = []

    matrices = ["enterprise", "mobile", "ics"] if args.matrix == "all" else [args.matrix]
    for m in matrices:
        data = summary[m]
        n_tools = data["tools"]
        print(f"[{m.upper()}] {n_tools} tools, "
              f"{len(data['tactics'])} tactics, "
              f"{len(data['techniques'])} techniques")

        rows = []
        for tid, count in sorted(data["tactics"].items()):
            label = _TACTIC_LABELS.get(tid, "(unknown tactic)")
            rows.append((tid, label, count))
        print(_format_table(rows))
        print()

        if args.min_per_tactic > 0 and n_tools > 0:
            for tid, count in data["tactics"].items():
                if count < args.min_per_tactic:
                    failed_thresholds.append(
                        f"{m}/{tid}: {count} tool(s), threshold = {args.min_per_tactic}"
                    )

    if failed_thresholds:
        print("FAIL: tactics below threshold:", file=sys.stderr)
        for f in failed_thresholds:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
