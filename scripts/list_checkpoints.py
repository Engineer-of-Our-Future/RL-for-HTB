"""List checkpoints in a runs/ directory with their metadata.

Usage:
    python scripts/list_checkpoints.py runs/
    python scripts/list_checkpoints.py runs/ --latest-per-run --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from htbrl.utils.checkpoint_registry import find_latest_per_run, scan_runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs_dir", type=Path)
    p.add_argument("--latest-per-run", action="store_true",
                   help="show only the most recent checkpoint per run dir")
    p.add_argument("--json", action="store_true",
                   help="emit JSON instead of a human-readable line per ckpt")
    args = p.parse_args(argv)

    metas = find_latest_per_run(args.runs_dir) if args.latest_per_run else scan_runs(args.runs_dir)
    if not metas:
        print(f"no checkpoints found under {args.runs_dir}")
        return 0

    if args.json:
        out = []
        for m in metas:
            d = {**asdict(m)}
            d["path"] = str(d["path"])
            d["run_dir"] = str(d["run_dir"])
            out.append(d)
        print(json.dumps(out, indent=2, default=str))
        return 0

    for m in metas:
        print(f"  {m.short_summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
