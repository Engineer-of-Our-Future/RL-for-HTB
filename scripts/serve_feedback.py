"""Launch the FastAPI feedback UI (Phase 8).

Examples:

    # serve labeling UI on localhost:8765
    python scripts/serve_feedback.py --db data/preferences.db --port 8765

    # add a new snippet from a text file (handy bootstrap before any rollouts)
    python scripts/serve_feedback.py --db data/preferences.db \\
        --add-snippet --matrix enterprise --source "manual" \\
        --content-file path/to/snippet.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from htbrl.data.preference_dataset import PreferenceStore
from htbrl.feedback.server import create_app


def _add_snippet(args: argparse.Namespace) -> int:
    if not args.content_file:
        print("ERROR: --content-file required for --add-snippet", file=sys.stderr)
        return 2
    text = Path(args.content_file).read_text(encoding="utf-8", errors="replace")
    store = PreferenceStore(args.db)
    sid = store.add_snippet(
        matrix=args.matrix,
        content_text=text,
        source=args.source,
    )
    store.close()
    print(f"added snippet id={sid} matrix={args.matrix} source={args.source!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="data/preferences.db", help="sqlite preference store")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    p.add_argument("--add-snippet", action="store_true",
                   help="instead of serving, insert a snippet from --content-file")
    p.add_argument("--matrix", default="enterprise",
                   choices=["enterprise", "mobile", "ics"])
    p.add_argument("--source", default="manual")
    p.add_argument("--content-file", default=None)
    args = p.parse_args(argv)

    if args.add_snippet:
        return _add_snippet(args)

    app = create_app(args.db)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
