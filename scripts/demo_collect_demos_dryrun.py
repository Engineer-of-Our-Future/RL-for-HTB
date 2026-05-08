"""Dry-run of scripts/collect_demos.py with a canned conversation.

Lets the operator preview the wizard's prompts + replies without
having an HTB box spawned. Monkeypatches the wizard's ``_ask`` so
each prompt gets a scripted answer; everything else (tool
suggestion, slot prompting, reward parsing, demo writing) is real.

Usage:

    python scripts/demo_collect_demos_dryrun.py
    python scripts/demo_collect_demos_dryrun.py --keep    # keep written demo

The fake target_id is ``demo:dryrun-meow``; the demo lands in
``data/demos/wizard_demo-dryrun-meow_<ts>.msgpack.gz`` (deleted
after the run unless ``--keep`` is set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))   # so ``import collect_demos`` works

from collections import deque

import collect_demos  # type: ignore[import-not-found]


# A canned conversation that drives the wizard through one full
# offline episode. Each entry is just the reply string; the actual
# wizard prompts are printed at runtime so the operator can see
# what triggered each reply.
_SCRIPTED = deque([
    # --- turn 1: nmap recon ---
    # ``nmap -sS`` matches both nmap_full_tcp + nmap_quick_tcp at
    # similarity 1.0; the wizard lists 3 suggestions and we pick #1
    # (nmap_full_tcp). nmap_full_tcp has ONE slot: ``ip``.
    "nmap -sS 10.129.42.42",
    # paste-output block; END with "." on its own line.
    "Starting Nmap 7.95 ( https://nmap.org )",
    "Nmap scan report for 10.129.42.42",
    "PORT   STATE SERVICE",
    "22/tcp open  ssh",
    "80/tcp open  http",
    ".",
    "1",                                          # pick suggestion #1
    "10.129.42.42",                               # ip slot
    "new_port",                                   # reward shortcut

    # --- finish episode ---
    "done",                                       # bash prompt: finish
    "n",                                          # foothold? (n for dry-run)
    "n",                                          # user_flag?
    "n",                                          # root_flag?
    "dry-run preview",                            # note
])


def _scripted_ask(prompt: str) -> str:
    """Replacement for ``collect_demos._ask`` that pulls from the script.

    When the script runs out we raise EOFError so the wizard's input
    handler returns "" and the episode wraps up cleanly instead of
    asking the same exhausted question forever.
    """
    if not _SCRIPTED:
        raise EOFError("dry-run script exhausted; episode ends here")
    value = _SCRIPTED.popleft()
    # Pretty-print so the operator sees prompt + scripted reply.
    print(f"\n>>> PROMPT: {prompt.rstrip()!r}")
    print(f"    REPLY:  {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", action="store_true",
                   help="don't delete the demo file written by the dry-run")
    args = p.parse_args(argv)

    # Patch the wizard's input fn for this run.
    collect_demos._ask = _scripted_ask

    out_dir = Path("data/demos")
    before = set(out_dir.glob("*.msgpack.gz")) if out_dir.exists() else set()

    # Disable auto-pick (--auto-suggest-threshold > 1) so the wizard
    # ALWAYS asks the operator to pick a suggestion. Otherwise the
    # demo skips that prompt and the operator can't see what the
    # picker UI looks like.
    fake_argv = [
        "--target-id", "demo:dryrun-meow",
        "--offline",
        "--auto-suggest-threshold", "1.01",
    ]
    rc = collect_demos.main(fake_argv)
    print(f"\n--- collect_demos returned {rc} ---")

    after = set(out_dir.glob("*.msgpack.gz")) if out_dir.exists() else set()
    new_files = after - before
    if new_files and not args.keep:
        for f in new_files:
            f.unlink()
            print(f"[dry-run] cleaned up {f}")
    elif new_files:
        for f in new_files:
            print(f"[dry-run] kept {f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
