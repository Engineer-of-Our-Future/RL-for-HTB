"""Train the byte-level BPE tokenizer on a shell-relevant corpus (Phase 2).

The default corpus pulls together:
  - Linux man pages from the connected Kali (via SSH)
  - The registry's per-tool ``example_output`` strings
  - HTB-style writeup-y text the user can drop into ``data/corpus/``

You can also point it at any list of plaintext files via ``--corpus-file``.

Outputs a single JSON with the tokenizer state to ``--output``. Resulting
file is what ``ByteLevelBPE.load(...)`` consumes.

Examples:

    # Train using just the registry's example_output strings (smallest corpus)
    python scripts/train_tokenizer.py --vocab-size 8192 --output tokenizer/v0.json

    # Add Linux man pages from a configured Kali
    python scripts/train_tokenizer.py \\
        --kali-host htbrl@127.0.0.1:2222 --kali-key ~/.ssh/htbrl_kali \\
        --man-pages 200 --vocab-size 32768 --output tokenizer/v1.json

    # Add user-provided text files
    python scripts/train_tokenizer.py \\
        --corpus-dir data/corpus \\
        --vocab-size 32768 --output tokenizer/v1.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterator

from htbrl.env.ssh_session import SSHCredentials, SSHSession
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


def _corpus_from_registry() -> Iterator[str]:
    vocab = load_registry()
    for tool in vocab.tools:
        if tool.example_output:
            yield tool.example_output
        if tool.command_template:
            yield tool.command_template
        if tool.description:
            yield tool.description


def _corpus_from_dir(root: Path) -> Iterator[str]:
    """Yield every text file under ``root`` (small / non-binary only)."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # 5 MB hard cap per file to avoid choking on stray binaries.
        if p.stat().st_size > 5_000_000:
            continue
        try:
            yield p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue


def _corpus_from_kali_man_pages(
    ssh: SSHSession,
    n_pages: int = 200,
    cmd_timeout_s: float = 30.0,
) -> Iterator[str]:
    """Pull the first ``n_pages`` distinct man pages from the connected Kali.

    Uses ``apropos .`` to enumerate available man pages, takes the first
    ``n_pages``, runs ``man <name>`` on each. ``apropos`` is part of
    ``manpages`` / ``man-db``; if it isn't installed the script raises a
    clear error.
    """
    list_cmd = f"apropos . 2>/dev/null | awk '{{print $1}}' | sort -u | head -n {n_pages}"
    listing = ssh.run(list_cmd, timeout=cmd_timeout_s)
    if listing.timed_out or not listing.stdout.strip():
        raise RuntimeError(
            "could not list man pages on Kali (is `man-db` installed?)"
        )
    names = [n.strip() for n in listing.stdout.splitlines() if n.strip()]
    print(f"[tok] pulling {len(names)} man pages from Kali...")
    for i, name in enumerate(names, 1):
        # COLUMNS=120 so the man output isn't reflowed too narrow.
        cmd = f"COLUMNS=120 MANWIDTH=120 man {name} 2>/dev/null | col -b"
        r = ssh.run(cmd, timeout=cmd_timeout_s)
        if r.timed_out or not r.stdout.strip():
            continue
        yield r.stdout
        if i % 25 == 0:
            print(f"[tok]   {i}/{len(names)} fetched")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="path to write the trained tokenizer JSON to")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--corpus-dir", type=Path, default=None,
                   help="directory of plain-text files to include in the corpus")
    p.add_argument("--corpus-file", action="append", default=[],
                   help="repeat to add individual text files to the corpus")
    p.add_argument("--no-registry", action="store_true",
                   help="skip the registry-derived corpus chunks")
    # Kali-driven man-page collection (optional)
    p.add_argument("--kali-host", default=os.environ.get("HTBRL_KALI_HOST"))
    p.add_argument("--kali-key", default=os.environ.get("HTBRL_KALI_KEY"))
    p.add_argument("--man-pages", type=int, default=0,
                   help="if > 0 and Kali is configured, pull this many man pages")
    p.add_argument("--man-cmd-timeout-s", type=float, default=30.0)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    print("=== BPE tokenizer training ===")
    print(f"  output       : {args.output}")
    print(f"  vocab_size   : {args.vocab_size}")
    print(f"  min_frequency: {args.min_frequency}")

    corpus: list[str] = []

    if not args.no_registry:
        chunk = list(_corpus_from_registry())
        print(f"  + registry strings: {len(chunk)} chunks")
        corpus.extend(chunk)

    if args.corpus_dir:
        chunk = list(_corpus_from_dir(args.corpus_dir))
        print(f"  + corpus_dir {args.corpus_dir}: {len(chunk)} chunks")
        corpus.extend(chunk)

    for cf in args.corpus_file:
        path = Path(cf)
        if not path.is_file():
            print(f"  ! skipping missing --corpus-file {path}", file=sys.stderr)
            continue
        corpus.append(path.read_text(encoding="utf-8", errors="replace"))
        print(f"  + corpus_file {cf}")

    ssh = None
    if args.man_pages > 0:
        if not args.kali_host:
            print("ERROR: --man-pages requires --kali-host (or $HTBRL_KALI_HOST)",
                  file=sys.stderr)
            return 2
        user_at_host, _, port_str = args.kali_host.partition(":")
        user, _, host = user_at_host.partition("@")
        creds = SSHCredentials(
            host=host,
            port=int(port_str) if port_str else 22,
            user=user,
            identity_file=os.path.expanduser(args.kali_key) if args.kali_key else None,
        )
        ssh = SSHSession(creds)
        ssh.open()
        chunk = list(_corpus_from_kali_man_pages(
            ssh, n_pages=args.man_pages, cmd_timeout_s=args.man_cmd_timeout_s,
        ))
        print(f"  + man pages: {len(chunk)} pages")
        corpus.extend(chunk)
        ssh.close()

    if not corpus:
        print("ERROR: empty corpus (use --corpus-dir / --corpus-file / --man-pages)",
              file=sys.stderr)
        return 2

    total_bytes = sum(len(s.encode("utf-8", errors="replace")) for s in corpus)
    print(f"  corpus       : {len(corpus)} chunks, {total_bytes/1e6:.1f} MB")

    tok = ByteLevelBPE.initialize()
    t0 = time.time()
    n_merges = tok.train(
        corpus,
        target_vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print(f"  trained {n_merges} merges in {elapsed:.1f}s "
          f"(final vocab_size = {tok.vocab_size})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.output)
    print(f"  saved -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
