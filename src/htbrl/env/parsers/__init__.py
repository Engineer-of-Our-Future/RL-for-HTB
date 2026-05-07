"""Per-tool output parsers (PLAN.md Phase 4).

Each tool in the registry declares an ``output_parser_id`` (raw / lines / nmap
/ ...). The env wrapper looks up that ID here and runs the parser on the
tool's stdout to produce a structured features dict that goes into the next
observation alongside the raw text.

Parsers are pure functions: ``(stdout: str) -> dict[str, Any]``. They never
raise on malformed input - they return whatever they can recover plus a
``parse_errors`` list so the policy can learn that a particular invocation
failed even when the tool's exit code looked clean.
"""

from __future__ import annotations

from typing import Callable

from .nmap import parse_nmap
from .simple import parse_lines, parse_raw
from .smb import parse_smbclient

ParserFn = Callable[[str], dict]


# Registry of all known parsers. Tool YAMLs reference these IDs in
# ``output_parser_id``. Adding a parser = drop a module here + register it.
PARSERS: dict[str, ParserFn] = {
    "raw": parse_raw,
    "lines": parse_lines,
    "nmap": parse_nmap,
    "smbclient": parse_smbclient,
}


def parse(parser_id: str, stdout: str) -> dict:
    """Dispatch to a parser by ID. Falls back to ``raw`` if the ID is unknown."""
    fn = PARSERS.get(parser_id, parse_raw)
    try:
        return fn(stdout)
    except Exception as exc:  # parsers shouldn't raise, but be defensive
        return {"text": stdout, "parse_errors": [f"{type(exc).__name__}: {exc}"]}
