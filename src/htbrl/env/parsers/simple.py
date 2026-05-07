"""Trivial parsers: raw and line-split."""

from __future__ import annotations


def parse_raw(stdout: str) -> dict:
    """Pass-through: no structured features extracted."""
    return {"text": stdout, "parse_errors": []}


def parse_lines(stdout: str) -> dict:
    """Split stdout into non-empty trimmed lines."""
    lines = [ln.rstrip() for ln in stdout.splitlines() if ln.strip()]
    return {"text": stdout, "lines": lines, "parse_errors": []}
