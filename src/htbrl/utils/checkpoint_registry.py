"""Checkpoint registry (PLAN.md Phase 12 polish).

Scans a runs directory and produces structured metadata about every saved
checkpoint: which run it came from, what config produced it, how many env
steps had elapsed, what eval metrics were recorded (if any), what git SHA
was current at the time. Used by the upcoming experiment-launcher and by
ad-hoc CLI inspection.

The trainer's checkpoint format is just a torch.save(...) of a dict with
keys ``model``, ``config``, ``rollout_idx``, ``env_steps``, ``args``. We
read that lazily (don't load the model weights) and join it with sibling
``log.jsonl`` if present + an optional ``eval.json`` from scripts/eval.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointMetadata:
    path: Path
    run_dir: Path
    run_name: str
    rollout_idx: int = 0
    env_steps: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] = field(default_factory=dict)
    last_log_entry: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None
    git_sha: str | None = None
    size_mb: float = 0.0

    def short_summary(self) -> str:
        bits = [
            self.path.name,
            f"steps={self.env_steps}",
            f"rollout={self.rollout_idx}",
            f"size={self.size_mb:.1f}MB",
        ]
        if self.last_log_entry:
            r = self.last_log_entry.get("avg_episode_return")
            if r is not None:
                bits.append(f"avg_ep_ret={r:+.3f}")
        if self.eval_summary:
            f_rate = self.eval_summary.get("foothold_rate")
            if f_rate is not None:
                bits.append(f"foothold_rate={f_rate:.2f}")
        if self.git_sha:
            bits.append(f"git={self.git_sha[:8]}")
        return " ".join(bits)


def _git_sha_of_repo(start: Path) -> str | None:
    """Walk up from ``start`` looking for .git, then read HEAD's commit SHA."""
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            try:
                out = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                return out.decode().strip() or None
            except Exception:
                return None
    return None


def _load_checkpoint_meta(path: Path) -> CheckpointMetadata | None:
    """Open the .pt file with weights_only=False so we can read the dict."""
    try:
        # Defensive: torch>=2.6 deprecated weights_only=False as a default;
        # pin it explicitly to keep behavior predictable.
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(ck, dict):
        return None
    return CheckpointMetadata(
        path=path,
        run_dir=path.parent,
        run_name=path.parent.name,
        rollout_idx=int(ck.get("rollout_idx", 0)),
        env_steps=int(ck.get("env_steps", 0)),
        config=dict(ck.get("config", {})),
        args=dict(ck.get("args", {})),
        size_mb=path.stat().st_size / (1024 * 1024),
    )


def _last_log_entry(run_dir: Path) -> dict[str, Any] | None:
    """Return the last JSON record in ``run_dir/log.jsonl`` if present."""
    log_path = run_dir / "log.jsonl"
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return None
    last: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


def _eval_summary(run_dir: Path) -> dict[str, Any] | None:
    """Optionally read scripts/eval.py's summary JSON if it sits next to the ckpt."""
    candidate = run_dir / "eval.json"
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None


def scan_runs(runs_dir: Path | str, *, recursive: bool = True) -> list[CheckpointMetadata]:
    """Walk ``runs_dir`` for ``ckpt-*.pt`` files and return rich metadata for each.

    If ``recursive=False`` only the immediate children are searched.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    pattern = "**/ckpt-*.pt" if recursive else "*/ckpt-*.pt"
    out: list[CheckpointMetadata] = []
    git_sha = _git_sha_of_repo(runs_dir)
    for path in sorted(runs_dir.glob(pattern)):
        meta = _load_checkpoint_meta(path)
        if meta is None:
            continue
        meta.git_sha = git_sha
        meta.last_log_entry = _last_log_entry(meta.run_dir)
        meta.eval_summary = _eval_summary(meta.run_dir)
        out.append(meta)
    return out


def find_latest_per_run(runs_dir: Path | str) -> list[CheckpointMetadata]:
    """Return the latest checkpoint (highest env_steps, then mtime) per run dir."""
    by_run: dict[Path, CheckpointMetadata] = {}
    for meta in scan_runs(runs_dir):
        prev = by_run.get(meta.run_dir)
        if prev is None or meta.env_steps > prev.env_steps:
            by_run[meta.run_dir] = meta
        elif meta.env_steps == prev.env_steps and meta.path.stat().st_mtime > prev.path.stat().st_mtime:
            by_run[meta.run_dir] = meta
    return sorted(by_run.values(), key=lambda m: m.run_dir.name)
