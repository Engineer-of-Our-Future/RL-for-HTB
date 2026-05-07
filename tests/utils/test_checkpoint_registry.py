"""Tests for the checkpoint registry."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from htbrl.utils.checkpoint_registry import (
    CheckpointMetadata,
    find_latest_per_run,
    scan_runs,
)


def _write_fake_ckpt(path: Path, *, env_steps: int, rollout_idx: int = 0,
                     config: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {"layer.weight": torch.randn(2, 2)},
        "config": config or {"d_model": 32, "n_layers": 2},
        "rollout_idx": rollout_idx,
        "env_steps": env_steps,
        "args": {"env_type": "stub"},
    }
    torch.save(payload, path)


def test_scan_runs_empty_dir(tmp_path: Path):
    assert scan_runs(tmp_path) == []


def test_scan_runs_finds_checkpoints(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00010.pt", env_steps=100)
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00020.pt", env_steps=200)
    _write_fake_ckpt(tmp_path / "run-b" / "ckpt-final.pt", env_steps=500)

    metas = scan_runs(tmp_path)
    assert len(metas) == 3
    paths = {m.path.name for m in metas}
    assert paths == {"ckpt-00010.pt", "ckpt-00020.pt", "ckpt-final.pt"}


def test_scan_runs_loads_last_log_entry(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00010.pt", env_steps=100)
    log = tmp_path / "run-a" / "log.jsonl"
    log.write_text(
        '{"rollout": 0, "avg_episode_return": 0.5}\n'
        '{"rollout": 1, "avg_episode_return": 0.7}\n',
        encoding="utf-8",
    )
    metas = scan_runs(tmp_path)
    m = metas[0]
    assert m.last_log_entry is not None
    assert m.last_log_entry["rollout"] == 1
    assert m.last_log_entry["avg_episode_return"] == 0.7


def test_scan_runs_loads_eval_summary(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00010.pt", env_steps=100)
    eval_path = tmp_path / "run-a" / "eval.json"
    eval_path.write_text(json.dumps({"foothold_rate": 0.42}), encoding="utf-8")
    metas = scan_runs(tmp_path)
    assert metas[0].eval_summary == {"foothold_rate": 0.42}


def test_find_latest_per_run_takes_highest_env_steps(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00010.pt", env_steps=100, rollout_idx=10)
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00020.pt", env_steps=200, rollout_idx=20)
    _write_fake_ckpt(tmp_path / "run-b" / "ckpt-final.pt", env_steps=500, rollout_idx=50)

    latest = find_latest_per_run(tmp_path)
    by_dir = {m.run_dir.name: m for m in latest}
    assert by_dir["run-a"].path.name == "ckpt-00020.pt"
    assert by_dir["run-b"].path.name == "ckpt-final.pt"


def test_short_summary_includes_key_fields(tmp_path: Path):
    _write_fake_ckpt(tmp_path / "run-a" / "ckpt-00010.pt", env_steps=100)
    metas = scan_runs(tmp_path)
    s = metas[0].short_summary()
    assert "ckpt-00010.pt" in s
    assert "steps=100" in s
    assert "MB" in s


def test_skips_unloadable_files(tmp_path: Path):
    """Files that match the glob but aren't valid torch checkpoints get skipped."""
    bad = tmp_path / "run-a" / "ckpt-bad.pt"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a torch file")
    metas = scan_runs(tmp_path)
    assert metas == []  # nothing loaded
