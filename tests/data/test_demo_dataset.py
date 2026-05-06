"""Tests for demo storage + DemoDataset filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from htbrl.data.demo_dataset import (
    Demonstration,
    DemoDataset,
    DemoOutcome,
    DemoTurn,
    load_demonstration,
    save_demonstration,
)


def _make_demo(matrix: str = "enterprise", target: str = "test", outcome_kw: dict | None = None) -> Demonstration:
    return Demonstration(
        matrix=matrix,
        target_id=target,
        turns=[
            DemoTurn(
                obs_text="22/tcp open ssh",
                action_tool_id=0,
                action_tool_name="nmap_quick_tcp",
                action_slots={"ip": "10.10.10.5"},
                action_render="nmap -sS -T4 --top-ports 1000 10.10.10.5",
                reward=0.1,
                techniques_attempted=["T1046"],
                techniques_succeeded=["T1046"],
            ),
            DemoTurn(
                obs_text="found /admin",
                action_tool_id=2,
                action_tool_name="gobuster_dir",
                action_slots={"ip": "10.10.10.5", "wordlist": "common.txt"},
                action_render="gobuster dir ...",
                reward=0.05,
            ),
        ],
        outcome=DemoOutcome(**(outcome_kw or {"foothold": True})),
        metadata={"author": "tester", "ts": 0},
    )


def test_demo_round_trip(tmp_path: Path):
    demo = _make_demo()
    p = tmp_path / "d1.msgpack.gz"
    save_demonstration(demo, p)
    loaded = load_demonstration(p)
    assert loaded.matrix == demo.matrix
    assert loaded.target_id == demo.target_id
    assert loaded.n_turns == 2
    assert loaded.outcome.foothold is True
    assert loaded.turns[0].techniques_attempted == ["T1046"]


def test_demo_uncompressed(tmp_path: Path):
    demo = _make_demo()
    p = tmp_path / "d1.msgpack"
    save_demonstration(demo, p, compress=False)
    loaded = load_demonstration(p)
    assert loaded.n_turns == demo.n_turns


def test_demo_rejects_unknown_matrix():
    with pytest.raises(ValueError, match="not in"):
        Demonstration(matrix="windows", target_id="x", turns=[], outcome=DemoOutcome())


def test_demo_total_reward():
    demo = _make_demo()
    assert abs(demo.total_reward - (0.1 + 0.05)) < 1e-6


def test_demo_load_unknown_version_raises(tmp_path: Path):
    import msgpack
    p = tmp_path / "bad.msgpack"
    p.write_bytes(msgpack.packb({"version": 99, "matrix": "enterprise"}))
    with pytest.raises(ValueError, match="unsupported demo file version"):
        load_demonstration(p)


def test_dataset_loads_and_iterates(tmp_path: Path):
    save_demonstration(_make_demo(target="a"), tmp_path / "a.msgpack.gz")
    save_demonstration(_make_demo(target="b"), tmp_path / "b.msgpack.gz")
    save_demonstration(_make_demo(target="c"), tmp_path / "c.msgpack.gz")
    ds = DemoDataset(tmp_path)
    assert len(ds) == 3
    targets = sorted(d.target_id for d in ds)
    assert targets == ["a", "b", "c"]


def test_dataset_indexing(tmp_path: Path):
    save_demonstration(_make_demo(target="a"), tmp_path / "a.msgpack.gz")
    ds = DemoDataset(tmp_path)
    demo = ds[0]
    assert demo.target_id == "a"


def test_dataset_filter_by_matrix(tmp_path: Path):
    save_demonstration(_make_demo(matrix="enterprise", target="ent"), tmp_path / "ent.msgpack.gz")
    save_demonstration(_make_demo(matrix="ics", target="ics1"), tmp_path / "ics1.msgpack.gz")
    save_demonstration(_make_demo(matrix="ics", target="ics2"), tmp_path / "ics2.msgpack.gz")
    ds = DemoDataset(tmp_path)
    ics_only = ds.filter(matrix="ics")
    assert len(ics_only) == 2
    assert all(d.matrix == "ics" for d in ics_only)


def test_dataset_filter_by_outcome(tmp_path: Path):
    save_demonstration(_make_demo(outcome_kw={"foothold": True}), tmp_path / "ok.msgpack.gz")
    save_demonstration(_make_demo(outcome_kw={"foothold": False}), tmp_path / "fail.msgpack.gz")
    ds = DemoDataset(tmp_path)
    ok = ds.filter(outcome="foothold")
    assert len(ok) == 1


def test_dataset_filter_min_turns(tmp_path: Path):
    save_demonstration(_make_demo(target="a"), tmp_path / "a.msgpack.gz")  # 2 turns
    ds = DemoDataset(tmp_path)
    assert len(ds.filter(min_turns=2)) == 1
    assert len(ds.filter(min_turns=5)) == 0


def test_dataset_filter_unknown_outcome_raises(tmp_path: Path):
    save_demonstration(_make_demo(), tmp_path / "a.msgpack.gz")
    ds = DemoDataset(tmp_path)
    with pytest.raises(ValueError, match="unknown outcome"):
        ds.filter(outcome="banana")


def test_dataset_stats_aggregation(tmp_path: Path):
    save_demonstration(_make_demo(matrix="enterprise"), tmp_path / "e1.msgpack.gz")
    save_demonstration(_make_demo(matrix="enterprise", outcome_kw={"foothold": True, "user_flag": True}), tmp_path / "e2.msgpack.gz")
    save_demonstration(_make_demo(matrix="ics"), tmp_path / "i1.msgpack.gz")
    s = DemoDataset(tmp_path).stats()
    assert s["n_demos"] == 3
    assert s["n_by_matrix"]["enterprise"] == 2
    assert s["n_by_matrix"]["ics"] == 1
    assert s["total_turns"] == 6  # 2 turns × 3 demos
    assert s["n_user_flag"] == 1


def test_dataset_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DemoDataset(tmp_path / "does-not-exist")
