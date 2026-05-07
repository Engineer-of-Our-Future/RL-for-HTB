"""Tests for ``scripts/htb_academy_walk_all.py`` (multi-module driver).

These tests don't need a live CDP attach: they monkey-patch the open_cdp /
fetch_modules_list / walk_one_module entry points so the harness exercises
the orchestration layer (planning, skip-existing, dry-run, gate-stop)
without ever opening Chrome.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# Load the script as a module without needing it to be on sys.path. Tests
# get to call ``walk_all.main([...])`` and ``walk_all.walk_one_module``
# while sharing the same import surface that the script uses.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "htb_academy_walk_all.py"


@pytest.fixture(scope="module")
def walk_all():
    spec = importlib.util.spec_from_file_location("htb_academy_walk_all", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["htb_academy_walk_all"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- ModuleListEntry parsing ------------------------------------------------


def test_module_list_entry_from_api_explicit_state(walk_all):
    raw = {"id": 9, "name": "Linux Fundamentals", "tier": 1,
           "cubes_to_unlock": 10, "state": "owned"}
    e = walk_all.ModuleListEntry.from_api(raw)
    assert e is not None
    assert e.id == 9
    assert e.title == "Linux Fundamentals"
    assert e.tier == 1
    assert e.cubes_to_unlock == 10
    assert e.state == "owned"


def test_module_list_entry_infers_state_from_pct(walk_all):
    # No explicit state -> inferred from percentage.
    e = walk_all.ModuleListEntry.from_api({
        "id": 18, "name": "X", "progress_percentage": 60.0,
    })
    assert e.state == "in_progress"

    e_done = walk_all.ModuleListEntry.from_api({
        "id": 19, "name": "Y", "progress_percentage": 100,
    })
    assert e_done.state == "completed"

    e_owned = walk_all.ModuleListEntry.from_api({
        "id": 20, "name": "Z", "owned": True, "progress_percentage": 0,
    })
    assert e_owned.state == "owned"


def test_filter_and_sort_modules_orders_by_tier_then_cubes(walk_all):
    entries = [
        walk_all.ModuleListEntry(id=3, title="C", tier=1, cubes_to_unlock=20, state="owned"),
        walk_all.ModuleListEntry(id=1, title="A", tier=0, cubes_to_unlock=0,  state="owned"),
        walk_all.ModuleListEntry(id=2, title="B", tier=0, cubes_to_unlock=10, state="in_progress"),
        walk_all.ModuleListEntry(id=4, title="D", tier=2, cubes_to_unlock=0,  state="locked"),
    ]
    plan = walk_all.filter_and_sort_modules(entries, {"owned", "in_progress"})
    assert [e.id for e in plan] == [1, 2, 3]


# ---- main() with mocked CDP -------------------------------------------------


@pytest.fixture
def fake_cdp(monkeypatch, walk_all):
    """Patch open_cdp + fetch_modules_list + walk_one_module + read_cube_balance.

    Returns a SimpleNamespace recording calls so each test can assert on
    what main() invoked.
    """
    rec = SimpleNamespace(
        opened=False, closed=False, walked=[],  # list of module_ids
        cube_reads=0,
    )

    class _FakeWS:
        def close(self):
            rec.closed = True

    fake_ws = _FakeWS()
    fake_cdp_client = MagicMock(name="cdp_client")

    def _fake_open_cdp(endpoint):
        rec.opened = True
        return fake_cdp_client, fake_ws, "ws://test"

    monkeypatch.setattr(walk_all, "open_cdp", _fake_open_cdp)

    def _fake_read_cube_balance(cdp):
        rec.cube_reads += 1
        return 50

    monkeypatch.setattr(walk_all, "read_cube_balance", _fake_read_cube_balance)

    return rec, fake_cdp_client, fake_ws


def _make_entries(walk_all, ids_states: list[tuple[int, str]]) -> list:
    return [
        walk_all.ModuleListEntry(id=i, title=f"Mod{i}", tier=0,
                                 cubes_to_unlock=0, state=s)
        for i, s in ids_states
    ]


def test_dry_run_lists_modules_without_walking(walk_all, fake_cdp, monkeypatch, tmp_path, capsys):
    rec, _client, _ws = fake_cdp
    entries = _make_entries(walk_all, [(9, "owned"), (15, "in_progress"), (99, "locked")])
    monkeypatch.setattr(walk_all, "fetch_modules_list", lambda cdp: (entries, None))
    # Sentinel: walk_one_module MUST NOT be called in dry-run.
    monkeypatch.setattr(walk_all, "walk_one_module",
                        lambda *a, **kw: pytest.fail("walk_one_module called in dry-run"))

    rc = walk_all.main([
        "--cdp", "http://test",
        "--auto-demo-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0
    assert rec.opened is True
    captured = capsys.readouterr().out
    # Both eligible modules appear in the plan; the locked one does not.
    assert "id=9" in captured
    assert "id=15" in captured
    assert "id=99" not in captured
    # No demo files were written.
    assert list(tmp_path.glob("*.msgpack.gz")) == []


def test_skip_existing_demos_default(walk_all, fake_cdp, monkeypatch, tmp_path):
    """A demo file already on disk causes that module to be skipped (default)."""
    rec, _client, _ws = fake_cdp
    # Pre-create a demo for module 9.
    (tmp_path / "academy_module_9.msgpack.gz").write_bytes(b"existing")
    entries = _make_entries(walk_all, [(9, "owned"), (15, "in_progress")])
    monkeypatch.setattr(walk_all, "fetch_modules_list", lambda cdp: (entries, None))

    walked: list[int] = []

    def _fake_walk_one_module(cdp, module_id, *, auto_demo_dir, max_sections,
                              answerer, cdp_endpoint, section_sleep_s=2.0):
        walked.append(module_id)
        return walk_all.ModuleWalkResult(
            module_id=module_id, title=f"Mod{module_id}",
            sections_walked=1, n_questions=2, n_attempted=2,
            n_turns=3, cheat_rows=0, demo_path=auto_demo_dir / f"academy_module_{module_id}.msgpack.gz",
        )

    monkeypatch.setattr(walk_all, "walk_one_module", _fake_walk_one_module)

    rc = walk_all.main([
        "--cdp", "http://test",
        "--auto-demo-dir", str(tmp_path),
        "--module-sleep-s", "0",
        "--section-sleep-s", "0",
    ])
    assert rc == 0
    assert walked == [15]  # 9 was skipped because its demo already existed.


def test_force_re_walks_existing(walk_all, fake_cdp, monkeypatch, tmp_path):
    """``--force`` makes the driver re-walk modules whose demo already exists."""
    rec, _client, _ws = fake_cdp
    (tmp_path / "academy_module_9.msgpack.gz").write_bytes(b"existing")
    entries = _make_entries(walk_all, [(9, "owned"), (15, "in_progress")])
    monkeypatch.setattr(walk_all, "fetch_modules_list", lambda cdp: (entries, None))

    walked: list[int] = []

    def _fake_walk_one_module(cdp, module_id, *, auto_demo_dir, max_sections,
                              answerer, cdp_endpoint, section_sleep_s=2.0):
        walked.append(module_id)
        return walk_all.ModuleWalkResult(
            module_id=module_id, title=f"Mod{module_id}",
            sections_walked=1, n_questions=1, n_attempted=1,
            n_turns=2, cheat_rows=0, demo_path=auto_demo_dir / f"academy_module_{module_id}.msgpack.gz",
        )

    monkeypatch.setattr(walk_all, "walk_one_module", _fake_walk_one_module)

    rc = walk_all.main([
        "--cdp", "http://test",
        "--auto-demo-dir", str(tmp_path),
        "--module-sleep-s", "0",
        "--section-sleep-s", "0",
        "--force",
    ])
    assert rc == 0
    assert walked == [9, 15]  # Both walked thanks to --force.


def test_unlock_gate_stops_loop_on_partial_walk(walk_all, fake_cdp, monkeypatch, tmp_path):
    """If a module reports n_attempted < n_questions, the driver halts.

    The operator's rule is "open new module only if ALL questions are
    answered". Study-only walks should always reach n_attempted ==
    n_questions, but if the answerer skipped some (rare) the gate must
    refuse to open the next module so the operator can investigate.
    """
    rec, _client, _ws = fake_cdp
    entries = _make_entries(walk_all, [(9, "owned"), (15, "in_progress"), (18, "owned")])
    monkeypatch.setattr(walk_all, "fetch_modules_list", lambda cdp: (entries, None))

    walked: list[int] = []

    def _fake_walk_one_module(cdp, module_id, *, auto_demo_dir, max_sections,
                              answerer, cdp_endpoint, section_sleep_s=2.0):
        walked.append(module_id)
        # Module 9 is partial (only attempted 1 of 3 questions); should
        # cause the driver to stop after its result is summarized.
        if module_id == 9:
            return walk_all.ModuleWalkResult(
                module_id=9, title="Mod9",
                sections_walked=1, n_questions=3, n_attempted=1,
                n_turns=2, cheat_rows=0,
                demo_path=auto_demo_dir / "academy_module_9.msgpack.gz",
            )
        return walk_all.ModuleWalkResult(
            module_id=module_id, title=f"Mod{module_id}",
            sections_walked=1, n_questions=1, n_attempted=1,
            n_turns=2, cheat_rows=0,
            demo_path=auto_demo_dir / f"academy_module_{module_id}.msgpack.gz",
        )

    monkeypatch.setattr(walk_all, "walk_one_module", _fake_walk_one_module)

    rc = walk_all.main([
        "--cdp", "http://test",
        "--auto-demo-dir", str(tmp_path),
        "--module-sleep-s", "0",
        "--section-sleep-s", "0",
    ])
    assert rc == 0
    # Only module 9 was attempted; the loop halted before walking 15 or 18.
    assert walked == [9]


def test_gate_open_when_all_questions_attempted(walk_all, fake_cdp, monkeypatch, tmp_path):
    """When n_attempted == n_questions, the gate opens and the loop continues."""
    rec, _client, _ws = fake_cdp
    entries = _make_entries(walk_all, [(9, "owned"), (15, "in_progress")])
    monkeypatch.setattr(walk_all, "fetch_modules_list", lambda cdp: (entries, None))

    walked: list[int] = []

    def _fake_walk_one_module(cdp, module_id, *, auto_demo_dir, max_sections,
                              answerer, cdp_endpoint, section_sleep_s=2.0):
        walked.append(module_id)
        return walk_all.ModuleWalkResult(
            module_id=module_id, title=f"Mod{module_id}",
            sections_walked=1, n_questions=2, n_attempted=2,
            n_turns=3, cheat_rows=1,
            demo_path=auto_demo_dir / f"academy_module_{module_id}.msgpack.gz",
        )

    monkeypatch.setattr(walk_all, "walk_one_module", _fake_walk_one_module)

    rc = walk_all.main([
        "--cdp", "http://test",
        "--auto-demo-dir", str(tmp_path),
        "--module-sleep-s", "0",
        "--section-sleep-s", "0",
    ])
    assert rc == 0
    assert walked == [9, 15]
