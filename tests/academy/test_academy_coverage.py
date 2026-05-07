"""Tests for ``scripts/academy_coverage.py``.

The script is exercised by importing its ``main`` / ``build_report`` entry
points directly - we keep the heavy lifting out of subprocess so tests
stay fast and don't depend on the .venv interpreter being on PATH.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from htbrl.data.demo_dataset import (
    Demonstration,
    DemoOutcome,
    DemoTurn,
    save_demonstration,
)


# ---- module loader ----------------------------------------------------------

# The coverage script lives under ``scripts/`` which isn't a Python package, so
# we load it explicitly the same way other ``scripts/`` smoke tests do in this
# repo. Doing this once at module-import time keeps the per-test boilerplate
# minimal.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "academy_coverage.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("academy_coverage", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


academy_coverage = _load_script()


# ---- fixtures ---------------------------------------------------------------


def _make_section_read_turn(
    section_index: int = 1,
    section_total: int = 5,
    title: str = "Introduction",
    techs: list[str] | None = None,
) -> DemoTurn:
    return DemoTurn(
        obs_text=f"[academy:{title}] body text " * 10,
        action_tool_id=-1,
        action_tool_name="academy_section_read",
        action_slots={
            "section_index": section_index,
            "section_total": section_total,
            "title": title,
            "n_inline_code": 2,
            "n_bullet_lists": 1,
            "n_code_blocks": 0,
            "n_questions": 1,
        },
        action_render=f"academy_read_section: {title!r}",
        reward=0.02,
        techniques_attempted=list(techs or []),
        techniques_succeeded=[],
    )


def _make_answer_turn(
    method: str = "inline_code_in_match",
    answer_text: str = "ls",
    cubes_reward: int = 0,
    reward: float = -0.1,
    techs: list[str] | None = None,
    accepted: bool = False,
) -> DemoTurn:
    techs = list(techs or [])
    return DemoTurn(
        obs_text="## Question\nWhat command lists files?",
        action_tool_id=-1,
        action_tool_name="academy_answer",
        action_slots={
            "answer": answer_text,
            "method": method,
            "confidence": 0.9,
            "cubes_reward": cubes_reward,
            "hp_reward": 0,
        },
        action_render=f"academy_answer({method}): {answer_text!r}",
        reward=reward,
        techniques_attempted=techs,
        techniques_succeeded=techs if accepted else [],
    )


def _make_demo(
    module_id: str = "1",
    title_section: str = "Introduction",
    n_questions: int = 2,
    n_attempts: int = 2,
    n_accepted: int | None = None,
    methods: list[str] | None = None,
    techs: list[str] | None = None,
    extra_md: dict | None = None,
) -> Demonstration:
    methods = methods or ["inline_code_in_match"]
    turns: list[DemoTurn] = []
    turns.append(_make_section_read_turn(
        section_index=1, section_total=3, title=title_section, techs=techs
    ))
    for i, m in enumerate(methods):
        turns.append(_make_answer_turn(
            method=m,
            cubes_reward=1 if (n_accepted and i < n_accepted) else 0,
            techs=techs,
            accepted=bool(n_accepted and i < n_accepted),
        ))
    md: dict = {
        "source": "htb_academy_auto_learner",
        "module_id": module_id,
        "module_tier": 0,
        "n_questions": n_questions,
        "n_attempts": n_attempts,
    }
    if n_accepted is not None:
        md["n_accepted"] = n_accepted
    if extra_md:
        md.update(extra_md)
    return Demonstration(
        matrix="enterprise",
        target_id=f"htb-academy:{module_id}",
        turns=turns,
        outcome=DemoOutcome(foothold=False, note=f"academy module: {module_id}"),
        metadata=md,
    )


def _write_demos(tmp_path: Path, demos: list[Demonstration]) -> None:
    for d in demos:
        mid = d.metadata.get("module_id", "x")
        save_demonstration(d, tmp_path / f"academy_module_{mid}.msgpack.gz")


# ---- happy path -------------------------------------------------------------


def test_table_contains_both_modules(tmp_path: Path, capsys):
    demos = [
        _make_demo(
            module_id="9",
            title_section="Way of Thinking",
            n_questions=0,
            n_attempts=0,
            methods=[],
            techs=["T1059"],
        ),
        _make_demo(
            module_id="18",
            title_section="Linux Structure",
            n_questions=26,
            n_attempts=26,
            n_accepted=20,
            methods=[
                "inline_code_in_match",
                "wizard_skip_low_conf",
                "howmany_count",
            ],
            techs=["T1083", "T1059.004"],
        ),
    ]
    _write_demos(tmp_path, demos)

    rc = academy_coverage.main(["--demos-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Both module IDs must show up in the table.
    assert "9" in out
    assert "18" in out
    # And titles (truncated form is fine, but our titles fit in 30 chars).
    assert "Way of Thinking" in out
    assert "Linux Structure" in out
    # Method totals across modules must include all methods we wrote.
    assert "inline_code_in_match=" in out
    assert "wizard_skip_low_conf=" in out
    assert "howmany_count=" in out
    # MITRE summary line must list the techniques we tagged.
    assert "T1083" in out
    assert "T1059.004" in out
    # The "Distinct techniques_attempted" header is always emitted when demos
    # are present.
    assert "Distinct techniques_attempted" in out


def test_accepted_falls_back_to_cubes_reward(tmp_path: Path):
    """For older demos without n_accepted, count cubes_reward>0 turns."""
    d = _make_demo(
        module_id="42",
        n_questions=3,
        n_attempts=3,
        n_accepted=None,  # missing -> fall-back path
        methods=["inline_code_in_match", "inline_code_in_match", "inline_code_in_match"],
    )
    # Patch one answer turn to have cubes_reward=1 -> "accepted" by inference.
    accepted_turn = [t for t in d.turns if t.action_tool_name == "academy_answer"][1]
    accepted_turn.action_slots["cubes_reward"] = 1
    _write_demos(tmp_path, [d])

    report = academy_coverage.build_report(tmp_path)
    assert report["n_demos"] == 1
    assert report["modules"][0]["n_accepted"] == 1


# ---- empty dir --------------------------------------------------------------


def test_empty_dir_graceful(tmp_path: Path, capsys):
    rc = academy_coverage.main(["--demos-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Plain-English message + zero exit code.
    assert "no demos found" in out.lower()


def test_missing_dir_graceful(tmp_path: Path, capsys):
    """Non-existent directories should not crash; just report 'no demos'."""
    missing = tmp_path / "does_not_exist"
    rc = academy_coverage.main(["--demos-dir", str(missing)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no demos found" in out.lower()


# ---- --json -----------------------------------------------------------------


def test_json_output_has_expected_keys(tmp_path: Path, capsys):
    demos = [
        _make_demo(
            module_id="9",
            title_section="Way of Thinking",
            n_questions=0,
            methods=[],
            techs=["T1059"],
        ),
        _make_demo(
            module_id="15",
            title_section="Network Foundations",
            n_questions=0,
            methods=[],
            techs=["T1046"],
        ),
    ]
    _write_demos(tmp_path, demos)

    rc = academy_coverage.main(["--demos-dir", str(tmp_path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)

    # Top-level keys.
    for k in (
        "demos_dir",
        "n_demos",
        "min_questions",
        "modules",
        "method_totals",
        "techniques_attempted",
        "techniques_succeeded",
        "module_techniques",
    ):
        assert k in payload, f"missing key {k!r} in JSON report"

    # Modules block.
    assert payload["n_demos"] == 2
    module_ids = {m["module_id"] for m in payload["modules"]}
    assert module_ids == {"9", "15"}
    # Each module entry must carry the per-demo stats fields used by
    # downstream eval / CI gates.
    sample = payload["modules"][0]
    for k in (
        "module_id",
        "title",
        "n_sections",
        "n_questions",
        "n_attempts",
        "n_accepted",
        "n_section_read_turns",
        "n_answer_turns",
        "method_counts",
        "techniques_attempted",
        "theory_bytes",
        "total_reward",
    ):
        assert k in sample, f"missing per-module key {k!r}"

    # Cross-module technique aggregation must dedupe across modules.
    assert set(payload["techniques_attempted"]) == {"T1046", "T1059"}


# ---- --min-questions filter -------------------------------------------------


def test_min_questions_filter_drops_theory_only(tmp_path: Path):
    demos = [
        _make_demo(module_id="9", n_questions=0, methods=[]),
        _make_demo(module_id="18", n_questions=26, n_attempts=26),
    ]
    _write_demos(tmp_path, demos)

    full = academy_coverage.build_report(tmp_path)
    assert full["n_demos"] == 2

    filtered = academy_coverage.build_report(tmp_path, min_questions=1)
    assert filtered["n_demos"] == 1
    assert filtered["modules"][0]["module_id"] == "18"
    assert filtered["n_demos_skipped"] == 1


# ---- corrupt-file resilience ------------------------------------------------


def test_corrupt_demo_warns_but_does_not_crash(tmp_path: Path, capsys):
    # Write one good demo and one obviously-broken file matching the glob.
    _write_demos(tmp_path, [_make_demo(module_id="7")])
    bad = tmp_path / "academy_module_999.msgpack.gz"
    bad.write_bytes(b"not a gzipped msgpack file")

    rc = academy_coverage.main(["--demos-dir", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    # The good demo must still appear in stdout.
    assert "7" in captured.out
    # The corrupt one warns on stderr.
    assert "warning: failed to load" in captured.err
    assert "academy_module_999" in captured.err
