"""Tests for ``htbrl.labs.demo_writer.write_lab_demo``.

Verifies the turn ordering pinned by the doc + the metadata + that
the demo round-trips through msgpack so train_bc.py can read it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from htbrl.data.demo_dataset import DemoOutcome, load_demonstration
from htbrl.labs.demo_writer import (
    LabFlagStep,
    LabFootholdStep,
    LabReconStep,
    LabTask,
    write_lab_demo,
)


# ---- helpers --------------------------------------------------------------


def _make_recon():
    """Two recon steps (port_scan + service_id) like Meow / Fawn use."""
    return [
        LabReconStep(
            kind="port_scan",
            summary="(python socket scan; only 21/tcp open)",
            ip="10.129.119.141",
            reward=0.10,
            techniques_attempted=("T1046",),
            techniques_succeeded=("T1046",),
        ),
        LabReconStep(
            kind="service_id",
            summary="(banner: vsFTPd 3.0.3)",
            ip="10.129.119.141",
            port=21,
            service="ftp",
            reward=0.20,
            techniques_attempted=("T1046",),
            techniques_succeeded=("T1046",),
        ),
    ]


def _make_tasks():
    return [
        LabTask(number=1, question="What does FTP stand for?",
                answer="File Transfer Protocol"),
        LabTask(number=2, question="Which port?", answer="21"),
        LabTask(number=4, question="ICMP echo tool?", answer="ping",
                operator_handled=True),
    ]


def _make_foothold():
    return LabFootholdStep(
        tool_name="ftp_anonymous_login",
        slots={"ip": "10.129.119.141", "port": 21, "username": "anonymous",
               "password": "anonymous@example.com"},
        render="ftp 10.129.119.141 -> USER anonymous -> 230 Login successful",
        reward=0.50,
        techniques_attempted=("T1078.001",),
        techniques_succeeded=("T1078.001",),
        proof="<ftp anonymous@... succeeded; 230 Login successful>",
    )


def _make_flag_step():
    return LabFlagStep(
        tool_name="ftp_get_flag",
        slots={"path": "flag.txt"},
        render="get flag.txt -> <REDACTED 32-hex-char flag>",
        reward=1.00,
    )


# ---- happy-path round-trip ------------------------------------------------


def test_write_lab_demo_round_trips_through_msgpack(tmp_path):
    """Write -> read back gives an identical-shape demo."""
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    assert path.exists()
    demo = load_demonstration(path)
    assert demo.matrix == "enterprise"
    assert demo.target_id == "htb-starting-point:fawn"
    assert demo.outcome.foothold is True
    assert demo.outcome.user_flag is True
    assert demo.outcome.root_flag is False


def test_turn_ordering_matches_canonical_lab_walk(tmp_path):
    """The pinned ordering: intro -> recon × M -> tasks × N -> foothold -> flag."""
    recon = _make_recon()
    tasks = _make_tasks()
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=recon,
        tasks=tasks,
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    expected = (
        ["lab_box_intro"]
        + ["lab_recon_port_scan", "lab_recon_service_id"]
        + ["lab_task_answer"] * len(tasks)
        + ["ftp_anonymous_login", "ftp_get_flag"]
    )
    actual = [t.action_tool_name for t in demo.turns]
    assert actual == expected


def test_per_task_reward_is_small_uniform(tmp_path):
    """All ``lab_task_answer`` turns get +0.05; the flag step gets +1.0."""
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    task_rewards = [
        t.reward for t in demo.turns if t.action_tool_name == "lab_task_answer"
    ]
    assert task_rewards == [0.05, 0.05, 0.05]
    flag_turn = next(t for t in demo.turns if t.action_tool_name == "ftp_get_flag")
    assert flag_turn.reward == pytest.approx(1.00)


def test_operator_handled_tasks_render_as_placeholder(tmp_path):
    """When ``operator_handled=True`` the demo's render shows
    "(operator)" so BC doesn't learn to memorize answers we
    didn't actually provide."""
    tasks = [
        LabTask(number=1, question="Q1?", answer="real-answer"),
        LabTask(number=4, question="Q4?", answer="ping",
                operator_handled=True),
    ]
    path = write_lab_demo(
        target_id="htb-starting-point:demo",
        ip="10.0.0.1",
        recon=[],                                  # no recon for this test
        tasks=tasks,
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    task_turns = [t for t in demo.turns if t.action_tool_name == "lab_task_answer"]
    assert "real-answer" in task_turns[0].action_render
    assert "(operator)" in task_turns[1].action_render
    # The operator_handled flag survives in slots so audits can find which
    # answers came from the agent vs the human.
    assert task_turns[0].action_slots["operator_handled"] is False
    assert task_turns[1].action_slots["operator_handled"] is True


def test_metadata_includes_walker_signature(tmp_path):
    """The demo should be self-describing: source, ts, kali_host, n_tasks."""
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    assert demo.metadata["source"] == "labs-walk-from-chat"
    assert "kali_host" in demo.metadata
    assert "vpn" in demo.metadata
    assert demo.metadata["n_tasks_completed"] == 3
    # ts is a recent unix timestamp
    assert demo.metadata["ts"] > time.time() - 60


def test_metadata_extras_merge_into_demo_metadata(tmp_path):
    """Caller-supplied extras don't clobber the defaults but do
    extend them."""
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        metadata_extras={"tools_observed": ["ftp", "vsftpd"]},
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    assert demo.metadata["tools_observed"] == ["ftp", "vsftpd"]
    assert demo.metadata["source"] == "labs-walk-from-chat"


def test_default_outcome_marks_foothold_and_user_flag(tmp_path):
    """Default outcome assumes both flags captured (foothold + user)."""
    path = write_lab_demo(
        target_id="htb-starting-point:fawn",
        ip="10.129.119.141",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    assert demo.outcome.foothold is True
    assert demo.outcome.user_flag is True
    assert demo.outcome.root_flag is False
    assert "redacted" in demo.outcome.note.lower()


def test_caller_can_override_outcome(tmp_path):
    """For boxes with both user + root flags, caller passes a custom
    outcome; the writer doesn't second-guess it."""
    custom = DemoOutcome(foothold=True, user_flag=True, root_flag=True,
                         note="root popped via dirtycow")
    path = write_lab_demo(
        target_id="htb-machines:lame",
        ip="10.10.10.3",
        recon=_make_recon(),
        tasks=_make_tasks(),
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        outcome=custom,
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    assert demo.outcome.root_flag is True
    assert "dirtycow" in demo.outcome.note


def test_demo_filename_uses_safe_target_id(tmp_path):
    """Colons in the target_id (htb-starting-point:meow) get replaced
    so the path is valid on Windows."""
    path = write_lab_demo(
        target_id="htb-starting-point:meow",
        ip="10.129.0.1",
        recon=[],
        tasks=[LabTask(number=1, question="?", answer="a")],
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    # No colon in the filename, even though the target_id has one.
    assert ":" not in path.name
    assert "htb-starting-point-meow" in path.name


def test_no_recon_steps_means_no_recon_turns(tmp_path):
    """Theory-only walks (no recon performed) skip the recon turns."""
    path = write_lab_demo(
        target_id="theory:noop",
        ip="0.0.0.0",
        recon=[],
        tasks=[LabTask(number=1, question="?", answer="a")],
        foothold=_make_foothold(),
        flag_step=_make_flag_step(),
        output_dir=tmp_path,
    )
    demo = load_demonstration(path)
    recon_turns = [
        t for t in demo.turns if t.action_tool_name.startswith("lab_recon_")
    ]
    assert recon_turns == []
