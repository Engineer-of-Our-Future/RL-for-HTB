"""Convert a finished labs walk into a ``Demonstration`` for BC training.

Pulled out of ``.local/write_meow_demo.py`` + ``.local/write_fawn_demo.py``
into a proper module so the next 8 Starting Point walks (and any future
Active Machines / Sherlocks) write demos via one consistent helper.

The demo turn ordering matches what ``train_bc.py`` expects:

  lab_box_intro
  lab_recon_port_scan
  lab_recon_service_id      (one per identified service)
  lab_task_answer × N
  <foothold-action>          (e.g. telnet_login / ftp_anonymous_login /
                              http_login / ssh_with_default_creds / ...)
  <flag-retrieval>           (e.g. cat_flag / ftp_get_flag / web_flag)

The flag string is REDACTED in the demo body — BC only needs to know
"got root flag → reward +1.0", not the literal flag string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from htbrl.data.demo_dataset import (
    Demonstration,
    DemoOutcome,
    DemoTurn,
    save_demonstration,
)


# ---- input dataclasses ------------------------------------------------------


@dataclass
class LabReconStep:
    """One recon step before the task questions begin.

    For port-scan steps, ``service`` should be the empty string and
    ``port`` left as None; for service-id steps, both should be set so
    the demo's reward + technique tags are accurate.
    """

    kind: str                          # "port_scan" | "service_id" | other
    summary: str                       # human-readable rationale
    ip: str
    port: int | None = None
    service: str = ""
    reward: float = 0.0
    techniques_attempted: tuple[str, ...] = ()
    techniques_succeeded: tuple[str, ...] = ()


@dataclass
class LabTask:
    """One guided task (question) on the box's page."""

    number: int                        # 1-indexed task number on app.hackthebox.com
    question: str                      # full prompt text
    answer: str                        # the operator-/agent-supplied answer
    operator_handled: bool = False     # True when operator answered without us
    hint: str = ""                     # optional revealed-hint text


@dataclass
class LabFootholdStep:
    """The actual exploitation step that yields shell / file access."""

    tool_name: str                     # e.g. "telnet_login", "ftp_anonymous_login"
    slots: dict                        # full action_slots dict for the demo turn
    render: str                        # human-readable command shape
    reward: float = 0.50               # default = user_shell-equivalent
    techniques_attempted: tuple[str, ...] = ("T1078.001",)  # Default Accounts
    techniques_succeeded: tuple[str, ...] = ("T1078.001",)
    # Observation snippet to seed into the next turn's ``obs_text``.
    proof: str = ""


@dataclass
class LabFlagStep:
    """The flag-retrieval step. The flag itself is REDACTED here."""

    tool_name: str                     # e.g. "cat_flag", "ftp_get_flag"
    slots: dict                        # action_slots
    render: str
    reward: float = 1.00               # user_flag default
    techniques_attempted: tuple[str, ...] = ("T1083",)  # File/Directory Discovery
    techniques_succeeded: tuple[str, ...] = ("T1083",)


# ---- writer -----------------------------------------------------------------


def write_lab_demo(
    *,
    target_id: str,
    ip: str,
    matrix: str = "enterprise",
    difficulty: str = "very_easy",
    os_name: str = "linux",
    recon: Sequence[LabReconStep],
    tasks: Sequence[LabTask],
    foothold: LabFootholdStep,
    flag_step: LabFlagStep,
    outcome: DemoOutcome | None = None,
    metadata_extras: dict | None = None,
    output_dir: str | Path = "data/demos",
) -> Path:
    """Compose the turn list, write a ``.msgpack.gz`` demo, return its path.

    Returns the final on-disk path. Does NOT mutate any inputs.

    The default ``outcome`` is ``foothold=True, user_flag=True,
    root_flag=False`` (Starting Point boxes don't all distinguish user
    vs root, and the SP "user flag" is the only flag); override
    for boxes where root flag exists separately.
    """
    turns: list[DemoTurn] = []

    # 1. Box intro: zero-action anchor turn so BC can context-condition
    #    on which target the rest of the demo applies to.
    intro_obs = f"<lab episode start: {target_id} @ {ip}>"
    turns.append(DemoTurn(
        obs_text=intro_obs,
        action_tool_id=-1,
        action_tool_name="lab_box_intro",
        action_slots={
            "target_id": target_id,
            "ip": ip,
            "difficulty": difficulty,
            "os": os_name,
        },
        action_render="(target overview)",
        reward=0.0,
        techniques_attempted=[],
        techniques_succeeded=[],
    ))
    last_obs = (
        f"<output of: lab_box_intro>\n"
        f"Target: {ip}  difficulty={difficulty}  os={os_name}"
    )

    # 2. Recon steps (port scan + per-service identification).
    for r in recon:
        turns.append(DemoTurn(
            obs_text=last_obs,
            action_tool_id=-1,
            action_tool_name=f"lab_recon_{r.kind}",
            action_slots={
                "ip": r.ip,
                "port": r.port,
                "service": r.service,
            } if r.port is not None else {"ip": r.ip},
            action_render=r.summary,
            reward=r.reward,
            techniques_attempted=list(r.techniques_attempted),
            techniques_succeeded=list(r.techniques_succeeded),
        ))
        last_obs = f"<output of: lab_recon_{r.kind}>\n{r.summary}"

    # 3. Guided tasks 1..N
    for t in tasks:
        rendered_value = "(operator)" if t.operator_handled else t.answer
        turns.append(DemoTurn(
            obs_text=last_obs,
            action_tool_id=-1,
            action_tool_name="lab_task_answer",
            action_slots={
                "task_number": t.number,
                "question": t.question,
                "answer": t.answer,
                "operator_handled": t.operator_handled,
                **({"hint": t.hint} if t.hint else {}),
            },
            action_render=f"task={t.number} -> {rendered_value!r}",
            reward=0.05,                            # small per-task signal
            techniques_attempted=[],
            techniques_succeeded=[],
        ))
        last_obs = f"<task {t.number} accepted>"

    # 4. Foothold (the actual exploit)
    turns.append(DemoTurn(
        obs_text=last_obs,
        action_tool_id=-1,
        action_tool_name=foothold.tool_name,
        action_slots=dict(foothold.slots),
        action_render=foothold.render,
        reward=foothold.reward,
        techniques_attempted=list(foothold.techniques_attempted),
        techniques_succeeded=list(foothold.techniques_succeeded),
    ))
    last_obs = foothold.proof or f"<foothold step succeeded: {foothold.tool_name}>"

    # 5. Flag retrieval (REDACTED in render)
    turns.append(DemoTurn(
        obs_text=last_obs,
        action_tool_id=-1,
        action_tool_name=flag_step.tool_name,
        action_slots=dict(flag_step.slots),
        action_render=flag_step.render,
        reward=flag_step.reward,
        techniques_attempted=list(flag_step.techniques_attempted),
        techniques_succeeded=list(flag_step.techniques_succeeded),
    ))

    if outcome is None:
        outcome = DemoOutcome(
            foothold=True, user_flag=True, root_flag=False,
            note=(
                f"{target_id} walked task-by-task. Flag captured + "
                f"submitted manually by operator; redacted in demo "
                f"per project policy."
            ),
        )

    metadata = {
        "source": "labs-walk-from-chat",
        "ts": time.time(),
        "kali_host": "claude@192.168.1.219:22",
        "vpn": "htb-starting-point eu-1",
        "n_tasks_completed": len(tasks),
    }
    if metadata_extras:
        metadata.update(metadata_extras)

    demo = Demonstration(
        matrix=matrix,
        target_id=target_id,
        turns=turns,
        outcome=outcome,
        metadata=metadata,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = target_id.replace(":", "-").replace("/", "_")
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    path = out_dir / f"wizard_{safe_id}_{ts}.msgpack.gz"
    save_demonstration(demo, path)
    return path


__all__ = [
    "LabReconStep",
    "LabTask",
    "LabFootholdStep",
    "LabFlagStep",
    "write_lab_demo",
]
