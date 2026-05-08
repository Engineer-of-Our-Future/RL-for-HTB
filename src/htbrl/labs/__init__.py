"""HTB Labs walker — Starting Point / Active Machines / Sherlocks / Challenges.

Sister package to ``htbrl.academy`` (which targets academy.hackthebox.com).
This one targets ``app.hackthebox.com`` — the labs side. Each box has a
sequence of guided tasks (questions) followed by a final flag submit,
structurally similar to academy module sections.

Public surfaces:

  - ``demo_writer.write_lab_demo`` — turn a finished walk's structured
    state into a ``Demonstration`` for BC training.
  - ``demo_writer.LabTask`` — dataclass for one task (number, question,
    answer, was-operator-handled, optional hint).
  - ``demo_writer.LabReconStep`` — dataclass for recon steps before the
    tasks (port scan, service id, etc).
  - ``cdp_walker`` — CDP-driven DOM walker. Stubs for now; full plumbing
    requires live verification against ``app.hackthebox.com``.

Used by ``scripts/htb_labs_wizard.py`` (parallel to ``htb_academy_wizard.py``).
"""

from htbrl.labs.demo_writer import (
    LabReconStep,
    LabTask,
    LabFootholdStep,
    LabFlagStep,
    write_lab_demo,
)

__all__ = [
    "LabReconStep",
    "LabTask",
    "LabFootholdStep",
    "LabFlagStep",
    "write_lab_demo",
]
