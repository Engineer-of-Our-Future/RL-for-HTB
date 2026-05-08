"""Tests for ``htbrl.labs.cdp_walker``.

The walker can't be live-tested without an attached Chrome on
``app.hackthebox.com``, so we mock ``CDPClient.evaluate`` to return
canned scraper results. The tests pin:

  - The dataclass shapes (LabsBoxPage / LabsTask).
  - That ``scrape_box_page`` lifts a JS-eval result into the dataclass
    cleanly (including the ``flag.input_id``-vs-``flag_input_id`` rename).
  - That ``submit_task_in_dom`` rejects when the JS reports failure.
  - That ``submit_task_in_dom`` polls scrape_box_page until the
    task shows ``accepted`` then returns ``"accepted"``.
  - That ``submit_flag_in_dom`` does the equivalent for the final flag.
  - That degenerate scraper output (no tasks, missing fields) yields
    a sane empty page rather than crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from htbrl.labs.cdp_walker import (
    LabsBoxPage,
    LabsTask,
    scrape_box_page,
    submit_flag_in_dom,
    submit_task_in_dom,
)


# ---- helpers --------------------------------------------------------------


def _scraper_result(*, target_ip="10.129.119.129", connected=True,
                    tasks=None, flag=None, url="https://app.hackthebox.com/...",
                    ) -> dict:
    """Build the dict shape the page-scraper JS returns."""
    return {
        "url": url,
        "target_ip": target_ip,
        "connected_to_htb": connected,
        "tasks": tasks if tasks is not None else [],
        "flag": flag if flag is not None else {
            "input_id": "", "locked": True, "accepted": False,
        },
    }


def _task_dict(n, q="?", **over):
    base = dict(
        number=n, question=q, placeholder="",
        accepted=False, locked=False, hint_visible=True,
        input_id=f"q{n}", submit_button_text="Submit Task",
    )
    base.update(over)
    return base


def _make_cdp(eval_returns):
    """A stub CDPClient whose .evaluate() returns the next canned value."""
    cdp = MagicMock()
    if not isinstance(eval_returns, list):
        eval_returns = [eval_returns]
    cdp.evaluate = MagicMock(side_effect=list(eval_returns))
    return cdp


# ---- scrape_box_page ------------------------------------------------------


def test_scrape_box_page_reads_target_ip_and_tasks():
    cdp = _make_cdp(_scraper_result(
        target_ip="10.129.119.167",
        tasks=[
            _task_dict(1, q="What does SQL stand for?"),
            _task_dict(2, q="Vuln name?", accepted=True),
        ],
        flag={"input_id": "fi", "locked": True, "accepted": False},
    ))
    page = scrape_box_page(cdp)
    assert isinstance(page, LabsBoxPage)
    assert page.target_ip == "10.129.119.167"
    assert page.connected_to_htb is True
    assert page.flag_input_id == "fi"
    assert page.flag_input_locked is True
    assert page.flag_accepted is False
    assert page.n_tasks == 2
    assert page.tasks[0].number == 1
    assert page.tasks[0].question == "What does SQL stand for?"
    assert page.tasks[1].accepted is True
    # Counters
    assert page.n_accepted_tasks == 1


def test_scrape_box_page_handles_empty_scraper_result():
    """If the page hasn't loaded / DOM doesn't match, scrape_box_page
    returns an empty LabsBoxPage rather than crashing."""
    cdp = _make_cdp({})
    page = scrape_box_page(cdp)
    assert page.target_ip == ""
    assert page.connected_to_htb is False
    assert page.tasks == []
    assert page.flag_input_id == ""
    assert page.flag_input_locked is True


def test_scrape_box_page_handles_evaluate_exception():
    """A JS-eval error mustn't propagate; we return an empty page."""
    cdp = MagicMock()
    cdp.evaluate = MagicMock(side_effect=RuntimeError("CDP boom"))
    page = scrape_box_page(cdp)
    assert page == LabsBoxPage()


# ---- submit_task_in_dom ---------------------------------------------------


def test_submit_task_returns_error_when_js_says_no_card():
    cdp = MagicMock()
    cdp.evaluate = MagicMock(return_value={"ok": False, "why": "no card matches Task 99"})
    state, detail = submit_task_in_dom(cdp, 99, "irrelevant")
    assert state == "error"
    assert "no card matches" in detail


def test_submit_task_polls_until_accepted():
    """First eval = fill-and-submit (ok); second eval onwards =
    scrape_box_page polling. We canned: scrape #1 -> task still
    pending, scrape #2 -> task accepted."""
    cdp = MagicMock()
    cdp.evaluate = MagicMock(side_effect=[
        # 1) fill_and_submit returns ok
        {"ok": True, "value": "Virtual Machine"},
        # 2) first poll: task not yet accepted
        _scraper_result(tasks=[_task_dict(1, accepted=False)]),
        # 3) second poll: accepted!
        _scraper_result(tasks=[_task_dict(1, accepted=True)]),
    ])
    state, detail = submit_task_in_dom(cdp, 1, "Virtual Machine", poll_seconds=2.0)
    assert state == "accepted"
    assert detail == "polled"


def test_submit_task_returns_pending_when_poll_window_expires():
    """If the task never flips to accepted within poll_seconds, we
    surface ``pending`` (not ``accepted``, not ``error``)."""
    cdp = MagicMock()
    cdp.evaluate = MagicMock(side_effect=[
        {"ok": True, "value": "x"},
        # endless "still pending" responses
        *([_scraper_result(tasks=[_task_dict(1, accepted=False)])] * 50),
    ])
    state, detail = submit_task_in_dom(cdp, 1, "x", poll_seconds=0.5)
    assert state == "pending"
    assert "no terminal state" in detail


# ---- submit_flag_in_dom ---------------------------------------------------


def test_submit_flag_returns_error_when_no_button():
    cdp = MagicMock()
    cdp.evaluate = MagicMock(return_value={"ok": False, "why": "no Submit Flag button"})
    state, detail = submit_flag_in_dom(cdp, "abcdef" * 5)
    assert state == "error"
    assert "Submit Flag" in detail


def test_submit_flag_polls_until_accepted():
    """flag_accepted=True on the second poll → returns 'accepted'."""
    cdp = MagicMock()
    cdp.evaluate = MagicMock(side_effect=[
        {"ok": True, "value": "deadbeef" * 4},
        _scraper_result(flag={"input_id": "fi", "locked": True, "accepted": False}),
        _scraper_result(flag={"input_id": "fi", "locked": True, "accepted": True}),
    ])
    state, detail = submit_flag_in_dom(cdp, "deadbeef" * 4, poll_seconds=2.0)
    assert state == "accepted"


# ---- LabsBoxPage helpers ---------------------------------------------------


def test_lab_box_page_n_accepted_counts_only_accepted_tasks():
    page = LabsBoxPage(
        target_ip="x", tasks=[
            LabsTask(number=1, question="?", accepted=True),
            LabsTask(number=2, question="?", accepted=False),
            LabsTask(number=3, question="?", accepted=True),
        ],
    )
    assert page.n_tasks == 3
    assert page.n_accepted_tasks == 2


def test_lab_box_page_default_initialisation_is_empty_and_safe():
    p = LabsBoxPage()
    assert p.target_ip == ""
    assert p.target_id == ""
    assert p.tasks == []
    assert p.connected_to_htb is False
    assert p.flag_input_locked is True
    assert p.flag_accepted is False
    # Counters survive empty list
    assert p.n_tasks == 0
    assert p.n_accepted_tasks == 0
