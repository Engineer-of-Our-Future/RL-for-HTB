"""Tests for path-aware curriculum + LabWalkthroughBuilder."""

from __future__ import annotations

import pytest

from htbrl.academy.curriculum import (
    CATEGORY_RANK,
    classify_module,
    next_module,
    progress_summary,
)
from htbrl.academy.page_models import (
    AcademyModule,
    AcademySection,
    ProgressState,
    QuestionType,
    AcademyQuestion,
)
from htbrl.academy.walkthrough import (
    find_candidate_flags,
    render_walkthrough,
)
from htbrl.data.demo_dataset import (
    Demonstration,
    DemoOutcome,
    DemoTurn,
)


# ---- classify_module --------------------------------------------------------


def _mod(mid: str, *, tier: int = 1, title: str = "", description: str = "") -> AcademyModule:
    return AcademyModule(
        id=mid, title=title or f"Module {mid}", tier=tier,
        description=description,
    )


def test_classify_tier0_is_general():
    m = _mod("a", tier=0, title="Penetration Testing")
    # tier=0 short-circuits: even an "offensive" title at tier 0 is 'general'.
    assert classify_module(m) == "general"


def test_classify_offensive_keywords():
    m = _mod("a", tier=2, title="Penetration Testing Process",
             description="how attackers exploit web apps")
    assert classify_module(m) == "offensive"


def test_classify_defensive_keywords():
    m = _mod("a", tier=2, title="SOC Hardening",
             description="incident response and detection workflows")
    assert classify_module(m) == "defensive"


def test_classify_general_at_higher_tier():
    m = _mod("a", tier=1, title="Linux Fundamentals",
             description="basics of the linux command line")
    assert classify_module(m) == "general"


def test_classify_falls_back_to_other():
    m = _mod("a", tier=2, title="Deep Dive Into Some Niche",
             description="specialized content unrelated to other categories")
    assert classify_module(m) == "other"


# ---- next_module curriculum ordering ----------------------------------------


def test_next_module_picks_general_before_offensive():
    mods = [
        _mod("offensive-1", tier=1, title="Attack the Web"),
        _mod("general-1",   tier=1, title="Linux Fundamentals"),
        _mod("defensive-1", tier=1, title="SOC Analyst"),
    ]
    # Pre-classify so tests don't depend on heuristic edge cases
    for m in mods:
        m.category = m.id.split("-")[0]
    state = ProgressState(user_id="u", cubes_balance=100)
    assert next_module(mods, state).id == "general-1"


def test_next_module_within_category_uses_tier_then_cubes():
    mods = [
        AcademyModule(id="a", title="x", tier=2, cubes_to_unlock=10, category="general"),
        AcademyModule(id="b", title="x", tier=1, cubes_to_unlock=20, category="general"),
        AcademyModule(id="c", title="x", tier=1, cubes_to_unlock=10, category="general"),
    ]
    state = ProgressState(user_id="u", cubes_balance=100)
    # tier ascending first => 1,1,2 ; then cubes ascending => c (10), b (20), a (20)
    assert next_module(mods, state).id == "c"


def test_next_module_full_curriculum_walk():
    mods = [
        AcademyModule(id="g", title="x", tier=0, category="general"),
        AcademyModule(id="o", title="x", tier=1, category="offensive"),
        AcademyModule(id="d", title="x", tier=1, category="defensive"),
        AcademyModule(id="x", title="x", tier=1, category="other"),
    ]
    state = ProgressState(user_id="u", cubes_balance=100)
    expected = ["g", "o", "d", "x"]
    for want in expected:
        m = next_module(mods, state)
        assert m is not None and m.id == want, f"expected {want}, got {m and m.id}"
        state.completed_module_ids.append(m.id)
    assert next_module(mods, state) is None


def test_next_module_preferred_order_overrides_category():
    mods = [
        AcademyModule(id="g", title="x", tier=0, category="general"),
        AcademyModule(id="o", title="x", tier=1, category="offensive"),
    ]
    state = ProgressState(user_id="u", cubes_balance=100)
    assert next_module(mods, state, preferred_order=["o"]).id == "o"


def test_progress_summary_includes_per_category_counts():
    mods = [
        AcademyModule(id="g1", title="x", tier=0, category="general"),
        AcademyModule(id="g2", title="x", tier=0, category="general"),
        AcademyModule(id="o1", title="x", tier=1, category="offensive"),
    ]
    state = ProgressState(user_id="u", completed_module_ids=["g1"])
    s = progress_summary(mods, state)
    assert s["by_category"]["general"]["total"] == 2
    assert s["by_category"]["general"]["done"] == 1
    assert s["by_category"]["offensive"]["total"] == 1
    assert s["by_category"]["offensive"]["done"] == 0


def test_category_rank_constants_present():
    assert CATEGORY_RANK == {
        "general": 0, "offensive": 1, "defensive": 2, "other": 3,
    }


# ---- LabWalkthroughBuilder --------------------------------------------------


def _demo_with_turns(*turns: DemoTurn,
                     foothold: bool = False, user_flag: bool = False,
                     root_flag: bool = False) -> Demonstration:
    return Demonstration(
        matrix="enterprise",
        target_id="htb:test-box",
        turns=list(turns),
        outcome=DemoOutcome(foothold=foothold, user_flag=user_flag, root_flag=root_flag),
    )


def _t(tool: str = "nmap_quick_tcp", out: str = "", reward: float = 0.1) -> DemoTurn:
    return DemoTurn(
        obs_text=out, action_tool_id=0, action_tool_name=tool,
        action_slots={"ip": "10.10.10.5"},
        action_render=f"{tool} 10.10.10.5", reward=reward,
        techniques_attempted=["T1046"], techniques_succeeded=["T1046"] if reward > 0 else [],
    )


def test_walkthrough_renders_step_per_turn():
    demo = _demo_with_turns(_t(), _t(reward=0.5))
    md = render_walkthrough(demo)
    assert "Walkthrough: htb:test-box" in md
    assert "Step 1" in md
    assert "Step 2" in md
    assert "Total steps: 2" in md


def test_walkthrough_emits_outcome_flags():
    demo = _demo_with_turns(_t(), foothold=True, user_flag=True)
    md = render_walkthrough(demo)
    assert "foothold=Y" in md
    assert "user_flag=Y" in md
    assert "root_flag=N" in md


def test_walkthrough_truncates_long_observations():
    long_blob = "X" * 12000
    demo = _demo_with_turns(_t(out=long_blob))
    md = render_walkthrough(demo, output_truncate_chars=4096)
    assert "bytes truncated" in md
    # Truncated form must not contain the entire 12 KB blob
    assert md.count("X") < 11000


def test_walkthrough_keep_full_obs_when_requested():
    long_blob = "X" * 12000
    demo = _demo_with_turns(_t(out=long_blob))
    md = render_walkthrough(demo, include_full_obs=True)
    assert md.count("X") >= 12000
    assert "bytes truncated" not in md


def test_find_candidate_flags_htb_braced():
    demo = _demo_with_turns(_t(out="some output\nHTB{this_is_the_flag}\nmore"))
    cands = find_candidate_flags(demo)
    assert any(c.flag == "HTB{this_is_the_flag}" for c in cands)


def test_find_candidate_flags_hex_line():
    demo = _demo_with_turns(_t(out="header\nabcdef0123456789abcdef0123456789\ntail"))
    cands = find_candidate_flags(demo)
    assert any(c.flag == "abcdef0123456789abcdef0123456789" for c in cands)


def test_find_candidate_flags_dedupe():
    """If the same flag appears twice, only one candidate is emitted."""
    out = "HTB{x}\nblah\nHTB{x}\n"
    demo = _demo_with_turns(_t(out=out))
    cands = find_candidate_flags(demo)
    assert len([c for c in cands if c.flag == "HTB{x}"]) == 1


def test_walkthrough_includes_warning_about_manual_submit():
    demo = _demo_with_turns(_t(out="HTB{flag}"))
    md = render_walkthrough(demo)
    assert "REVIEW BEFORE SUBMITTING" in md
    assert "submit manually" in md.lower()


def test_walkthrough_no_candidates_section_still_emitted():
    """Even if no flags found, the candidate-flags section + warning must appear."""
    demo = _demo_with_turns(_t(out="just some text"))
    md = render_walkthrough(demo)
    assert "Candidate flags" in md
    assert "No candidate flags detected" in md
