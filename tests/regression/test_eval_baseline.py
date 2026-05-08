"""Regression baselines for the eval harness (PLAN.md Phase 11).

The eval harness emits a JSON metrics blob with foothold rate, user
flag rate, and per-tactic technique coverage. This test pins the
*shape* of that blob and the floor for the must-have keys. It does
NOT pin numeric thresholds — those are checkpoint-dependent and
move with each training run; the per-checkpoint pass criteria are
in `LABS_PLAN.md` and evaluated at training time, not in pytest.

The role of this file is "lock the eval contract": if a refactor
removes the ``foothold_rate`` key or renames ``vocab_coverage``,
this test fails immediately so the contract drift is caught in
CI rather than in a 24-hour training run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from htbrl.eval.metrics import EpisodeResult, SuiteResult


_EXPECTED_TOP_LEVEL_KEYS = {
    # Episode counts — sanity proves the harness ran at least N episodes.
    "n_episodes",
    # Headline rates that gate "production-ready" per PLAN.md Phase 11.
    "foothold_rate",
    "user_flag_rate",
    "root_flag_rate",
    # Reward + length sanity (lets us spot a policy that's stuck in a loop).
    "avg_total_reward",
    "avg_episode_length",
    # MITRE ATT&CK breakdown — proves the agent isn't using one tool only.
    "technique_attempt_coverage",
    "technique_success_coverage",
    "tactic_completion_rate",
    "avg_killchain_depth",
}


def _make_episode(*, foothold=False, user_flag=False, root_flag=False,
                  matrix="enterprise", techs_attempt=(), techs_success=(),
                  tactics=(), reward=0.0, n_steps=10, killchain_depth=0,
                  tools=()) -> EpisodeResult:
    """Tiny constructor so each test reads as ``_make_episode(foothold=True)``
    instead of repeating the full kwarg list."""
    return EpisodeResult(
        matrix=matrix,
        target_id=f"test:{matrix}",
        total_reward=reward,
        n_steps=n_steps,
        foothold=foothold,
        user_flag=user_flag,
        root_flag=root_flag,
        techniques_attempted=set(techs_attempt),
        techniques_succeeded=set(techs_success),
        tactics_completed=set(tactics),
        killchain_depth=killchain_depth,
        tools_used=set(tools),
    )


def test_eval_json_contract_keyset_is_a_superset_of_phase11_baseline():
    """The shape of the JSON eval harness writes must NOT change without
    bumping this test. Driver test ``tests/eval/test_metrics.py`` covers
    the values; this one covers the keyset.

    Synthesised here (not run end-to-end) to avoid a dependency on
    GPU + a checkpoint; the metrics module is the source of truth
    and we verify that what it emits is what eval.py writes.
    """
    suite = SuiteResult(episodes=[
        _make_episode(
            foothold=True, user_flag=True, reward=2.0, n_steps=10,
            techs_attempt=("T1046", "T1190"),
            techs_success=("T1190",),
            tactics=("TA0043", "TA0001"),
            killchain_depth=2, tools=(1, 2),
        ),
        _make_episode(
            reward=-0.05, n_steps=5,
            techs_attempt=("T1046",),
            tactics=("TA0043",),
            tools=(1,),
        ),
    ])
    blob = suite.summary_dict()

    # Top-level keyset is a *superset* of the contract. Extra keys are OK
    # (forward-compat); missing keys are a regression.
    missing = _EXPECTED_TOP_LEVEL_KEYS - set(blob.keys())
    assert not missing, (
        f"eval JSON contract regression: keys missing -> {sorted(missing)!r}. "
        "If you intentionally removed/renamed a key, update "
        "tests/regression/test_eval_baseline.py to match (and bump README's "
        "Phase 11 Status section)."
    )


def test_eval_headline_rates_are_floats_in_unit_interval():
    """foothold/user_flag/root_flag rates must be float in [0, 1]."""
    suite = SuiteResult(episodes=[
        _make_episode(foothold=True, user_flag=True, root_flag=True),
        _make_episode(foothold=True),
        _make_episode(),
    ])
    blob = suite.summary_dict()
    for k in ("foothold_rate", "user_flag_rate", "root_flag_rate"):
        v = blob[k]
        assert isinstance(v, float), f"{k} must be float, got {type(v).__name__}"
        assert 0.0 <= v <= 1.0, f"{k}={v} not in [0, 1]"
    # Sanity: 2/3, 1/3, 1/3 with current synthetic data.
    assert blob["foothold_rate"] == pytest.approx(2 / 3)
    assert blob["user_flag_rate"] == pytest.approx(1 / 3)
    assert blob["root_flag_rate"] == pytest.approx(1 / 3)


def test_eval_per_matrix_breakdown_keyed_by_matrix_name():
    """The technique/tactic dicts split by matrix name so future
    Mobile/ICS suites can co-exist with Enterprise."""
    suite = SuiteResult(episodes=[
        _make_episode(matrix="enterprise", techs_attempt=("T1046",)),
        _make_episode(matrix="mobile", techs_attempt=("T1635",)),
    ])
    blob = suite.summary_dict()
    assert "enterprise" in blob["technique_attempt_coverage"]
    assert "mobile" in blob["technique_attempt_coverage"]
    # The values are sorted lists of technique IDs.
    assert blob["technique_attempt_coverage"]["enterprise"] == ["T1046"]
    assert blob["technique_attempt_coverage"]["mobile"] == ["T1635"]


def test_eval_json_round_trips_through_disk(tmp_path):
    """The JSON metric blob must survive a write→read round-trip
    byte-identical (so external pipelines like dashboards can rely
    on a stable schema)."""
    suite = SuiteResult(episodes=[
        _make_episode(foothold=True, user_flag=True, reward=1.5),
    ])
    blob = suite.summary_dict()
    path = tmp_path / "eval.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(blob, f, sort_keys=True, default=str)

    with path.open("r", encoding="utf-8") as f:
        reloaded = json.load(f)

    # Compare with a JSON round-trip of the original to normalise dict
    # iteration order (which Python 3.7+ guarantees but the JSON spec
    # doesn't).
    expected = json.loads(json.dumps(blob, sort_keys=True, default=str))
    assert reloaded == expected


def test_phase_11_pass_criteria_are_present_in_both_plan_docs():
    """Sanity: both `PLAN.md` and `LABS_PLAN.md` mention the same final
    pass thresholds so the operator can't drift them silently.

    PLAN.md Phase 11 specifies:
      - Foothold rate ≥ 60 % on held-out easy boxes
      - User-flag rate ≥ 30 %
    """
    repo_root = Path(__file__).resolve().parents[2]
    plan = (repo_root / "PLAN.md").read_text(encoding="utf-8").lower()
    labs = (repo_root / "LABS_PLAN.md").read_text(encoding="utf-8").lower()

    # Each doc must mention foothold + user-flag and the 60 / 30 numerics.
    # Use loose matching (regex with flexible whitespace) so tweaking
    # punctuation (% vs ' %', "user-flag" vs "user flag") doesn't break this.
    foothold_60 = re.compile(r"foothold[^.]*60\s*%", re.DOTALL)
    user_flag_30 = re.compile(r"user.flag[^.]*30\s*%", re.DOTALL)

    assert foothold_60.search(plan), \
        "PLAN.md missing the 'foothold ... 60%' Phase-11 threshold"
    assert user_flag_30.search(plan), \
        "PLAN.md missing the 'user-flag ... 30%' Phase-11 threshold"
    assert foothold_60.search(labs), \
        "LABS_PLAN.md missing the 'foothold ... 60%' Phase-11 threshold"
    assert user_flag_30.search(labs), \
        "LABS_PLAN.md missing the 'user-flag ... 30%' Phase-11 threshold"
