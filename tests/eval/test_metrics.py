"""Tests for the eval metrics + harness."""

from __future__ import annotations

import pytest
import torch

from htbrl.env.base import StepInfo
from htbrl.env.stub_env import StubPentestEnv
from htbrl.eval.harness import run_episode, run_suite
from htbrl.eval.metrics import (
    EpisodeResult,
    SuiteResult,
    killchain_depth_from_steps,
)
from htbrl.model.policy import ActorCriticPolicy, PolicyConfig
from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tools.loader import load_registry


# ---- killchain depth ---------------------------------------------------------


def test_killchain_depth_counts_distinct_successful_tactics():
    steps = [
        StepInfo(techniques_succeeded=["T1046"], tactic_ids=["TA0043"]),
        StepInfo(techniques_succeeded=["T1018"], tactic_ids=["TA0007"]),
        StepInfo(techniques_succeeded=["T1078"], tactic_ids=["TA0001"]),
    ]
    assert killchain_depth_from_steps(steps) == 3


def test_killchain_depth_ignores_failures():
    steps = [
        StepInfo(techniques_succeeded=["T1046"], tactic_ids=["TA0043"]),
        StepInfo(techniques_succeeded=[], tactic_ids=["TA0007"]),  # failed
        StepInfo(techniques_succeeded=["T1018"], tactic_ids=["TA0007"]),
    ]
    assert killchain_depth_from_steps(steps) == 2


def test_killchain_depth_no_double_count_same_tactic():
    steps = [
        StepInfo(techniques_succeeded=["T1046"], tactic_ids=["TA0043"]),
        StepInfo(techniques_succeeded=["T1595"], tactic_ids=["TA0043"]),  # same tactic
    ]
    assert killchain_depth_from_steps(steps) == 1


def test_killchain_depth_empty():
    assert killchain_depth_from_steps([]) == 0


# ---- EpisodeResult / SuiteResult ---------------------------------------------


def _ep(matrix="enterprise", **kw) -> EpisodeResult:
    return EpisodeResult(
        matrix=matrix,
        target_id=kw.get("target_id", "x"),
        total_reward=kw.get("total_reward", 0.0),
        n_steps=kw.get("n_steps", 1),
        foothold=kw.get("foothold", False),
        user_flag=kw.get("user_flag", False),
        root_flag=kw.get("root_flag", False),
        techniques_attempted=set(kw.get("techniques_attempted", set())),
        techniques_succeeded=set(kw.get("techniques_succeeded", set())),
        tactics_completed=set(kw.get("tactics_completed", set())),
        killchain_depth=kw.get("killchain_depth", 0),
        tools_used=set(kw.get("tools_used", set())),
        step_infos=kw.get("step_infos", []),
    )


def test_suite_outcome_rates():
    s = SuiteResult(episodes=[
        _ep(foothold=True, user_flag=True),
        _ep(foothold=True, user_flag=False),
        _ep(foothold=False),
        _ep(foothold=True, user_flag=True, root_flag=True),
    ])
    assert s.foothold_rate == 0.75
    assert s.user_flag_rate == 0.5
    assert s.root_flag_rate == 0.25


def test_suite_avg_metrics_handle_empty():
    s = SuiteResult()
    assert s.foothold_rate == 0.0
    assert s.avg_episode_length == 0.0


def test_suite_technique_coverage_per_matrix():
    s = SuiteResult(episodes=[
        _ep(matrix="enterprise", techniques_attempted={"T1046", "T1018"}),
        _ep(matrix="enterprise", techniques_attempted={"T1135"}),
        _ep(matrix="ics", techniques_attempted={"T0830"}),
    ])
    cov = s.technique_attempt_coverage()
    assert cov["enterprise"] == {"T1046", "T1018", "T1135"}
    assert cov["ics"] == {"T0830"}


def test_suite_tactic_completion_rate():
    """If TA0007 was attempted in all 3 enterprise episodes and completed in 2
    of them, the rate should be 2/3."""
    s = SuiteResult(episodes=[
        _ep(matrix="enterprise",
            tactics_completed={"TA0007"},
            step_infos=[StepInfo(tactic_ids=["TA0007"], techniques_succeeded=["T1046"])]),
        _ep(matrix="enterprise",
            tactics_completed={"TA0007"},
            step_infos=[StepInfo(tactic_ids=["TA0007"], techniques_succeeded=["T1046"])]),
        _ep(matrix="enterprise",
            tactics_completed=set(),
            step_infos=[StepInfo(tactic_ids=["TA0007"], techniques_succeeded=[])]),
    ])
    rate = s.tactic_completion_rate()
    assert "enterprise" in rate
    assert rate["enterprise"]["TA0007"] == pytest.approx(2 / 3)


def test_suite_avg_killchain_depth():
    s = SuiteResult(episodes=[
        _ep(matrix="enterprise", killchain_depth=3),
        _ep(matrix="enterprise", killchain_depth=5),
        _ep(matrix="ics", killchain_depth=1),
    ])
    avg = s.avg_killchain_depth()
    assert avg["enterprise"] == 4.0
    assert avg["ics"] == 1.0


def test_suite_vocab_coverage():
    s = SuiteResult(episodes=[
        _ep(matrix="enterprise", tools_used={1, 2, 3}),
        _ep(matrix="enterprise", tools_used={3, 4}),
    ])
    cov = s.vocab_coverage(total_tools_per_matrix={"enterprise": 10})
    assert cov["enterprise"] == 0.4  # 4 distinct tools out of 10


def test_suite_summary_dict_is_json_safe():
    """summary_dict should produce only types serializable to JSON without manual coercion."""
    import json
    s = SuiteResult(episodes=[
        _ep(matrix="enterprise", foothold=True, techniques_attempted={"T1046"}),
    ])
    summary = s.summary_dict()
    json.dumps(summary, default=str)  # must not raise


# ---- run_episode against the stub env ----------------------------------------


def _tiny_policy(vocab_size: int, n_tools: int) -> ActorCriticPolicy:
    return ActorCriticPolicy(PolicyConfig(
        vocab_size=vocab_size, n_tools=n_tools,
        d_model=32, n_layers=2, n_heads=4, d_ff=64,
        max_seq_len=64, dropout=0.0,
    ))


def test_run_episode_returns_result_with_attack_metadata():
    torch.manual_seed(0)
    vocab = load_registry()
    tokenizer = ByteLevelBPE.initialize()
    policy = _tiny_policy(tokenizer.vocab_size, vocab.n_tools)
    env = StubPentestEnv(vocab, max_steps=10, seed=0)
    res = run_episode(env, policy, tokenizer, max_seq_len=64, max_steps=10)
    env.close()
    assert res.n_steps > 0
    assert res.matrix == "enterprise"
    # Stub env reports techniques_attempted on every step from real registry.
    assert len(res.techniques_attempted) > 0


def test_run_suite_aggregates_episodes():
    torch.manual_seed(0)
    vocab = load_registry()
    tokenizer = ByteLevelBPE.initialize()
    policy = _tiny_policy(tokenizer.vocab_size, vocab.n_tools)
    factories = [
        lambda i=i: StubPentestEnv(vocab, max_steps=8, seed=i)
        for i in range(2)
    ]
    suite = run_suite(
        env_factories=factories,
        policy=policy,
        tokenizer=tokenizer,
        max_seq_len=64,
        n_episodes_per_env=2,
        max_steps_per_episode=8,
    )
    assert len(suite.episodes) == 4
    cov = suite.technique_attempt_coverage()
    # All episodes are enterprise via the stub.
    assert "enterprise" in cov
