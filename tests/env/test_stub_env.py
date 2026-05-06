"""Tests for the stub env."""

from __future__ import annotations

import pytest

from htbrl.env.base import Action, PentestEnv
from htbrl.env.stub_env import StubPentestEnv
from htbrl.tools.loader import load_registry


@pytest.fixture
def vocab():
    return load_registry()


def test_stub_env_is_pentest_env(vocab):
    env = StubPentestEnv(vocab)
    assert isinstance(env, PentestEnv)
    assert env.matrix == "enterprise"


def test_stub_env_reset_returns_observation(vocab):
    env = StubPentestEnv(vocab)
    obs = env.reset(seed=0)
    assert "stub" in obs.obs_text.lower()
    assert obs.last_reward == 0.0
    assert "favored_first_tool" in obs.parsed_features


def test_stub_env_step_validates_tool_id(vocab):
    env = StubPentestEnv(vocab)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="out of range"):
        env.step(Action(tool_id=99999))


def test_stub_env_terminates_after_max_steps(vocab):
    env = StubPentestEnv(vocab, max_steps=3)
    env.reset(seed=0)
    # Use a deliberately wrong tool every step so we never advance through
    # stages. The env should still terminate after max_steps.
    favored_first = env._favored[0]
    wrong_tool = (favored_first + 1) % vocab.n_tools
    done = False
    steps = 0
    while not done:
        steps += 1
        _, _, done, _ = env.step(Action(tool_id=wrong_tool))
        if steps > 10:
            pytest.fail("env did not terminate within max_steps")
    assert steps == 3


def test_stub_env_advances_stage_on_favored_tool(vocab):
    env = StubPentestEnv(vocab, max_steps=20)
    obs = env.reset(seed=42)
    initial_stage = env._stage
    favored = env._favored[initial_stage]
    next_obs, reward, _, info = env.step(Action(tool_id=favored))
    assert reward > 0  # got the recon stage bonus
    assert env._stage == initial_stage + 1
    # techniques_succeeded should be non-empty when we advanced.
    assert info.techniques_succeeded


def test_stub_env_full_solve_gives_positive_total(vocab):
    """Walking through the favored sequence should net a positive return."""
    env = StubPentestEnv(vocab, max_steps=10)
    env.reset(seed=99)
    total = 0.0
    done = False
    while not done:
        favored = env._favored[env._stage]
        _, r, done, _ = env.step(Action(tool_id=favored))
        total += r
    # Sum of stage rewards minus per-step penalty: ~0.1+0.1+0.5+1.0+1.5+2.0 - 6*0.01
    assert total > 4.0


def test_stub_env_close_blocks_further_steps(vocab):
    env = StubPentestEnv(vocab)
    env.reset(seed=0)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(Action(tool_id=0))


def test_stub_env_context_manager(vocab):
    with StubPentestEnv(vocab) as env:
        env.reset(seed=0)
    # After context exit, env is closed.
    with pytest.raises(RuntimeError):
        env.step(Action(tool_id=0))


def test_stub_env_emits_attack_metadata(vocab):
    env = StubPentestEnv(vocab)
    env.reset(seed=0)
    _, _, _, info = env.step(Action(tool_id=0))
    # Tool 0 is some real tool with ATT&CK metadata.
    assert info.techniques_attempted  # at least one technique tagged
    assert info.tactic_ids
