"""Tests for the Mobile + ICS stub envs (Phase 13/16 prep)."""

from __future__ import annotations

import pytest

from htbrl.env.base import Action, PentestEnv
from htbrl.env.ics_stub_env import ICSStubEnv
from htbrl.env.mobile_stub_env import MobileStubEnv
from htbrl.tools.loader import load_registry


@pytest.fixture
def vocab():
    return load_registry()


# ---- mobile -----------------------------------------------------------------


def test_mobile_stub_is_pentest_env(vocab):
    env = MobileStubEnv(vocab)
    assert isinstance(env, PentestEnv)
    assert env.matrix == "mobile"


def test_mobile_stub_reset_returns_obs(vocab):
    env = MobileStubEnv(vocab, seed=0)
    obs = env.reset()
    assert "mobile-stub" in obs.obs_text
    assert "favored_first_tool" in obs.parsed_features


def test_mobile_stub_advances_on_favored_tool(vocab):
    env = MobileStubEnv(vocab, max_steps=20, seed=1)
    env.reset()
    initial_stage = env._stage
    favored = env._favored[initial_stage]
    _, reward, _, info = env.step(Action(tool_id=favored))
    assert reward > 0
    assert env._stage == initial_stage + 1
    assert "TA0032" in info.tactic_ids or info.techniques_succeeded


def test_mobile_stub_full_solve(vocab):
    env = MobileStubEnv(vocab, max_steps=20, seed=42)
    env.reset()
    total = 0.0
    done = False
    while not done:
        favored = env._favored[env._stage]
        _, r, done, _ = env.step(Action(tool_id=favored))
        total += r
    # Sum: 0.1 + 0.5 + 0.7 + 1.0 + 1.5 minus 5 step penalties
    assert total > 3.5


def test_mobile_stub_close_blocks_step(vocab):
    env = MobileStubEnv(vocab)
    env.reset()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(Action(tool_id=0))


# ---- ICS --------------------------------------------------------------------


def test_ics_stub_is_pentest_env(vocab):
    env = ICSStubEnv(vocab)
    assert isinstance(env, PentestEnv)
    assert env.matrix == "ics"


def test_ics_stub_reset_returns_obs(vocab):
    env = ICSStubEnv(vocab, seed=0)
    obs = env.reset()
    assert "ics-stub" in obs.obs_text
    assert "favored_first_tool" in obs.parsed_features


def test_ics_stub_advances_on_favored_tool(vocab):
    env = ICSStubEnv(vocab, max_steps=20, seed=2)
    env.reset()
    favored = env._favored[0]
    _, reward, _, info = env.step(Action(tool_id=favored))
    assert reward > 0
    assert env._stage == 1


def test_ics_stub_full_solve(vocab):
    env = ICSStubEnv(vocab, max_steps=20, seed=99)
    env.reset()
    total = 0.0
    done = False
    while not done:
        favored = env._favored[env._stage]
        _, r, done, _ = env.step(Action(tool_id=favored))
        total += r
    # Sum: 0.1 + 0.2 + 0.5 + 1.0 + 2.0 minus 5 step penalties
    assert total > 3.5


def test_ics_stub_terminates_on_max_steps(vocab):
    env = ICSStubEnv(vocab, max_steps=3, seed=7)
    env.reset()
    favored_0 = env._favored[0]
    wrong = (favored_0 + 1) % vocab.n_tools
    done = False
    n = 0
    while not done:
        n += 1
        _, _, done, _ = env.step(Action(tool_id=wrong))
        if n > 10:
            pytest.fail("env did not terminate within max_steps")
    assert n == 3
