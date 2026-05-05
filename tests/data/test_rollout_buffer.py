"""Tests for RolloutBuffer."""

from __future__ import annotations

import pytest
import torch

from htbrl.data.rollout_buffer import RolloutBatch, RolloutBuffer


def _populate(buf: RolloutBuffer, fill_value: float = 0.0) -> None:
    """Write n_steps full rows into the buffer."""
    n_envs = buf.n_envs
    seq = buf.max_seq_len
    for t in range(buf.n_steps):
        buf.add(
            obs_tokens=torch.full((n_envs, seq), int(t)),
            attn_mask=torch.ones(n_envs, seq, dtype=torch.bool),
            matrix_id=torch.zeros(n_envs, dtype=torch.int64),
            action_tool_id=torch.full((n_envs,), int(t)),
            log_prob=torch.full((n_envs,), -float(t) - 0.1),
            value=torch.full((n_envs,), float(t)),
            reward=torch.full((n_envs,), fill_value + float(t)),
            done=torch.zeros(n_envs, dtype=torch.bool),
        )


def test_empty_buffer_state():
    buf = RolloutBuffer(n_steps=4, n_envs=2, max_seq_len=8, pin_memory=False)
    assert not buf.is_full
    assert buf.total_transitions == 0


def test_add_advances_pointer_and_fills():
    buf = RolloutBuffer(n_steps=3, n_envs=2, max_seq_len=4, pin_memory=False)
    _populate(buf)
    assert buf.is_full
    assert buf.total_transitions == 6
    # Step 0 obs tokens are all 0
    assert (buf.obs_tokens[0] == 0).all()
    # Step 2 obs tokens are all 2
    assert (buf.obs_tokens[2] == 2).all()


def test_add_after_full_raises():
    buf = RolloutBuffer(n_steps=2, n_envs=1, max_seq_len=2, pin_memory=False)
    _populate(buf)
    with pytest.raises(RuntimeError, match="full"):
        buf.add(
            obs_tokens=torch.zeros(1, 2, dtype=torch.int64),
            attn_mask=torch.ones(1, 2, dtype=torch.bool),
            matrix_id=torch.zeros(1, dtype=torch.int64),
            action_tool_id=torch.zeros(1, dtype=torch.int64),
            log_prob=torch.zeros(1),
            value=torch.zeros(1),
            reward=torch.zeros(1),
            done=torch.zeros(1, dtype=torch.bool),
        )


def test_iter_minibatches_before_full_raises():
    buf = RolloutBuffer(n_steps=2, n_envs=1, max_seq_len=2, pin_memory=False)
    with pytest.raises(RuntimeError, match="not full"):
        list(buf.iter_minibatches(minibatch_size=1))


def test_iter_minibatches_yields_correct_total():
    buf = RolloutBuffer(n_steps=4, n_envs=2, max_seq_len=2, pin_memory=False)
    _populate(buf)
    buf.set_advantages_and_returns(torch.zeros(4, 2), torch.zeros(4, 2))
    batches = list(buf.iter_minibatches(minibatch_size=3, shuffle=False))
    # 8 transitions total, mb=3 -> 3 batches of sizes 3, 3, 2
    assert sum(b.obs_tokens.shape[0] for b in batches) == 8
    assert all(isinstance(b, RolloutBatch) for b in batches)


def test_iter_minibatches_obs_shape():
    buf = RolloutBuffer(n_steps=2, n_envs=2, max_seq_len=5, pin_memory=False)
    _populate(buf)
    buf.set_advantages_and_returns(torch.zeros(2, 2), torch.zeros(2, 2))
    for b in buf.iter_minibatches(minibatch_size=2):
        assert b.obs_tokens.shape == (2, 5)
        assert b.attn_mask.shape == (2, 5)
        assert b.matrix_id.shape == (2,)
        break


def test_set_advantages_and_returns_shape_check():
    buf = RolloutBuffer(n_steps=2, n_envs=2, max_seq_len=2, pin_memory=False)
    with pytest.raises(ValueError, match="advantage shape"):
        buf.set_advantages_and_returns(
            torch.zeros(3, 2), torch.zeros(3, 2)
        )


def test_obs_tokens_shape_check_in_add():
    buf = RolloutBuffer(n_steps=2, n_envs=2, max_seq_len=4, pin_memory=False)
    with pytest.raises(ValueError, match="obs_tokens shape"):
        buf.add(
            obs_tokens=torch.zeros(2, 99, dtype=torch.int64),
            attn_mask=torch.ones(2, 4, dtype=torch.bool),
            matrix_id=torch.zeros(2, dtype=torch.int64),
            action_tool_id=torch.zeros(2, dtype=torch.int64),
            log_prob=torch.zeros(2),
            value=torch.zeros(2),
            reward=torch.zeros(2),
            done=torch.zeros(2, dtype=torch.bool),
        )


def test_reset_clears_pointer():
    buf = RolloutBuffer(n_steps=2, n_envs=1, max_seq_len=2, pin_memory=False)
    _populate(buf)
    assert buf.is_full
    buf.reset()
    assert not buf.is_full
    # Should be writable again
    _populate(buf)
    assert buf.is_full
