"""Tests for the turn-structured state encoder."""

from __future__ import annotations

import pytest

from htbrl.data.encode_state import Turn, all_special_token_names, encode_episode_window
from htbrl.tokenizer.bpe import ByteLevelBPE


def _new_tokenizer() -> ByteLevelBPE:
    return ByteLevelBPE.initialize()


def test_encode_emits_bos_and_matrix_token_first():
    tok = _new_tokenizer()
    ids, mask = encode_episode_window(tok, "enterprise", [], max_seq_len=64)
    assert ids[0].item() == tok.special_id("<bos>")
    assert ids[1].item() == tok.special_id("<matrix:enterprise>")


@pytest.mark.parametrize("matrix", ["enterprise", "mobile", "ics"])
def test_encode_each_matrix_uses_its_token(matrix):
    tok = _new_tokenizer()
    ids, _ = encode_episode_window(tok, matrix, [], max_seq_len=64)
    assert ids[1].item() == tok.special_id(f"<matrix:{matrix}>")


def test_encode_unknown_matrix_raises():
    tok = _new_tokenizer()
    with pytest.raises(ValueError, match="unknown matrix"):
        encode_episode_window(tok, "windows", [], max_seq_len=64)


def test_pad_token_fills_remainder():
    tok = _new_tokenizer()
    ids, mask = encode_episode_window(tok, "enterprise", [], max_seq_len=128)
    pad_id = tok.special_id("<pad>")
    assert ids.shape == (128,)
    assert mask.shape == (128,)
    # Real tokens at the start, pad afterwards
    real_count = mask.sum().item()
    assert (ids[:real_count] != pad_id).all()
    assert (ids[real_count:] == pad_id).all()


def test_one_turn_renders_obs_act_rew():
    tok = _new_tokenizer()
    turn = Turn(
        obs_text="22/tcp open ssh",
        action_tool_name="nmap_quick_tcp",
        action_render="nmap -sS 10.10.10.5",
        reward=0.1,
    )
    ids, mask = encode_episode_window(tok, "enterprise", [turn], max_seq_len=256)
    id_set = set(int(x) for x in ids.tolist())
    # Turn-boundary specials should all appear.
    for name in ("<obs>", "<act>", "<out>", "<rew>"):
        assert tok.special_id(name) in id_set


def test_overflow_drops_oldest_turn_first():
    """When the window can't hold all turns, oldest go first; newest survives."""
    tok = _new_tokenizer()
    # Tight turns so multiple fit within max_seq_len=512 but not all 5.
    # Each turn is ~ 40-50 untrained-BPE tokens; ~ 250 tokens for 5 turns.
    turns = [
        Turn(
            obs_text=f"obs_data_{i}",
            action_tool_name=f"tool_{i}",
            action_render=f"r_{i}",
            reward=0.01 * i,
        )
        for i in range(50)  # way more than fit
    ]
    ids, mask = encode_episode_window(tok, "enterprise", turns, max_seq_len=128)
    decoded = tok.decode(ids[mask].tolist())
    # The newest turn (#49) must survive; some of the oldest must have been dropped.
    assert "tool_49" in decoded
    assert "tool_0" not in decoded


def test_all_special_token_names_returns_list():
    names = all_special_token_names()
    assert "<bos>" in names
    assert "<matrix:enterprise>" in names


def test_invalid_max_seq_len_raises():
    tok = _new_tokenizer()
    with pytest.raises(ValueError, match="max_seq_len must be positive"):
        encode_episode_window(tok, "enterprise", [], max_seq_len=0)


def test_attn_mask_aligns_with_real_tokens():
    tok = _new_tokenizer()
    turn = Turn(
        obs_text="ports: 22 80 443",
        action_tool_name="nmap_full_tcp",
        action_render="nmap -p- target",
        reward=0.5,
    )
    ids, mask = encode_episode_window(tok, "ics", [turn], max_seq_len=128)
    pad_id = tok.special_id("<pad>")
    # mask True iff token is not pad
    for tid, m in zip(ids.tolist(), mask.tolist()):
        if tid == pad_id:
            assert not m
        else:
            assert m
