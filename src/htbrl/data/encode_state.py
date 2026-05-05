"""Turn-structured state encoding (PLAN.md Phase 2 - State encoder input).

A trajectory is a sequence of turns. Each turn has:
- an observation (tool output, parser-extracted features)
- an action (tool name + slot values)
- a scalar reward

We serialize the most recent K turns as a single token sequence the trunk
reads:

    <bos> <matrix:enterprise>
    <obs> tok... <act> tool_name slot=val ... <out> tok... <rew> r
    <obs> ...

Older turns get summarized (left for Phase 5 - LSTM compression head). For now
we just truncate to the latest K turns and right-pad the result to ``max_seq``.

Why turn boundaries are first-class tokens: the policy needs to attend
selectively to obs vs act vs reward; explicit special tokens make those
boundaries learnable rather than hidden inside punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tokenizer.special import DEFAULT_SPECIALS


@dataclass
class Turn:
    """One step of trajectory in encoder-friendly form."""

    obs_text: str             # parsed/raw stdout from the previous action
    action_tool_name: str     # e.g. "nmap_quick_tcp"
    action_render: str        # the rendered shell command, for context
    reward: float


# Mapping from matrix string to its corresponding special-token name.
_MATRIX_TOKEN = {
    "enterprise": "<matrix:enterprise>",
    "mobile":     "<matrix:mobile>",
    "ics":        "<matrix:ics>",
}


def encode_episode_window(
    tokenizer: ByteLevelBPE,
    matrix: str,
    turns: list[Turn],
    *,
    max_seq_len: int,
    pad_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the most recent ``len(turns)`` turns into a token sequence.

    Returns:
        token_ids: (max_seq_len,) int64
        attn_mask: (max_seq_len,) bool ; True = real token, False = pad

    The output is right-padded with ``<pad>`` tokens. If the rendered turns
    overflow ``max_seq_len``, the OLDEST turns are dropped first (so the most
    recent context is always preserved at the tail of the sequence).
    """
    if matrix not in _MATRIX_TOKEN:
        raise ValueError(f"unknown matrix {matrix!r}; expected one of {list(_MATRIX_TOKEN)}")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    if pad_id is None:
        pad_id = tokenizer.special_id("<pad>")

    bos_id = tokenizer.special_id("<bos>")
    matrix_id = tokenizer.special_id(_MATRIX_TOKEN[matrix])
    obs_id = tokenizer.special_id("<obs>")
    act_id = tokenizer.special_id("<act>")
    out_id = tokenizer.special_id("<out>")
    rew_id = tokenizer.special_id("<rew>")
    sep_id = tokenizer.special_id("<sep>")

    # Build turn-token-lists from newest to oldest, accumulating until we'd
    # overflow. We want the newest turns at the END of the resulting sequence.
    head = [bos_id, matrix_id]
    head_len = len(head)

    rendered_turns: list[list[int]] = []
    used = head_len
    for turn in reversed(turns):  # newest first; we'll reverse back at the end
        action_str = f"{turn.action_tool_name} {turn.action_render}"
        chunk: list[int] = (
            [obs_id]
            + tokenizer.encode(turn.obs_text)
            + [act_id]
            + tokenizer.encode(action_str)
            + [sep_id, out_id]
            + [rew_id]
            + tokenizer.encode(f"{turn.reward:+.4f}")
        )
        # If even adding this one chunk overflows, stop (we drop older turns).
        if used + len(chunk) > max_seq_len:
            break
        rendered_turns.append(chunk)
        used += len(chunk)

    body: list[int] = []
    for chunk in reversed(rendered_turns):  # back to chronological order
        body.extend(chunk)

    seq = head + body
    if len(seq) > max_seq_len:
        # Should not happen given the budgeting above, but be defensive.
        seq = seq[-max_seq_len:]

    pad_len = max_seq_len - len(seq)
    token_ids = torch.tensor(seq + [pad_id] * pad_len, dtype=torch.int64)
    attn_mask = torch.tensor(
        [True] * len(seq) + [False] * pad_len,
        dtype=torch.bool,
    )
    return token_ids, attn_mask


def all_special_token_names() -> list[str]:
    """Convenience accessor for the special-token list (consumed by Hydra configs)."""
    return list(DEFAULT_SPECIALS)
