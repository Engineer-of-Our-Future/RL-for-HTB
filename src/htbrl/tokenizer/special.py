"""Special tokens used by the state encoder.

These tokens are reserved at the bottom of the BPE vocab (IDs 0..N-1) so they
never collide with byte tokens or learned merges. The state encoder in
`src/htbrl/data/encode_state.py` uses them to delimit observations / actions /
rewards / matrix selectors / typed payloads in the rolling-window sequence
that the transformer reads.

Adding a token is forward-compatible (just append to DEFAULT_SPECIALS); changing
or removing one breaks any saved tokenizer artifact.
"""

from __future__ import annotations

# Order is the source of truth: the index in this list IS the token ID.
# Never reorder or remove entries; only append new ones to the end.
DEFAULT_SPECIALS: tuple[str, ...] = (
    "<pad>",       # 0  - padding to fixed sequence length
    "<bos>",       # 1  - beginning of an episode
    "<eos>",       # 2  - end of an episode
    "<obs>",       # 3  - start of an observation chunk
    "<act>",       # 4  - start of an action serialization
    "<rew>",       # 5  - reward scalar that follows
    "<cmd>",       # 6  - rendered shell command
    "<out>",       # 7  - tool output (stdout/stderr)
    "<prompt>",    # 8  - shell prompt seen at the end of output
    "<ip>",        # 9  - placeholder for an IP address (so the model isn't
                   #      forced to memorize them as character sequences)
    "<port>",      # 10 - placeholder for a port number
    "<hash>",      # 11 - placeholder for a hash / hex blob
    "<sep>",       # 12 - generic separator inside structured payloads
    "<matrix:enterprise>",  # 13 - episode matrix selector (Enterprise)
    "<matrix:mobile>",      # 14 - episode matrix selector (Mobile)
    "<matrix:ics>",         # 15 - episode matrix selector (ICS)
    "<tactic>",    # 16 - prefix for tactic ID emitted by the env tracker
    "<technique>", # 17 - prefix for technique ID emitted by the env tracker
)

N_SPECIAL = len(DEFAULT_SPECIALS)


def special_id(name: str) -> int:
    """Return the canonical ID for a special token name. Raises KeyError if unknown."""
    return DEFAULT_SPECIALS.index(name)
