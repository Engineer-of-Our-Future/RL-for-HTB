"""Byte-level BPE tokenizer, trained from scratch.

Vocab layout (indices are stable across saves/loads):
- ``0 .. N_SPECIAL - 1``       : special tokens (from ``special.py``)
- ``N_SPECIAL .. N_SPECIAL+255`` : the 256 raw byte values
- ``N_SPECIAL+256 .. vocab_size-1`` : learned BPE merges, in the order they were
  added (so the BPE rank used at encode time is stable).

Why byte-level: shell output frequently contains non-UTF8 bytes (binary blobs,
ANSI escapes, raw memory dumps). Character-level BPE breaks on those. Byte-level
guarantees a round-trip on *any* byte string.

Why from scratch: project rule says "no LLMs / no pretrained models / no
HuggingFace tokenizers". This module is the only tokenizer the project uses;
it loads zero external state at startup.

Implementation choices:
- Training is greedy max-frequency pair merging (Sennrich-Haddow-Birch 2016),
  with min-frequency cutoff and an explicit vocab-size budget.
- Encoding uses a priority-queue-free implementation: scan the token list,
  find the lowest-rank mergeable pair, merge in place, repeat. O(N * V_merges)
  worst case but simple and correct, which matters way more than speed for the
  one-time encoding of demo trajectories. We can swap in a faster algorithm
  later if tokenization shows up in profiles.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .special import DEFAULT_SPECIALS, N_SPECIAL


class ByteLevelBPE:
    """Byte-level BPE.  Construct via ``initialize()``, then ``train()`` or ``load()``."""

    def __init__(self) -> None:
        # token_id -> bytes content for byte / merged tokens
        self._content: dict[int, bytes] = {}
        # id -> name for special tokens
        self._special_names: dict[int, str] = {}
        # name -> id for special tokens
        self._special_by_name: dict[str, int] = {}
        # 1-byte content -> id for fast initial tokenization
        self._byte_to_id: dict[bytes, int] = {}
        # ordered list of (a, b) merges; index in list IS the rank
        self._merges: list[tuple[int, int]] = []
        # (a, b) -> rank, for fast lookup at encode time
        self._merge_rank: dict[tuple[int, int], int] = {}
        # (a, b) -> new_id produced when merging
        self._merge_target: dict[tuple[int, int], int] = {}

    # ----- construction -------------------------------------------------------

    @classmethod
    def initialize(cls, specials: tuple[str, ...] | None = None) -> "ByteLevelBPE":
        """Empty tokenizer: bytes 0..255 + special tokens, no merges yet."""
        t = cls()
        spec = specials if specials is not None else DEFAULT_SPECIALS
        for i, name in enumerate(spec):
            t._special_names[i] = name
            t._special_by_name[name] = i
        for b in range(256):
            tid = len(spec) + b
            content = bytes([b])
            t._content[tid] = content
            t._byte_to_id[content] = tid
        return t

    # ----- properties ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._special_names) + len(self._content)

    @property
    def n_special(self) -> int:
        return len(self._special_names)

    @property
    def n_merges(self) -> int:
        return len(self._merges)

    def special_id(self, name: str) -> int:
        return self._special_by_name[name]

    def is_special(self, token_id: int) -> bool:
        return token_id in self._special_names

    # ----- training -----------------------------------------------------------

    def train(
        self,
        corpus: Iterable[str | bytes],
        target_vocab_size: int = 32768,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> int:
        """Train BPE merges until ``target_vocab_size`` is reached.

        Returns the number of merges actually added (may be less than the budget
        if the corpus runs out of frequency-≥-min_frequency pairs).
        """
        if target_vocab_size <= self.vocab_size:
            return 0

        # Initialize encoded corpus as byte token IDs
        encoded: list[list[int]] = []
        for entry in corpus:
            if isinstance(entry, str):
                entry = entry.encode("utf-8")
            encoded.append([self._byte_to_id[bytes([b])] for b in entry])

        merges_added = 0
        while self.vocab_size < target_vocab_size:
            pair_counts: Counter = Counter()
            for tokens in encoded:
                for i in range(len(tokens) - 1):
                    pair_counts[(tokens[i], tokens[i + 1])] += 1

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            new_id = self.vocab_size
            self._content[new_id] = self._content[best_pair[0]] + self._content[best_pair[1]]
            self._merges.append(best_pair)
            self._merge_rank[best_pair] = len(self._merges) - 1
            self._merge_target[best_pair] = new_id
            merges_added += 1

            # Apply merge to the corpus in-place, single left-to-right pass.
            for idx, tokens in enumerate(encoded):
                new_tokens: list[int] = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i + 1 < n and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                encoded[idx] = new_tokens

            if verbose and merges_added % 500 == 0:
                print(
                    f"  merge #{merges_added}: "
                    f"({best_pair}) -> {new_id}  "
                    f"vocab_size={self.vocab_size}  "
                    f"freq={best_count}"
                )

        return merges_added

    # ----- encoding -----------------------------------------------------------

    def encode(self, text: str | bytes, add_bos_eos: bool = False) -> list[int]:
        """Encode ``text`` to token IDs.

        Special tokens given as substrings of ``text`` are treated as raw bytes
        (no special-token recognition during encoding). To insert special tokens,
        call ``special_id("<bos>")`` and splice the result yourself - this avoids
        accidental splits when shell output happens to contain a string like
        ``<obs>`` for some unrelated reason.
        """
        if isinstance(text, str):
            text = text.encode("utf-8")
        if not text:
            return [self.special_id("<bos>"), self.special_id("<eos>")] if add_bos_eos else []

        tokens: list[int] = [self._byte_to_id[bytes([b])] for b in text]
        # Greedy lowest-rank-first merging. We re-scan after each merge because
        # merging can create new mergeable pairs at the boundary.
        while True:
            best_rank = len(self._merges)  # sentinel
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            new_id = self._merge_target[(tokens[best_idx], tokens[best_idx + 1])]
            tokens = tokens[:best_idx] + [new_id] + tokens[best_idx + 2 :]

        if add_bos_eos:
            return [self.special_id("<bos>"), *tokens, self.special_id("<eos>")]
        return tokens

    # ----- decoding -----------------------------------------------------------

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        for tid in ids:
            if tid in self._special_names:
                if skip_special:
                    continue
                out.extend(self._special_names[tid].encode("utf-8"))
            else:
                out.extend(self._content[tid])
        return out.decode("utf-8", errors="replace")

    # ----- persistence --------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist tokenizer state as a single JSON file (bytes hex-encoded).

        Format is intentionally simple and version-tagged so the tokenizer is
        portable and reviewable as text.
        """
        path = Path(path)
        payload = {
            "version": 1,
            "specials": [self._special_names[i] for i in sorted(self._special_names)],
            "merges": self._merges,  # list of (a, b) tuples become JSON arrays
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ByteLevelBPE":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported tokenizer file version: {payload.get('version')}")

        t = cls.initialize(tuple(payload["specials"]))
        for pair in payload["merges"]:
            a, b = int(pair[0]), int(pair[1])
            new_id = t.vocab_size
            t._content[new_id] = t._content[a] + t._content[b]
            t._merges.append((a, b))
            t._merge_rank[(a, b)] = len(t._merges) - 1
            t._merge_target[(a, b)] = new_id
        return t
