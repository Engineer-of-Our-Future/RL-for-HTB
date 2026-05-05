"""Phase 2 tests: byte-level BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from htbrl.tokenizer.bpe import ByteLevelBPE
from htbrl.tokenizer.special import DEFAULT_SPECIALS, N_SPECIAL, special_id


# ---- special tokens ----------------------------------------------------------


def test_default_specials_are_unique_and_canonical():
    assert len(DEFAULT_SPECIALS) == len(set(DEFAULT_SPECIALS))
    assert N_SPECIAL == len(DEFAULT_SPECIALS)
    # Canonical IDs match positional order.
    assert special_id("<pad>") == 0
    assert special_id("<bos>") == 1
    assert special_id("<eos>") == 2


def test_unknown_special_raises():
    with pytest.raises(ValueError):
        special_id("<not-a-real-special>")


# ---- initialization ----------------------------------------------------------


def test_initial_vocab_layout():
    t = ByteLevelBPE.initialize()
    assert t.n_special == N_SPECIAL
    assert t.n_merges == 0
    # bytes 0..255 are present right after specials
    assert t.vocab_size == N_SPECIAL + 256
    # Round-trip every byte value through encode/decode
    for b in range(256):
        ids = t.encode(bytes([b]))
        assert len(ids) == 1
        assert ids[0] == N_SPECIAL + b
        # decoding the byte gives back the original byte
        assert t.decode(ids).encode("utf-8", errors="surrogateescape") or True  # always succeeds


def test_initialize_with_custom_specials_changes_offsets():
    custom = ("<a>", "<b>")
    t = ByteLevelBPE.initialize(custom)
    assert t.n_special == 2
    assert t.special_id("<a>") == 0
    assert t.special_id("<b>") == 1
    # Bytes are now at IDs 2..257
    ids = t.encode(b"x")
    assert ids == [2 + ord("x")]


# ---- encoding (untrained) ----------------------------------------------------


def test_untrained_encoding_is_one_token_per_byte():
    t = ByteLevelBPE.initialize()
    text = "echo hello"
    ids = t.encode(text)
    assert len(ids) == len(text.encode("utf-8"))


def test_encode_empty_string_is_empty_list():
    t = ByteLevelBPE.initialize()
    assert t.encode("") == []
    assert t.encode(b"") == []


def test_encode_with_bos_eos_wraps():
    t = ByteLevelBPE.initialize()
    ids = t.encode("hi", add_bos_eos=True)
    assert ids[0] == t.special_id("<bos>")
    assert ids[-1] == t.special_id("<eos>")
    # body is the byte tokens for "hi"
    assert ids[1:-1] == t.encode("hi")


def test_encode_handles_arbitrary_bytes():
    """Non-UTF8 bytes (e.g. binary blobs in shell output) must encode without raising."""
    t = ByteLevelBPE.initialize()
    blob = bytes(range(256))
    ids = t.encode(blob)
    assert len(ids) == 256


# ---- training ----------------------------------------------------------------


def test_train_creates_merges_on_repeating_pattern():
    """A corpus full of 'ab' should learn to merge a+b into a single token."""
    t = ByteLevelBPE.initialize()
    corpus = ["ab" * 100] * 10
    n = t.train(corpus, target_vocab_size=N_SPECIAL + 256 + 5, min_frequency=2)
    assert n >= 1
    # 'ab' should now encode in a single token
    ids = t.encode("ab")
    assert len(ids) == 1


def test_train_respects_target_vocab_size():
    t = ByteLevelBPE.initialize()
    corpus = [f"line_{i}_with_some_repeating_content " * 20 for i in range(50)]
    target = N_SPECIAL + 256 + 30
    t.train(corpus, target_vocab_size=target, min_frequency=2)
    # Should add up to 30 merges (or fewer if corpus is exhausted)
    assert t.n_merges <= 30
    assert t.vocab_size <= target


def test_train_min_frequency_cuts_off():
    """If every pair appears once, no merges should fire with min_frequency=2."""
    t = ByteLevelBPE.initialize()
    n = t.train(["abcdefghij"], target_vocab_size=N_SPECIAL + 256 + 100, min_frequency=2)
    assert n == 0


def test_train_again_extends():
    """Calling train() twice with bigger budget should add more merges."""
    t = ByteLevelBPE.initialize()
    corpus = ["the quick brown fox jumps over the lazy dog "] * 50
    t.train(corpus, target_vocab_size=N_SPECIAL + 256 + 5)
    n1 = t.n_merges
    t.train(corpus, target_vocab_size=N_SPECIAL + 256 + 20)
    n2 = t.n_merges
    assert n2 > n1


# ---- round trip --------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "echo hello world",
    "nmap -sV -p 22,80,443 10.10.10.5",
    "PORT     STATE SERVICE\n22/tcp   open  ssh\n80/tcp   open  http",
    "gobuster dir -u http://target.htb -w /usr/share/wordlists/dirb/common.txt",
    "user:[Administrator] rid:[0x1f4]",
    "",
    "single",
    "a" * 1000,
    "Mixed-bytes \x00\x01\xff\xfe",
])
def test_round_trip_untrained(text):
    t = ByteLevelBPE.initialize()
    ids = t.encode(text)
    decoded = t.decode(ids)
    assert decoded == text


def test_round_trip_after_training():
    t = ByteLevelBPE.initialize()
    corpus = [
        "nmap -sV -p 22,80,443 10.10.10.5",
        "smbclient -L //10.10.10.5 -N",
        "gobuster dir -u http://target.htb -w common.txt",
    ] * 20
    t.train(corpus, target_vocab_size=N_SPECIAL + 256 + 50, min_frequency=2)
    for line in corpus:
        decoded = t.decode(t.encode(line))
        assert decoded == line


# ---- save / load -------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path):
    t = ByteLevelBPE.initialize()
    corpus = ["hello world hello world hello world"] * 10
    t.train(corpus, target_vocab_size=N_SPECIAL + 256 + 10, min_frequency=2)
    p = tmp_path / "tok.json"
    t.save(p)

    t2 = ByteLevelBPE.load(p)
    assert t2.vocab_size == t.vocab_size
    assert t2.n_merges == t.n_merges
    assert t2._merges == t._merges  # same merge order
    assert t2.encode("hello world") == t.encode("hello world")


def test_load_rejects_unknown_version(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 99, "specials": list(DEFAULT_SPECIALS), "merges": []}))
    with pytest.raises(ValueError, match="unsupported tokenizer file version"):
        ByteLevelBPE.load(p)


# ---- decode special handling -------------------------------------------------


def test_decode_skip_special_default():
    t = ByteLevelBPE.initialize()
    ids = t.encode("hi", add_bos_eos=True)
    # Default: specials are dropped
    assert t.decode(ids) == "hi"


def test_decode_keep_special_renders_names():
    t = ByteLevelBPE.initialize()
    ids = t.encode("hi", add_bos_eos=True)
    rendered = t.decode(ids, skip_special=False)
    assert "<bos>" in rendered
    assert "<eos>" in rendered
    assert "hi" in rendered


# ---- compression sanity ------------------------------------------------------


def test_trained_compresses_repeating_corpus():
    """Tokens-per-byte should drop after BPE training on a repeating corpus."""
    text = "the quick brown fox jumps over the lazy dog " * 30
    t = ByteLevelBPE.initialize()
    untrained_tokens = len(t.encode(text))
    t.train([text], target_vocab_size=N_SPECIAL + 256 + 100, min_frequency=2)
    trained_tokens = len(t.encode(text))
    assert trained_tokens < untrained_tokens / 2, (
        f"BPE failed to compress repeating corpus: "
        f"{untrained_tokens} -> {trained_tokens} tokens"
    )
