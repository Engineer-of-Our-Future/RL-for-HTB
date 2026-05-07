"""Tests for wizard-mode helpers (suggest_tools, parse_reward, coerce_slot_value)."""

from __future__ import annotations

import pytest

from htbrl.data.wizard import (
    _REWARD_ALIASES,
    coerce_slot_value,
    parse_reward,
    render_or_fallback,
    suggest_tools,
)
from htbrl.tools.loader import SlotValidationError, load_registry
from htbrl.tools.schema import EnumSlot, FreeStringSlot, IntSlot, IpSlot, PortSlot


# ---- suggest_tools ----------------------------------------------------------


def test_suggest_tools_top_match_for_nmap():
    vocab = load_registry()
    suggestions = suggest_tools(vocab, "nmap -sV 10.10.10.5", n=3)
    assert len(suggestions) == 3
    # Top suggestion should be one of the nmap_* tools (boost: same first token).
    top_score, top_tool = suggestions[0]
    assert top_tool.name.startswith("nmap_")
    assert top_score >= 0.85


def test_suggest_tools_top_match_for_smbclient():
    vocab = load_registry()
    suggestions = suggest_tools(vocab, "smbclient -L //10.10.10.5 -N", n=3)
    top_score, top_tool = suggestions[0]
    assert top_tool.name.startswith("smbclient_")
    assert top_score >= 0.85


def test_suggest_tools_empty_command_returns_empty():
    vocab = load_registry()
    assert suggest_tools(vocab, "") == []
    assert suggest_tools(vocab, "   ") == []


def test_suggest_tools_handles_unmatched_quotes():
    """shlex.split raises on unbalanced quotes; suggest_tools must not crash."""
    vocab = load_registry()
    suggestions = suggest_tools(vocab, "echo \"unclosed", n=3)
    assert isinstance(suggestions, list)


def test_suggest_tools_returns_at_most_n():
    vocab = load_registry()
    assert len(suggest_tools(vocab, "nmap", n=1)) == 1
    assert len(suggest_tools(vocab, "nmap", n=5)) == 5
    assert len(suggest_tools(vocab, "nmap", n=999)) == vocab.n_tools


# ---- parse_reward -----------------------------------------------------------


def test_parse_reward_default_on_empty():
    assert parse_reward("", default=-0.01) == -0.01
    assert parse_reward("   ", default=0.5) == 0.5


def test_parse_reward_numeric():
    assert parse_reward("0.5") == 0.5
    assert parse_reward("+0.5") == 0.5
    assert parse_reward("-0.1") == -0.1
    assert parse_reward("1") == 1.0


def test_parse_reward_aliases():
    assert parse_reward("user_flag") == _REWARD_ALIASES["user_flag"]
    assert parse_reward("root_flag") == _REWARD_ALIASES["root_flag"]
    assert parse_reward("step") == _REWARD_ALIASES["step"]
    assert parse_reward("blocked") == _REWARD_ALIASES["blocked"]


def test_parse_reward_unknown_falls_back_to_default():
    assert parse_reward("not_a_reward", default=99.0) == 99.0


# ---- coerce_slot_value ------------------------------------------------------


def test_coerce_int_slot():
    s = IntSlot(name="x", min=0, max=100)
    assert coerce_slot_value(s, "42") == 42


def test_coerce_int_slot_rejects_garbage():
    s = IntSlot(name="x")
    with pytest.raises(SlotValidationError):
        coerce_slot_value(s, "forty two")


def test_coerce_port_slot():
    s = PortSlot(name="port")
    assert coerce_slot_value(s, "22") == 22


def test_coerce_passthrough_for_string_slots():
    """Free-string / enum / IP slots return their input unchanged - the
    registry's render() does the actual validation."""
    assert coerce_slot_value(IpSlot(name="ip"), "10.10.10.5") == "10.10.10.5"
    assert coerce_slot_value(FreeStringSlot(name="s"), "hello") == "hello"
    assert coerce_slot_value(EnumSlot(name="e", values=["a", "b"]), "a") == "a"


# ---- render_or_fallback -----------------------------------------------------


def test_render_or_fallback_success():
    vocab = load_registry()
    nmap = vocab.get("nmap_quick_tcp")
    rendered, warnings = render_or_fallback(
        vocab, nmap, {"ip": "10.10.10.5"}, fallback="ECHO FALLBACK"
    )
    assert "10.10.10.5" in rendered
    assert "nmap" in rendered
    assert warnings == []


def test_render_or_fallback_returns_fallback_on_validation_error():
    vocab = load_registry()
    nmap = vocab.get("nmap_quick_tcp")
    rendered, warnings = render_or_fallback(
        vocab, nmap, {"ip": "not-an-ip"}, fallback="ECHO FALLBACK"
    )
    assert rendered == "ECHO FALLBACK"
    assert warnings  # got a warning string


def test_render_or_fallback_missing_required_slot():
    vocab = load_registry()
    nmap = vocab.get("nmap_quick_tcp")
    rendered, warnings = render_or_fallback(
        vocab, nmap, {}, fallback="raw cmd"
    )
    assert rendered == "raw cmd"
    assert warnings
