"""Tests for the subscription-tier filter (htbrl.env.subscription).

The filter never reaches the network in normal mode (free / vip).
Only ``"auto"`` does, and we mock the urlopen call so the test
doesn't actually hit HTB's API.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from htbrl.env.subscription import (
    SubscriptionInfo,
    filter_boxes_by_tier,
    is_box_free,
    resolve,
)


# ---- explicit tiers --------------------------------------------------------


def test_resolve_explicit_free_returns_free():
    info = resolve("free")
    assert info.tier == "free"
    assert info.has_vip is False
    assert info.source == "explicit:free"


def test_resolve_explicit_vip_returns_vip():
    info = resolve("vip")
    assert info.tier == "vip"
    assert info.has_vip is True
    assert info.source == "explicit:vip"


def test_resolve_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown subscription tier"):
        resolve("ultra-platinum")  # type: ignore[arg-type]


# ---- auto with no token ----------------------------------------------------


def test_auto_with_no_token_falls_back_to_free(monkeypatch):
    """Without a token we can't probe; safer default is 'free'."""
    monkeypatch.delenv("HTB_API_TOKEN", raising=False)
    info = resolve("auto")
    assert info.tier == "free"
    assert "no-token" in info.source


def test_auto_picks_up_token_from_env(monkeypatch):
    """An HTB_API_TOKEN env var is enough to enable the probe."""
    monkeypatch.setenv("HTB_API_TOKEN", "test-token-123")
    fake_payload = b'{"info": {"subscription": "vip-plus"}}'
    fake_resp = _FakeResponse(fake_payload)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        info = resolve("auto")
    assert info.tier == "vip"
    assert "vip-plus" in info.source


def test_auto_classifies_basic_subscription_as_free(monkeypatch):
    monkeypatch.setenv("HTB_API_TOKEN", "tok")
    fake_resp = _FakeResponse(b'{"info": {"subscription": "basic"}}')
    with patch("urllib.request.urlopen", return_value=fake_resp):
        info = resolve("auto")
    assert info.tier == "free"
    assert "basic" in info.source


def test_auto_classifies_null_subscription_as_free(monkeypatch):
    monkeypatch.setenv("HTB_API_TOKEN", "tok")
    fake_resp = _FakeResponse(b'{"info": {"subscription": null}}')
    with patch("urllib.request.urlopen", return_value=fake_resp):
        info = resolve("auto")
    assert info.tier == "free"


def test_auto_falls_back_to_free_on_api_failure(monkeypatch):
    """Network glitch / 401 / schema drift -> default to 'free'."""
    monkeypatch.setenv("HTB_API_TOKEN", "tok")
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        info = resolve("auto")
    assert info.tier == "free"
    assert "probe-failed" in info.source


def test_auto_explicit_token_takes_precedence_over_env(monkeypatch):
    """``api_token=`` arg beats the HTB_API_TOKEN env var."""
    monkeypatch.setenv("HTB_API_TOKEN", "env-token")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["auth"] = req.headers.get("Authorization")
        return _FakeResponse(b'{"info": {"subscription": "vip"}}')

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        info = resolve("auto", api_token="explicit-token")
    assert info.tier == "vip"
    assert captured["auth"] == "Bearer explicit-token"


# ---- per-box helpers -------------------------------------------------------


def test_is_box_free_default_when_field_missing():
    """Boxes without ``vip_only`` are treated as free (back-compat)."""
    assert is_box_free({"id": "anything"}) is True


def test_is_box_free_respects_vip_only_flag():
    assert is_box_free({"id": "x", "vip_only": True}) is False
    assert is_box_free({"id": "x", "vip_only": False}) is True


def test_is_box_free_starting_point_overrides_vip_only_flag():
    """A Starting Point box mistakenly tagged ``vip_only`` is still
    free in practice; the helper protects against config drift."""
    assert is_box_free({
        "id": "htb-starting-point:meow", "vip_only": True,
    }) is True


# ---- pool filtering --------------------------------------------------------


def _make_pool() -> list[dict]:
    return [
        {"id": "htb-starting-point:meow", "vip_only": False},
        {"id": "htb-machines:active-easy-1", "vip_only": False},
        {"id": "htb-machines:retired-medium-7", "vip_only": True},
        {"id": "htb-machines:retired-hard-12", "vip_only": True},
    ]


def test_filter_boxes_vip_returns_everything():
    info = SubscriptionInfo(tier="vip")
    out = filter_boxes_by_tier(_make_pool(), info)
    assert len(out) == 4


def test_filter_boxes_free_drops_vip_only():
    info = SubscriptionInfo(tier="free")
    out = filter_boxes_by_tier(_make_pool(), info)
    assert [b["id"] for b in out] == [
        "htb-starting-point:meow",
        "htb-machines:active-easy-1",
    ]


def test_filter_boxes_preserves_ordering():
    """Round-robin selection over the filtered list must match the
    YAML's stated order, not be re-sorted."""
    info = SubscriptionInfo(tier="free")
    pool = [
        {"id": "z-free", "vip_only": False},
        {"id": "a-vip", "vip_only": True},
        {"id": "m-free", "vip_only": False},
    ]
    out = filter_boxes_by_tier(pool, info)
    assert [b["id"] for b in out] == ["z-free", "m-free"]


def test_filter_boxes_empty_pool_is_empty_pool():
    info = SubscriptionInfo(tier="free")
    assert filter_boxes_by_tier([], info) == []


# ---- helper ---------------------------------------------------------------


class _FakeResponse:
    """Minimal urllib.urlopen response stand-in."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False
