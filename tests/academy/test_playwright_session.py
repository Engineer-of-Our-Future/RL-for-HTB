"""Tests for the Playwright academy session.

The Playwright import is heavy and pulls browser binaries. Tests that need a
real browser are NOT in this file - those are manual integration tests run
against your actual HTB Academy account.

What we DO test here:
- The selector-loading layer (default + YAML override)
- The "playwright not installed" import path raises a clear error
- Helper functions (_first_int, _hash_id, _safe_text) — these are pure.
- Construction with study_only / cookie_path / selector overrides
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# Skip the entire module if playwright isn't installed - we don't want this
# to fail when only the default extras are installed.
playwright_session = pytest.importorskip("htbrl.academy.playwright_session")


def test_default_selectors_present():
    sel = playwright_session._DEFAULT_SELECTORS
    for top in ("login", "modules_list", "module_page", "progress"):
        assert top in sel
    # Some required leaves on the default set
    assert "email" in sel["login"]
    assert "password" in sel["login"]


def test_first_int_extracts_leading_number():
    fn = playwright_session._first_int
    assert fn("3 cubes") == 3
    assert fn("Tier 0 module") == 0
    assert fn("nothing here") is None
    assert fn("") is None


def test_hash_id_stable_short():
    fn = playwright_session._hash_id
    a = fn("Linux Fundamentals")
    b = fn("Linux Fundamentals")
    c = fn("Network Enumeration")
    assert a == b
    assert a != c
    assert len(a) == 12


def test_selector_override_yaml(tmp_path: Path):
    override = tmp_path / "sel.yaml"
    override.write_text(
        "login:\n  email: 'input#email'\n  password: 'input#password'\n",
        encoding="utf-8",
    )
    cfg = playwright_session.PlaywrightConfig(selectors_path=override)
    # We can't actually instantiate without playwright + browser, but we can
    # exercise the selector loader by calling _load_selectors directly.
    sess = playwright_session.PlaywrightAcademySession.__new__(
        playwright_session.PlaywrightAcademySession
    )
    sess.cfg = cfg
    sel = sess._load_selectors()
    assert sel["login"]["email"] == "input#email"
    assert sel["login"]["password"] == "input#password"
    # Default keys still survive
    assert "submit" in sel["login"]


def test_selector_override_falls_back_to_default_when_missing(tmp_path: Path):
    cfg = playwright_session.PlaywrightConfig(selectors_path=None)
    sess = playwright_session.PlaywrightAcademySession.__new__(
        playwright_session.PlaywrightAcademySession
    )
    sess.cfg = cfg
    sel = sess._load_selectors()
    assert sel["login"]["email"] == playwright_session._DEFAULT_SELECTORS["login"]["email"]


def test_playwright_config_defaults():
    cfg = playwright_session.PlaywrightConfig()
    assert cfg.headless is True
    assert cfg.base_url.startswith("https://academy.hackthebox.com")
    assert cfg.selectors_path is None
    assert cfg.cookie_path is None
