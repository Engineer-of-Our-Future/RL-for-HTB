"""Tests for the per-tool output parsers."""

from __future__ import annotations

import pytest

from htbrl.env.parsers import PARSERS, parse
from htbrl.env.parsers.nmap import parse_nmap
from htbrl.env.parsers.simple import parse_lines, parse_raw
from htbrl.env.parsers.smb import parse_smbclient


def test_raw_parser_passthrough():
    out = parse_raw("hello\nworld\n")
    assert out["text"] == "hello\nworld\n"
    assert out["parse_errors"] == []


def test_lines_parser_strips_blank_and_trailing():
    out = parse_lines("a\n\n  b  \n\nc\n")
    assert out["lines"] == ["a", "  b", "c"]


def test_nmap_extracts_open_ports():
    output = """\
Starting Nmap 7.93 ( https://nmap.org )
Nmap scan report for 10.10.10.5
Host is up (0.020s latency).
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 8.4p1 Debian 5+deb11u1
80/tcp   open  http    Apache httpd 2.4.51
443/tcp  closed https
8080/tcp open  http-proxy
"""
    parsed = parse_nmap(output)
    assert parsed["host_up"] is True
    assert len(parsed["ports"]) == 3  # closed dropped
    p22 = next(p for p in parsed["ports"] if p["port"] == 22)
    assert p22["service"] == "ssh"
    assert p22["version"] == "OpenSSH 8.4p1 Debian 5+deb11u1"


def test_nmap_no_open_ports_returns_empty_list():
    parsed = parse_nmap("Host seems down (no response)")
    assert parsed["host_up"] is False
    assert parsed["ports"] == []


def test_nmap_robust_to_garbage():
    parsed = parse_nmap("the cat sat on the mat\nrandom log line")
    assert parsed["ports"] == []
    assert parsed["host_up"] is None
    assert parsed["parse_errors"] == []


def test_smbclient_extracts_shares():
    output = """\
        Sharename       Type      Comment
        ---------       ----      -------
        ADMIN$          Disk      Remote Admin
        IPC$            IPC       Remote IPC
        public          Disk      Public files

SMB1 disabled
"""
    parsed = parse_smbclient(output)
    names = {s["name"] for s in parsed["shares"]}
    assert names == {"ADMIN$", "IPC$", "public"}


def test_smbclient_empty():
    parsed = parse_smbclient("")
    assert parsed["shares"] == []


# ---- dispatcher --------------------------------------------------------------


def test_parse_dispatches_by_id():
    out = parse("nmap", "Host is up\n22/tcp open ssh OpenSSH")
    assert "ports" in out


def test_parse_unknown_id_falls_back_to_raw():
    out = parse("does-not-exist", "hello")
    assert out["text"] == "hello"


def test_parse_swallows_parser_exceptions():
    """If a parser is buggy, dispatch should still return a dict."""
    bad = "not really nmap"
    out = parse("nmap", bad)
    # Just shouldn't raise, regardless of content.
    assert isinstance(out, dict)


def test_parsers_registry_has_expected_ids():
    for tid in ("raw", "lines", "nmap", "smbclient"):
        assert tid in PARSERS
