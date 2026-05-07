"""Tests for the persistent SSH session.

These tests need a reachable Kali (or any Linux SSH server) configured via
HTBRL_KALI_HOST + HTBRL_KALI_KEY env vars. They auto-skip otherwise so the
suite stays runnable on machines without Kali.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from htbrl.env.ssh_session import SSHCredentials, SSHSession


_KALI_HOST = os.environ.get("HTBRL_KALI_HOST")
_KALI_KEY = os.environ.get("HTBRL_KALI_KEY")
pytestmark = pytest.mark.ssh

skip_if_no_kali = pytest.mark.skipif(
    not (_KALI_HOST and _KALI_KEY),
    reason="HTBRL_KALI_HOST or HTBRL_KALI_KEY not set",
)


def _creds() -> SSHCredentials:
    user_at_host, _, port_str = _KALI_HOST.partition(":")
    user, _, host = user_at_host.partition("@")
    return SSHCredentials(
        host=host,
        port=int(port_str) if port_str else 22,
        user=user,
        identity_file=os.path.expanduser(_KALI_KEY),
        connect_timeout_seconds=5.0,
    )


@skip_if_no_kali
def test_ssh_open_close_idempotent():
    sess = SSHSession(_creds())
    assert not sess.is_open
    sess.open()
    assert sess.is_open
    sess.open()  # idempotent
    sess.close()
    assert not sess.is_open
    sess.close()  # idempotent


@skip_if_no_kali
def test_ssh_run_simple_command():
    with SSHSession(_creds()) as sess:
        r = sess.run("whoami", timeout=10.0)
        assert not r.timed_out
        assert "htbrl" in r.stdout
        assert r.exit_code == 0


@skip_if_no_kali
def test_ssh_persistent_state_across_runs():
    """cd in one run; pwd in the next must reflect the new directory."""
    with SSHSession(_creds()) as sess:
        sess.run("cd /tmp", timeout=5.0)
        r = sess.run("pwd", timeout=5.0)
        assert "/tmp" in r.stdout


@skip_if_no_kali
def test_ssh_timeout_on_long_running_command():
    with SSHSession(_creds()) as sess:
        t0 = time.time()
        r = sess.run("sleep 5", timeout=1.0)
        elapsed = time.time() - t0
        assert r.timed_out
        # Should have given up around the timeout, not waited the full 5s.
        assert elapsed < 3.0


@skip_if_no_kali
def test_ssh_exit_code_captured_for_failing_command():
    with SSHSession(_creds()) as sess:
        r = sess.run("false", timeout=5.0)
        assert r.exit_code == 1


@skip_if_no_kali
def test_ssh_runs_pentest_tool():
    """Sanity check: nmap is installed and runs against the loopback."""
    with SSHSession(_creds()) as sess:
        r = sess.run("nmap -sn 127.0.0.1 -oN -", timeout=30.0)
        assert not r.timed_out
        assert "127.0.0.1" in r.stdout
        assert r.exit_code == 0
