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
_KALI_PASSWORD = os.environ.get("HTBRL_KALI_PASSWORD")
pytestmark = pytest.mark.ssh

# Kali tests run when host + (key OR password) are set. This way both
# the WSL2 + key-auth path (README's recommended setup) and the
# password-auth dev path (operator's external Kali laptop) light up.
skip_if_no_kali = pytest.mark.skipif(
    not (_KALI_HOST and (_KALI_KEY or _KALI_PASSWORD)),
    reason="HTBRL_KALI_HOST + (HTBRL_KALI_KEY or HTBRL_KALI_PASSWORD) not set",
)


def _creds() -> SSHCredentials:
    user_at_host, _, port_str = _KALI_HOST.partition(":")
    user, _, host = user_at_host.partition("@")
    return SSHCredentials(
        host=host,
        port=int(port_str) if port_str else 22,
        user=user,
        identity_file=os.path.expanduser(_KALI_KEY) if _KALI_KEY else None,
        password=_KALI_PASSWORD,
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
    """``whoami`` must echo back the username we authenticated as.

    The user can vary across setups (``htbrl`` for the README's WSL
    install, but operators with their own Kali laptop may use any
    name), so we check ``whoami`` matches the user portion of
    ``HTBRL_KALI_HOST`` rather than hard-coding a specific name.
    """
    expected_user = _KALI_HOST.split("@", 1)[0]
    with SSHSession(_creds()) as sess:
        r = sess.run("whoami", timeout=10.0)
        assert not r.timed_out
        assert expected_user in r.stdout
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
    """Sanity check: nmap is installed on Kali and runs against the loopback.

    Auto-skips if nmap isn't installed yet. On a fresh Kali laptop:
    ``sudo apt-get install -y nmap``. The README's WSL2 setup snippet
    pre-installs it, but standalone installs may not have it.
    """
    with SSHSession(_creds()) as sess:
        # Probe for nmap before running the real test so the skip
        # message is informative on machines that don't have it.
        probe = sess.run("command -v nmap && echo HAS_NMAP || echo NO_NMAP",
                         timeout=5.0)
        if "HAS_NMAP" not in probe.stdout:
            pytest.skip(
                "nmap not installed on Kali host. Install with: "
                "sudo apt-get install -y nmap"
            )
        r = sess.run("nmap -sn 127.0.0.1 -oN -", timeout=30.0)
        assert not r.timed_out
        assert "127.0.0.1" in r.stdout
        assert r.exit_code == 0
