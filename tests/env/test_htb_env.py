"""Tests for HTBEnv: allowlist enforcement, parser dispatch, real-Kali smoke."""

from __future__ import annotations

import os

import pytest

from htbrl.env.base import Action
from htbrl.env.htb_env import HTBEnv, _extract_ips_from_command, _ip_in_any_cidr
from htbrl.env.ssh_session import SSHCredentials
from htbrl.tools.loader import load_registry

import ipaddress


_KALI_HOST = os.environ.get("HTBRL_KALI_HOST")
_KALI_KEY = os.environ.get("HTBRL_KALI_KEY")

skip_if_no_kali = pytest.mark.skipif(
    not (_KALI_HOST and _KALI_KEY),
    reason="HTBRL_KALI_HOST or HTBRL_KALI_KEY not set",
)


# ---- pure-function helpers ---------------------------------------------------


def test_ip_in_cidr_membership():
    nets = [ipaddress.ip_network("10.10.10.0/24"), ipaddress.ip_network("10.10.11.0/24")]
    assert _ip_in_any_cidr("10.10.10.5", nets)
    assert _ip_in_any_cidr("10.10.11.99", nets)
    assert not _ip_in_any_cidr("8.8.8.8", nets)
    assert not _ip_in_any_cidr("not-an-ip", nets)


def test_extract_ips_from_command_finds_all():
    cmd = "nmap -sV -p 22,80 10.10.10.5; curl http://1.2.3.4/"
    ips = _extract_ips_from_command(cmd)
    assert "10.10.10.5" in ips
    assert "1.2.3.4" in ips


# ---- env construction --------------------------------------------------------


def test_htb_env_requires_allowlist():
    vocab = load_registry()
    creds = SSHCredentials(host="127.0.0.1", user="u", port=2222)
    with pytest.raises(ValueError, match="allowlist_cidrs is required"):
        HTBEnv(vocab=vocab, ssh_creds=creds, allowlist_cidrs=[])


def test_htb_env_blocks_off_allowlist_immediately():
    """A rendered command targeting an outside-allowlist IP must terminate the episode."""
    vocab = load_registry()
    creds = SSHCredentials(host="127.0.0.1", user="u", port=2222)
    env = HTBEnv(vocab=vocab, ssh_creds=creds, allowlist_cidrs=["10.10.10.0/24"])
    env.reset()

    nmap_id = vocab.id_of("nmap_quick_tcp")
    obs, reward, done, info = env.step(Action(tool_id=nmap_id, slots={"ip": "8.8.8.8"}))
    env.close()

    assert "allowlist" in obs.obs_text
    assert info.extras["allowlist_violation"] == "8.8.8.8"
    assert reward < 0
    assert done is True


# ---- real-Kali smoke ---------------------------------------------------------


@skip_if_no_kali
@pytest.mark.ssh
@pytest.mark.slow
def test_htb_env_runs_nmap_against_loopback_via_kali():
    """End-to-end: render nmap, ship through Kali, parse the result.

    Allowlist includes 127.0.0.0/8 so the env permits scanning loopback.
    Actual scan is against 127.0.0.1 so we don't need a separate target VM.
    """
    user_at_host, _, port_str = _KALI_HOST.partition(":")
    user, _, host = user_at_host.partition("@")
    creds = SSHCredentials(
        host=host,
        user=user,
        port=int(port_str) if port_str else 22,
        identity_file=os.path.expanduser(_KALI_KEY),
        connect_timeout_seconds=5.0,
    )
    vocab = load_registry()
    env = HTBEnv(
        vocab=vocab,
        ssh_creds=creds,
        allowlist_cidrs=["127.0.0.0/8"],
        max_steps=2,
    )
    env.reset()
    try:
        # nmap_targeted_ports avoids the long top-1000 scan.
        tool_id = vocab.id_of("nmap_targeted_ports")
        obs, reward, done, info = env.step(
            Action(tool_id=tool_id, slots={"ip": "127.0.0.1", "ports": "22"})
        )
    finally:
        env.close()

    assert info.rendered_command.startswith("nmap")
    assert info.parser_id == "nmap"
    # Shouldn't have time-out unless the box is in serious trouble.
    assert not info.timed_out
    # The parsed features should at least carry the parser shape, even if
    # nothing's listening on 127.0.0.1:22 (which it likely isn't here).
    assert "ports" in obs.parsed_features
    # Reward includes the per-command penalty.
    assert reward <= 0.2
