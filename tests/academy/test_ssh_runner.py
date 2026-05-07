"""Unit tests for the academy SSH probe pipeline.

We don't have a live academy box at test time, so the runner's
:meth:`run` is exercised indirectly by mocking ``paramiko.SSHClient``.
The credential parser is exercised against the literal prompt
shapes seen in modules 18 / 23 / 33 / 49 to lock the regex against
academy-side wording drift.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from htbrl.academy.ssh_runner import (
    SshPromptCreds,
    SshResult,
    SshTargetRunner,
    parse_ssh_credentials,
)


# ---- credential parser ------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # Full form with host inline (Module 18, sec 'Find Files').
        (
            'SSH to 10.129.100.38 (ACADEMY-NIXFUND), with user "htb-student"'
            ' and password "HTB_@cademy_stdnt!"',
            SshPromptCreds(
                host="10.129.100.38",
                user="htb-student",
                password="HTB_@cademy_stdnt!",
            ),
        ),
        # Host omitted, creds only (Module 23 'File Inclusion Prevention').
        (
            'Authenticate to with user "root" and password "password"',
            SshPromptCreds(host=None, user="root", password="password"),
        ),
        # Host inline, no parenthetical name.
        (
            'SSH to 10.129.100.38 with user "htb" and password "x!Y2"',
            SshPromptCreds(host="10.129.100.38", user="htb", password="x!Y2"),
        ),
        # Different ordering: "user 'X' password 'Y'".
        (
            "ssh user 'admin' password 'p4ss'",
            SshPromptCreds(host=None, user="admin", password="p4ss"),
        ),
    ],
)
def test_parse_ssh_credentials_known_shapes(prompt, expected):
    got = parse_ssh_credentials(prompt)
    assert got == expected
    assert got.complete


def test_parse_ssh_credentials_no_match():
    """Prompt with no SSH directive returns an empty (incomplete) creds."""
    got = parse_ssh_credentials("Just a normal question with no creds.")
    assert got.host is None
    assert got.user == ""
    assert got.password == ""
    assert not got.complete


def test_parse_ssh_credentials_handles_special_chars_in_password():
    """Academy passwords often include ! @ # $ & — must survive the parser."""
    got = parse_ssh_credentials(
        'SSH to 1.2.3.4 with user "u" and password "p!@#$&*-_+="'
    )
    assert got.password == "p!@#$&*-_+="


# ---- runner: success path with mocked paramiko -----------------------------


class _FakeChannel:
    def __init__(self, exit_status: int):
        self._rc = exit_status

    def recv_exit_status(self) -> int:
        return self._rc


class _FakeStream:
    def __init__(self, data: bytes, channel=None):
        self._data = data
        self.channel = channel

    def read(self, _max):
        return self._data


@pytest.fixture
def fake_paramiko(monkeypatch):
    """Patch paramiko so the runner doesn't need a real SSH server."""

    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    chan = _FakeChannel(0)
    out = _FakeStream(b"hello\n", channel=chan)
    err = _FakeStream(b"")
    fake_client.exec_command.return_value = (object(), out, err)
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()

    # ssh_runner imports paramiko lazily inside _connect, so patch the
    # module entry that import will resolve.
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)
    return fake_paramiko_mod, fake_client, chan


def test_run_returns_stdout_and_zero_rc(fake_paramiko):
    runner = SshTargetRunner(host="1.2.3.4", username="u", password="p")
    result = runner.run("echo hello")
    assert isinstance(result, SshResult)
    assert result.ok
    assert result.rc == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.error == ""


def test_run_caps_output_at_max_bytes(monkeypatch):
    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    big = b"A" * 50_000
    out = _FakeStream(big, channel=_FakeChannel(0))
    err = _FakeStream(b"")
    fake_client.exec_command.return_value = (object(), out, err)
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)

    runner = SshTargetRunner(
        host="1.2.3.4", username="u", password="p", max_bytes=1024,
    )
    result = runner.run("yes A | head -c 50000")
    assert result.truncated is True
    # We capped to max_bytes (the +1 byte we read for overflow detection
    # is sliced off before decoding).
    assert len(result.stdout) == 1024


def test_run_returns_error_on_connect_failure(monkeypatch):
    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    fake_client.connect.side_effect = OSError("Connection refused")
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)

    runner = SshTargetRunner(host="1.2.3.4", username="u", password="p")
    result = runner.run("whoami")
    assert not result.ok
    assert result.rc == -1
    assert "OSError" in result.error
    assert "Connection refused" in result.error


def test_run_propagates_nonzero_exit(monkeypatch):
    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    out = _FakeStream(b"", channel=_FakeChannel(127))
    err = _FakeStream(b"sh: bogus: not found\n")
    fake_client.exec_command.return_value = (object(), out, err)
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)

    runner = SshTargetRunner(host="1.2.3.4", username="u", password="p")
    result = runner.run("bogus")
    assert not result.ok                # rc != 0 even though no transport error
    assert result.rc == 127
    assert "not found" in result.stderr
    assert result.error == ""           # transport itself was fine


# ---- runner: batch run reuses a single transport ---------------------------


def test_run_many_reuses_transport(monkeypatch):
    """run_many opens ONE SSHClient, runs N commands, closes it."""

    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    # Each exec_command call returns a fresh (stdin, stdout, stderr).
    pairs = [
        (b"a\n", b""), (b"b\n", b""), (b"c\n", b""),
    ]
    call_iter = iter(pairs)

    def _exec(cmd, timeout=None):
        out, err = next(call_iter)
        return (object(), _FakeStream(out, channel=_FakeChannel(0)),
                _FakeStream(err))

    fake_client.exec_command.side_effect = _exec
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)

    runner = SshTargetRunner(host="1.2.3.4", username="u", password="p")
    results = runner.run_many(["echo a", "echo b", "echo c"])
    assert [r.stdout.strip() for r in results] == ["a", "b", "c"]
    assert all(r.ok for r in results)
    # Only one connect() call for the whole batch, even though three
    # exec_command() calls happen.
    assert fake_client.connect.call_count == 1
    assert fake_client.exec_command.call_count == 3
    assert fake_client.close.call_count == 1


def test_run_many_returns_per_command_error_on_connect_failure(monkeypatch):
    """If we can't connect at all, every cmd in the batch reports the same error."""

    fake_paramiko_mod = MagicMock()
    fake_client = MagicMock()
    fake_client.connect.side_effect = TimeoutError("connect timeout")
    fake_paramiko_mod.SSHClient.return_value = fake_client
    fake_paramiko_mod.AutoAddPolicy.return_value = object()
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko_mod)

    runner = SshTargetRunner(host="1.2.3.4", username="u", password="p")
    results = runner.run_many(["a", "b", "c"])
    assert len(results) == 3
    for r, expected_cmd in zip(results, ["a", "b", "c"]):
        assert r.cmd == expected_cmd
        assert r.error.startswith("TimeoutError")
        assert r.rc == -1
