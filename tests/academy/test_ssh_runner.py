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
    probe_via_ssh,
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


# ---- probe_via_ssh: question-pattern dispatcher ----------------------------


class _ScriptedRunner:
    """SshTargetRunner stand-in that returns canned ``run`` results.

    Maps each command (full string) to ``(stdout, rc)``. Records every
    command attempted so tests can assert which patterns fired.
    """

    def __init__(self, scripted: dict[str, tuple[str, int]]):
        self.scripted = scripted
        self.calls: list[str] = []

    def run(self, cmd: str, *, timeout_s=None) -> SshResult:
        self.calls.append(cmd)
        if cmd in self.scripted:
            stdout, rc = self.scripted[cmd]
            return SshResult(cmd=cmd, rc=rc, stdout=stdout)
        return SshResult(cmd=cmd, rc=1, stderr="not scripted")


def test_probe_via_ssh_kernel_version():
    """`uname -r` shape: prompt asks for kernel version."""
    runner = _ScriptedRunner({"uname -r": ("4.15.0-123-generic\n", 0)})
    result = probe_via_ssh(
        runner, "What is the kernel version of the target workstation?",
    )
    assert result is not None
    answer, rationale, conf = result
    assert answer == "4.15.0-123-generic"
    assert "uname -r" in rationale
    assert conf == pytest.approx(0.85)


def test_probe_via_ssh_inode_of_path():
    """``stat -c %i`` shape: prompt asks for the inode of /etc/shadow.bak."""
    runner = _ScriptedRunner({"stat -c %i /etc/shadow.bak": ("265293\n", 0)})
    result = probe_via_ssh(runner, 'What is the inode of "/etc/shadow.bak"?')
    assert result is not None
    assert result[0] == "265293"


def test_probe_via_ssh_last_modified_file_in_dir():
    """``ls -1t`` shape: last-modified file in /var/backups."""
    runner = _ScriptedRunner({
        "ls -1t /var/backups 2>/dev/null | head -n 1": ("apt.extended_states.0\n", 0),
    })
    result = probe_via_ssh(
        runner,
        "What is the name of the last modified file in /var/backups?",
    )
    assert result is not None
    assert result[0] == "apt.extended_states.0"


def test_probe_via_ssh_count_files_by_extension():
    """``find / -name '*.bak'`` shape: counts files of a given extension."""
    runner = _ScriptedRunner({
        "find / -name '*.bak' 2>/dev/null | wc -l": ("4\n", 0),
    })
    result = probe_via_ssh(
        runner,
        'How many files exist on the system that have the ".bak" extension?',
    )
    assert result is not None
    assert result[0] == "4"


def test_probe_via_ssh_total_packages_installed():
    """``dpkg -l | grep -c ^ii`` shape: package count."""
    runner = _ScriptedRunner({
        "dpkg -l 2>/dev/null | grep -c '^ii'": ("737\n", 0),
    })
    result = probe_via_ssh(
        runner, "How many total packages are installed on the target system?",
    )
    assert result is not None
    assert result[0] == "737"


def test_probe_via_ssh_systemd_unit_by_description():
    """``systemctl list-units | grep '<desc>'`` shape: find unit by description."""
    runner = _ScriptedRunner({
        "systemctl list-units --all --no-pager 2>/dev/null | grep -F "
        "'Load AppArmor profiles managed internally by snapd' | awk '{print $1}'":
            ("snapd.apparmor.service\n", 0),
    })
    result = probe_via_ssh(
        runner,
        'systemctl ... unit name with the description '
        '"Load AppArmor profiles managed internally by snapd" as the answer.',
    )
    assert result is not None
    assert result[0] == "snapd.apparmor.service"


def test_probe_via_ssh_full_path_of_binary():
    """``command -v xxd`` shape: full path of a binary."""
    runner = _ScriptedRunner({
        "command -v xxd 2>/dev/null || which xxd": ("/usr/bin/xxd\n", 0),
    })
    result = probe_via_ssh(
        runner, 'Submit the full path of the "xxd" binary.',
    )
    assert result is not None
    assert result[0] == "/usr/bin/xxd"


def test_probe_via_ssh_returns_none_on_unmatched_prompt():
    """Prompts that don't fit any known shape must return None *without*
    issuing any SSH command (the runner shouldn't be touched)."""
    runner = _ScriptedRunner({})
    result = probe_via_ssh(runner, "Just a random philosophical question?")
    assert result is None
    assert runner.calls == []   # no shell commands executed


def test_probe_via_ssh_returns_none_when_command_fails():
    """When the matched command exits non-zero (or stdout is empty),
    we don't return a phantom answer."""
    runner = _ScriptedRunner({"uname -r": ("", 1)})
    result = probe_via_ssh(runner, "What is the kernel version?")
    assert result is None
