"""Persistent SSH session into the Kali attacker (PLAN.md Phase 4).

One ``SSHSession`` == one persistent shell on the Kali host. Across many
``run`` calls, environment variables, current directory, opened SOCKS proxies,
and background processes survive. This is what lets the policy use
multi-step pentest workflows (e.g. ``cd /tmp/loot``, then ``ls``, then
``cat`` something) without each command starting from a fresh shell.

We use ``paramiko``'s ``invoke_shell`` rather than per-command ``exec_command``
so the shell stays alive. Output is delimited by an explicit sentinel string
that the wrapper appends to each command, since SSH doesn't natively give us
"command finished" boundaries on a tty.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import paramiko


_PROMPT_SENTINEL_PREFIX = "__HTBRL_DONE__"


@dataclass
class SSHCredentials:
    host: str
    user: str
    port: int = 22
    identity_file: str | None = None       # absolute path to private key
    password: str | None = None            # optional fallback (NOT for production)
    connect_timeout_seconds: float = 10.0


@dataclass
class CommandResult:
    """Output of one ``run`` call."""

    stdout: str
    timed_out: bool
    wallclock_seconds: float
    # Best-effort exit code (parsed from ``$?`` if we appended its capture).
    exit_code: int | None = None


class SSHSession:
    """Persistent shell on a remote host, opened once and reused.

    Use as a context manager:

        with SSHSession(creds) as sh:
            r = sh.run("nmap -sS 10.10.10.5", timeout=120)
            print(r.stdout)
    """

    def __init__(self, creds: SSHCredentials) -> None:
        self.creds = creds
        self._client: paramiko.SSHClient | None = None
        self._channel: paramiko.Channel | None = None
        self._opened = False

    # ----- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.creds.host,
            "port": self.creds.port,
            "username": self.creds.user,
            "timeout": self.creds.connect_timeout_seconds,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if self.creds.identity_file:
            connect_kwargs["key_filename"] = self.creds.identity_file
        if self.creds.password:
            connect_kwargs["password"] = self.creds.password
            connect_kwargs["allow_agent"] = False
            connect_kwargs["look_for_keys"] = False

        client.connect(**connect_kwargs)
        # invoke_shell gives us a persistent tty.
        chan = client.invoke_shell(term="xterm", width=200, height=50)
        chan.settimeout(0.0)  # non-blocking reads

        # Drain the initial banner / motd / prompt so subsequent reads start
        # from a clean slate.
        time.sleep(0.4)
        self._drain(chan, max_seconds=2.0)

        self._client = client
        self._channel = chan
        self._opened = True

        # Configure the shell to disable echo + bracketed paste, which simplifies
        # output parsing. We also export PS1 to a benign value so prompt strings
        # don't pollute stdout between commands.
        self._send_raw("stty -echo; export PS1='$ '; export TERM=dumb; clear\n")
        time.sleep(0.2)
        self._drain(self._channel, max_seconds=1.0)

    def close(self) -> None:
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._channel = None
        self._client = None
        self._opened = False

    def __enter__(self) -> "SSHSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._opened

    # ----- run a single command ----------------------------------------------

    def run(self, command: str, timeout: float = 60.0) -> CommandResult:
        """Send a command, wait for the sentinel, return everything in between."""
        if not self._opened:
            raise RuntimeError("SSHSession not open; call open() or use 'with'")
        assert self._channel is not None

        sentinel = f"{_PROMPT_SENTINEL_PREFIX}{secrets.token_hex(8)}"
        # The trailing semicolon ensures the sentinel runs even if the command fails.
        full = f"{command}; printf '\\n%s_RC=%s\\n' '{sentinel}' \"$?\"\n"
        self._send_raw(full)

        out = []
        deadline = time.time() + timeout
        timed_out = False
        while True:
            chunk = self._read_available(self._channel)
            if chunk:
                out.append(chunk)
            joined = "".join(out)
            # Check for the sentinel + return-code marker on its own line.
            marker_idx = joined.rfind(f"{sentinel}_RC=")
            if marker_idx != -1:
                break
            if time.time() > deadline:
                timed_out = True
                break
            time.sleep(0.05)

        wall = max(0.0, time.time() - (deadline - timeout))
        joined = "".join(out)
        exit_code: int | None = None
        if not timed_out:
            marker = f"{sentinel}_RC="
            idx = joined.rfind(marker)
            tail = joined[idx + len(marker) :]
            rc_str = tail.split("\n", 1)[0].strip()
            try:
                exit_code = int(rc_str)
            except ValueError:
                exit_code = None
            # Strip everything after (and including) the sentinel line.
            sentinel_line_start = joined.rfind(sentinel)
            joined = joined[:sentinel_line_start].rstrip()
            # Strip the echoed command line at the start, if present.
            first_nl = joined.find("\n")
            if first_nl != -1 and command.split("\n", 1)[0] in joined[:first_nl]:
                joined = joined[first_nl + 1 :]

        return CommandResult(
            stdout=joined,
            timed_out=timed_out,
            wallclock_seconds=wall,
            exit_code=exit_code,
        )

    # ----- raw I/O ------------------------------------------------------------

    def _send_raw(self, data: str) -> None:
        assert self._channel is not None
        self._channel.send(data.encode("utf-8", errors="replace"))

    @staticmethod
    def _read_available(chan: paramiko.Channel, max_bytes: int = 1 << 20) -> str:
        """Read whatever is currently available without blocking."""
        out = b""
        if chan.recv_ready():
            try:
                out = chan.recv(max_bytes)
            except Exception:
                out = b""
        return out.decode("utf-8", errors="replace")

    @staticmethod
    def _drain(chan: paramiko.Channel, max_seconds: float = 1.0) -> None:
        """Read and discard any pending output for up to ``max_seconds``."""
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            if chan.recv_ready():
                try:
                    chan.recv(1 << 16)
                except Exception:
                    break
            else:
                time.sleep(0.05)
