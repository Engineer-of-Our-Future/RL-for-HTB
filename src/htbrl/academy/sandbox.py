"""Per-module sandbox SSH wrapper.

HTB Academy modules sometimes spin up a per-section sandbox VM you SSH into
(``ssh <user>@<spawn-id>.htb.binary.ninja`` style). This module wraps that
lifecycle around the existing ``SSHSession`` so the answerer can run a single
command and get its output back as a string.

We deliberately keep this thin - it's just an adapter. SSHSession owns the
shell-state details.
"""

from __future__ import annotations

from htbrl.academy.page_models import AcademySandbox
from htbrl.env.ssh_session import SSHCredentials, SSHSession


class SandboxRunner:
    """Open an SSHSession to an academy sandbox; run() executes commands."""

    def __init__(self, sandbox: AcademySandbox, command_timeout_s: float = 60.0) -> None:
        self.sandbox = sandbox
        self.command_timeout_s = command_timeout_s
        self._session: SSHSession | None = None

    @property
    def is_open(self) -> bool:
        return self._session is not None and self._session.is_open

    def open(self) -> None:
        if self.is_open:
            return
        creds = SSHCredentials(
            host=self.sandbox.host,
            port=self.sandbox.port,
            user=self.sandbox.user,
            password=self.sandbox.password,
            identity_file=None,  # academy sandboxes typically use password auth
            connect_timeout_seconds=10.0,
        )
        sess = SSHSession(creds)
        sess.open()
        self._session = sess

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None

    def run(self, command: str) -> str:
        """Execute a single shell command, return its stdout. Raises on timeout."""
        if not self.is_open:
            self.open()
        assert self._session is not None
        result = self._session.run(command, timeout=self.command_timeout_s)
        if result.timed_out:
            raise TimeoutError(f"sandbox command timed out: {command!r}")
        return result.stdout

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # Make the runner look like the SandboxRunner Callable used by HeuristicAnswerer.
    def __call__(self, command: str) -> str:
        return self.run(command)
