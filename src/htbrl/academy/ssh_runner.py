"""Run probes against an academy target via SSH (paramiko).

Many academy questions ask the operator to ``SSH to <target> with
user "<u>" and password "<p>"`` and then run a shell command to find
the answer (e.g. ``find / -name flag.txt`` or ``cat /etc/release``).
This module is the SSH counterpart to :mod:`target_runner`'s HTTP
runner.

Why paramiko (not subprocess+ssh):
  - Already in the project's allowed dep list (no external CLI).
  - Lets us cap output bytes + per-command timeout from Python.
  - No password-handling on the command line (sshpass-style),
    so the password never appears in process listings or shell
    history.

Security:
  - Per-call timeout caps so a hung target can't lock the wizard.
  - Output capped to ``max_bytes`` so we don't pull a multi-GB
    file back through the wire.
  - Credentials never written to disk - they live only in this
    process for the lifetime of the runner.
  - The runner refuses to talk to anything outside the academy
    target IP supplied at construction time. The wizard's spawn
    helper hands us the IP; we don't accept arbitrary hosts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import paramiko  # noqa: F401


@dataclass
class SshResult:
    """One SSH command run against a target."""

    cmd: str
    rc: int = -1                 # process exit code; -1 if connect/exec failed
    stdout: str = ""             # decoded stdout, capped
    stderr: str = ""             # decoded stderr, capped
    error: str = ""              # connect/auth/timeout error description
    elapsed_s: float = 0.0       # wall-clock seconds for the command
    truncated: bool = False      # True if output hit the byte cap

    @property
    def ok(self) -> bool:
        """True iff connect+exec succeeded *and* the process exited 0."""
        return self.error == "" and self.rc == 0


@dataclass
class SshTargetRunner:
    """Stateful SSH runner bound to a specific academy target.

    Each :meth:`run` call opens a fresh transport. We deliberately do
    NOT keep a long-lived ``SSHClient`` alive between calls because:

      * Academy targets often hibernate / get reaped between probe
        attempts; a stale transport then fails opaquely.
      * One-shot connects play well with the wizard's "answer one
        question, move on" loop.

    For multi-command exploration (e.g. discovering the flag file
    name then ``cat``-ing it), prefer :meth:`run_many` which reuses
    a single transport for the batch.
    """

    host: str
    username: str
    password: str
    port: int = 22
    timeout_s: float = 12.0
    max_bytes: int = 16_384
    # Paramiko's host-key policy: academy targets rotate keys per
    # spawn so AutoAdd is the only sane default. We never persist the
    # key (`load_system_host_keys=False` is the implicit default for
    # a fresh ``SSHClient`` instance).
    auto_add_keys: bool = True

    def _connect(self):
        """Open a new SSHClient + transport. Raises on connect/auth failure."""
        # Lazy import: paramiko is a heavy dep, no need to pay its
        # ~200ms import cost when nobody's using SSH.
        import paramiko

        client = paramiko.SSHClient()
        if self.auto_add_keys:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.timeout_s,
            banner_timeout=self.timeout_s,
            auth_timeout=self.timeout_s,
            look_for_keys=False,
            allow_agent=False,
        )
        return client

    def run(self, cmd: str, *, timeout_s: float | None = None) -> SshResult:
        """Connect, run one command, capture rc/stdout/stderr, disconnect.

        Returns an :class:`SshResult` describing the outcome. Connect
        failures (timeouts, auth errors, dropped connections) populate
        ``error`` and leave ``rc`` at -1; the caller can branch on
        ``result.ok`` for "did it succeed end-to-end".
        """
        t = self.timeout_s if timeout_s is None else timeout_s
        start = time.monotonic()
        try:
            client = self._connect()
        except Exception as exc:
            return SshResult(
                cmd=cmd,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_s=time.monotonic() - start,
            )
        try:
            _stdin, stdout, stderr = client.exec_command(cmd, timeout=t)
            out_raw = stdout.read(self.max_bytes + 1)
            err_raw = stderr.read(self.max_bytes + 1)
            rc = stdout.channel.recv_exit_status()
        except Exception as exc:
            return SshResult(
                cmd=cmd,
                error=f"exec: {type(exc).__name__}: {exc}",
                elapsed_s=time.monotonic() - start,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

        truncated = len(out_raw) > self.max_bytes or len(err_raw) > self.max_bytes
        out_raw = out_raw[: self.max_bytes]
        err_raw = err_raw[: self.max_bytes]
        return SshResult(
            cmd=cmd,
            rc=rc,
            stdout=out_raw.decode("utf-8", errors="replace"),
            stderr=err_raw.decode("utf-8", errors="replace"),
            elapsed_s=time.monotonic() - start,
            truncated=truncated,
        )

    def run_many(
        self, cmds: list[str], *, timeout_s: float | None = None,
    ) -> list[SshResult]:
        """Run a batch of commands over a single transport.

        Cheaper than calling :meth:`run` per command (saves the
        per-command auth round-trip ~300-700ms) when you know up-front
        you'll fire several. If any connect step fails the whole
        batch returns with that error and no commands run.
        """
        t = self.timeout_s if timeout_s is None else timeout_s
        start = time.monotonic()
        try:
            client = self._connect()
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            return [SshResult(cmd=c, error=err, elapsed_s=0.0) for c in cmds]
        results: list[SshResult] = []
        try:
            for cmd in cmds:
                cmd_start = time.monotonic()
                try:
                    _stdin, stdout, stderr = client.exec_command(cmd, timeout=t)
                    out_raw = stdout.read(self.max_bytes + 1)
                    err_raw = stderr.read(self.max_bytes + 1)
                    rc = stdout.channel.recv_exit_status()
                    truncated = (
                        len(out_raw) > self.max_bytes or len(err_raw) > self.max_bytes
                    )
                    out_raw = out_raw[: self.max_bytes]
                    err_raw = err_raw[: self.max_bytes]
                    results.append(
                        SshResult(
                            cmd=cmd,
                            rc=rc,
                            stdout=out_raw.decode("utf-8", errors="replace"),
                            stderr=err_raw.decode("utf-8", errors="replace"),
                            elapsed_s=time.monotonic() - cmd_start,
                            truncated=truncated,
                        )
                    )
                except Exception as exc:
                    results.append(
                        SshResult(
                            cmd=cmd,
                            error=f"exec: {type(exc).__name__}: {exc}",
                            elapsed_s=time.monotonic() - cmd_start,
                        )
                    )
        finally:
            try:
                client.close()
            except Exception:
                pass
        # Account for the overall wall-clock vs the sum of per-cmd
        # timings (the difference is connect overhead).
        total = time.monotonic() - start
        if results and total > sum(r.elapsed_s for r in results):
            results[0].elapsed_s = max(results[0].elapsed_s, total - sum(
                r.elapsed_s for r in results[1:]
            ))
        return results


# ---- prompt parsing ----------------------------------------------------------

# Academy prompts ship the SSH credentials in plain English directly in
# the question card body, in one of two slightly-different forms:
#
#   1.  "SSH to 10.129.100.38 (ACADEMY-NIXFUND), with user \"htb-student\"
#        and password \"HTB_@cademy_stdnt!\""
#
#   2.  "SSH to with user \"htb-student\" and password \"HTB_@cademy_stdnt!\""
#       (host omitted because it's already shown in the section's Target
#       panel; the wizard fills the host from the panel side-channel)
#
# We extract (host?, user, password) so the wizard can wire up
# :class:`SshTargetRunner` automatically.

_USER_PASS_RE = re.compile(
    r"""user[\s:]*['"]?(?P<user>[A-Za-z0-9_.\-+]+)['"]?
        \s+(?:and\s+)?password[\s:]*['"](?P<pw>[^'"]+)['"]""",
    re.IGNORECASE | re.VERBOSE,
)
_HOST_RE = re.compile(r"SSH\s+to\s+([0-9]{1,3}(?:\.[0-9]{1,3}){3})", re.IGNORECASE)


@dataclass
class SshPromptCreds:
    """Parsed SSH credentials from an academy question prompt."""

    host: str | None = None
    user: str = ""
    password: str = ""

    @property
    def complete(self) -> bool:
        return bool(self.user and self.password)


def parse_ssh_credentials(prompt: str) -> SshPromptCreds:
    """Pull SSH (host?, user, password) out of an academy question prompt.

    Returns an :class:`SshPromptCreds`. ``host`` may be None when the
    prompt only mentions credentials and expects the wizard to know
    the target IP from the section's Target panel.
    """
    creds = SshPromptCreds()
    m = _HOST_RE.search(prompt or "")
    if m:
        creds.host = m.group(1)
    m2 = _USER_PASS_RE.search(prompt or "")
    if m2:
        creds.user = m2.group("user")
        creds.password = m2.group("pw")
    return creds


__all__ = [
    "SshResult",
    "SshTargetRunner",
    "SshPromptCreds",
    "parse_ssh_credentials",
]
