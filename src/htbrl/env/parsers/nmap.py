"""Nmap output parser.

Extracts open ports, services, and (when present) versions from the
human-readable text output (``-oN``) that all our nmap_* tools produce.
Robust to extra noise around the port table: walks the output line-by-line
and only acts on lines that match the canonical port-row pattern.

Returns a dict shaped like::

    {
        "text": "<original stdout>",
        "ports": [
            {"port": 22, "proto": "tcp", "state": "open", "service": "ssh", "version": "OpenSSH 8.4p1"},
            ...
        ],
        "host_up": True | False | None,
        "parse_errors": [],
    }

Only ports with state "open" (or "open|filtered") are emitted; closed/filtered
get dropped because the policy doesn't typically care about them.
"""

from __future__ import annotations

import re

# Lines like:  22/tcp   open  ssh    OpenSSH 8.4p1 Debian 5+deb11u1
_PORT_RE = re.compile(
    r"^\s*(?P<port>\d{1,5})/(?P<proto>tcp|udp)\s+"
    r"(?P<state>open|open\|filtered|filtered|closed)\s+"
    r"(?P<service>\S+)"
    r"(?:\s+(?P<version>.+?))?\s*$"
)

_HOST_DOWN_RE = re.compile(r"Host seems down", re.IGNORECASE)
_HOST_UP_RE = re.compile(r"Host is up", re.IGNORECASE)


def parse_nmap(stdout: str) -> dict:
    ports: list[dict] = []
    host_up: bool | None = None
    for line in stdout.splitlines():
        if host_up is None:
            if _HOST_UP_RE.search(line):
                host_up = True
            elif _HOST_DOWN_RE.search(line):
                host_up = False
        m = _PORT_RE.match(line)
        if not m:
            continue
        state = m.group("state")
        if state in {"closed", "filtered"}:
            continue
        ports.append({
            "port": int(m.group("port")),
            "proto": m.group("proto"),
            "state": state,
            "service": m.group("service"),
            "version": (m.group("version") or "").strip() or None,
        })
    return {
        "text": stdout,
        "ports": ports,
        "host_up": host_up,
        "parse_errors": [],
    }
