"""smbclient -L share-list parser."""

from __future__ import annotations

import re


# Lines look like:
#         Sharename       Type      Comment
#         ---------       ----      -------
#         ADMIN$          Disk      Remote Admin
#         IPC$            IPC       Remote IPC
_SHARE_RE = re.compile(
    r"^\s*(?P<name>[\w$\-\.]+)\s+(?P<type>Disk|IPC|Printer)\s*(?P<comment>.*?)\s*$"
)


def parse_smbclient(stdout: str) -> dict:
    shares: list[dict] = []
    in_table = False
    for line in stdout.splitlines():
        if "Sharename" in line and "Type" in line:
            in_table = True
            continue
        if in_table and set(line.strip()) <= {"-", " "}:
            continue
        if in_table and not line.strip():
            in_table = False
            continue
        if not in_table:
            continue
        m = _SHARE_RE.match(line)
        if m:
            shares.append({
                "name": m.group("name"),
                "type": m.group("type"),
                "comment": m.group("comment").strip(),
            })
    return {"text": stdout, "shares": shares, "parse_errors": []}
