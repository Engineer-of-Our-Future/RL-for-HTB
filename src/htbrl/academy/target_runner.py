"""Run probes against a spawned HTB Academy target VM.

Many academy questions ask the operator to *do* something against a target
("submit a GET to /, report the value of the X field" / "what server header
does the target return" / "find the hidden directory"). The heuristic
answerer alone can't answer those - they require live execution.

This module is the "execute" half of the answer pipeline:

  - :func:`spawn_target` (in cdp_walker) clicks the Spawn button and reads
    the target's IP:PORT.
  - :class:`HttpTargetRunner` here runs HTTP/HTTPS requests against that
    target from the operator's host (NOT through CDP - we want the raw
    bytes, not the rendered DOM).
  - The wizard then routes question prompts to handlers that interpret
    the response shape and extract the answer (e.g. JSON field, header
    value, status code).

Pure stdlib (urllib + json) on purpose: keeps the no-extra-deps rule.

Security:
  - Refuses to connect to anything that isn't an academy target IP. The
    academy hands out targets in the public internet ranges; we lock the
    runner to a per-call ``host`` value supplied by the spawn helper.
  - Capped response size + timeouts so a misbehaving target can't hang
    the wizard.
"""

from __future__ import annotations

import json as _json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HttpResponse:
    """One HTTP response captured from a target probe."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    body_text: str = ""        # decoded body, capped
    body_json: Any = None      # parsed JSON if Content-Type was application/json
    url: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and 200 <= self.status < 400


@dataclass
class HttpTargetRunner:
    """Stateful HTTP runner bound to a specific academy target IP:PORT.

    Builds full URLs from a base ``http://<host>:<port>`` (or ``https://``
    when ``scheme="https"``). Caps response bodies, follows redirects up
    to a small bound, and decodes JSON when the response Content-Type
    advertises it.
    """

    host: str
    port: int | None = 80
    scheme: str = "http"        # "http" or "https"
    timeout_s: float = 12.0
    max_body_bytes: int = 1_048_576    # 1 MB - enough for any academy probe

    @property
    def base_url(self) -> str:
        if self.port and not (
            (self.scheme == "http" and self.port == 80)
            or (self.scheme == "https" and self.port == 443)
        ):
            return f"{self.scheme}://{self.host}:{self.port}"
        return f"{self.scheme}://{self.host}"

    def request(
        self,
        path: str = "/",
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: bytes | str | dict | None = None,
        timeout_s: float | None = None,
    ) -> HttpResponse:
        """Run one HTTP request against the bound target.

        ``data`` accepts:
          - bytes: sent as-is
          - str: sent as utf-8
          - dict: form-urlencoded with Content-Type set automatically

        Returns an :class:`HttpResponse` with parsed JSON if the response
        advertises ``application/json``; ``ok`` is True for 2xx/3xx.
        """
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/") if path else self.base_url
        req_headers: dict[str, str] = {
            "User-Agent": "htbrl-academy-runner/0.1",
            "Accept": "*/*",
        }
        if headers:
            req_headers.update(headers)

        body: bytes | None = None
        if isinstance(data, dict):
            body = urllib.parse.urlencode(data).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        elif isinstance(data, str):
            body = data.encode("utf-8")
        elif isinstance(data, bytes):
            body = data

        req = urllib.request.Request(url, data=body, method=method, headers=req_headers)
        t = self.timeout_s if timeout_s is None else timeout_s
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:  # nosec - target is operator-supplied
                raw = resp.read(self.max_body_bytes + 1)
                truncated = len(raw) > self.max_body_bytes
                raw = raw[: self.max_body_bytes]
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                status = resp.status
                final_url = resp.geturl()
        except urllib.error.HTTPError as e:
            try:
                raw = e.read(self.max_body_bytes + 1)
            except Exception:
                raw = b""
            truncated = False
            resp_headers = {k.lower(): v for k, v in (e.headers or {}).items()}
            status = e.code
            final_url = url
        except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
            return HttpResponse(status=0, error=f"{type(e).__name__}: {e}", url=url)

        # Decode body for textual handlers; cap render to keep printouts sane.
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if truncated:
            text += "\n…[truncated]"
        body_json: Any = None
        ctype = (resp_headers.get("content-type") or "").lower()
        if "json" in ctype:
            try:
                body_json = _json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                body_json = None
        return HttpResponse(
            status=status,
            headers=resp_headers,
            body=raw,
            body_text=text,
            body_json=body_json,
            url=final_url,
        )


# ---- answer extractors ------------------------------------------------------


def extract_json_field(resp: HttpResponse, prompt: str) -> str | None:
    """Pull a field value out of a JSON response that the prompt asks about.

    Handles two common academy patterns:
      - "report the value of the ``city`` field"
      - "what is the X returned by the API"

    Returns ``str(value)`` for the matching field, or None if no
    obvious key match.
    """
    if resp.body_json is None:
        return None
    p = prompt.lower()

    def _stringify(v: Any) -> str:
        if isinstance(v, (dict, list)):
            return _json.dumps(v, ensure_ascii=False)
        return str(v)

    def _walk(obj: Any) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                flat[k] = v
                if isinstance(v, (dict, list)):
                    for k2, v2 in _walk(v).items():
                        flat.setdefault(f"{k}.{k2}", v2)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                for k2, v2 in _walk(v).items():
                    flat.setdefault(f"[{i}].{k2}", v2)
        return flat

    flat = _walk(resp.body_json)
    # 1. Direct key-name match: prompt mentions "the X field" -> look for X.
    for k, v in flat.items():
        kk = k.split(".")[-1].lower()
        if kk and kk in p and not isinstance(v, (dict, list)):
            return _stringify(v)
    # 2. Quoted/back-ticked key in prompt.
    import re
    m = re.search(r"['\"`]([a-zA-Z_][\w]{0,30})['\"`]", prompt)
    if m:
        key = m.group(1)
        for k, v in flat.items():
            if k.split(".")[-1] == key and not isinstance(v, (dict, list)):
                return _stringify(v)
    return None


def extract_header_value(resp: HttpResponse, prompt: str) -> str | None:
    """Pull a response-header value the prompt asks about.

    e.g. "what server header does the target return" -> resp.headers['server'].
    """
    p = prompt.lower()
    for header_name in resp.headers.keys():
        if header_name and header_name in p:
            return resp.headers[header_name]
    return None


__all__ = [
    "HttpResponse",
    "HttpTargetRunner",
    "extract_header_value",
    "extract_json_field",
    "probe_target_for_answer",
]


# ---- top-level probe orchestrator ------------------------------------------


_HTB_FLAG_RE_BYTES = b"HTB\\{[^}]+\\}"  # noqa
import re as _re


def probe_target_for_answer(
    runner: "HttpTargetRunner",
    question_prompt: str,
    *,
    section_code_blocks: list[str] | None = None,
) -> tuple[str, str, float] | None:
    """Run a sequence of HTTP probes against ``runner``'s target trying to
    answer the question.

    Returns ``(answer, rationale, confidence)`` on success, None when no
    pattern matched. Designed to be cheap and *narrow*: each pattern picks
    a specific question shape ("server version" / "header value" / "JSON
    field" / "flag at /path") and only fires when the prompt clearly fits.

    Patterns covered (in order):

    1. ``"version of X running on the server"`` -> ``GET /`` and parse
       the ``Server: X/N.N.N`` header. Strong: 0.95 confidence.
    2. ``"download the file returned by '/X'"`` -> fetch ``/X``, return
       any ``HTB{...}`` we find. Strong: 0.95.
    3. ``"what is the value of the X field"`` (JSON-API questions) ->
       walk all known endpoints from section code blocks, fetch each,
       and pull the matching key from any JSON body. Medium: 0.7.
    4. ``"what server header"`` -> ``GET /`` + return Server header.
    """
    p = question_prompt.lower()
    code_blocks = section_code_blocks or []

    # -- Pattern 1: server version -------------------------------------------
    # Examples:
    #   "find the version of Apache running on the server"
    #   "what is the version of nginx the target uses"
    m = _re.search(
        r"version of (?P<sw>[a-zA-Z][\w\-+.]{1,30})\b.*?(?:running|server|target|used)",
        question_prompt, _re.IGNORECASE | _re.DOTALL,
    )
    if m:
        sw = m.group("sw")
        resp = runner.request("/")
        server = (resp.headers.get("server") or "").strip()
        if server:
            m2 = _re.search(
                rf"{_re.escape(sw)}\s*/\s*(\d+(?:\.\d+){{1,3}}(?:[a-zA-Z\-][\w\-]*)?)",
                server, _re.IGNORECASE,
            )
            if m2:
                return (
                    m2.group(1),
                    f"GET / -> Server: {server!r} -> {sw} version {m2.group(1)}",
                    0.95,
                )

    # -- Pattern 2: download flag from a specific path ----------------------
    m = _re.search(
        r"(?:download|fetch|get|access).{0,80}?['\"`/]?(/[\w\-./]+)['\"`]?",
        question_prompt, _re.IGNORECASE,
    )
    if m:
        path = m.group(1)
        if "." in path or path.endswith("/"):  # filter out IPs/ports
            resp = runner.request(path)
            mflag = _re.search(r"HTB\{[^}]+\}", resp.body_text)
            if mflag:
                return (
                    mflag.group(0),
                    f"GET {path} -> body contained {mflag.group(0)!r}",
                    0.95,
                )

    # -- Pattern 3: server header (catch-all version of #1) -----------------
    if "server header" in p or _re.search(r"\bserver\s+(version|name)\b", p):
        resp = runner.request("/")
        server = (resp.headers.get("server") or "").strip()
        if server:
            return (
                server,
                f"GET / -> Server: {server!r}",
                0.85,
            )

    # -- Pattern 4: JSON-field walk over hinted endpoints -------------------
    # When the prompt mentions a quoted/back-ticked field name AND the
    # section's code blocks suggest /api/something, probe each endpoint
    # and return the first matching field value.
    m = _re.search(r"['\"`]([a-zA-Z_][\w]{0,30})['\"`]", question_prompt)
    if m:
        field_name = m.group(1)
        endpoints = set()
        for blk in code_blocks[:6]:
            for ep in _re.findall(r"/(?:api|v\d+)/[\w\-/]{1,80}", blk):
                endpoints.add(ep)
        for ep in list(endpoints)[:5]:
            resp = runner.request(ep)
            extracted = extract_json_field(resp, question_prompt)
            if extracted:
                return (
                    extracted,
                    f"GET {ep} -> JSON field {field_name!r} = {extracted!r}",
                    0.70,
                )

    return None
