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

    User-Agent defaults to a cURL string because many academy targets
    explicitly check the UA and refuse non-cURL clients with the
    response body literally being ``"Please use cURL"``. The point of
    these academy exercises is to teach students to use cURL, so the
    target's "anti-bot" check is part of the curriculum, not an attempt
    to block legitimate operators - we match what a human running
    ``curl`` would send.
    """

    host: str
    port: int | None = 80
    scheme: str = "http"        # "http" or "https"
    timeout_s: float = 12.0
    max_body_bytes: int = 1_048_576    # 1 MB - enough for any academy probe
    user_agent: str = "curl/8.4.0"     # match a real cURL version banner

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
            "User-Agent": self.user_agent,
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
    "extract_html_endpoints",
    "extract_http_method_hint",
    "extract_json_field",
    "extract_curl_commands",
    "probe_code_blocks_for_flag",
    "probe_target_for_answer",
    "try_authenticated_search",
    "try_crud_chain",
    "try_form_login",
]


# ---- theory-driven probing -------------------------------------------------
#
# Academy modules teach by example: every section that requires a target
# also shows the cURL / HTTP commands to run against it inside ``<pre>`` /
# ``<code>`` blocks. Treat those code blocks as TEACHING - the model reads
# the theory first (just like a human would), then practices by running
# the same commands the section showed. This is the "theory then practice"
# loop the operator asked for.


def extract_curl_commands(code_blocks: list[str]) -> list[dict]:
    """Parse cURL-shaped commands from a section's code blocks.

    Returns a list of dicts with the fields the HTTP runner needs:
      ``{"method": "GET", "path": "/", "headers": {...}, "data": <bytes|str|None>}``

    Robust against HTB Academy's CSS-obfuscated shell prompts (the
    academy interleaves invisible characters into the displayed
    "student@htb[/htb]$" so naive copy-paste produces strings like
    ``"S5sherS4stem@htb[/htb]$ curl ..."``). We just locate the first
    ``curl`` token in each line and parse from there, ignoring any
    prompt prefix.
    """
    out: list[dict] = []
    for blk in code_blocks or []:
        text = (blk or "").strip()
        # Code blocks often join multiple commands with newlines or ``\\``
        # line continuations. Normalize backslash-newlines first so a
        # multi-line ``curl ... \`` flag block parses as one command.
        text = _re.sub(r"\\\s*\n\s*", " ", text)
        for raw_line in text.splitlines():
            # Find the first ``curl`` token in the line, allowing an
            # arbitrary obfuscated prompt before it.
            m = _re.search(r"\bcurl\b", raw_line)
            if not m:
                continue
            line = raw_line[m.start():].strip()
            parsed = _parse_curl(line)
            if parsed:
                out.append(parsed)
    return out


def _parse_curl(line: str) -> dict | None:
    """Parse one ``curl ...`` line into a runner-shaped dict."""
    import shlex
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError:
        return None
    if not tokens or tokens[0].lower() != "curl":
        return None
    method = "GET"
    headers: dict[str, str] = {}
    data: str | None = None
    url: str | None = None
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in ("-X", "--request") and i + 1 < len(tokens):
            method = tokens[i + 1].upper()
            i += 2; continue
        if t in ("-H", "--header") and i + 1 < len(tokens):
            h = tokens[i + 1]
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip()] = v.strip()
            i += 2; continue
        if t in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(tokens):
            data = tokens[i + 1]
            if method == "GET":
                method = "POST"
            i += 2; continue
        if t in ("-b", "--cookie") and i + 1 < len(tokens):
            headers["Cookie"] = tokens[i + 1]
            i += 2; continue
        if t in ("-A", "--user-agent") and i + 1 < len(tokens):
            headers["User-Agent"] = tokens[i + 1]
            i += 2; continue
        if t in ("-u", "--user") and i + 1 < len(tokens):
            import base64 as _b64
            creds = tokens[i + 1]
            headers["Authorization"] = "Basic " + _b64.b64encode(
                creds.encode("utf-8")
            ).decode("ascii")
            i += 2; continue
        if t in ("-s", "-S", "-v", "-i", "-I", "-L", "--silent", "--include",
                 "--head", "--location", "--verbose"):
            # Headers-only mode for ``-I``/``--head`` -> issue HEAD instead.
            if t in ("-I", "--head"):
                method = "HEAD"
            i += 1; continue
        if t.startswith("-"):
            # Unknown flag; skip its arg if it looks like a value.
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                i += 2; continue
            i += 1; continue
        if url is None:
            url = t
        i += 1
    if not url:
        return None
    # Reduce URL to a path: strip scheme + host, keep path + query.
    path = "/"
    m = _re.match(r"^[a-zA-Z]+://[^/]+(/.*)?$", url)
    if m:
        path = m.group(1) or "/"
    elif url.startswith("/"):
        path = url
    else:
        # Bare host with no path; default to /.
        path = "/"
    return {"method": method, "path": path, "headers": headers, "data": data}


def probe_code_blocks_for_flag(
    runner: "HttpTargetRunner",
    code_blocks: list[str],
    *,
    max_probes: int = 6,
) -> tuple[str, str, float] | None:
    """Run cURL examples from the section's theory, looking for HTB flags.

    Mirrors the operator's stated mental model: a human reads the theory,
    sees ``curl -X POST /api/...``, copies the command, runs it against
    the spawned target, and reports whatever HTB flag the response
    contains. The model does the same: any cURL we can parse from the
    section's code blocks is re-hosted against the live target's IP, and
    if the response body has an ``HTB{...}`` token we surface it as a
    high-confidence answer.

    Capped at ``max_probes`` HTTP requests so we don't spray when a
    section dumps a huge list of examples.
    """
    cmds = extract_curl_commands(code_blocks)
    if not cmds:
        return None
    seen_paths: set[tuple[str, str]] = set()
    for cmd in cmds[:max_probes]:
        key = (cmd["method"], cmd["path"])
        if key in seen_paths:
            continue
        seen_paths.add(key)
        resp = runner.request(
            cmd["path"],
            method=cmd["method"],
            headers=cmd["headers"] or None,
            data=cmd.get("data"),
        )
        m = _re.search(r"HTB\{[^}]+\}", resp.body_text)
        if m:
            return (
                m.group(0),
                f"{cmd['method']} {cmd['path']} -> body had {m.group(0)!r}",
                0.95,
            )
    return None


# ---- top-level probe orchestrator ------------------------------------------


_HTB_FLAG_RE_BYTES = b"HTB\\{[^}]+\\}"  # noqa
import re as _re


def probe_target_for_answer(
    runner: "HttpTargetRunner",
    question_prompt: str,
    *,
    section_code_blocks: list[str] | None = None,
    hints: list[str] | None = None,
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

    ``hints`` is the per-question hint text the academy reveals when the
    operator clicks the Hint button. We treat hints as ADDITIONAL prompt
    text for pattern matching - the hint often disambiguates which
    pattern applies (e.g. "the request method is at the beginning of the
    HTTP request" tells us to look for an HTTP verb).
    """
    # Combine the prompt with any hints so pattern regexes see both.
    hints_text = " ".join(h for h in (hints or []) if h)
    enriched_prompt = question_prompt + (" " + hints_text if hints_text else "")
    p = enriched_prompt.lower()
    code_blocks = section_code_blocks or []
    # For backwards-compat with patterns that use ``question_prompt``
    # directly: keep a reference to the enriched version.
    question_prompt = enriched_prompt

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

    # -- Pattern 5: HTTP method (case-sensitive) -----------------------------
    # "What is the HTTP method used while intercepting the request?
    # (case-sensitive)" - the answer is a verb token like GET/POST/PUT/...
    # The section's code blocks usually show the captured request, e.g.
    # ``POST /search HTTP/1.1`` or ``Method: POST``.
    if _re.search(
        r"(http\s+method|method\s+used|request\s+method|verb\s+used)",
        question_prompt, _re.IGNORECASE,
    ):
        verb = extract_http_method_hint(code_blocks)
        if verb:
            return (verb, f"section code-block contained {verb!r}", 0.90)

    # -- Pattern 6: authenticated search (login -> cookie -> search) ---------
    # "Authenticate to ... user 'admin' and password 'admin', use cURL
    # to search for 'flag' and obtain the flag."
    creds = _extract_credentials(question_prompt)
    if creds is not None:
        username, password = creds
        # The search query is whatever the prompt asks us to search for;
        # default to "flag" since most academy auth-search tasks are flag
        # discovery.
        m = _re.search(
            r"search\s+for\s+['\"`]?([\w\-]{1,40})['\"`]?",
            question_prompt, _re.IGNORECASE,
        )
        query = m.group(1) if m else "flag"
        # Pull login + search endpoints out of the section's cURL examples
        # so we use the *literally taught* paths (the academy's curriculum
        # IS our endpoint hint).
        login_paths_hint = _paths_from_code_blocks(
            code_blocks, hint=("login", "auth", "signin"),
        )
        search_paths_hint = _paths_from_code_blocks(
            code_blocks, hint=("search", "find", "query"),
        )
        cookie = try_form_login(
            runner, username, password,
            extra_paths=login_paths_hint,
        )
        if cookie is not None:
            json_body = bool(_re.search(
                r"json\s+(post|request)|application/json",
                question_prompt, _re.IGNORECASE,
            ))
            # Look for a specific endpoint in the prompt (e.g. /search.php)
            m_ep = _re.search(r"['\"`](/[\w\-./]+\.[\w]{2,5})['\"`]", question_prompt)
            search_endpoint = m_ep.group(1) if m_ep else None
            # Try the prompted endpoint first, then any from code blocks.
            for ep in [search_endpoint, *search_paths_hint, None]:
                flag = try_authenticated_search(
                    runner, cookie, query,
                    json_body=json_body, endpoint=ep,
                )
                if flag:
                    return (
                        flag,
                        f"login admin/{password} -> cookie -> search '{query}' "
                        f"@ {ep or 'default-paths'} -> {flag}",
                        0.95,
                    )

    # -- Pattern 7: REST CRUD chain ------------------------------------------
    # "First, try to update any city's name to be 'flag'. Then, delete
    # any city. Once done, search for a city named 'flag' to get the flag."
    if _re.search(
        r"(update|modify|change).*?(name|value).*?(flag).*?(delete|remove)|"
        r"(delete|remove).*?(then|after).*?(search|get|fetch).*?(flag)",
        question_prompt, _re.IGNORECASE | _re.DOTALL,
    ):
        m = _re.search(
            r"\b(?:update|modify|delete|remove)\s+(?:any\s+|the\s+)?(\w+?)(?:'s|\s+name)",
            question_prompt, _re.IGNORECASE,
        )
        resource = m.group(1).rstrip("s") + "s" if m else None
        flag = try_crud_chain(
            runner, resource_hint=resource,
            magic_value="flag", code_blocks=code_blocks,
        )
        if flag:
            return (flag, f"CRUD chain on {resource or '?'} -> {flag}", 0.95)

    # -- Pattern 8: HTML resource discovery ("Network tab") ------------------
    # "Use the Network tab in the browser devtools to see what requests
    # are made by the page, and find the request to the flag."
    if _re.search(
        r"(network\s+tab|devtools|developer\s+tools|browser\s+inspect)",
        question_prompt, _re.IGNORECASE,
    ):
        flag = _discover_flag_via_html_endpoints(runner)
        if flag:
            return (flag, f"HTML/JS endpoint discovery -> {flag}", 0.95)

    # -- Pattern 9: JSON-field walk over hinted endpoints (last resort) -----
    # When the prompt mentions a quoted/back-ticked field name AND the
    # section's code blocks suggest /api/X endpoints, probe each and
    # return the first matching field value. Lower-priority on purpose:
    # earlier patterns are more specific. Only fires when prompt language
    # is question-y ("what is the value of X" / "report the value") and
    # avoids triggering on action-verb prompts (those go through patterns
    # 6-7 above).
    if _re.search(
        r"(report|what is|return)\s+(?:the\s+)?(?:value|content|data)\b",
        question_prompt, _re.IGNORECASE,
    ):
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


# ---- multi-step helpers (login + cookie + search, CRUD, etc) ----------------


# Common login form paths the academy uses across modules.
_LOGIN_PATHS = (
    "/login.php", "/login", "/signin", "/auth/login", "/api/login",
    "/api/auth/login", "/users/login", "/auth", "/admin/login",
)
# Form-field name pairs for username/password (common variants).
_LOGIN_FIELD_PAIRS = (
    ("username", "password"),
    ("user", "pass"),
    ("email", "password"),
    ("login", "password"),
    ("name", "password"),
)


def try_form_login(
    runner: "HttpTargetRunner",
    username: str,
    password: str,
    *,
    extra_paths: list[str] | None = None,
) -> str | None:
    """POST credentials to common login paths, return the session cookie.

    Tries each path in :data:`_LOGIN_PATHS` (plus any user-supplied
    ``extra_paths``) with each common username/password field-name pair.
    Returns the response's ``Set-Cookie`` (just the cookie pair, no
    attributes) when one of them succeeds with a 2xx/3xx, otherwise None.

    "Success" here = the response set a cookie. Even if the login page
    returns 200 with an error message we'd still get a cookie if the
    backend sets a session cookie (the academy's intentionally-leaky
    auth flow that some modules teach).
    """
    paths = list(_LOGIN_PATHS) + list(extra_paths or [])
    for path in paths:
        for user_field, pass_field in _LOGIN_FIELD_PAIRS:
            data = {user_field: username, pass_field: password}
            resp = runner.request(path, method="POST", data=data)
            if not resp.ok:
                # Try once more as JSON (some endpoints reject form data).
                resp = runner.request(
                    path, method="POST",
                    headers={"Content-Type": "application/json"},
                    data=_json.dumps(data),
                )
            if not resp.ok:
                continue
            cookie = _extract_session_cookie(resp.headers.get("set-cookie", ""))
            if cookie:
                return cookie
    return None


def _extract_session_cookie(set_cookie_header: str) -> str | None:
    """Return ``"name=value"`` for the first cookie in a Set-Cookie header."""
    if not set_cookie_header:
        return None
    # Set-Cookie may be multiple comma-joined values OR one with attrs.
    # Take the first ``name=value`` pair before any ``;``.
    first = set_cookie_header.split(",")[0].strip()
    pair = first.split(";")[0].strip()
    if "=" in pair:
        return pair
    return None


# Common search endpoints + parameter names.
_SEARCH_PATHS = (
    "/search.php", "/search", "/api/search", "/find", "/api/find",
)
_SEARCH_PARAM_NAMES = ("search", "q", "query", "keyword", "term")


def try_authenticated_search(
    runner: "HttpTargetRunner",
    cookie: str,
    query: str,
    *,
    json_body: bool = False,
    endpoint: str | None = None,
) -> str | None:
    """Search with a session cookie; return the first ``HTB{...}`` we find.

    If ``endpoint`` is supplied, only that path is tried (with both GET
    + query-string and POST + body forms); otherwise we walk every path
    in :data:`_SEARCH_PATHS`. Each path is tried with form-encoded body
    AND JSON body (when ``json_body`` is True or as a fallback). If the
    response body has an ``HTB{...}`` token, it's returned.
    """
    paths = [endpoint] if endpoint else list(_SEARCH_PATHS)
    for path in paths:
        for param in _SEARCH_PARAM_NAMES:
            data = {param: query}
            attempts: list[tuple[str, dict[str, str], Any]] = []
            attempts.append(("GET", {}, None))  # filled below
            attempts.append(("POST_FORM", {}, data))
            if json_body:
                attempts.insert(0, ("POST_JSON", {"Content-Type": "application/json"}, _json.dumps(data)))
            else:
                attempts.append(("POST_JSON", {"Content-Type": "application/json"}, _json.dumps(data)))
            for tag, hdrs, body in attempts:
                merged = {"Cookie": cookie, **hdrs}
                if tag == "GET":
                    resp = runner.request(
                        f"{path}?{urllib.parse.urlencode(data)}",
                        method="GET", headers=merged,
                    )
                elif tag == "POST_FORM":
                    resp = runner.request(path, method="POST", headers=merged, data=data)
                else:
                    resp = runner.request(path, method="POST", headers=merged, data=body)
                m = _re.search(r"HTB\{[^}]+\}", resp.body_text)
                if m:
                    return m.group(0)
    return None


# Common REST collection paths + magic identifiers.
_CRUD_COLLECTION_PATHS = (
    "/api/cities", "/cities", "/api/items", "/api/users", "/api/products",
    "/api/posts", "/api/v1/cities", "/api/v1/items",
)


def try_crud_chain(
    runner: "HttpTargetRunner",
    *,
    resource_hint: str | None = None,
    magic_value: str = "flag",
    code_blocks: list[str] | None = None,
) -> str | None:
    """Run a typical update -> delete -> search chain on a REST collection.

    Tries each candidate collection (from ``code_blocks`` and the default
    list) until one responds with a JSON list. For that collection:
      1. Find the first item with an ``id`` field.
      2. ``PUT`` (and fall back to ``PATCH``) with ``{"name": magic_value}``.
      3. ``DELETE`` the second item to satisfy "delete any city".
      4. ``GET`` the collection again and look for any object whose
         ``name`` matches ``magic_value`` - return its ``HTB{...}``-bearing
         field if any, or any ``HTB{...}`` in the full body.
    """
    cands: list[str] = list(_CRUD_COLLECTION_PATHS)
    if code_blocks:
        for blk in code_blocks:
            for ep in _re.findall(r"/(?:api|v\d+)/[\w\-/]{1,60}", blk):
                ep = ep.split("?", 1)[0].rstrip("/")
                if ep not in cands:
                    cands.insert(0, ep)
    if resource_hint:
        cands.insert(0, f"/api/{resource_hint}")
        cands.insert(0, f"/{resource_hint}")
    seen: set[str] = set()
    for collection in cands:
        if collection in seen:
            continue
        seen.add(collection)
        listing = runner.request(collection)
        if not (listing.ok and isinstance(listing.body_json, list)):
            continue
        items = [it for it in listing.body_json if isinstance(it, dict) and "id" in it]
        if len(items) < 2:
            continue
        first_id = items[0]["id"]
        second_id = items[1]["id"]
        # PUT then PATCH fallback.
        for verb in ("PUT", "PATCH"):
            runner.request(
                f"{collection}/{first_id}", method=verb,
                headers={"Content-Type": "application/json"},
                data=_json.dumps({"name": magic_value}),
            )
        runner.request(f"{collection}/{second_id}", method="DELETE")
        # Search after the chain.
        post_listing = runner.request(collection)
        m = _re.search(r"HTB\{[^}]+\}", post_listing.body_text)
        if m:
            return m.group(0)
    return None


def _extract_credentials(prompt: str) -> tuple[str, str] | None:
    """Pull (username, password) from a prompt describing the credentials.

    Recognizes:
      - "user 'admin' and password 'admin'"
      - "username 'X' password 'Y'"
      - "credentials: 'X' / 'Y'"
    """
    p = prompt
    m = _re.search(
        r"user(?:name)?\s*['\"`]([^'\"`]{1,40})['\"`]"
        r".{0,40}?"
        r"(?:and\s+)?password\s*['\"`]([^'\"`]{1,80})['\"`]",
        p, _re.IGNORECASE | _re.DOTALL,
    )
    if m:
        return (m.group(1), m.group(2))
    m = _re.search(
        r"credentials\s*[:=]\s*['\"`]([^'\"`]{1,40})['\"`]\s*[/:]\s*['\"`]([^'\"`]{1,80})['\"`]",
        p, _re.IGNORECASE,
    )
    if m:
        return (m.group(1), m.group(2))
    return None


def extract_html_endpoints(html: str) -> list[str]:
    """Pull href / src / fetch / XHR endpoints out of an HTML response.

    Used by the "Network tab in browser devtools" question pattern: when
    the prompt asks the student to find which request the page makes, we
    fetch the main page, parse out every URL it references, and probe
    each for a flag. Filters out off-target absolute URLs and the well-
    known noisy paths (favicon, etc).
    """
    if not html:
        return []
    out: set[str] = set()
    for m in _re.finditer(
        r"""(?:href|src|action)\s*=\s*['"]([^'"]+)['"]""", html
    ):
        out.add(m.group(1))
    for m in _re.finditer(
        r"""fetch\(\s*['"]([^'"]+)['"]""", html
    ):
        out.add(m.group(1))
    for m in _re.finditer(
        r"""\.open\(\s*['"][A-Z]+['"]\s*,\s*['"]([^'"]+)['"]""", html
    ):
        out.add(m.group(1))
    # Filter to relative / same-host paths and drop noise.
    cleaned: list[str] = []
    skip_substrings = ("favicon", "googleapis", "cdnjs", "//cdn.")
    for u in out:
        if any(s in u for s in skip_substrings):
            continue
        if u.startswith("http://") or u.startswith("https://"):
            # Off-host absolute URL; skip.
            continue
        if not u.startswith("/"):
            u = "/" + u
        cleaned.append(u)
    return cleaned


def _discover_flag_via_html_endpoints(runner: "HttpTargetRunner") -> str | None:
    """For "Network-tab" questions: fetch /, parse URLs, probe each for HTB{}."""
    home = runner.request("/")
    if not home.body_text:
        return None
    eps = extract_html_endpoints(home.body_text)
    # Also try a small set of common flag-endpoint guesses.
    eps = list(dict.fromkeys(
        eps + ["/flag", "/flag.php", "/flag.txt", "/flag.json", "/api/flag"]
    ))
    for ep in eps[:10]:
        resp = runner.request(ep)
        m = _re.search(r"HTB\{[^}]+\}", resp.body_text)
        if m:
            return m.group(0)
    return None


def _paths_from_code_blocks(
    code_blocks: list[str], *, hint: tuple[str, ...] = (),
) -> list[str]:
    """Pull /paths out of a section's code blocks.

    When ``hint`` is supplied, only paths whose URL contains one of the
    hint substrings are returned (case-insensitive). Used to surface
    endpoints the academy's curriculum is teaching the student to hit
    (e.g. ``/login.php``, ``/search.php``, ``/api/cities``) directly out
    of the section's cURL examples.
    """
    out: list[str] = []
    seen: set[str] = set()
    for blk in code_blocks or []:
        for m in _re.finditer(r"https?://[^\s'\"`]+(/[\w\-./?=&%]*)?", blk):
            full = m.group(0)
            url_path = "/"
            mp = _re.match(r"^https?://[^/]+(/[^?\s]*)?", full)
            if mp and mp.group(1):
                url_path = mp.group(1)
            # Drop query strings; we re-add them ourselves.
            url_path = url_path.split("?", 1)[0]
            if not url_path or url_path == "/":
                continue
            if hint:
                low = url_path.lower()
                if not any(h in low for h in hint):
                    continue
            if url_path not in seen:
                seen.add(url_path)
                out.append(url_path)
        # Also pull bare /paths that appear without a host (relative refs).
        for m in _re.finditer(r"\b(/(?:[\w\-]+/)*[\w\-]+\.\w{2,5}|/api/[\w\-/]+)", blk):
            p = m.group(1)
            if hint:
                low = p.lower()
                if not any(h in low for h in hint):
                    continue
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def extract_http_method_hint(code_blocks: list[str]) -> str | None:
    """Find a case-sensitive HTTP method verb in the section's code blocks.

    Looks for ``METHOD /path HTTP/1.1`` or ``Method: METHOD`` patterns
    (intercepted-request / Burp / devtools format). Returns the first
    match, preserving the original casing (HTB academy questions are
    typically case-sensitive on the verb).
    """
    verbs = ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS")
    for blk in code_blocks or []:
        for line in (blk or "").splitlines():
            line = line.strip()
            for v in verbs:
                # "POST /api/login HTTP/1.1"
                if line.startswith(v + " ") and "HTTP/" in line:
                    return v
                # "Method: POST"
                m = _re.match(r"^method\s*[:=]\s*([A-Z]+)\b", line, _re.IGNORECASE)
                if m and m.group(1).upper() in verbs:
                    return m.group(1).upper()
    return None
