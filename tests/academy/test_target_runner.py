"""Unit tests for the academy target probe pipeline.

We mock ``HttpTargetRunner.request`` so the tests don't need a live academy
target. The aim is to exercise every probe pattern + multi-step helper at
least once, locking in the question shapes we know how to answer so future
edits to the regexes don't silently regress real-academy accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from htbrl.academy.target_runner import (
    HttpResponse,
    HttpTargetRunner,
    extract_curl_commands,
    extract_html_endpoints,
    extract_http_method_hint,
    extract_json_field,
    probe_code_blocks_for_flag,
    probe_target_for_answer,
    try_authenticated_search,
    try_crud_chain,
    try_form_login,
)


# ---- request-mock helper ----------------------------------------------------


@dataclass
class FakeRunner:
    """Stand-in for :class:`HttpTargetRunner` that returns canned responses.

    Tests register ``(method, path) -> response`` mappings and an
    optional fallback. Records every call for assertions.
    """

    host: str = "1.2.3.4"
    port: int = 80
    scheme: str = "http"
    user_agent: str = "curl/8.4.0"
    routes: dict[tuple[str, str], HttpResponse] = field(default_factory=dict)
    default: HttpResponse | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def request(self, path="/", *, method="GET", headers=None, data=None,
                timeout_s=None) -> HttpResponse:
        # Normalize path (strip query for keying; keep for record).
        key = (method.upper(), path.split("?", 1)[0])
        self.calls.append({"method": method.upper(), "path": path,
                           "headers": dict(headers or {}), "data": data})
        if key in self.routes:
            return self.routes[key]
        if path.split("?", 1)[0] in self.routes:
            return self.routes[path.split("?", 1)[0]]  # type: ignore[index]
        if self.default is not None:
            return self.default
        return HttpResponse(status=404, body_text="not found", url=path)


def _resp(body: str = "", *, status: int = 200,
          headers: dict[str, str] | None = None,
          body_json: Any = None) -> HttpResponse:
    return HttpResponse(
        status=status, body_text=body,
        headers={k.lower(): v for k, v in (headers or {}).items()},
        body_json=body_json, url="",
    )


# ---- pattern dispatcher ----------------------------------------------------


def test_probe_pattern_server_version_extracts_apache():
    runner = FakeRunner(routes={
        ("GET", "/"): _resp(headers={"Server": "Apache/2.4.41 (Ubuntu)"}),
    })
    out = probe_target_for_answer(
        runner,
        "Send a GET request to find the version of Apache running on the server. "
        "(answer format: X.Y.ZZ)",
    )
    assert out is not None
    answer, _, conf = out
    assert answer == "2.4.41"
    assert conf == 0.95


def test_probe_pattern_path_flag_uses_curl_user_agent():
    runner = FakeRunner(routes={
        ("GET", "/download.php"): _resp(body="HTB{cURL_only_pls}"),
    })
    out = probe_target_for_answer(
        runner,
        "use cURL to download the file returned by '/download.php' to get the flag",
    )
    assert out is not None
    assert out[0] == "HTB{cURL_only_pls}"
    assert out[2] == 0.95


def test_probe_pattern_server_header_fallback():
    runner = FakeRunner(routes={
        ("GET", "/"): _resp(headers={"Server": "nginx/1.18.0"}),
    })
    out = probe_target_for_answer(runner, "what server header does the target return?")
    assert out is not None
    assert "nginx" in out[0]


def test_probe_pattern_http_method_case_sensitive():
    code_blocks = ["POST /search HTTP/1.1\nHost: target\n\n{...}"]
    out = probe_target_for_answer(
        FakeRunner(),
        "What is the HTTP method used while intercepting the request? (case-sensitive)",
        section_code_blocks=code_blocks,
    )
    assert out is not None
    assert out[0] == "POST"


def test_probe_pattern_authenticated_search_captures_flag():
    """End-to-end: prompt asks to log in admin/admin and search for 'flag'."""
    runner = FakeRunner(routes={
        ("POST", "/login.php"): _resp(headers={"Set-Cookie": "PHPSESSID=abc; Path=/"}),
        ("POST", "/search.php"): _resp(body="<html>Result: HTB{auth_flow_works}</html>"),
    }, default=_resp(status=404))
    out = probe_target_for_answer(
        runner,
        "Authenticate to with user 'admin' and password 'admin', then use cURL "
        "to search for 'flag' through a JSON POST request to '/search.php'",
    )
    assert out is not None
    assert out[0] == "HTB{auth_flow_works}"


def test_probe_pattern_crud_chain():
    """Update/delete/search REST chain returns a flag from the post-chain listing."""
    cities_v1 = [{"id": 1, "name": "Boston"}, {"id": 2, "name": "Tokyo"}]
    cities_v2 = [{"id": 1, "name": "flag", "extra": "HTB{crud_chain_complete}"}]
    listing_seq = iter([cities_v1, cities_v2])

    def _fake_request(path="/", *, method="GET", headers=None, data=None,
                      timeout_s=None):
        if path == "/api/cities" and method == "GET":
            try:
                body = next(listing_seq)
            except StopIteration:
                body = cities_v2
            import json
            return HttpResponse(
                status=200, body_text=json.dumps(body),
                headers={"content-type": "application/json"},
                body_json=body, url=path,
            )
        if path.startswith("/api/cities/") and method in ("PUT", "PATCH", "DELETE"):
            return HttpResponse(status=200, body_text="ok", url=path)
        return HttpResponse(status=404, body_text="", url=path)

    runner = HttpTargetRunner(host="1.2.3.4", port=80)
    with patch.object(runner, "request", side_effect=_fake_request):
        out = probe_target_for_answer(
            runner,
            "First, try to update any city's name to be 'flag'. Then, delete "
            "any city. Once done, search for a city named 'flag' to get the flag.",
            section_code_blocks=["curl http://target/api/cities"],
        )
    assert out is not None
    assert out[0] == "HTB{crud_chain_complete}"


def test_probe_pattern_html_network_tab_discovery():
    """Network-tab question: page references /flag.php, fetch reveals flag."""
    runner = FakeRunner(routes={
        ("GET", "/"): _resp(
            body='<html><script>fetch("/api/flag")</script></html>',
        ),
        ("GET", "/api/flag"): _resp(body="HTB{network_tab}"),
    }, default=_resp(status=404))
    out = probe_target_for_answer(
        runner,
        "Use the Network tab in the browser devtools to find the request to the flag.",
    )
    assert out is not None
    assert out[0] == "HTB{network_tab}"


# ---- extract_curl_commands -------------------------------------------------


def test_extract_curl_strips_obfuscated_prompt():
    """HTB Academy CSS-interleaves the prompt; extractor must skip the prefix."""
    blocks = [
        "S5sherS4stem@htb[/htb]$ curl http://info.cern.ch/index.html",
        "$ curl -X POST -d 'a=1' http://target/api/items",
    ]
    cmds = extract_curl_commands(blocks)
    assert len(cmds) == 2
    assert cmds[0]["method"] == "GET"
    assert cmds[0]["path"] == "/index.html"
    assert cmds[1]["method"] == "POST"
    assert cmds[1]["path"] == "/api/items"
    assert cmds[1]["data"] == "a=1"


def test_extract_curl_handles_headers_cookies_basicauth():
    blocks = [
        "curl -H 'X-Foo: bar' -b 'session=abc' -u admin:admin http://target/api/x",
    ]
    cmds = extract_curl_commands(blocks)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd["headers"]["X-Foo"] == "bar"
    assert cmd["headers"]["Cookie"] == "session=abc"
    assert "Authorization" in cmd["headers"]
    assert cmd["headers"]["Authorization"].startswith("Basic ")


def test_extract_curl_method_inference():
    blocks = [
        "curl -d 'k=v' http://t/api  # implicit POST",
        "curl -X DELETE http://t/api/1",
        "curl -I http://t/  # head request",
    ]
    cmds = extract_curl_commands(blocks)
    methods = [c["method"] for c in cmds]
    assert "POST" in methods
    assert "DELETE" in methods
    assert "HEAD" in methods


# ---- probe_code_blocks_for_flag --------------------------------------------


def test_probe_code_blocks_replays_curls_finds_flag():
    runner = FakeRunner(routes={
        ("POST", "/api/cities"): _resp(body="HTB{from_code_block}"),
    }, default=_resp(status=404))
    blocks = ["curl -X POST -d 'name=flag' http://target/api/cities"]
    out = probe_code_blocks_for_flag(runner, blocks)
    assert out is not None
    assert out[0] == "HTB{from_code_block}"


def test_probe_code_blocks_returns_none_when_no_curl():
    runner = FakeRunner(default=_resp(status=200, body="hello"))
    out = probe_code_blocks_for_flag(runner, ["just plain text", "no commands"])
    assert out is None


# ---- try_form_login --------------------------------------------------------


def test_try_form_login_captures_phpsessid():
    runner = FakeRunner(routes={
        ("POST", "/login.php"): _resp(
            headers={"Set-Cookie": "PHPSESSID=xyz123; Path=/; HttpOnly"},
        ),
    }, default=_resp(status=404))
    cookie = try_form_login(runner, "admin", "admin")
    assert cookie == "PHPSESSID=xyz123"


def test_try_form_login_returns_none_when_no_cookie():
    runner = FakeRunner(default=_resp(status=200, body="login form"))
    cookie = try_form_login(runner, "admin", "admin")
    assert cookie is None


# ---- try_authenticated_search ----------------------------------------------


def test_try_authenticated_search_finds_flag_via_cookie_post():
    runner = FakeRunner(routes={
        ("POST", "/search.php"): _resp(body="result: HTB{auth_search}"),
    }, default=_resp(status=404))
    flag = try_authenticated_search(
        runner, cookie="PHPSESSID=xyz", query="flag",
        json_body=True, endpoint="/search.php",
    )
    assert flag == "HTB{auth_search}"


def test_try_authenticated_search_walks_default_paths():
    runner = FakeRunner(routes={
        ("GET", "/api/search"): _resp(body="HTB{walk_paths}"),
    }, default=_resp(status=404))
    flag = try_authenticated_search(
        runner, cookie="PHPSESSID=xyz", query="flag",
    )
    assert flag == "HTB{walk_paths}"


# ---- extract_html_endpoints / extract_http_method_hint ---------------------


def test_extract_html_endpoints_collects_hrefs_srcs_fetches():
    html = """
    <html>
      <head><script src='/js/app.js'></script></head>
      <body>
        <a href='/page2'>x</a>
        <script>fetch('/api/secret')</script>
        <img src='https://cdn.example.com/img.png'>
        <link href='/favicon.ico'>
      </body>
    </html>
    """
    eps = extract_html_endpoints(html)
    assert "/js/app.js" in eps
    assert "/page2" in eps
    assert "/api/secret" in eps
    # off-host CDN dropped, favicon dropped
    assert all("cdn.example.com" not in e for e in eps)
    assert all("favicon" not in e for e in eps)


def test_extract_http_method_hint_finds_post():
    blocks = [
        "GET /home HTTP/1.1\nHost: x",
        "POST /api/login HTTP/1.1\nHost: x\n\nuser=admin",
    ]
    # First match wins => GET (not what we want for "intercepting");
    # but for THIS shape the section's intercepted block will only have
    # the one captured request. Verify both verbs are recoverable.
    assert extract_http_method_hint(blocks[1:]) == "POST"
    assert extract_http_method_hint(blocks[:1]) == "GET"


def test_extract_http_method_hint_recognizes_method_label():
    blocks = ["Method: PUT\nUrl: /api/x\nBody: ..."]
    assert extract_http_method_hint(blocks) == "PUT"


def test_extract_http_method_hint_returns_none_without_match():
    assert extract_http_method_hint(["plain prose, no verb"]) is None


# ---- extract_json_field ----------------------------------------------------


def test_extract_json_field_direct_key_in_prompt():
    resp = HttpResponse(
        status=200,
        body_json={"city": "Tokyo", "country": "JP"},
    )
    out = extract_json_field(resp, "report the value of the city field")
    assert out == "Tokyo"


def test_extract_json_field_quoted_key_in_prompt():
    resp = HttpResponse(
        status=200,
        body_json={"items": [{"id": 1, "secret": "HTB{nested}"}]},
    )
    out = extract_json_field(resp, "what is the value of the 'secret' field")
    assert out == "HTB{nested}"


# ---- pattern dispatcher: backward-compat / null cases ----------------------


def test_paths_from_code_blocks_filters_by_hint():
    from htbrl.academy.target_runner import _paths_from_code_blocks
    blocks = [
        "curl -X POST -d 'u=a&p=b' http://target/auth/login",
        "curl -X POST http://target/search?q=flag",
        "curl http://target/api/cities",
        "curl http://target/static/logo.png",
    ]
    login_paths = _paths_from_code_blocks(blocks, hint=("login", "auth"))
    assert "/auth/login" in login_paths
    assert "/search" not in login_paths

    search_paths = _paths_from_code_blocks(blocks, hint=("search",))
    assert "/search" in search_paths
    assert "/auth/login" not in search_paths


def test_probe_auth_search_uses_section_endpoints():
    """Pattern 6 should consult code_blocks for the actual login + search
    paths the section teaches, not just guess from defaults."""
    runner = FakeRunner(routes={
        ("POST", "/auth/v2/login"): _resp(headers={"Set-Cookie": "sid=zzz; Path=/"}),
        ("POST", "/v2/find"): _resp(body="hit: HTB{section_taught_paths}"),
    }, default=_resp(status=404))
    out = probe_target_for_answer(
        runner,
        "Authenticate with user 'admin' and password 'admin', then search for 'flag'.",
        section_code_blocks=[
            "curl -X POST -d 'u=a' http://target/auth/v2/login",
            "curl -X POST -H 'Content-Type: application/json' http://target/v2/find -d '{\"q\":\"x\"}'",
        ],
    )
    assert out is not None
    assert out[0] == "HTB{section_taught_paths}"


def test_probe_returns_none_for_unrelated_prompt():
    runner = FakeRunner(default=_resp(status=200, body="ok"))
    out = probe_target_for_answer(runner, "hello world how are you")
    assert out is None
