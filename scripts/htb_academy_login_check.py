r"""Login + screenshot validator for HTB Academy (Phase 5b real-account bring-up).

Reads credentials ONLY from env vars; never logs the password; screenshots go
under ``.local/`` which the repo's .gitignore protects. Use this to validate
the Playwright selectors against the real DOM before pointing the orchestrator
at the account.

Required env vars:
  HTB_ACADEMY_USER
  HTB_ACADEMY_PASS
Optional:
  HTBRL_ACADEMY_COOKIES  - path to a cookie file (read+save). Default
                            .local/academy_cookies.json.
  HTBRL_ACADEMY_HEADLESS - "0" to launch a visible browser (debugging),
                            anything else = headless (default)

This version is a low-level Playwright probe that takes screenshots at every
stage so we can see exactly which page the SSO flow lands us on. The
PlaywrightAcademySession's higher-level helpers come AFTER we know which
selectors actually work.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# Pin Playwright to a project-local browser cache so it's independent of the
# shell user's %LOCALAPPDATA% (admin vs non-admin PowerShell, multiple Windows
# accounts, etc. all resolve identically). Must be set BEFORE importing
# playwright.sync_api.
_PROJECT_BROWSERS = Path(__file__).resolve().parent.parent / ".local" / "playwright-browsers"
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and _PROJECT_BROWSERS.exists():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_PROJECT_BROWSERS)

from playwright.sync_api import sync_playwright


def _redacted_email(s: str) -> str:
    if "@" not in s:
        return "***"
    user, _, host = s.partition("@")
    if len(user) <= 2:
        return f"{user[:1]}***@{host}"
    return f"{user[:2]}***@{host}"


def _run_cdp_attach(cdp_endpoint: str, cookies_path: Path, shots_dir: Path) -> int:
    """CDP-attach mode: connect to a Chrome the user launched themselves.

    Workflow:
      1. User runs scripts/start_chrome_for_htb.ps1 - launches Chrome with
         --remote-debugging-port=9222 + a dedicated profile dir.
      2. User logs in to HTB normally (Google OAuth works because Chrome is
         user-launched, no Playwright automation markers visible to Google or
         Cloudflare).
      3. User runs this script with HTBRL_ACADEMY_CDP set.
      4. We attach via CDP, find the academy tab, poll until it's on
         academy.hackthebox.com/app/*, save cookies, exit.
    """
    import time

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_endpoint)
        except Exception as exc:
            print(f"ERROR: could not connect to {cdp_endpoint}: {exc}", file=sys.stderr)
            print("Did you run scripts/start_chrome_for_htb.ps1 first?", file=sys.stderr)
            return 5

        if not browser.contexts:
            print("ERROR: connected but no browser contexts. Open a tab in Chrome "
                  "and navigate to https://academy.hackthebox.com/app/dashboard.",
                  file=sys.stderr)
            return 5

        ctx = browser.contexts[0]
        # Find or create an academy page.
        academy_page = None
        for page in ctx.pages:
            try:
                if "hackthebox.com" in page.url:
                    academy_page = page
                    break
            except Exception:
                continue
        if academy_page is None:
            academy_page = ctx.new_page()
            academy_page.goto("https://academy.hackthebox.com/app/dashboard")

        print(f"[cdp] attached. Current URL: {academy_page.url}")
        print("[cdp] If you're not logged in yet, log in NOW in the Chrome window.")
        print("[cdp] Polling until you reach academy.hackthebox.com/app/*  (10 min cap).")

        deadline = time.time() + 600.0
        last_url = ""
        authed = False
        while time.time() < deadline:
            try:
                cur = academy_page.url
            except Exception:
                cur = ""
            if cur != last_url:
                print(f"[cdp]   nav -> {cur}")
                last_url = cur
            on_academy_app = (
                "academy.hackthebox.com/app/" in cur
                or "academy.hackthebox.com/dashboard" in cur
            )
            if on_academy_app:
                # Probe for an authenticated marker. If still on a login screen
                # within /app/* (unlikely), keep polling.
                try:
                    has_login_form = academy_page.locator(
                        "#loginEmail, input[name='email'][type='email']"
                    ).count() > 0
                except Exception:
                    has_login_form = False
                if not has_login_form:
                    authed = True
                    print(f"[cdp] authenticated. URL={cur}")
                    break
            time.sleep(1.0)

        if not authed:
            print("[cdp] timed out waiting for academy/app URL. Re-run after you "
                  "complete login.", file=sys.stderr)

        # Save the storage_state from the connected context. NB: connect_over_cdp's
        # default context can save cookies via ctx.storage_state.
        try:
            state = ctx.storage_state()
            cookies_path.write_text(json.dumps(state), encoding="utf-8")
            print(f"[cdp] saved cookies -> {cookies_path}")
        except Exception as exc:
            print(f"[cdp] saving cookies failed: {exc}", file=sys.stderr)

        # Don't close the user's browser - just disconnect.
        try:
            _shot(academy_page, shots_dir / "06_cdp_after_login.png", "cdp-after-login")
        except Exception:
            pass

    return 0 if authed else 4


def _find_system_chrome() -> str | None:
    """Auto-detect a system Chrome (or Edge) install on Windows."""
    for candidate in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _shot(page, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(dest), full_page=True)
        print(f"[check] saved {dest}  (URL: {page.url})  [{label}]")
    except Exception as exc:
        print(f"[check] screenshot failed at {label}: {exc}")


def main(argv: list[str] | None = None) -> int:
    no_login = os.environ.get("HTBRL_ACADEMY_NO_LOGIN", "0") == "1"
    manual_login = os.environ.get("HTBRL_ACADEMY_MANUAL_LOGIN", "0") == "1"
    cdp_endpoint = os.environ.get("HTBRL_ACADEMY_CDP", "")
    user = os.environ.get("HTB_ACADEMY_USER", "")
    pw = os.environ.get("HTB_ACADEMY_PASS", "")
    if not no_login and not manual_login and not cdp_endpoint and not (user and pw):
        print("ERROR: no auth method configured. Choose one:", file=sys.stderr)
        print("  HTBRL_ACADEMY_CDP=http://127.0.0.1:9222   (recommended; "
              "attach to user-launched Chrome, no auth via Playwright)", file=sys.stderr)
        print("  HTBRL_ACADEMY_MANUAL_LOGIN=1              (log in by hand in "
              "the Playwright window)", file=sys.stderr)
        print("  HTB_ACADEMY_USER + HTB_ACADEMY_PASS       (auto-fill the form; "
              "Cloudflare typically blocks)", file=sys.stderr)
        print("  HTBRL_ACADEMY_NO_LOGIN=1                  (chromium-launch smoke "
              "test only)", file=sys.stderr)
        return 2

    headless = os.environ.get("HTBRL_ACADEMY_HEADLESS", "1") != "0"
    cookies_path = Path(os.environ.get("HTBRL_ACADEMY_COOKIES", ".local/academy_cookies.json"))
    shots_dir = Path(".local/screenshots")
    shots_dir.mkdir(parents=True, exist_ok=True)
    cookies_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[check] user            : {_redacted_email(user)}")
    print(f"[check] cookies         : {cookies_path}")
    print(f"[check] screenshots dir : {shots_dir}")
    print(f"[check] headless        : {headless}")
    if cdp_endpoint:
        print(f"[check] mode            : CDP attach -> {cdp_endpoint}")
        return _run_cdp_attach(cdp_endpoint, cookies_path, shots_dir)

    # Optional: use the user's installed Chrome instead of Playwright's chromium
    # build (sometimes resolves Cloudflare false positives, but we do NOT layer
    # bot-evasion patches on top - if Cloudflare blocks us, the user logs in
    # manually via HTBRL_ACADEMY_MANUAL_LOGIN=1).
    chrome_path = os.environ.get("HTBRL_CHROME_PATH") or _find_system_chrome()
    if chrome_path:
        print(f"[check] using system Chrome at {chrome_path}")

    if manual_login:
        print("[check] HTBRL_ACADEMY_MANUAL_LOGIN=1 -> you log in manually in the "
              "browser window; the script will save cookies once you reach the dashboard")

    with sync_playwright() as pw_runtime:
        launch_kwargs: dict[str, Any] = {"headless": headless, "slow_mo": 100}
        if chrome_path:
            launch_kwargs["executable_path"] = chrome_path
        browser = pw_runtime.chromium.launch(**launch_kwargs)
        ctx_kwargs: dict[str, Any] = {}
        if cookies_path.exists():
            ctx_kwargs["storage_state"] = str(cookies_path)
            print("[check] using existing cookie state")
        ctx = browser.new_context(**ctx_kwargs)
        ctx.set_default_navigation_timeout(45_000)
        ctx.set_default_timeout(15_000)
        page = ctx.new_page()

        # Stage 1: navigate to the academy dashboard. The site is a SPA that
        # checks auth via API and EITHER renders the dashboard (if our cookies
        # are valid) OR redirects to account.hackthebox.com/login.
        print("[check] step 1: GET /app/dashboard ...")
        try:
            page.goto("https://academy.hackthebox.com/app/dashboard")
        except Exception as exc:
            print(f"[check] goto raised: {exc}")
        # Wait for the SPA to settle: either the login form appears (we need
        # to log in) or the academy chrome renders (we're authenticated).
        try:
            page.wait_for_function(
                "() => document.querySelector('#loginEmail') || "
                "document.querySelector('.htb-user-avatar') || "
                "document.querySelector('[data-test=\"user-menu\"]') || "
                "document.querySelector('aside') ||  /* academy nav rail */ "
                "document.querySelector('.htb-app-bar')",
                timeout=20_000,
            )
        except Exception:
            print("[check]   SPA never resolved; continuing with whatever's on screen")
        _shot(page, shots_dir / "01_after_initial_get.png", "post-initial-get")
        print(f"[check]   landed URL: {page.url}")

        if no_login:
            print("[check] HTBRL_ACADEMY_NO_LOGIN=1 -> chromium-launch smoke OK; exiting")
            ctx.close()
            browser.close()
            return 0

        # Stage 2: detect via DOM, not URL. If the login form selector is on
        # the page, we need to authenticate; otherwise we're already in.
        login_form_present = page.locator("#loginEmail, input[name='email']").count() > 0
        print(f"[check] step 2: login form present in DOM = {login_form_present}")

        if login_form_present and manual_login:
            # Hand the wheel to the human. We poll until the URL is back on
            # academy.hackthebox.com (or app.hackthebox.com) AND no login form
            # is present - this catches users who go through Google/Github
            # OAuth, where the form-gone signal alone fires too early
            # (browser is mid-OAuth at accounts.google.com).
            print("=" * 64)
            print(" MANUAL LOGIN MODE")
            print(" 1. The browser window in front of you is on the HTB SSO page.")
            print(" 2. Log in normally - email/password OR Sign in with Google /")
            print("    Github / LinkedIn. Solve any Cloudflare challenge that")
            print("    appears.")
            print(" 3. Land on the HTB academy dashboard (URL should contain")
            print("    'academy.hackthebox.com/app/'). The script will detect")
            print("    that, save cookies, and exit on its own.")
            print(" 4. Timeout: 10 minutes. Ctrl+C to abort.")
            print("=" * 64)

            import time
            deadline = time.time() + 600.0
            authed = False
            last_url_logged = ""
            while time.time() < deadline:
                try:
                    cur = page.url
                except Exception:
                    cur = ""
                if cur != last_url_logged:
                    print(f"[check]   nav -> {cur}")
                    last_url_logged = cur
                # Authentication signal: we're back on academy + no login form.
                on_academy_app = (
                    "academy.hackthebox.com/app" in cur
                    or "app.hackthebox.com" in cur
                    or "academy.hackthebox.com/dashboard" in cur
                )
                form_gone = page.locator(
                    "#loginEmail, input[name='email'][type='email']"
                ).count() == 0
                if on_academy_app and form_gone:
                    authed = True
                    print(f"[check]   authed=True, URL={cur}")
                    break
                time.sleep(1.0)

            if not authed:
                print("[check] manual login timed out. The browser may still be in "
                      "an intermediate OAuth state. Re-run when you've completed "
                      "the login.")

            try:
                ctx.storage_state(path=str(cookies_path))
                print(f"[check] saved cookies -> {cookies_path}")
            except Exception as exc:
                print(f"[check] saving cookies failed: {exc}")
            _shot(page, shots_dir / "06_after_manual_login.png", "after-manual-login")
            ctx.close()
            browser.close()
            return 0 if authed else 4

        if login_form_present:
            print(f"[check] step 2: SSO login form expected at {page.url}")

            # Real selectors (captured from the live DOM at
            # https://account.hackthebox.com/login). The probe still tries a
            # small fallback list so future redesigns don't require a code
            # change to discover the new IDs.
            email_sels = ["#loginEmail", "input[name='email'][type='email']", "input[type='email']"]
            pw_sels = ["#loginPassword", "input[name='password']", "input[type='password']"]
            submit_sels = [
                "form#loginForm button[type='submit']",
                "button:has-text('Sign in')",
                "button[type='submit']",
            ]

            # Cloudflare Turnstile: the submit button has the `disabled`
            # attribute until the challenge token is acquired. We poll for
            # the enabled state up to 30 s after filling.
            submit_enabled_sels = [
                "form#loginForm button[type='submit']:not([disabled])",
                "button[type='submit']:not([disabled])",
            ]

            # Vuetify mounts the form via JS, so DOMContentLoaded fires before
            # the inputs exist. Wait up to 20 s for the first selector that
            # matches anything in our fallback lists to appear.
            try:
                page.wait_for_selector(
                    ", ".join(email_sels), timeout=20_000, state="attached",
                )
                print("[check]   form mounted")
            except Exception as exc:
                print(f"[check]   WARNING: timed out waiting for form to mount: {exc}")

            email_loc = pw_loc = submit_loc = None
            for sel in email_sels:
                if page.locator(sel).count() > 0:
                    email_loc = page.locator(sel).first
                    print(f"[check]   email field selector: {sel!r}")
                    break
            for sel in pw_sels:
                if page.locator(sel).count() > 0:
                    pw_loc = page.locator(sel).first
                    print(f"[check]   password field selector: {sel!r}")
                    break
            for sel in submit_sels:
                if page.locator(sel).count() > 0:
                    submit_loc = page.locator(sel).first
                    print(f"[check]   submit button selector: {sel!r}")
                    break

            if not (email_loc and pw_loc and submit_loc):
                print("[check] could not find one of email/password/submit; aborting")
                _shot(page, shots_dir / "02_login_form_unfound.png", "login-form-unfound")
                ctx.close()
                browser.close()
                return 3

            print("[check] step 3: filling form...")
            email_loc.fill(user)
            pw_loc.fill(pw)
            _shot(page, shots_dir / "02_form_filled.png", "form-filled")

            # Wait for Cloudflare Turnstile to enable the submit button.
            print("[check] waiting up to 30 s for Cloudflare Turnstile to enable submit...")
            submit_enabled = False
            for sel in submit_enabled_sels:
                try:
                    page.wait_for_selector(sel, timeout=30_000)
                    submit_enabled = True
                    print(f"[check]   submit became enabled (selector: {sel!r})")
                    break
                except Exception:
                    continue
            if not submit_enabled:
                print("[check] WARNING: submit never became enabled. Cloudflare Turnstile "
                      "may be blocking automation. If you're running headless, try "
                      "HTBRL_ACADEMY_HEADLESS=0 so the challenge can resolve naturally.")

            submit_loc.click()
            try:
                page.wait_for_url(
                    lambda u: "login" not in u.lower() and "account.hackthebox.com" not in u.lower(),
                    timeout=45_000,
                )
            except Exception:
                pass
            _shot(page, shots_dir / "03_after_submit.png", "after-submit")

        else:
            print("[check] no login challenge encountered; cookies likely valid")

        # Stage 3: navigate to /app/library/modules (the user's link) and capture.
        print("[check] step 4: GET /app/library/modules ...")
        try:
            page.goto("https://academy.hackthebox.com/app/library/modules")
            page.wait_for_load_state("networkidle")
        except Exception as exc:
            print(f"[check] modules-page load: {exc}")
        _shot(page, shots_dir / "04_modules_page.png", "modules-page")

        # Stage 4: paths page (the user's other link).
        print("[check] step 5: GET /app/library/paths ...")
        try:
            page.goto("https://academy.hackthebox.com/app/library/paths")
            page.wait_for_load_state("networkidle")
        except Exception as exc:
            print(f"[check] paths-page load: {exc}")
        _shot(page, shots_dir / "05_paths_page.png", "paths-page")

        # Save cookies for next run.
        try:
            ctx.storage_state(path=str(cookies_path))
            print(f"[check] saved cookies -> {cookies_path}")
        except Exception as exc:
            print(f"[check] saving cookies failed: {exc}")

        ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
