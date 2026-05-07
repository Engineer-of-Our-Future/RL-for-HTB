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


def _shot(page, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(dest), full_page=True)
        print(f"[check] saved {dest}  (URL: {page.url})  [{label}]")
    except Exception as exc:
        print(f"[check] screenshot failed at {label}: {exc}")


def main(argv: list[str] | None = None) -> int:
    no_login = os.environ.get("HTBRL_ACADEMY_NO_LOGIN", "0") == "1"
    user = os.environ.get("HTB_ACADEMY_USER", "")
    pw = os.environ.get("HTB_ACADEMY_PASS", "")
    if not no_login and not (user and pw):
        print("ERROR: HTB_ACADEMY_USER and HTB_ACADEMY_PASS env vars are required.",
              file=sys.stderr)
        print("       (set HTBRL_ACADEMY_NO_LOGIN=1 for a chromium-launch smoke test)",
              file=sys.stderr)
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

    with sync_playwright() as pw_runtime:
        browser = pw_runtime.chromium.launch(headless=headless, slow_mo=100)
        ctx_kwargs: dict[str, Any] = {}
        if cookies_path.exists():
            ctx_kwargs["storage_state"] = str(cookies_path)
            print("[check] using existing cookie state")
        ctx = browser.new_context(**ctx_kwargs)
        ctx.set_default_navigation_timeout(45_000)
        ctx.set_default_timeout(15_000)
        page = ctx.new_page()

        # Stage 1: navigate to the academy dashboard. If we're already logged in
        # (cookies), this is a fast load. Otherwise the SSO flow kicks in.
        print("[check] step 1: GET /app/dashboard ...")
        try:
            page.goto("https://academy.hackthebox.com/app/dashboard")
        except Exception as exc:
            print(f"[check] goto raised: {exc}")
        page.wait_for_load_state("domcontentloaded")
        _shot(page, shots_dir / "01_after_initial_get.png", "post-initial-get")

        if no_login:
            print("[check] HTBRL_ACADEMY_NO_LOGIN=1 -> chromium-launch smoke OK; exiting")
            ctx.close()
            browser.close()
            return 0

        # Stage 2: if URL contains 'login' or 'sso' / 'account', we need creds.
        url = page.url.lower()
        if any(s in url for s in ("login", "sso", "account.hackthebox.com")):
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
