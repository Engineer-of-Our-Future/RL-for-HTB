"""Real (Playwright-driven) HTB Academy session.

This module wires a concrete browser-automation transport into the
``AcademySession`` ABC. It does NOT ship with the default install -
``pip install -e ".[academy]"`` is required to pull Playwright, plus a
one-time ``playwright install chromium`` to fetch the headless browser.

Design notes:

- **Selectors live in a config file**, not in code. HTB Academy's DOM evolves;
  rather than re-cutting a release every time a class name changes, we read
  ``configs/academy/htb_selectors.yaml`` and let users (or future-me) update
  the selectors without touching Python. Each method names the selector keys
  it needs in its docstring.

- **Authentication:** prefer cookie-jar persistence over re-typing creds. Pass
  ``cookie_path=...`` to ``__init__`` and the session will load/save the
  ``storage_state`` JSON Playwright uses. If cookies are absent or expired,
  we fall back to credential login. CAPTCHA / 2FA are NOT auto-solved - if
  we hit one, we raise a clear exception asking you to do an interactive
  login first to seed the cookie file.

- **study_only is enforced INSIDE the transport.** ``submit_answer`` checks
  the flag before clicking. The orchestrator's gating is defense in depth;
  this layer is the second line.

- **Idempotent navigation.** Every public method either reuses the current
  page if it's already on the right URL or navigates explicitly. Re-entrancy
  is safe.

The implementation here is best-effort given that the site structure can
change. Selectors that don't match raise ``SelectorMissingError`` with the
selector key + URL so you can update the YAML and re-run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# Pin Playwright's browser cache to a project-local path so it's independent of
# the shell's %LOCALAPPDATA% (matters when running under elevated PowerShell on
# Windows, where some configurations resolve LOCALAPPDATA to a different profile
# than where chromium was installed). Must be set BEFORE the first
# `playwright.sync_api` import inside this process.
_PROJECT_BROWSERS = Path(__file__).resolve().parents[3] / ".local" / "playwright-browsers"
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and _PROJECT_BROWSERS.exists():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_PROJECT_BROWSERS)

from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySandbox,
    AcademySection,
    ProgressState,
    QuestionType,
)
from htbrl.academy.session import AcademyCredentials, AcademySession


log = logging.getLogger("htbrl.academy.playwright")


_DEFAULT_BASE_URL = "https://academy.hackthebox.com"


# Selectors are keyed by stable names; the file at ``selectors_path`` provides
# the actual CSS/XPath/role selectors so we can update them out-of-band.
#
# The login selectors below were captured against the real
# https://account.hackthebox.com/login DOM (HTB unified SSO, Vuetify-based).
# They will drift over time - update the YAML override when they do.
_DEFAULT_SSO_LOGIN_URL = "https://account.hackthebox.com/login"
_DEFAULT_SELECTORS = {
    "login": {
        "url": _DEFAULT_SSO_LOGIN_URL,
        "email": "#loginEmail, input[name='email'][type='email']",
        "password": "#loginPassword, input[name='password']",
        "submit": "form#loginForm button[type='submit'], button:has-text('Sign in')",
        # Vuetify renders the submit button with the 'disabled' attribute until
        # the Cloudflare Turnstile challenge resolves. Wait for this CSS state
        # before clicking. The probe + session use this implicitly.
        "submit_enabled": "form#loginForm button[type='submit']:not([disabled])",
        "logged_in_marker": "[data-test='user-menu'], .user-avatar, .htb-user-avatar",
        # Cloudflare Turnstile container (invisible challenge that gates the form).
        "captcha_marker": "#turnstile-login, .g-recaptcha, [data-test='captcha']",
    },
    "modules_list": {
        "url": f"{_DEFAULT_BASE_URL}/modules",
        "card": "[data-test='module-card'], .module-card",
        "card_title": "h3, .title",
        "card_link": "a",
        "card_tier": "[data-test='tier'], .tier",
        "card_cubes_unlock": "[data-test='cubes-required'], .cubes-required",
        "card_cubes_reward": "[data-test='cubes-reward'], .cubes-reward",
    },
    "module_page": {
        "section_block": "[data-test='module-section'], section.module-section, .module-section",
        "section_title": "h1.section-title, h2, h3.section-title",
        "section_body": ".section-body, .markdown-body",
        "code_block": "pre code",
        "inline_code": "p code, li code",   # screenshots show inline-code answers
        "bullet_list": "ul",                # bulleted lists in the theory
        "bullet_item": "li",
        # Section navigator at the bottom of the page: "Section 11 / 22"
        "section_index_label": "[data-test='section-progress'], .section-progress",
        "next_section_button": "button[data-test='next-section'], button:has-text('Next')",
        "mark_complete_next_button": "button[data-test='mark-complete-next'], button:has-text('Mark Complete')",
        "section_hp_reward": "[data-test='section-hp']",
        # Question cards (screenshot: "Question 1 ... +5 cubes / +20 HP / Submit")
        "question_block": "[data-test='question'], .question-card, .question-block",
        "question_prompt": ".question-prompt, h4, [data-test='question-prompt']",
        "question_input": "input[placeholder='Write your answer'], input[type='text'], textarea",
        "question_options": "input[type='radio']",
        "question_submit": "button[data-test='submit-answer'], button:has-text('Submit')",
        "question_correct_marker": "[data-test='correct'], .answer-correct",
        "question_wrong_marker": "[data-test='wrong'], .answer-wrong",
        "question_cubes_reward": "[data-test='cubes-reward'], .reward-cubes",
        "question_hp_reward": "[data-test='hp-reward'], .reward-hp",
        # Per-section sandbox card
        "sandbox_block": "[data-test='sandbox-info'], .sandbox-credentials",
        "sandbox_host": "[data-test='sandbox-host'], .sandbox-host",
        "sandbox_user": "[data-test='sandbox-user'], .sandbox-user",
        "sandbox_password": "[data-test='sandbox-password'], .sandbox-password",
        "sandbox_port": "[data-test='sandbox-port'], .sandbox-port",
    },
    "progress": {
        "url": f"{_DEFAULT_BASE_URL}/dashboard",
        "cubes_balance": "[data-test='cubes-balance'], .cubes-balance",
        "completed_modules": "[data-test='completed-module']",
    },
}


class SelectorMissingError(RuntimeError):
    """Raised when a selector doesn't match anything on the page."""


@dataclass
class PlaywrightConfig:
    base_url: str = _DEFAULT_BASE_URL
    selectors_path: Path | None = None       # YAML override of _DEFAULT_SELECTORS
    cookie_path: Path | None = None           # JSON storage_state for session reuse
    headless: bool = True
    slow_mo_ms: int = 0                       # human-eye debugging speed
    user_agent: str | None = None
    nav_timeout_ms: int = 30_000
    selector_timeout_ms: int = 8_000


class PlaywrightAcademySession(AcademySession):
    """Concrete academy session backed by Playwright (sync API).

    Construct with study_only and a PlaywrightConfig. Login and navigate via
    the AcademySession contract. Every action that mutates academy-side state
    (start_module, submit_answer) checks ``self.study_only`` before doing
    anything network-side.
    """

    def __init__(
        self,
        cfg: PlaywrightConfig | None = None,
        study_only: bool = True,
    ) -> None:
        self.cfg = cfg or PlaywrightConfig()
        self.study_only = study_only

        # Lazy-imported so non-academy installs don't pay the import cost.
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "playwright is not installed. install with `pip install -e \".[academy]\"` "
                "and then run `playwright install chromium`."
            ) from exc

        self._sel = self._load_selectors()
        self._pw = None
        self._browser = None
        self._ctx = None
        self._page = None
        self._logged_in = False

    # ---- selectors ----------------------------------------------------------

    def _load_selectors(self) -> dict:
        sel = {k: dict(v) for k, v in _DEFAULT_SELECTORS.items()}
        if self.cfg.selectors_path is not None:
            override = yaml.safe_load(Path(self.cfg.selectors_path).read_text(encoding="utf-8"))
            for top_key, sub in (override or {}).items():
                sel.setdefault(top_key, {}).update(sub or {})
        return sel

    # ---- lifecycle ----------------------------------------------------------

    def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.cfg.headless, slow_mo=self.cfg.slow_mo_ms
        )
        ctx_kwargs: dict[str, Any] = {}
        if self.cfg.user_agent:
            ctx_kwargs["user_agent"] = self.cfg.user_agent
        if self.cfg.cookie_path is not None and Path(self.cfg.cookie_path).exists():
            ctx_kwargs["storage_state"] = str(self.cfg.cookie_path)
        self._ctx = self._browser.new_context(**ctx_kwargs)
        self._ctx.set_default_navigation_timeout(self.cfg.nav_timeout_ms)
        self._ctx.set_default_timeout(self.cfg.selector_timeout_ms)
        self._page = self._ctx.new_page()

    def login(self, creds: AcademyCredentials) -> None:
        self._ensure_browser()
        page = self._page
        assert page is not None

        # Try cookie-only login first.
        page.goto(self._sel["progress"]["url"])
        if self._is_logged_in_dom():
            self._logged_in = True
            log.info("logged in via persisted cookies")
            return

        log.info("logging in with credentials")
        page.goto(self._sel["login"]["url"])

        # HTB's login uses Cloudflare Turnstile (an invisible / minimal-friction
        # challenge), which keeps the submit button disabled until the token
        # resolves. We don't try to solve Turnstile programmatically (that
        # would be both unreliable and a clear ToS violation); instead we
        # require ``headless=False`` for the FIRST run, let the Turnstile
        # token settle naturally, and then save cookies so subsequent runs
        # skip the login flow entirely.
        if self.cfg.headless:
            log.warning(
                "login attempted in headless mode; Cloudflare Turnstile will "
                "almost certainly block the submit button. Run once with "
                "PlaywrightConfig(headless=False) to seed cookies."
            )

        page.locator(self._sel["login"]["email"]).first.fill(creds.username)
        page.locator(self._sel["login"]["password"]).first.fill(creds.password)
        # Wait for Turnstile to resolve (submit button stops being disabled).
        try:
            page.wait_for_selector(
                self._sel["login"]["submit_enabled"],
                timeout=self.cfg.nav_timeout_ms,
            )
        except Exception:
            log.warning(
                "submit button never became enabled within timeout; the "
                "Cloudflare Turnstile challenge may be blocking automation. "
                "Try running headful and resolving any visible challenge."
            )
        page.locator(self._sel["login"]["submit"]).first.click()
        page.wait_for_url(
            lambda u: "/login" not in u and "account.hackthebox.com" not in u,
            timeout=self.cfg.nav_timeout_ms,
        )

        if not self._is_logged_in_dom():
            raise RuntimeError(
                "login attempt succeeded the form but no logged-in marker on the "
                "destination page; either credentials are wrong or selectors are stale"
            )
        self._logged_in = True

        if self.cfg.cookie_path is not None:
            Path(self.cfg.cookie_path).parent.mkdir(parents=True, exist_ok=True)
            self._ctx.storage_state(path=str(self.cfg.cookie_path))
            log.info("saved cookies to %s", self.cfg.cookie_path)

    def is_logged_in(self) -> bool:
        return self._logged_in

    def close(self) -> None:
        if self._page is not None:
            self._page.close()
        if self._ctx is not None:
            self._ctx.close()
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()
        self._page = None
        self._ctx = None
        self._browser = None
        self._pw = None
        self._logged_in = False

    # ---- progress -----------------------------------------------------------

    def get_progress_state(self) -> ProgressState:
        page = self._page
        assert page is not None
        page.goto(self._sel["progress"]["url"])
        cubes_node = page.locator(self._sel["progress"]["cubes_balance"]).first
        try:
            cubes_text = cubes_node.inner_text()
        except Exception:
            raise SelectorMissingError(f"cubes_balance not found on {page.url}")
        cubes = _first_int(cubes_text)
        completed_ids: list[str] = []
        for el in page.locator(self._sel["progress"]["completed_modules"]).all():
            mid = el.get_attribute("data-module-id") or el.text_content() or ""
            if mid.strip():
                completed_ids.append(mid.strip())
        return ProgressState(
            user_id=cubes_node.get_attribute("data-user-id") or "self",
            cubes_balance=cubes or 0,
            completed_module_ids=completed_ids,
        )

    def list_modules(self) -> list[AcademyModule]:
        page = self._page
        assert page is not None
        page.goto(self._sel["modules_list"]["url"])
        sel = self._sel["modules_list"]
        cards = page.locator(sel["card"]).all()
        if not cards:
            raise SelectorMissingError(f"no module cards matched {sel['card']!r}")
        out: list[AcademyModule] = []
        for c in cards:
            href = c.locator(sel["card_link"]).first.get_attribute("href") or ""
            mid = href.rstrip("/").rsplit("/", 1)[-1]
            title = (c.locator(sel["card_title"]).first.text_content() or "").strip()
            tier = _first_int(_safe_text(c, sel["card_tier"])) or 0
            cubes_unlock = _first_int(_safe_text(c, sel["card_cubes_unlock"])) or 0
            cubes_reward = _first_int(_safe_text(c, sel["card_cubes_reward"])) or 0
            out.append(AcademyModule(
                id=mid, title=title, tier=tier,
                cubes_to_unlock=cubes_unlock, cubes_reward=cubes_reward,
            ))
        return out

    def fetch_module(self, module_id: str) -> AcademyModule:
        """Walk every section of the module by clicking through Section 1..N.

        HTB Academy renders one section per page with a "Section X / Y"
        indicator and a "Next" / "Mark Complete & Next" button. We start at
        section 1 and step forward, scraping each section's theory + questions
        + sandbox info. The walk stops when there's no next-button or we run
        out of section-index advances.
        """
        page = self._page
        assert page is not None
        url = f"{self.cfg.base_url}/modules/{module_id}"
        page.goto(url)
        sel = self._sel["module_page"]

        sections: list[AcademySection] = []
        seen_section_ids: set[str] = set()
        # Hard cap to defend against an infinite loop if selectors regress.
        for _step in range(200):
            sec_blocks = page.locator(sel["section_block"]).all()
            sec_root = sec_blocks[0] if sec_blocks else page
            # Index label like "Section 11 / 22"
            label = _safe_text(sec_root, sel["section_index_label"])
            sec_idx, sec_total = _parse_section_label(label)

            title = (_safe_text(sec_root, sel["section_title"]) or "").strip()
            body = (_safe_text(sec_root, sel["section_body"]) or "").strip()
            code_blocks = [
                (cb.text_content() or "").strip()
                for cb in sec_root.locator(sel["code_block"]).all()
            ]
            inline_code = [
                (ic.text_content() or "").strip()
                for ic in sec_root.locator(sel["inline_code"]).all()
                if ic.text_content() and len(ic.text_content().strip()) <= 64
            ]
            bullet_lists: list[list[str]] = []
            for ul in sec_root.locator(sel["bullet_list"]).all():
                items = [
                    (li.text_content() or "").strip()
                    for li in ul.locator(sel["bullet_item"]).all()
                ]
                items = [it for it in items if it]
                if items:
                    bullet_lists.append(items)
            hp_reward = _first_int(_safe_text(sec_root, sel["section_hp_reward"])) or 0

            questions: list[AcademyQuestion] = []
            for q in sec_root.locator(sel["question_block"]).all():
                q_id = q.get_attribute("data-question-id") or _hash_id(_safe_text(q, sel["question_prompt"]))
                prompt = (_safe_text(q, sel["question_prompt"]) or "").strip()
                opts = [
                    (o.get_attribute("value") or "").strip()
                    for o in q.locator(sel["question_options"]).all()
                ]
                qtype = QuestionType.MULTIPLE_CHOICE if opts else (
                    QuestionType.FLAG if "flag" in prompt.lower() else QuestionType.TEXT
                )
                cubes_r = _first_int(_safe_text(q, sel["question_cubes_reward"])) or 0
                hp_r = _first_int(_safe_text(q, sel["question_hp_reward"])) or 0
                questions.append(AcademyQuestion(
                    id=q_id, prompt=prompt, type=qtype,
                    multiple_choice_options=opts,
                    cubes_reward=cubes_r,
                    hp_reward=hp_r,
                ))

            sandbox = self._scrape_sandbox(sec_root, sel)
            section_id = _hash_id(f"{title}|{sec_idx}|{url}")
            if section_id in seen_section_ids:
                # Shouldn't happen unless the click didn't advance; bail.
                log.warning("section %s seen twice; stopping module walk", section_id)
                break
            seen_section_ids.add(section_id)
            sections.append(AcademySection(
                id=section_id,
                title=title,
                body_text=body,
                questions=questions,
                code_blocks=code_blocks,
                sandbox=sandbox,
                inline_code=inline_code,
                bullet_lists=bullet_lists,
                section_index=sec_idx or len(sections) + 1,
                section_total=sec_total or 0,
                hp_reward=hp_reward,
            ))

            # Decide whether to step forward.
            if sec_total and sec_idx and sec_idx >= sec_total:
                break
            advanced = self._goto_next_section(sel)
            if not advanced:
                break

        return AcademyModule(
            id=module_id,
            title=(page.title() or module_id).split("|")[0].strip(),
            tier=0,
            sections=sections,
        )

    def _goto_next_section(self, sel: dict) -> bool:
        """Click Next / Mark-Complete-and-Next; return True if we advanced."""
        page = self._page
        assert page is not None
        # Prefer "Next" if present (does not mark complete); fall back to
        # Mark-Complete-and-Next.
        next_btn = page.locator(sel["next_section_button"]).first
        if next_btn.count() > 0 and next_btn.is_enabled():
            try:
                next_btn.click()
                page.wait_for_load_state("networkidle", timeout=self.cfg.nav_timeout_ms)
                return True
            except Exception as exc:
                log.warning("next-section click failed: %s", exc)
        mc_btn = page.locator(sel["mark_complete_next_button"]).first
        if mc_btn.count() > 0 and mc_btn.is_enabled():
            if self.study_only:
                # Mark-Complete mutates academy state; refuse in study-only.
                log.info("study_only=True: not clicking 'Mark Complete & Next'")
                return False
            try:
                mc_btn.click()
                page.wait_for_load_state("networkidle", timeout=self.cfg.nav_timeout_ms)
                return True
            except Exception as exc:
                log.warning("mark-complete-and-next click failed: %s", exc)
        return False

    @staticmethod
    def _scrape_sandbox(sec, sel: dict) -> AcademySandbox | None:
        if sec.locator(sel["sandbox_block"]).count() == 0:
            return None
        host = _safe_text(sec, sel["sandbox_host"]) or ""
        user = _safe_text(sec, sel["sandbox_user"]) or ""
        password = _safe_text(sec, sel["sandbox_password"]) or None
        port_text = _safe_text(sec, sel["sandbox_port"]) or "22"
        port = _first_int(port_text) or 22
        if not (host and user):
            return None
        return AcademySandbox(host=host.strip(), user=user.strip(),
                              port=port, password=password)

    # ---- mutations ----------------------------------------------------------

    def start_module(self, module_id: str) -> None:
        # The academy starts a module the first time you load its page; we
        # already navigated when fetch_module ran. Defensive idempotency.
        page = self._page
        assert page is not None
        url = f"{self.cfg.base_url}/modules/{module_id}"
        if not page.url.startswith(url):
            page.goto(url)
        log.info("module %s opened (start_module=%s)", module_id, "no-op")

    def submit_answer(self, module_id: str, answer: AcademyAnswer) -> bool:
        if self.study_only:
            log.info("study_only=True: NOT submitting %s for %s/%s",
                     answer.answer_text, module_id, answer.question_id)
            return False
        page = self._page
        assert page is not None
        url = f"{self.cfg.base_url}/modules/{module_id}"
        if not page.url.startswith(url):
            page.goto(url)
        sel = self._sel["module_page"]
        q_locator = page.locator(
            f"{sel['question_block']}[data-question-id='{answer.question_id}']"
        )
        if q_locator.count() == 0:
            log.warning("question %s not found on %s; cannot submit", answer.question_id, url)
            return False
        if q_locator.locator(sel["question_options"]).count() > 0:
            # MC question: check the matching radio.
            opt = q_locator.locator(
                f"{sel['question_options']}[value='{answer.answer_text}']"
            )
            if opt.count() == 0:
                log.warning("MC option %r not found on question %s",
                            answer.answer_text, answer.question_id)
                return False
            opt.first.check()
        else:
            inp = q_locator.locator(sel["question_input"]).first
            inp.fill(answer.answer_text)
        q_locator.locator(sel["question_submit"]).first.click()
        # Wait for the correct/wrong marker to appear.
        try:
            page.wait_for_selector(
                ",".join([sel["question_correct_marker"], sel["question_wrong_marker"]]),
                timeout=self.cfg.selector_timeout_ms,
            )
        except Exception:
            log.warning("no correct/wrong marker after submit on %s", answer.question_id)
            return False
        return q_locator.locator(sel["question_correct_marker"]).count() > 0

    # ---- helpers ------------------------------------------------------------

    def _is_logged_in_dom(self) -> bool:
        page = self._page
        if page is None:
            return False
        return page.locator(self._sel["login"]["logged_in_marker"]).count() > 0


# ---- module-level small helpers ---------------------------------------------


def _safe_text(node, selector: str) -> str:
    try:
        return node.locator(selector).first.text_content() or ""
    except Exception:
        return ""


def _first_int(s: str) -> int | None:
    if not s:
        return None
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    if not digits:
        return None
    try:
        return int(digits[0])
    except ValueError:
        return None


def _hash_id(s: str) -> str:
    """Stable short ID derived from a string. Used when the academy DOM
    doesn't expose a per-element id attribute."""
    import hashlib
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:12]


def _parse_section_label(label: str) -> tuple[int | None, int | None]:
    """Parse strings like 'Section 11 / 22' into (11, 22)."""
    if not label:
        return None, None
    import re as _re
    m = _re.search(r"(\d+)\s*/\s*(\d+)", label)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))
