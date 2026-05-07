"""Shared Chrome DevTools Protocol helpers for academy walkers.

Both ``scripts/htb_academy_run.py`` (study-only) and
``scripts/htb_academy_wizard.py`` (model-or-user answers + optional auto-submit)
attach to a user-launched Chrome with a remote-debugging port and walk the
academy SPA. They share:

- a tiny synchronous CDP client over a single websocket
- the section scraper JS that pulls theory + questions + inline-code + bullets
- a robust "enter module" routine that retries the Vuetify entry button
- a "click Next" helper to advance sections
- a "fill answer + click Submit" helper for the wizard's auto-submit path

The auto-submit helper is *only* called for module text/MC questions. Lab FLAG
questions are explicitly defended at the wizard level: per the project rule
("labs flags model will not auto-sign this is user task only !") we never have
the model push a flag answer into the page.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from websockets.sync.client import connect

from htbrl.academy.page_models import (
    AcademyQuestion,
    AcademySection,
    QuestionType,
)


# JavaScript that scrapes a single rendered section into a structured dict.
# The question card layout is ``<li class="mb-4"><div class="collapse...">...
# <input placeholder="Write your answer">...<button>Submit</button>...</li>``
# and the card's full text is "Question N\n+M\n+K\n<actual prompt>\nSubmit
# \nHint" so we strip the boilerplate to leave the prompt body.
SECTION_SCRAPER_JS = r"""
(function() {
  function txt(el) { return el ? (el.innerText || el.textContent || '').trim() : ''; }
  const allText = document.body.innerText || '';
  const m = allText.match(/Section\s+(\d+)\s*\/\s*(\d+)/);
  const sec_idx = m ? parseInt(m[1], 10) : 0;
  const sec_total = m ? parseInt(m[2], 10) : 0;
  const h1 = document.querySelector('main h1, article h1, h1');
  const title = txt(h1);
  const body_el = document.querySelector('article, main [class*="content"], main') || document.body;
  const body_text = txt(body_el);
  const inline = Array.from(document.querySelectorAll('p code, li code, h2 code, h3 code, h4 code'))
    .map(c => txt(c)).filter(s => s && s.length <= 80);
  const code_blocks = Array.from(document.querySelectorAll('pre code, pre'))
    .map(c => txt(c)).filter(s => s.length > 0);
  const bullet_lists = Array.from(document.querySelectorAll('ul')).map(ul =>
    Array.from(ul.querySelectorAll('li')).map(li => txt(li)).filter(s => s)
  ).filter(l => l.length > 0);

  function findCard(inp) {
    let card = inp;
    for (let i = 0; i < 8 && card; i++) {
      if ((card.className || '').includes('collapse')) break;
      card = card.parentElement;
    }
    if (!card) card = inp.closest('li') || inp.parentElement;
    return card;
  }

  function extractQuestion(inp, idx) {
    const card = findCard(inp);
    const cardText = txt(card);
    const rewardMatches = cardText.match(/\+(\d+)/g) || [];
    const rewards = rewardMatches.map(s => parseInt(s.replace('+',''), 10));
    // Strip the "Question N", "+N", "Submit", "Hint" boilerplate.
    const lines = cardText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const promptLines = lines.filter(l => {
      if (/^Question\s+\d+$/i.test(l)) return false;
      if (/^\+\d+$/.test(l)) return false;
      if (/^Submit$/i.test(l)) return false;
      if (/^(Show\s+)?Hint$/i.test(l)) return false;
      if (/^(Show\s+)?Answer$/i.test(l)) return false;
      return true;
    });
    const prompt = promptLines.join(' ').slice(0, 800);
    const cardCodeBlocks = Array.from(card.querySelectorAll('pre code, pre'))
      .map(c => txt(c)).filter(s => s.length > 0);
    // "Already answered" indicator: some HTB layouts show a green check or
    // a disabled input once the user got it right. We approximate with
    // input.disabled / readonly / value already populated.
    const answered = !!(inp.disabled || inp.readOnly || (inp.value || '').trim().length > 0);
    return {
      idx, prompt,
      placeholder: inp.placeholder, name: inp.name, id: inp.id,
      reward_markers: rewards, card_code_blocks: cardCodeBlocks,
      already_answered: answered,
    };
  }

  const inputs = Array.from(document.querySelectorAll(
    'input[placeholder*="answer" i], input[placeholder*="Write your"]'
  ));
  const questions = inputs.map(extractQuestion);
  return {
    sec_idx, sec_total, title, body_text, inline, code_blocks, bullet_lists,
    questions, url: window.location.href,
  };
})()
"""


_NEXT_BUTTON_JS = r"""
(function() {
  // The academy renders three variants of the section-advance button
  // depending on completion state:
  //   "Next"                   - section not marked complete (we just
  //                              advance without crediting completion)
  //   "Complete & Next" or
  //   "Mark Complete & Next"   - all section questions answered: this
  //                              button advances AND marks the section
  //                              complete (which awards section HP/cubes
  //                              + counts toward module completion).
  //   "Finish" or
  //   "Complete Module"        - last section: marks the entire module
  //                              complete (awards module-completion
  //                              bonus cubes_awarded_upon_completion).
  // Prefer the strongest variant available so we maximize cube earnings.
  const btns = Array.from(document.querySelectorAll('button')).filter(b => !b.disabled);
  const byPriority = [
    /^Finish$/i,
    /^Complete\s+Module$/i,
    /^(Mark\s+)?Complete\s*&\s*Next$/i,
    /^(Mark\s+)?Complete\s+and\s+Next$/i,
    /^Next$/i,
  ];
  for (const re of byPriority) {
    const cand = btns.find(b => re.test((b.innerText||'').trim()));
    if (cand) {
      cand.click();
      return 'clicked ' + (cand.innerText||'').trim();
    }
  }
  return 'no Next button';
})()
"""


_ENTRY_BUTTON_JS = """
(function(idx){
    const btns = Array.from(document.querySelectorAll('button, a'));
    // The academy ships several variants of the "enter the module" button
    // across module states + responsive viewports:
    //   "Start Module"     - first time
    //   "Continue"         - already started, current section unfinished
    //   "Continue Module"  - same, mobile-prefixed variant
    //   "Resume Module"    - timed-out session
    //   "Revisit Module"   - completed module, view again
    // Match by *prefix* so we tolerate the "Module" suffix variants without
    // needing to enumerate every cross-product.
    const ENTRY_LABELS = ['Revisit', 'Continue', 'Start Module', 'Resume Module', 'Start'];
    const cands = btns.filter(b => {
        const t = (b.innerText||'').trim();
        return ENTRY_LABELS.some(l => t === l || t === l + ' Module');
    });
    if (idx >= cands.length) return {clicked:false, count:cands.length};
    cands[idx].click();
    return {clicked:true, text:cands[idx].innerText.trim(), count:cands.length, idx};
})
"""


# JS that fills a question's input by index in the current section view, fires
# Vue-friendly events, and clicks the matching ``Submit`` button. The wizard
# polls afterwards to detect accept/reject.
#
# Two phases needed because Vue/Vuetify enables the Submit button reactively
# AFTER the input event fires - if we click immediately the button is still
# in its `disabled` state. We `await` a couple of macrotasks (setTimeout 0 +
# requestAnimationFrame) before reading `btn.disabled`, which gives Vue's
# scheduler time to flush the binding update.
_FILL_AND_SUBMIT_JS = r"""
(async function(qIdx, answer) {
  const inputs = Array.from(document.querySelectorAll(
    'input[placeholder*="answer" i], input[placeholder*="Write your"]'
  ));
  const inp = inputs[qIdx];
  if (!inp) return {ok:false, why:'no input at idx ' + qIdx};
  if (inp.disabled || inp.readOnly) return {ok:false, why:'input disabled'};
  // Find the enclosing collapse-card and locate its Submit button.
  let card = inp;
  for (let i=0; i<8 && card; i++) {
    if ((card.className || '').includes('collapse')) break;
    card = card.parentElement;
  }
  if (!card) card = inp.closest('li') || inp.parentElement;
  const findBtn = () => Array.from(card.querySelectorAll('button')).find(
    b => (b.innerText||'').trim() === 'Submit'
  );
  // Vue 3 wraps the input's ``value`` setter via Object.defineProperty;
  // we bypass it so the change reaches the bound v-model. Then dispatch
  // input + change + a focus/blur pair so any debounced validators fire.
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(inp, answer);
  inp.dispatchEvent(new Event('input',  {bubbles:true}));
  inp.dispatchEvent(new Event('change', {bubbles:true}));
  inp.focus();
  // Vue's reactive flush takes ~500-600ms in HTB Academy's bundle (the
  // bound disabled-class transitions ``htb-button--disabled`` -> normal
  // around the 600ms mark). Poll up to 2.5s for the button to enable.
  const deadline = Date.now() + 2500;
  let btn = findBtn();
  while (Date.now() < deadline) {
    btn = findBtn();
    if (btn && !btn.disabled
        && !((btn.className || '').toString().includes('htb-button--disabled'))) {
      break;
    }
    await new Promise(r => setTimeout(r, 100));
  }
  if (!btn) return {ok:false, why:'no submit button'};
  if (btn.disabled
      || (btn.className || '').toString().includes('htb-button--disabled')) {
    // Click anyway as a last resort; the wizard will see "pending" and
    // we record the attempt rather than silently dropping the question.
    btn.click();
    return {ok:true, why:'clicked while still gated', value:inp.value};
  }
  btn.click();
  return {ok:true, value:inp.value};
})
"""


# Polls the question card at index for an accepted/rejected indicator.
#
# HTB Academy reliably signals success two ways:
#   - The input gets ``disabled`` / ``readOnly`` once the answer is correct
#   - A toast/banner with text "Correct!" / "+N HP" appears near the card
# We trust the disabled flag as the strongest signal (it ALSO appears when
# a question was already answered before this run - see below). For
# rejection, HTB leaves the input enabled and pops a red toast / error
# helper-text near the input. We look for the literal words "incorrect"
# or "wrong" inside the immediate ancestor (NOT the whole card) since
# generic class strings like "text-success" appear on the page in CSS
# variables even before any submission.
_QUESTION_RESULT_JS = r"""
(function(qIdx) {
  const inputs = Array.from(document.querySelectorAll(
    'input[placeholder*="answer" i], input[placeholder*="Write your"]'
  ));
  const inp = inputs[qIdx];
  if (!inp) return 'no_input';
  // Strongest accept signal: HTB locks the input once accepted.
  if (inp.disabled || inp.readOnly) return 'accepted';
  let card = inp;
  for (let i=0; i<8 && card; i++) {
    if ((card.className || '').includes('collapse')) break;
    card = card.parentElement;
  }
  if (!card) return 'no_card';
  // Look at visible text near the input, not the full innerHTML (which
  // contains too many CSS class names that match "text-success" / "correct"
  // even before any submission).
  const txt = (card.innerText || '').toLowerCase();
  if (/incorrect|wrong\s+answer|try\s+again/.test(txt)) return 'rejected';
  // Toast banners are appended to <body>, not the card, so check there too.
  const toastTxt = (document.body.innerText || '').toLowerCase();
  if (/correct[!\s]|\+\s*\d+\s*hp/.test(toastTxt)
      && !/incorrect/.test(toastTxt)) {
    // A "Correct!" toast is rare without input.disabled, but accept it.
    return 'accepted';
  }
  return 'pending';
})
"""


@dataclass
class CDPClient:
    """Tiny synchronous CDP client over a single WebSocket."""

    ws: object
    _id: int = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params or {}}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self._id:
                return r

    def evaluate(self, expression: str, await_promise: bool = False):
        r = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
        )
        if "exceptionDetails" in r.get("result", {}):
            raise RuntimeError(
                r["result"]["exceptionDetails"].get("text", "?") + ": "
                + str(r["result"]["exceptionDetails"].get("exception", {}).get("description", "?"))
            )
        return r["result"]["result"].get("value")


def pick_academy_tab(cdp_endpoint: str) -> str:
    """Return the WebSocket URL of an academy.hackthebox.com page tab."""
    tabs = json.loads(urllib.request.urlopen(f"{cdp_endpoint}/json").read())
    pages = [
        t for t in tabs
        if t.get("type") == "page" and "academy.hackthebox.com" in t.get("url", "")
    ]
    if not pages:
        pages = [t for t in tabs if t.get("type") == "page"]
    if not pages:
        raise RuntimeError(f"no page tabs in {cdp_endpoint}/json")
    return pages[0]["webSocketDebuggerUrl"]


def open_cdp(cdp_endpoint: str):
    """Yield a connected CDPClient + raw websocket. Caller must close ws."""
    ws_url = pick_academy_tab(cdp_endpoint)
    ws = connect(ws_url, max_size=20_000_000, open_timeout=30)
    cdp = CDPClient(ws=ws)
    cdp.call("Page.enable")
    return cdp, ws, ws_url


def navigate_and_wait(cdp: CDPClient, url: str, timeout_s: float = 30.0) -> str:
    """Navigate to ``url`` and poll until window.location matches."""
    cdp.call("Page.navigate", {"url": url})
    deadline = time.time() + timeout_s
    cur = ""
    while time.time() < deadline:
        try:
            cur = cdp.evaluate("window.location.href") or ""
            if cur.rstrip("/") == url.rstrip("/") or url in cur:
                break
        except Exception:
            pass
        time.sleep(0.3)
    time.sleep(2.0)
    return cur


def enter_module(cdp: CDPClient, module_id: int | str) -> tuple[bool, str]:
    """Click an entry button on the module landing until the URL hits /section/.

    Returns ``(entered, url)``. Skips the hard navigate when we're already on
    the correct module's landing/section page (the academy SPA's Vue Router
    re-binds button handlers a beat after every full navigate, which is what
    causes "click was a no-op" hangs in the wizard's auto-submit branch).
    """
    landing = f"https://academy.hackthebox.com/app/module/{module_id}"
    cur = cdp.evaluate("window.location.href") or ""
    on_module = (
        f"/app/module/{module_id}" in cur
        and ("hackthebox.com" in cur)
    )
    if not on_module:
        navigate_and_wait(cdp, landing, timeout_s=30.0)
        # Give Vue Router a beat to bind click handlers on the freshly mounted page.
        time.sleep(2.5)
    # If we're already on a section URL, no need to click any entry button.
    if "/section/" in (cdp.evaluate("window.location.href") or ""):
        # Just wait for SPA hydration like below and return.
        for _ in range(40):
            ready = cdp.evaluate(
                "(function(){"
                "const t=document.body.innerText||'';"
                "if (/Section\\s+\\d+\\s*\\/\\s*\\d+/.test(t)) return true;"
                "return false;"
                "})()"
            )
            if ready:
                break
            time.sleep(0.3)
        return True, cdp.evaluate("window.location.href") or ""

    entered = False
    for try_idx in range(6):
        res = cdp.evaluate(f"{_ENTRY_BUTTON_JS}({try_idx})") or {}
        if not res.get("clicked"):
            break
        # 12s per click. Vue Router transitions take ~1.5s in practice but the
        # first click after a fresh navigate can take longer if handlers are
        # still binding.
        for _ in range(60):
            time.sleep(0.2)
            url = cdp.evaluate("window.location.href") or ""
            if "/section/" in url:
                entered = True
                break
        if entered:
            break
    if entered:
        for _ in range(40):
            ready = cdp.evaluate(
                "(function(){"
                "const t=document.body.innerText||'';"
                "if (/Section\\s+\\d+\\s*\\/\\s*\\d+/.test(t)) return true;"
                "return false;"
                "})()"
            )
            if ready:
                break
            time.sleep(0.3)
        time.sleep(1.0)
    url = cdp.evaluate("window.location.href") or ""
    return entered, url


def scrape_section(cdp: CDPClient) -> dict:
    """Run the section scraper and return its dict, after waiting for h1."""
    for _ in range(20):
        try:
            h = cdp.evaluate(
                "(document.querySelector('main h1, article h1, h1') || {}).innerText || ''"
            ) or ""
            if h.strip():
                break
        except Exception:
            pass
        time.sleep(0.3)
    return cdp.evaluate(SECTION_SCRAPER_JS) or {}


def click_next(cdp: CDPClient) -> bool:
    """Click the section's section-advance button.

    Prefers the strongest variant available (``Finish`` >
    ``Complete Module`` > ``Complete & Next`` > ``Next``) so the academy
    gets to award completion bonuses when we've actually answered the
    section's questions. Returns True iff *any* advance button was
    clicked.
    """
    msg = cdp.evaluate(_NEXT_BUTTON_JS) or ""
    return msg.startswith("clicked")


def click_next_and_advance(cdp: CDPClient, *, current_idx: int = 0,
                           timeout_s: float = 8.0) -> bool:
    """Click Next, then poll until the section_index increments.

    The vanilla ``click_next`` returns immediately after the DOM click event
    fires, but Vue Router's transition can take 1-3s before the new section
    is rendered. If the walker scrapes too soon it gets the *previous*
    section back, which trips the "already seen" loop-protection check and
    halts the walk early.

    This variant clicks, then polls ``Section X / Y`` until X > current_idx
    OR the deadline expires. Returns True iff the index advanced.
    """
    if not click_next(cdp):
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            scraped = cdp.evaluate(SECTION_SCRAPER_JS) or {}
        except Exception:
            continue
        new_idx = int(scraped.get("sec_idx") or 0)
        if new_idx > current_idx:
            return True
    return False


_PREV_BUTTON_JS = """
(function() {
  const btns = Array.from(document.querySelectorAll('button'));
  const prev = btns.find(b => (b.innerText || '').trim() === 'Previous' && !b.disabled);
  if (prev) { prev.click(); return 'clicked Previous'; }
  return 'no Previous button';
})()
"""


def go_to_first_section(cdp: CDPClient, *, max_steps: int = 60) -> bool:
    """Click Previous until ``sec_idx == 1`` or no Previous button is enabled.

    Walks ALWAYS need to start from section 1/N to be reproducible. Without
    this, ``enter_module`` lands the walker on whichever section the operator
    was last on (HTB's "Revisit Module" is sticky), which makes the demo file
    sec-coverage non-deterministic.
    """
    for _ in range(max_steps):
        scraped = cdp.evaluate(SECTION_SCRAPER_JS) or {}
        idx = int(scraped.get("sec_idx") or 0)
        if idx == 1:
            return True
        if idx == 0:
            # Page hasn't hydrated yet. Wait a beat and re-scrape.
            time.sleep(0.5)
            continue
        msg = cdp.evaluate(_PREV_BUTTON_JS)
        if msg != "clicked Previous":
            return False  # already at section 1, or no nav at all
        # Wait for Vue Router to transition + DOM to update.
        time.sleep(1.5)
    return False


def build_section_from_scrape(scraped: dict) -> AcademySection:
    """Convert a scraper-output dict into an :class:`AcademySection`.

    Builds one :class:`AcademyQuestion` per scraped question. Reward magnitudes
    on the card map to (cubes, hp) by sort: smaller = cubes, larger = HP.
    """
    questions: list[AcademyQuestion] = []
    for i, q in enumerate(scraped.get("questions") or []):
        prompt = (q.get("prompt") or "").strip() or f"Question {i+1}"
        rewards = q.get("reward_markers") or []
        cubes_reward, hp_reward = 0, 0
        if len(rewards) == 1:
            hp_reward = int(rewards[0])
        elif len(rewards) >= 2:
            sorted_r = sorted(rewards)
            cubes_reward, hp_reward = sorted_r[0], sorted_r[-1]
        question_hints = q.get("card_code_blocks") or []
        questions.append(AcademyQuestion(
            id=f"q-{scraped.get('sec_idx', 0)}-{i}",
            prompt=prompt,
            type=QuestionType.TEXT,
            cubes_reward=cubes_reward,
            hp_reward=hp_reward,
            hints=question_hints,
        ))
    return AcademySection(
        id=f"sec-{scraped.get('sec_idx', 0)}",
        title=scraped.get("title") or f"Section {scraped.get('sec_idx', 0)}",
        body_text=scraped.get("body_text") or "",
        questions=questions,
        sandbox=None,
        code_blocks=list(scraped.get("code_blocks") or []),
        inline_code=list(scraped.get("inline") or []),
        bullet_lists=list(scraped.get("bullet_lists") or []),
        section_index=int(scraped.get("sec_idx") or 0),
        section_total=int(scraped.get("sec_total") or 0),
    )


_LAB_FLAG_HINTS = (
    "submit the flag",
    "submit your flag",
    "what is the flag",
    "what's the flag",
    "find the flag",
    "the flag value",
    "flag in the format",
    "htb{",
    "user flag",
    "root flag",
)


def is_lab_flag_question(q: AcademyQuestion) -> bool:
    """Return True for questions where auto-submit would be unsafe.

    The wizard never auto-submits lab-flag questions (per the rule that lab
    flag submission is operator-only - automating it risks the research account
    being banned). Module *theory* questions (acronym lookup, "how many", etc.)
    are always safe.

    A question is treated as a lab flag if either:
      - its declared :class:`QuestionType` is ``FLAG``, or
      - its prompt mentions "submit the flag", "user flag", "HTB{", etc.
    """
    if q.type == QuestionType.FLAG:
        return True
    p = (q.prompt or "").lower()
    return any(h in p for h in _LAB_FLAG_HINTS)


_CUBE_BALANCE_JS = r"""
(function(){
  // The HTB Academy header shows the cube balance as a small <button> whose
  // text is exactly the integer (e.g. ``"50"``). It's the first standalone-
  // integer button on the page; the "Reviews" / "Last Updated" buttons have
  // multi-line text that won't match this regex.
  const btns = Array.from(document.querySelectorAll('button'));
  for (const b of btns) {
    const t = (b.innerText || '').trim();
    if (/^\d+$/.test(t) && t.length >= 1 && t.length <= 6) {
      return parseInt(t, 10);
    }
  }
  return null;
})()
"""


def read_cube_balance(cdp: CDPClient) -> int | None:
    """Return the cube-balance integer from the academy header, or None.

    Used by the wizard's post-walk gate-readiness check: if the balance
    didn't change after a question-bearing module, the academy's reward
    signal hasn't landed yet and opening a new module is unsafe.
    """
    try:
        v = cdp.evaluate(_CUBE_BALANCE_JS)
    except Exception:
        return None
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_TARGET_PANEL_JS = r"""
(function(){
    // Walk DOM for an element whose direct text is exactly 'Target(s)'.
    // The academy renders this above a card that, when running, shows
    // BOTH a "Time left: N min(s)" timer in the header AND an IP:PORT
    // body row beneath it. The header and body are sibling divs inside
    // a wrapper card, so we have to walk up far enough that both are
    // covered. ``textContent`` (not innerText - that hides spacing
    // descendants of monospace lines) is what surfaces the IP.
    const all = Array.from(document.querySelectorAll('*'));
    const labels = all.filter(el => {
        const child = Array.from(el.childNodes).find(n => n.nodeType === 3);
        return child && child.textContent.trim() === 'Target(s)';
    });
    if (!labels.length) return {present: false};
    let panel = labels[0];
    // Walk up until we see either (a) the IP under the timer (running),
    // or (b) the Spawn button (not running). Cap at 18 levels so we
    // don't end up at <html>.
    for (let i = 0; i < 18; i++) {
        if (!panel.parentElement) break;
        panel = panel.parentElement;
        const tc = panel.textContent || '';
        const hasIP = /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.test(tc);
        const hasSpawn = /Spawn\s+the\s+target|Click\s+Here\s+to\s+Spawn|Spawn\s+Target/i.test(tc);
        const hasTimer = /Time\s+left/i.test(tc);
        // Running target: walk up until BOTH timer + IP are visible.
        if (hasTimer && hasIP) break;
        // Not-running: walk up until we see the spawn button.
        if (hasSpawn) break;
    }
    const text = (panel.textContent || '').slice(0, 1500);
    const ipPortMatch = text.match(/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d{1,5}))?/);
    const timerMatch = text.match(/Time\s+left:?\s*(\d+)\s*min/i);
    const buttons = Array.from(panel.querySelectorAll('button')).map(b => ({
        text: ((b.innerText||b.textContent)||'').trim(),
        disabled: b.disabled,
    }));
    return {
        present: true,
        text,
        ip: ipPortMatch ? ipPortMatch[1] : null,
        port: ipPortMatch && ipPortMatch[2] ? parseInt(ipPortMatch[2], 10) : null,
        ttl_minutes: timerMatch ? parseInt(timerMatch[1], 10) : null,
        buttons,
        // Convenience: is a target currently running?
        running: !!ipPortMatch,
    };
})()
"""


_SPAWN_TARGET_BTN_JS = r"""
(function(){
    // Match "Spawn the target system", "Click Here to Spawn Target",
    // "Spawn Target", or just "Start Target" - all observed variants
    // across academy modules.
    const btns = Array.from(document.querySelectorAll('button'));
    const cand = btns.find(b => {
        const t = ((b.innerText||b.textContent)||'').trim();
        return /^(Spawn\s+the\s+target(\s+system)?|Click\s+Here\s+to\s+Spawn(\s+Target)?|Spawn\s+Target|Start\s+Target)$/i.test(t);
    });
    if (!cand) return {clicked: false, why: 'no spawn button'};
    if (cand.disabled) return {clicked: false, why: 'spawn button disabled'};
    cand.click();
    return {clicked: true, text: ((cand.innerText||cand.textContent)||'').trim()};
})()
"""


_STOP_TARGET_BTN_JS = r"""
(function(){
    // The academy renders the running-target panel with two icon-only
    // buttons:
    //   - htb-square-button--secondary (orange refresh)  aria-label
    //                                                     "reset-target"
    //   - htb-square-button--danger (red X)              aria-label
    //                                                     "terminate-target"
    // Match by aria-label first (most reliable), then class fallback.
    const btns = Array.from(document.querySelectorAll('button'));
    const cand = btns.find(b => {
        const aria = (b.getAttribute('aria-label') || '').toLowerCase();
        const title = (b.getAttribute('title') || '').toLowerCase();
        const tip = (b.getAttribute('tooltip') || '').toLowerCase();
        const cls = (b.className || '').toString();
        const t = ((b.innerText||b.textContent)||'').trim();
        if (aria.includes('terminate-target') || aria.includes('terminate target')
            || aria.includes('stop-target') || aria.includes('stop target')) return true;
        if (title.includes('terminate target') || title.includes('stop target')
            || tip.includes('terminate target') || tip.includes('stop target')) return true;
        if (/^(Stop\s+the\s+target|Stop\s+Target|Stop\s+Machine|Terminate\s+Target)$/i.test(t)) return true;
        // Fallback: the red-X danger square button inside the target panel.
        if (cls.includes('htb-square-button--danger')) return true;
        return false;
    });
    if (!cand) return {clicked: false, why: 'no stop button'};
    if (cand.disabled) return {clicked: false, why: 'stop button disabled'};
    cand.click();
    return {clicked: true, text: ((cand.innerText||cand.textContent)||'').trim() || 'terminate-target icon'};
})()
"""


_STOP_CONFIRM_JS = r"""
(function(){
    // Clicking the red-X "terminate-target" pops a confirmation modal
    // titled "Terminate Target" with two buttons: "Terminate" (green
    // primary, htb-button--primary) and "Go Back" (secondary).
    //
    // We MUST scope to currently-visible buttons to avoid clicking a
    // stale primary-action button from an unrelated dismissed modal
    // that's still in the DOM. The previous matcher was fooled by an
    // "I understand" button left over from a different dialog and
    // returned ok=True without actually terminating.
    function isVisible(el) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return false;
        const cs = window.getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none') return false;
        if (parseFloat(cs.opacity || '1') === 0) return false;
        return true;
    }
    const btns = Array.from(document.querySelectorAll('button')).filter(isVisible);

    // 1. Most reliable: literal "Terminate" text (the actual confirm
    //    button in the Terminate Target dialog). Reject any button
    //    that contains the word "cancel" or "Go Back".
    const terminateBtn = btns.find(b => {
        const t = ((b.innerText||b.textContent)||'').trim();
        return /^Terminate\b/i.test(t)
            && !/cancel|go\s*back/i.test(t);
    });
    if (terminateBtn) {
        terminateBtn.click();
        return 'clicked Terminate button';
    }
    // 2. Fallback: primary-action-btn class (used by some other
    //    confirmation dialogs like "I understand").
    const primary = btns.find(b =>
        (b.className || '').toString().includes('primary-action-btn')
    );
    if (primary) {
        const t = ((primary.innerText||primary.textContent)||'').trim();
        // Don't click a Cancel-shaped button even if it has the class.
        if (!/cancel|go\s*back|abort/i.test(t)) {
            primary.click();
            return 'clicked primary-action: ' + t;
        }
    }
    // 3. Fallback: any other accept-shaped label.
    const cand = btns.find(b => {
        const t = ((b.innerText||b.textContent)||'').trim();
        return /^(I\s+understand|Continue|Confirm|Yes|OK(ay)?|Stop)\b/i.test(t)
            && !/cancel|go\s*back/i.test(t);
    });
    if (cand) { cand.click(); return 'clicked: ' + (cand.innerText||cand.textContent||'').trim(); }
    return 'no confirm dialog';
})()
"""


@dataclass
class TargetInfo:
    """A spawned academy target's connection info."""

    ip: str
    port: int | None
    ttl_minutes: int | None = None
    panel_text: str = ""

    @property
    def host_port(self) -> str:
        return f"{self.ip}:{self.port}" if self.port else self.ip


def read_target_info(cdp: CDPClient) -> TargetInfo | None:
    """Return the spawned target's IP:PORT + TTL, or None if not running.

    Treats "panel exists but no IP" as not-running (the spawn button is
    showing instead). Use ``spawn_target`` to start one.
    """
    res = cdp.evaluate(_TARGET_PANEL_JS) or {}
    if not res.get("present") or not res.get("running"):
        return None
    ip = res.get("ip")
    if not ip:
        return None
    return TargetInfo(
        ip=ip,
        port=res.get("port"),
        ttl_minutes=res.get("ttl_minutes"),
        panel_text=str(res.get("text", ""))[:500],
    )


def spawn_target(cdp: CDPClient, *, timeout_s: float = 90.0) -> tuple[bool, TargetInfo | str]:
    """Click the section's "Spawn the target system" button + wait for IP.

    Returns ``(ok, target)`` where ``target`` is a :class:`TargetInfo` on
    success or a diagnostic string on failure. Polls the target panel
    until an IP:PORT appears (academy targets typically take 20-60s to
    boot). Skips the click if a target is already running on this section.
    """
    # Already running? Use it.
    existing = read_target_info(cdp)
    if existing is not None:
        return True, existing
    res = cdp.evaluate(_SPAWN_TARGET_BTN_JS) or {}
    if not res.get("clicked"):
        return False, str(res.get("why") or "unknown")
    # Poll the panel for an IP:PORT.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2.0)
        info = read_target_info(cdp)
        if info is not None:
            return True, info
    return False, f"clicked spawn but no IP appeared in {timeout_s:.0f}s"


def stop_target(cdp: CDPClient) -> tuple[bool, str]:
    """Click "Stop the target system" if a target is currently running.

    Returns ``(False, "no running target")`` when nothing's spawned.
    On success, polls briefly for the panel to clear (the academy
    sometimes shows a confirmation dialog we need to dismiss first).
    """
    if read_target_info(cdp) is None:
        return False, "no running target"
    res = cdp.evaluate(_STOP_TARGET_BTN_JS) or {}
    if not res.get("clicked"):
        return False, str(res.get("why") or "unknown")
    # Some modules confirm the stop with a dialog.
    time.sleep(0.8)
    cdp.evaluate(_STOP_CONFIRM_JS)
    # Poll until the panel goes back to "Spawn the target system" state.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        time.sleep(0.5)
        if read_target_info(cdp) is None:
            return True, str(res.get("text") or "stopped")
    return True, "clicked but panel still shows running after 8s"


_HINT_BUTTON_JS = r"""
(function(qIdx){
    const inputs = Array.from(document.querySelectorAll(
        'input[placeholder*="answer" i], input[placeholder*="Write your"]'
    ));
    const inp = inputs[qIdx];
    if (!inp) return {clicked:false, why:'no input at idx ' + qIdx};
    let card = inp;
    for (let i=0; i<8 && card; i++) {
        if ((card.className || '').includes('collapse')) break;
        card = card.parentElement;
    }
    if (!card) card = inp.closest('li') || inp.parentElement;
    const btn = Array.from(card.querySelectorAll('button')).find(b =>
        /^(Hint|Show\s+Hint)$/i.test(((b.innerText||b.textContent)||'').trim())
    );
    if (!btn) return {clicked:false, why:'no Hint button'};
    if (btn.disabled) return {clicked:false, why:'Hint button disabled'};
    btn.click();
    return {clicked:true};
})
"""


_HINT_MODAL_JS = r"""
(function(){
    // The Hint modal renders as a <dialog class="modal modal-open">
    // outside the question card. Its full innerText is:
    //   "Hint\n\n<the hint text>\n\nOkay\nClose"
    // We strip the "Hint" header + the trailing button labels to leave
    // just the body.
    const dlg = document.querySelector('dialog.modal-open, dialog[open]');
    if (!dlg) return null;
    const t = (dlg.innerText || '').trim();
    return t
        .replace(/^Hint\s*/i, '')
        .replace(/\s*(Okay|Close|Got\s+it|Cancel)\s*(Okay|Close|Got\s+it|Cancel)?\s*$/i, '')
        .trim();
})()
"""


_DISMISS_HINT_JS = r"""
(function(){
    const dlg = document.querySelector('dialog.modal-open, dialog[open]');
    if (!dlg) return 'no modal';
    const btn = Array.from(dlg.querySelectorAll('button')).find(b =>
        /^(Okay|Close|Got\s+it)$/i.test(((b.innerText||b.textContent)||'').trim())
    );
    if (btn) { btn.click(); return 'dismissed'; }
    // Fallback: try the dialog's native close.
    if (typeof dlg.close === 'function') { dlg.close(); return 'closed via dialog.close'; }
    return 'could not dismiss';
})()
"""


def read_hint_for_question(
    cdp: CDPClient, q_idx: int, *, settle_s: float = 1.5,
) -> str | None:
    """Click the Hint button for question ``q_idx`` and return the revealed text.

    HTB Academy renders hints in a modal ``<dialog class="modal-open">``
    outside the question card; we click Hint, wait for the modal to
    paint, scrape its body text (stripping the "Hint" header and
    "Okay"/"Close" buttons), then dismiss the modal.

    Returns the hint string on success, None if no Hint button exists or
    the modal didn't open. The wizard threads the hint into the
    question's ``hints`` list so the answerer can use it as additional
    reading-comprehension context.
    """
    res = cdp.evaluate(f"({_HINT_BUTTON_JS})({q_idx})") or {}
    if not res.get("clicked"):
        return None
    time.sleep(settle_s)
    text = cdp.evaluate(_HINT_MODAL_JS)
    # Always try to dismiss whatever opened, even on read failure.
    try:
        cdp.evaluate(_DISMISS_HINT_JS)
    except Exception:
        pass
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


_UNLOCK_BUTTON_JS = r"""
(function(){
    // The "Unlock Module - N Cubes" button on the module landing page.
    // HTB renders it whenever module.state === 'locked' and the user has
    // enough cubes. Match by prefix "Unlock Module" so the cube-count
    // suffix doesn't trip exact-match.
    const btns = Array.from(document.querySelectorAll('button'));
    const cand = btns.find(b => /^Unlock\s+Module\b/i.test((b.innerText||'').trim()));
    if (!cand) return {clicked: false, why: 'no Unlock Module button'};
    if (cand.disabled) return {clicked: false, why: 'unlock button disabled'};
    cand.click();
    return {clicked: true, text: (cand.innerText||'').trim()};
})()
"""


_CONFIRM_UNLOCK_JS = r"""
(function(){
    // Some HTB modules pop a confirmation dialog ("Are you sure you want
    // to spend N cubes?") with a primary button containing "Unlock" or
    // "Confirm". Click it if present.
    const btns = Array.from(document.querySelectorAll('button'));
    const cand = btns.find(b => {
        const t = (b.innerText||'').trim();
        return /^(Unlock|Confirm|Yes|Continue)\b/i.test(t)
            && !/cancel/i.test(t);
    });
    if (cand) { cand.click(); return 'clicked: ' + (cand.innerText||'').trim(); }
    return 'no confirm dialog';
})()
"""


def unlock_module(cdp: CDPClient, module_id: int | str, *,
                  poll_seconds: float = 8.0) -> tuple[bool, str]:
    """Spend cubes to unlock a locked module via the academy UI.

    The operator's standing rule (set 2026-05-07 in conversation) is "you can
    always open new modules, don't ask for manual unlocking". This helper
    encapsulates the "Unlock Module - N Cubes" button click + optional
    confirmation dialog, polls the module API to verify the state changed
    away from ``locked``, and reports the outcome.

    Returns ``(unlocked, detail)`` where ``unlocked`` is True iff the API
    no longer reports ``state == 'locked'`` after the click sequence.
    Skips the click entirely (and reports True) if the module is already
    in any non-locked state.
    """
    pre = fetch_module_via_api(cdp, module_id) or {}
    pre_state = pre.get("state")
    if pre_state and pre_state != "locked":
        return True, f"already {pre_state!r}; skipping unlock"

    # Make sure we're on the module's landing page (the unlock button is
    # only rendered there). If we're on a different module's section, the
    # button won't exist.
    landing = f"https://academy.hackthebox.com/app/module/{module_id}"
    cur = cdp.evaluate("window.location.href") or ""
    if f"/app/module/{module_id}" not in cur:
        navigate_and_wait(cdp, landing, timeout_s=20.0)
        time.sleep(2.0)

    res = cdp.evaluate(_UNLOCK_BUTTON_JS) or {}
    if not res.get("clicked"):
        return False, str(res.get("why") or "unknown")
    # Some modules confirm; click the dialog's primary button if present.
    time.sleep(1.0)
    confirm = cdp.evaluate(_CONFIRM_UNLOCK_JS)
    # Poll the modules API until state is no longer 'locked'.
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        time.sleep(0.5)
        post = fetch_module_via_api(cdp, module_id) or {}
        post_state = post.get("state")
        if post_state and post_state != "locked":
            return True, f"unlocked: {pre_state!r} -> {post_state!r} ({confirm})"
    return False, f"clicked but state still {pre_state!r} after {poll_seconds:.0f}s"


def fetch_module_via_api(cdp: CDPClient, module_id: int | str) -> dict | None:
    """Fetch ``/api/v2/modules/<id>`` from inside the authenticated page.

    Returns the unwrapped ``data`` object (containing ``cheatsheet``,
    ``prelude``, ``conclusion``, ``takeaways``, ``name``, ``sections``,
    etc.) or None if the request failed.

    This is much cheaper than walking sections one-by-one: one fetch covers
    the whole module's metadata. The walker still scrapes section bodies
    via DOM (the API doesn't return rendered theory HTML), but cheatsheet
    + prelude come from here.
    """
    expr = f"""
    (async () => {{
        try {{
            const r = await fetch('/api/v2/modules/{module_id}', {{credentials: 'include'}});
            if (!r.ok) return {{__error: 'http ' + r.status}};
            const j = await r.json();
            return j.data || j;
        }} catch (e) {{
            return {{__error: String(e)}};
        }}
    }})()
    """
    try:
        v = cdp.evaluate(expr, await_promise=True)
    except Exception as exc:
        return {"__error": str(exc)}
    if not isinstance(v, dict):
        return None
    return v


def parse_cheatsheet_markdown(md: str) -> list[dict[str, str]]:
    """Parse an HTB Academy cheatsheet markdown table into structured rows.

    Input format (from ``data.cheatsheet`` on the modules API):

        | **Command** | **Description** |
        |-------------|-----------------|
        | ``man <tool>`` | Opens man pages for the specified tool. |
        | ``<tool> -h`` | Prints the help page of the tool. |

    Returns a list of dicts, one per data row, with keys taken from the
    header row (lowercased, whitespace-collapsed). Markdown emphasis
    (``**foo**``) and inline code backticks are stripped from cell values
    so the answerer can compare commands directly.

    Lines that aren't pipe-separated (intro paragraphs etc.) are ignored.
    Sub-headers / category dividers ("**Filesystem commands**" with no
    pipes) are also skipped.
    """
    if not md:
        return []
    rows: list[list[str]] = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        # Drop the leading/trailing pipe then split on |, trimming each cell.
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        rows.append(cells)
    if not rows:
        return []
    # First row = header, second = separator (---|---), rest = data.
    header_cells = rows[0]
    keys = [_clean_cell(c).lower().replace(" ", "_") or f"col_{i}"
            for i, c in enumerate(header_cells)]
    out: list[dict[str, str]] = []
    for cells in rows[1:]:
        # Skip the markdown separator row ("---|---|").
        if all(set(c) <= set("-: ") for c in cells):
            continue
        # Pad / truncate to header length.
        if len(cells) < len(keys):
            cells = cells + [""] * (len(keys) - len(cells))
        cells = cells[: len(keys)]
        row = {k: _clean_cell(v) for k, v in zip(keys, cells)}
        # Drop fully-empty rows.
        if any(v for v in row.values()):
            out.append(row)
    return out


def _clean_cell(s: str) -> str:
    """Strip markdown bold/italic and inline-code ticks from a cell value."""
    s = (s or "").strip()
    # Remove **bold** and *italic*; keep contents.
    while s.startswith("**") and s.endswith("**") and len(s) > 4:
        s = s[2:-2].strip()
    while s.startswith("*") and s.endswith("*") and len(s) > 2:
        s = s[1:-1].strip()
    # Strip a leading/trailing single backtick if present.
    if s.startswith("`") and s.endswith("`") and len(s) > 2:
        s = s[1:-1].strip()
    # Replace common HTML entities the academy uses.
    s = s.replace("&nbsp;", " ").replace("\xa0", " ")
    return s


def already_answered_flags(scraped: dict) -> list[bool]:
    """Per-question 'is this already accepted on the page' booleans.

    Used by the wizard to skip questions HTB has already marked correct.
    """
    return [bool(q.get("already_answered")) for q in (scraped.get("questions") or [])]


def submit_answer_in_dom(
    cdp: CDPClient, q_idx: int, answer_text: str, *, poll_seconds: float = 8.0,
) -> tuple[str, str]:
    """Fill the question's input + click Submit + poll for the result.

    Returns ``(state, detail)`` where ``state`` is one of:
        - ``"accepted"``  -- HTB acknowledged the answer (green check / disabled input)
        - ``"rejected"``  -- HTB rejected the answer (red X / wrong indicator)
        - ``"pending"``   -- no clear indicator after polling (timeout)
        - ``"error"``     -- could not click Submit (input disabled, etc.)
    ``detail`` carries the JS rationale.

    NOTE: this is for *module* text/MC questions only. The wizard MUST NOT
    call this for ``QuestionType.FLAG`` (lab flag); doing so risks an account
    ban per the project's stated rule.
    """
    js = f"({_FILL_AND_SUBMIT_JS})({q_idx}, {json.dumps(answer_text)})"
    # The fill-and-submit JS is async (it awaits Vue's reactive flush
    # before clicking Submit) so we must await the returned promise.
    res = cdp.evaluate(js, await_promise=True) or {}
    if not res.get("ok"):
        return "error", str(res.get("why") or "unknown")
    deadline = time.time() + poll_seconds
    last = "pending"
    while time.time() < deadline:
        last = cdp.evaluate(f"({_QUESTION_RESULT_JS})({q_idx})") or "pending"
        if last in ("accepted", "rejected"):
            return last, "polled"
        time.sleep(0.4)
    return last, f"polled but no terminal state in {poll_seconds:.0f}s"


__all__ = [
    "CDPClient",
    "SECTION_SCRAPER_JS",
    "TargetInfo",
    "already_answered_flags",
    "build_section_from_scrape",
    "click_next",
    "click_next_and_advance",
    "enter_module",
    "fetch_module_via_api",
    "go_to_first_section",
    "is_lab_flag_question",
    "navigate_and_wait",
    "open_cdp",
    "parse_cheatsheet_markdown",
    "pick_academy_tab",
    "read_cube_balance",
    "read_hint_for_question",
    "read_target_info",
    "scrape_section",
    "spawn_target",
    "stop_target",
    "submit_answer_in_dom",
    "unlock_module",
]
