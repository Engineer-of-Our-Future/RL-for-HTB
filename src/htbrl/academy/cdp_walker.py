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


_NEXT_BUTTON_JS = """
(function() {
  const btns = Array.from(document.querySelectorAll('button'));
  const next = btns.find(b => (b.innerText || '').trim() === 'Next' && !b.disabled);
  if (next) { next.click(); return 'clicked Next'; }
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
_FILL_AND_SUBMIT_JS = r"""
(function(qIdx, answer) {
  const inputs = Array.from(document.querySelectorAll(
    'input[placeholder*="answer" i], input[placeholder*="Write your"]'
  ));
  const inp = inputs[qIdx];
  if (!inp) return {ok:false, why:'no input at idx ' + qIdx};
  if (inp.disabled || inp.readOnly) return {ok:false, why:'input disabled'};
  // Vue 3 / Vuetify wraps value via prototype setter; bypass to actually set.
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(inp, answer);
  inp.dispatchEvent(new Event('input',  {bubbles:true}));
  inp.dispatchEvent(new Event('change', {bubbles:true}));
  // Find the enclosing collapse-card and locate its Submit button.
  let card = inp;
  for (let i=0; i<8 && card; i++) {
    if ((card.className || '').includes('collapse')) break;
    card = card.parentElement;
  }
  if (!card) card = inp.closest('li') || inp.parentElement;
  const btn = Array.from(card.querySelectorAll('button')).find(b =>
    (b.innerText||'').trim() === 'Submit'
  );
  if (!btn)        return {ok:false, why:'no submit button'};
  if (btn.disabled) return {ok:false, why:'submit disabled'};
  btn.click();
  return {ok:true, value:inp.value};
})
"""


# Polls the question card at index for an accepted/rejected indicator.
# Vuetify renders a green check-circle on success and a red error icon on
# failure; we look for class fragments that survive minification.
_QUESTION_RESULT_JS = r"""
(function(qIdx) {
  const inputs = Array.from(document.querySelectorAll(
    'input[placeholder*="answer" i], input[placeholder*="Write your"]'
  ));
  const inp = inputs[qIdx];
  if (!inp) return 'no_input';
  let card = inp;
  for (let i=0; i<8 && card; i++) {
    if ((card.className || '').includes('collapse')) break;
    card = card.parentElement;
  }
  if (!card) return 'no_card';
  const html = card.innerHTML.toLowerCase();
  if (html.includes('mdi-check') || html.includes('text-success') ||
      html.includes('correct') || inp.disabled || inp.readOnly) {
    return 'accepted';
  }
  if (html.includes('mdi-close') || html.includes('text-error') ||
      html.includes('incorrect') || html.includes('wrong')) {
    return 'rejected';
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
    """Click the section's Next button. Returns True iff a Next button was clicked."""
    return cdp.evaluate(_NEXT_BUTTON_JS) == "clicked Next"


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
    res = cdp.evaluate(js) or {}
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
    "scrape_section",
    "submit_answer_in_dom",
]
