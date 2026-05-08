"""CDP-driven DOM walker for ``app.hackthebox.com``'s labs section.

Sister of ``htbrl.academy.cdp_walker`` — same plumbing (CDPClient,
websocket attach, JS-eval scraping) but targets the labs DOM:

  - Box page = list of numbered Tasks
  - Each Task = (question text, optional hint button, answer input,
                 "Submit Task" button)
  - Last entry = "Submit Flag" input (locks until all tasks are done)
  - Side panel = target IP + connection state + spawn/stop

Reuses :class:`htbrl.academy.cdp_walker.CDPClient` directly so we
don't duplicate the websocket bits. The JS payloads here are
labs-specific.

**Live-verification status (2026-05-08):** the JS scrapers below
are designed against the visible DOM seen in operator screenshots
during the Tier-0 + Tier-1 walks. They have NOT yet been
exercised against a live attached Chrome — the operator will
attach + we'll iterate on selector specifics in the next session.
Until then, every scraper has a fallback path that returns an
empty/None result rather than crashing, so a wrong selector
just degrades to "wizard couldn't read the DOM" rather than a
hard failure.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from htbrl.academy.cdp_walker import CDPClient


# ---- tab picker -------------------------------------------------------------


def pick_labs_tab(cdp_endpoint: str) -> str:
    """Return the WebSocket URL of an ``app.hackthebox.com`` page tab.

    Falls back to any page tab if no labs tab is open (so the operator
    can see "no labs tab found" rather than a hard error).
    """
    tabs = json.loads(urllib.request.urlopen(f"{cdp_endpoint}/json").read())
    pages = [
        t for t in tabs
        if t.get("type") == "page" and "app.hackthebox.com" in t.get("url", "")
    ]
    if not pages:
        pages = [t for t in tabs if t.get("type") == "page"]
    if not pages:
        raise RuntimeError(f"no page tabs found in {cdp_endpoint}/json")
    return pages[0]["webSocketDebuggerUrl"]


def open_cdp(cdp_endpoint: str):
    """Yield a connected CDPClient + raw websocket pinned to a labs tab."""
    from websockets.sync.client import connect       # local import keeps cost low
    ws_url = pick_labs_tab(cdp_endpoint)
    ws = connect(ws_url, max_size=20_000_000, open_timeout=30)
    cdp = CDPClient(ws=ws)
    cdp.call("Page.enable")
    return cdp, ws, ws_url


# ---- dataclasses for scraped state ------------------------------------------


@dataclass
class LabsTask:
    """One task as scraped from the labs box page."""

    number: int                           # 1-indexed
    question: str
    placeholder: str = ""                 # input placeholder if visible
    accepted: bool = False                # True if green check shown
    locked: bool = False                  # True if input/submit are disabled
    hint_visible: bool = False            # True if hint button is rendered
    input_id: str = ""                    # the `id` attr of the answer input
    submit_button_text: str = "Submit Task"


@dataclass
class LabsBoxPage:
    """Snapshot of the box page state."""

    target_id: str = ""                   # e.g. "htb-starting-point:meow"
    target_ip: str = ""                   # e.g. "10.129.119.129"
    connected_to_htb: bool = False
    tasks: list[LabsTask] = field(default_factory=list)
    flag_input_id: str = ""               # the id of the final flag input
    flag_input_locked: bool = True        # True until all tasks accepted
    flag_accepted: bool = False           # True if flag was submitted + accepted
    url: str = ""

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_accepted_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.accepted)


# ---- JS scrapers ------------------------------------------------------------

# Pull all the on-page state in ONE evaluate() so we don't pay the round-trip
# cost N times. Returns a structured dict that ``scrape_box_page`` lifts into
# :class:`LabsBoxPage`.
#
# Selector strategy: HTB's labs SPA uses Vue/React-style numbered cards.
# We look for elements containing the literal text "Task <N>" + an
# adjacent input + "Submit Task" button. That's robust to class-name
# changes which HTB does periodically.
_BOX_PAGE_SCRAPER_JS = r"""
(function() {
  function txt(el) { return el ? (el.innerText || el.textContent || '').trim() : ''; }
  // Target IP — usually rendered next to "Target IP Address" / "Target(s)".
  function findTargetIp() {
    // Look for an IPv4 octet pattern in elements near "Target" labels.
    const labels = Array.from(document.querySelectorAll('*')).filter(el => {
      const t = (el.innerText || el.textContent || '').trim();
      return /^Target\b/i.test(t) && t.length < 80;
    });
    const ipRe = /\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[0-1]))\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\b/;
    for (const el of labels) {
      // Walk up + over to find a sibling/descendant containing an IP.
      let n = el;
      for (let i = 0; i < 6 && n; i++) {
        const m = (n.innerText || '').match(ipRe);
        if (m) return m[0];
        n = n.parentElement;
      }
    }
    // Fallback: any IP-shaped string on the page.
    const all = document.body.innerText.match(ipRe);
    return all ? all[0] : '';
  }
  function findConnectedToHTB() {
    // Heuristic: page mentions "Connected with OpenVPN" near a green dot.
    const t = (document.body.innerText || '').toLowerCase();
    return t.includes('connected with openvpn') || t.includes('connected to hack the box');
  }
  // Tasks: find elements that contain "Task <N>" header + a question + an input.
  function findTasks() {
    const tasks = [];
    // Prefer explicit class hints if HTB shipped them; fall back to text scan.
    // Iterate through candidate cards: any block-level element whose
    // innerText starts with "Task <N>" and contains an <input>.
    const candidates = Array.from(document.querySelectorAll('div, section, article, li'))
      .filter(el => /^Task\s+\d+\b/i.test(txt(el).split('\n')[0] || ''));
    const seen = new Set();
    for (const el of candidates) {
      // Only keep the OUTERMOST card matching that header (skip nested dupes).
      let parent = el.parentElement;
      let isInner = false;
      while (parent) {
        if (candidates.includes(parent)) { isInner = true; break; }
        parent = parent.parentElement;
      }
      if (isInner) continue;
      const head = (txt(el).split('\n')[0] || '').match(/^Task\s+(\d+)/i);
      if (!head) continue;
      const n = parseInt(head[1], 10);
      if (seen.has(n)) continue;
      seen.add(n);
      // Scrape question + input + locked state
      const lines = txt(el).split('\n').map(s => s.trim()).filter(s => s);
      // Drop the "Task N" header + Hint/Submit chrome.
      const promptLines = lines.filter(l => {
        if (/^Task\s+\d+$/i.test(l)) return false;
        if (/^(Hint|Submit Task|Submit Flag|Accepted)$/i.test(l)) return false;
        return true;
      });
      const inp = el.querySelector('input[type="text"], input[type="password"], input:not([type])');
      const btn = Array.from(el.querySelectorAll('button')).find(
        b => /^Submit\s+Task$/i.test(txt(b))
      );
      const hintBtn = Array.from(el.querySelectorAll('button')).find(
        b => /^Hint$/i.test(txt(b))
      );
      const accepted = !!(
        inp && (inp.disabled || inp.readOnly)
        || el.querySelector('[class*="check"], [class*="success"], [class*="accepted"]')
      );
      const locked = !inp || (btn && btn.disabled);
      tasks.push({
        number: n,
        question: promptLines.slice(0, 5).join(' ').slice(0, 600),
        placeholder: inp ? (inp.placeholder || '') : '',
        accepted: accepted,
        locked: !!locked,
        hint_visible: !!hintBtn,
        input_id: inp ? (inp.id || '') : '',
        submit_button_text: btn ? txt(btn) : 'Submit Task',
      });
    }
    tasks.sort((a, b) => a.number - b.number);
    return tasks;
  }
  // Final flag input
  function findFlagInput() {
    const submitFlagBtns = Array.from(document.querySelectorAll('button'))
      .filter(b => /^Submit\s+Flag$/i.test(txt(b)));
    if (!submitFlagBtns.length) return {input_id: '', locked: true, accepted: false};
    const btn = submitFlagBtns[0];
    let card = btn;
    for (let i = 0; i < 6 && card; i++) {
      if (card.querySelector && card.querySelector('input')) break;
      card = card.parentElement;
    }
    const inp = card ? card.querySelector('input[type="text"], input:not([type])') : null;
    return {
      input_id: inp ? (inp.id || '') : '',
      locked: !inp || btn.disabled,
      accepted: !!(inp && (inp.disabled || inp.readOnly)),
    };
  }
  return {
    url: window.location.href,
    target_ip: findTargetIp(),
    connected_to_htb: findConnectedToHTB(),
    tasks: findTasks(),
    flag: findFlagInput(),
  };
})()
"""


def scrape_box_page(cdp: CDPClient) -> LabsBoxPage:
    """Pull the current box-page state into a :class:`LabsBoxPage`.

    Doesn't navigate — the operator (or wizard) is responsible for being
    on a box page in Chrome. Returns an empty-ish ``LabsBoxPage`` if the
    DOM didn't match (no exception raised so the caller can inspect).
    """
    try:
        raw = cdp.evaluate(_BOX_PAGE_SCRAPER_JS) or {}
    except Exception:
        return LabsBoxPage()
    flag = raw.get("flag") or {}
    page = LabsBoxPage(
        target_ip=raw.get("target_ip") or "",
        connected_to_htb=bool(raw.get("connected_to_htb")),
        url=raw.get("url") or "",
        flag_input_id=flag.get("input_id") or "",
        flag_input_locked=bool(flag.get("locked", True)),
        flag_accepted=bool(flag.get("accepted", False)),
    )
    for t in raw.get("tasks") or []:
        page.tasks.append(LabsTask(
            number=int(t.get("number", 0)),
            question=t.get("question", ""),
            placeholder=t.get("placeholder", ""),
            accepted=bool(t.get("accepted", False)),
            locked=bool(t.get("locked", False)),
            hint_visible=bool(t.get("hint_visible", False)),
            input_id=t.get("input_id", ""),
            submit_button_text=t.get("submit_button_text", "Submit Task"),
        ))
    return page


# ---- task submission --------------------------------------------------------

# Fill a task's input + click its Submit Task button. Vue-style reactive
# bindings need the native value setter (same trick as the academy walker).
_FILL_TASK_JS = r"""
(async function(taskNumber, answer) {
  function txt(el) { return el ? (el.innerText || el.textContent || '').trim() : ''; }
  // Locate the card containing "Task N" header.
  const cards = Array.from(document.querySelectorAll('div, section, article, li'))
    .filter(el => new RegExp('^Task\\s+' + taskNumber + '\\b').test(
      (txt(el).split('\n')[0] || '')
    ));
  if (!cards.length) return {ok: false, why: 'no card matches Task ' + taskNumber};
  // Pick the smallest card (innermost) that has an input — outermost might
  // be the page wrapper containing every task.
  let card = null;
  for (const c of cards) {
    if (c.querySelector('input[type="text"], input[type="password"], input:not([type])')) {
      if (!card || c.contains(card) === false) card = c;
    }
  }
  if (!card) return {ok: false, why: 'no input under Task ' + taskNumber};
  const inp = card.querySelector('input[type="text"], input[type="password"], input:not([type])');
  if (inp.disabled || inp.readOnly) return {ok: false, why: 'task input is disabled (already submitted?)'};
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(inp, answer);
  inp.dispatchEvent(new Event('input', {bubbles: true}));
  inp.dispatchEvent(new Event('change', {bubbles: true}));
  // Find the Submit Task button — wait briefly for Vue to flush the binding
  // so the disabled class clears.
  let btn = null;
  const findBtn = () => Array.from(card.querySelectorAll('button'))
    .find(b => /^Submit\s+Task$/i.test(txt(b)));
  const deadline = Date.now() + 2500;
  while (Date.now() < deadline) {
    btn = findBtn();
    if (btn && !btn.disabled) break;
    await new Promise(r => setTimeout(r, 80));
  }
  if (!btn) return {ok: false, why: 'no Submit Task button in card'};
  btn.click();
  return {ok: true, value: inp.value};
})
"""


def submit_task_in_dom(
    cdp: CDPClient, task_number: int, answer: str,
    *, poll_seconds: float = 8.0,
) -> tuple[str, str]:
    """Fill task ``N``'s input + click Submit + poll for accept/reject.

    Returns ``(state, detail)`` where ``state`` is one of:
      - ``"accepted"`` — input went disabled / green check appeared
      - ``"rejected"`` — error toast / unchanged state after poll window
      - ``"pending"`` — inconclusive
      - ``"error"`` — no card / no button / input already disabled
    """
    js = f"({_FILL_TASK_JS})({task_number}, {json.dumps(answer)})"
    res = cdp.evaluate(js, await_promise=True) or {}
    if not res.get("ok"):
        return "error", str(res.get("why") or "unknown")
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        page = scrape_box_page(cdp)
        for t in page.tasks:
            if t.number == task_number:
                if t.accepted:
                    return "accepted", "polled"
                break
        time.sleep(0.4)
    return "pending", f"no terminal state in {poll_seconds:.0f}s"


# ---- final flag submission --------------------------------------------------

# Fill the final flag input + click Submit Flag. Same Vue-friendly fill.
_SUBMIT_FLAG_JS = r"""
(async function(flag) {
  function txt(el) { return el ? (el.innerText || el.textContent || '').trim() : ''; }
  const submitBtn = Array.from(document.querySelectorAll('button'))
    .find(b => /^Submit\s+Flag$/i.test(txt(b)));
  if (!submitBtn) return {ok: false, why: 'no Submit Flag button'};
  // Walk up to the card containing the Submit Flag button + an input.
  let card = submitBtn;
  for (let i = 0; i < 8 && card; i++) {
    if (card.querySelector && card.querySelector('input[type="text"], input:not([type])')) break;
    card = card.parentElement;
  }
  const inp = card ? card.querySelector('input[type="text"], input:not([type])') : null;
  if (!inp) return {ok: false, why: 'no flag input near Submit Flag button'};
  if (inp.disabled || inp.readOnly) return {ok: false, why: 'flag input disabled (already submitted?)'};
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(inp, flag);
  inp.dispatchEvent(new Event('input', {bubbles: true}));
  inp.dispatchEvent(new Event('change', {bubbles: true}));
  // Wait for the button to enable.
  const deadline = Date.now() + 2500;
  let btn = submitBtn;
  while (Date.now() < deadline) {
    btn = Array.from(document.querySelectorAll('button'))
      .find(b => /^Submit\s+Flag$/i.test(txt(b)));
    if (btn && !btn.disabled) break;
    await new Promise(r => setTimeout(r, 80));
  }
  if (!btn) return {ok: false, why: 'Submit Flag button vanished after fill'};
  btn.click();
  return {ok: true, value: inp.value};
})
"""


def submit_flag_in_dom(
    cdp: CDPClient, flag: str, *, poll_seconds: float = 8.0,
) -> tuple[str, str]:
    """Fill + click Submit Flag. Returns ``(state, detail)`` like
    :func:`submit_task_in_dom`.

    NOTE: per project rule the wizard never auto-submits lab flags
    when ``is_lab_flag_question`` would have flagged them. This helper
    is for the operator-confirmed flow only — the wizard caller must
    explicitly opt in.
    """
    js = f"({_SUBMIT_FLAG_JS})({json.dumps(flag)})"
    res = cdp.evaluate(js, await_promise=True) or {}
    if not res.get("ok"):
        return "error", str(res.get("why") or "unknown")
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        page = scrape_box_page(cdp)
        if page.flag_accepted:
            return "accepted", "polled"
        time.sleep(0.4)
    return "pending", f"no terminal state in {poll_seconds:.0f}s"


__all__ = [
    "LabsBoxPage",
    "LabsTask",
    "open_cdp",
    "pick_labs_tab",
    "scrape_box_page",
    "submit_flag_in_dom",
    "submit_task_in_dom",
]
