"""End-to-end academy run using CDP attach to user-launched Chrome.

Walks every section of a target module, scrapes theory + questions + inline
code + bullet lists, runs the HeuristicAnswerer on any questions it finds,
and writes one Demonstration to ``data/auto_demos/``.

Hard requirements before running:

  1. User has run ``scripts/start_chrome_for_htb.ps1`` and is logged into
     HTB Academy in that Chrome window.
  2. The Chrome debug endpoint is reachable at ``$HTBRL_ACADEMY_CDP``
     (default ``http://127.0.0.1:9222``).

study_only=True (default) means we never click "Mark Complete & Next" - we
read content and walk via the regular "Next" button only. No academy-side
state is mutated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

from htbrl.academy.answerer import HeuristicAnswerer
from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySection,
    QuestionType,
)
from htbrl.data.demo_dataset import save_demonstration


# JavaScript that scrapes a single rendered section into a structured dict.
# Question-prompt extraction was rewritten to match the actual academy DOM
# (each question is an <li class="mb-4"><div class="collapse...">). The
# expanded card's full text is "Question N\n+M\n+K\n<actual prompt>\nSubmit
# \nHint" - we strip the Question/reward/button noise to leave the prompt.
_SECTION_SCRAPER_JS = r"""
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

  // Question extraction: walk up from each answer input to its enclosing
  // collapse-card, then take the card's innerText and strip the boilerplate.
  function extractQuestion(inp, idx) {
    let card = inp;
    for (let i = 0; i < 8 && card; i++) {
      if ((card.className || '').includes('collapse')) break;
      card = card.parentElement;
    }
    if (!card) card = inp.closest('li') || inp.parentElement;
    const cardText = txt(card);
    // Capture "+N" reward markers (cubes have a green-cube icon, HP has purple).
    // The DOM only gives us the magnitudes; we can't easily distinguish which is
    // cubes vs HP without color/icon inspection, so we record both.
    const rewardMatches = cardText.match(/\+(\d+)/g) || [];
    const rewards = rewardMatches.map(s => parseInt(s.replace('+',''), 10));
    // Strip noise: lines containing "Question N", standalone "+N", "Submit",
    // "Hint", "Show Hint", "Show Answer". Keep the prompt body.
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
    // Detect any code blocks INSIDE the question card (sandbox SSH creds, etc.)
    const cardCodeBlocks = Array.from(card.querySelectorAll('pre code, pre'))
      .map(c => txt(c)).filter(s => s.length > 0);
    return { idx, prompt, placeholder: inp.placeholder, name: inp.name, id: inp.id,
             reward_markers: rewards, card_code_blocks: cardCodeBlocks };
  }

  const inputs = Array.from(document.querySelectorAll('input[placeholder*="answer" i], input[placeholder*="Write your"]'));
  const questions = inputs.map(extractQuestion);
  return { sec_idx, sec_total, title, body_text, inline, code_blocks, bullet_lists, questions, url: window.location.href };
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


class CDP:
    """Tiny synchronous CDP client over a single WebSocket."""

    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    def call(self, method: str, params: dict | None = None):
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
                r["result"]["exceptionDetails"].get("text", "?")
                + ": "
                + str(r["result"]["exceptionDetails"].get("exception", {}).get("description", "?"))
            )
        return r["result"]["result"].get("value")


def _pick_academy_tab(cdp_endpoint: str) -> str:
    """Return the WebSocket URL of an academy.hackthebox.com page tab."""
    tabs = json.loads(urllib.request.urlopen(f"{cdp_endpoint}/json").read())
    pages = [t for t in tabs if t.get("type") == "page" and "academy.hackthebox.com" in t.get("url", "")]
    if not pages:
        # Fall back to any page tab; we'll navigate it.
        pages = [t for t in tabs if t.get("type") == "page"]
    if not pages:
        raise RuntimeError(f"no page tabs in {cdp_endpoint}/json")
    return pages[0]["webSocketDebuggerUrl"]


def _navigate_and_wait(cdp: CDP, url: str, timeout_s: float = 30.0) -> str:
    """Navigate the attached page to `url` and poll until window.location matches."""
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
    time.sleep(2.0)  # let JS finish hydrating
    return cur


def _scrape_section(cdp: CDP) -> dict:
    """Run the section scraper JS and return its dict."""
    # Wait for an h1 to appear (SPA hydration).
    for _ in range(20):
        try:
            h = cdp.evaluate("(document.querySelector('main h1, article h1, h1') || {}).innerText || ''") or ""
            if h.strip():
                break
        except Exception:
            pass
        time.sleep(0.3)
    return cdp.evaluate(_SECTION_SCRAPER_JS) or {}


def _click_next(cdp: CDP) -> bool:
    msg = cdp.evaluate(_NEXT_BUTTON_JS)
    return msg == "clicked Next"


def _build_section(scraped: dict) -> AcademySection:
    questions: list[AcademyQuestion] = []
    for i, q in enumerate(scraped.get("questions") or []):
        prompt = (q.get("prompt") or "").strip() or f"Question {i+1}"
        rewards = q.get("reward_markers") or []
        # If two reward magnitudes were detected, the smaller is cubes and the
        # larger is HP (academy convention: cubes are 1-5, HP is 10-50).
        cubes_reward, hp_reward = 0, 0
        if len(rewards) == 1:
            hp_reward = int(rewards[0])
        elif len(rewards) >= 2:
            sorted_r = sorted(rewards)
            cubes_reward, hp_reward = sorted_r[0], sorted_r[-1]
        # Card-internal code blocks often contain the SSH sandbox credentials
        # the question expects you to run commands against; merge them with the
        # section's other hints so the answerer can find them.
        question_hints = q.get("card_code_blocks") or []
        questions.append(AcademyQuestion(
            id=f"q-{scraped.get('sec_idx', 0)}-{i}",
            prompt=prompt,
            type=QuestionType.TEXT,  # academy text inputs are free-text by default
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


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-id", type=int, required=True,
                   help="numeric module ID, e.g. 15 for Intro To Academy")
    p.add_argument("--module-title", default="",
                   help="optional title used in the Demonstration")
    p.add_argument("--cdp", default=os.environ.get("HTBRL_ACADEMY_CDP", "http://127.0.0.1:9222"))
    p.add_argument("--max-sections", type=int, default=50)
    p.add_argument("--auto-demo-dir", type=Path, default=Path("data/auto_demos"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    print(f"[run] module_id={args.module_id} cdp={args.cdp}")

    ws_url = _pick_academy_tab(args.cdp)
    print(f"[run] attaching to {ws_url}")

    sections: list[AcademySection] = []
    seen_ids: set[str] = set()

    # Navigate via the module landing page so the academy "starts" the
    # module if needed (the HTB SPA tracks last-visited section, etc.).
    module_landing = f"https://academy.hackthebox.com/app/module/{args.module_id}"

    with connect(ws_url, max_size=20_000_000, open_timeout=30) as ws:
        cdp = CDP(ws)
        cdp.call("Page.enable")
        print("[run] -> module landing page")
        url = _navigate_and_wait(cdp, module_landing, timeout_s=30.0)
        print(f"[run]   landed at {url}")
        # Click "Revisit Module" / "Continue" / first section to enter content.
        cdp.evaluate(
            "(function(){"
            "const btns = Array.from(document.querySelectorAll('button, a'));"
            "const t = btns.find(b => ['Revisit Module', 'Continue', 'Start Module'].includes((b.innerText||'').trim()));"
            "if (t) { t.click(); return t.innerText.trim(); } return 'no entry button';"
            "})()"
        )
        time.sleep(3)
        url = cdp.evaluate("window.location.href") or ""
        print(f"[run]   in section view: {url}")

        for hop in range(args.max_sections):
            scraped = _scrape_section(cdp)
            url = scraped.get("url", "")
            section = _build_section(scraped)
            if section.id in seen_ids:
                print(f"[run] section {section.id!r} already seen; stopping walk")
                break
            seen_ids.add(section.id)
            sections.append(section)
            print(
                f"[run]   sec {section.section_index}/{section.section_total} "
                f"{section.title!r} body={len(section.body_text)} "
                f"inline={len(section.inline_code)} bullets={len(section.bullet_lists)} "
                f"questions={len(section.questions)}"
            )
            if section.section_total and section.section_index >= section.section_total:
                print("[run] reached final section")
                break
            if not _click_next(cdp):
                print("[run] no Next button; stopping walk")
                break
            time.sleep(2)

    # Run the answerer over any questions and build submission tuples.
    answerer = HeuristicAnswerer()
    submissions: list[tuple[str, AcademyAnswer, bool]] = []
    n_questions = sum(len(s.questions) for s in sections)
    if n_questions == 0:
        print(f"[run] {len(sections)} section(s) walked; module has no questions")
    else:
        print(f"[run] running answerer on {n_questions} question(s)")
    for s in sections:
        for q in s.questions:
            ans = answerer.answer(q, s)
            print(f"[run]   q={q.id!r} method={ans.method} conf={ans.confidence:.2f} "
                  f"answer={ans.answer_text[:60]!r}")
            # study_only: we never submit to the academy. accepted=False unconditionally.
            submissions.append((str(args.module_id), ans, False))

    module = AcademyModule(
        id=str(args.module_id),
        title=args.module_title or f"Module {args.module_id}",
        tier=0,
        sections=sections,
        category="general",
    )
    demo = session_to_demonstration(module, submissions, study_only=True,
                                    extra_metadata={"cdp_endpoint": args.cdp})
    args.auto_demo_dir.mkdir(parents=True, exist_ok=True)
    out = args.auto_demo_dir / f"academy_module_{args.module_id}.msgpack.gz"
    save_demonstration(demo, out, compress=True)
    print(f"[run] wrote demo -> {out}")
    print(f"[run]   sections={len(sections)} questions={n_questions} "
          f"turns={len(demo.turns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
