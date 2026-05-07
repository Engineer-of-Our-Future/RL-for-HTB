"""End-to-end academy run using CDP attach to user-launched Chrome.

Walks every section of a target module, scrapes theory + questions + inline
code + bullet lists, runs the :class:`HeuristicAnswerer` on any questions it
finds, and writes one :class:`Demonstration` to ``data/auto_demos/``.

Hard requirements before running:

  1. User has run ``scripts/start_chrome_for_htb.ps1`` and is logged into
     HTB Academy in that Chrome window.
  2. The Chrome debug endpoint is reachable at ``$HTBRL_ACADEMY_CDP``
     (default ``http://127.0.0.1:9222``).

study_only=True (default) means we never click Submit on the page - we read
content and walk via the regular "Next" button only. No academy-side state is
mutated. Use ``scripts/htb_academy_wizard.py`` for the model-or-user answering
loop with optional auto-submit.

The CDP / scraper plumbing lives in ``htbrl.academy.cdp_walker`` so this
script and the wizard can share one source of truth.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from htbrl.academy.answerer import HeuristicAnswerer
from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.cdp_walker import (
    build_section_from_scrape,
    click_next,
    enter_module,
    go_to_first_section,
    open_cdp,
    scrape_section,
)
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademySection,
)
from htbrl.data.demo_dataset import save_demonstration


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-id", type=int, required=True,
                   help="numeric module ID, e.g. 15 for Intro To Academy")
    p.add_argument("--module-title", default="",
                   help="optional title used in the Demonstration")
    p.add_argument("--cdp",
                   default=os.environ.get("HTBRL_ACADEMY_CDP", "http://127.0.0.1:9222"))
    p.add_argument("--max-sections", type=int, default=50)
    p.add_argument("--auto-demo-dir", type=Path, default=Path("data/auto_demos"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    print(f"[run] module_id={args.module_id} cdp={args.cdp}")

    sections: list[AcademySection] = []
    seen_ids: set[str] = set()

    cdp, ws, ws_url = open_cdp(args.cdp)
    print(f"[run] attaching to {ws_url}")
    try:
        entered, url = enter_module(cdp, args.module_id)
        print(f"[run]   in section view: {url}  (entered={entered})")
        if not entered:
            print("[run] WARN: never reached a /section/ URL; module may not be unlocked")
        else:
            # "Revisit Module" returns the operator to whichever section they
            # last visited; for a reproducible walk we rewind to section 1/N.
            rewound = go_to_first_section(cdp)
            print(f"[run]   rewound to first section: {rewound}")

        for hop in range(args.max_sections):
            scraped = scrape_section(cdp)
            section = build_section_from_scrape(scraped)
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
            if not click_next(cdp):
                print("[run] no Next button; stopping walk")
                break
            time.sleep(2)
    finally:
        try:
            ws.close()
        except Exception:
            pass

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
            # study_only: we never submit; accepted=False unconditionally.
            submissions.append((str(args.module_id), ans, False))

    module = AcademyModule(
        id=str(args.module_id),
        title=args.module_title or f"Module {args.module_id}",
        tier=0,
        sections=sections,
        category="general",
    )
    demo = session_to_demonstration(
        module, submissions, study_only=True,
        extra_metadata={"cdp_endpoint": args.cdp},
    )
    args.auto_demo_dir.mkdir(parents=True, exist_ok=True)
    out = args.auto_demo_dir / f"academy_module_{args.module_id}.msgpack.gz"
    save_demonstration(demo, out, compress=True)
    print(f"[run] wrote demo -> {out}")
    print(f"[run]   sections={len(sections)} questions={n_questions} "
          f"turns={len(demo.turns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
