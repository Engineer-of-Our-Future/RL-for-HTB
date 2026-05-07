"""Wizard mode for academy module questions.

The flow combines two answering paths in one walker so the operator can stay
in the loop *while* the model gets to act on its own when it's confident:

1. Walk every section of a target module via CDP-attached Chrome (same plumbing
   as ``htb_academy_run.py``; see ``src/htbrl/academy/cdp_walker.py``).
2. For each question, run :class:`HeuristicAnswerer.propose` to get a ranked
   list of candidate answers.
3. **Auto-submit path** (``--auto-submit``): if the top candidate's confidence
   meets ``--auto-confidence``, fill it into the page and click Submit
   without asking. Lab-flag-shaped questions are *always* excluded - per the
   project rule, the model must never submit a lab flag (it would risk the
   research account being banned).
4. **Manual path**: otherwise show the prompt + theory excerpt + top-N
   candidates and prompt the operator. The operator can:
       <Enter>   accept the top candidate
       1..9      pick an alternate candidate by rank
       text      type a custom answer
       s         skip this question (study only)
       auto      switch to auto-submit for the rest of this module
       q         quit the walker (saves what we have so far)
5. Selected answers are submitted via the page DOM (Vue 3 reactive set + click
   Submit), and the result (accepted / rejected / pending) is polled.
6. Save a Demonstration tagged with method strings the demo writer can carry
   into BC training data: ``wizard_auto`` / ``wizard_accepted`` /
   ``wizard_alt`` / ``wizard_user`` / ``wizard_skip``.

Hard requirements:

  - ``scripts/start_chrome_for_htb.ps1`` has been run and the operator is
    logged into HTB Academy in that Chrome window.
  - The Chrome debug endpoint is reachable at ``$HTBRL_ACADEMY_CDP``
    (default ``http://127.0.0.1:9222``).
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
    already_answered_flags,
    build_section_from_scrape,
    click_next,
    enter_module,
    fetch_module_via_api,
    go_to_first_section,
    is_lab_flag_question,
    open_cdp,
    parse_cheatsheet_markdown,
    read_cube_balance,
    scrape_section,
    submit_answer_in_dom,
)
from htbrl.academy.curriculum import check_unlock_gate
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySection,
    QuestionType,
)
from htbrl.data.demo_dataset import save_demonstration


# ---- terminal UI helpers ----------------------------------------------------


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _render_candidates(candidates: list[AcademyAnswer]) -> str:
    if not candidates:
        return "  (no candidates)"
    out = []
    for i, c in enumerate(candidates, start=1):
        out.append(
            f"  {i}. [{c.confidence:.2f}] {c.method:24s} {c.answer_text!r}"
        )
        if c.rationale:
            out.append(f"       why: {_truncate(c.rationale, 100)}")
    return "\n".join(out)


def _show_question_panel(
    section: AcademySection, question: AcademyQuestion,
    candidates: list[AcademyAnswer],
) -> None:
    print()
    print("=" * 78)
    print(f"[wizard] Section {section.section_index}/{section.section_total} {section.title!r}")
    print(f"[wizard] Question {question.id}:")
    print(f"  > {question.prompt}")
    if question.cubes_reward or question.hp_reward:
        print(f"  ({question.cubes_reward} cubes / +{question.hp_reward} HP)")
    print()
    if section.body_text:
        excerpt = section.body_text[-1200:] if len(section.body_text) > 1200 else section.body_text
        print("  Theory excerpt (last 1200 chars):")
        for line in excerpt.splitlines()[-12:]:
            print(f"    {line}")
        print()
    if section.inline_code:
        codes = ", ".join(repr(c) for c in section.inline_code[:12])
        print(f"  Inline code: {codes}")
    if section.bullet_lists:
        first = section.bullet_lists[0]
        bullets_preview = " | ".join(_truncate(b, 60) for b in first[:4])
        print(f"  Bullets[0] ({len(first)} items): {bullets_preview}")
    print()
    print("  Top candidates:")
    print(_render_candidates(candidates))
    print()
    if is_lab_flag_question(question):
        print("  *** LAB FLAG question detected -- auto-submit DISABLED. ***")
        print("  *** Model proposes only; you must verify and submit by hand. ***")
    print()


def _prompt_user(candidates: list[AcademyAnswer]) -> tuple[str, str]:
    """Read one wizard input. Returns (action, value).

    action ∈ {"accept", "alt", "custom", "skip", "auto", "quit"}.
    value is the chosen answer text (empty for skip/quit/auto).
    """
    n_cand = len(candidates)
    keys = "[Enter]=accept top  [1..%d]=alt  [text]=custom  [s]=skip  [auto]=auto-rest  [q]=quit" % (
        max(n_cand, 1)
    )
    print(keys)
    try:
        line = input("  > ").strip()
    except EOFError:
        return "quit", ""
    if not line:
        if not candidates:
            return "skip", ""
        return "accept", candidates[0].answer_text
    low = line.lower()
    if low in ("q", "quit", "exit"):
        return "quit", ""
    if low in ("s", "skip"):
        return "skip", ""
    if low in ("auto", "all"):
        return "auto", ""
    if line.isdigit():
        idx = int(line) - 1
        if 0 <= idx < n_cand:
            return "alt", candidates[idx].answer_text
        # Fall through and treat as custom
    return "custom", line


def _method_tag(action: str) -> str:
    return {
        "accept": "wizard_accepted",
        "alt":    "wizard_alt",
        "custom": "wizard_user",
        "skip":   "wizard_skip",
        "auto-skip": "wizard_skip_already_answered",
    }.get(action, "wizard_user")


# ---- main loop --------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module-id", type=int, required=True,
                   help="numeric module ID, e.g. 18 for Linux Fundamentals")
    p.add_argument("--module-title", default="",
                   help="optional title used in the Demonstration")
    p.add_argument("--cdp", default=os.environ.get("HTBRL_ACADEMY_CDP", "http://127.0.0.1:9222"))
    p.add_argument("--max-sections", type=int, default=50)
    p.add_argument("--auto-demo-dir", type=Path, default=Path("data/auto_demos"))
    p.add_argument(
        "--auto-submit", action="store_true",
        help="when set, fill+submit confident answers (>= --auto-confidence) "
             "into the page automatically. Lab flag questions are NEVER "
             "auto-submitted regardless of this flag.",
    )
    p.add_argument(
        "--auto-confidence", type=float, default=0.85,
        help="minimum top-candidate confidence to auto-submit (default 0.85)",
    )
    p.add_argument(
        "--top-n", type=int, default=5,
        help="how many ranked candidates to show in the wizard (default 5)",
    )
    p.add_argument(
        "--no-prompt", action="store_true",
        help="purely automated mode: auto-submit everything that meets the "
             "confidence threshold, skip everything else. No operator prompts.",
    )
    p.add_argument(
        "--skip-already-answered", action="store_true", default=True,
        help="skip questions HTB shows as already answered (default True)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    print(f"[wizard] module_id={args.module_id} cdp={args.cdp} "
          f"auto_submit={args.auto_submit} auto_conf={args.auto_confidence}")

    answerer = HeuristicAnswerer()
    submissions: list[tuple[str, AcademyAnswer, bool]] = []
    sections: list[AcademySection] = []
    seen_ids: set[str] = set()
    auto_for_remainder = False  # toggled when operator types "auto"

    try:
        cdp, ws, ws_url = open_cdp(args.cdp)
    except Exception as exc:
        print(f"[wizard] failed to attach to {args.cdp}: {exc}")
        print("[wizard] is start_chrome_for_htb.ps1 running?")
        return 2
    print(f"[wizard] attached to {ws_url}")

    cubes_before: int | None = None
    cheat_sheet_rows: list[dict[str, str]] = []
    api_prelude = ""
    api_conclusion = ""
    api_takeaways = ""
    api_title = ""
    # Minimal default module so the finalize block always has something to
    # save - replaced inside the try once API metadata is fetched.
    module = AcademyModule(
        id=str(args.module_id),
        title=args.module_title or f"Module {args.module_id}",
        tier=0, sections=sections, category="general",
    )
    try:
        # Capture cube balance BEFORE we attempt anything, for the unlock-gate
        # readiness report at the end of the run.
        cubes_before = read_cube_balance(cdp)
        if cubes_before is not None:
            print(f"[wizard] cubes_balance before run: {cubes_before}")

        # Fetch module-level metadata via the academy's authenticated API:
        # cheat sheet (markdown table of canonical commands), prelude,
        # conclusion, takeaways. The cheat sheet is high-precision context
        # for the answerer; prelude/conclusion become extra theory bytes.
        api = fetch_module_via_api(cdp, args.module_id)
        if api and not api.get("__error"):
            cheat_sheet_rows = parse_cheatsheet_markdown(api.get("cheatsheet") or "")
            api_prelude = (api.get("prelude") or "").strip()
            api_conclusion = (api.get("conclusion") or "").strip()
            api_takeaways = (api.get("takeaways") or "").strip()
            api_title = (api.get("name") or "").strip()
            print(f"[wizard] API metadata: cheat_rows={len(cheat_sheet_rows)} "
                  f"prelude={len(api_prelude)} title={api_title!r}")
        else:
            print(f"[wizard] API metadata unavailable: {api}")

        # Build the module wrapper ONCE - reused by the answerer (every
        # question) AND by session_to_demonstration at the end. The
        # ``sections`` list is shared by reference so as the walk discovers
        # new sections they're visible to both consumers without rewriting.
        module = AcademyModule(
            id=str(args.module_id),
            title=args.module_title or api_title or f"Module {args.module_id}",
            tier=0, sections=sections, category="general",
            cheat_sheet=cheat_sheet_rows,
            prelude=api_prelude,
            conclusion=api_conclusion,
            takeaways=api_takeaways,
        )

        entered, url = enter_module(cdp, args.module_id)
        print(f"[wizard] entered module: {url}  (entered={entered})")
        if not entered:
            print("[wizard] could not reach a /section/ URL; module may need to be unlocked manually.")
        else:
            rewound = go_to_first_section(cdp)
            print(f"[wizard] rewound to first section: {rewound}")

        for hop in range(args.max_sections):
            scraped = scrape_section(cdp)
            section = build_section_from_scrape(scraped)
            answered_flags = already_answered_flags(scraped)
            if section.id in seen_ids:
                print(f"[wizard] section {section.id!r} already seen; stopping walk")
                break
            seen_ids.add(section.id)
            sections.append(section)
            print(
                f"[wizard]   sec {section.section_index}/{section.section_total} "
                f"{section.title!r} body={len(section.body_text)} "
                f"questions={len(section.questions)}"
            )

            for q_idx, question in enumerate(section.questions):
                already = q_idx < len(answered_flags) and answered_flags[q_idx]
                if already and args.skip_already_answered:
                    print(f"[wizard]   q={question.id!r} already answered; skipping")
                    submissions.append((str(args.module_id), AcademyAnswer(
                        question_id=question.id, answer_text="",
                        confidence=0.0, method="wizard_skip_already_answered",
                        rationale="HTB shows this question as already answered",
                    ), True))  # accepted=True because the academy already counts it
                    continue

                candidates = answerer.propose(
                    question, section, top_n=args.top_n,
                    module=module,
                )
                top = candidates[0] if candidates else None
                lab_flag = is_lab_flag_question(question)
                # -- decide path -----------------------------------------------------
                if lab_flag:
                    # NEVER auto-submit lab flags; always defer.
                    if args.no_prompt:
                        print(f"[wizard]   q={question.id!r} LAB FLAG; --no-prompt set; skipping")
                        submissions.append((str(args.module_id), AcademyAnswer(
                            question_id=question.id, answer_text=top.answer_text if top else "",
                            confidence=top.confidence if top else 0.0,
                            method="wizard_skip_lab_flag",
                            rationale="lab flag - operator not present (no-prompt); deferred",
                        ), False))
                        continue
                    _show_question_panel(section, question, candidates)
                    print("  This is a LAB FLAG question. The model will NOT submit it.")
                    print("  Type the flag yourself or 's' to skip.")
                    action, value = _prompt_user(candidates)
                    method = _method_tag(action)
                    if action in ("accept", "alt"):
                        # Operator pressed Enter / 1..N - we still don't auto-fill
                        # the page; we record the operator's chosen answer
                        # locally (so the demo carries the model's proposal +
                        # operator pick) but the operator must submit by hand.
                        method = "wizard_user_lab_flag"
                    submissions.append((str(args.module_id), AcademyAnswer(
                        question_id=question.id, answer_text=value,
                        confidence=1.0 if action == "custom" else (top.confidence if top else 0.0),
                        method=method,
                        rationale="lab flag: operator handles submission",
                    ), False))
                    if action == "quit":
                        raise KeyboardInterrupt
                    continue

                if (auto_for_remainder or args.no_prompt or args.auto_submit) and top \
                        and top.confidence >= args.auto_confidence:
                    print(f"[wizard]   q={question.id!r} AUTO method={top.method} "
                          f"conf={top.confidence:.2f} ans={top.answer_text!r}")
                    state, detail = ("pending", "auto-submit disabled")
                    if args.auto_submit or auto_for_remainder or args.no_prompt:
                        state, detail = submit_answer_in_dom(cdp, q_idx, top.answer_text)
                        print(f"[wizard]     -> {state} ({detail})")
                    accepted = (state == "accepted")
                    submissions.append((str(args.module_id), AcademyAnswer(
                        question_id=question.id, answer_text=top.answer_text,
                        confidence=top.confidence,
                        method="wizard_auto",
                        rationale=f"{top.rationale} | submit_state={state}",
                    ), accepted))
                    continue

                if args.no_prompt:
                    # Below threshold and no-prompt: skip.
                    print(f"[wizard]   q={question.id!r} below threshold "
                          f"({(top.confidence if top else 0.0):.2f} < {args.auto_confidence}); "
                          f"skipping (no-prompt)")
                    submissions.append((str(args.module_id), AcademyAnswer(
                        question_id=question.id,
                        answer_text=top.answer_text if top else "",
                        confidence=top.confidence if top else 0.0,
                        method="wizard_skip_low_conf",
                        rationale=f"top conf {(top.confidence if top else 0.0):.2f} below {args.auto_confidence}",
                    ), False))
                    continue

                # -- interactive path -----------------------------------------------
                _show_question_panel(section, question, candidates)
                action, value = _prompt_user(candidates)
                if action == "quit":
                    raise KeyboardInterrupt
                if action == "auto":
                    auto_for_remainder = True
                    print("[wizard]   auto-mode ON for the rest of this module.")
                    # Re-evaluate this question via the auto branch.
                    if top and top.confidence >= args.auto_confidence:
                        state, detail = submit_answer_in_dom(cdp, q_idx, top.answer_text)
                        print(f"[wizard]     -> {state} ({detail})")
                        accepted = (state == "accepted")
                        submissions.append((str(args.module_id), AcademyAnswer(
                            question_id=question.id, answer_text=top.answer_text,
                            confidence=top.confidence,
                            method="wizard_auto_after_switch",
                            rationale=f"{top.rationale} | submit_state={state}",
                        ), accepted))
                    else:
                        print(f"[wizard]     top conf {(top.confidence if top else 0.0):.2f} "
                              f"< {args.auto_confidence}; auto-skip")
                        submissions.append((str(args.module_id), AcademyAnswer(
                            question_id=question.id,
                            answer_text=top.answer_text if top else "",
                            confidence=top.confidence if top else 0.0,
                            method="wizard_skip_low_conf",
                            rationale="auto-skip after switch (low confidence)",
                        ), False))
                    continue
                if action == "skip":
                    submissions.append((str(args.module_id), AcademyAnswer(
                        question_id=question.id, answer_text="",
                        confidence=0.0, method="wizard_skip",
                        rationale="operator skipped",
                    ), False))
                    continue
                # accept / alt / custom -> fill + submit
                state, detail = submit_answer_in_dom(cdp, q_idx, value)
                print(f"[wizard]   -> {state} ({detail})")
                accepted = (state == "accepted")
                if action == "accept" and top is not None:
                    conf = top.confidence
                elif action == "alt":
                    # find the chosen candidate's recorded confidence
                    conf = next(
                        (c.confidence for c in candidates if c.answer_text == value),
                        0.5,
                    )
                else:
                    # custom answer typed by operator -> treat as ground-truth confidence 1.0
                    conf = 1.0
                submissions.append((str(args.module_id), AcademyAnswer(
                    question_id=question.id, answer_text=value,
                    confidence=conf,
                    method=_method_tag(action),
                    rationale=f"operator {action}; submit_state={state}",
                ), accepted))

            # advance to next section
            if section.section_total and section.section_index >= section.section_total:
                print("[wizard] reached final section")
                break
            if not click_next(cdp):
                print("[wizard] no Next button; stopping walk")
                break
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[wizard] interrupted; saving demo with progress so far")

    # Re-read cube balance BEFORE closing the websocket so the gate
    # readiness check has fresh data.
    cubes_after: int | None = None
    try:
        cubes_after = read_cube_balance(cdp)
        if cubes_after is not None:
            print(f"[wizard] cubes_balance after run: {cubes_after}  "
                  f"(delta {cubes_after - (cubes_before or 0):+d})")
    finally:
        try:
            ws.close()
        except Exception:
            pass

    # -- finalize ----------------------------------------------------------------
    n_q = sum(len(s.questions) for s in sections)
    n_attempts = len(submissions)
    n_accepted = sum(1 for _, _, ok in submissions if ok)
    print(f"[wizard] done: sections={len(sections)} questions={n_q} "
          f"attempts={n_attempts} accepted={n_accepted}")

    # Reuse the wrapper we built before the walk - sections were populated
    # in-place during the walk, so the same instance is fully formed now.
    extra = {
        "cdp_endpoint": args.cdp,
        "wizard": True,
        "auto_submit": args.auto_submit,
        "auto_confidence": args.auto_confidence,
        "no_prompt": args.no_prompt,
        "n_accepted": n_accepted,
    }
    demo = session_to_demonstration(
        module, submissions, study_only=not args.auto_submit, extra_metadata=extra,
    )
    args.auto_demo_dir.mkdir(parents=True, exist_ok=True)
    out = args.auto_demo_dir / f"academy_module_{args.module_id}_wizard.msgpack.gz"
    save_demonstration(demo, out)
    print(f"[wizard] wrote demo -> {out}")
    print(f"[wizard]   sections={len(sections)} questions={n_q} turns={len(demo.turns)}")

    # -- unlock gate readiness report --------------------------------------------
    # Per the operator rule "open new module only if all questions are answered
    # and cube balance are updated", we run the gate at end-of-walk and report
    # whether the next module would be safe to open. This is informational for
    # the wizard (single-module per invocation); the orchestrator enforces the
    # check programmatically before each new module open.
    attempted_qids = {a.question_id for _, a, _ in submissions}
    gate = check_unlock_gate(
        current_module=module,
        answered_question_ids=attempted_qids,
        cube_balance_before=cubes_before if cubes_before is not None else 0,
        cube_balance_after=cubes_after if cubes_after is not None else (cubes_before or 0),
        require_all_answered=True,
        require_cube_refresh=(cubes_before is not None and cubes_after is not None),
    )
    print(f"[wizard] unlock gate: allowed={gate.allowed} - {gate.reason}")
    if gate.allowed:
        print("[wizard] safe to start the next module.")
    else:
        print("[wizard] NOT safe to start a new module yet "
              "(refresh state and rerun, or relax gate flags).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
