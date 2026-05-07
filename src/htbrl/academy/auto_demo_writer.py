"""Convert an academy session log into ``Demonstration`` objects.

Each module attempt produces one Demonstration:
- ``matrix``: always 'enterprise' (HTB Academy is Enterprise-aligned)
- ``target_id``: 'htb-academy:<module_id>'
- ``turns``: one DemoTurn per question attempt with the rendered answer +
  reward. Sandbox-driven flag-questions also include a turn for the actual
  sandbox command they ran.
- ``outcome``: foothold=True if any question accepted; user_flag/root_flag
  set when corresponding flag-question was answered correctly.
- ``metadata``: includes the academy session's submission_log, agent's
  confidence per answer, and a top-level mode tag ('study_only' or 'auto_submit').

These demos go into ``data/auto_demos/`` (separate from manual ``data/demos/``)
so BC training can mix them at controlled ratios.
"""

from __future__ import annotations

import time
from typing import Iterable

from htbrl.academy.mitre_mapping import (
    techniques_for_module,
    techniques_for_section,
)
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
)
from htbrl.data.demo_dataset import (
    Demonstration,
    DemoOutcome,
    DemoTurn,
)


def answer_to_demo_turn(
    question: AcademyQuestion,
    answer: AcademyAnswer,
    accepted: bool,
    section_title: str = "",
    section_body: str = "",
    inline_code: list[str] | None = None,
    *,
    module_techniques: list[str] | None = None,
    section_techniques: list[str] | None = None,
) -> DemoTurn:
    """Render one Q+A as a DemoTurn the BC trainer can learn from.

    Crucially, the section's *theory text* travels with the question in
    ``obs_text`` so the policy learns the reading-comprehension mapping
    (theory -> answer), which is the user's stated goal for the academy
    track. We cap the theory at 4 KB to keep token counts bounded; if the
    section is longer, we keep the last 4 KB which is usually closer to the
    questions on the page.

    The 'tool' here is a synthetic 'academy_answer' marker since academy
    answers don't map cleanly to the registry's pentest tools. BC training
    can choose to ignore academy_answer turns or treat them as a separate
    head-target via a small extension to the action vocab.
    """
    parts: list[str] = []
    parts.append(f"[academy:{section_title}]" if section_title else "[academy]")
    if section_body:
        body = section_body if len(section_body) <= 4096 else section_body[-4096:]
        parts.append("## Theory\n" + body)
    if inline_code:
        parts.append("## Inline code: " + " | ".join(inline_code[:16]))
    parts.append("## Question\n" + question.prompt)
    if question.multiple_choice_options:
        parts.append("Options: " + " | ".join(question.multiple_choice_options))
    obs_text = "\n\n".join(parts)

    reward = _reward_for_answer(answer, accepted, question.points)
    # Per-question reward bonus reflects the academy's own +cubes / +HP signal.
    if accepted and (question.cubes_reward or question.hp_reward):
        reward += 0.01 * question.cubes_reward + 0.001 * question.hp_reward

    # ATT&CK tagging: an accepted answer demonstrates the technique was
    # internalized; a rejected/skipped answer still counts as "attempted" so
    # coverage metrics see negative samples too.
    section_tags = list(section_techniques) if section_techniques else []
    return DemoTurn(
        obs_text=obs_text,
        action_tool_id=-1,        # synthetic; not in the registry
        action_tool_name="academy_answer",
        action_slots={
            "answer": answer.answer_text,
            "method": answer.method,
            "confidence": answer.confidence,
            "cubes_reward": question.cubes_reward,
            "hp_reward": question.hp_reward,
        },
        action_render=f"academy_answer({answer.method}): {answer.answer_text!r}",
        reward=reward,
        techniques_attempted=section_tags,
        techniques_succeeded=section_tags if accepted else [],
    )


def section_read_turn(
    section,
    module_title: str = "",
    *,
    module_techniques: list[str] | None = None,
) -> DemoTurn:
    """Emit one synthetic 'I read this theory section' turn.

    Even sections with zero questions are valuable training data - the user
    explicitly noted "don't skip theory there are some critical areas
    sometimes!". A section_read turn captures the full section body, inline
    code spans, and bullet-list highlights into ``obs_text`` so BC can learn
    to attend to theory content even when no question follows.

    ATT&CK tagging: when ``module_techniques`` is provided we narrow it to
    the section via ``techniques_for_section`` and report it as
    ``techniques_attempted``. Reading theory does not by itself "succeed" at
    any technique, so ``techniques_succeeded`` stays empty.
    """
    parts: list[str] = []
    parts.append(
        f"[academy:{section.title}] (Section {section.section_index}/{section.section_total})"
        if section.title
        else "[academy section]"
    )
    body = section.body_text or ""
    if len(body) > 4096:
        body = body[-4096:]
    if body:
        parts.append("## Theory\n" + body)
    if section.inline_code:
        parts.append("## Inline code: " + " | ".join(section.inline_code[:24]))
    if section.bullet_lists:
        flat = " | ".join(
            ", ".join(lst[:8]) for lst in section.bullet_lists[:3]
        )
        parts.append("## Bullets: " + flat)
    if section.code_blocks:
        # Only the FIRST code block, capped, to keep turn size bounded.
        cb = section.code_blocks[0]
        parts.append("## Code:\n" + (cb[:512] + "...[truncated]" if len(cb) > 512 else cb))
    obs_text = "\n\n".join(parts)
    # Small positive reward: engaging with theory is valuable behavior, even
    # without a downstream question. Cap so it can't dominate the env reward.
    section_tags: list[str]
    if module_techniques:
        section_tags = techniques_for_section(section, module_techniques)
    else:
        section_tags = []
    return DemoTurn(
        obs_text=obs_text,
        action_tool_id=-1,
        action_tool_name="academy_section_read",
        action_slots={
            "section_index": section.section_index,
            "section_total": section.section_total,
            "title": section.title,
            "n_inline_code": len(section.inline_code),
            "n_bullet_lists": len(section.bullet_lists),
            "n_code_blocks": len(section.code_blocks),
            "n_questions": len(section.questions),
        },
        action_render=f"academy_read_section: {section.title!r}",
        reward=0.02,
        techniques_attempted=section_tags,
        techniques_succeeded=[],
    )


def module_intro_turn(
    module: AcademyModule,
    *,
    module_techniques: list[str] | None = None,
) -> DemoTurn | None:
    """Emit one synthetic 'I read the module intro' turn, if any.

    Combines the module's ``prelude`` (intro paragraph, what this module
    teaches), ``takeaways`` (learning objectives), and ``conclusion``
    (wrap-up summary) into a single training turn placed at the very
    start of the demo. These are pure theory text - no commands - so
    they don't belong in the cheat sheet turn but are still valuable
    BC signal because they often state the literal answer to summary
    questions verbatim.

    Returns None when the module has no intro/takeaways/conclusion text
    (e.g. theory-only modules whose API record is sparse). Caps total
    text at ~6 KB to stay token-bounded.
    """
    prelude = (module.prelude or "").strip()
    takeaways = (module.takeaways or "").strip()
    conclusion = (module.conclusion or "").strip()
    if not (prelude or takeaways or conclusion):
        return None
    parts: list[str] = [f"[academy-intro:{module.title}]"]

    def _add(label: str, text: str, cap: int) -> None:
        if not text:
            return
        if len(text) > cap:
            text = text[:cap] + "…"
        parts.append(f"## {label}\n{text}")

    _add("Prelude", prelude, 2_500)
    _add("Takeaways", takeaways, 2_000)
    _add("Conclusion", conclusion, 1_500)
    obs_text = "\n\n".join(parts)
    return DemoTurn(
        obs_text=obs_text,
        action_tool_id=-1,
        action_tool_name="academy_module_intro",
        action_slots={
            "module_id": module.id,
            "module_title": module.title,
            "has_prelude": bool(prelude),
            "has_takeaways": bool(takeaways),
            "has_conclusion": bool(conclusion),
        },
        action_render=f"academy_module_intro: {module.title!r}",
        # Modest reward: intro reading is valuable but cheaper than a
        # cheatsheet row that maps directly to a question answer.
        reward=0.03,
        techniques_attempted=list(module_techniques or []),
        techniques_succeeded=[],
    )


def cheat_sheet_turn(
    module: AcademyModule,
    *,
    module_techniques: list[str] | None = None,
) -> DemoTurn | None:
    """Emit one synthetic 'I read the module's cheat sheet' turn, if any.

    Returns None when the module has no cheatsheet rows (theory-only modules
    like "Intro To Academy" or "Learning Process" don't carry one). Otherwise
    renders the cheatsheet rows as a compact markdown table in ``obs_text``,
    capped at ~6 KB so a giant cheatsheet doesn't dominate token budgets.

    The cheatsheet is hugely valuable training data: it's the academy's own
    canonical command -> description map for the module, and many text
    questions are essentially "which row of the cheatsheet does this match?"
    BC training on this turn lets the policy attend to the table verbatim.
    Module-level ATT&CK techniques tag the turn so the technique-coverage
    report credits the demo for reading the canonical reference.

    Note: ``module_intro_turn`` separately captures prelude/takeaways/
    conclusion - those don't repeat here, so the cheat-sheet turn is
    purely the table.
    """
    rows = module.cheat_sheet or []
    if not rows:
        return None
    parts: list[str] = [f"[academy-cheatsheet:{module.title}]"]
    parts.append("## Cheatsheet")
    # Render as Markdown table for the policy's consumption (matches the
    # academy's own format, so BC can fall back on training-data verbatim).
    keys = list(rows[0].keys()) if rows else []
    parts.append("| " + " | ".join(keys) + " |")
    parts.append("|" + "|".join("---" for _ in keys) + "|")
    rendered = 0
    for row in rows:
        line = "| " + " | ".join(row.get(k, "") for k in keys) + " |"
        # Cap total cheatsheet text at ~6 KB to stay token-bounded.
        if sum(len(p) + 2 for p in parts) + len(line) > 6_500:
            parts.append(f"...[truncated; {len(rows) - rendered} more rows]")
            break
        parts.append(line)
        rendered += 1
    obs_text = "\n".join(parts)
    return DemoTurn(
        obs_text=obs_text,
        action_tool_id=-1,
        action_tool_name="academy_cheat_sheet",
        action_slots={
            "module_id": module.id,
            "module_title": module.title,
            "n_rows": len(rows),
            "n_rows_rendered": rendered,
            "columns": keys,
        },
        action_render=f"academy_cheat_sheet: {len(rows)} rows for {module.title!r}",
        reward=0.05,  # higher than section_read - the cheatsheet IS the answer key
        techniques_attempted=list(module_techniques or []),
        techniques_succeeded=[],
    )


def sandbox_cmd_to_demo_turn(
    answer: AcademyAnswer,
    accepted: bool,
) -> DemoTurn | None:
    """If the answer was produced by running a sandbox command, log THAT
    turn too (helps BC learn the underlying tool-using behavior).
    """
    if not answer.sandbox_command or answer.method != "sandbox_cmd":
        return None
    obs_text = (answer.sandbox_output or "")[:8192]
    return DemoTurn(
        obs_text=f"[sandbox-output]\n{obs_text}",
        action_tool_id=-1,
        action_tool_name="academy_sandbox_cmd",
        action_slots={"command": answer.sandbox_command},
        action_render=answer.sandbox_command,
        reward=0.05 if accepted else -0.01,
        techniques_attempted=[],
        techniques_succeeded=[],
    )


def session_to_demonstration(
    module: AcademyModule,
    submissions: Iterable[tuple[str, AcademyAnswer, bool]],
    *,
    study_only: bool = True,
    extra_metadata: dict | None = None,
    include_section_read_turns: bool = True,
) -> Demonstration:
    """Roll a list of submissions into one Demonstration.

    With ``include_section_read_turns=True`` (default), every section emits
    one ``academy_section_read`` turn carrying its theory + inline code +
    bullets BEFORE any question turns from that section. This way modules
    with zero questions still produce useful training data - critical when
    theory-only sections cover important concepts.
    """
    turns: list[DemoTurn] = []
    # Index for fast lookup; we still emit section-order turns by walking
    # ``module.sections`` so the demo's turn order matches the academy's
    # presentation order.
    submissions_by_qid: dict[str, list[tuple[str, AcademyAnswer, bool]]] = {}
    for mod_id, ans, accepted in submissions:
        if mod_id != module.id:
            continue
        submissions_by_qid.setdefault(ans.question_id, []).append((mod_id, ans, accepted))

    foothold = False
    user_flag = False
    root_flag = False

    # Precompute the module-level ATT&CK technique list once - section
    # narrowing reuses it for every section.
    module_techniques = techniques_for_module(module)

    # Emit a module-level intro turn (prelude + takeaways + conclusion)
    # FIRST. This is the canonical "what does this module teach" text and
    # often states the literal answer to summary questions.
    intro_turn = module_intro_turn(module, module_techniques=module_techniques)
    if intro_turn is not None:
        turns.append(intro_turn)
    # Then the cheat sheet (canonical command -> description map). BC
    # therefore sees the answer key BEFORE any questions, turning
    # "what command does X" into near-trivial pattern matches.
    cheat_turn = cheat_sheet_turn(module, module_techniques=module_techniques)
    if cheat_turn is not None:
        turns.append(cheat_turn)

    for section in module.sections:
        section_tags = techniques_for_section(section, module_techniques)
        if include_section_read_turns:
            turns.append(section_read_turn(
                section,
                module_title=module.title,
                module_techniques=module_techniques,
            ))
        for question in section.questions:
            for mod_id, ans, accepted in submissions_by_qid.get(question.id, []):
                cmd_turn = sandbox_cmd_to_demo_turn(ans, accepted)
                if cmd_turn is not None:
                    turns.append(cmd_turn)
                turns.append(answer_to_demo_turn(
                    question, ans, accepted,
                    section_title=section.title,
                    section_body=section.body_text,
                    inline_code=section.inline_code,
                    module_techniques=module_techniques,
                    section_techniques=section_tags,
                ))
                if accepted:
                    foothold = True
                    if question.type.value == "flag":
                        # First accepted flag = user, second = root, mirroring
                        # the HTB box conventions.
                        if not user_flag:
                            user_flag = True
                        else:
                            root_flag = True

    outcome = DemoOutcome(
        foothold=foothold,
        user_flag=user_flag,
        root_flag=root_flag,
        note=f"academy module: {module.title}",
    )

    md = {
        "source": "htb_academy_auto_learner",
        "module_id": module.id,
        "module_tier": module.tier,
        "study_only": study_only,
        "ts": time.time(),
        "n_questions": len(module.all_questions),
        "n_attempts": sum(1 for s in submissions if s[0] == module.id),
        # Module-level ATT&CK coverage so downstream tools (curriculum
        # weighting, coverage report) can see what this demo is teaching
        # without re-deriving it from the per-turn lists.
        "module_techniques": list(module_techniques),
        # Cheat-sheet shape so the coverage report can show "X rows" without
        # round-tripping through the turn list.
        "n_cheat_sheet_rows": len(module.cheat_sheet or []),
        "has_prelude": bool(module.prelude),
        "has_conclusion": bool(module.conclusion),
        "has_takeaways": bool(module.takeaways),
    }
    if extra_metadata:
        md.update(extra_metadata)

    return Demonstration(
        matrix="enterprise",
        target_id=f"htb-academy:{module.id}",
        turns=turns,
        outcome=outcome,
        metadata=md,
    )


def _reward_for_answer(answer: AcademyAnswer, accepted: bool, points: int) -> float:
    if answer.method == "skipped":
        return -0.01
    if accepted:
        # Bonus for low-confidence-but-accepted (the agent took a smart guess).
        risk_bonus = max(0.0, 0.5 - answer.confidence) * 0.2
        return 0.5 + 0.05 * max(points, 1) + risk_bonus
    # Wrong answer: small negative, scaled by confidence (overconfidence is penalized harder).
    return -0.1 - 0.2 * answer.confidence
