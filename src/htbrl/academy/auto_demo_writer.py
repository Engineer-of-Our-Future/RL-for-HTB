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
) -> DemoTurn:
    """Render one Q+A as a DemoTurn the BC trainer can learn from.

    The 'tool' here is a synthetic 'academy_answer' marker since academy
    answers don't map cleanly to the registry's pentest tools. BC training
    can choose to ignore academy_answer turns or treat them as a separate
    head-target via a small extension to the action vocab.
    """
    obs_text = (
        f"[academy:{section_title}] {question.prompt}"
        if section_title else f"[academy] {question.prompt}"
    )
    if question.multiple_choice_options:
        obs_text += "\nOptions: " + " | ".join(question.multiple_choice_options)

    reward = _reward_for_answer(answer, accepted, question.points)
    return DemoTurn(
        obs_text=obs_text,
        action_tool_id=-1,        # synthetic; not in the registry
        action_tool_name="academy_answer",
        action_slots={
            "answer": answer.answer_text,
            "method": answer.method,
            "confidence": answer.confidence,
        },
        action_render=f"academy_answer({answer.method}): {answer.answer_text!r}",
        reward=reward,
        techniques_attempted=[],
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
) -> Demonstration:
    """Roll a list of submissions into one Demonstration."""
    turns: list[DemoTurn] = []
    questions_by_id = {q.id: (q, s.title) for s in module.sections for q in s.questions}

    foothold = False
    user_flag = False
    root_flag = False

    for mod_id, ans, accepted in submissions:
        if mod_id != module.id:
            continue
        q_info = questions_by_id.get(ans.question_id)
        if q_info is None:
            continue
        question, section_title = q_info
        # Sandbox command turn first (the 'how'), then the answer turn (the 'what').
        cmd_turn = sandbox_cmd_to_demo_turn(ans, accepted)
        if cmd_turn is not None:
            turns.append(cmd_turn)
        turns.append(answer_to_demo_turn(question, ans, accepted, section_title))
        if accepted:
            foothold = True
            if question.type.value == "flag":
                # First accepted flag = user, second = root, mirroring the
                # HTB box conventions.
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
