"""Tests for the academy auto-learner package."""

from __future__ import annotations

from pathlib import Path

import pytest

from htbrl.academy.answerer import HeuristicAnswerer, _jaccard, _tokenize
from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.curriculum import (
    eligible_modules,
    is_eligible,
    next_module,
    progress_summary,
)
from htbrl.academy.orchestrator import AutoLearner, OrchestratorConfig
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySandbox,
    AcademySection,
    ProgressState,
    QuestionType,
)
from htbrl.academy.session import (
    AcademyCredentials,
    MockAcademySession,
)
from htbrl.data.demo_dataset import load_demonstration


# ---- fixtures ---------------------------------------------------------------


def _make_module(
    mod_id: str = "m1",
    tier: int = 0,
    cubes_to_unlock: int = 0,
    cubes_reward: int = 10,
    prereqs: list[str] | None = None,
) -> AcademyModule:
    return AcademyModule(
        id=mod_id,
        title=f"Module {mod_id}",
        tier=tier,
        cubes_to_unlock=cubes_to_unlock,
        cubes_reward=cubes_reward,
        prerequisites=prereqs or [],
        sections=[
            AcademySection(
                id=f"{mod_id}.s1",
                title="basics",
                body_text=(
                    "The pwd command prints the current working directory. "
                    'The "ls" command lists files inside a directory.'
                ),
                code_blocks=["echo HTB{example}"],
                questions=[
                    AcademyQuestion(
                        id=f"{mod_id}.q1",
                        prompt="Which command lists files in a directory?",
                        type=QuestionType.MULTIPLE_CHOICE,
                        multiple_choice_options=["ls", "pwd", "cat", "rm"],
                    ),
                    AcademyQuestion(
                        id=f"{mod_id}.q2",
                        prompt="Which command prints the current working directory?",
                        type=QuestionType.TEXT,
                    ),
                ],
            ),
        ],
    )


# ---- credentials redaction --------------------------------------------------


def test_credentials_repr_redacts_password():
    c = AcademyCredentials(username="alice", password="hunter2")
    r = repr(c)
    assert "hunter2" not in r
    assert "***" in r
    assert "alice" in r


# ---- dataclasses ------------------------------------------------------------


def test_mc_question_requires_options():
    with pytest.raises(ValueError, match="no options"):
        AcademyQuestion(
            id="q",
            prompt="?",
            type=QuestionType.MULTIPLE_CHOICE,
            multiple_choice_options=[],
        )


def test_module_all_questions_collects_across_sections():
    mod = AcademyModule(
        id="x", title="x", tier=0,
        sections=[
            AcademySection(id="s1", title="t", body_text="",
                           questions=[AcademyQuestion(id="q1", prompt="p", type=QuestionType.TEXT)]),
            AcademySection(id="s2", title="t", body_text="",
                           questions=[AcademyQuestion(id="q2", prompt="p", type=QuestionType.TEXT)]),
        ],
    )
    assert len(mod.all_questions) == 2


# ---- mock session -----------------------------------------------------------


def test_mock_login_and_logout():
    sess = MockAcademySession(modules=[], state=ProgressState(user_id="u"))
    assert not sess.is_logged_in()
    sess.login(AcademyCredentials("u", "p"))
    assert sess.is_logged_in()
    sess.close()
    assert not sess.is_logged_in()


def test_mock_login_rejects_empty_creds():
    sess = MockAcademySession(modules=[], state=ProgressState(user_id="u"))
    with pytest.raises(ValueError):
        sess.login(AcademyCredentials("", ""))


def test_mock_start_module_requires_cubes():
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(modules=[_make_module(cubes_to_unlock=10)], state=state)
    sess.login(AcademyCredentials("u", "p"))
    with pytest.raises(RuntimeError, match="insufficient cubes"):
        sess.start_module("m1")


def test_mock_submit_in_study_only_does_not_mutate_state():
    """study_only=True: submission_log records the attempt but state is untouched."""
    mod = _make_module()
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(
        modules=[mod], state=state, study_only=True,
        accept_predicate=lambda mid, ans: True,
    )
    sess.login(AcademyCredentials("u", "p"))
    sess.start_module("m1")
    ans = AcademyAnswer(
        question_id="m1.q1", answer_text="ls", confidence=1.0, method="mc_match"
    )
    accepted = sess.submit_answer("m1", ans)
    assert accepted is True
    assert sess.submission_log == [("m1", ans, True)]
    assert state.cubes_balance == 0
    assert "m1" not in state.completed_module_ids


def test_mock_submit_in_auto_mode_marks_completion_when_all_questions_answered():
    mod = _make_module()
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(
        modules=[mod], state=state, study_only=False,
        accept_predicate=lambda mid, ans: True,
    )
    sess.login(AcademyCredentials("u", "p"))
    sess.start_module("m1")
    for q in mod.all_questions:
        sess.submit_answer("m1", AcademyAnswer(
            question_id=q.id, answer_text="ls", confidence=0.9, method="mc_match"
        ))
    assert state.has_completed("m1")
    assert state.cubes_balance == mod.cubes_reward


# ---- answerer ---------------------------------------------------------------


def test_jaccard_basic():
    assert _jaccard(set(), set("abc")) == 0.0
    assert _jaccard(set("abc"), set("abc")) == 1.0


def test_tokenize_strips_short():
    assert _tokenize("a in this is the cat sat") == {"this", "the", "cat", "sat"}


def test_answerer_mc_picks_overlapping_option():
    section = AcademySection(
        id="s", title="t",
        body_text=(
            'The "ls" command lists files. '
            "It accepts options like -l for long listing."
        ),
    )
    q = AcademyQuestion(
        id="q", prompt="Which command lists files?",
        type=QuestionType.MULTIPLE_CHOICE,
        multiple_choice_options=["ls", "pwd", "cd", "rm"],
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.answer_text == "ls"
    assert ans.method == "mc_match"


def test_answerer_text_extracts_quoted_span():
    section = AcademySection(
        id="s", title="t",
        body_text='The current working directory can be printed with "pwd".',
    )
    q = AcademyQuestion(
        id="q", prompt="Which command prints the current working directory?",
        type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert "pwd" in ans.answer_text


def test_answerer_flag_no_runner_skips():
    section = AcademySection(
        id="s", title="t", body_text="",
        code_blocks=["echo HTB{flag}"],
    )
    q = AcademyQuestion(
        id="q", prompt="What's the flag?", type=QuestionType.FLAG,
    )
    ans = HeuristicAnswerer(sandbox_runner=None).answer(q, section)
    assert ans.method == "skipped"
    assert ans.confidence == 0.0


def test_answerer_flag_with_runner_extracts_htb_flag():
    section = AcademySection(
        id="s", title="t", body_text="",
        code_blocks=["cat /flag.txt"],
    )
    q = AcademyQuestion(
        id="q", prompt="What's the flag?", type=QuestionType.FLAG,
    )

    def runner(cmd: str) -> str:
        assert cmd == "cat /flag.txt"
        return "HTB{the_real_flag_here}\n"

    ans = HeuristicAnswerer(sandbox_runner=runner).answer(q, section)
    assert ans.answer_text == "HTB{the_real_flag_here}"
    assert ans.confidence > 0.9
    assert ans.sandbox_command == "cat /flag.txt"


def test_answerer_unsupported_returns_skip():
    section = AcademySection(id="s", title="t", body_text="anything")
    q = AcademyQuestion(id="q", prompt="?", type=QuestionType.UNSUPPORTED)
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.method == "skipped"


# ---- curriculum -------------------------------------------------------------


def test_eligibility_blocks_completed():
    mod = _make_module()
    state = ProgressState(user_id="u", completed_module_ids=["m1"])
    assert not is_eligible(mod, state)


def test_eligibility_requires_prereqs():
    mod = _make_module(mod_id="m2", prereqs=["m1"])
    state = ProgressState(user_id="u")
    assert not is_eligible(mod, state)
    state.completed_module_ids.append("m1")
    assert is_eligible(mod, state)


def test_eligibility_requires_cubes():
    mod = _make_module(cubes_to_unlock=10)
    state = ProgressState(user_id="u", cubes_balance=5)
    assert not is_eligible(mod, state)
    state.cubes_balance = 10
    assert is_eligible(mod, state)


def test_next_module_picks_lowest_tier_then_cheapest():
    mods = [
        _make_module(mod_id="A", tier=1, cubes_to_unlock=5),
        _make_module(mod_id="B", tier=0, cubes_to_unlock=20),
        _make_module(mod_id="C", tier=0, cubes_to_unlock=5),
    ]
    state = ProgressState(user_id="u", cubes_balance=100)
    assert next_module(mods, state).id == "C"


def test_next_module_honors_preferred_order():
    mods = [
        _make_module(mod_id="A", tier=0, cubes_to_unlock=5),
        _make_module(mod_id="B", tier=0, cubes_to_unlock=5),
    ]
    state = ProgressState(user_id="u", cubes_balance=100)
    assert next_module(mods, state, preferred_order=["B"]).id == "B"


def test_next_module_returns_none_when_nothing_eligible():
    mods = [_make_module(mod_id="A", cubes_to_unlock=999)]
    state = ProgressState(user_id="u", cubes_balance=0)
    assert next_module(mods, state) is None


def test_progress_summary_basic():
    mods = [_make_module(mod_id="A"), _make_module(mod_id="B"), _make_module(mod_id="C")]
    state = ProgressState(user_id="u", completed_module_ids=["A"], cubes_balance=10)
    s = progress_summary(mods, state)
    assert s["n_total"] == 3
    assert s["n_completed"] == 1


# ---- orchestrator end-to-end -----------------------------------------------


def test_orchestrator_study_only_writes_demo_no_submission(tmp_path: Path):
    mod = _make_module()
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(modules=[mod], state=state, study_only=True)
    sess.login(AcademyCredentials("u", "p"))

    learner = AutoLearner(
        session=sess,
        cfg=OrchestratorConfig(
            study_only=True, auto_demo_dir=tmp_path, max_modules_per_run=1,
        ),
    )
    result = learner.run()

    assert result.modules_attempted == ["m1"]
    assert result.modules_completed == []  # study_only -> no completion
    assert len(result.demos_written) == 1
    demo_path = result.demos_written[0]
    assert demo_path.exists()
    # Loaded demo carries academy metadata
    demo = load_demonstration(demo_path)
    assert demo.target_id == "htb-academy:m1"
    assert demo.metadata.get("source") == "htb_academy_auto_learner"
    assert demo.metadata.get("study_only") is True


def test_orchestrator_auto_submit_completes_module(tmp_path: Path):
    mod = _make_module()
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(
        modules=[mod], state=state, study_only=False,
        accept_predicate=lambda mid, ans: True,
    )
    sess.login(AcademyCredentials("u", "p"))

    learner = AutoLearner(
        session=sess,
        cfg=OrchestratorConfig(
            study_only=False,
            submit_confidence_threshold=0.0,  # always submit
            manual_review_threshold=0.0,
            auto_demo_dir=tmp_path,
            max_modules_per_run=1,
        ),
    )
    result = learner.run()
    assert "m1" in result.modules_completed
    assert state.cubes_balance == mod.cubes_reward


def test_orchestrator_pauses_on_low_confidence(tmp_path: Path):
    """Force the answerer below the manual_review_threshold so we get a manual_review item."""
    mod = AcademyModule(
        id="m1", title="x", tier=0,
        sections=[
            AcademySection(
                id="s1", title="t",
                # body has ZERO overlap with the question prompt -> low confidence text answer
                body_text="cats sleep on mats.",
                questions=[AcademyQuestion(
                    id="m1.q1", prompt="Quantum chromodynamics?", type=QuestionType.TEXT
                )],
            ),
        ],
    )
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(modules=[mod], state=state, study_only=True)
    sess.login(AcademyCredentials("u", "p"))
    learner = AutoLearner(
        session=sess,
        cfg=OrchestratorConfig(
            study_only=True,
            manual_review_threshold=0.5,
            auto_demo_dir=tmp_path,
            max_modules_per_run=1,
        ),
    )
    result = learner.run()
    assert len(result.manual_review) == 1


# ---- auto_demo_writer -------------------------------------------------------


def test_session_to_demonstration_emits_synthetic_tool():
    mod = _make_module()
    answer = AcademyAnswer(
        question_id="m1.q1", answer_text="ls", confidence=0.9, method="mc_match"
    )
    submissions = [("m1", answer, True)]
    demo = session_to_demonstration(mod, submissions, study_only=True)
    assert demo.matrix == "enterprise"
    assert demo.target_id == "htb-academy:m1"
    assert any(t.action_tool_name == "academy_answer" for t in demo.turns)
    assert demo.outcome.foothold is True


def test_session_to_demonstration_includes_sandbox_turn():
    """When the answerer ran a sandbox command, the demo should contain that turn."""
    mod = AcademyModule(
        id="m1", title="x", tier=0,
        sections=[
            AcademySection(
                id="s1", title="t", body_text="",
                questions=[AcademyQuestion(
                    id="m1.q1", prompt="flag?", type=QuestionType.FLAG
                )],
            ),
        ],
    )
    answer = AcademyAnswer(
        question_id="m1.q1",
        answer_text="HTB{x}",
        confidence=0.9,
        method="sandbox_cmd",
        sandbox_command="cat /flag.txt",
        sandbox_output="HTB{x}\n",
    )
    submissions = [("m1", answer, True)]
    demo = session_to_demonstration(mod, submissions, study_only=False)
    tool_names = [t.action_tool_name for t in demo.turns]
    assert "academy_sandbox_cmd" in tool_names
    assert "academy_answer" in tool_names
