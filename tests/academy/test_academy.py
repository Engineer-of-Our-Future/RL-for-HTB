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


# ---- theory-aware patterns (modeled directly on the user's screenshots) ----


def test_acronym_question_uses_paren_expansion():
    """Body says 'PAM (Pluggable Authentication Modules)'; question is the canonical
    'What does the acronym X stand for?' format from the screenshot."""
    section = AcademySection(
        id="s", title="VPS Hardening",
        body_text=(
            "We can use Linux PAM (Pluggable Authentication Modules) to enforce "
            "password policy on our VPS. PAM is configured in /etc/pam.d/."
        ),
    )
    q = AcademyQuestion(
        id="q", prompt="What does the acronym Linux PAM stand for?",
        type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.method == "acronym_expansion"
    assert "Pluggable Authentication Modules" in ans.answer_text
    assert ans.confidence >= 0.8


def test_acronym_question_with_stands_for_phrasing():
    section = AcademySection(
        id="s", title="x",
        body_text="VPN stands for Virtual Private Network. It is used to ...",
    )
    q = AcademyQuestion(
        id="q", prompt="What does VPN stand for?", type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert "Virtual Private Network" in ans.answer_text
    assert ans.method == "acronym_expansion"


def test_acronym_question_with_reverse_paren_expansion():
    section = AcademySection(
        id="s", title="x",
        body_text="The Pluggable Authentication Modules (PAM) framework ...",
    )
    q = AcademyQuestion(
        id="q", prompt="What does the acronym PAM stand for?",
        type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.method == "acronym_expansion"
    assert "Pluggable Authentication Modules" in ans.answer_text


def test_inline_code_in_match_preferred_over_tail():
    """Screenshot's `up-to-date` case: question matches a sentence that has an
    inline code span; we should return the code, not the tail of the sentence."""
    section = AcademySection(
        id="s", title="x",
        body_text=(
            "One of the first steps in hardening our system is updating "
            "and bringing the system up-to-date."
        ),
        inline_code=["up-to-date"],
    )
    q = AcademyQuestion(
        id="q",
        prompt="What is the term for bringing the system to its latest state?",
        type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.answer_text == "up-to-date"
    assert ans.method == "inline_code_in_match"


def test_lone_inline_code_fallback():
    """When nothing else matches but the section has a single inline-code span."""
    section = AcademySection(
        id="s", title="x",
        body_text="Some text with no overlap at all.",
        inline_code=["sshd_config"],
    )
    q = AcademyQuestion(
        id="q", prompt="Quantum chromodynamics?", type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.answer_text == "sshd_config"
    assert ans.method == "lone_inline_code"


def test_howmany_question_counts_bullet_list():
    """Direct port of the screenshot's bullet list of 10 hardening precautions."""
    bullets = [
        "Install Fail2ban",
        "Working only with SSH keys",
        "Reduce Idle timeout interval",
        "Disable passwords",
        "Disable x11 forwarding",
        "Use a different port",
        "Limit users' SSH access",
        "Disable root logins",
        "Use SSH proto 2",
        "Enable 2FA Authentication for SSH",
    ]
    section = AcademySection(
        id="s", title="VPS Hardening",
        body_text="There are many ways to harden our VPS, including the following:",
        bullet_lists=[bullets],
    )
    q = AcademyQuestion(
        id="q",
        prompt="How many SSH hardening precautions does the section list?",
        type=QuestionType.TEXT,
    )
    ans = HeuristicAnswerer().answer(q, section)
    assert ans.answer_text == str(len(bullets))
    assert ans.method == "howmany_count"


def test_question_carries_reward_metadata():
    q = AcademyQuestion(
        id="q1", prompt="?", type=QuestionType.TEXT,
        cubes_reward=5, hp_reward=20,
    )
    assert q.cubes_reward == 5
    assert q.hp_reward == 20


def test_demo_turn_includes_theory_in_obs_text(tmp_path: Path):
    """auto_demo_writer must put the section's theory in the obs_text so BC
    learns the read-then-answer pattern."""
    from htbrl.academy.auto_demo_writer import answer_to_demo_turn

    q = AcademyQuestion(
        id="q1",
        prompt="What does the acronym Linux PAM stand for?",
        type=QuestionType.TEXT,
        cubes_reward=5, hp_reward=20,
    )
    ans = AcademyAnswer(
        question_id="q1",
        answer_text="Pluggable Authentication Modules",
        confidence=0.85,
        method="acronym_expansion",
    )
    turn = answer_to_demo_turn(
        q, ans, accepted=True,
        section_title="VPS Hardening",
        section_body=(
            "We can use Linux PAM (Pluggable Authentication Modules) to enforce "
            "password policy on our VPS."
        ),
        inline_code=["sshd_config"],
    )
    assert "VPS Hardening" in turn.obs_text
    assert "## Theory" in turn.obs_text
    assert "Pluggable Authentication Modules" in turn.obs_text
    assert "## Question" in turn.obs_text
    # Reward includes the +cubes/+HP bonus for an accepted answer
    assert turn.reward > 0.5


def test_demo_turn_truncates_long_theory():
    """Cap theory at 4 KB; keep the last 4 KB which is closer to the question."""
    from htbrl.academy.auto_demo_writer import answer_to_demo_turn

    long_body = "PADDING. " * 800 + "ANSWER_MARKER_AT_END"
    q = AcademyQuestion(id="q1", prompt="?", type=QuestionType.TEXT)
    ans = AcademyAnswer(
        question_id="q1", answer_text="x", confidence=0.5, method="heuristic_text"
    )
    turn = answer_to_demo_turn(
        q, ans, accepted=False,
        section_title="t", section_body=long_body, inline_code=[],
    )
    # The last 4 KB of theory should be present (so the marker we put at the
    # end survives the truncation).
    assert "ANSWER_MARKER_AT_END" in turn.obs_text
    # And the header before truncation should be dropped (we cut from the front).
    assert turn.obs_text.count("PADDING.") < 800


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


def test_orchestrator_unlock_gate_blocks_second_module_in_study_only(tmp_path: Path):
    """In study_only mode the cube balance never changes; with the gate's
    cube-refresh check on, the orchestrator should attempt module #1, then
    refuse to open module #2 even though more eligible modules exist."""
    m1 = AcademyModule(
        id="m1", title="general 1", tier=0, category="general",
        sections=[
            AcademySection(
                id="s1", title="t", body_text="ls lists files",
                questions=[AcademyQuestion(
                    id="m1.q1", prompt="Which command lists files?",
                    type=QuestionType.MULTIPLE_CHOICE,
                    multiple_choice_options=["ls", "pwd", "cat"],
                )],
            ),
        ],
    )
    m2 = AcademyModule(
        id="m2", title="general 2", tier=0, category="general",
        sections=[
            AcademySection(
                id="s2", title="t", body_text="cd changes directory",
                questions=[AcademyQuestion(
                    id="m2.q1", prompt="Which command changes directory?",
                    type=QuestionType.MULTIPLE_CHOICE,
                    multiple_choice_options=["cd", "ls", "pwd"],
                )],
            ),
        ],
    )
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(modules=[m1, m2], state=state, study_only=True)
    sess.login(AcademyCredentials("u", "p"))
    learner = AutoLearner(
        session=sess,
        cfg=OrchestratorConfig(
            study_only=True, auto_demo_dir=tmp_path,
            max_modules_per_run=5,  # enough headroom; gate should bound it.
            require_all_answered_before_unlock=True,
            require_cube_refresh_before_unlock=True,
        ),
    )
    result = learner.run()
    # Module 1 attempted, module 2 NOT attempted (gate closed because no cube delta).
    assert "m1" in result.modules_attempted
    assert "m2" not in result.modules_attempted
    # Two gate decisions recorded: pre-m1 (open) and pre-m2 (closed).
    assert len(result.unlock_gates) >= 2
    assert result.unlock_gates[0].allowed is True
    assert result.unlock_gates[1].allowed is False
    assert any("gate" in e.lower() for e in result.errors)


def test_orchestrator_unlock_gate_relaxed_walks_multiple_modules(tmp_path: Path):
    """With both gate-relax flags off, the orchestrator walks consecutive
    modules even when no submissions land - useful for offline demo harvesting."""
    m1 = AcademyModule(
        id="m1", title="general 1", tier=0, category="general",
        sections=[AcademySection(
            id="s1", title="t", body_text="ls",
            questions=[AcademyQuestion(
                id="m1.q1", prompt="?", type=QuestionType.MULTIPLE_CHOICE,
                multiple_choice_options=["a", "b"],
            )],
        )],
    )
    m2 = AcademyModule(
        id="m2", title="general 2", tier=0, category="general",
        sections=[AcademySection(
            id="s2", title="t", body_text="cd",
            questions=[AcademyQuestion(
                id="m2.q1", prompt="?", type=QuestionType.MULTIPLE_CHOICE,
                multiple_choice_options=["a", "b"],
            )],
        )],
    )
    state = ProgressState(user_id="u", cubes_balance=0)
    sess = MockAcademySession(modules=[m1, m2], state=state, study_only=True)
    sess.login(AcademyCredentials("u", "p"))
    learner = AutoLearner(
        session=sess,
        cfg=OrchestratorConfig(
            study_only=True, auto_demo_dir=tmp_path,
            max_modules_per_run=5,
            require_all_answered_before_unlock=False,
            require_cube_refresh_before_unlock=False,
        ),
    )
    result = learner.run()
    assert "m1" in result.modules_attempted
    assert "m2" in result.modules_attempted


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


# ---- propose() ranked-candidate API ----------------------------------------


def test_propose_returns_ranked_list_with_unique_answers():
    """propose() must dedupe by answer_text and rank by confidence desc."""
    section = AcademySection(
        id="s", title="ls",
        body_text="The ls command lists files in a directory. It is a built-in.",
        inline_code=["ls"],
    )
    q = AcademyQuestion(
        id="q", prompt="Which command lists files?", type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section, top_n=5)
    # All confidence values are descending
    confs = [c.confidence for c in cands]
    assert confs == sorted(confs, reverse=True)
    # Answers are unique (case-insensitive)
    texts = [c.answer_text.strip().lower() for c in cands]
    assert len(set(texts)) == len(texts)
    # The top candidate uses the inline-code span
    assert cands[0].answer_text == "ls"


def test_propose_acronym_beats_inline_code():
    """When an acronym expansion exists, it must rank above inline-code spans."""
    section = AcademySection(
        id="s", title="PAM",
        body_text="Linux PAM (Pluggable Authentication Modules) lets us add modules.",
        inline_code=["pam.so"],
    )
    q = AcademyQuestion(
        id="q", prompt="What does the acronym Linux PAM stand for?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section, top_n=3)
    assert cands
    assert cands[0].answer_text == "Pluggable Authentication Modules"
    assert cands[0].method == "acronym_expansion"


def test_propose_path_question_extracts_path():
    section = AcademySection(
        id="s", title="passwd",
        body_text="The user database is stored in /etc/passwd which holds account info.",
    )
    q = AcademyQuestion(
        id="q", prompt="What is the path to the user database file?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section)
    assert any(c.answer_text == "/etc/passwd" for c in cands)
    # The path should be among the higher-ranked candidates (top 3 at worst)
    top3 = cands[:3]
    assert any(c.answer_text == "/etc/passwd" for c in top3)


def test_propose_port_question_extracts_port():
    section = AcademySection(
        id="s", title="ssh",
        body_text="OpenSSH listens on TCP port 22 by default. Connections are encrypted.",
    )
    q = AcademyQuestion(
        id="q", prompt="On which TCP port does OpenSSH listen?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section)
    assert any(c.answer_text == "22" for c in cands)


def test_propose_mc_returns_all_options_ranked():
    section = AcademySection(
        id="s", title="basics",
        body_text="The ls command lists files in a directory.",
    )
    q = AcademyQuestion(
        id="q", prompt="Which command lists files?",
        type=QuestionType.MULTIPLE_CHOICE,
        multiple_choice_options=["ls", "pwd", "cat"],
    )
    cands = HeuristicAnswerer().propose(q, section)
    assert len(cands) == 3
    # 'ls' should be the top because it appears in the body
    assert cands[0].answer_text == "ls"
    # All candidates use the mc_match method tag
    assert all(c.method == "mc_match" for c in cands)


def test_propose_skipped_when_nothing_to_lean_on():
    section = AcademySection(id="s", title="x", body_text="")
    q = AcademyQuestion(
        id="q", prompt="What is the meaning of life?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section, top_n=3)
    # Either an empty list, or a single skipped marker
    assert all(c.answer_text == "" or c.confidence < 0.5 for c in cands)


def test_propose_inline_code_ranked_uses_command_hint():
    """When the prompt includes a 'command'/'flag'/'option' hint, inline-code
    spans whose surrounding sentence overlaps the prompt should outrank ones
    that don't."""
    section = AcademySection(
        id="s", title="grep",
        body_text=(
            "The grep command searches for patterns. The -i option makes the "
            "search case-insensitive. The cat command concatenates files."
        ),
        inline_code=["grep", "-i", "cat"],
    )
    q = AcademyQuestion(
        id="q", prompt="Which option makes the search case-insensitive?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section, top_n=3)
    # '-i' appears in the case-insensitive sentence so should be at the top
    assert cands[0].answer_text == "-i"


# ---- wizard safety: lab-flag detection -------------------------------------


def test_lab_flag_detection_by_question_type():
    from htbrl.academy.cdp_walker import is_lab_flag_question
    q = AcademyQuestion(id="q", prompt="give me the flag",
                        type=QuestionType.FLAG)
    assert is_lab_flag_question(q) is True


def test_lab_flag_detection_by_prompt_keywords():
    """Even TEXT-typed questions whose prompt mentions a lab flag must be
    treated as auto-submit-unsafe."""
    from htbrl.academy.cdp_walker import is_lab_flag_question
    flag_phrases = [
        "Submit the flag from /root/flag.txt",
        "What is the flag value shown on the box?",
        "Find the flag in HTB{...} format",
        "What's the user flag?",
        "What's the root flag?",
    ]
    for prompt in flag_phrases:
        q = AcademyQuestion(id="q", prompt=prompt, type=QuestionType.TEXT)
        assert is_lab_flag_question(q) is True, f"failed to flag-detect: {prompt!r}"


def test_lab_flag_detection_negative_for_theory_question():
    """Plain theory questions must NOT trigger the lab-flag guard, otherwise
    the wizard could never auto-submit them."""
    from htbrl.academy.cdp_walker import is_lab_flag_question
    theory_prompts = [
        "What does the acronym PAM stand for?",
        "Which command lists files in a directory?",
        "How many ports does the example show as open?",
        "What is the path to the user database?",
        "On which TCP port does OpenSSH listen by default?",
    ]
    for prompt in theory_prompts:
        q = AcademyQuestion(id="q", prompt=prompt, type=QuestionType.TEXT)
        assert is_lab_flag_question(q) is False, f"false-positive on theory: {prompt!r}"


# ---- junk-candidate filter (regression for "h" / "." / "/" answers) ------


def test_propose_filters_single_char_inline_code():
    """The earlier answerer happily emitted ``'h'`` and ``'.'`` at conf 0.80
    when an academy section had those characters as standalone <code> spans.
    Those are noise (regex highlights / ASCII art) and would corrupt BC.
    The filter must reject them so the *meaningful* candidate wins instead."""
    section = AcademySection(
        id="s", title="grep",
        body_text=(
            "The grep command searches for patterns. The -i option makes "
            "the search case-insensitive."
        ),
        # Mix junk with a real candidate.
        inline_code=["h", ".", "/", "-i", "grep"],
    )
    q = AcademyQuestion(
        id="q", prompt="Which option makes the search case-insensitive?",
        type=QuestionType.TEXT,
    )
    cands = HeuristicAnswerer().propose(q, section, top_n=10)
    texts = [c.answer_text for c in cands]
    # Junk is gone:
    assert "h" not in texts
    assert "." not in texts
    assert "/" not in texts
    # Real answer survives and ranks at the top.
    assert "-i" in texts
    assert cands[0].answer_text == "-i"


def test_propose_filters_pure_punctuation_inline_code():
    section = AcademySection(
        id="s", title="x",
        body_text="The example uses -- and := as separators.",
        inline_code=["--", ":=", "?", ";"],
    )
    q = AcademyQuestion(id="q", prompt="What separator is used?", type=QuestionType.TEXT)
    cands = HeuristicAnswerer().propose(q, section)
    # Pure-punctuation candidates are dropped (no \w characters).
    assert all(c.answer_text not in {"?", ";"} for c in cands)


def test_propose_filters_stopword_quoted_spans():
    """A quoted span like \"the\" or \"is\" is a stopword we don't want to
    surface as an answer."""
    section = AcademySection(
        id="s", title="x",
        body_text='The token "the" appears often in English text.',
    )
    q = AcademyQuestion(id="q", prompt="What appears often in English?", type=QuestionType.TEXT)
    cands = HeuristicAnswerer().propose(q, section)
    assert all(c.answer_text.lower() != "the" for c in cands)


def test_is_meaningful_answer_directly():
    from htbrl.academy.answerer import _is_meaningful_answer
    # Reject:
    assert _is_meaningful_answer("") is False
    assert _is_meaningful_answer(" ") is False
    assert _is_meaningful_answer("h") is False
    assert _is_meaningful_answer(".") is False
    assert _is_meaningful_answer("///") is False
    assert _is_meaningful_answer("the") is False
    assert _is_meaningful_answer("OF") is False  # case-insensitive stopword
    # Accept:
    assert _is_meaningful_answer("ls") is True
    assert _is_meaningful_answer("-i") is True
    assert _is_meaningful_answer("--verbose") is True
    assert _is_meaningful_answer("/etc/passwd") is True
    assert _is_meaningful_answer("Pluggable Authentication Modules") is True


def test_answer_uses_top_propose_candidate():
    """Legacy single-best ``answer()`` is just ``propose()[0]``."""
    section = AcademySection(
        id="s", title="ssh",
        body_text="OpenSSH listens on TCP port 22 by default.",
    )
    q = AcademyQuestion(
        id="q", prompt="On which TCP port does OpenSSH listen?",
        type=QuestionType.TEXT,
    )
    a = HeuristicAnswerer()
    top = a.propose(q, section)[0]
    legacy = a.answer(q, section)
    assert top.answer_text == legacy.answer_text
    assert top.method == legacy.method
