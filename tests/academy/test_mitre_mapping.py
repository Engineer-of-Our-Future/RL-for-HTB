"""Tests for the academy -> MITRE ATT&CK mapping wiring."""

from __future__ import annotations

from htbrl.academy.auto_demo_writer import session_to_demonstration
from htbrl.academy.mitre_mapping import (
    ACADEMY_MODULE_TECHNIQUES,
    techniques_for_module,
    techniques_for_section,
)
from htbrl.academy.page_models import (
    AcademyAnswer,
    AcademyModule,
    AcademyQuestion,
    AcademySection,
    QuestionType,
)


# ---- module-level mapping ---------------------------------------------------


def test_linux_fundamentals_module_includes_unix_shell_and_files():
    mod = AcademyModule(
        id="linux-fundamentals",
        title="Linux Fundamentals",
        tier=0,
    )
    techs = techniques_for_module(mod)
    assert "T1059.004" in techs   # Unix Shell
    assert "T1083" in techs       # File and Directory Discovery


def test_windows_fundamentals_module_includes_powershell():
    mod = AcademyModule(
        id="windows-fundamentals",
        title="Windows Fundamentals",
        tier=0,
    )
    techs = techniques_for_module(mod)
    assert "T1059.001" in techs    # PowerShell
    assert "T1059.003" in techs    # Windows Command Shell


def test_network_enumeration_module_includes_t1046():
    mod = AcademyModule(
        id="network-enum-with-nmap",
        title="Network Enumeration with Nmap",
        tier=0,
    )
    techs = techniques_for_module(mod)
    assert "T1046" in techs        # Network Service Discovery
    assert "T1018" in techs        # Remote System Discovery


def test_intro_to_academy_returns_empty_list():
    mod = AcademyModule(id="intro", title="Intro to Academy", tier=0)
    assert techniques_for_module(mod) == []


def test_learning_process_returns_empty_list():
    mod = AcademyModule(id="lp", title="Learning Process", tier=0)
    assert techniques_for_module(mod) == []


def test_unknown_module_returns_empty_list():
    """A module whose title matches no keyword must produce no false-positive
    technique tags."""
    mod = AcademyModule(id="m", title="An entirely unrelated topic", tier=0)
    assert techniques_for_module(mod) == []


def test_active_directory_module_includes_kerberos_techniques():
    mod = AcademyModule(id="ad", title="Active Directory Enumeration", tier=2)
    techs = techniques_for_module(mod)
    assert "T1558" in techs       # Steal or Forge Kerberos Tickets
    assert "T1003.006" in techs   # DCSync
    assert "T1208" in techs       # Kerberoasting (legacy)


def test_password_attacks_module_includes_brute_force_and_dumping():
    mod = AcademyModule(id="pwd", title="Password Attacks", tier=1)
    techs = techniques_for_module(mod)
    assert "T1110" in techs       # Brute Force
    assert "T1003" in techs       # OS Credential Dumping


def test_buffer_overflow_module_includes_exploitation():
    mod = AcademyModule(id="bof", title="Stack-Based Buffer Overflows", tier=2)
    techs = techniques_for_module(mod)
    assert "T1203" in techs       # Exploitation for Client Execution
    assert "T1068" in techs       # Exploitation for Privilege Escalation


def test_file_transfer_module_includes_ingress_tool_transfer():
    mod = AcademyModule(id="ft", title="File Transfers", tier=1)
    assert "T1105" in techniques_for_module(mod)


def test_tunneling_module_includes_protocol_tunneling_and_proxy():
    mod = AcademyModule(id="tun", title="Pivoting, Tunneling, and Port Forwarding", tier=2)
    techs = techniques_for_module(mod)
    assert "T1572" in techs   # Protocol Tunneling
    assert "T1090" in techs   # Proxy


def test_module_techniques_table_has_only_valid_ids():
    """Every entry in the table must match the schema's technique-ID regex."""
    import re
    pat = re.compile(r"^T\d{4}(?:\.\d{3})?$")
    for keyword, techs in ACADEMY_MODULE_TECHNIQUES.items():
        for t in techs:
            assert pat.match(t), f"bad technique id {t!r} under keyword {keyword!r}"


# ---- section-level narrowing ------------------------------------------------


def test_section_about_finding_files_biases_to_t1083():
    """Within Linux Fundamentals, a section titled 'Find Files and Directories'
    should narrow to T1083 (File and Directory Discovery), not the full set."""
    mod = AcademyModule(id="lf", title="Linux Fundamentals", tier=0)
    module_techs = techniques_for_module(mod)
    section = AcademySection(
        id="lf.s1",
        title="Find Files and Directories",
        body_text="We use the find command to locate files in the file system.",
    )
    section_techs = techniques_for_section(section, module_techs)
    assert "T1083" in section_techs
    # And critically, the section narrowing should have *dropped* unrelated
    # techniques like T1057 (Process Discovery) since this section doesn't
    # talk about processes.
    assert "T1057" not in section_techs


def test_section_about_processes_biases_to_t1057():
    mod = AcademyModule(id="lf", title="Linux Fundamentals", tier=0)
    module_techs = techniques_for_module(mod)
    section = AcademySection(
        id="lf.s2",
        title="Process Management",
        body_text="The ps command lists running processes.",
    )
    section_techs = techniques_for_section(section, module_techs)
    assert "T1057" in section_techs


def test_section_with_no_keyword_falls_back_to_module_level():
    """A theory-only section with no specific keyword should inherit the
    module's full coverage, not collapse to []."""
    mod = AcademyModule(id="lf", title="Linux Fundamentals", tier=0)
    module_techs = techniques_for_module(mod)
    section = AcademySection(
        id="lf.s3",
        title="Why Linux",
        body_text="Some general motivation for using Linux as an attacker.",
    )
    section_techs = techniques_for_section(section, module_techs)
    assert section_techs == module_techs


def test_section_with_empty_module_techniques_returns_empty():
    """If the module has no techniques, every section is also empty."""
    section = AcademySection(id="s", title="Find Files", body_text="find -name foo")
    assert techniques_for_section(section, []) == []


def test_section_narrowing_never_invents_techniques_outside_module():
    """Even if section text matches a hint (e.g. 'kerberoast'), if the parent
    module isn't an AD module, the section must NOT acquire that technique."""
    mod = AcademyModule(id="lf", title="Linux Fundamentals", tier=0)
    module_techs = techniques_for_module(mod)
    assert "T1208" not in module_techs    # sanity: not in linux module
    # Even if the section text mentions 'kerberoast', narrowing must respect
    # the module-level set.
    section = AcademySection(
        id="lf.s",
        title="Misnamed section",
        body_text="This page mentions kerberoast for some reason.",
    )
    section_techs = techniques_for_section(section, module_techs)
    assert "T1208" not in section_techs


# ---- end-to-end demo writer behaviour --------------------------------------


def test_session_to_demonstration_emits_nonempty_techniques_for_linux_fundamentals():
    """Linux Fundamentals demo turns must carry non-empty technique lists."""
    mod = AcademyModule(
        id="lf",
        title="Linux Fundamentals",
        tier=0,
        sections=[
            AcademySection(
                id="lf.s1",
                title="Find Files and Directories",
                body_text="We use the find command to locate files.",
                questions=[
                    AcademyQuestion(
                        id="lf.q1",
                        prompt="What command finds files?",
                        type=QuestionType.TEXT,
                    ),
                ],
            ),
        ],
    )
    answer = AcademyAnswer(
        question_id="lf.q1",
        answer_text="find",
        confidence=0.9,
        method="heuristic_text",
    )
    submissions = [("lf", answer, True)]
    demo = session_to_demonstration(mod, submissions, study_only=True)

    # Module-level metadata must include the technique list.
    assert "T1083" in demo.metadata["module_techniques"]

    # At least one turn must carry techniques_attempted with T1083.
    attempted = [t for turn in demo.turns for t in turn.techniques_attempted]
    assert "T1083" in attempted

    # The accepted-answer turn must record T1083 in techniques_succeeded.
    answer_turns = [t for t in demo.turns if t.action_tool_name == "academy_answer"]
    assert answer_turns, "expected an academy_answer turn"
    assert "T1083" in answer_turns[0].techniques_succeeded


def test_session_to_demonstration_for_intro_module_has_empty_techniques():
    """Modules that legitimately teach no offensive technique must NOT acquire
    any technique tags - empty lists are correct here."""
    mod = AcademyModule(
        id="intro",
        title="Intro to Academy",
        tier=0,
        sections=[
            AcademySection(
                id="intro.s1",
                title="Welcome",
                body_text="Welcome to the academy.",
                questions=[
                    AcademyQuestion(
                        id="intro.q1",
                        prompt="What is this?",
                        type=QuestionType.TEXT,
                    ),
                ],
            ),
        ],
    )
    answer = AcademyAnswer(
        question_id="intro.q1",
        answer_text="academy",
        confidence=0.9,
        method="heuristic_text",
    )
    submissions = [("intro", answer, True)]
    demo = session_to_demonstration(mod, submissions, study_only=True)
    assert demo.metadata["module_techniques"] == []
    for turn in demo.turns:
        assert turn.techniques_attempted == []
        assert turn.techniques_succeeded == []


def test_section_read_turn_techniques_for_unrecognized_module_are_empty():
    """A section_read turn for a module with no keyword match must have
    empty technique lists - the academy track shouldn't fabricate ATT&CK
    coverage for unrelated material."""
    mod = AcademyModule(
        id="random",
        title="Some Unrelated Module",
        tier=0,
        sections=[
            AcademySection(
                id="r.s1",
                title="A Section",
                body_text="Some body text.",
            ),
        ],
    )
    demo = session_to_demonstration(mod, [], study_only=True)
    read_turns = [t for t in demo.turns if t.action_tool_name == "academy_section_read"]
    assert read_turns
    for t in read_turns:
        assert t.techniques_attempted == []
        assert t.techniques_succeeded == []


def test_rejected_answer_records_attempt_but_not_success():
    """A rejected answer should still be in techniques_attempted (we tried),
    but NOT in techniques_succeeded."""
    mod = AcademyModule(
        id="lf",
        title="Linux Fundamentals",
        tier=0,
        sections=[
            AcademySection(
                id="lf.s1",
                title="Find Files",
                body_text="The find command.",
                questions=[
                    AcademyQuestion(
                        id="lf.q1",
                        prompt="?",
                        type=QuestionType.TEXT,
                    ),
                ],
            ),
        ],
    )
    answer = AcademyAnswer(
        question_id="lf.q1",
        answer_text="wrong",
        confidence=0.4,
        method="heuristic_text",
    )
    submissions = [("lf", answer, False)]
    demo = session_to_demonstration(mod, submissions, study_only=True)
    answer_turns = [t for t in demo.turns if t.action_tool_name == "academy_answer"]
    assert answer_turns
    # Attempted but not succeeded
    assert "T1083" in answer_turns[0].techniques_attempted
    assert answer_turns[0].techniques_succeeded == []


# ---- regression: broad coverage of academy themes --------------------------


def test_mapping_table_covers_offensive_themes_broadly():
    """Locks in that the major offensive themes have entries in
    ACADEMY_MODULE_TECHNIQUES. A future edit accidentally dropping a
    section of the mapping fires this test.
    """
    expected = {
        "linux fundamentals": "T1059.004",
        "windows fundamentals": "T1059.001",
        "network enumeration": "T1046",
        "footprinting": "T1595.001",
        "active directory": "T1558",
        "kerberos": "T1558",
        "buffer overflow": "T1203",
        "password attacks": "T1110",
        "file inclusion": "T1190",
        "command injection": "T1059",
        "sql injection": "T1190",
        "metasploit": "T1059",
        "ntlm relay": "T1557.001",
        "lateral movement": "T1021",
        "siem fundamentals": "T1078",
        "threat hunting": "T1059",
        "malware analysis": "T1027",
        "javascript deobfusc": "T1027",
        "file transfer": "T1105",
        "tunneling": "T1572",
    }
    for kw, expect_id in expected.items():
        techs = ACADEMY_MODULE_TECHNIQUES.get(kw, [])
        assert expect_id in techs, (
            f"keyword {kw!r} should yield {expect_id!r}, got {techs!r}"
        )


def test_mapping_table_intentional_empties_stay_empty():
    """Process / foundational / AI-attack modules deliberately have empty
    technique lists. This locks that in so a future edit doesn't tag them
    with random techniques.
    """
    intentional_empties = [
        "intro to academy",
        "learning process",
        "setting up",
        "documentation & reporting",
        "incident handling process",
        "bug bounty hunting process",
        "penetration testing process",
        "fundamentals of ai",
        "applications of ai in infosec",
        "red teaming ai",
        "ai evasion",
        "ai defense",
        "ai privacy",
        "llm output attacks",
    ]
    for kw in intentional_empties:
        techs = ACADEMY_MODULE_TECHNIQUES.get(kw, None)
        assert techs is not None, f"keyword {kw!r} missing from table"
        assert techs == [], f"keyword {kw!r} should be empty, got {techs!r}"


def test_distinct_technique_count_is_substantial():
    """The mapping table should produce coverage of at least 50 distinct
    ATT&CK technique IDs across the academy curriculum. Sanity guard
    against accidental table truncation.
    """
    all_ids: set[str] = set()
    for techs in ACADEMY_MODULE_TECHNIQUES.values():
        all_ids.update(techs)
    assert len(all_ids) >= 50, (
        f"only {len(all_ids)} distinct techniques in table; expected >= 50"
    )
