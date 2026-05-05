"""Phase 1 tests: schema validation, registry loading, command rendering, safety.

These tests are pure-Python (no torch, no SSH); they should pass on any machine
that has the project installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from htbrl.tools.loader import (
    REGISTRY_DIR,
    ActionVocabulary,
    SlotValidationError,
    ToolNotFoundError,
    load_registry,
)
from htbrl.tools.schema import (
    Category,
    EnumSlot,
    FilePathSlot,
    FreeStringSlot,
    HashSlot,
    HostnameSlot,
    IntSlot,
    IpSlot,
    PortSlot,
    ToolDefinition,
    ToolRegistryFile,
    _extract_template_slots,
    has_forbidden_shell_chars,
)


# ---- registry loading --------------------------------------------------------

def test_registry_dir_exists():
    assert REGISTRY_DIR.is_dir(), f"missing registry dir at {REGISTRY_DIR}"
    yaml_files = sorted(REGISTRY_DIR.glob("*.yaml"))
    assert yaml_files, "no YAML files in registry dir - did Phase 1 populate it?"


def test_load_registry_succeeds():
    vocab = load_registry()
    assert isinstance(vocab, ActionVocabulary)
    assert vocab.n_tools >= 15, f"expected >=15 starter tools, got {vocab.n_tools}"


def test_tool_ids_are_stable_and_dense():
    vocab = load_registry()
    ids = [vocab.id_of(t.name) for t in vocab.tools]
    assert ids == sorted(ids) == list(range(vocab.n_tools))


def test_tool_names_are_unique_globally():
    vocab = load_registry()
    names = [t.name for t in vocab.tools]
    assert len(names) == len(set(names))


def test_every_tool_renders_its_example_invocation():
    """If a tool defines example_invocation, the loader must render it without errors."""
    vocab = load_registry()
    rendered_count = 0
    for tool in vocab.tools:
        if tool.example_invocation is None:
            continue
        cmd = vocab.render(tool.name, tool.example_invocation)
        assert isinstance(cmd, str) and cmd
        rendered_count += 1
    # We require all starter tools to have examples - this catches "I added a
    # tool but forgot the example" regressions.
    assert rendered_count == vocab.n_tools, (
        f"only {rendered_count}/{vocab.n_tools} tools have example_invocation; all "
        f"starter tools must define one"
    )


def test_all_categories_used_by_starter_set():
    """We expect at least recon, enum, web in the starter set."""
    vocab = load_registry()
    cats = {t.category for t in vocab.tools}
    assert Category.RECON in cats
    assert Category.ENUM in cats
    assert Category.WEB in cats


# ---- schema-level: template <-> slot symmetry --------------------------------

def test_template_slots_must_match_defined_slots():
    bad = {
        "tools": [
            {
                "name": "broken",
                "category": "recon",
                "command_template": "echo {missing}",
                "slots": [],
            }
        ]
    }
    with pytest.raises(ValueError, match="references slots that are not defined"):
        ToolRegistryFile.model_validate(bad)


def test_unused_slots_are_rejected():
    bad = {
        "tools": [
            {
                "name": "broken",
                "category": "recon",
                "command_template": "echo hi",
                "slots": [{"name": "extra", "type": "ip"}],
            }
        ]
    }
    with pytest.raises(ValueError, match="not used in command_template"):
        ToolRegistryFile.model_validate(bad)


def test_template_format_specs_are_rejected():
    """Slot placeholders must be plain names; {x:>3} or {x!r} are forbidden."""
    bad = {
        "tools": [
            {
                "name": "fmt",
                "category": "recon",
                "command_template": "echo {x:>3}",
                "slots": [{"name": "x", "type": "free_string"}],
            }
        ]
    }
    with pytest.raises(ValueError, match="format spec"):
        ToolRegistryFile.model_validate(bad)


def test_extract_template_slots_handles_escaped_braces():
    assert _extract_template_slots("echo {a} and {{b}}") == {"a"}
    assert _extract_template_slots("plain") == set()
    assert _extract_template_slots("{a} {b} {a}") == {"a", "b"}


def test_tool_name_must_be_identifier():
    bad = {
        "tools": [
            {
                "name": "9bad-name",
                "category": "recon",
                "command_template": "echo hi",
                "slots": [],
            }
        ]
    }
    with pytest.raises(ValueError):
        ToolRegistryFile.model_validate(bad)


# ---- slot type validation ----------------------------------------------------

def _vocab_with(td_dict: dict) -> ActionVocabulary:
    return ActionVocabulary([ToolDefinition.model_validate(td_dict)])


def test_int_slot_range_enforced():
    vocab = _vocab_with({
        "name": "test_int",
        "category": "recon",
        "command_template": "echo {n}",
        "slots": [{"name": "n", "type": "int", "min": 1, "max": 10}],
    })
    assert vocab.render("test_int", {"n": 5}) == "echo 5"
    with pytest.raises(SlotValidationError, match="out of range"):
        vocab.render("test_int", {"n": 11})
    with pytest.raises(SlotValidationError, match="expected int"):
        vocab.render("test_int", {"n": "5"})
    # Bools are sneaky in Python (True == 1), reject explicitly.
    with pytest.raises(SlotValidationError):
        vocab.render("test_int", {"n": True})


def test_ip_slot_validates_address():
    vocab = _vocab_with({
        "name": "test_ip",
        "category": "recon",
        "command_template": "ping {ip}",
        "slots": [{"name": "ip", "type": "ip"}],
    })
    assert vocab.render("test_ip", {"ip": "10.10.10.5"}) == "ping 10.10.10.5"
    assert vocab.render("test_ip", {"ip": "::1"}) == "ping ::1"
    with pytest.raises(SlotValidationError, match="invalid IP"):
        vocab.render("test_ip", {"ip": "not-an-ip"})


def test_port_list_slot_rejects_oversize():
    vocab = _vocab_with({
        "name": "test_pl",
        "category": "recon",
        "command_template": "echo {ports}",
        "slots": [{"name": "ports", "type": "port_list", "max_count": 3}],
    })
    assert vocab.render("test_pl", {"ports": "80, 443, 22"}) == "echo 80,443,22"
    with pytest.raises(SlotValidationError, match="max"):
        vocab.render("test_pl", {"ports": "1,2,3,4"})
    with pytest.raises(SlotValidationError, match="non-integer"):
        vocab.render("test_pl", {"ports": "80,abc"})
    with pytest.raises(SlotValidationError, match="out of"):
        vocab.render("test_pl", {"ports": "80,99999"})


def test_enum_slot_default_must_be_in_values():
    """Schema-level: declaring a default not in values fails at parse time."""
    with pytest.raises(ValueError, match="not in values"):
        EnumSlot(name="x", values=["a", "b"], default="c")


def test_enum_slot_render_rejects_unknown_value():
    vocab = _vocab_with({
        "name": "test_enum",
        "category": "recon",
        "command_template": "echo {mode}",
        "slots": [{"name": "mode", "type": "enum", "values": ["fast", "slow"], "default": "fast"}],
    })
    assert vocab.render("test_enum", {}) == "echo fast"  # uses default
    assert vocab.render("test_enum", {"mode": "slow"}) == "echo slow"
    with pytest.raises(SlotValidationError, match="not in enum"):
        vocab.render("test_enum", {"mode": "medium"})


def test_hash_slot_validates_hex():
    vocab = _vocab_with({
        "name": "test_hash",
        "category": "exploit",
        "command_template": "echo {h}",
        "slots": [{"name": "h", "type": "hash"}],
    })
    assert vocab.render("test_hash", {"h": "DEADBEEF"}) == "echo deadbeef"
    with pytest.raises(SlotValidationError, match="not hex"):
        vocab.render("test_hash", {"h": "not-hex"})


# ---- safety: shell metacharacter blocking -----------------------------------

@pytest.mark.parametrize("payload", [
    "foo;rm -rf /",
    "foo && bad",
    "foo|cat /etc/passwd",
    "foo`whoami`",
    "foo$(id)",
    "foo>out",
    "foo<in",
    "foo\nbar",
    'foo"bar',
    "foo'bar",
    "foo\\bar",
])
def test_free_string_slot_rejects_shell_metas(payload):
    vocab = _vocab_with({
        "name": "test_free",
        "category": "recon",
        "command_template": "echo {s}",
        "slots": [{"name": "s", "type": "free_string"}],
    })
    with pytest.raises(SlotValidationError, match="forbidden shell metacharacters"):
        vocab.render("test_free", {"s": payload})
    assert has_forbidden_shell_chars(payload)


def test_free_string_slot_accepts_safe_value():
    vocab = _vocab_with({
        "name": "test_free",
        "category": "recon",
        "command_template": "echo {s}",
        "slots": [{"name": "s", "type": "free_string"}],
    })
    assert vocab.render("test_free", {"s": "hello-world.txt"}) == "echo hello-world.txt"


def test_file_path_slot_rejects_dash_prefix():
    """Prevents 'ls -la' being smuggled in via a path slot."""
    vocab = _vocab_with({
        "name": "test_path",
        "category": "recon",
        "command_template": "ls {p}",
        "slots": [{"name": "p", "type": "file_path"}],
    })
    assert vocab.render("test_path", {"p": "/tmp/foo"}) == "ls /tmp/foo"
    with pytest.raises(SlotValidationError, match="may not start with '-'"):
        vocab.render("test_path", {"p": "-la"})


# ---- defaults & required-ness ------------------------------------------------

def test_missing_required_slot_fails():
    vocab = _vocab_with({
        "name": "test_req",
        "category": "recon",
        "command_template": "ping {ip}",
        "slots": [{"name": "ip", "type": "ip"}],
    })
    with pytest.raises(SlotValidationError, match="missing required slot"):
        vocab.render("test_req", {})


def test_optional_slot_without_default_is_skipped():
    # If a slot is optional and has no default and is referenced by template,
    # rendering must fail at .format() time. Schema enforces template <-> slot
    # symmetry, so this catches the loop where required=False + no default +
    # not provided is impossible to render. We test that error path.
    vocab = _vocab_with({
        "name": "test_opt",
        "category": "recon",
        "command_template": "ping {ip}",
        "slots": [{"name": "ip", "type": "ip", "required": False}],
    })
    with pytest.raises(SlotValidationError):
        vocab.render("test_opt", {})


# ---- vocabulary indexing ----------------------------------------------------

def test_unknown_tool_raises():
    vocab = load_registry()
    with pytest.raises(ToolNotFoundError):
        vocab.get("nonexistent_tool")
    with pytest.raises(ToolNotFoundError):
        vocab.id_of("nonexistent_tool")
    with pytest.raises(ToolNotFoundError):
        vocab.name_of(99999)


def test_id_round_trip():
    vocab = load_registry()
    for tool in vocab.tools:
        assert vocab.name_of(vocab.id_of(tool.name)) == tool.name


# ---- per-file YAML hygiene ---------------------------------------------------

@pytest.mark.parametrize("yaml_path", sorted(REGISTRY_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_yaml_file_parses_cleanly(yaml_path: Path):
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "tools" in data, f"{yaml_path.name}: missing top-level 'tools'"
    ToolRegistryFile.model_validate(data)
