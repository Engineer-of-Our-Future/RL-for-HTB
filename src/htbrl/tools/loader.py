"""Tool registry loader + ActionVocabulary indexing (Phase 1).

Loads every YAML file in `src/htbrl/tools/registry/` into a single
`ActionVocabulary`, which is the authoritative action space the policy emits
into. The vocabulary assigns each tool a stable integer ID (alphabetical by
name, deterministic across runs) and provides:

- name <-> id mapping
- safe rendering of (tool_id, slot_values) -> bash command string
- typed validation of slot values prior to rendering
- MITRE ATT&CK lookups (by tactic, technique, matrix) and a coverage summary
  for curriculum + eval metrics
"""

from __future__ import annotations

import ipaddress
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import (
    CidrSlot,
    EnumSlot,
    FilePathSlot,
    FreeStringSlot,
    HashSlot,
    HostnameSlot,
    IntSlot,
    IpSlot,
    Matrix,
    PortListSlot,
    PortSlot,
    Slot,
    ToolDefinition,
    ToolRegistryFile,
    WordlistSlot,
    has_forbidden_shell_chars,
)

REGISTRY_DIR = Path(__file__).parent / "registry"


class SlotValidationError(ValueError):
    """A supplied slot value did not satisfy its slot's type constraints."""


class ToolNotFoundError(KeyError):
    """The vocabulary has no tool with the given name or ID."""


class ActionVocabulary:
    """In-memory index over all loaded tool definitions.

    The vocabulary is immutable after construction. Tools are sorted by name so
    IDs are stable across runs as long as the registry contents don't change.
    """

    def __init__(self, tools: list[ToolDefinition]) -> None:
        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate tool names across registry files: {dups}")

        sorted_tools = sorted(tools, key=lambda t: t.name)
        self._tools: list[ToolDefinition] = sorted_tools
        self._id_by_name: dict[str, int] = {t.name: i for i, t in enumerate(sorted_tools)}

    @property
    def tools(self) -> list[ToolDefinition]:
        return list(self._tools)

    @property
    def n_tools(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[self._id_by_name[name]]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def id_of(self, name: str) -> int:
        try:
            return self._id_by_name[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def name_of(self, tool_id: int) -> str:
        if not (0 <= tool_id < len(self._tools)):
            raise ToolNotFoundError(tool_id)
        return self._tools[tool_id].name

    # ---- MITRE ATT&CK lookups ------------------------------------------------

    def tools_for_technique(self, technique_id: str) -> list[ToolDefinition]:
        """Return all tools tagged with the given technique ID (e.g. 'T1046').

        Sub-techniques are matched exactly: 'T1595.002' returns only tools
        explicitly tagged with that sub-technique, not parents.
        """
        return [t for t in self._tools if technique_id in t.attack.techniques]

    def tools_for_tactic(self, tactic_id: str) -> list[ToolDefinition]:
        """Return all tools tagged with the given tactic ID (e.g. 'TA0007')."""
        return [t for t in self._tools if tactic_id in t.attack.tactics]

    def tools_for_matrix(self, matrix: str | Matrix) -> list[ToolDefinition]:
        """Return all tools applicable to the given matrix.

        A tool that's tagged with multiple matrices (e.g. nmap for both
        Enterprise and ICS reconnaissance) shows up in each lookup.
        """
        m = Matrix(matrix) if isinstance(matrix, str) else matrix
        return [t for t in self._tools if m in t.attack.matrices]

    @property
    def all_tactics(self) -> list[str]:
        """Sorted unique list of every tactic ID covered by the registry."""
        s: set[str] = set()
        for t in self._tools:
            s.update(t.attack.tactics)
        return sorted(s)

    @property
    def all_techniques(self) -> list[str]:
        """Sorted unique list of every technique ID covered by the registry."""
        s: set[str] = set()
        for t in self._tools:
            s.update(t.attack.techniques)
        return sorted(s)

    def coverage_summary(self) -> dict[str, dict[str, int]]:
        """Per-matrix tool-count breakdown by tactic and technique.

        Returns:
            {
                "<matrix_name>": {
                    "tactics":    {<tactic_id>: <tool_count>, ...},
                    "techniques": {<technique_id>: <tool_count>, ...},
                    "tools":      <total tool count for this matrix>,
                },
                ...
            }

        Used by `scripts/coverage_report.py` and CI to gate that no tactic is
        understaffed in the registry.
        """
        out: dict[str, dict[str, Any]] = {}
        for matrix in Matrix:
            tactics = Counter()
            techniques = Counter()
            n_tools = 0
            for t in self._tools:
                if matrix not in t.attack.matrices:
                    continue
                n_tools += 1
                tactics.update(t.attack.tactics)
                techniques.update(t.attack.techniques)
            out[matrix.value] = {
                "tactics": dict(tactics),
                "techniques": dict(techniques),
                "tools": n_tools,
            }
        return out

    # ---- rendering -----------------------------------------------------------

    def render(self, tool: str | int, slot_values: Mapping[str, Any]) -> str:
        """Render a tool invocation to a bash command, validating each slot.

        Raises SlotValidationError on type/constraint failure, ToolNotFoundError
        if the tool is unknown.
        """
        td = self.get(tool) if isinstance(tool, str) else self._tools[tool]

        rendered: dict[str, str] = {}
        for slot in td.slots:
            if slot.name in slot_values:
                value = slot_values[slot.name]
            else:
                default = getattr(slot, "default", None)
                if default is None:
                    if slot.required:
                        raise SlotValidationError(
                            f"tool {td.name!r}: missing required slot {slot.name!r}"
                        )
                    continue
                value = default
            rendered[slot.name] = _validate_and_format(td.name, slot, value)

        try:
            return td.command_template.format(**rendered)
        except KeyError as exc:
            # Should be impossible because schema validation already enforced
            # template <-> slot symmetry, but be defensive.
            raise SlotValidationError(
                f"tool {td.name!r}: template references undefined slot {exc.args[0]!r}"
            ) from exc


# ---- per-slot-type validators -----------------------------------------------

def _validate_and_format(tool_name: str, slot: Slot, value: Any) -> str:
    """Validate `value` against `slot` and return its string rendering."""
    fail = lambda msg: SlotValidationError(f"tool {tool_name!r}, slot {slot.name!r}: {msg}")

    if isinstance(slot, EnumSlot):
        if value not in slot.values:
            raise fail(f"value {value!r} not in enum {slot.values}")
        return str(value)

    if isinstance(slot, IntSlot):
        if not isinstance(value, int) or isinstance(value, bool):
            raise fail(f"expected int, got {type(value).__name__}")
        if not (slot.min <= value <= slot.max):
            raise fail(f"value {value} out of range [{slot.min}, {slot.max}]")
        return str(value)

    if isinstance(slot, IpSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (ip), got {type(value).__name__}")
        try:
            ipaddress.ip_address(value)
        except ValueError as e:
            raise fail(f"invalid IP {value!r}: {e}") from None
        return value

    if isinstance(slot, CidrSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (cidr), got {type(value).__name__}")
        try:
            ipaddress.ip_network(value, strict=False)
        except ValueError as e:
            raise fail(f"invalid CIDR {value!r}: {e}") from None
        return value

    if isinstance(slot, PortSlot):
        if not isinstance(value, int) or isinstance(value, bool):
            raise fail(f"expected int (port), got {type(value).__name__}")
        if not (1 <= value <= 65535):
            raise fail(f"port {value} out of 1..65535")
        return str(value)

    if isinstance(slot, PortListSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (port_list), got {type(value).__name__}")
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise fail("empty port list")
        if len(parts) > slot.max_count:
            raise fail(f"port list has {len(parts)} entries, max {slot.max_count}")
        for p in parts:
            try:
                pi = int(p)
            except ValueError:
                raise fail(f"non-integer entry {p!r} in port list") from None
            if not (1 <= pi <= 65535):
                raise fail(f"port {pi} out of 1..65535 in port list")
        return ",".join(parts)

    if isinstance(slot, WordlistSlot):
        if value not in slot.values:
            raise fail(f"wordlist id {value!r} not in {slot.values}")
        return str(value)

    if isinstance(slot, FreeStringSlot):
        if not isinstance(value, str):
            raise fail(f"expected str, got {type(value).__name__}")
        if len(value) > slot.max_length:
            raise fail(f"length {len(value)} > max {slot.max_length}")
        if has_forbidden_shell_chars(value):
            raise fail(f"value contains forbidden shell metacharacters: {value!r}")
        return value

    if isinstance(slot, FilePathSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (path), got {type(value).__name__}")
        if has_forbidden_shell_chars(value):
            raise fail(f"path contains forbidden shell metacharacters: {value!r}")
        if value.startswith("-"):
            # Prevents "ls -la" being interpreted as a flag injected via path.
            raise fail(f"path may not start with '-': {value!r}")
        return value

    if isinstance(slot, HostnameSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (hostname), got {type(value).__name__}")
        if has_forbidden_shell_chars(value):
            raise fail(f"hostname contains forbidden shell metacharacters: {value!r}")
        if not value or len(value) > 253:
            raise fail(f"hostname length {len(value)} not in 1..253")
        return value

    if isinstance(slot, HashSlot):
        if not isinstance(value, str):
            raise fail(f"expected str (hash), got {type(value).__name__}")
        if not all(c in "0123456789abcdefABCDEF" for c in value):
            raise fail(f"hash {value!r} is not hex")
        return value.lower()

    raise fail(f"unhandled slot type {type(slot).__name__}")


# ---- entry points ------------------------------------------------------------

def load_registry(registry_dir: Path | None = None) -> ActionVocabulary:
    """Load and merge every YAML in `registry_dir` (default: bundled registry/)."""
    d = Path(registry_dir) if registry_dir is not None else REGISTRY_DIR
    if not d.is_dir():
        raise FileNotFoundError(f"registry directory does not exist: {d}")

    all_tools: list[ToolDefinition] = []
    seen_files: list[str] = []
    for yaml_path in sorted(d.glob("*.yaml")):
        seen_files.append(yaml_path.name)
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        registry_file = ToolRegistryFile.model_validate(data)
        all_tools.extend(registry_file.tools)

    if not all_tools:
        raise RuntimeError(
            f"no tools loaded from {d} (saw files: {seen_files}). Phase 1 should "
            f"have populated this directory."
        )
    return ActionVocabulary(all_tools)
