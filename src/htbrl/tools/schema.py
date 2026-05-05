"""Tool registry schema (Phase 1).

This module defines the typed action vocabulary the policy chooses from. Every
tool has a fixed name, a bash command template with named slots, and a typed
slot definition for each slot. The policy emits a (tool_id, slot_values) tuple;
the loader in `loader.py` validates and renders that into a runnable command.

Why typed slots: a free-form text policy on consumer hardware is not feasible
to train from scratch. Constraining each slot to a small typed vocabulary
collapses the action space by orders of magnitude and lets the per-slot heads
in the model use ordinary categorical sampling.
"""

from __future__ import annotations

import ipaddress
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    RECON = "recon"
    ENUM = "enum"
    WEB = "web"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post-exploit"
    LATERAL = "lateral"
    CLEANUP = "cleanup"


# ----- slot type definitions ---------------------------------------------------

class _SlotBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    required: bool = True


class EnumSlot(_SlotBase):
    type: Literal["enum"] = "enum"
    values: list[str] = Field(min_length=1)
    default: str | None = None

    @model_validator(mode="after")
    def _default_in_values(self) -> "EnumSlot":
        if self.default is not None and self.default not in self.values:
            raise ValueError(f"default {self.default!r} not in values {self.values}")
        return self


class IntSlot(_SlotBase):
    type: Literal["int"] = "int"
    min: int = 0
    max: int = 65535
    default: int | None = None

    @model_validator(mode="after")
    def _default_in_range(self) -> "IntSlot":
        if self.default is not None and not (self.min <= self.default <= self.max):
            raise ValueError(f"default {self.default} not in [{self.min}, {self.max}]")
        return self


class IpSlot(_SlotBase):
    """A single IPv4 / IPv6 address. Resolved + allowlist-checked at execution time."""
    type: Literal["ip"] = "ip"
    default: str | None = None

    @field_validator("default")
    @classmethod
    def _check_default(cls, v: str | None) -> str | None:
        if v is not None:
            ipaddress.ip_address(v)
        return v


class CidrSlot(_SlotBase):
    type: Literal["cidr"] = "cidr"
    default: str | None = None

    @field_validator("default")
    @classmethod
    def _check_default(cls, v: str | None) -> str | None:
        if v is not None:
            ipaddress.ip_network(v, strict=False)
        return v


class PortSlot(_SlotBase):
    """Single port 1-65535."""
    type: Literal["port"] = "port"
    default: int | None = None

    @model_validator(mode="after")
    def _check_default(self) -> "PortSlot":
        if self.default is not None and not (1 <= self.default <= 65535):
            raise ValueError(f"port default {self.default} out of 1..65535")
        return self


class PortListSlot(_SlotBase):
    """Comma-separated list of ports, e.g. '22,80,443'. Rendered verbatim."""
    type: Literal["port_list"] = "port_list"
    default: str | None = None
    max_count: int = 64


class WordlistSlot(_SlotBase):
    """An ID into a known-wordlist registry. Resolved to a path at exec time."""
    type: Literal["wordlist_id"] = "wordlist_id"
    values: list[str] = Field(min_length=1)
    default: str | None = None

    @model_validator(mode="after")
    def _default_in_values(self) -> "WordlistSlot":
        if self.default is not None and self.default not in self.values:
            raise ValueError(f"default {self.default!r} not in values {self.values}")
        return self


class FreeStringSlot(_SlotBase):
    """Free-text slot (filenames, custom payloads, hostnames not pre-known).

    Free-string slots are emitted by the BPE tokenizer head in Phase 2+ rather
    than a categorical head. Validation here is just length + no-shell-injection.
    """
    type: Literal["free_string"] = "free_string"
    max_length: int = 256
    default: str | None = None


class FilePathSlot(_SlotBase):
    """Path on the Kali attacker filesystem. No shell metacharacters allowed."""
    type: Literal["file_path"] = "file_path"
    default: str | None = None


class HostnameSlot(_SlotBase):
    """DNS hostname or IP-as-hostname."""
    type: Literal["hostname"] = "hostname"
    default: str | None = None


class HashSlot(_SlotBase):
    """A hex-encoded hash string. Used for cracking-tool slots."""
    type: Literal["hash"] = "hash"
    default: str | None = None


# Discriminated union — pydantic v2 picks the right subclass by the `type` field.
Slot = Annotated[
    Union[
        EnumSlot,
        IntSlot,
        IpSlot,
        CidrSlot,
        PortSlot,
        PortListSlot,
        WordlistSlot,
        FreeStringSlot,
        FilePathSlot,
        HostnameSlot,
        HashSlot,
    ],
    Field(discriminator="type"),
]


# ----- tool definition ---------------------------------------------------------

# Shell metacharacters we forbid in any rendered argument. These are the
# characters that let an attacker (or a confused policy) break out of the
# intended command structure: command separators, pipes, redirections,
# substitution, and quoting tricks. The tool template itself is allowed to use
# them - this rule applies only to slot values supplied by the policy.
_FORBIDDEN_SHELL_CHARS = frozenset(";&|`$()<>\n\r\t\"'\\")


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    category: Category
    description: str = ""
    command_template: str = Field(min_length=1)
    slots: list[Slot] = Field(default_factory=list)
    runtime_cap_seconds: int = Field(default=60, ge=1, le=3600)
    output_parser_id: str = "raw"
    requires_root: bool = False
    requires_target_in_allowlist: bool = True
    example_invocation: dict[str, Any] | None = None
    example_output: str = ""

    @field_validator("name")
    @classmethod
    def _name_kebab(cls, v: str) -> str:
        # tool names: lowercase letters, digits, underscores. Easy to map to identifiers.
        if not all(c.isalnum() or c == "_" for c in v) or not v[0].isalpha():
            raise ValueError(f"tool name {v!r} must match [a-z][a-z0-9_]*")
        return v

    @model_validator(mode="after")
    def _slots_unique_and_referenced(self) -> "ToolDefinition":
        # All slot names unique.
        names = [s.name for s in self.slots]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"tool {self.name!r}: duplicate slot names {dups}")
        # Every {placeholder} in the template references a defined slot, and vice
        # versa. We use a tiny home-grown scan rather than .format() introspection
        # because we want to flag mismatches with helpful messages.
        template_slots = _extract_template_slots(self.command_template)
        defined = set(names)
        missing = template_slots - defined
        unused = defined - template_slots
        if missing:
            raise ValueError(
                f"tool {self.name!r}: command_template references slots that are not "
                f"defined: {sorted(missing)}"
            )
        if unused:
            raise ValueError(
                f"tool {self.name!r}: defined slots are not used in command_template: "
                f"{sorted(unused)}"
            )
        return self


class ToolRegistryFile(BaseModel):
    """Top-level structure of a registry/*.yaml file."""
    model_config = ConfigDict(extra="forbid")
    tools: list[ToolDefinition]


# ----- helpers -----------------------------------------------------------------

def _extract_template_slots(template: str) -> set[str]:
    """Return the set of {slot_name} placeholders used in `template`.

    We don't allow format-spec or conversion suffixes ({x:>3}, {x!r}) because
    every slot value is rendered as-is and we want syntactic simplicity.
    """
    out: set[str] = set()
    i = 0
    n = len(template)
    while i < n:
        c = template[i]
        if c == "{" and i + 1 < n and template[i + 1] == "{":
            i += 2
            continue
        if c == "}" and i + 1 < n and template[i + 1] == "}":
            i += 2
            continue
        if c == "{":
            j = template.find("}", i + 1)
            if j == -1:
                raise ValueError(f"unbalanced '{{' in template: {template!r}")
            inner = template[i + 1 : j]
            if not inner or any(ch in inner for ch in ":!{} "):
                raise ValueError(
                    f"slot placeholder {{{inner}}} in template {template!r} must be a "
                    f"plain name with no format spec"
                )
            out.add(inner)
            i = j + 1
            continue
        i += 1
    return out


def has_forbidden_shell_chars(value: str) -> bool:
    """Return True if `value` contains any character we refuse to render verbatim."""
    return any(c in _FORBIDDEN_SHELL_CHARS for c in value)
