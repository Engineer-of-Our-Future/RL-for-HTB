"""Helpers for the wizard-mode demo collection CLI (Phase 5).

The interactive loop lives in ``scripts/collect_demos.py``; the testable
helper logic lives here so we can unit-test it without simulating stdin.

What we expose:
- ``suggest_tools(vocab, command, n=3)`` -> ranked candidate tools whose
  command templates / names look like ``command``.
- ``parse_reward(text, default)`` -> tolerant float parser (accepts "+0.5", "1",
  "user_flag" alias).
- ``coerce_slot_value(slot, raw)`` -> type-aware coercion so
  prompts can defer validation to the registry's existing validators.
"""

from __future__ import annotations

import difflib
import shlex
from typing import Any

from htbrl.tools.loader import ActionVocabulary, SlotValidationError, _validate_and_format
from htbrl.tools.schema import (
    EnumSlot,
    IntSlot,
    IpSlot,
    PortSlot,
    Slot,
    ToolDefinition,
    WordlistSlot,
)


# ---- tool suggestion --------------------------------------------------------


# Reward-shortcut aliases the wizard accepts in place of a numeric reward.
# These mirror the Phase 4 reward primitives in src/htbrl/env/htb_env.py so a
# manual demo session reports the same way an automated rollout would.
_REWARD_ALIASES = {
    "step": -0.01,
    "timeout": -0.1,
    "blocked": -0.5,
    "new_port": 0.1,
    "new_service": 0.2,
    "user_shell": 0.5,
    "user_flag": 1.0,
    "root_shell": 1.5,
    "root_flag": 2.0,
}


def suggest_tools(
    vocab: ActionVocabulary,
    command: str,
    n: int = 3,
) -> list[tuple[float, ToolDefinition]]:
    """Rank tools by string similarity between the user's typed bash and the
    tool's name + command_template's first token. Returns up to ``n`` results
    sorted by descending score.

    A score >= 0.95 indicates the first token matched exactly (e.g. user typed
    ``nmap ...`` and we found ``nmap_*`` tools); the wizard treats that as
    "auto-pick eligible" by default.
    """
    if not command.strip():
        return []
    try:
        cmd_tokens = shlex.split(command)
    except ValueError:
        cmd_tokens = command.split()
    cmd_t0 = cmd_tokens[0] if cmd_tokens else ""

    scored: list[tuple[float, ToolDefinition]] = []
    for tool in vocab.tools:
        tmpl_t0 = ""
        try:
            tmpl_t0 = shlex.split(tool.command_template)[0] if tool.command_template else ""
        except ValueError:
            tmpl_t0 = tool.command_template.split()[0] if tool.command_template else ""

        s_name = difflib.SequenceMatcher(None, cmd_t0, tool.name).ratio()
        s_tmpl = difflib.SequenceMatcher(None, cmd_t0, tmpl_t0).ratio()
        score = max(s_name, s_tmpl)
        # Hard boost: exact match on the leading binary
        if cmd_t0 and tmpl_t0 and cmd_t0 == tmpl_t0:
            score = max(score, 0.95)
        # Soft boost: cmd substring of tool name (e.g. "smbclient" -> "smbclient_list")
        if cmd_t0 and cmd_t0 in tool.name:
            score = max(score, 0.85)
        scored.append((score, tool))

    scored.sort(key=lambda x: (-x[0], x[1].name))
    return scored[:n]


# ---- reward parser ----------------------------------------------------------


def parse_reward(text: str, default: float = -0.01) -> float:
    """Parse a reward literal or alias.

    Accepts:
      - ``""``  -> default
      - ``+0.5`` / ``-0.1`` / ``1`` -> float
      - ``user_flag`` / ``new_port`` / ``timeout`` / ``...`` -> alias lookup
    """
    s = (text or "").strip()
    if not s:
        return default
    if s in _REWARD_ALIASES:
        return _REWARD_ALIASES[s]
    try:
        return float(s)
    except ValueError:
        return default


# ---- slot coercion ----------------------------------------------------------


def coerce_slot_value(slot: Slot, raw: str) -> Any:
    """Coerce a raw string from input() into the right Python type for ``slot``.

    The downstream call to ``ActionVocabulary.render`` does the actual
    validation; this just gets the type right so render's checks fire on
    a sensible value (e.g. "22" -> 22 for a port).
    """
    if isinstance(slot, (IntSlot, PortSlot)):
        try:
            return int(raw)
        except ValueError as exc:
            raise SlotValidationError(
                f"slot {slot.name!r}: expected integer, got {raw!r}"
            ) from exc
    return raw


def render_or_fallback(
    vocab: ActionVocabulary,
    tool: ToolDefinition,
    slots: dict[str, Any],
    fallback: str,
) -> tuple[str, list[str]]:
    """Render a tool invocation, capturing validation errors as warnings.

    Returns ``(rendered_command, warnings)`` so the wizard can show the user
    why a render failed without crashing the whole episode.
    """
    warnings: list[str] = []
    try:
        return vocab.render(tool.name, slots), warnings
    except SlotValidationError as exc:
        warnings.append(str(exc))
        return fallback, warnings


# Re-export so callers can validate slots one at a time without importing
# the loader's underscore-private helper.
validate_slot_value = _validate_and_format
