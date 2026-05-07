"""Wizard-mode demo collection CLI (Phase 5).

Drives one demo-recording episode against a Kali attacker over SSH:

  1. You type a bash command at the > prompt.
  2. The wizard runs it via SSH, shows the (truncated) output.
  3. The wizard suggests the top-3 tools from the registry whose
     command_templates / names look like what you typed; you pick one
     (or 'manual' to log it as a free-text turn, or 'skip' to drop it).
  4. The wizard prompts for each slot value, then computes a reward
     (you can type a number or an alias like ``user_flag`` / ``new_port``).
  5. Repeat. Type ``done`` when the episode is over, ``abort`` to discard.
  6. The wizard asks for outcome flags (foothold / user_flag / root_flag),
     then writes a Demonstration to ``data/demos/`` in the same format
     ``scripts/train_bc.py`` reads.

Examples:

    # Connect to the WSL Kali we set up in Phase 4 and record one demo
    python scripts/collect_demos.py \\
        --target-id htb-starting-point:meow \\
        --kali-host htbrl@127.0.0.1:2222 \\
        --kali-key ~/.ssh/htbrl_kali

    # Offline mode (no SSH; you paste tool output manually)
    python scripts/collect_demos.py --target-id vulnhub:metasploitable2 --offline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from htbrl.data.demo_dataset import (
    Demonstration,
    DemoOutcome,
    DemoTurn,
    save_demonstration,
)
from htbrl.data.wizard import (
    coerce_slot_value,
    parse_reward,
    render_or_fallback,
    suggest_tools,
)
from htbrl.env.ssh_session import SSHCredentials, SSHSession
from htbrl.tools.loader import ActionVocabulary, SlotValidationError, load_registry


# ---- I/O helpers (override-able for testing) --------------------------------


def _ask(prompt: str) -> str:
    """input() wrapper so tests can monkeypatch this in one place."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def _print_help() -> None:
    print("  done    -> finish the episode and save the Demonstration")
    print("  abort   -> discard everything and exit")
    print("  :help   -> show this help")
    print("  reward shortcuts: " + ", ".join(
        ["step", "timeout", "blocked", "new_port", "new_service",
         "user_shell", "user_flag", "root_shell", "root_flag"]
    ))


def _prompt_slots(tool, vocab: ActionVocabulary) -> dict | None:
    """Walk through each slot of the tool, return the slot dict or None on abort."""
    slots: dict = {}
    for slot in tool.slots:
        default = getattr(slot, "default", None)
        type_str = getattr(slot, "type", "?")
        prompt = f"  {slot.name} ({type_str}"
        if default is not None:
            prompt += f", default={default}"
        prompt += "): "
        while True:
            v = _ask(prompt).strip()
            if v == ":abort":
                return None
            if not v and default is not None:
                slots[slot.name] = default
                break
            if not v and not slot.required:
                break
            try:
                slots[slot.name] = coerce_slot_value(slot, v)
                break
            except SlotValidationError as exc:
                print(f"  -> {exc}; try again or type ':abort' to skip this turn")
    return slots


# ---- main loop --------------------------------------------------------------


def run_episode(
    *,
    vocab: ActionVocabulary,
    ssh: SSHSession | None,
    target_id: str,
    auto_suggest_threshold: float = 0.95,
    cmd_timeout_s: float = 120.0,
) -> list[DemoTurn]:
    """Drive one interactive episode. Returns the list of recorded turns."""
    turns: list[DemoTurn] = []
    print(f"[wizard] target={target_id}; {vocab.n_tools} tools available")
    print("[wizard] type a bash command (or 'done' / 'abort' / ':help')")

    last_obs = "<wizard episode start>"
    while True:
        cmd = _ask("> ").strip()
        if not cmd:
            continue
        if cmd == "done":
            break
        if cmd == "abort":
            print("[wizard] aborted; nothing will be saved")
            return []
        if cmd == ":help":
            _print_help()
            continue

        # Execute (or fake-execute in offline mode)
        if ssh is not None:
            result = ssh.run(cmd, timeout=cmd_timeout_s)
            output = result.stdout
            timed_out = result.timed_out
        else:
            print("[wizard] paste output (END with a single '.' on its own line):")
            output_lines: list[str] = []
            while True:
                line = _ask("")
                if line.strip() == ".":
                    break
                output_lines.append(line)
            output = "\n".join(output_lines)
            timed_out = False

        if timed_out:
            print("[wizard] (command timed out)")
        truncated = output if len(output) <= 2048 else output[:2048] + f"\n... ({len(output)} bytes total)"
        print(f"[wizard] output:\n{truncated}")

        # Suggest matching tool(s)
        suggestions = suggest_tools(vocab, cmd, n=3)
        chosen = None
        slots: dict = {}
        if suggestions and suggestions[0][0] >= auto_suggest_threshold:
            score, chosen = suggestions[0]
            print(f"[wizard] auto-pick: {chosen.name} (score={score:.2f})")
        elif suggestions:
            print("[wizard] top matches:")
            for i, (score, t) in enumerate(suggestions, 1):
                print(f"  [{i}] {t.name} ({t.category.value})  score={score:.2f}")
                print(f"      template: {t.command_template}")
            pick = _ask("[wizard] pick [1-3] / 'manual' / 'skip': ").strip().lower()
            if pick == "skip":
                print("[wizard] skipped (turn not logged)")
                last_obs = f"<output of: {cmd}>\n{output[:4096]}"
                continue
            if pick == "manual":
                chosen = None
            else:
                try:
                    idx = int(pick) - 1
                    chosen = suggestions[idx][1]
                except (ValueError, IndexError):
                    print("[wizard] invalid pick; skipping turn")
                    continue
        else:
            print("[wizard] no tool suggestions; logging as 'manual_command' (raw bash)")
            chosen = None

        if chosen is not None:
            slots = _prompt_slots(chosen, vocab) or {}
            if not slots and chosen.slots:
                # User typed :abort during slot prompts.
                print("[wizard] turn skipped")
                continue

        reward = parse_reward(_ask("[wizard] reward (default: 'step' = -0.01): "))
        # Render the command via the registry so the demo records what the
        # agent's renderer would have produced (not just what the user typed).
        if chosen is not None:
            rendered, warns = render_or_fallback(vocab, chosen, slots, fallback=cmd)
            for w in warns:
                print(f"[wizard] WARN: {w}")
            tool_id = vocab.id_of(chosen.name)
            tool_name = chosen.name
            techniques_attempted = list(chosen.attack.techniques)
        else:
            rendered = cmd
            tool_id = -1
            tool_name = "manual_command"
            techniques_attempted = []
            slots = {"command": cmd}

        techniques_succeeded = techniques_attempted if reward > 0 else []
        turn = DemoTurn(
            obs_text=last_obs,
            action_tool_id=tool_id,
            action_tool_name=tool_name,
            action_slots=slots,
            action_render=rendered,
            reward=reward,
            techniques_attempted=techniques_attempted,
            techniques_succeeded=techniques_succeeded,
        )
        turns.append(turn)
        last_obs = f"<output of: {rendered}>\n{output[:4096]}"
        print(f"[wizard] logged turn {len(turns)}: {tool_name} reward={reward:+.3f}")

    return turns


def _open_ssh(args: argparse.Namespace) -> SSHSession | None:
    if args.offline or not args.kali_host:
        return None
    user_at_host, _, port_str = args.kali_host.partition(":")
    user, _, host = user_at_host.partition("@")
    creds = SSHCredentials(
        host=host,
        port=int(port_str) if port_str else 22,
        user=user,
        identity_file=os.path.expanduser(args.kali_key) if args.kali_key else None,
        connect_timeout_seconds=10.0,
    )
    sess = SSHSession(creds)
    sess.open()
    print(f"[wizard] SSH connected -> {creds.user}@{creds.host}:{creds.port}")
    return sess


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-id", required=True,
                   help="e.g. 'htb-starting-point:meow' or 'vulnhub:metasploitable2'")
    p.add_argument("--matrix", default="enterprise",
                   choices=["enterprise", "mobile", "ics"])
    p.add_argument("--kali-host", default=os.environ.get("HTBRL_KALI_HOST"),
                   help="user@host[:port]; default = $HTBRL_KALI_HOST")
    p.add_argument("--kali-key", default=os.environ.get("HTBRL_KALI_KEY"),
                   help="SSH private key path; default = $HTBRL_KALI_KEY")
    p.add_argument("--offline", action="store_true",
                   help="no SSH; paste tool output manually")
    p.add_argument("--output-dir", type=Path, default=Path("data/demos"))
    p.add_argument("--auto-suggest-threshold", type=float, default=0.95,
                   help="auto-pick top match if score >= this; 1.0 disables")
    p.add_argument("--cmd-timeout-s", type=float, default=120.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    vocab = load_registry()
    ssh = _open_ssh(args)

    try:
        turns = run_episode(
            vocab=vocab, ssh=ssh, target_id=args.target_id,
            auto_suggest_threshold=args.auto_suggest_threshold,
            cmd_timeout_s=args.cmd_timeout_s,
        )
    finally:
        if ssh is not None:
            ssh.close()

    if not turns:
        print("[wizard] no turns recorded; nothing saved")
        return 0

    print(f"[wizard] {len(turns)} turn(s) recorded. Outcome flags:")
    foothold = _ask("  foothold? [y/N]: ").strip().lower() == "y"
    user_flag = _ask("  user_flag? [y/N]: ").strip().lower() == "y"
    root_flag = _ask("  root_flag? [y/N]: ").strip().lower() == "y"
    note = _ask("  note (optional): ").strip()

    demo = Demonstration(
        matrix=args.matrix,
        target_id=args.target_id,
        turns=turns,
        outcome=DemoOutcome(foothold=foothold, user_flag=user_flag,
                            root_flag=root_flag, note=note),
        metadata={
            "source": "wizard-mode",
            "ts": time.time(),
            "kali_host": args.kali_host or "(offline)",
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_safe = args.target_id.replace(":", "-").replace("/", "_")
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    path = args.output_dir / f"wizard_{target_safe}_{ts}.msgpack.gz"
    save_demonstration(demo, path)
    print(f"[wizard] saved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
