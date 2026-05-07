"""LabWalkthroughBuilder - turn a Demonstration into a human-reviewable Markdown.

User directive (Phase 4 / lab-flag handling):

    "labs flags model will not auto-sign this is user task only ! ...
     model will provide to user lab flags with full walkthrough!"

Rule of the road: when the agent is working a real HTB lab box (not academy),
it MUST NOT auto-submit the flag (account-ban risk). Instead, it produces a
walkthrough document the user reviews + uses to manually submit the flag.
This module is the renderer - it eats a ``Demonstration`` (the same format the
trainer reads) and emits Markdown.

Output format:

    # Walkthrough: <target_id>
    Outcome: foothold=Y user_flag=Y root_flag=N
    Total steps: N
    Total reward: +X.XX

    ## Step 1 - <tool_name>  (reward +0.10, techniques: T1046, ...)
    **Command:** `nmap -sS -T4 ...`
    **Output (truncated):**
    ```
    PORT     STATE SERVICE
    22/tcp   open  ssh
    ```

    ## Candidate flags (REVIEW BEFORE SUBMITTING)
    - HTB{...} -- found in step 5 output
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from htbrl.data.demo_dataset import Demonstration


# Same flag detector as HTBEnv.  We require the 32-hex-char string to appear
# alone on a line (cat user.txt convention) to avoid matching version banners.
_FLAG_LINE_RE = re.compile(r"^[a-f0-9]{32}$", re.MULTILINE)
_HTB_BRACED_RE = re.compile(r"HTB\{[^}]+\}")


@dataclass
class CandidateFlag:
    flag: str
    step_index: int      # 1-based
    rationale: str       # which regex matched + which output excerpt


def find_candidate_flags(demo: Demonstration) -> list[CandidateFlag]:
    out: list[CandidateFlag] = []
    seen: set[str] = set()
    for i, turn in enumerate(demo.turns, start=1):
        if not turn.obs_text:
            continue
        for m in _HTB_BRACED_RE.findall(turn.obs_text):
            if m in seen:
                continue
            seen.add(m)
            out.append(CandidateFlag(
                flag=m, step_index=i,
                rationale=f"HTB{{...}} match in observation following step {i}",
            ))
        for m in _FLAG_LINE_RE.findall(turn.obs_text):
            if m in seen:
                continue
            seen.add(m)
            out.append(CandidateFlag(
                flag=m, step_index=i,
                rationale=f"32-hex-char flag-only line in observation following step {i}",
            ))
    return out


def render_walkthrough(
    demo: Demonstration,
    *,
    output_truncate_chars: int = 4096,
    include_full_obs: bool = False,
) -> str:
    """Render `demo` as a Markdown walkthrough.

    ``include_full_obs=True`` keeps every observation byte; default truncates
    each output to ``output_truncate_chars`` so the doc stays readable.
    """
    lines: list[str] = []
    lines.append(f"# Walkthrough: {demo.target_id}")
    o = demo.outcome
    lines.append(
        f"_Outcome: foothold={'Y' if o.foothold else 'N'}, "
        f"user_flag={'Y' if o.user_flag else 'N'}, "
        f"root_flag={'Y' if o.root_flag else 'N'}_"
    )
    lines.append(f"_Total steps: {len(demo.turns)} ; "
                 f"Total reward: {demo.total_reward:+.3f}_")
    if o.note:
        lines.append(f"_Note: {o.note}_")
    if demo.metadata:
        lines.append("")
        lines.append("**Metadata:**")
        for k in sorted(demo.metadata):
            lines.append(f"- `{k}` = `{demo.metadata[k]!r}`")
    lines.append("")

    # Step-by-step
    for i, turn in enumerate(demo.turns, start=1):
        techs = ", ".join(turn.techniques_attempted) or "(none)"
        succ = ", ".join(turn.techniques_succeeded)
        succ_part = f", succeeded: {succ}" if succ else ""
        lines.append(
            f"## Step {i} — `{turn.action_tool_name}` "
            f"(reward {turn.reward:+.3f}; ATT&CK: {techs}{succ_part})"
        )
        if turn.action_render:
            lines.append("**Command:**")
            lines.append("")
            lines.append("```")
            lines.append(turn.action_render)
            lines.append("```")
        if turn.obs_text:
            obs = turn.obs_text
            if not include_full_obs and len(obs) > output_truncate_chars:
                obs = obs[:output_truncate_chars] + (
                    f"\n... <{len(turn.obs_text) - output_truncate_chars} bytes truncated>"
                )
            lines.append("")
            lines.append("**Observation (output of previous step):**")
            lines.append("")
            lines.append("```")
            lines.append(obs)
            lines.append("```")
        if turn.action_slots:
            lines.append("")
            lines.append("**Slot values:**")
            for k in sorted(turn.action_slots):
                lines.append(f"- `{k}` = `{turn.action_slots[k]!r}`")
        lines.append("")

    # Candidate flags - this is the section the user actually wants when
    # working a lab box. We always emit it, even if no flags found, with a
    # reminder that manual submission is the user's job.
    lines.append("---")
    lines.append("")
    lines.append("## Candidate flags (REVIEW BEFORE SUBMITTING)")
    lines.append("")
    lines.append("> The agent does **not** auto-submit lab flags. Confirm each "
                 "candidate, then submit manually on app.hackthebox.com.")
    lines.append("")
    cands = find_candidate_flags(demo)
    if not cands:
        lines.append("_No candidate flags detected in the trajectory._")
    else:
        for c in cands:
            lines.append(f"- `{c.flag}` — {c.rationale}")
    lines.append("")
    return "\n".join(lines)
