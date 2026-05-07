"""Real PentestEnv: agent runs structured tool calls in a Kali shell over SSH.

Compared to the stub env, this one:
- actually executes commands and reads stdout
- enforces a target-IP allowlist (refuses any rendered command that mentions
  an IP outside the allowlist - critical safety boundary)
- runs the registered output parser to produce structured features
- detects flags and shell milestones for reward shaping

Phase 4 scope: covers the pre-foothold portion of the kill-chain (Recon /
Enum / Web / Initial Access). Post-foothold reward signals (user shell, root
shell) are detected from output but the actual session-state tracking they'd
need (which target the agent has shells on) is intentionally minimal here -
expanded in Phases 5-7 when the multi-target rollout layer lands.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any

from htbrl.env.base import Action, Observation, PentestEnv, StepInfo
from htbrl.env.parsers import parse as run_parser
from htbrl.env.ssh_session import SSHCredentials, SSHSession
from htbrl.tools.loader import ActionVocabulary


# Markers we look for in tool output as proxies for milestones.
# Flags are HTB-style 32-hex-char strings. We require the hex to appear ALONE
# on its line (`cat user.txt` style output) so we don't false-match on hex
# fingerprints embedded in nmap version banners or SSH host keys.
_FLAG_LINE_RE = re.compile(r"^[a-f0-9]{32}$", re.MULTILINE)
_USER_SHELL_MARKERS = (
    re.compile(r"\$\s*$"),
    re.compile(r"\bbash-\d", re.IGNORECASE),
)
_ROOT_SHELL_MARKERS = (
    re.compile(r"^\s*#\s*$", re.MULTILINE),
    re.compile(r"\buid=0\(root\)"),
)


# Default reward primitives matching configs/env/htb.yaml.
DEFAULT_REWARDS = {
    "per_command_penalty": -0.01,
    "timeout_penalty": -0.1,
    "reset_penalty": -0.5,
    "new_port": 0.1,
    "new_service_version": 0.2,
    "user_shell": 0.5,
    "user_flag": 1.0,
    "root_shell": 1.5,
    "root_flag": 2.0,
}


def _ip_in_any_cidr(ip: str, cidrs: list[ipaddress._BaseNetwork]) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(ip_obj in net for net in cidrs)


def _extract_ips_from_command(rendered: str) -> list[str]:
    """Pull IPv4 / IPv6 / CIDR-style strings out of the rendered command line."""
    # Simple IPv4 / IPv4-CIDR regex - we don't need IPv6 in initial release.
    out: list[str] = []
    for m in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", rendered):
        token = m.group(0)
        # Strip CIDR for the check; allowlist contains CIDRs, IPs check for membership.
        ip_part = token.split("/", 1)[0]
        out.append(ip_part)
    return out


class HTBEnv(PentestEnv):
    """Real env. One env instance == one persistent SSH session into Kali."""

    matrix = "enterprise"

    def __init__(
        self,
        vocab: ActionVocabulary,
        ssh_creds: SSHCredentials,
        allowlist_cidrs: list[str],
        max_steps: int = 60,
        rewards: dict[str, float] | None = None,
        episode_target_id: str = "unknown",
    ) -> None:
        if not allowlist_cidrs:
            raise ValueError(
                "allowlist_cidrs is required - the env refuses to run commands "
                "without an explicit allowlist of target subnets"
            )
        self.vocab = vocab
        self.ssh_creds = ssh_creds
        self.allowlist = [ipaddress.ip_network(c, strict=False) for c in allowlist_cidrs]
        self.max_steps = max_steps
        self.rewards = dict(DEFAULT_REWARDS)
        if rewards is not None:
            self.rewards.update(rewards)
        self.episode_target_id = episode_target_id

        self._session: SSHSession | None = None
        self._step = 0
        # Per-episode discovery memory (so we only reward the FIRST sighting).
        self._seen_ports: set[tuple[str, int, str]] = set()
        self._seen_versions: set[tuple[str, int, str]] = set()
        self._seen_flags: set[str] = set()
        self._user_shell_acquired = False
        self._root_shell_acquired = False
        self._closed = False

    # ----- lifecycle ----------------------------------------------------------

    def _ensure_session(self) -> SSHSession:
        if self._session is None:
            self._session = SSHSession(self.ssh_creds)
            self._session.open()
        return self._session

    def reset(self, seed: int | None = None) -> Observation:
        # We don't recycle SSH sessions across episodes - cleanest reset is a
        # brand-new shell, which avoids leaked env vars / shell state.
        if self._session is not None:
            self._session.close()
            self._session = None
        self._step = 0
        self._seen_ports.clear()
        self._seen_versions.clear()
        self._seen_flags.clear()
        self._user_shell_acquired = False
        self._root_shell_acquired = False
        return Observation(
            obs_text="<htb-env>: episode reset; SSH session not yet opened.",
            parsed_features={"step": 0},
            last_reward=0.0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, StepInfo]:
        if self._closed:
            raise RuntimeError("step() on closed env")

        self._step += 1
        info = StepInfo()

        # Render via the registry. SlotValidationError surfaces directly.
        try:
            rendered = self.vocab.render(action.tool_id, action.slots)
        except Exception as exc:
            info.extras["render_error"] = str(exc)
            obs = Observation(
                obs_text=f"<render-error>: {exc}",
                parsed_features={"render_error": str(exc)},
                last_reward=self.rewards["per_command_penalty"],
            )
            return obs, self.rewards["per_command_penalty"], self._step >= self.max_steps, info

        info.rendered_command = rendered

        # Allowlist enforcement: every IP-shaped token in the rendered command
        # must fall inside the allowlist. CIDR slots use ip_network membership.
        ips_in_cmd = _extract_ips_from_command(rendered)
        for ip in ips_in_cmd:
            if not _ip_in_any_cidr(ip, self.allowlist):
                info.extras["allowlist_violation"] = ip
                obs = Observation(
                    obs_text=f"<allowlist-block>: {ip} not in allowlist",
                    parsed_features={"allowlist_violation": ip},
                    last_reward=self.rewards["reset_penalty"],
                )
                return (
                    obs,
                    self.rewards["reset_penalty"],
                    True,  # terminate this episode immediately
                    info,
                )

        # Pull tool def for parser_id + cap + ATT&CK metadata.
        tool_def = self.vocab.tools[action.tool_id]
        info.parser_id = tool_def.output_parser_id
        info.techniques_attempted = list(tool_def.attack.techniques)
        info.tactic_ids = list(tool_def.attack.tactics)

        # Execute over SSH.
        sess = self._ensure_session()
        result = sess.run(rendered, timeout=float(tool_def.runtime_cap_seconds))
        info.timed_out = result.timed_out

        # Parse.
        parsed = run_parser(tool_def.output_parser_id, result.stdout)

        # Reward shaping.
        reward = self.rewards["per_command_penalty"]
        if result.timed_out:
            reward += self.rewards["timeout_penalty"]
        else:
            # Discovery rewards from parsed features.
            if isinstance(parsed.get("ports"), list):
                target_ip = ips_in_cmd[0] if ips_in_cmd else "unknown"
                for p in parsed["ports"]:
                    key = (target_ip, int(p["port"]), p.get("proto", "tcp"))
                    if key not in self._seen_ports:
                        self._seen_ports.add(key)
                        reward += self.rewards["new_port"]
                    if p.get("version"):
                        vkey = (target_ip, int(p["port"]), p["version"])
                        if vkey not in self._seen_versions:
                            self._seen_versions.add(vkey)
                            reward += self.rewards["new_service_version"]
            # Flag detection - HTB flags appear alone on a line, not embedded.
            for m in _FLAG_LINE_RE.findall(result.stdout):
                if m not in self._seen_flags:
                    self._seen_flags.add(m)
                    if not self._user_shell_acquired:
                        # Heuristic: first flag seen without a shell is likely user.txt.
                        reward += self.rewards["user_flag"]
                    else:
                        reward += self.rewards["root_flag"]
            # Shell-milestone heuristic from output text.
            if not self._user_shell_acquired and any(rx.search(result.stdout) for rx in _USER_SHELL_MARKERS):
                self._user_shell_acquired = True
                reward += self.rewards["user_shell"]
            if not self._root_shell_acquired and any(rx.search(result.stdout) for rx in _ROOT_SHELL_MARKERS):
                self._root_shell_acquired = True
                reward += self.rewards["root_shell"]

            # Outcome flags exposed to the eval harness.
            info.extras["foothold"] = self._user_shell_acquired
            info.extras["user_flag"] = self._user_shell_acquired and bool(self._seen_flags)
            info.extras["root_flag"] = self._root_shell_acquired and len(self._seen_flags) >= 2
            # techniques_succeeded heuristic: parser got real structured content
            if parsed.get("ports") or parsed.get("shares") or parsed.get("lines"):
                info.techniques_succeeded = list(tool_def.attack.techniques)

        # Build observation.
        obs_text_lines = []
        if result.stdout:
            obs_text_lines.append(result.stdout[:8192])  # cap to keep tokens bounded
        if result.timed_out:
            obs_text_lines.append("<timed-out>")
        obs_text = "\n".join(obs_text_lines) or "<no-output>"
        obs = Observation(
            obs_text=obs_text,
            parsed_features={k: v for k, v in parsed.items() if k != "text"},
            last_reward=reward,
        )

        done = self._step >= self.max_steps or info.extras.get("root_flag", False)
        return obs, reward, done, info

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._closed = True
