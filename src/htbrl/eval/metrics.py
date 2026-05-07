"""Evaluation metrics (PLAN.md Phase 11).

Episode-level results aggregate up to a SuiteResult that's reported per matrix.
Metrics organized by:
- Outcome: foothold rate, root rate, flag-time
- ATT&CK coverage: technique attempt coverage, tactic completion rate, killchain depth
- Health: vocab coverage, KL drift (computed externally)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from htbrl.env.base import StepInfo


@dataclass
class EpisodeResult:
    matrix: str
    target_id: str
    total_reward: float
    n_steps: int
    foothold: bool = False
    user_flag: bool = False
    root_flag: bool = False
    techniques_attempted: set[str] = field(default_factory=set)
    techniques_succeeded: set[str] = field(default_factory=set)
    # tactic_id -> True if at least one technique under it succeeded this episode
    tactics_completed: set[str] = field(default_factory=set)
    # max sequential tactics chained in a single episode
    killchain_depth: int = 0
    # tools the policy used at least once this episode (by tool_id)
    tools_used: set[int] = field(default_factory=set)
    # accumulated step infos for debugging / replay
    step_infos: list[StepInfo] = field(default_factory=list)


@dataclass
class SuiteResult:
    episodes: list[EpisodeResult] = field(default_factory=list)

    # ----- outcome metrics ----------------------------------------------------

    @property
    def foothold_rate(self) -> float:
        return _rate(e.foothold for e in self.episodes)

    @property
    def user_flag_rate(self) -> float:
        return _rate(e.user_flag for e in self.episodes)

    @property
    def root_flag_rate(self) -> float:
        return _rate(e.root_flag for e in self.episodes)

    @property
    def root_rate(self) -> float:
        """Fraction of episodes that achieved any root-level outcome."""
        return _rate(e.root_flag for e in self.episodes)

    @property
    def avg_episode_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.n_steps for e in self.episodes) / len(self.episodes)

    @property
    def avg_total_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_reward for e in self.episodes) / len(self.episodes)

    # ----- ATT&CK metrics -----------------------------------------------------

    def technique_attempt_coverage(self) -> dict[str, set[str]]:
        """Union of techniques attempted, grouped by matrix."""
        out: dict[str, set[str]] = defaultdict(set)
        for e in self.episodes:
            out[e.matrix] |= e.techniques_attempted
        return dict(out)

    def technique_success_coverage(self) -> dict[str, set[str]]:
        """Union of techniques *successfully* exercised, grouped by matrix."""
        out: dict[str, set[str]] = defaultdict(set)
        for e in self.episodes:
            out[e.matrix] |= e.techniques_succeeded
        return dict(out)

    def tactic_completion_rate(self) -> dict[str, dict[str, float]]:
        """For each matrix, for each tactic seen, fraction of episodes where it completed."""
        # matrix -> tactic_id -> [n_episodes_with_that_tactic_seen, n_completed]
        cnt: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        # We count an episode as "exposed to" a tactic if the tactic was attempted
        # at least once across all episodes for that matrix; alternatively we
        # could only count tactics the registry knows about. Keep it simple here.
        all_tactics_per_matrix: dict[str, set[str]] = defaultdict(set)
        for e in self.episodes:
            for s in e.step_infos:
                all_tactics_per_matrix[e.matrix] |= set(s.tactic_ids)

        for e in self.episodes:
            for tac in all_tactics_per_matrix[e.matrix]:
                cnt[e.matrix][tac][0] += 1
                if tac in e.tactics_completed:
                    cnt[e.matrix][tac][1] += 1
        return {
            m: {tac: (n_done / max(n_total, 1)) for tac, (n_total, n_done) in tactics.items()}
            for m, tactics in cnt.items()
        }

    def avg_killchain_depth(self) -> dict[str, float]:
        out: dict[str, list[int]] = defaultdict(list)
        for e in self.episodes:
            out[e.matrix].append(e.killchain_depth)
        return {m: sum(xs) / len(xs) for m, xs in out.items() if xs}

    # ----- vocab health -------------------------------------------------------

    def vocab_coverage(self, total_tools_per_matrix: dict[str, int]) -> dict[str, float]:
        """Fraction of available tools that were used at least once per matrix."""
        used_per_matrix: dict[str, set[int]] = defaultdict(set)
        for e in self.episodes:
            used_per_matrix[e.matrix] |= e.tools_used
        return {
            m: len(used_per_matrix[m]) / max(total_tools_per_matrix.get(m, 1), 1)
            for m in used_per_matrix
        }

    # ----- summary ------------------------------------------------------------

    def summary_dict(self) -> dict:
        """Compact dict suitable for JSON / TensorBoard logging."""
        return {
            "n_episodes": len(self.episodes),
            "foothold_rate": self.foothold_rate,
            "user_flag_rate": self.user_flag_rate,
            "root_flag_rate": self.root_flag_rate,
            "avg_total_reward": self.avg_total_reward,
            "avg_episode_length": self.avg_episode_length,
            "technique_attempt_coverage": {
                m: sorted(s) for m, s in self.technique_attempt_coverage().items()
            },
            "technique_success_coverage": {
                m: sorted(s) for m, s in self.technique_success_coverage().items()
            },
            "tactic_completion_rate": self.tactic_completion_rate(),
            "avg_killchain_depth": self.avg_killchain_depth(),
        }


def _rate(it: Iterable[bool]) -> float:
    xs = list(it)
    return sum(1 for x in xs if x) / max(len(xs), 1)


def killchain_depth_from_steps(step_infos: list[StepInfo]) -> int:
    """Compute max contiguous distinct-tactic chain over a sequence of steps.

    A tactic transitions count toward depth only when the new step *succeeded*
    (techniques_succeeded non-empty). A failed step in between does not break
    the chain - the chain measures distinct successful tactics in order.
    """
    seen_in_chain: list[str] = []
    max_depth = 0
    for info in step_infos:
        if not info.techniques_succeeded:
            continue
        for tac in info.tactic_ids:
            if tac not in seen_in_chain:
                seen_in_chain.append(tac)
                max_depth = max(max_depth, len(seen_in_chain))
    return max_depth
