"""On-disk demonstration trajectory format and PyTorch Dataset.

A demonstration is one episode the user "wizard-mode-played" through (Phase 5
in PLAN.md). It stores everything the BC + PPO loops need:

- Which matrix the episode belongs to (enterprise / mobile / ics)
- Target identifier (HTB box name, Vulnhub VM, APK hash, PLC address)
- Per-turn obs / action / reward / ATT&CK technique tags
- Final outcome (foothold? user flag? root flag?)

Storage format: one ``.msgpack`` file per trajectory, gzip-compressed by default.
Files are loaded lazily by ``DemoDataset`` so a 5 GB dataset never lives in
RAM all at once.

Why msgpack and not JSON: msgpack handles bytes natively (raw nmap output may
contain non-UTF8) and serializes ~3x faster + ~2x smaller. msgpack-numpy adds
ndarray support which we use sparingly (token-id sequences if they're already
tokenized at write time).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import msgpack


_VALID_MATRICES = frozenset({"enterprise", "mobile", "ics"})


@dataclass
class DemoTurn:
    obs_text: str
    action_tool_id: int            # index into ActionVocabulary
    action_tool_name: str          # for human inspection / forwards-compat
    action_slots: dict[str, Any]   # raw slot values as authored in YAML
    action_render: str             # the rendered shell command
    reward: float                  # env reward (no RM, no RND)
    techniques_attempted: list[str] = field(default_factory=list)  # T1046 etc.
    techniques_succeeded: list[str] = field(default_factory=list)


@dataclass
class DemoOutcome:
    foothold: bool = False
    user_flag: bool = False
    root_flag: bool = False
    note: str = ""


@dataclass
class Demonstration:
    matrix: str
    target_id: str                 # e.g. "htb:starting-point:meow", "vulnhub:metasploitable2"
    turns: list[DemoTurn]
    outcome: DemoOutcome
    metadata: dict[str, Any] = field(default_factory=dict)  # author, timestamp, plugin versions

    def __post_init__(self) -> None:
        if self.matrix not in _VALID_MATRICES:
            raise ValueError(
                f"matrix {self.matrix!r} not in {sorted(_VALID_MATRICES)}"
            )

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.turns)


# ----- on-disk format ---------------------------------------------------------


def save_demonstration(demo: Demonstration, path: Path | str, compress: bool = True) -> None:
    """Write a single trajectory to disk. ``.msgpack.gz`` if compressed, else ``.msgpack``."""
    path = Path(path)
    payload = {
        "version": 1,
        "matrix": demo.matrix,
        "target_id": demo.target_id,
        "turns": [asdict(t) for t in demo.turns],
        "outcome": asdict(demo.outcome),
        "metadata": demo.metadata,
    }
    blob = msgpack.packb(payload, use_bin_type=True)
    if compress:
        with gzip.open(path, "wb", compresslevel=4) as f:
            f.write(blob)
    else:
        path.write_bytes(blob)


def load_demonstration(path: Path | str) -> Demonstration:
    """Read a single trajectory from disk."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            blob = f.read()
    else:
        blob = path.read_bytes()
    payload = msgpack.unpackb(blob, raw=False)
    if payload.get("version") != 1:
        raise ValueError(f"unsupported demo file version {payload.get('version')!r}")
    turns = [DemoTurn(**t) for t in payload["turns"]]
    outcome = DemoOutcome(**payload["outcome"])
    return Demonstration(
        matrix=payload["matrix"],
        target_id=payload["target_id"],
        turns=turns,
        outcome=outcome,
        metadata=payload.get("metadata", {}),
    )


# ----- dataset ----------------------------------------------------------------


class DemoDataset:
    """Lazy loader over a directory of demonstration files.

    Mirrors PyTorch's Dataset interface (``__len__`` and ``__getitem__``) but
    is intentionally framework-agnostic so we don't pay the import cost in
    tests that don't need torch.

    Usage:
        ds = DemoDataset(Path("data/demos"))
        ds = ds.filter(matrix="enterprise", outcome="foothold")
        for demo in ds: ...
    """

    def __init__(self, root: Path | str, glob: str = "*.msgpack*") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"demo root does not exist: {self.root}")
        # Sort for reproducibility across operating systems / filesystems.
        self.paths: list[Path] = sorted(p for p in self.root.glob(glob) if p.is_file())

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Demonstration:
        return load_demonstration(self.paths[idx])

    def __iter__(self):
        for p in self.paths:
            yield load_demonstration(p)

    # ----- filtering ----------------------------------------------------------

    def filter(
        self,
        matrix: str | None = None,
        outcome: str | None = None,
        min_turns: int = 0,
    ) -> "DemoDataset":
        """Return a new dataset over the subset matching the given predicates.

        Filtering re-loads each demo (cheap; metadata is small). The returned
        dataset preserves the same root path but holds a filtered ``paths`` list.
        """
        if outcome is not None and outcome not in {"foothold", "user_flag", "root_flag", "any_success"}:
            raise ValueError(f"unknown outcome filter {outcome!r}")
        out = DemoDataset.__new__(DemoDataset)
        out.root = self.root
        out.paths = []
        for p in self.paths:
            demo = load_demonstration(p)
            if matrix is not None and demo.matrix != matrix:
                continue
            if outcome is not None:
                if outcome == "foothold" and not demo.outcome.foothold:
                    continue
                elif outcome == "user_flag" and not demo.outcome.user_flag:
                    continue
                elif outcome == "root_flag" and not demo.outcome.root_flag:
                    continue
                elif outcome == "any_success" and not (
                    demo.outcome.foothold or demo.outcome.user_flag or demo.outcome.root_flag
                ):
                    continue
            if demo.n_turns < min_turns:
                continue
            out.paths.append(p)
        return out

    def stats(self) -> dict[str, Any]:
        """Aggregate counts useful for sanity-checking the dataset."""
        n_total = len(self.paths)
        n_by_matrix: dict[str, int] = {}
        n_foothold = 0
        n_user_flag = 0
        n_root_flag = 0
        total_turns = 0
        for demo in self:
            n_by_matrix[demo.matrix] = n_by_matrix.get(demo.matrix, 0) + 1
            if demo.outcome.foothold:
                n_foothold += 1
            if demo.outcome.user_flag:
                n_user_flag += 1
            if demo.outcome.root_flag:
                n_root_flag += 1
            total_turns += demo.n_turns
        return {
            "n_demos": n_total,
            "n_by_matrix": n_by_matrix,
            "n_foothold": n_foothold,
            "n_user_flag": n_user_flag,
            "n_root_flag": n_root_flag,
            "total_turns": total_turns,
            "avg_turns_per_demo": total_turns / n_total if n_total else 0.0,
        }
