"""SQLite-backed store for trajectory snippets and pairwise preferences (Phase 8).

Two tables:

  ``snippets`` (id, matrix, content_text, source, created_at, metadata_json)
  ``preferences`` (id, left_id, right_id, label, labeler, created_at,
                   notes, label_confidence)

  ``label`` is one of:  ``"left"``, ``"right"``, ``"tie"``, ``"discard"``.

The DB path is treated as a single source of truth across the FastAPI UI
(adds new preferences) and the RM training script (reads them). All access
goes through ``PreferenceStore`` so we can swap to a different backend later
without touching callers.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


_VALID_LABELS = frozenset({"left", "right", "tie", "discard"})


@dataclass
class Snippet:
    id: int
    matrix: str
    content_text: str
    source: str = "unknown"
    created_at: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass
class Preference:
    id: int
    left_id: int
    right_id: int
    label: str
    labeler: str
    created_at: float
    notes: str = ""
    label_confidence: float = 1.0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS snippets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    matrix        TEXT NOT NULL,
    content_text  TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'unknown',
    created_at    REAL NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_snippets_matrix ON snippets(matrix);

CREATE TABLE IF NOT EXISTS preferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    left_id          INTEGER NOT NULL,
    right_id         INTEGER NOT NULL,
    label            TEXT NOT NULL,
    labeler          TEXT NOT NULL DEFAULT 'human',
    created_at       REAL NOT NULL,
    notes            TEXT NOT NULL DEFAULT '',
    label_confidence REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY(left_id)  REFERENCES snippets(id),
    FOREIGN KEY(right_id) REFERENCES snippets(id)
);
"""


class PreferenceStore:
    """Connection-per-instance wrapper around the sqlite preference DB."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is fine here: we're a single-process app,
        # write contention is low (one human labeling at a time), and isolation
        # is autocommit so no cross-thread transaction state to worry about.
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        # Enable foreign-key enforcement and row factory for nicer reads.
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- snippets -----------------------------------------------------------

    def add_snippet(
        self,
        matrix: str,
        content_text: str,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if matrix not in {"enterprise", "mobile", "ics"}:
            raise ValueError(f"invalid matrix {matrix!r}")
        cur = self._conn.execute(
            "INSERT INTO snippets (matrix, content_text, source, created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (matrix, content_text, source, time.time(),
             json.dumps(metadata) if metadata is not None else None),
        )
        return int(cur.lastrowid)

    def get_snippet(self, snippet_id: int) -> Snippet:
        row = self._conn.execute(
            "SELECT * FROM snippets WHERE id = ?", (snippet_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snippet_id)
        return _snippet_from_row(row)

    def n_snippets(self, matrix: str | None = None) -> int:
        if matrix is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM snippets WHERE matrix = ?", (matrix,)
            ).fetchone()[0]
        )

    def random_pair(self, matrix: str | None = None) -> tuple[Snippet, Snippet]:
        """Return two distinct snippets sampled uniformly at random."""
        where = ""
        params: tuple = ()
        if matrix is not None:
            where = "WHERE matrix = ?"
            params = (matrix,)
        rows = self._conn.execute(
            f"SELECT * FROM snippets {where} ORDER BY RANDOM() LIMIT 2",
            params,
        ).fetchall()
        if len(rows) < 2:
            raise RuntimeError(
                f"need at least 2 snippets to form a pair (matrix={matrix})"
            )
        return _snippet_from_row(rows[0]), _snippet_from_row(rows[1])

    def all_snippets(self, matrix: str | None = None) -> list[Snippet]:
        where = ""
        params: tuple = ()
        if matrix is not None:
            where = "WHERE matrix = ?"
            params = (matrix,)
        rows = self._conn.execute(
            f"SELECT * FROM snippets {where} ORDER BY id ASC", params
        ).fetchall()
        return [_snippet_from_row(r) for r in rows]

    # ----- preferences --------------------------------------------------------

    def add_preference(
        self,
        left_id: int,
        right_id: int,
        label: str,
        labeler: str = "human",
        notes: str = "",
        label_confidence: float = 1.0,
    ) -> int:
        if label not in _VALID_LABELS:
            raise ValueError(f"label {label!r} not in {_VALID_LABELS}")
        if left_id == right_id:
            raise ValueError("left and right must be different snippets")
        cur = self._conn.execute(
            "INSERT INTO preferences (left_id, right_id, label, labeler, created_at, notes, label_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (left_id, right_id, label, labeler, time.time(), notes, label_confidence),
        )
        return int(cur.lastrowid)

    def n_preferences(self, exclude_discard: bool = False) -> int:
        if exclude_discard:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM preferences WHERE label != 'discard'"
                ).fetchone()[0]
            )
        return int(self._conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0])

    def all_preferences(self, exclude_discard: bool = True) -> list[Preference]:
        sql = "SELECT * FROM preferences"
        if exclude_discard:
            sql += " WHERE label != 'discard'"
        sql += " ORDER BY id ASC"
        return [_pref_from_row(r) for r in self._conn.execute(sql).fetchall()]

    # ----- training-time accessors --------------------------------------------

    def iter_training_pairs(
        self, exclude_discard: bool = True, exclude_ties: bool = True
    ) -> Iterable[tuple[Snippet, Snippet, str]]:
        """Yield ``(preferred, other, original_label)`` triples ready for BT loss.

        Ties are skipped by default. ``"left"`` -> preferred=left snippet;
        ``"right"`` -> preferred=right.
        """
        for pref in self.all_preferences(exclude_discard=exclude_discard):
            if pref.label == "tie" and exclude_ties:
                continue
            left = self.get_snippet(pref.left_id)
            right = self.get_snippet(pref.right_id)
            if pref.label == "left":
                yield left, right, pref.label
            elif pref.label == "right":
                yield right, left, pref.label
            elif pref.label == "tie":
                yield left, right, pref.label  # caller decides what to do


def _snippet_from_row(row: sqlite3.Row) -> Snippet:
    return Snippet(
        id=int(row["id"]),
        matrix=str(row["matrix"]),
        content_text=str(row["content_text"]),
        source=str(row["source"]),
        created_at=float(row["created_at"]),
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
    )


def _pref_from_row(row: sqlite3.Row) -> Preference:
    return Preference(
        id=int(row["id"]),
        left_id=int(row["left_id"]),
        right_id=int(row["right_id"]),
        label=str(row["label"]),
        labeler=str(row["labeler"]),
        created_at=float(row["created_at"]),
        notes=str(row["notes"]),
        label_confidence=float(row["label_confidence"]),
    )
