"""Tests for the box-pool YAML loader + subscription filter integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from htbrl.env.pool_loader import BoxPool, load_pool


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


# ---- happy path -----------------------------------------------------------


def test_load_pool_starting_point_is_unaffected_by_free_filter(tmp_path):
    """Starting Point boxes are tagged ``vip_only: false`` in the
    bundled YAMLs and survive the free-tier filter."""
    pool_path = _write(tmp_path, "sp.yaml", {
        "name": "sp",
        "matrix": "enterprise",
        "boxes": [
            {"id": "htb-starting-point:meow", "vip_only": False},
            {"id": "htb-starting-point:fawn", "vip_only": False},
        ],
    })
    pool = load_pool(pool_path, subscription="free")
    assert pool.name == "sp"
    assert pool.matrix == "enterprise"
    assert [b["id"] for b in pool.boxes] == [
        "htb-starting-point:meow",
        "htb-starting-point:fawn",
    ]
    assert pool.n_filtered_out == 0
    assert pool.subscription is not None
    assert pool.subscription.tier == "free"


def test_load_pool_filters_vip_only_in_free_mode(tmp_path):
    pool_path = _write(tmp_path, "mixed.yaml", {
        "name": "mixed",
        "boxes": [
            {"id": "htb-machines:active-1", "vip_only": False},
            {"id": "htb-machines:retired-1", "vip_only": True},
            {"id": "htb-machines:retired-2", "vip_only": True},
        ],
    })
    pool = load_pool(pool_path, subscription="free")
    assert [b["id"] for b in pool.boxes] == ["htb-machines:active-1"]
    assert pool.n_filtered_out == 2
    assert len(pool.all_boxes) == 3


def test_load_pool_vip_keeps_everything(tmp_path):
    pool_path = _write(tmp_path, "mixed.yaml", {
        "name": "mixed",
        "boxes": [
            {"id": "free-1", "vip_only": False},
            {"id": "vip-1", "vip_only": True},
            {"id": "vip-2", "vip_only": True},
        ],
    })
    pool = load_pool(pool_path, subscription="vip")
    assert [b["id"] for b in pool.boxes] == ["free-1", "vip-1", "vip-2"]
    assert pool.n_filtered_out == 0


def test_load_pool_preserves_box_ordering(tmp_path):
    """Round-robin selection over the filtered list should match the
    YAML's stated order. The loader must NOT re-sort."""
    pool_path = _write(tmp_path, "ordered.yaml", {
        "name": "ordered",
        "boxes": [
            {"id": "z-free", "vip_only": False},
            {"id": "m-vip", "vip_only": True},
            {"id": "a-free", "vip_only": False},
        ],
    })
    pool = load_pool(pool_path, subscription="free")
    assert [b["id"] for b in pool.boxes] == ["z-free", "a-free"]


def test_load_pool_passes_through_extra_box_metadata(tmp_path):
    """Loader doesn't validate per-box fields beyond ``id``+``vip_only``;
    arbitrary metadata (difficulty, expected_techniques, notes) should
    pass through verbatim."""
    pool_path = _write(tmp_path, "rich.yaml", {
        "name": "rich",
        "boxes": [{
            "id": "htb-starting-point:meow",
            "vip_only": False,
            "difficulty": "trivial",
            "expected_techniques": ["T1190", "T1078"],
            "notes": "default creds",
        }],
    })
    pool = load_pool(pool_path)
    box = pool.boxes[0]
    assert box["difficulty"] == "trivial"
    assert box["expected_techniques"] == ["T1190", "T1078"]
    assert box["notes"] == "default creds"


# ---- failure modes --------------------------------------------------------


def test_load_pool_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pool(tmp_path / "doesnotexist.yaml")


def test_load_pool_missing_boxes_key_raises_valueerror(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nmatrix: enterprise\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing top-level"):
        load_pool(p)


def test_load_pool_boxes_not_a_list_raises_valueerror(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nboxes:\n  meow: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing top-level"):
        load_pool(p)


# ---- empty pool sanity ----------------------------------------------------


def test_load_pool_all_filtered_out_yields_empty_pool(tmp_path):
    pool_path = _write(tmp_path, "all_vip.yaml", {
        "name": "all-vip",
        "boxes": [
            {"id": "vip-1", "vip_only": True},
            {"id": "vip-2", "vip_only": True},
        ],
    })
    pool = load_pool(pool_path, subscription="free")
    assert pool.empty
    assert pool.n_filtered_out == 2


# ---- bundled configs ------------------------------------------------------
# These tests double as schema-pinning for the actual YAMLs we ship.


def test_bundled_starting_pool_loads_clean_under_free(repo_root):
    """The repo-bundled Starting Point pool YAML must parse + survive
    the free-tier filter."""
    p = repo_root / "configs" / "env" / "htb_starting_pool.yaml"
    pool = load_pool(p, subscription="free")
    assert not pool.empty
    # Every box in Starting Point is free.
    assert all("htb-starting-point:" in b["id"] for b in pool.boxes)
    assert pool.n_filtered_out == 0


def test_bundled_machines_pool_filters_to_just_free_under_free(repo_root):
    """The bundled Machines pool has both free + vip_only entries; under
    free-tier we should keep only the free ones (currently the active
    rotation), not the retired-VIP-only ones."""
    p = repo_root / "configs" / "env" / "htb_machines_pool.yaml"
    pool = load_pool(p, subscription="free")
    # Should drop at least some boxes (the retired-VIP-only entries).
    assert pool.n_filtered_out > 0
    # No remaining box should be vip_only.
    for box in pool.boxes:
        # Use the same logic the filter uses — Starting Point boxes
        # are always free regardless of the flag, others must have
        # ``vip_only: False``.
        if not box.get("id", "").startswith("htb-starting-point:"):
            assert box.get("vip_only", False) is False


def test_bundled_holdout_eval_loads_clean(repo_root):
    p = repo_root / "configs" / "eval" / "holdout_v1.yaml"
    pool = load_pool(p, subscription="free")
    assert not pool.empty
    # Held-out boxes must NOT overlap with the training Starting Point
    # pool. Cheap check: id sets are disjoint.
    sp = load_pool(repo_root / "configs" / "env" / "htb_starting_pool.yaml",
                   subscription="free")
    sp_ids = {b["id"] for b in sp.boxes}
    holdout_ids = {b["id"] for b in pool.boxes}
    overlap = sp_ids & holdout_ids
    assert not overlap, (
        f"holdout suite overlaps training pool: {sorted(overlap)}. "
        "Held-out boxes must NEVER appear in training; either rename "
        "them in configs/eval/holdout_v1.yaml or drop the duplicates "
        "from configs/env/htb_starting_pool.yaml."
    )


# ---- fixture --------------------------------------------------------------


@pytest.fixture
def repo_root() -> Path:
    """Resolve the repo root from this test file's location."""
    return Path(__file__).resolve().parents[2]
