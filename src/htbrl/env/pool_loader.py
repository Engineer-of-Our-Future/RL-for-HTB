"""Box-pool YAML loader with subscription filtering.

Used by ``scripts/train_ppo.py`` (training pool) and ``scripts/eval.py``
(held-out suite) to read a YAML config and apply the operator's
subscription filter so unreachable VIP-only boxes are dropped before
the rollout loop ever sees them.

Two responsibilities:

1. **Parse** the YAML (top-level metadata + ``boxes`` list).
2. **Filter** by the resolved subscription tier so the agent doesn't
   waste connect attempts on boxes it can't reach.

The pool format mirrors what's in ``configs/env/htb_starting_pool.yaml``
+ ``configs/env/htb_machines_pool.yaml`` + ``configs/eval/holdout_v1.yaml``:

    name: <str>
    matrix: enterprise|mobile|ics
    selection_policy: round_robin|random|...
    max_steps: <int>
    wallclock_cap_seconds: <int>
    boxes:
      - id: <str>
        difficulty: trivial|easy|medium|hard|insane
        vip_only: <bool>
        ... (free-form per box)

The loader doesn't validate every field -- it surfaces the dict
unchanged so domain-specific code (e.g. eval thresholds) can read
its own keys directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from htbrl.env.subscription import (
    SubscriptionInfo,
    SubscriptionTier,
    filter_boxes_by_tier,
    resolve,
)


@dataclass
class BoxPool:
    """Parsed + filtered box-pool config."""

    name: str
    matrix: str
    boxes: list[dict[str, Any]]            # filtered boxes (post-subscription)
    raw: dict[str, Any] = field(default_factory=dict)
    # The full unfiltered list -- handy for telemetry ("dropped 3 of 8
    # boxes due to free-tier filter") so the operator sees what got
    # gated out.
    all_boxes: list[dict[str, Any]] = field(default_factory=list)
    subscription: SubscriptionInfo | None = None

    @property
    def n_filtered_out(self) -> int:
        return len(self.all_boxes) - len(self.boxes)

    @property
    def empty(self) -> bool:
        return not self.boxes


def load_pool(
    yaml_path: str | Path,
    *,
    subscription: SubscriptionTier = "free",
    api_token: str | None = None,
) -> BoxPool:
    """Read ``yaml_path`` and return the post-subscription-filter pool.

    ``subscription``:
      - ``"free"`` (default, safe): only free-tier boxes pass.
      - ``"vip"``: every box passes.
      - ``"auto"``: probes HTB's API for the live tier; falls back to
        ``"free"`` if no token / probe fails.

    Raises FileNotFoundError if the YAML doesn't exist; ValueError if
    none of the recognised list keys (``boxes``, ``sherlocks``,
    ``challenges``) is present.

    Different pool types use different top-level list keys to keep
    YAMLs readable:
      - Machines / Starting Point: ``boxes:``
      - Sherlocks (DFIR): ``sherlocks:``
      - Challenges (CTF): ``challenges:``
    The loader accepts any of these and treats them uniformly. Use
    the ``raw`` dict on the returned pool to read pool-type-specific
    metadata (e.g. ``target_type`` to choose env class).
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"pool YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    list_keys = ("boxes", "sherlocks", "challenges")
    found_key = next((k for k in list_keys if isinstance(raw.get(k), list)), None)
    if found_key is None:
        raise ValueError(
            f"pool YAML missing top-level list (expected one of "
            f"{list_keys}): {path}"
        )

    info = resolve(subscription, api_token=api_token)
    all_boxes = list(raw[found_key])
    filtered = filter_boxes_by_tier(all_boxes, info)
    return BoxPool(
        name=raw.get("name", path.stem),
        matrix=raw.get("matrix", "enterprise"),
        boxes=filtered,
        raw=raw,
        all_boxes=all_boxes,
        subscription=info,
    )


__all__ = [
    "BoxPool",
    "load_pool",
]
