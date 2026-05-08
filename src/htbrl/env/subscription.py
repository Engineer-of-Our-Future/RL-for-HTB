"""HTB subscription-tier awareness for the box pool selector.

HTB has two relevant tiers for the agent:

  - **free**: Starting Point boxes + the small rotating set of Active
    Machines that HTB exposes to free accounts.
  - **vip / vip+**: every retired machine + every active machine.

The agent's training pool and eval suite are configured in YAML
(``configs/env/*.yaml``, ``configs/eval/*.yaml``) with a per-box
``vip_only: bool`` field. At rollout time, the pool loader filters
out boxes the operator can't actually reach so we don't waste
SSH connect attempts on unreachable targets.

Three subscription resolution modes:

  - ``"free"``: hard-pin to free-only filtering. The operator's
    currently on a free account, or wants to dogfood free-only
    behavior even on a VIP account (useful for collecting demos
    that a future free user could reproduce).
  - ``"vip"``: no filtering; every box in the pool is allowed.
  - ``"auto"``: probe HTB's API with an API token to determine the
    actual subscription tier. Falls back to ``"free"`` if the token
    is missing or the API rejects it (safer default).

This module is intentionally small + pure -- it doesn't reach out to
HTB at all unless ``"auto"`` is requested AND a token is provided.
The pool YAMLs are the source of truth for which boxes are free vs
VIP; this code's job is just to apply that label correctly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

if TYPE_CHECKING:  # pragma: no cover
    pass


SubscriptionTier = Literal["free", "vip", "auto"]


@dataclass(frozen=True)
class SubscriptionInfo:
    """Resolved subscription state."""

    # The actual tier the agent should treat the operator as having.
    # Only "free" and "vip" appear here -- "auto" is collapsed to one
    # of these by ``resolve``.
    tier: Literal["free", "vip"]
    # How we landed on that tier (for logs / debugging).
    source: str = "explicit"

    @property
    def has_vip(self) -> bool:
        return self.tier == "vip"


# Known-free Starting Point box names. These never need a VIP check;
# they're free to all HTB accounts. Used by ``filter_boxes_by_tier``
# as a fast path so the operator can rely on Starting Point demos
# never being silently dropped.
_KNOWN_FREE_BOX_PREFIXES = (
    "htb-starting-point:",
)


def resolve(tier: SubscriptionTier, *, api_token: str | None = None) -> SubscriptionInfo:
    """Decide which subscription state to apply.

    ``"free"`` and ``"vip"`` are returned verbatim. ``"auto"`` calls
    HTB's profile API with the supplied token; on any failure it
    falls back to ``"free"`` to be safe (fewer boxes attempted, no
    silent over-reach).
    """
    if tier == "free":
        return SubscriptionInfo(tier="free", source="explicit:free")
    if tier == "vip":
        return SubscriptionInfo(tier="vip", source="explicit:vip")
    if tier == "auto":
        if not api_token:
            api_token = os.environ.get("HTB_API_TOKEN")
        if not api_token:
            return SubscriptionInfo(
                tier="free", source="auto:no-token-fallback-to-free",
            )
        try:
            return _probe_htb_subscription(api_token)
        except Exception:
            # Network glitches, API rate-limits, schema changes -- all
            # land here. Defaulting to "free" keeps the agent within
            # what's certainly reachable.
            return SubscriptionInfo(
                tier="free", source="auto:probe-failed-fallback-to-free",
            )
    raise ValueError(f"unknown subscription tier: {tier!r}")


def _probe_htb_subscription(api_token: str) -> SubscriptionInfo:
    """Hit HTB's /api/v4/profile to determine the user's tier.

    Endpoint shape (paraphrased; HTB rotates the schema occasionally):

        GET https://www.hackthebox.com/api/v4/user/profile/basic
        Authorization: Bearer <token>
        ->
        { "info": { "subscription": "vip"|"vip-plus"|"basic"|null, ... } }

    A "basic" / null subscription is "free". Everything else is "vip"
    for our purposes (we don't distinguish vip vs vip+ -- both have
    full machine access).
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "https://www.hackthebox.com/api/v4/user/profile/basic",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
            "User-Agent": "htbrl/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=8) as resp:  # nosec - operator-supplied
        if resp.status != 200:
            raise RuntimeError(f"HTB profile API returned {resp.status}")
        body = json.loads(resp.read().decode("utf-8"))
    sub = ((body or {}).get("info") or {}).get("subscription")
    sub_str = (sub or "").lower()
    is_vip = sub_str.startswith("vip") or sub_str == "premium"
    return SubscriptionInfo(
        tier="vip" if is_vip else "free",
        source=f"auto:probe-ok:subscription={sub!r}",
    )


def is_box_free(box: dict) -> bool:
    """True if the box should be considered reachable on a free account.

    Reads the YAML's ``vip_only`` flag (default False for backward
    compat -- old configs without the flag are treated as free, which
    matches the original Starting Point intent).

    Also honors ``_KNOWN_FREE_BOX_PREFIXES``: a Starting Point box
    that's mistakenly tagged ``vip_only: true`` in a config is still
    considered free, since HTB makes those free for all accounts.
    """
    box_id = box.get("id", "")
    if isinstance(box_id, str):
        for prefix in _KNOWN_FREE_BOX_PREFIXES:
            if box_id.startswith(prefix):
                return True
    return not bool(box.get("vip_only", False))


def filter_boxes_by_tier(
    boxes: Iterable[dict],
    info: SubscriptionInfo,
) -> list[dict]:
    """Return the subset of ``boxes`` reachable by the given subscription.

    VIP gets everything; free gets ``is_box_free`` matches. Order is
    preserved so round-robin selection over the filtered result still
    matches the YAML's stated order.
    """
    boxes_list = list(boxes)
    if info.has_vip:
        return boxes_list
    return [b for b in boxes_list if is_box_free(b)]


__all__ = [
    "SubscriptionInfo",
    "SubscriptionTier",
    "filter_boxes_by_tier",
    "is_box_free",
    "resolve",
]
