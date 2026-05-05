"""Adaptive KL coefficient (Schulman et al., PPO paper appendix A.2).

The PPO + RLHF setup multiplies the policy reward by ``-c * KL(pi || pi_ref)``
to keep the policy from drifting too far from a reference policy (the BC prior
in Phase 6, the BC + Phase 9 frozen snapshot in Phase 9). ``c`` is updated each
batch based on the *observed* KL: if KL is too high, raise c; if too low, lower it.

Constants taken from PPO paper section 4 and the OpenAI baselines defaults,
with tighter bounds for our use case where KL drift past ~0.05 starts hurting
the BC prior in our small-vocab structured action space.
"""

from __future__ import annotations


class AdaptiveKLController:
    """Schulman adaptive-beta. Multiplicative update with a target KL window."""

    __slots__ = ("coef", "target_kl", "scale_up", "scale_down", "min_coef", "max_coef")

    def __init__(
        self,
        init_coef: float = 0.2,
        target_kl: float = 0.02,
        scale_up: float = 1.5,
        scale_down: float = 1.5,
        min_coef: float = 1e-4,
        max_coef: float = 10.0,
    ) -> None:
        if init_coef <= 0 or target_kl <= 0:
            raise ValueError("init_coef and target_kl must be positive")
        if scale_up <= 1 or scale_down <= 1:
            raise ValueError("scale_up and scale_down must be > 1 (they are multipliers)")
        if not (min_coef <= init_coef <= max_coef):
            raise ValueError("init_coef must lie inside [min_coef, max_coef]")
        self.coef = init_coef
        self.target_kl = target_kl
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_coef = min_coef
        self.max_coef = max_coef

    def update(self, observed_kl: float) -> float:
        """Update ``self.coef`` based on the most recent measured KL.

        Returns the new coefficient. Schulman's rule:
        - if kl > 1.5 * target: coef *= scale_up
        - if kl < target / 1.5: coef /= scale_down
        - otherwise unchanged
        Coef clamped to ``[min_coef, max_coef]``.
        """
        if observed_kl > 1.5 * self.target_kl:
            self.coef *= self.scale_up
        elif observed_kl < self.target_kl / 1.5:
            self.coef /= self.scale_down
        self.coef = max(self.min_coef, min(self.max_coef, self.coef))
        return self.coef

    def state_dict(self) -> dict:
        return {
            "coef": self.coef,
            "target_kl": self.target_kl,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "min_coef": self.min_coef,
            "max_coef": self.max_coef,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.coef = float(sd["coef"])
        self.target_kl = float(sd["target_kl"])
        self.scale_up = float(sd["scale_up"])
        self.scale_down = float(sd["scale_down"])
        self.min_coef = float(sd["min_coef"])
        self.max_coef = float(sd["max_coef"])
