"""Pure reward-shaping functions (no physics state mutation).

Keeping these pure makes the reward easy to reason about and unit-test, and
lets enemy ("predator") roles reuse the same building blocks with a flipped
sign on the target term.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    progress: float = 0.1      # per unit of world-distance closed toward target
    eat: float = 10.0          # bonus on eating a food pellet
    energy_cost: float = 0.02  # penalty per unit of total motor command
    time_cost: float = 0.005   # tiny per-step penalty to encourage promptness


def forager_reward(
    prev_dist: float,
    dist: float,
    ate: bool,
    action_magnitude: float,
    weights: RewardWeights,
) -> float:
    """Reward for a food-seeking creature.

    ``prev_dist``/``dist`` are distances to the nearest food before/after the
    step; ``action_magnitude`` is sum(|action|).
    """
    r = 0.0
    r += weights.progress * (prev_dist - dist)
    if ate:
        r += weights.eat
    r -= weights.energy_cost * action_magnitude
    r -= weights.time_cost
    return r
