"""A Gymnasium environment wrapping a single creature in a small world.

This is the heart of training: PPO drives the leg motors through ``step`` and
learns to crawl toward food by smell. The same observation builder is reused
later for live inference, so a trained brain transfers straight to the sandbox.
"""

from __future__ import annotations

import math
import random
from typing import Any

import gymnasium as gym
import numpy as np

from .. import config
from ..sim.morphology import CreatureMorphology
from ..sim.world import World
from . import spaces as space_builder
from .rewards import RewardWeights, forager_reward


class CreatureEnv(gym.Env):
    """One creature, one food source, continuous foraging by smell."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        morph: CreatureMorphology | None = None,
        max_steps: int = config.EPISODE_MAX_STEPS,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.morph = morph or CreatureMorphology()
        self.max_steps = max_steps
        self.weights = RewardWeights()
        # Metabolism: energy lost per step so the creature must keep eating.
        self.energy_decay = 1.0 / float(max_steps)

        self.observation_space = space_builder.make_observation_space(self.morph)
        self.action_space = space_builder.make_action_space(self.morph)

        self._rng = random.Random(seed)
        self.world: World | None = None
        self.creature = None
        self._steps = 0
        self._prev_dist = 0.0

    # -- helpers ------------------------------------------------------------
    def _spawn_food(self, near: tuple[float, float], radius: float) -> None:
        assert self.world is not None
        cx, cy = near
        margin = 80.0
        for _ in range(20):
            ang = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(radius * 0.5, radius)
            fx = cx + math.cos(ang) * r
            fy = cy + math.sin(ang) * r
            if margin < fx < self.world.width - margin and (
                margin < fy < self.world.height - margin
            ):
                self.world.add_food(fx, fy)
                return
        # Fallback: centre-ish.
        self.world.add_food(self.world.width * 0.5, self.world.height * 0.5)

    def _dist_to_food(self) -> float:
        assert self.world is not None and self.creature is not None
        cx, cy = self.creature.position
        food = self.world.nearest_food(cx, cy)
        if food is None:
            return 0.0
        return math.hypot(food.x - cx, food.y - cy)

    # -- gym API ------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self.world = World()
        cx = self.world.width * 0.5
        cy = self.world.height * 0.5
        angle = self._rng.uniform(0, 2 * math.pi)
        self.creature = self.world.add_creature(self.morph, (cx, cy), angle)
        self.creature.energy = config.START_ENERGY

        self._spawn_food((cx, cy), radius=300.0)
        # Pre-warm the smell field so the gradient is meaningful at step 0.
        for _ in range(20):
            for f in self.world.food:
                self.world.chem.emit(f.x, f.y, f.smell_strength)
            self.world.chem.step()

        self._steps = 0
        self._prev_dist = self._dist_to_food()
        obs = space_builder.observe(self.world, self.creature)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.world is not None and self.creature is not None
        action = np.asarray(action, dtype=np.float32)
        self.creature.apply_action(action)

        eats = self.world.step(config.SIM_DT)
        ate = any(c is self.creature for c, _ in eats)

        # Metabolism.
        self.creature.energy = max(0.0, self.creature.energy - self.energy_decay)

        if ate:
            # Remove eaten food and drop a fresh one to keep foraging going.
            self.world.food = [f for f in self.world.food if not f.eaten]
            self._spawn_food(self.creature.position, radius=350.0)
            dist = self._dist_to_food()
            self._prev_dist = dist  # no spurious progress term on the eat step

        dist = self._dist_to_food()
        action_mag = float(np.sum(np.abs(np.clip(action, -1.0, 1.0))))
        reward = forager_reward(
            prev_dist=self._prev_dist,
            dist=dist,
            ate=ate,
            action_magnitude=action_mag,
            weights=self.weights,
        )
        self._prev_dist = dist

        self._steps += 1
        terminated = self.creature.energy <= 0.0
        truncated = self._steps >= self.max_steps

        obs = space_builder.observe(self.world, self.creature)
        info = {"ate": ate, "energy": self.creature.energy}
        return obs, float(reward), terminated, truncated, info
