"""A Gymnasium environment wrapping a single creature in a small world.

Two roles share this env:

* ``forager`` — learns to crawl to food by smell (curriculum: food starts close
  and moves farther away as it improves).
* ``predator`` — learns to chase down a fleeing prey creature.

The same observation builder is reused for live inference, so a trained brain
transfers straight to the sandbox.
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
from .rewards import RewardWeights, forager_reward, predator_reward

_PREY_FLEE_SPEED = 3.2


class CreatureEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        morph: CreatureMorphology | None = None,
        max_steps: int = config.EPISODE_MAX_STEPS,
        role: str = "forager",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.morph = morph or CreatureMorphology()
        self.role = role
        self.max_steps = max_steps
        self.weights = RewardWeights()
        self.energy_decay = 1.0 / float(max_steps)

        self.observation_space = space_builder.make_observation_space(self.morph)
        self.action_space = space_builder.make_action_space(self.morph)

        self._rng = random.Random(seed)
        self.world: World | None = None
        self.creature = None
        self.prey = None
        self._steps = 0
        self._episode = 0
        self._prev_dist = 0.0

    # -- spawning helpers ---------------------------------------------------
    def _spawn_food(self, near: tuple[float, float], radius: float) -> None:
        assert self.world is not None
        cx, cy = near
        margin = 80.0
        for _ in range(20):
            ang = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(radius * 0.5, radius)
            fx, fy = cx + math.cos(ang) * r, cy + math.sin(ang) * r
            if margin < fx < self.world.width - margin and (
                margin < fy < self.world.height - margin
            ):
                self.world.add_food(fx, fy)
                return
        self.world.add_food(self.world.width * 0.5, self.world.height * 0.5)

    def _random_point(self, near, radius):
        assert self.world is not None
        cx, cy = near
        m = 80.0
        for _ in range(20):
            a = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(radius * 0.5, radius)
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            if m < x < self.world.width - m and m < y < self.world.height - m:
                return x, y
        return self.world.width * 0.5, self.world.height * 0.5

    def _curriculum_radius(self) -> float:
        # Ramp the target distance from easy to hard over ~50 episodes.
        t = min(1.0, self._episode / 50.0)
        return 150.0 + t * 250.0

    def _target_dist(self) -> float:
        assert self.world is not None and self.creature is not None
        cx, cy = self.creature.position
        if self.role == "predator":
            prey = self.world.nearest_creature(cx, cy, role="forager",
                                               exclude=self.creature)
            if prey is None:
                return 0.0
            return math.hypot(prey.position[0] - cx, prey.position[1] - cy)
        food = self.world.nearest_food(cx, cy)
        return 0.0 if food is None else math.hypot(food.x - cx, food.y - cy)

    def _move_prey(self) -> None:
        """Scripted prey: flee from the predator with a little wander."""
        if self.prey is None or self.creature is None or self.world is None:
            return
        px, py = self.prey.position
        cx, cy = self.creature.position
        dx, dy = px - cx, py - cy
        d = math.hypot(dx, dy) + 1e-6
        jx, jy = self._rng.uniform(-1, 1), self._rng.uniform(-1, 1)
        nx = px + (dx / d) * _PREY_FLEE_SPEED + jx
        ny = py + (dy / d) * _PREY_FLEE_SPEED + jy
        m = 60.0
        nx = min(max(nx, m), self.world.width - m)
        ny = min(max(ny, m), self.world.height - m)
        self.prey.teleport(nx, ny)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self.world = World()
        cx, cy = self.world.width * 0.5, self.world.height * 0.5
        angle = self._rng.uniform(0, 2 * math.pi)
        self.creature = self.world.add_creature(self.morph, (cx, cy), angle,
                                                role=self.role)
        self.creature.energy = config.START_ENERGY

        radius = self._curriculum_radius()
        if self.role == "predator":
            ppos = self._random_point((cx, cy), radius)
            self.prey = self.world.add_creature(
                CreatureMorphology(), ppos, role="forager")
        else:
            self._spawn_food((cx, cy), radius=radius)
            # Warm the smell field so the gradient is meaningful at step 0.
            for _ in range(20):
                for f in self.world.food:
                    self.world.chem.emit(f.x, f.y, f.smell_strength)
                self.world.chem.step()

        self._steps = 0
        self._episode += 1
        self._prev_dist = self._target_dist()
        return space_builder.observe(self.world, self.creature), {}

    def step(self, action):
        assert self.world is not None and self.creature is not None
        action = np.asarray(action, dtype=np.float32)
        self.creature.apply_action(action)
        if self.role == "predator":
            self._move_prey()

        eats = self.world.step(config.SIM_DT)
        ate = any(c is self.creature for c, _ in eats)
        caught = any(p is self.creature for p, _ in self.world.catches)

        self.creature.energy = max(0.0, self.creature.energy - self.energy_decay)

        if ate:
            self.world.food = [f for f in self.world.food if not f.eaten]
            self._spawn_food(self.creature.position,
                             radius=self._curriculum_radius())
            self._prev_dist = self._target_dist()
        if caught and self.prey is not None:
            self.prey.alive = True  # respawn the prey to keep hunting
            self.prey.teleport(*self._random_point(self.creature.position,
                                                   self._curriculum_radius()))
            self._prev_dist = self._target_dist()

        dist = self._target_dist()
        action_mag = float(np.sum(np.abs(np.clip(action, -1.0, 1.0))))
        if self.role == "predator":
            reward = predator_reward(self._prev_dist, dist, caught, action_mag,
                                     self.weights)
        else:
            reward = forager_reward(self._prev_dist, dist, ate, action_mag,
                                    self.weights)
        self._prev_dist = dist

        self._steps += 1
        terminated = self.creature.energy <= 0.0
        truncated = self._steps >= self.max_steps
        obs = space_builder.observe(self.world, self.creature)
        return obs, float(reward), terminated, truncated, {"ate": ate, "caught": caught}
