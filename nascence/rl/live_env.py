"""A live, interactive training environment.

Unlike :class:`CreatureEnv` (which builds a fresh private world each episode and
runs as fast as possible), this env trains a creature inside a *shared,
persistent* world that the GUI renders and the user meddles with in real time:

* the world is created once and never torn down on reset (so the creature stays
  put and the sandbox feels continuous);
* every step it drains a :class:`SharedControl` for user commands (place/move
  food, drag the creature, add walls) and manual "treat/scold" reward;
* it throttles itself to the user's chosen speed and obeys pause.

The same lock guards world mutation + stepping so the GUI thread can safely
snapshot positions for drawing.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .. import config
from ..sim.morphology import CreatureMorphology
from ..sim.world import World
from . import spaces as space_builder
from .creature_env import CreatureEnv
from .live_control import Command, SharedControl

# Energy never fully drains in live mode, so the show never stops abruptly.
_ENERGY_FLOOR = 0.05
_LIVE_MAX_STEPS = 2000


class LiveCreatureEnv(CreatureEnv):
    def __init__(
        self,
        world: World,
        ctrl: SharedControl,
        morph: CreatureMorphology | None = None,
    ) -> None:
        super().__init__(morph=morph, max_steps=_LIVE_MAX_STEPS)
        self.world = world
        self.ctrl = ctrl
        cx, cy = world.width * 0.5, world.height * 0.5
        self.creature = world.add_creature(self.morph, (cx, cy))
        self.creature.energy = config.START_ENERGY
        if not world.food:
            self._spawn_food((cx, cy), radius=300.0)
        self._steps = 0
        self._prev_dist = self._dist_to_food()
        self._target_dt = 0.0

    # -- continuous, non-destructive reset ---------------------------------
    def reset(self, *, seed=None, options=None):  # noqa: D401
        # Soft reset: keep the world, creature and walls; just re-energise and
        # make sure there is something to chase.
        if self.creature is not None:
            self.creature.energy = config.START_ENERGY
        if self.world is not None and not self.world.food:
            self._spawn_food(self.creature.position, radius=300.0)
        self._steps = 0
        self._prev_dist = self._dist_to_food()
        obs = space_builder.observe(self.world, self.creature)
        return obs, {}

    # -- user influence -----------------------------------------------------
    def _apply_commands(self) -> None:
        for cmd in self.ctrl.drain_commands():
            if cmd.kind == "food":
                self.world.add_food(cmd.x, cmd.y)
            elif cmd.kind == "clear_food":
                self.world.clear_food()
            elif cmd.kind == "drag":
                self.creature.teleport(cmd.x, cmd.y)
            elif cmd.kind == "reset_pose":
                self.creature.teleport(self.world.width * 0.5,
                                       self.world.height * 0.5)
            elif cmd.kind == "wall":
                self.world.add_wall(cmd.x, cmd.y, cmd.x2, cmd.y2)

    def _await_clock(self) -> None:
        """Block while paused; pace the loop to the chosen speed."""
        while True:
            paused, sps = self.ctrl.snapshot_speed()
            if not paused:
                self._target_dt = (1.0 / sps) if sps > 0 else 0.0
                return
            time.sleep(0.05)

    # -- step ---------------------------------------------------------------
    def step(self, action):
        self._await_clock()
        t0 = time.perf_counter()

        with self.ctrl.lock:
            self._apply_commands()
            action = np.asarray(action, dtype=np.float32)
            self.creature.apply_action(action)
            eats = self.world.step(config.SIM_DT)
            ate = any(c is self.creature for c, _ in eats)

            self.creature.energy = max(
                _ENERGY_FLOOR, self.creature.energy - self.energy_decay
            )
            if ate:
                self.world.food = [f for f in self.world.food if not f.eaten]
                self._spawn_food(self.creature.position, radius=350.0)
                self._prev_dist = self._dist_to_food()

            dist = self._dist_to_food()
            obs = space_builder.observe(self.world, self.creature)

        action_mag = float(np.sum(np.abs(np.clip(action, -1.0, 1.0))))
        from .rewards import forager_reward

        reward = forager_reward(
            prev_dist=self._prev_dist, dist=dist, ate=ate,
            action_magnitude=action_mag, weights=self.weights,
        )
        reward += self.ctrl.take_reward()  # manual treat / scold
        self._prev_dist = dist

        self._steps += 1
        truncated = self._steps >= self.max_steps
        terminated = False  # never "die" during a live show

        # Pace to the requested speed (sleep outside the lock).
        if self._target_dt > 0.0:
            elapsed = time.perf_counter() - t0
            if elapsed < self._target_dt:
                time.sleep(self._target_dt - elapsed)

        return obs, float(reward), terminated, truncated, {"ate": ate}
