"""A creature instance: the physics body plus its drivable legs and energy."""

from __future__ import annotations

import math

import numpy as np

from .. import config
from .morphology import BuiltBody, CreatureMorphology, build_body


class Creature:
    def __init__(
        self,
        morph: CreatureMorphology,
        position: tuple[float, float],
        angle: float = 0.0,
        role: str = "forager",
    ) -> None:
        self.morph = morph
        self.role = role  # "forager" | "predator"
        self.body: BuiltBody = build_body(morph, position, angle)
        self.energy = config.START_ENERGY
        self.alive = True

    # -- physics handles ----------------------------------------------------
    @property
    def hub(self):
        return self.body.hub

    @property
    def position(self) -> tuple[float, float]:
        p = self.body.hub.position
        return (p.x, p.y)

    @property
    def angle(self) -> float:
        return self.body.hub.angle

    def _motors(self):
        for leg in self.body.legs:
            for j, motor in enumerate(leg.motors):
                yield leg, j, motor

    # -- placement ----------------------------------------------------------
    def teleport(self, x: float, y: float) -> None:
        """Move the whole creature (hub + legs + jelly ring) keeping its shape."""
        hub = self.body.hub
        dx, dy = x - hub.position.x, y - hub.position.y
        for b in self.body.all_bodies():
            b.position = (b.position.x + dx, b.position.y + dy)
            b.velocity = (0.0, 0.0)
            b.angular_velocity = 0.0

    # -- control ------------------------------------------------------------
    def apply_action(self, action) -> None:
        """Set every joint motor's target rate from a [-1, 1] action vector."""
        rmax = self.morph.leg_max_rate
        n = len(action)
        k = 0
        for _leg, _j, motor in self._motors():
            if k >= n:
                break
            a = float(action[k])
            # Plain-python clip (np.clip on scalars is many times slower).
            if a < -1.0: a = -1.0
            elif a > 1.0: a = 1.0
            motor.rate = a * rmax
            k += 1

    # -- proprioception -----------------------------------------------------
    def actuator_states(self) -> list[tuple[float, float]]:
        """Per-joint (relative_angle, relative_angular_velocity)."""
        states = []
        for leg, j, _motor in self._motors():
            seg = leg.segments[j]
            parent = leg.parents[j]
            neutral = leg.neutral_angle if j == 0 else 0.0
            rel_angle = seg.angle - parent.angle - neutral
            rel_vel = seg.angular_velocity - parent.angular_velocity
            states.append((rel_angle, rel_vel))
        return states

    def body_velocity_local(self) -> tuple[float, float, float]:
        hub = self.body.hub
        v = hub.velocity
        a = -hub.angle
        ca, sa = math.cos(a), math.sin(a)
        vx = v.x * ca - v.y * sa
        vy = v.x * sa + v.y * ca
        return (vx, vy, hub.angular_velocity)
