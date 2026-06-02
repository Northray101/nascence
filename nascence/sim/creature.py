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
    ) -> None:
        self.morph = morph
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

    # -- control ------------------------------------------------------------
    def apply_action(self, action: np.ndarray) -> None:
        """Set each leg motor's target rate from a [-1, 1] action vector."""
        for i, leg in enumerate(self.body.legs):
            a = float(np.clip(action[i], -1.0, 1.0))
            leg.motor.rate = a * self.morph.leg_max_rate

    # -- proprioception -----------------------------------------------------
    def leg_states(self) -> list[tuple[float, float, float]]:
        """Per-leg (relative_angle, sin, cos, angular_velocity) proprioception.

        Returns a list of (rel_angle, ang_vel) but encoded for the observation
        builder; kept simple here and expanded in rl/spaces.py.
        """
        states = []
        hub = self.body.hub
        for leg in self.body.legs:
            rel_angle = leg.segment.angle - hub.angle - leg.neutral_angle
            ang_vel = leg.segment.angular_velocity - hub.angular_velocity
            states.append((rel_angle, ang_vel, leg.neutral_angle))
        return states

    def body_velocity_local(self) -> tuple[float, float, float]:
        """Body-frame (vx, vy, angular_velocity)."""
        hub = self.body.hub
        v = hub.velocity
        a = -hub.angle
        ca, sa = math.cos(a), math.sin(a)
        vx = v.x * ca - v.y * sa
        vy = v.x * sa + v.y * ca
        return (vx, vy, hub.angular_velocity)
