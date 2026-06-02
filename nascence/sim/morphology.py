"""Creature body plans and the Pymunk builder that turns them into physics.

A :class:`CreatureMorphology` is a plain, serialisable description of a body
(how big, how many legs, how stiff). :func:`build_body` turns one into a set of
linked Pymunk bodies and the leg motors the brain will drive.

For the Phase-1 vertical slice the body is a single rigid disc with simple
one-joint legs. The dataclass already carries the fields (e.g. ``firmness``)
that a later soft-body ("jelly") phase will use, so saved species stay
forward-compatible.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pymunk

from .. import config
from .collision import CT_BODY, CT_LEG


@dataclass
class CreatureMorphology:
    """Serialisable description of a creature's body."""

    body_radius: float = config.DEFAULT_BODY_RADIUS
    num_legs: int = config.DEFAULT_NUM_LEGS
    leg_length: float = config.DEFAULT_LEG_LENGTH
    leg_width: float = config.DEFAULT_LEG_WIDTH
    leg_max_rate: float = config.LEG_MAX_RATE
    leg_swing_limit: float = config.LEG_SWING_LIMIT
    # Reserved for the Phase-3 soft jelly body (1 = rigid, lower = jellier).
    firmness: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreatureMorphology":
        # Ignore unknown keys so older/newer save files still load.
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class Leg:
    """One actuated leg: a segment body plus the motor the brain controls."""

    segment: pymunk.Body
    shape: pymunk.Shape
    motor: pymunk.SimpleMotor
    neutral_angle: float  # leg's rest orientation relative to the body


@dataclass
class BuiltBody:
    """Everything :func:`build_body` produced, ready to add to a Space."""

    hub: pymunk.Body
    hub_shape: pymunk.Shape
    legs: list[Leg] = field(default_factory=list)
    constraints: list[pymunk.Constraint] = field(default_factory=list)

    def all_bodies(self) -> list[pymunk.Body]:
        return [self.hub] + [leg.segment for leg in self.legs]

    def all_shapes(self) -> list[pymunk.Shape]:
        return [self.hub_shape] + [leg.shape for leg in self.legs]


def build_body(
    morph: CreatureMorphology,
    position: tuple[float, float],
    angle: float = 0.0,
) -> BuiltBody:
    """Construct the Pymunk bodies/joints for one creature at ``position``.

    The caller is responsible for adding the returned bodies, shapes and
    constraints to a :class:`pymunk.Space`.
    """
    px, py = position

    # --- Hub: a rigid disc that is the creature's core body ---------------
    hub_mass = 1.0
    hub_moment = pymunk.moment_for_circle(hub_mass, 0, morph.body_radius)
    hub = pymunk.Body(hub_mass, hub_moment)
    hub.position = (px, py)
    hub.angle = angle
    hub_shape = pymunk.Circle(hub, morph.body_radius)
    hub_shape.friction = 0.6
    hub_shape.collision_type = CT_BODY

    built = BuiltBody(hub=hub, hub_shape=hub_shape)

    # --- Legs: evenly spaced around the hub -------------------------------
    leg_mass = 0.15
    for i in range(morph.num_legs):
        neutral = (2.0 * math.pi * i) / morph.num_legs
        # Attachment point on the hub rim, in world coordinates.
        anchor_local = (
            math.cos(neutral) * morph.body_radius,
            math.sin(neutral) * morph.body_radius,
        )
        anchor_world = hub.local_to_world(anchor_local)

        # Leg segment extends radially outward from the anchor.
        seg_moment = pymunk.moment_for_box(
            leg_mass, (morph.leg_length, morph.leg_width)
        )
        segment = pymunk.Body(leg_mass, seg_moment)
        outward = (math.cos(neutral + angle), math.sin(neutral + angle))
        seg_center = (
            anchor_world[0] + outward[0] * morph.leg_length * 0.5,
            anchor_world[1] + outward[1] * morph.leg_length * 0.5,
        )
        segment.position = seg_center
        segment.angle = neutral + angle

        shape = pymunk.Poly.create_box(
            segment, (morph.leg_length, morph.leg_width)
        )
        shape.friction = 0.4
        shape.collision_type = CT_LEG

        # Pin the inner end of the leg to the hub rim.
        pivot = pymunk.PivotJoint(hub, segment, anchor_world)
        pivot.collide_bodies = False

        # The motor the brain drives: relative angular velocity hub<->leg.
        motor = pymunk.SimpleMotor(hub, segment, 0.0)
        motor.max_force = 4.0e6  # strong enough to move against the fluid

        # Keep the leg's swing within a comfortable range.
        limit = pymunk.RotaryLimitJoint(
            hub, segment, -morph.leg_swing_limit, morph.leg_swing_limit
        )

        built.legs.append(
            Leg(segment=segment, shape=shape, motor=motor, neutral_angle=neutral)
        )
        built.constraints.extend([pivot, motor, limit])

    return built
