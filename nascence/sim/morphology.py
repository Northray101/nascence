"""Creature body plans and the Pymunk builder that turns them into physics.

A :class:`CreatureMorphology` is a plain, serialisable description of a body.
:func:`build_body` turns one into linked Pymunk bodies plus the leg motors the
brain drives. Supported variations (all forward-compatible via metadata):

* ``body_type``  — ``"rigid"`` disc, or ``"jelly"`` (a springy ring of nodes
  around the hub that wobbles; the hub stays as the control/sensor anchor).
* ``leg_segments`` — 1 (simple leg, one motor) or 2 (hip + knee, two motors).
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
    leg_segments: int = 1          # 1 = simple, 2 = hip+knee
    body_type: str = "rigid"       # "rigid" | "jelly"
    ring_nodes: int = 10           # jelly only
    firmness: float = 1.0          # jelly only: 1 = firm, lower = wobblier

    @property
    def num_actuators(self) -> int:
        return self.num_legs * self.leg_segments

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreatureMorphology":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class Leg:
    """One leg: 1-2 segments and the motors (one per joint) the brain drives."""

    segments: list[pymunk.Body]
    shapes: list[pymunk.Shape]
    motors: list[pymunk.SimpleMotor]
    parents: list[pymunk.Body]  # the body each segment/motor pivots against
    neutral_angle: float


@dataclass
class BuiltBody:
    hub: pymunk.Body
    hub_shape: pymunk.Shape
    legs: list[Leg] = field(default_factory=list)
    constraints: list[pymunk.Constraint] = field(default_factory=list)
    ring_nodes: list[pymunk.Body] = field(default_factory=list)
    ring_shapes: list[pymunk.Shape] = field(default_factory=list)

    def all_bodies(self) -> list[pymunk.Body]:
        bodies = [self.hub]
        for leg in self.legs:
            bodies.extend(leg.segments)
        bodies.extend(self.ring_nodes)
        return bodies

    def all_shapes(self) -> list[pymunk.Shape]:
        shapes = [self.hub_shape]
        for leg in self.legs:
            shapes.extend(leg.shapes)
        shapes.extend(self.ring_shapes)
        return shapes


def _make_segment(mass, length, width, center, angle, collision_type):
    moment = pymunk.moment_for_box(mass, (length, width))
    body = pymunk.Body(mass, moment)
    body.position = center
    body.angle = angle
    shape = pymunk.Poly.create_box(body, (length, width))
    shape.friction = 0.4
    shape.collision_type = collision_type
    return body, shape


def build_body(
    morph: CreatureMorphology,
    position: tuple[float, float],
    angle: float = 0.0,
) -> BuiltBody:
    px, py = position

    hub_mass = 1.0
    hub_moment = pymunk.moment_for_circle(hub_mass, 0, morph.body_radius)
    hub = pymunk.Body(hub_mass, hub_moment)
    hub.position = (px, py)
    hub.angle = angle
    hub_shape = pymunk.Circle(hub, morph.body_radius)
    hub_shape.friction = 0.6
    hub_shape.collision_type = CT_BODY

    built = BuiltBody(hub=hub, hub_shape=hub_shape)

    # --- Optional jelly ring -------------------------------------------
    if morph.body_type == "jelly":
        _build_jelly_ring(built, morph, hub, angle)

    # --- Legs ----------------------------------------------------------
    leg_mass = 0.15
    seg2_scale = 0.8  # the knee segment is a bit shorter
    for i in range(morph.num_legs):
        neutral = (2.0 * math.pi * i) / morph.num_legs
        anchor_world = hub.local_to_world(
            (math.cos(neutral) * morph.body_radius,
             math.sin(neutral) * morph.body_radius)
        )
        outward = (math.cos(neutral + angle), math.sin(neutral + angle))

        segments, shapes, motors, parents = [], [], [], []
        parent = hub
        attach = anchor_world
        seg_len = morph.leg_length
        for s in range(morph.leg_segments):
            center = (attach[0] + outward[0] * seg_len * 0.5,
                      attach[1] + outward[1] * seg_len * 0.5)
            body, shape = _make_segment(
                leg_mass, seg_len, morph.leg_width, center,
                neutral + angle, CT_LEG)
            pivot = pymunk.PivotJoint(parent, body, attach)
            pivot.collide_bodies = False
            motor = pymunk.SimpleMotor(parent, body, 0.0)
            motor.max_force = 4.0e6
            limit = pymunk.RotaryLimitJoint(
                parent, body, -morph.leg_swing_limit, morph.leg_swing_limit)

            segments.append(body)
            shapes.append(shape)
            motors.append(motor)
            parents.append(parent)
            built.constraints.extend([pivot, motor, limit])

            # Next segment attaches at the far end of this one.
            attach = (attach[0] + outward[0] * seg_len,
                      attach[1] + outward[1] * seg_len)
            parent = body
            seg_len *= seg2_scale

        built.legs.append(Leg(segments=segments, shapes=shapes, motors=motors,
                              parents=parents, neutral_angle=neutral))

    return built


def _build_jelly_ring(built, morph, hub, angle) -> None:
    """A ring of light nodes spring-linked to neighbours and the hub."""
    n = max(6, morph.ring_nodes)
    radius = morph.body_radius * 1.35
    node_mass = 0.05
    # firmness 1 -> stiff; lower -> wobblier.
    stiffness = 400.0 * max(0.1, morph.firmness)
    damping = 12.0
    nodes = []
    for i in range(n):
        a = angle + (2 * math.pi * i) / n
        pos = hub.local_to_world((math.cos(a) * radius, math.sin(a) * radius))
        moment = pymunk.moment_for_circle(node_mass, 0, 4)
        body = pymunk.Body(node_mass, moment)
        body.position = pos
        shape = pymunk.Circle(body, 4)
        shape.collision_type = CT_BODY
        shape.sensor = True  # decorative; don't collide with legs/food
        nodes.append(body)
        built.ring_nodes.append(body)
        built.ring_shapes.append(shape)
        # Radial spring to the hub (keeps the blob's volume).
        built.constraints.append(
            pymunk.DampedSpring(hub, body, (0, 0), (0, 0), radius,
                                stiffness, damping))
    # Springs between neighbours (perimeter elasticity).
    rest = 2 * radius * math.sin(math.pi / n)
    for i in range(n):
        built.constraints.append(
            pymunk.DampedSpring(nodes[i], nodes[(i + 1) % n], (0, 0), (0, 0),
                                rest, stiffness * 0.5, damping))
