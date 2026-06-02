"""Build observation/action spaces from a morphology and assemble observations.

This module is the single source of truth for what a creature "sees" and how
it "acts", shared by both training (CreatureEnv) and live inference (sandbox),
so a saved brain always meets the body it was trained on.

Observation layout (all values roughly normalised to [-1, 1]):
  per leg (N):  sin(rel_angle), cos(rel_angle), ang_vel        -> 3 * N
  body:         vx, vy, angular_velocity                       -> 3
  smell:        concentration at K nostrils                    -> K
  smell:        gradient direction (gx, gy) in body frame      -> 2
  food:         nearest-food bearing (sin, cos), distance      -> 3
  enemy:        nearest-enemy bearing (sin, cos), distance     -> 3  (0 in Phase 1)
  energy:       normalised energy                              -> 1
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from .. import config
from ..sim.creature import Creature
from ..sim.morphology import CreatureMorphology
from ..sim.world import World

_NOSTRILS = config.NUM_NOSTRILS


def _clip(x: float, lo: float, hi: float) -> float:
    """Plain-python scalar clip — far cheaper than np.clip in a tight loop."""
    return lo if x < lo else (hi if x > hi else x)


def _tanh(x: float) -> float:
    """Plain-python tanh (math.tanh is much faster than np.tanh on scalars)."""
    return math.tanh(x)


def observation_dim(morph: CreatureMorphology) -> int:
    return 3 * morph.num_actuators + 3 + _NOSTRILS + 2 + 3 + 3 + 1


def make_observation_space(morph: CreatureMorphology) -> spaces.Box:
    n = observation_dim(morph)
    return spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)


def make_action_space(morph: CreatureMorphology) -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(morph.num_actuators,),
                      dtype=np.float32)


def _world_diag(world: World) -> float:
    return math.hypot(world.width, world.height)


def _rotate_to_body(dx: float, dy: float, angle: float) -> tuple[float, float]:
    a = -angle
    ca, sa = math.cos(a), math.sin(a)
    return dx * ca - dy * sa, dx * sa + dy * ca


def _emit_vision(obs, cx, cy, ang, point) -> None:
    """Append (sin bearing, cos bearing, distance) for a point *only if* it falls
    inside the creature's forward vision cone and within range; otherwise emit
    "nothing seen" (0, 0, 1). Vision is invisible but finite: the creature must
    face roughly toward a target, and be close enough, to perceive it."""
    if point is None:
        obs.extend([0.0, 0.0, 1.0])
        return
    dx, dy = point[0] - cx, point[1] - cy
    dist = math.hypot(dx, dy)
    if dist > config.VISION_RANGE:
        obs.extend([0.0, 0.0, 1.0])
        return
    xb, yb = _rotate_to_body(dx, dy, ang)
    bearing = math.atan2(yb, xb)  # 0 = straight ahead
    if abs(bearing) > config.VISION_HALF_ANGLE:
        obs.extend([0.0, 0.0, 1.0])
        return
    obs.append(math.sin(bearing))
    obs.append(math.cos(bearing))
    obs.append(_clip(dist / config.VISION_RANGE, 0.0, 1.0))


def _target_and_threat(world: World, creature: Creature):
    cx, cy = creature.position
    if creature.role == "predator":
        prey = world.nearest_creature(cx, cy, role="forager", exclude=creature)
        return (prey.position if prey else None), None
    food = world.nearest_food(cx, cy)
    pred = world.nearest_creature(cx, cy, role="predator", exclude=creature)
    return (food.position if food else None), (pred.position if pred else None)


def observe(world: World, creature: Creature) -> np.ndarray:
    """Assemble the normalised observation vector for ``creature``."""
    obs: list[float] = []
    cx, cy = creature.position
    ang = creature.angle
    diag = _world_diag(world)

    rmax = config.LEG_MAX_RATE
    for rel_angle, ang_vel in creature.actuator_states():
        obs.append(math.sin(rel_angle))
        obs.append(math.cos(rel_angle))
        obs.append(_clip(ang_vel / rmax, -1.0, 1.0))

    # Body velocity (body frame). Scale by a nominal max speed.
    vx, vy, av = creature.body_velocity_local()
    obs.append(_clip(vx / 200.0, -1.0, 1.0))
    obs.append(_clip(vy / 200.0, -1.0, 1.0))
    obs.append(_clip(av / 6.0, -1.0, 1.0))

    # Smell at nostrils around the body.
    br = creature.morph.body_radius * 1.5
    chem = world.chem
    for i in range(_NOSTRILS):
        a = ang + (2.0 * math.pi * i) / _NOSTRILS
        nx = cx + math.cos(a) * br
        ny = cy + math.sin(a) * br
        obs.append(_tanh(chem.sample(nx, ny)))

    # Smell gradient in body frame.
    gx, gy = chem.gradient(cx, cy)
    gxb, gyb = _rotate_to_body(gx, gy, ang)
    obs.append(_tanh(gxb))
    obs.append(_tanh(gyb))

    # Target + threat (vision cone — finite range, forward FOV).
    target, threat = _target_and_threat(world, creature)
    _emit_vision(obs, cx, cy, ang, target)
    _emit_vision(obs, cx, cy, ang, threat)

    obs.append(_clip(creature.energy, 0.0, 1.0))
    return np.asarray(obs, dtype=np.float32)
