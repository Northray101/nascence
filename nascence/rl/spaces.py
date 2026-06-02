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


def observation_dim(morph: CreatureMorphology) -> int:
    return 3 * morph.num_legs + 3 + _NOSTRILS + 2 + 3 + 3 + 1


def make_observation_space(morph: CreatureMorphology) -> spaces.Box:
    n = observation_dim(morph)
    return spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)


def make_action_space(morph: CreatureMorphology) -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(morph.num_legs,), dtype=np.float32)


def _world_diag(world: World) -> float:
    return math.hypot(world.width, world.height)


def _rotate_to_body(dx: float, dy: float, angle: float) -> tuple[float, float]:
    a = -angle
    ca, sa = math.cos(a), math.sin(a)
    return dx * ca - dy * sa, dx * sa + dy * ca


def observe(world: World, creature: Creature) -> np.ndarray:
    """Assemble the normalised observation vector for ``creature``."""
    obs: list[float] = []
    cx, cy = creature.position
    ang = creature.angle
    diag = _world_diag(world)

    # Per-leg proprioception.
    for rel_angle, ang_vel, _neutral in creature.leg_states():
        obs.append(math.sin(rel_angle))
        obs.append(math.cos(rel_angle))
        obs.append(float(np.clip(ang_vel / config.LEG_MAX_RATE, -1.0, 1.0)))

    # Body velocity (body frame). Scale by a nominal max speed.
    vx, vy, av = creature.body_velocity_local()
    vscale = 200.0
    obs.append(float(np.clip(vx / vscale, -1.0, 1.0)))
    obs.append(float(np.clip(vy / vscale, -1.0, 1.0)))
    obs.append(float(np.clip(av / 6.0, -1.0, 1.0)))

    # Smell at nostrils around the body.
    for i in range(_NOSTRILS):
        a = ang + (2.0 * math.pi * i) / _NOSTRILS
        nx = cx + math.cos(a) * creature.morph.body_radius * 1.5
        ny = cy + math.sin(a) * creature.morph.body_radius * 1.5
        c = world.chem.sample(nx, ny)
        obs.append(float(np.tanh(c)))

    # Smell gradient in body frame.
    gx, gy = world.chem.gradient(cx, cy)
    gxb, gyb = _rotate_to_body(gx, gy, ang)
    gmag = math.hypot(gxb, gyb) + 1e-9
    obs.append(float(np.tanh(gxb)))
    obs.append(float(np.tanh(gyb)))

    # Nearest food bearing + distance (body frame).
    food = world.nearest_food(cx, cy)
    if food is not None:
        dx, dy = food.x - cx, food.y - cy
        fxb, fyb = _rotate_to_body(dx, dy, ang)
        bearing = math.atan2(fyb, fxb)
        dist = math.hypot(dx, dy)
        obs.append(math.sin(bearing))
        obs.append(math.cos(bearing))
        obs.append(float(np.clip(dist / diag, 0.0, 1.0)))
    else:
        obs.extend([0.0, 0.0, 1.0])

    # Nearest enemy bearing + distance (Phase 1: none -> zeros, far).
    obs.extend([0.0, 0.0, 1.0])

    # Energy.
    obs.append(float(np.clip(creature.energy, 0.0, 1.0)))

    return np.asarray(obs, dtype=np.float32)
