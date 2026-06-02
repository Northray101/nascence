"""Sanity tests for the simulation and the Gymnasium environment.

These run without torch/stable-baselines3 installed (they only need numpy,
gymnasium and pymunk), so the physics + RL contract can be checked quickly.
"""

from __future__ import annotations

import numpy as np

from nascence.rl import spaces as space_builder
from nascence.rl.creature_env import CreatureEnv
from nascence.sim.morphology import CreatureMorphology
from nascence.sim.world import World


def test_observation_dim_matches_space():
    morph = CreatureMorphology(num_legs=3)
    env = CreatureEnv(morph=morph)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == space_builder.observation_dim(morph)
    assert env.observation_space.contains(obs)


def test_env_checker_passes():
    from gymnasium.utils.env_checker import check_env

    env = CreatureEnv(seed=0)
    check_env(env, skip_render_check=True)


def test_step_contract():
    env = CreatureEnv(seed=1)
    env.reset(seed=1)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break


def test_eating_gives_energy_and_reward():
    morph = CreatureMorphology()
    world = World()
    cx, cy = world.width / 2, world.height / 2
    creature = world.add_creature(morph, (cx, cy))
    creature.energy = 0.1
    world.add_food(cx + 5, cy)  # within eat radius
    eats = world.step()
    assert len(eats) == 1
    assert creature.energy > 0.1


def test_morphology_roundtrip():
    morph = CreatureMorphology(num_legs=5, body_radius=30.0)
    restored = CreatureMorphology.from_dict(morph.to_dict())
    assert restored == morph
