"""Save/load the trained PPO brain for a species.

The brain is stored as Stable-Baselines3's standard ``policy.zip`` next to the
species ``metadata.json``. Heavy imports (torch/SB3) are deferred so the rest
of the app loads quickly and stays testable without them installed.
"""

from __future__ import annotations

from typing import Any

from ..species import registry


def save_model(species_name: str, model: Any) -> None:
    """Persist a trained PPO model for ``species_name``."""
    path = registry.policy_path(species_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path.with_suffix("")))  # SB3 appends .zip


def save_policy(species_name: str, policy: Any) -> None:
    """Persist an evolved numpy ``MLPPolicy`` brain for ``species_name``."""
    path = registry.policy_npz_path(species_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.save_npz(path)


def load_model(species_name: str) -> Any:
    """Load a trained brain. Prefers an evolved numpy brain (``policy.npz``),
    falling back to a PPO ``policy.zip``. Both expose ``predict(obs)`` so the
    Sandbox can drive either. Raises FileNotFoundError if never trained."""
    npz = registry.policy_npz_path(species_name)
    if npz.is_file():
        from .neuro import MLPPolicy

        return MLPPolicy.load_npz(npz)

    path = registry.policy_path(species_name)
    if path.is_file():
        from stable_baselines3 import PPO

        return PPO.load(str(path.with_suffix("")), device="cpu")
    raise FileNotFoundError(f"No trained brain for species '{species_name}'.")
