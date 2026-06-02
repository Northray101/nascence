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


def load_model(species_name: str) -> Any:
    """Load a trained PPO model. Raises FileNotFoundError if never trained."""
    from stable_baselines3 import PPO

    path = registry.policy_path(species_name)
    if not path.is_file():
        raise FileNotFoundError(f"No trained brain for species '{species_name}'.")
    return PPO.load(str(path.with_suffix("")), device="cpu")
