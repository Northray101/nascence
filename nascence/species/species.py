"""The Species record: a named body plan plus training metadata.

A species lives in its own folder under the data directory:

    <data>/species/<name>/metadata.json   # this record (the "contract")
    <data>/species/<name>/policy.zip      # the trained PPO brain (once trained)

``metadata.json`` is what lets the sandbox rebuild the exact body and sensor
layout a saved brain expects.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..sim.morphology import CreatureMorphology

SCHEMA_VERSION = 1


@dataclass
class TrainingStats:
    timesteps: int = 0
    mean_reward: float = 0.0
    last_trained: float = 0.0  # unix time, 0 = never


@dataclass
class Species:
    name: str
    morphology: CreatureMorphology = field(default_factory=CreatureMorphology)
    role: str = "forager"  # "forager" | "predator"
    obs_dim: int = 0
    act_dim: int = 0
    created: float = field(default_factory=time.time)
    stats: TrainingStats = field(default_factory=TrainingStats)
    schema_version: int = SCHEMA_VERSION

    @property
    def trained(self) -> bool:
        return self.stats.timesteps > 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["morphology"] = self.morphology.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Species":
        morph = CreatureMorphology.from_dict(data.get("morphology", {}))
        stats_data = data.get("stats", {}) or {}
        stats = TrainingStats(
            timesteps=int(stats_data.get("timesteps", 0)),
            mean_reward=float(stats_data.get("mean_reward", 0.0)),
            last_trained=float(stats_data.get("last_trained", 0.0)),
        )
        return cls(
            name=data["name"],
            morphology=morph,
            role=data.get("role", "forager"),
            obs_dim=int(data.get("obs_dim", 0)),
            act_dim=int(data.get("act_dim", 0)),
            created=float(data.get("created", time.time())),
            stats=stats,
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )
