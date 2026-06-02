"""List, create, load, save and delete species on disk."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .. import paths
from ..rl import spaces as space_builder
from ..sim.morphology import CreatureMorphology
from .species import Species

_METADATA = "metadata.json"
_POLICY = "policy.zip"
_POLICY_NPZ = "policy.npz"


def _safe_name(name: str) -> str:
    """Folder-safe version of a species name (keeps it human-readable)."""
    cleaned = re.sub(r"[^\w\- ]", "", name).strip()
    return cleaned or "unnamed"


def species_path(name: str) -> Path:
    return paths.species_dir() / _safe_name(name)


def metadata_path(name: str) -> Path:
    return species_path(name) / _METADATA


def policy_path(name: str) -> Path:
    return species_path(name) / _POLICY


def policy_npz_path(name: str) -> Path:
    return species_path(name) / _POLICY_NPZ


def exists(name: str) -> bool:
    return metadata_path(name).is_file()


def has_trained_policy(name: str) -> bool:
    return policy_npz_path(name).is_file() or policy_path(name).is_file()


def list_species() -> list[Species]:
    out: list[Species] = []
    base = paths.species_dir()
    for d in sorted(base.iterdir()):
        meta = d / _METADATA
        if meta.is_file():
            try:
                out.append(load(d.name))
            except Exception:
                continue  # skip corrupt entries rather than crash the UI
    return out


def load(name: str) -> Species:
    with open(metadata_path(name), "r", encoding="utf-8") as f:
        data = json.load(f)
    return Species.from_dict(data)


def save(species: Species) -> None:
    species_path(species.name).mkdir(parents=True, exist_ok=True)
    with open(metadata_path(species.name), "w", encoding="utf-8") as f:
        json.dump(species.to_dict(), f, indent=2)


def create(name: str, morphology: CreatureMorphology | None = None,
           role: str = "forager") -> Species:
    """Create a new (untrained) species and persist its metadata."""
    morph = morphology or CreatureMorphology()
    species = Species(
        name=_safe_name(name),
        morphology=morph,
        role=role,
        obs_dim=space_builder.observation_dim(morph),
        act_dim=morph.num_legs,
    )
    save(species)
    return species


def delete(name: str) -> None:
    p = species_path(name)
    if p.is_dir():
        shutil.rmtree(p)
