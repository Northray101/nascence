"""Filesystem locations for saved species.

On macOS we keep saved brains in the standard per-user application-support
directory so they survive app updates and never clutter the project folder.
"""

from __future__ import annotations

import sys
from pathlib import Path


def data_dir() -> Path:
    """Return (and create) the base directory for nascence's saved data."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "nascence"
    elif sys.platform.startswith("win"):
        base = Path.home() / "AppData" / "Roaming" / "nascence"
    else:  # Linux and others
        base = Path.home() / ".local" / "share" / "nascence"
    base.mkdir(parents=True, exist_ok=True)
    return base


def species_dir() -> Path:
    """Directory that holds one sub-folder per saved species."""
    d = data_dir() / "species"
    d.mkdir(parents=True, exist_ok=True)
    return d
