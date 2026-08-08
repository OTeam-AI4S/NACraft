"""Resolve model assets without binding NACraft to one filesystem layout."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_boltz_asset(name: str) -> Path:
    """Return a required Boltz asset from a configured or standard cache."""
    module_root = Path(__file__).resolve().parents[1]
    candidates = []
    configured_cache = os.environ.get("NACRAFT_BOLTZ_CACHE")
    if configured_cache:
        candidates.append(Path(configured_cache).expanduser() / name)
    candidates.extend(
        [
            module_root / "boltz" / name,
            Path.home() / ".boltz" / name,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Boltz asset {name!r} was not found. Checked: {checked}. "
        "Download the Boltz-1 assets or set NACRAFT_BOLTZ_CACHE."
    )
