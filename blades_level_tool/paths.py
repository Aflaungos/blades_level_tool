"""Locate the exported game data (AssetRipper ExportedProject)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_assets_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Return the ExportedProject Assets folder containing
    export/gameplaymetadata/RoomsData.asset, or None."""
    candidates: list = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("BLADES_GAME_DATA")
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates += [
        home / "blades_game_data/Joined/ExportedProject/Assets",
        home / "blades_game_data/ExportedProject/Assets",
        home / "blades_game_data/blades_data/ExportedProject/Assets",
        home / "blades_game_data/blades_app_data/ExportedProject/Assets",
    ]
    for c in candidates:
        if (c / "export/gameplaymetadata/RoomsData.asset").exists():
            return c
    return None


def find_guid_cache_dir() -> Path:
    return Path.home() / ".blades_level_tool"
