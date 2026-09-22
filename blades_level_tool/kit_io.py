"""Parse Kit `.asset` files (BGS.Game.Level.KitScriptableObject).

A Kit is a tileset (AyleidDungeon, CaveDungeon, StoneDungeon, ...) and
carries the all-important `_gridSize` (1.828125 world units per grid
cell in every shipped kit we checked) plus the kit's runtime Uid, which
becomes the `_uidParent` of every `_roomPointer` in a FixedLayout.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Kit:
    name: str                       # display name (falls back to file stem)
    asset_path: Path
    guid: str                       # Unity asset guid (from .meta)
    uid: str                        # runtime Uid (_uid._id inside the asset)
    grid_size: float = 1.828125
    snap: bool = True
    location_type: int = 0          # 0 interior, 1 exterior
    max_room_size: int = 64
    doorway_piece_size: int = 4
    small_corridor_piece_size: int = 4
    room_guids: List[str] = field(default_factory=list)  # unused for now


_UID_RE = re.compile(r"_uid:\s*\n\s*_id:\s*([0-9a-fA-F-]{36})")
_META_GUID_RE = re.compile(r"guid:\s*([0-9a-fA-F]{32})")


def load_kit(asset_path: Path) -> Optional[Kit]:
    """Load a single KitScriptableObject .asset file."""
    asset_path = Path(asset_path)
    text = asset_path.read_text(encoding="utf-8", errors="replace")

    if "_gridSize" not in text:
        return None  # not a Kit asset

    uid_m = _UID_RE.search(text)
    uid = uid_m.group(1) if uid_m else ""

    guid = ""
    meta_path = asset_path.with_suffix(asset_path.suffix + ".meta")
    if meta_path.exists():
        gm = _META_GUID_RE.search(meta_path.read_text(encoding="utf-8", errors="replace"))
        if gm:
            guid = gm.group(1)

    name_m = re.search(r"^  m_Name:\s*(.*)$", text, re.M)
    name = name_m.group(1).strip() if name_m else asset_path.stem
    if not name or name == "_":
        name = asset_path.stem

    def _float(pattern: str, default: float) -> float:
        m = re.search(pattern + r":\s*(-?[\d.eE+]+)", text)
        return float(m.group(1)) if m else default

    def _int(pattern: str, default: int) -> int:
        m = re.search(pattern + r":\s*(-?\d+)", text)
        return int(m.group(1)) if m else default

    snap_m = re.search(r"_snap:\s*(\d)", text)
    name_key_m = re.search(r"_name:\s*\n\s*_key:\s*(.*)", text)
    display = name_key_m.group(1).strip() if name_key_m else ""
    if not display or display.startswith("Kit_"):
        # "Kit_Ayleid_Name" -> "Ayleid"
        m = re.match(r"Kit_(\w+?)_Name", display or "")
        if m:
            display = m.group(1)
        else:
            display = asset_path.stem

    return Kit(
        name=display,
        asset_path=asset_path,
        guid=guid,
        uid=uid,
        grid_size=_float(r"_gridSize", 1.828125),
        snap=bool(int(snap_m.group(1))) if snap_m else True,
        location_type=_int(r"_locationType", 0),
        max_room_size=_int(r"_maxRoomSize", 64),
        doorway_piece_size=_int(r"_doorwayPieceSize", 4),
        small_corridor_piece_size=_int(r"_smallCorridorPieceSize", 4),
    )


def load_kits(kits_dir: Path) -> Dict[str, Kit]:
    """Load every Kit asset in a directory. Keyed by guid, which is what
    FixedLayout._kit references."""
    kits: Dict[str, Kit] = {}
    for path in sorted(Path(kits_dir).glob("*.asset")):
        kit = load_kit(path)
        if kit and kit.guid:
            kits[kit.guid] = kit
    return kits
