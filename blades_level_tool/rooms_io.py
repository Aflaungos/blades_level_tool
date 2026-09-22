"""Parse RoomsData.asset (Assets/export/gameplaymetadata/RoomsData.asset).

RoomsData is the master catalog of every placeable room in the game:
name, runtime Uid, kit reference, grid size/offset, and the joint list
(uid, type, grid position, direction). The joint UIDs match the
RoomJoint components inside each room prefab under
Assets/export/resources/<room_name_without _Room suffix>/.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .models import Joint, Room, Vec3

_KIT_RE = re.compile(r"_kit:\s*\{fileID:\s*(-?\d+),\s*guid:\s*([0-9a-fA-F]*),\s*type:\s*\d+\}")
_ID_RE = re.compile(r"_id:\s*([0-9a-fA-F-]{36})")


def _parse_int_block(chunk: str, field: str) -> Optional[dict]:
    """Parse an `x: y: z:` int block for `field` if present."""
    m = re.search(
        re.escape(field) + r":\s*\n\s*x:\s*(-?\d+)\s*\n\s*y:\s*(-?\d+)\s*\n\s*z:\s*(-?\d+)",
        chunk,
    )
    if not m:
        return None
    return {"x": int(m.group(1)), "y": int(m.group(2)), "z": int(m.group(3))}


def _parse_joints(chunk: str) -> List[Joint]:
    """Parse the `_joints:` list within one room entry."""
    joints: List[Joint] = []
    m = re.search(r"(?ms)^    _joints:\n(.*?)(?=^    _spawners:|\Z)", chunk)
    if not m:
        return joints
    body = m.group(1)
    if body.strip() == "[]":
        return joints
    # entries look like:
    #   - _uid:\n        _id: <uuid>\n      _type: 1\n ... _occludedJoints:
    for entry in re.split(r"(?m)^    - _uid:", body)[1:]:
        chunk_j = "_uid:" + entry
        uid_m = _ID_RE.search(chunk_j)
        if not uid_m:
            continue
        type_m = re.search(r"_type:\s*(-?\d+)", chunk_j)
        forbid_m = re.search(r"_forbidAsRandomEntrance:\s*(\d)", chunk_j)
        pos = _parse_int_block(chunk_j, "_position") or {"x": 0, "y": 0, "z": 0}
        dir_m = re.search(r"_direction:\s*\{x:\s*(-?[\d.]+),\s*y:\s*(-?[\d.]+)\}", chunk_j)
        dx = float(dir_m.group(1)) if dir_m else 1.0
        dz = float(dir_m.group(2)) if dir_m else 0.0
        joints.append(Joint(
            uid=uid_m.group(1),
            type=int(type_m.group(1)) if type_m else 1,
            position=Vec3(pos["x"], pos["y"], pos["z"]),
            direction=(dx, dz),
            forbid_as_random_entrance=bool(int(forbid_m.group(1))) if forbid_m else False,
        ))
    return joints


def parse_rooms_data(path: Path) -> List[Room]:
    """Parse every room entry from RoomsData.asset."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    rooms: List[Room] = []
    # Room entries start with "  - _name: ..." at 2-space indent.
    entries = re.split(r"(?m)^  - _name:\s*", text)[1:]
    for entry in entries:
        chunk = "_name:" + entry
        name = chunk.split("\n", 1)[0].strip()
        if name.startswith("_name:"):
            name = name[len("_name:"):].strip()

        uid_m = re.search(r"^    _uid:\s*\n\s*_id:\s*([0-9a-fA-F-]{36})", chunk, re.M)
        if not uid_m:
            continue
        kit_m = _KIT_RE.search(chunk)
        size = _parse_int_block(chunk, "_size") or {"x": 0, "y": 0, "z": 0}
        offset = _parse_int_block(chunk, "_offset") or {"x": 0, "y": 0, "z": 0}
        scene_m = re.search(r"_workScenePath:\s*(.*)", chunk)
        cat_m = re.search(r"_sizeCategory:\s*(\d+)", chunk)
        ars_m = re.search(r"_allowRandomSelection:\s*(\d)", chunk)
        ae_m = re.search(r"_useAsRandomEntrance:\s*(\d)", chunk)
        ax_m = re.search(r"_useAsRandomExit:\s*(\d)", chunk)

        rooms.append(Room(
            uid=uid_m.group(1),
            name=name,
            kit_file_id=int(kit_m.group(1)) if kit_m else None,
            kit_guid=kit_m.group(2) if kit_m else None,
            size=Vec3(size["x"], size["y"], size["z"]),
            offset=Vec3(offset["x"], offset["y"], offset["z"]),
            work_scene_path=scene_m.group(1).strip() if scene_m else "",
            size_category=int(cat_m.group(1)) if cat_m else 0,
            allow_random_selection=bool(int(ars_m.group(1))) if ars_m else False,
            use_as_random_entrance=bool(int(ae_m.group(1))) if ae_m else False,
            use_as_random_exit=bool(int(ax_m.group(1))) if ax_m else False,
            joints=_parse_joints(chunk),
        ))
    return rooms


class RoomsCatalog:
    """All rooms, indexed by uid and by kit guid."""

    def __init__(self, rooms: List[Room]):
        self.rooms = rooms
        self.by_uid: Dict[str, Room] = {r.uid: r for r in rooms}
        self.by_kit: Dict[str, List[Room]] = {}
        for r in rooms:
            if r.kit_guid:
                self.by_kit.setdefault(r.kit_guid, []).append(r)

    def __len__(self) -> int:
        return len(self.rooms)


def load_rooms(rooms_data_path: Path) -> RoomsCatalog:
    return RoomsCatalog(parse_rooms_data(Path(rooms_data_path)))
