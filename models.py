"""Core data models for Elder Scrolls Blades dungeon layout editing.

These mirror the game's own C# classes (DungeonSettingsFixedLayout,
RoomsData, the room prefab Joint component) closely enough to round-trip
the Unity YAML asset format used for FixedLayout `.asset` files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import uuid


def new_uid() -> str:
    """Generate a fresh UUID in the same textual form the game uses."""
    return str(uuid.uuid4())


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Joint:
    """A connection point on a room prefab (Joint_XX GameObject)."""
    uid: str
    type: int
    game_object_name: str
    local_position: Vec3
    local_rotation: Quat
    forbid_as_random_entrance: bool = False


@dataclass
class Room:
    """One entry from RoomsData.asset, plus joints once cross-referenced
    against its prefab file via `catalog.attach_joints`.
    """
    uid: str
    name: str
    kit_file_id: Optional[int]
    kit_guid: Optional[str]
    size: Vec3
    offset: Vec3
    work_scene_path: str
    joints: List[Joint] = field(default_factory=list)

    def joint_by_uid(self, joint_uid: str) -> Optional[Joint]:
        for j in self.joints:
            if j.uid == joint_uid:
                return j
        return None


@dataclass
class RoomInfo:
    """One entry in a FixedLayout's `_roomInfos` list: a placed instance
    of a Room within this specific layout.
    """
    room_info_uid: str           # this placement's own identity
    room_uid: str                 # resolved Room.uid (the _roomPointer target)
    is_critical_path: bool = True
    section_index: int = 0


@dataclass
class RoomLink:
    """One entry in a FixedLayout's `_roomLinks` list: joint A on room
    placement A connects to joint B on room placement B.
    """
    first_room_info_uid: str
    first_joint_uid: str
    second_room_info_uid: str
    second_joint_uid: str


@dataclass
class FixedLayout:
    name: str
    fixed_layout_uid: str
    kit_file_id: int
    kit_guid: str
    kit_uid: str  # the Kit's own runtime Uid (distinct from the asset guid) -
                  # used as _uidParent on every _roomPointer in this layout
    room_infos: List[RoomInfo] = field(default_factory=list)
    room_links: List[RoomLink] = field(default_factory=list)
    save_on_modification: int = 0

    def room_info_by_uid(self, room_info_uid: str) -> Optional[RoomInfo]:
        for ri in self.room_infos:
            if ri.room_info_uid == room_info_uid:
                return ri
        return None

    def used_joint_uids(self) -> set:
        used = set()
        for link in self.room_links:
            used.add(link.first_joint_uid)
            used.add(link.second_joint_uid)
        return used
