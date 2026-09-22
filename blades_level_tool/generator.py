"""Helpers for building and validating a new FixedLayout by hand or
programmatically. This intentionally does NOT auto-generate random
dungeons -- joint "type" semantics (which joint can connect to which)
aren't confirmed yet (see README), so an editor should let you choose
each connection explicitly and validate afterward.
"""
from __future__ import annotations

from typing import Dict, List
import random

from .models import FixedLayout, Room, RoomInfo, RoomLink, new_uid


class LayoutBuilder:
    def __init__(self, name: str, kit_file_id: int, kit_guid: str, kit_uid: str):
        self.layout = FixedLayout(
            name=name,
            fixed_layout_uid=new_uid(),
            kit_file_id=kit_file_id,
            kit_guid=kit_guid,
            kit_uid=kit_uid,
        )

    def add_room(self, room: Room, critical_path: bool = True, section_index: int = 0) -> str:
        """Place an instance of `room` in the layout. Returns the new
        room_info_uid you'll use to reference this placement in links.
        """
        room_info_uid = new_uid()
        self.layout.room_infos.append(RoomInfo(
            room_info_uid=room_info_uid,
            room_uid=room.uid,
            is_critical_path=critical_path,
            section_index=section_index,
        ))
        return room_info_uid

    def link(self, room_info_uid_a: str, joint_uid_a: str,
              room_info_uid_b: str, joint_uid_b: str) -> None:
        self.layout.room_links.append(RoomLink(
            first_room_info_uid=room_info_uid_a,
            first_joint_uid=joint_uid_a,
            second_room_info_uid=room_info_uid_b,
            second_joint_uid=joint_uid_b,
        ))

    def build(self) -> FixedLayout:
        return self.layout


def validate(layout: FixedLayout, rooms_by_uid: Dict[str, Room]) -> List[str]:
    """Sanity-check a layout against the room catalog. Returns a list of
    problem descriptions; empty list means it looks structurally sound.
    NOTE: this checks structure (existence, no double-use of a joint,
    dangling references) -- it can NOT confirm the dungeon is actually
    playable/reachable in-game, and it does not know joint-type
    compatibility rules since those haven't been confirmed yet.
    """
    problems: List[str] = []

    room_info_by_uid = {ri.room_info_uid: ri for ri in layout.room_infos}
    joint_usage: Dict[str, int] = {}

    for ri in layout.room_infos:
        room = rooms_by_uid.get(ri.room_uid)
        if room is None:
            problems.append(f"room_info {ri.room_info_uid}: references unknown room uid {ri.room_uid}")

    for idx, link in enumerate(layout.room_links):
        for side, room_info_uid, joint_uid in (
            ("first", link.first_room_info_uid, link.first_joint_uid),
            ("second", link.second_room_info_uid, link.second_joint_uid),
        ):
            ri = room_info_by_uid.get(room_info_uid)
            if ri is None:
                problems.append(f"link[{idx}].{side}: unknown room_info_uid {room_info_uid}")
                continue
            room = rooms_by_uid.get(ri.room_uid)
            if room is None:
                continue  # already reported above
            joint = room.joint_by_uid(joint_uid)
            if joint is None:
                problems.append(
                    f"link[{idx}].{side}: room '{room.name}' has no joint with uid {joint_uid} "
                    f"(known joints: {[j.uid for j in room.joints]})"
                )
            key = f"{room_info_uid}:{joint_uid}"
            joint_usage[key] = joint_usage.get(key, 0) + 1

    for key, count in joint_usage.items():
        if count > 1:
            problems.append(f"joint {key} is used in {count} links -- a joint can only connect once")

    # every room_info should be reachable from at least one link, except
    # when there's only a single room in the whole layout.
    if len(layout.room_infos) > 1:
        linked_room_infos = set()
        for link in layout.room_links:
            linked_room_infos.add(link.first_room_info_uid)
            linked_room_infos.add(link.second_room_info_uid)
        for ri in layout.room_infos:
            if ri.room_info_uid not in linked_room_infos:
                problems.append(f"room_info {ri.room_info_uid} ({ri.room_uid}) has no links at all")

    return problems
