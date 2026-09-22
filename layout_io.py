"""Read and write DungeonSettingsFixedLayout `.asset` files.

The format was reverse engineered from MQ17_FixedLayout.asset and the
decompiled DungeonSettingsFixedLayout.cs class -- field names below match
the C# serialized field names exactly, so this round-trips real files.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import FixedLayout, RoomInfo, RoomLink

# m_Script guid for DungeonSettingsFixedLayout, confirmed against MQ17_FixedLayout.asset.
FIXEDLAYOUT_SCRIPT_GUID = "b67e675204aed0ed2ec3de54bbc173e4"

_UID_RE = re.compile(r"_id:\s*([0-9a-fA-F-]+)")
_KIT_RE = re.compile(r"_kit:\s*\{fileID:\s*(-?\d+),\s*guid:\s*([0-9a-fA-F]*),\s*type:\s*(\d+)\}")
_NAME_RE = re.compile(r"m_Name:\s*(.*)")


def _pointer_block(chunk: str, label: str) -> dict:
    """Parse a `_uid: / _uidParent: / _searchType:` pointer block whose
    key is `label` (e.g. '_roomPointer', '_firstRoomJointPointer').
    Returns {'uid':..., 'uid_parent':..., 'search_type':...}.
    """
    m = re.search(
        re.escape(label) + r":\s*\n"
        r"\s*_uid:\s*\n\s*_id:\s*([0-9a-fA-F-]+)\s*\n"
        r"\s*_uidParent:\s*\n\s*_id:\s*([0-9a-fA-F-]+)\s*\n"
        r"\s*_searchType:\s*(-?\d+)",
        chunk,
    )
    if not m:
        return {"uid": None, "uid_parent": None, "search_type": None}
    return {"uid": m.group(1), "uid_parent": m.group(2), "search_type": int(m.group(3))}


def parse_fixed_layout(path: Path) -> FixedLayout:
    text = path.read_text(encoding="utf-8", errors="replace")

    name_m = _NAME_RE.search(text)
    name = name_m.group(1).strip() if name_m else path.stem

    layout_uid_m = re.search(r"_fixedLayoutUid:\s*\n\s*_id:\s*([0-9a-fA-F-]+)", text)
    fixed_layout_uid = layout_uid_m.group(1) if layout_uid_m else ""

    kit_m = _KIT_RE.search(text)
    kit_file_id = int(kit_m.group(1)) if kit_m else 0
    kit_guid = kit_m.group(2) if kit_m else ""

    # --- room infos ---
    room_infos: List[RoomInfo] = []
    kit_uid = ""
    ri_section_m = re.search(r"(?ms)^  _roomInfos:\n(.*?)(?=^  _roomLinks:)", text)
    if ri_section_m:
        for entry in re.split(r"(?m)^  - _roomInfoUID:", ri_section_m.group(1))[1:]:
            chunk = "_roomInfoUID:" + entry
            uid_m = re.search(r"_roomInfoUID:\s*\n\s*_uid:\s*\n\s*_id:\s*([0-9a-fA-F-]+)", chunk)
            ptr = _pointer_block(chunk, "_roomPointer")
            crit_m = re.search(r"_isConcideredOnCriticalPath:\s*(\d+)", chunk)
            sect_m = re.search(r"_sectionIndex:\s*(\d+)", chunk)
            if not uid_m or not ptr["uid"]:
                continue
            if ptr["uid_parent"] and not kit_uid:
                kit_uid = ptr["uid_parent"]
            room_infos.append(RoomInfo(
                room_info_uid=uid_m.group(1),
                room_uid=ptr["uid"],
                is_critical_path=bool(int(crit_m.group(1))) if crit_m else True,
                section_index=int(sect_m.group(1)) if sect_m else 0,
            ))

    # --- room links ---
    room_links: List[RoomLink] = []
    rl_section_m = re.search(r"(?ms)^  _roomLinks:\n(.*?)(?=^  _saveOnModification:)", text)
    if rl_section_m:
        for entry in re.split(r"(?m)^  - _firstRoomInfoPointer:", rl_section_m.group(1))[1:]:
            chunk = "_firstRoomInfoPointer:" + entry
            first_room = _pointer_block(chunk, "_firstRoomInfoPointer")
            first_joint = _pointer_block(chunk, "_firstRoomJointPointer")
            second_room = _pointer_block(chunk, "_secondRoomInfoPointer")
            second_joint = _pointer_block(chunk, "_secondRoomJointPointer")
            if not all([first_room["uid"], first_joint["uid"], second_room["uid"], second_joint["uid"]]):
                continue
            room_links.append(RoomLink(
                first_room_info_uid=first_room["uid"],
                first_joint_uid=first_joint["uid"],
                second_room_info_uid=second_room["uid"],
                second_joint_uid=second_joint["uid"],
            ))

    save_m = re.search(r"_saveOnModification:\s*(\d+)", text)
    save_on_modification = int(save_m.group(1)) if save_m else 0

    return FixedLayout(
        name=name,
        fixed_layout_uid=fixed_layout_uid,
        kit_file_id=kit_file_id,
        kit_guid=kit_guid,
        kit_uid=kit_uid,
        room_infos=room_infos,
        room_links=room_links,
        save_on_modification=save_on_modification,
    )


def _pointer_yaml(indent: str, label: str, uid: str, uid_parent: str, search_type: int) -> str:
    return (
        f"{indent}{label}:\n"
        f"{indent}  _uid:\n"
        f"{indent}    _id: {uid}\n"
        f"{indent}  _uidParent:\n"
        f"{indent}    _id: {uid_parent}\n"
        f"{indent}  _searchType: {search_type}\n"
    )


def write_fixed_layout(layout: FixedLayout, path: Path) -> None:
    """Serialize a FixedLayout back into Unity's exact YAML asset shape,
    matching MQ17_FixedLayout.asset byte-for-byte in structure (only the
    data differs). search_type is always 1 (TypeOnFixParent), matching
    every reference observed in real layout files.
    """
    lines = []
    lines.append("%YAML 1.1")
    lines.append("%TAG !u! tag:unity3d.com,2011:")
    lines.append("--- !u!114 &11400000")
    lines.append("MonoBehaviour:")
    lines.append("  m_ObjectHideFlags: 0")
    lines.append("  m_CorrespondingSourceObject: {fileID: 0}")
    lines.append("  m_PrefabInstance: {fileID: 0}")
    lines.append("  m_PrefabAsset: {fileID: 0}")
    lines.append("  m_GameObject: {fileID: 0}")
    lines.append("  m_Enabled: 1")
    lines.append("  m_EditorHideFlags: 0")
    lines.append(f"  m_Script: {{fileID: 11500000, guid: {FIXEDLAYOUT_SCRIPT_GUID}, type: 3}}")
    lines.append(f"  m_Name: {layout.name}")
    lines.append("  m_EditorClassIdentifier:")
    lines.append("  _fixedLayoutUid:")
    lines.append(f"    _id: {layout.fixed_layout_uid}")
    lines.append(f"  _kit: {{fileID: {layout.kit_file_id}, guid: {layout.kit_guid}, type: 2}}")
    lines.append("  _roomInfos:")
    for ri in layout.room_infos:
        lines.append("  - _roomInfoUID:")
        lines.append("      _uid:")
        lines.append(f"        _id: {ri.room_info_uid}")
        lines.append(_pointer_yaml("    ", "_roomPointer", ri.room_uid, layout.kit_uid, 1).rstrip("\n"))
        lines.append(f"    _isConcideredOnCriticalPath: {int(ri.is_critical_path)}")
        lines.append(f"    _sectionIndex: {ri.section_index}")
    lines.append("  _roomLinks:")
    for link in layout.room_links:
        first_ri = layout.room_info_by_uid(link.first_room_info_uid)
        second_ri = layout.room_info_by_uid(link.second_room_info_uid)
        # A joint pointer's _uidParent is the ROOM's own uid (the
        # _roomPointer target), not the room_info placement uid --
        # confirmed by round-tripping MQ17_FixedLayout.asset.
        first_room_uid = first_ri.room_uid if first_ri else link.first_room_info_uid
        second_room_uid = second_ri.room_uid if second_ri else link.second_room_info_uid

        lines.append("  - _firstRoomInfoPointer:")
        lines.append(f"      _uid:")
        lines.append(f"        _id: {link.first_room_info_uid}")
        lines.append(f"      _uidParent:")
        lines.append(f"        _id: {layout.fixed_layout_uid}")
        lines.append(f"      _searchType: 1")
        lines.append(f"    _firstRoomJointPointer:")
        lines.append(f"      _uid:")
        lines.append(f"        _id: {link.first_joint_uid}")
        lines.append(f"      _uidParent:")
        lines.append(f"        _id: {first_room_uid}")
        lines.append(f"      _searchType: 1")
        lines.append(f"    _secondRoomInfoPointer:")
        lines.append(f"      _uid:")
        lines.append(f"        _id: {link.second_room_info_uid}")
        lines.append(f"      _uidParent:")
        lines.append(f"        _id: {layout.fixed_layout_uid}")
        lines.append(f"      _searchType: 1")
        lines.append(f"    _secondRoomJointPointer:")
        lines.append(f"      _uid:")
        lines.append(f"        _id: {link.second_joint_uid}")
        lines.append(f"      _uidParent:")
        lines.append(f"        _id: {second_room_uid}")
        lines.append(f"      _searchType: 1")
    lines.append(f"  _saveOnModification: {layout.save_on_modification}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
