"""A tool for building Elder Scrolls Blades dungeon/town FixedLayouts.

Modules:
    models      - data classes for rooms, joints, layouts
    kit_io      - parse Kit .asset files (grid size, guid, uid)
    rooms_io    - parse RoomsData.asset into a room catalog
    mesh_io     - extract room preview meshes from Unity room prefabs
    layout_io   - read/write DungeonSettingsFixedLayout .asset files
    joints      - joint snapping math for the 3D editor
    editor_3d   - the pyglet 3D editor application
    cli         - command-line entry points (gui, validate, export)
"""
from .models import (
    Vec3, Quat, Joint, Room, RoomInfo, RoomLink, FixedLayout, new_uid,
)
from .kit_io import Kit, load_kit, load_kits
from .rooms_io import load_rooms, RoomsCatalog

__all__ = [
    "Vec3", "Quat", "Joint", "Room", "RoomInfo", "RoomLink", "FixedLayout",
    "new_uid", "Kit", "load_kit", "load_kits", "load_rooms", "RoomsCatalog",
]
