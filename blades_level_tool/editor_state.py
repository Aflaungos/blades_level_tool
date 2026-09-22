"""Editor state: pure (non-GUI) logic for the layout editor.

Holds room placements and links, converts them to/from FixedLayout
models, and implements picking/snapping math so the pyglet window in
editor_window.py stays a thin shell.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .generator import validate
from .joints import joint_world, snap_transform, yaw_radians
from .kit_io import Kit
from .layout_io import parse_fixed_layout, write_fixed_layout
from .models import FixedLayout, Room, RoomInfo, RoomLink, new_uid
from .rooms_io import RoomsCatalog


class Placement:
    """One placed room instance in the editor."""

    _counter = 0

    def __init__(self, room: Room, pos: Tuple[float, float, float], rot: int):
        Placement._counter += 1
        self.room = room
        self.pos = pos            # world translation (x, y, z)
        self.rot = rot % 4        # quarter-turns CCW around +Y
        # uid assigned when converted to a FixedLayout (or parsed from one);
        # uses a per-session placeholder until then
        self.room_info_uid: Optional[str] = None

    def ensure_uid(self) -> str:
        if not self.room_info_uid:
            self.room_info_uid = new_uid()
        return self.room_info_uid

    @property
    def yaw(self) -> float:
        return yaw_radians(self.rot)

    def joint_worlds(self, grid: float) -> List[Tuple[Joint, Tuple[float, float, float], Tuple[float, float]]]:
        """[(joint, world_pos, world_dir), ...] for every joint."""
        out = []
        for joint in self.room.joints:
            wpos, wdir = joint_world(joint, grid, self.pos, self.rot)
            out.append((joint, wpos, wdir))
        return out


# import after class definition for the type annotation
from .models import Joint  # noqa: E402


class EditorState:
    """Placements + links + kit context, independent of any GUI."""

    def __init__(self, catalog: RoomsCatalog, kit: Kit, layout_name: str = "MyLayout"):
        self.catalog = catalog
        self.kit = kit
        self.layout_name = layout_name
        self.placements: List[Placement] = []
        # links reference (placement, joint_uid) pairs; placement objects
        # stay stable while editing
        self.links: List[Tuple[Placement, str, Placement, str]] = []
        self.source_path: Optional[Path] = None   # last loaded/saved .asset
        self.fixed_layout_uid: str = new_uid()

    # ------------------------------------------------------------------
    # room queries
    # ------------------------------------------------------------------

    def rooms_for_kit(self) -> List[Room]:
        return self.catalog.by_kit.get(self.kit.guid, [])

    def used_joints(self) -> set:
        used = set()
        for p, ju, _, _ in self.links:
            used.add((id(p), ju))
        return used

    def joint_is_used(self, placement: Placement, joint_uid: str) -> bool:
        return any(
            (a is placement and ju == joint_uid) or (b is placement and jv == joint_uid)
            for a, ju, b, jv in self.links
        )

    def snap_placement(self, target: Placement, target_joint_uid: str,
                       moving_room: Room) -> Optional[Tuple[Tuple[float, float, float], int, str]]:
        """Snap a new/moving room onto one of target's joints.

        Returns (pos, rot, moving_joint_uid) using the first moving joint
        that can oppose the target joint, or None.
        """
        target_joint = target.room.joint_by_uid(target_joint_uid)
        if target_joint is None:
            return None
        for joint in moving_room.joints:
            result = snap_transform(self.kit.grid_size, target.pos, target_joint, target.rot, joint)
            if result is not None:
                return (result[0], result[1], joint.uid)
        return None

    def can_link(self, a: Placement, a_joint_uid: str, b: Placement, b_joint_uid: str) -> Optional[str]:
        """Return an error string if the link is invalid, else None."""
        if a is b:
            return "cannot link a room to itself"
        ja = a.room.joint_by_uid(a_joint_uid)
        jb = b.room.joint_by_uid(b_joint_uid)
        if ja is None or jb is None:
            return "unknown joint"
        if self.joint_is_used(a, a_joint_uid) or self.joint_is_used(b, b_joint_uid):
            return "that joint is already linked"
        wa = joint_world(ja, self.kit.grid_size, a.pos, a.rot)
        wb = joint_world(jb, self.kit.grid_size, b.pos, b.rot)
        if abs(wa[1][0] + wb[1][0]) > 1e-6 or abs(wa[1][1] + wb[1][1]) > 1e-6:
            return "joint directions do not oppose each other"
        if ja.type != jb.type:
            return f"joint type mismatch ({ja.type} vs {jb.type})"
        return None

    # ------------------------------------------------------------------
    # editing operations
    # ------------------------------------------------------------------

    def add_link(self, a: Placement, a_joint_uid: str, b: Placement, b_joint_uid: str) -> Optional[str]:
        err = self.can_link(a, a_joint_uid, b, b_joint_uid)
        if err:
            return err
        self.links.append((a, a_joint_uid, b, b_joint_uid))
        return None

    def remove_placement(self, placement: Placement) -> None:
        self.placements = [p for p in self.placements if p is not placement]
        self.links = [
            (a, ju, b, jv) for (a, ju, b, jv) in self.links
            if a is not placement and b is not placement
        ]

    def clear(self) -> None:
        self.placements = []
        self.links = []
        self.source_path = None
        self.fixed_layout_uid = new_uid()

    # ------------------------------------------------------------------
    # FixedLayout conversion
    # ------------------------------------------------------------------

    def to_fixed_layout(self) -> FixedLayout:
        room_infos: List[RoomInfo] = []
        placement_uid: Dict[int, str] = {}
        for p in self.placements:
            uid = p.ensure_uid()
            placement_uid[id(p)] = uid
            room_infos.append(RoomInfo(
                room_info_uid=uid,
                room_uid=p.room.uid,
                is_critical_path=True,
                section_index=0,
            ))
        room_links: List[RoomLink] = []
        for a, ju, b, jv in self.links:
            room_links.append(RoomLink(
                first_room_info_uid=placement_uid[id(a)],
                first_joint_uid=ju,
                second_room_info_uid=placement_uid[id(b)],
                second_joint_uid=jv,
            ))
        return FixedLayout(
            name=self.layout_name,
            fixed_layout_uid=self.fixed_layout_uid,
            kit_file_id=11400000,
            kit_guid=self.kit.guid,
            kit_uid=self.kit.uid,
            room_infos=room_infos,
            room_links=room_links,
        )

    def save(self, path: Path) -> List[str]:
        layout = self.to_fixed_layout()
        rooms_by_uid = self.catalog.by_uid
        problems = validate(layout, rooms_by_uid)
        write_fixed_layout(layout, Path(path))
        self.source_path = Path(path)
        return problems

    def load(self, path: Path) -> None:
        layout = parse_fixed_layout(Path(path))
        self.layout_name = layout.name
        self.fixed_layout_uid = layout.fixed_layout_uid or new_uid()
        self.placements = []
        self.links = []
        by_rid: Dict[str, Placement] = {}
        for ri in layout.room_infos:
            room = self.catalog.by_uid.get(ri.room_uid)
            if room is None:
                continue  # unknown room (different kit?) -- skip
            p = Placement(room, (0.0, 0.0, 0.0), 0)
            p.room_info_uid = ri.room_info_uid
            self.placements.append(p)
            by_rid[ri.room_info_uid] = p
        # reflow placements along links so the dungeon assembles itself
        self._relayout_from_links(layout, by_rid)
        for link in layout.room_links:
            a = by_rid.get(link.first_room_info_uid)
            b = by_rid.get(link.second_room_info_uid)
            if a is None or b is None:
                continue
            self.links.append((a, link.first_joint_uid, b, link.second_joint_uid))
        self.source_path = Path(path)

    def _relayout_from_links(self, layout: FixedLayout, by_rid: Dict[str, Placement]) -> None:
        """Loaded layouts store no world transforms; rebuild approximate
        positions by snapping each room onto an already-placed neighbour
        (BFS over the link graph). Because snap_transform is deterministic
        (each joint pair has exactly one opposing-yaw solution), this
        reproduces the original arrangement up to the placement of the
        first room.
        """
        if not self.placements:
            return
        placed: set = set()
        first = self.placements[0]
        first.pos = (0.0, 0.0, 0.0)
        first.rot = 0
        placed.add(id(first))
        queue = [first]
        while queue:
            current = queue.pop(0)
            for link in layout.room_links:
                a = by_rid.get(link.first_room_info_uid)
                b = by_rid.get(link.second_room_info_uid)
                if a is None or b is None:
                    continue
                if a is current and id(b) not in placed:
                    self._snap_loaded(b, link.second_joint_uid, a, link.first_joint_uid)
                    placed.add(id(b))
                    queue.append(b)
                elif b is current and id(a) not in placed:
                    self._snap_loaded(a, link.first_joint_uid, b, link.second_joint_uid)
                    placed.add(id(a))
                    queue.append(a)
        # anything not reached (unlinked rooms): place on a grid row below
        slot = 0
        for p in self.placements:
            if id(p) not in placed:
                p.pos = (slot * 24.0, 0.0, -30.0)
                p.rot = 0
                slot += 1

    def _snap_loaded(self, moving: Placement, moving_joint_uid: str,
                     target: Placement, target_joint_uid: str) -> None:
        tj = target.room.joint_by_uid(target_joint_uid)
        mj = moving.room.joint_by_uid(moving_joint_uid)
        if tj is None or mj is None:
            return
        result = snap_transform(self.kit.grid_size, target.pos, tj, target.rot, mj)
        if result is not None:
            moving.pos, moving.rot = result


# ----------------------------------------------------------------------
# ray picking helpers (numpy)
# ----------------------------------------------------------------------

def ray_from_screen(mx: float, my: float, width: float, height: float,
                    proj: np.ndarray, view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Screen point -> (origin, direction) in world space."""
    ndc_x = (2.0 * mx / width - 1.0)
    ndc_y = (1.0 - 2.0 * my / height)
    inv_vp = np.linalg.inv(proj @ view)
    near = inv_vp @ np.array([ndc_x, ndc_y, -1.0, 1.0])
    far = inv_vp @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    origin = near[:3] / near[3]
    far_pt = far[:3] / far[3]
    direction = far_pt - origin
    n = np.linalg.norm(direction)
    if n == 0:
        return origin, np.array([0.0, 0.0, -1.0])
    return origin, direction / n


def ray_ground_point(origin: np.ndarray, direction: np.ndarray, y: float = 0.0) -> Optional[np.ndarray]:
    if abs(direction[1]) < 1e-8:
        return None
    t = (y - origin[1]) / direction[1]
    if t < 0:
        return None
    return origin + direction * t


def ray_aabb(origin: np.ndarray, direction: np.ndarray,
             box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """Slab test in the box's own space."""
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(direction[i]) < 1e-9:
            if origin[i] < box_min[i] or origin[i] > box_max[i]:
                return False
            continue
        t1 = (box_min[i] - origin[i]) / direction[i]
        t2 = (box_max[i] - origin[i]) / direction[i]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    return tmax >= tmin and tmax > 0


def placement_hit(placement: Placement, grid: float, origin: np.ndarray,
                  direction: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> bool:
    """Ray vs placement OBB: transform the ray into room-local space
    (translate, then rotate by -yaw around Y)."""
    inv_c, inv_s = math.cos(-placement.yaw), math.sin(-placement.yaw)
    dx, dy, dz = direction
    local_origin = np.array([
        (origin[0] - placement.pos[0]) * inv_c + (origin[2] - placement.pos[2]) * inv_s,
        origin[1] - placement.pos[1],
        -(origin[0] - placement.pos[0]) * inv_s + (origin[2] - placement.pos[2]) * inv_c,
    ])
    local_dir = np.array([dx * inv_c + dz * inv_s, dy, -dx * inv_s + dz * inv_c])
    return ray_aabb(local_origin, local_dir, bounds[0], bounds[1])


def world_to_screen(point, width: float, height: float,
                    proj: np.ndarray, view: np.ndarray) -> Optional[Tuple[float, float]]:
    v = proj @ view @ np.array([point[0], point[1], point[2], 1.0])
    if abs(v[3]) < 1e-9:
        return None
    ndc = v[:3] / v[3]
    if v[3] <= 0:
        return None
    return ((ndc[0] + 1.0) * 0.5 * width, (ndc[1] + 1.0) * 0.5 * height)
