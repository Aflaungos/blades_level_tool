"""Joint snapping math for the layout editor.

The game snaps two rooms together by aligning a joint on each side:
their world positions coincide and their directions point opposite ways.

Room transform model:
  - every placement has an integer 90-degree yaw `rot` (0..3 quarters,
    Unity/OpenGL +Y yaw convention: positive yaw maps +Z -> +X) and a
    world translation `pos` (unity units)
  - a room joint at grid cell (jx, jz) with grid direction (dx, dz) sits
    at local world offset (jx + dx/2, 0, jz + dz/2) * grid_size, and its
    world direction is (dx, dz) rotated by the placement yaw.

Snapping placement B to a joint of placed room A therefore means: choose
B's joint J and yaw q such that
    pos_B + R(q) * joint_offset_B  ==  world joint position of A
    world direction of B's joint   ==  -world direction of A's joint
which has exactly one solution for every (A joint, B joint) pair.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .models import Joint

# 90-degree yaw rotations (x, z) for rot = 0..3. Convention matches both
# Unity (Quaternion.Euler(0, rot*90, 0)) and OpenGL glRotatef(+-Y):
# positive yaw maps +Z -> +X, i.e. (x, z) -> (x c + z s, -x s + z c).
_YAW: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((1.0, 0.0), (0.0, 1.0)),     # rot 0: identity
    ((0.0, 1.0), (-1.0, 0.0)),    # rot 1: +90 yaw, (x, z) -> (z, -x)
    ((-1.0, 0.0), (0.0, -1.0)),   # rot 2: 180
    ((0.0, -1.0), (1.0, 0.0)),    # rot 3: +270
]


def rotate_xz(x: float, z: float, rot: int) -> Tuple[float, float]:
    """Rotate an (x, z) offset by rot quarter-turns of Unity +Y yaw."""
    (a, b), (c, d) = _YAW[rot % 4]
    return (x * a + z * b, x * c + z * d)


def joint_world(joint: Joint, grid: float, pos: Tuple[float, float, float],
                rot: int) -> Tuple[Tuple[float, float, float], Tuple[float, float]]:
    """World position and direction of a joint on a placed room.

    Joint grid positions are exact room-face coordinates (verified against
    the prefabs: Hall_00's joint at grid (8, 0, 2) sits at world
    (8*grid, 0, 2*grid) = (14.625, 0, 3.65625)), so no half-cell offsets
    are involved. `pos` is the room placement translation; the joint's y
    comes from the placement (joints are floor-level in the grid data).
    """
    rx, rz = rotate_xz(joint.position.x, joint.position.z, rot)
    wx = pos[0] + rx * grid
    wz = pos[2] + rz * grid
    dx, dz = rotate_xz(joint.direction[0], joint.direction[1], rot)
    return (wx, pos[1], wz), (dx, dz)


def snap_transform(grid: float, target_pos: Tuple[float, float, float],
                   target_joint: Joint, target_rot: int,
                   moving_joint: Joint) -> Optional[Tuple[Tuple[float, float, float], int]]:
    """Compute (pos, rot) that snaps a room's `moving_joint` onto an
    already-placed room's joint (target_pos/target_rot/target_joint),
    with directions opposing. Returns None if the joint has no direction
    or no opposing yaw exists.
    """
    twx, twy, twz = joint_world(target_joint, grid, target_pos, target_rot)[0]
    mdx, mdz = moving_joint.direction
    if mdx == 0 and mdz == 0:
        return None
    tdx, tdz = joint_world(target_joint, grid, target_pos, target_rot)[1]
    for rot in range(4):
        dx, dz = rotate_xz(mdx, mdz, rot)
        # direction must oppose the target's world direction
        if abs(dx + tdx) > 1e-6 or abs(dz + tdz) > 1e-6:
            continue
        # world offset of the moving joint at this rot must be subtracted
        rx, rz = rotate_xz(moving_joint.position.x, moving_joint.position.z, rot)
        px = twx - rx * grid
        pz = twz - rz * grid
        return ((px, twy, pz), rot)
    return None


def yaw_radians(rot: int) -> float:
    """Rotation in radians around +Y (Unity/OpenGL convention)."""
    return (rot % 4) * (math.pi / 2.0)
