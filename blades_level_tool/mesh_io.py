"""Build a guid -> path index over the exported Unity project, and load
colliders from room prefabs.

Every asset gets a guid in its `.meta` file, and every reference to it
is `{fileID: ..., guid: <guid>, type: 2}`. Room prefabs reference their
Mesh and MeshCollider assets by guid, so we map guid -> first file that
owns it.
"""
from __future__ import annotations

import base64
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_GUID_LINE_RE = re.compile(r"guid:\s*([0-9a-fA-F]{32})")
_MESH_PTR_RE = re.compile(r"m_Mesh:\s*\{fileID:\s*(-?\d+),\s*guid:\s*([0-9a-fA-F]{32})")
_FILEID_RE = re.compile(r"^--- !u!(\d+) &(-?\d+)", re.M)
_HEX_RE = re.compile(r"[0-9a-fA-F]")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{8,}")


def build_guid_index(assets_root: Path, progress=None) -> Dict[str, Path]:
    """Scan every `.meta` file under an exported project's Assets folder
    and return {guid: first path owning that guid}.
    """
    assets_root = Path(assets_root)
    index: Dict[str, Path] = {}
    metas = list(assets_root.rglob("*.meta"))
    for i, meta in enumerate(metas):
        if progress and i % 5000 == 0:
            progress(i, len(metas))
        try:
            text = meta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = _GUID_LINE_RE.search(text)
        if m:
            guid = m.group(1)
            if guid not in index:
                asset_path = meta.with_suffix("")  # strip ".meta"
                if asset_path.exists():
                    index[guid] = asset_path
    return index


# ---------------------------------------------------------------------------
# Unity Mesh (.asset files with class id 43, e.g. under Assets/BGS/art/)
# ---------------------------------------------------------------------------

_FMT_SIZES = {0: 4, 1: 1, 2: 1, 3: 2, 4: 1, 5: 4, 8: 4, 9: 2, 10: 4, 11: 2, 12: 1}


def _parse_channels(text: str) -> Tuple[int, List[Tuple[int, int, int, int]]]:
    """Parse m_Channels -> (vertex_count, [(stream, offset, format, dim)]).
    Indentation varies between serialized versions, so we don't anchor it.
    """
    count_m = re.search(r"m_VertexCount:\s*(\d+)", text)
    vcount = int(count_m.group(1)) if count_m else 0
    channels = []
    m = re.search(r"(?ms)m_Channels:\s*\n(.*?)(?=^\s*m_\w+|\Z)", text)
    if not m:
        return vcount, channels
    for entry in re.split(r"(?m)^\s*- ", m.group(1))[1:]:
        stream_m = re.search(r"stream:\s*(-?\d+)", entry)
        off_m = re.search(r"offset:\s*(\d+)", entry)
        fmt_m = re.search(r"format:\s*(\d+)", entry)
        dim_m = re.search(r"dimension:\s*(\d+)", entry)
        channels.append((
            int(stream_m.group(1)) if stream_m else 0,
            int(off_m.group(1)) if off_m else 0,
            int(fmt_m.group(1)) if fmt_m else 0,
            int(dim_m.group(1)) if dim_m else 0,
        ))
    return vcount, channels


def _decode_mesh_vertices(data: bytes, vcount: int, channels) -> Optional[np.ndarray]:
    """Extract positions (channel 0) from a Unity vertex-data blob.

    The vertex stride is derived from the blob size itself (len/vcount);
    AssetRipper payloads don't always match the channels list (padding
    varies), so we trust the byte count over the declared layout.
    """
    if not vcount or not data:
        return None
    stride_exact = len(data) / vcount
    stride = int(stride_exact)
    if stride != stride_exact or stride < 12:
        # fall back: assume tightly packed 12-byte positions
        if len(data) >= vcount * 12:
            return np.frombuffer(data, dtype="<f4", count=vcount * 3).reshape(vcount, 3).copy()
        return None
    # sanity: channel 0 (positions) must fit in the stride
    if channels:
        stream, offset, fmt, dim = channels[0]
        if dim == 3 and offset + 12 > stride:
            return None
    arr = np.frombuffer(data, dtype=np.uint8, count=vcount * stride)
    arr = arr.reshape(vcount, stride)
    pos = arr[:, :12].copy().view("<f4").reshape(vcount, 3)
    return pos


def _extract_vertex_blobs(text: str) -> List[str]:
    """Pull the vertex-data `m_DataSize:` payload. AssetRipper serializes
    it as hex inside `m_VertexData: / _typelessdata: ...`.
    """
    blobs = []
    vd = text.find("m_VertexData:")
    search_text = text[vd:] if vd >= 0 else text
    for m in re.finditer(r"(?ms)_typelessdata:\s*([0-9a-fA-F\s]{64,}?)(?=^\s*m_\w+:|\Z)", search_text):
        blobs.append(m.group(1))
    return blobs


def load_mesh_asset_positions(path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Load vertex positions (and index buffer if present) from a Unity
    Mesh `.asset` file (class id 43)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not re.search(r"^--- !u!43", text, re.M):
        return None

    vcount, channels = _parse_channels(text)

    # index buffer: hex string, 2 bytes per entry (16-bit) or 4 (32-bit)
    idx_m = re.search(r"(?ms)m_IndexBuffer: ([0-9a-fA-F\s]+?)(?=^\s*m_\w+:|\Z)", text)
    idx_format_m = re.search(r"m_IndexFormat:\s*(\d+)", text)
    indices = None
    if idx_m:
        hexstr = re.sub(r"\s+", "", idx_m.group(1))
        if idx_format_m and idx_format_m.group(1) == "1":  # 32-bit
            indices = np.frombuffer(bytes.fromhex(hexstr), dtype="<u4").copy()
        else:  # 16-bit
            indices = np.frombuffer(bytes.fromhex(hexstr), dtype="<u2").astype(np.uint32)
        if len(hexstr) % 4:  # malformed
            indices = None

    positions = None
    for blob in _extract_vertex_blobs(text):
        hexstr = re.sub(r"\s+", "", blob)
        if len(hexstr) < 128 or len(hexstr) % 2:
            continue
        try:
            data = bytes.fromhex(hexstr)
        except ValueError:
            continue
        positions = _decode_mesh_vertices(data, vcount, channels)
        if positions is not None:
            break
    if positions is None:
        return None
    return positions, indices


# ---------------------------------------------------------------------------
# Room prefabs -> collider triangles in room-local space
# ---------------------------------------------------------------------------

class PrefabColliders:
    """Mesh-collider geometry found inside one room prefab, plus the room
    root's transform so it can be converted to room-local space."""

    def __init__(self) -> None:
        self.triangles: List[np.ndarray] = []   # world-space (room-local) tris, np (n,3,3)
        self.failed_guids: List[str] = []


def _parse_doc_objects(text: str) -> Dict[int, str]:
    """Split a prefab into {fileID: chunk_text}."""
    parts = re.split(r"(?=^--- !u!)", text, flags=re.M)
    out: Dict[int, str] = {}
    for part in parts:
        m = re.match(r"--- !u!\d+ &(-?\d+)", part)
        if m:
            out[int(m.group(1))] = part
    return out


def _get_transform(objs: Dict[int, str], file_id: int) -> Optional[dict]:
    """Parse one Transform chunk into rotation quat + position."""
    chunk = objs.get(file_id)
    if chunk is None:
        return None
    rot_m = re.search(
        r"m_LocalRotation:\s*\{x:\s*(-?[\d.eE+]+),\s*y:\s*(-?[\d.eE+]+),\s*z:\s*(-?[\d.eE+]+),\s*w:\s*(-?[\d.eE+]+)\}",
        chunk)
    pos_m = re.search(
        r"m_LocalPosition:\s*\{x:\s*(-?[\d.eE+]+),\s*y:\s*(-?[\d.eE+]+),\s*z:\s*(-?[\d.eE+]+)\}",
        chunk)
    if not pos_m:
        return None
    rot = None
    if rot_m:
        rot = (float(rot_m.group(1)), float(rot_m.group(2)),
               float(rot_m.group(3)), float(rot_m.group(4)))
    return {
        "rotation": rot,
        "position": (float(pos_m.group(1)), float(pos_m.group(2)), float(pos_m.group(3))),
    }


def load_room_geometry(prefab_path: Path, guid_index: Dict[str, Path],
                       max_meshes: int = 80) -> PrefabColliders:
    """Extract world-space (room-local) triangles from a room prefab:
    every MeshFilter's referenced mesh (the full visible geometry),
    transformed by its Transform chain up to the prefab root.

    The prefab root sits at the room origin with identity rotation, so
    the result can be drawn directly with the placement transform.
    """
    prefab_path = Path(prefab_path)
    text = prefab_path.read_text(encoding="utf-8", errors="replace")
    objs = _parse_doc_objects(text)
    out = PrefabColliders()

    # map: GameObject fileID -> transform fileID
    go_transform: Dict[int, int] = {}
    for fid, chunk in objs.items():
        if re.search(r"^Transform:", chunk, re.M):
            go_m = re.search(r"m_GameObject:\s*\{fileID:\s*(-?\d+)\}", chunk)
            if go_m:
                go_transform[int(go_m.group(1))] = fid

    def _chain_to_root(transform_fid: int) -> List[int]:
        """Return list of transform fileIDs from this node up to root."""
        chain = []
        current = transform_fid
        seen = set()
        while current and current in objs and current not in seen:
            seen.add(current)
            chain.append(current)
            father_m = re.search(r"m_Father:\s*\{fileID:\s*(-?\d+)\}", objs[current])
            if not father_m or father_m.group(1) == "0":
                break
            current = int(father_m.group(1))
        return chain

    quat_cache: Dict[Tuple[float, float, float, float], np.ndarray] = {}

    def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        key = (x, y, z, w)
        mat = quat_cache.get(key)
        if mat is not None:
            return mat
        n = x * x + y * y + z * z + w * w
        if n == 0:
            mat = np.eye(3)
        else:
            s = 2.0 / n
            mat = np.array([
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ])
        quat_cache[key] = mat
        return mat

    # find MeshFilters (visible geometry); MeshColliders often only cover
    # clutter and can be skipped entirely
    mesh_count = 0
    for fid, chunk in objs.items():
        if not re.search(r"^MeshFilter:", chunk, re.M):
            continue
        mesh_m = _MESH_PTR_RE.search(chunk)
        if not mesh_m:
            continue
        guid = mesh_m.group(2)
        mesh_path = guid_index.get(guid)
        if mesh_path is None:
            out.failed_guids.append(guid)
            continue
        mesh_data = load_mesh_asset_positions(mesh_path)
        if mesh_data is None:
            out.failed_guids.append(guid)
            continue
        positions, indices = mesh_data
        if indices is None or len(indices) < 3:
            continue
        go_m = re.search(r"m_GameObject:\s*\{fileID:\s*(-?\d+)\}", chunk)
        if not go_m:
            continue
        t_fid = go_transform.get(int(go_m.group(1)))
        if t_fid is None:
            continue
        chain = _chain_to_root(t_fid)

        # accumulate transform (root -> leaf)
        pos = np.zeros(3)
        rot = np.eye(3)
        for cfid in reversed(chain):
            info = _get_transform(objs, cfid)
            if info is None:
                continue
            if info["rotation"] is not None:
                x, y, z, w = info["rotation"]
                rot = rot @ _quat_matrix(x, y, z, w)
            pos = pos + rot @ np.asarray(info["position"], dtype=np.float64)

        verts = positions[indices]  # (n_tri*3, 3)
        tris = verts.reshape(-1, 3, 3)
        tris = tris @ rot.T + pos
        out.triangles.append(tris.astype(np.float32))
        mesh_count += 1
        if mesh_count >= max_meshes:
            break
    return out


def load_room_colliders(prefab_path: Path, guid_index: Dict[str, Path],
                        max_meshes: int = 40) -> PrefabColliders:
    """Extract only MeshCollider geometry from a room prefab (kept for
    analysis/debugging -- the editor renders load_room_geometry output).
    """
    prefab_path = Path(prefab_path)
    text = prefab_path.read_text(encoding="utf-8", errors="replace")
    objs = _parse_doc_objects(text)
    out = PrefabColliders()

    go_transform: Dict[int, int] = {}
    for fid, chunk in objs.items():
        if re.search(r"^Transform:", chunk, re.M):
            go_m = re.search(r"m_GameObject:\s*\{fileID:\s*(-?\d+)\}", chunk)
            if go_m:
                go_transform[int(go_m.group(1))] = fid

    def _chain_to_root(transform_fid: int) -> List[int]:
        chain = []
        current = transform_fid
        seen = set()
        while current and current in objs and current not in seen:
            seen.add(current)
            chain.append(current)
            father_m = re.search(r"m_Father:\s*\{fileID:\s*(-?\d+)\}", objs[current])
            if not father_m or father_m.group(1) == "0":
                break
            current = int(father_m.group(1))
        return chain

    quat_cache: Dict[Tuple[float, float, float, float], np.ndarray] = {}

    def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        key = (x, y, z, w)
        mat = quat_cache.get(key)
        if mat is not None:
            return mat
        n = x * x + y * y + z * z + w * w
        if n == 0:
            mat = np.eye(3)
        else:
            s = 2.0 / n
            mat = np.array([
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ])
        quat_cache[key] = mat
        return mat

    collider_count = 0
    for fid, chunk in objs.items():
        if not re.search(r"^MeshCollider:", chunk, re.M):
            continue
        mesh_m = _MESH_PTR_RE.search(chunk)
        if not mesh_m:
            continue
        guid = mesh_m.group(2)
        mesh_path = guid_index.get(guid)
        if mesh_path is None:
            out.failed_guids.append(guid)
            continue
        mesh_data = load_mesh_asset_positions(mesh_path)
        if mesh_data is None:
            out.failed_guids.append(guid)
            continue
        positions, indices = mesh_data
        if indices is None or len(indices) < 3:
            continue
        go_m = re.search(r"m_GameObject:\s*\{fileID:\s*(-?\d+)\}", chunk)
        if not go_m:
            continue
        t_fid = go_transform.get(int(go_m.group(1)))
        if t_fid is None:
            continue
        chain = _chain_to_root(t_fid)

        pos = np.zeros(3)
        rot = np.eye(3)
        for cfid in reversed(chain):  # root -> leaf
            info = _get_transform(objs, cfid)
            if info is None:
                continue
            if info["rotation"] is not None:
                x, y, z, w = info["rotation"]
                rot = rot @ _quat_matrix(x, y, z, w)
            pos = pos + rot @ np.asarray(info["position"], dtype=np.float64)

        verts = positions[indices]
        tris = verts.reshape(-1, 3, 3)
        tris = tris @ rot.T + pos
        out.triangles.append(tris.astype(np.float32))
        collider_count += 1
        if collider_count >= max_meshes:
            break
    return out


def triangles_to_arrays(triangle_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate triangle lists into (positions, flat indices) arrays."""
    total_tris = sum(len(t) for t in triangle_list)
    if total_tris == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.uint32)
    positions = np.concatenate(triangle_list, axis=0)
    indices = np.arange(total_tris * 3, dtype=np.uint32)
    return positions, indices
