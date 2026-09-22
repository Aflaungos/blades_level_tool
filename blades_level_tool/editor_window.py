"""The 3D layout editor (pyglet 2.x, OpenGL 3.3 core shaders).

Controls:
  Tab               open the room list (Up/Down + Enter to place, Esc closes)
  Left click        select room / drop the room being placed or moved
  G                 grab the selected room (move with mouse, click to drop)
  R / Shift+R       rotate the placed/grabbed/selected room by 90 degrees
  F                 snap the room being placed/moved onto the nearest free joint
  X / Delete        delete the selected room
  V                 toggle joint markers
  Ctrl+S            save layout (save-as dialog on first save)
  Ctrl+Shift+S      save layout as...
  Ctrl+L            load a layout
  Esc               cancel grab / deselect
  Right-drag        orbit camera | Middle-drag pan | wheel zoom | WASD+QE pan

Rooms render their actual exported geometry (walls, floors, props).
Rooms snap together at joints; links are written to a
DungeonSettingsFixedLayout .asset the game can load.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyglet
    from pyglet import gl
    from pyglet.graphics.shader import Shader, ShaderProgram
    from pyglet.math import Mat4, Vec3
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "pyglet 2.x is required for the 3D editor. Install with: pip install pyglet\n"
        f"(import error: {e})")

from .editor_state import (EditorState, Placement, placement_hit, ray_from_screen,
                           ray_ground_point)
from .joints import joint_world, yaw_radians
from .kit_io import load_kits
from .mesh_io import build_guid_index, load_room_geometry, triangles_to_arrays
from .paths import find_assets_root

# ----------------------------------------------------------------------
# shaders
# ----------------------------------------------------------------------

ROOM_VERTEX_SOURCE = """#version 330 core
    in vec3 position;
    in vec3 normals;
    in vec3 colors;

    out vec3 world_pos;
    out vec3 normal_v;
    out vec3 color_v;

    uniform mat4 u_projection;
    uniform mat4 u_view;
    uniform mat4 u_model;

    void main()
    {
        world_pos = (u_model * vec4(position, 1.0)).xyz;
        normal_v = mat3(u_model) * normals;
        color_v = colors;
        gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
    }
"""

ROOM_FRAGMENT_SOURCE = """#version 330 core
    in vec3 world_pos;
    in vec3 normal_v;
    in vec3 color_v;
    out vec4 out_color;

    uniform vec3 u_light_dir;   // normalized, points FROM the light
    uniform vec3 u_camera_pos;
    uniform vec3 u_tint;
    uniform float u_alpha;

    void main()
    {
        vec3 n = normalize(normal_v);
        if (!gl_FrontFacing)
            n = -n;                     // two-sided lighting
        float diff = max(dot(n, -u_light_dir), 0.0);
        float ambient = 0.42;
        vec3 view_dir = normalize(u_camera_pos - world_pos);
        vec3 halfv = normalize(view_dir - u_light_dir);
        float spec = pow(max(dot(n, halfv), 0.0), 24.0) * 0.15;
        vec3 rgb = color_v * (ambient + diff * 0.75) + vec3(spec);
        rgb *= u_tint;
        out_color = vec4(rgb, u_alpha);
    }
"""

FLAT_VERTEX_SOURCE = """#version 330 core
    in vec3 position;
    in vec3 colors;
    out vec3 color_v;
    uniform mat4 u_projection;
    uniform mat4 u_view;
    void main()
    {
        color_v = colors;
        gl_Position = u_projection * u_view * vec4(position, 1.0);
    }
"""

FLAT_FRAGMENT_SOURCE = """#version 330 core
    in vec3 color_v;
    out vec4 out_color;
    uniform float u_alpha;
    void main()
    {
        out_color = vec4(color_v, u_alpha);
    }
"""


def _room_program() -> ShaderProgram:
    return ShaderProgram(Shader(ROOM_VERTEX_SOURCE, "vertex"),
                         Shader(ROOM_FRAGMENT_SOURCE, "fragment"))


def _flat_program() -> ShaderProgram:
    return ShaderProgram(Shader(FLAT_VERTEX_SOURCE, "vertex"),
                         Shader(FLAT_FRAGMENT_SOURCE, "fragment"))


def _m4_to_np(m: Mat4) -> np.ndarray:
    """pyglet Mat4 (column-major storage) -> math-standard numpy 4x4."""
    return np.asarray(tuple(m), dtype=np.float64).reshape(4, 4).T


class RoomGeometryGPU:
    """Static room geometry in a vertex list, drawn with a per-room
    u_model matrix. Vertex colors are derived from the normal direction
    (floors lighter, walls darker) so rooms read visually without
    textures."""

    def __init__(self, positions: np.ndarray, program: ShaderProgram):
        verts = positions.reshape(-1, 3).astype(np.float32)
        tris = verts.reshape(-1, 3, 3)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        n = (n / norm).astype(np.float32)

        upness = np.clip(n[:, 1], 0.0, 1.0)
        wallness = np.clip(1.0 - abs(n[:, 1]), 0.0, 1.0)
        base = 0.30 + 0.35 * upness + 0.12 * wallness
        tint = np.stack([base * 1.0, base * 0.97, base * 0.92], axis=1).astype(np.float32)
        colors = np.repeat(tint, 3, axis=0)

        self.tri_count = len(tris)
        # attribute data must be flat float sequences (count * 3 floats)
        self.vertex_list = program.vertex_list(
            self.tri_count * 3, gl.GL_TRIANGLES,
            position=("f3", verts.reshape(-1)),
            normals=("f3", np.repeat(n, 3, axis=0).reshape(-1)),
            colors=("f3", colors.reshape(-1)),
        )

    def draw(self):
        self.vertex_list.draw(gl.GL_TRIANGLES)

    def dispose(self):
        try:
            self.vertex_list.delete()
        except Exception:
            pass


class LinePool:
    """Dynamic line renderer over a persistent VBO/VAO with an explicit
    draw count (no fixed-function, no per-frame reallocation)."""

    CAPACITY = 8000

    def __init__(self, program: ShaderProgram):
        self.program = program
        self.count = 0  # number of lines currently set
        self._pos = np.zeros((self.CAPACITY * 2, 3), np.float32)
        self._col = np.zeros((self.CAPACITY * 2, 3), np.float32)
        self._prev_count = 0

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)

        byte_stride = 6 * 4
        program.use()
        loc_pos = program.attributes["position"]["location"]
        loc_col = program.attributes["colors"]["location"]
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.CAPACITY * 2 * byte_stride, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(loc_pos)
        gl.glVertexAttribPointer(loc_pos, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 byte_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(loc_col)
        gl.glVertexAttribPointer(loc_col, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 byte_stride, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        program.stop()

    def begin(self):
        self.count = 0

    def add(self, a, b, color):
        if self.count >= self.CAPACITY:
            return
        i = self.count
        self._pos[i * 2 + 0] = a
        self._pos[i * 2 + 1] = b
        self._col[i * 2 + 0] = color
        self._col[i * 2 + 1] = color
        self.count += 1

    def add_box(self, bmin, bmax, color):
        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
             (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            self.add(c[a], c[b], color)

    def end(self):
        n = self.count * 2
        if self._prev_count > n:
            # hide stale lines from previous frames
            self._pos[n:self._prev_count] = 0.0
            self._col[n:self._prev_count] = 0.0
        self._prev_count = max(n, self._prev_count)

    def draw(self):
        if self.count == 0 and self._prev_count == 0:
            return
        n = self.count * 2
        upload = max(n, self._prev_count)
        interleaved = np.empty((upload, 6), np.float32)
        interleaved[:, :3] = self._pos[:upload]
        interleaved[:, 3:] = self._col[:upload]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, interleaved.nbytes,
                           interleaved.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, upload)
        gl.glBindVertexArray(0)
        self._prev_count = n


import ctypes  # noqa: E402


class EditorWindow(pyglet.window.Window):
    JOINT_FREE = (0.3, 1.0, 0.4)
    JOINT_USED = (1.0, 0.45, 0.2)
    GRID_COLOR = (0.30, 0.32, 0.38)
    GRID_MAJOR = (0.42, 0.45, 0.52)
    LIGHT_DIR = (0.45, 0.8, 0.5)   # normalized below

    def __init__(self, state: EditorState, geometries: Dict[str, RoomGeometryGPU],
                 room_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        super().__init__(1280, 720, resizable=True, caption="Blades Level Editor")
        self.state = state
        self.geometries = geometries
        self.room_bounds = room_bounds

        self.room_program = _room_program()
        self.flat_program = _flat_program()
        self.lines = LinePool(self.flat_program)
        self.room_program.use()
        self.room_program["u_model"] = Mat4()
        self.room_program["u_tint"] = (1.0, 1.0, 1.0)
        self.room_program["u_alpha"] = 1.0
        self.room_program.stop()

        self.selected: Optional[Placement] = None
        self.grabbed: Optional[Placement] = None
        self.grab_offset = np.zeros(3)
        self.ghost: Optional[Placement] = None
        self.ghost_snap_joint_uid: Optional[str] = None
        self.show_joints = True
        self.cam_yaw = math.radians(25)
        self.cam_pitch = math.radians(-38)
        self.cam_dist = 45.0
        self.cam_target = np.array([8.0, 0.0, 8.0])
        self._mouse_pos = (0, 0)
        self._keys = set()
        self._status = "Ready. Press Tab to add a room."
        self._status_until = 0.0
        self.room_list_visible = False
        self.room_list_index = 0
        self._proj_np = np.eye(4)
        self._view_np = np.eye(4)
        self._eye = np.array([0.0, 0.0, 40.0])

    def flash(self, text: str, seconds: float = 4.0):
        self._status = text
        self._status_until = time.time() + seconds

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------

    def _update_camera(self, dt):
        speed = 24.0 * dt
        keys = self._keys
        from pyglet.window import key
        if key.LSHIFT in keys:
            speed *= 3.0
        fwd = np.array([math.sin(self.cam_yaw), 0, math.cos(self.cam_yaw)])
        right = np.array([math.cos(self.cam_yaw), 0, -math.sin(self.cam_yaw)])
        if key.W in keys:
            self.cam_target += fwd * speed
        if key.S in keys:
            self.cam_target -= fwd * speed
        if key.A in keys:
            self.cam_target -= right * speed
        if key.D in keys:
            self.cam_target += right * speed
        if key.Q in keys:
            self.cam_target[1] -= speed
        if key.E in keys:
            self.cam_target[1] += speed

    def _apply_camera(self):
        aspect = self.width / max(1.0, float(self.height))
        proj = Mat4.perspective_projection(aspect, 0.5, 1500.0, fov=55)
        eye = self.cam_target + self.cam_dist * np.array([
            math.sin(self.cam_yaw) * math.cos(self.cam_pitch),
            -math.sin(self.cam_pitch),
            math.cos(self.cam_yaw) * math.cos(self.cam_pitch)])
        self._eye = eye
        view = Mat4.look_at(Vec3(*eye), Vec3(*self.cam_target), Vec3(0.0, 1.0, 0.0))
        self.projection = proj   # window UBO (used by text later)
        self.view = view
        self._proj_np = _m4_to_np(proj)
        self._view_np = _m4_to_np(view)

        self.room_program["u_projection"] = proj
        self.room_program["u_view"] = view
        self.flat_program["u_projection"] = proj
        self.flat_program["u_view"] = view

    # ------------------------------------------------------------------
    # picking
    # ------------------------------------------------------------------

    def _ray(self, mx, my):
        return ray_from_screen(mx, my, self.width, self.height,
                               self._proj_np, self._view_np)

    def _placement_bounds(self, p: Placement) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Cached bounds, with a grid-size fallback so picking still works
        for rooms whose geometry failed to extract."""
        bounds = self.room_bounds.get(p.room.uid)
        if bounds is None:
            g = self.state.kit.grid_size
            r = p.room
            if r.size.x > 0 and r.size.z > 0:
                bounds = (np.array([0.0, -1.0, 0.0]),
                          np.array([r.size.x * g, r.size.y * g, r.size.z * g]))
                self.room_bounds[p.room.uid] = bounds
        return bounds

    def _pick_placement(self, mx, my) -> Optional[Placement]:
        origin, direction = self._ray(mx, my)
        best, best_t = None, None
        for p in self.state.placements:
            bounds = self._placement_bounds(p)
            if bounds is None:
                continue
            if placement_hit(p, self.state.kit.grid_size, origin, direction, bounds):
                t = np.dot((np.asarray(p.pos) - origin), direction)
                if best_t is None or t < best_t:
                    best, best_t = p, t
        return best

    def _mouse_ground(self, y=0.0) -> Optional[np.ndarray]:
        origin, direction = self._ray(*self._mouse_pos)
        return ray_ground_point(origin, direction, y)

    # ------------------------------------------------------------------
    # editing ops
    # ------------------------------------------------------------------

    def start_ghost(self, room):
        self.ghost = Placement(room, (0.0, 0.0, -25.0), 0)
        self.ghost_snap_joint_uid = None
        self.selected = None
        self.flash(f"Placing {room.name} -- click to drop, R rotates, F snaps to nearest joint")

    def grab_selected(self):
        if not self.selected:
            return
        self.grabbed = self.selected
        ground = self._mouse_ground()
        if ground is not None:
            self.grab_offset = np.asarray(self.grabbed.pos) - ground
            self.grab_offset[1] = 0.0
        self.flash("Moving room -- left click to drop, R rotates")

    def drop_grabbed(self):
        if self.grabbed:
            self.selected = self.grabbed
            self.grabbed = None
            self.flash("Room moved")

    def try_snap_to_nearest_joint(self):
        moving = self.ghost or self.grabbed
        if not moving:
            return
        best_p, best_joint, best_d = None, None, 1e18
        for p in self.state.placements:
            if p is moving:
                continue
            for joint, wpos, wdir in p.joint_worlds(self.state.kit.grid_size):
                if self.state.joint_is_used(p, joint.uid):
                    continue
                d = math.hypot(wpos[0] - moving.pos[0], wpos[2] - moving.pos[2])
                if d < best_d:
                    best_p, best_joint, best_d = p, joint, d
        if best_p is None:
            self.ghost_snap_joint_uid = None
            return
        result = self.state.snap_placement(best_p, best_joint.uid, moving.room)
        if result is None:
            self.ghost_snap_joint_uid = None
            return
        pos, rot, joint_uid = result
        moving.pos, moving.rot = pos, rot
        self.ghost_snap_joint_uid = joint_uid
        self.flash(f"Snapped to joint of {best_p.room.name[:-5]}")

    def commit_ghost(self):
        if not self.ghost:
            return
        moving = self.ghost
        target_p, target_joint_uid = None, None
        if self.ghost_snap_joint_uid:
            mj = moving.room.joint_by_uid(self.ghost_snap_joint_uid)
            if mj is not None:
                mw_pos, _ = joint_world(mj, self.state.kit.grid_size, moving.pos, moving.rot)
                for p in self.state.placements:
                    for joint, wpos, wdir in p.joint_worlds(self.state.kit.grid_size):
                        if (abs(mw_pos[0] - wpos[0]) < 1e-3 and
                                abs(mw_pos[2] - wpos[2]) < 1e-3):
                            target_p, target_joint_uid = p, joint.uid
                            break
                    if target_p:
                        break
        if target_p:
            err = self.state.add_link(target_p, target_joint_uid, moving,
                                      self.ghost_snap_joint_uid)
            if err:
                self.flash(f"Cannot link: {err}", 6)
        self.state.placements.append(moving)
        self.selected = moving
        self.ghost = None
        self.ghost_snap_joint_uid = None
        self.flash("Room placed" + (" and linked" if target_p else " (unlinked)"))

    def delete_selected(self):
        if not self.selected:
            return
        self.state.remove_placement(self.selected)
        self.selected = None
        self.flash("Deleted")

    def save_layout(self, path: Path):
        problems = self.state.save(path)
        if problems:
            self.flash("Saved with warnings: " + "; ".join(problems[:3]), 8)
        else:
            self.flash(f"Saved {path.name} ({len(self.state.placements)} rooms, "
                       f"{len(self.state.links)} links)")

    def _file_dialog(self, save: bool):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if save:
                path = filedialog.asksaveasfilename(
                    defaultextension=".asset",
                    filetypes=[("FixedLayout asset", "*.asset"), ("All files", "*.*")],
                    initialfile=f"{self.state.layout_name}.asset", parent=root)
            else:
                path = filedialog.askopenfilename(
                    filetypes=[("FixedLayout asset", "*.asset"), ("All files", "*.*")],
                    parent=root)
        finally:
            root.destroy()
        return path

    def open_save_dialog(self):
        try:
            path = self._file_dialog(save=True)
        except Exception as e:
            self.flash(f"Dialog failed ({e}); see console", 6)
            path = input("Save to: ").strip().strip('"')
        if path:
            self.save_layout(Path(path))

    def open_load_dialog(self):
        try:
            path = self._file_dialog(save=False)
        except Exception:
            return
        if path:
            try:
                self.state.load(Path(path))
                self.flash(f"Loaded {Path(path).name}: {len(self.state.placements)} rooms, "
                           f"{len(self.state.links)} links")
            except Exception as e:
                self.flash(f"Load failed: {e}", 8)

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def on_mouse_motion(self, x, y, dx, dy):
        self._mouse_pos = (x, y)
        moving = self.ghost or self.grabbed
        if moving:
            ground = self._mouse_ground()
            if ground is not None:
                if self.grabbed:
                    newpos = ground + self.grab_offset
                    moving.pos = (float(newpos[0]), moving.pos[1], float(newpos[2]))
                else:
                    moving.pos = (float(ground[0]), moving.pos[1], float(ground[2]))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        from pyglet.window import mouse
        moving = self.ghost or self.grabbed
        if moving and buttons & mouse.LEFT:
            self.on_mouse_motion(x, y, dx, dy)
            return
        if buttons & mouse.RIGHT:
            self.cam_yaw -= dx * 0.008
            self.cam_pitch = max(-1.45, min(-0.05, self.cam_pitch + dy * 0.006))
        elif buttons & mouse.MIDDLE:
            fwd = np.array([math.sin(self.cam_yaw), 0, math.cos(self.cam_yaw)])
            right = np.array([math.cos(self.cam_yaw), 0, -math.sin(self.cam_yaw)])
            self.cam_target -= right * dx * 0.04
            self.cam_target += fwd * dy * 0.04
        self._mouse_pos = (x, y)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.cam_dist = max(5.0, min(250.0, self.cam_dist * (1.0 - scroll_y * 0.1)))

    def on_mouse_press(self, x, y, button, modifiers):
        from pyglet.window import mouse
        if button == mouse.LEFT:
            if self.ghost:
                self.commit_ghost()
                return
            if self.grabbed:
                self.drop_grabbed()
                return
            self.selected = self._pick_placement(x, y)

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        self._keys.add(symbol)
        ctrl = bool(modifiers & key.MOD_CTRL)
        shift = bool(modifiers & key.MOD_SHIFT)

        if symbol == key.ESCAPE:
            if self.room_list_visible:
                self.room_list_visible = False
            elif self.ghost:
                self.ghost = None
                self.flash("Cancelled placement")
            elif self.grabbed:
                self.grabbed = None
                self.flash("Cancelled move")
            else:
                self.selected = None
            return
        if ctrl and symbol == key.S:
            if shift or not self.state.source_path:
                self.open_save_dialog()
            else:
                self.save_layout(self.state.source_path)
            return
        if ctrl and symbol == key.L:
            self.open_load_dialog()
            return
        if self.room_list_visible:
            rooms = self.state.rooms_for_kit()
            if symbol == key.UP:
                self.room_list_index = max(0, self.room_list_index - 1)
            elif symbol == key.DOWN:
                self.room_list_index = min(len(rooms) - 1, self.room_list_index + 1)
            elif symbol == key.ENTER:
                if rooms:
                    self.room_list_visible = False
                    self.start_ghost(rooms[self.room_list_index % len(rooms)])
            elif symbol == key.TAB:
                self.room_list_visible = False
            return

        if symbol == key.TAB:
            self.room_list_visible = True
            self.room_list_index = 0
        elif symbol == key.G and self.selected:
            self.grab_selected()
        elif symbol == key.R:
            moving = self.ghost or self.grabbed or self.selected
            if moving:
                moving.rot = (moving.rot + (-1 if shift else 1)) % 4
        elif symbol == key.F:
            if self.ghost or self.grabbed:
                self.try_snap_to_nearest_joint()
            elif self.selected:
                self.grab_selected()
                self.try_snap_to_nearest_joint()
        elif symbol in (key.DELETE, key.X) and not ctrl:
            self.delete_selected()
        elif symbol == key.V:
            self.show_joints = not self.show_joints

    def on_key_release(self, symbol, modifiers):
        self._keys.discard(symbol)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        gl.glViewport(0, 0, width, height)

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def on_draw(self):
        self._update_camera(1 / 60.0)
        self._apply_camera()
        gl.glClearColor(0.09, 0.10, 0.12, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        # ---- rooms (per-room model matrix) ----
        prog = self.room_program
        prog.use()
        light = np.array(self.LIGHT_DIR, dtype=np.float32)
        light /= np.linalg.norm(light)
        prog["u_light_dir"] = tuple(light)
        prog["u_camera_pos"] = tuple(self._eye)

        for p in self.state.placements:
            self._draw_room(p)
        if self.grabbed:
            prog["u_tint"] = (1.5, 1.5, 1.2)
            prog["u_alpha"] = 1.0
            prog["u_model"] = self._model_mat(self.grabbed)
            self._geo_draw(self.grabbed)
        if self.ghost:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)
            prog["u_tint"] = (1.2, 1.6, 1.9)
            prog["u_alpha"] = 0.55
            prog["u_model"] = self._model_mat(self.ghost)
            self._geo_draw(self.ghost)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_BLEND)
        prog.stop()

        # ---- lines (grid, bounds, joints) ----
        self._build_lines()
        self.flat_program.use()
        self.flat_program["u_alpha"] = 1.0
        self.lines.draw()
        self.flat_program.stop()
        gl.glDisable(gl.GL_CULL_FACE)

        # ---- HUD (2D ortho for the default text shader) ----
        self.projection = Mat4.orthogonal_projection(
            0, self.width, 0, self.height, -255, 255)
        self.view = Mat4()
        self._draw_hud()

    def _model_mat(self, p: Placement) -> Mat4:
        return Mat4.from_translation(Vec3(*p.pos)) @ Mat4.from_rotation(p.yaw, Vec3(0, 1, 0))

    def _draw_room(self, p: Placement):
        prog = self.room_program
        prog["u_tint"] = (1.55, 1.55, 1.2) if p is self.selected else (1.0, 1.0, 1.0)
        prog["u_alpha"] = 1.0
        prog["u_model"] = self._model_mat(p)
        self._geo_draw(p)

    def _geo_draw(self, p: Placement):
        geo = self.geometries.get(p.room.uid)
        if geo is not None:
            geo.draw()

    def _build_lines(self):
        L = self.lines
        L.begin()
        g = self.state.kit.grid_size
        ext = 40
        for i in range(-ext, ext + 1):
            t = i * g
            c = self.GRID_MAJOR if i % 4 == 0 else self.GRID_COLOR
            L.add((t, -0.02, -ext * g), (t, -0.02, ext * g), c)
            L.add((-ext * g, -0.02, t), (ext * g, -0.02, t), c)
        for p in self.state.placements:
            bounds = self._placement_bounds(p)
            if bounds is None:
                continue
            color = (1.0, 0.8, 0.3) if p is self.selected else (0.32, 0.55, 0.85)
            L.add_box(bounds[0], bounds[1], color)
        if self.show_joints:
            r = g * 0.3
            drawn = set()
            for p in self.state.placements + ([self.ghost] if self.ghost else []):
                for joint, wpos, wdir in p.joint_worlds(g):
                    key = (round(wpos[0], 3), round(wpos[2], 3))
                    used = self.state.joint_is_used(p, joint.uid) or key in drawn
                    color = self.JOINT_USED if used else self.JOINT_FREE
                    if key not in drawn:
                        drawn.add(key)
                    y = wpos[1] + 0.06
                    L.add((wpos[0] - r, y, wpos[2] - r), (wpos[0] + r, y, wpos[2] + r), color)
                    L.add((wpos[0] - r, y, wpos[2] + r), (wpos[0] + r, y, wpos[2] - r), color)
                    L.add((wpos[0], y, wpos[2]),
                          (wpos[0] + wdir[0] * g * 0.5, y, wpos[2] + wdir[1] * g * 0.5), color)
        L.end()

    def _draw_hud(self):
        lines = []
        if time.time() < self._status_until:
            lines.append(">> " + self._status)
        lines.append("Tab rooms | LMB select/drop | G move | R rotate | F snap | X delete | "
                     "V joints | Ctrl+S save | Ctrl+L load | RMB orbit | wheel zoom")
        lines.append(f"Kit: {self.state.kit.name}  Rooms: {len(self.state.placements)}  "
                     f"Links: {len(self.state.links)}  "
                     f"File: {self.state.source_path.name if self.state.source_path else '(unsaved)'}")
        for i, text in enumerate(lines):
            label = pyglet.text.Label(text, font_size=10, x=10, y=self.height - 20 - i * 16,
                                      anchor_y="top", color=(235, 235, 240, 255))
            label.draw()

        if self.room_list_visible:
            rooms = self.state.rooms_for_kit()
            for i, room in enumerate(rooms[:32]):
                color = (255, 230, 120, 255) if i == self.room_list_index else (220, 220, 225, 255)
                label = pyglet.text.Label(room.name, font_size=9, x=16,
                                          y=self.height - 38 - i * 14,
                                          anchor_y="top", color=color)
                label.draw()
            hint = pyglet.text.Label("Up/Down + Enter to place, Esc to close",
                                     font_size=9, x=16, y=self.height - 38 - min(len(rooms), 32) * 14,
                                     anchor_y="top", color=(160, 200, 255, 255))
            hint.draw()


# ----------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------

def run_editor(assets: Optional[str] = None, kit_name: Optional[str] = None,
               kit_guid: Optional[str] = None, layout: Optional[Path] = None) -> int:
    assets_root = find_assets_root(assets)
    if not assets_root:
        print("Could not find the exported game data automatically.")
        print("Pass --assets pointing at the ExportedProject/Assets folder.")
        return 1
    print(f"Game data: {assets_root}")

    print("Indexing asset guids...")
    t0 = time.time()
    guid_index = build_guid_index(assets_root)
    print(f"  {len(guid_index)} assets indexed in {time.time() - t0:.1f}s")

    rooms = load_rooms(assets_root / "export/gameplaymetadata/RoomsData.asset")
    kits = load_kits(assets_root / "BGS/scriptableobjects/kits")
    kit = None
    if kit_guid and kit_guid in kits:
        kit = kits[kit_guid]
    elif kit_name:
        for k in kits.values():
            if k.name.lower() == kit_name.lower():
                kit = k
    if kit is None:
        kit = kits.get("63d3a7591920a734e92cddaf5e664e8d")  # Ayleid
    if kit is None:
        kit = next(iter(kits.values()))
    print(f"Kit: {kit.name}")

    state = EditorState(rooms, kit)

    if layout:
        state.load(Path(layout))
        print(f"Loaded layout: {Path(layout).name} ({len(state.placements)} rooms)")

    # geometry cache
    cache_dir = Path.home() / ".blades_level_tool"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"geom_cache_{kit.guid}.npz"
    cached_arrays: Dict[str, np.ndarray] = {}
    if cache_file.exists():
        try:
            with np.load(cache_file) as npz:
                cached_arrays = {k: npz[k] for k in npz.files}
        except Exception:
            cached_arrays = {}

    room_list = state.rooms_for_kit()
    todo = [r for r in room_list if r.uid not in cached_arrays]
    print(f"Room geometry: {len(cached_arrays)} cached, {len(todo)} to extract")
    for i, room in enumerate(todo):
        base = room.name[:-len("_Room")]
        prefab = assets_root / "export/resources" / base.lower() / f"{room.name}.prefab"
        if not prefab.exists():
            alt = list((assets_root / "export/resources").glob(f"*/{room.name}.prefab"))
            prefab = alt[0] if alt else None
        if prefab is None:
            print(f"  [{i + 1}/{len(todo)}] {room.name}: prefab not found")
            cached_arrays[room.uid] = np.zeros((0, 3), np.float32)
            continue
        geo = load_room_geometry(prefab, guid_index)
        positions, _ = triangles_to_arrays(geo.triangles)
        cached_arrays[room.uid] = positions.astype(np.float32)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"  [{i + 1}/{len(todo)}] extracted {room.name}")
    if todo:
        try:
            np.savez_compressed(cache_file, **cached_arrays)
        except Exception as e:
            print(f"  (cache save failed: {e})")

    # sanitize positions (drop NaNs) and compute bounds
    clean_arrays: Dict[str, np.ndarray] = {}
    bounds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for uid, positions in cached_arrays.items():
        if len(positions):
            finite = positions[np.isfinite(positions).all(axis=1)]
            if len(finite) >= 3:
                clean_arrays[uid] = finite
                bounds[uid] = (finite.min(axis=0), finite.max(axis=0))
        if uid not in bounds:
            room = rooms.by_uid.get(uid)
            if room is not None:
                g = kit.grid_size
                bounds[uid] = (np.array([0.0, -1.0, 0.0]),
                               np.array([room.size.x * g, room.size.y * g, room.size.z * g]))

    window = EditorWindow(state, {}, bounds)

    def _make_geometries(dt):
        program = window.room_program
        for uid, positions in clean_arrays.items():
            try:
                window.geometries[uid] = RoomGeometryGPU(positions, program)
            except Exception as e:
                print(f"  geometry upload failed for {uid}: {e}")
        print(f"Uploaded {len(window.geometries)} room meshes to GPU")
        pyglet.clock.unschedule(_make_geometries)

    pyglet.clock.schedule_once(_make_geometries, 0.0)
    pyglet.app.run()
    return 0
