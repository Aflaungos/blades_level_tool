"""Smoke tests for blades_level_tool using the real game data.

Run:  python tests/test_smoke.py
Skips automatically if the game data isn't present.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blades_level_tool.kit_io import load_kits
from blades_level_tool.paths import find_assets_root
from blades_level_tool.rooms_io import load_rooms

ASSETS = find_assets_root()


@unittest.skipIf(ASSETS is None, "game data not found")
class TestGameData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rooms = load_rooms(ASSETS / "export/gameplaymetadata/RoomsData.asset")
        cls.kits = load_kits(ASSETS / "BGS/scriptableobjects/kits")

    def test_rooms_loaded(self):
        self.assertGreater(len(self.rooms), 200)

    def test_kits_loaded(self):
        self.assertGreater(len(self.kits), 10)
        # dungeon kits (the ones with joint-connected rooms) use the standard grid
        ayleid = self.kits["63d3a7591920a734e92cddaf5e664e8d"]
        self.assertAlmostEqual(ayleid.grid_size, 1.828125, places=6)
        for guid, kit in self.kits.items():
            if any(r.joints for r in self.rooms.by_kit.get(guid, [])):
                self.assertGreater(kit.grid_size, 0, f"{kit.name} has joint rooms but no grid size")

    def test_ayleid_rooms_have_joints(self):
        kit = self.kits["63d3a7591920a734e92cddaf5e664e8d"]
        rooms = self.rooms.by_kit[kit.guid]
        self.assertTrue(all(r.joints for r in rooms))


@unittest.skipIf(ASSETS is None, "game data not found")
class TestEditorFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rooms = load_rooms(ASSETS / "export/gameplaymetadata/RoomsData.asset")
        cls.kits = load_kits(ASSETS / "BGS/scriptableobjects/kits")
        cls.kit = cls.kits["63d3a7591920a734e92cddaf5e664e8d"]

    def _state(self):
        from blades_level_tool.editor_state import EditorState
        return EditorState(self.rooms, self.kit, "SmokeTest")

    def _hall(self, st):
        return next(r for r in st.rooms_for_kit() if r.name.endswith("Hall_00_Room"))

    def test_snap_and_link(self):
        from blades_level_tool.editor_state import Placement
        st = self._state()
        hall = self._hall(st)
        p1 = Placement(hall, (0, 0, 0), 0)
        st.placements.append(p1)
        entrance = next(r for r in st.rooms_for_kit() if r.name.endswith("EntranceRoom_04_Room"))
        pos, rot, joint_uid = st.snap_placement(p1, hall.joints[1].uid, entrance)
        p2 = Placement(entrance, pos, rot)
        st.placements.append(p2)
        self.assertIsNone(st.add_link(p1, hall.joints[1].uid, p2, joint_uid))
        self.assertEqual(len(st.links), 1)

    def test_save_load_roundtrip(self):
        from blades_level_tool.editor_state import Placement
        st = self._state()
        hall = self._hall(st)
        p1 = Placement(hall, (0, 0, 0), 0)
        st.placements.append(p1)
        entrance = next(r for r in st.rooms_for_kit() if r.name.endswith("EntranceRoom_04_Room"))
        pos, rot, joint_uid = st.snap_placement(p1, hall.joints[1].uid, entrance)
        p2 = Placement(entrance, pos, rot)
        st.placements.append(p2)
        st.add_link(p1, hall.joints[1].uid, p2, joint_uid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Smoke.asset"
            problems = st.save(path)
            self.assertEqual(problems, [])
            text = path.read_text()
            self.assertIn("m_Script: {fileID: 11500000, guid: b67e675204aed0ed2ec3de54bbc173e4, type: 3}", text)
            st2 = self._state()
            st2.load(path)
            self.assertEqual(len(st2.placements), 2)
            self.assertEqual(len(st2.links), 1)
            # the reflow on load should reproduce the same arrangement we saved
            self.assertAlmostEqual(st2.placements[1].pos[0], pos[0], places=4)
            self.assertAlmostEqual(st2.placements[1].pos[2], pos[2], places=4)
            self.assertEqual(st2.placements[1].rot, rot)

    def test_validate_catches_double_joint_use(self):
        from blades_level_tool.editor_state import Placement
        st = self._state()
        hall = self._hall(st)
        p1 = Placement(hall, (0, 0, 0), 0)
        p2 = Placement(hall, (50, 0, 0), 0)
        st.placements += [p1, p2]
        # link the same joint twice on p1
        err = st.add_link(p1, hall.joints[0].uid, p2, hall.joints[1].uid)
        self.assertIsNone(err)
        st.links.append((p1, hall.joints[0].uid, p2, hall.joints[0].uid))
        layout = st.to_fixed_layout()
        from blades_level_tool.generator import validate
        problems = validate(layout, self.rooms.by_uid)
        self.assertTrue(any("used in 2 links" in p for p in problems))


class TestJointMath(unittest.TestCase):
    def test_yaw_matches_unity(self):
        from blades_level_tool.joints import rotate_xz
        # Unity +Y yaw of +90 degrees maps +Z -> +X
        self.assertEqual(rotate_xz(0, 1, 1), (1.0, 0.0))

    def test_snap_opposes_directions(self):
        from blades_level_tool.joints import snap_transform, joint_world
        from blades_level_tool.models import Joint, Vec3
        grid = 1.828125
        ja = Joint(uid="a", type=1, position=Vec3(8, 0, 2), direction=(1, 0))
        jb = Joint(uid="b", type=1, position=Vec3(0, 0, 2), direction=(-1, 0))
        pos, rot = snap_transform(grid, (3, 0, 4), ja, 2, jb)
        tpos, tdir = joint_world(ja, grid, (3, 0, 4), 2)
        mpos, mdir = joint_world(jb, grid, pos, rot)
        self.assertAlmostEqual(tpos[0], mpos[0], places=6)
        self.assertAlmostEqual(tpos[2], mpos[2], places=6)
        self.assertAlmostEqual(tdir[0] + mdir[0], 0.0, places=6)
        self.assertAlmostEqual(tdir[1] + mdir[1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
