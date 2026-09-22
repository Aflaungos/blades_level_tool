"""Command-line entry points for blades_level_tool.

`python -m blades_level_tool gui`      open the 3D layout editor
`python -m blades_level_tool rooms`    list rooms in a kit
`python -m blades_level_tool validate` validate a FixedLayout .asset
`python -m blades_level_tool build-exe` build a standalone Windows .exe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .kit_io import load_kits
from .paths import find_assets_root
from .rooms_io import load_rooms


def _pick_kit(kits, guid=None, name=None):
    if guid and guid in kits:
        return kits[guid]
    if name:
        for k in kits.values():
            if k.name.lower() == name.lower():
                return k
    return None


def cmd_rooms(args):
    assets = find_assets_root(args.assets)
    if not assets:
        print("Could not locate game data. Use --assets <path to ExportedProject/Assets>")
        return 1
    rooms = load_rooms(assets / "export/gameplaymetadata/RoomsData.asset")
    kits = load_kits(assets / "BGS/scriptableobjects/kits")
    kit = _pick_kit(kits, args.kit_guid, args.kit)
    if kit is None:
        print(f"{len(kits)} kits available:")
        for guid, k in kits.items():
            count = len(rooms.by_kit.get(guid, []))
            print(f"  {k.name:20s} guid={guid} rooms={count}")
        return 0
    kit_rooms = rooms.by_kit.get(kit.guid, [])
    print(f"Kit '{kit.name}' ({len(kit_rooms)} rooms, grid={kit.grid_size}):")
    for r in kit_rooms:
        print(f"  {r.name:60s} size=({r.size.x},{r.size.y},{r.size.z}) joints={len(r.joints)}")
    return 0


def cmd_validate(args):
    from .editor_state import EditorState
    assets = find_assets_root(args.assets)
    if not assets:
        print("Could not locate game data. Use --assets <path>")
        return 1
    rooms = load_rooms(assets / "export/gameplaymetadata/RoomsData.asset")
    kits = load_kits(assets / "BGS/scriptableobjects/kits")
    state = EditorState(rooms, next(iter(kits.values())))
    state.load(Path(args.layout))
    # restore the layout's own kit for validation
    from .layout_io import parse_fixed_layout
    layout = parse_fixed_layout(Path(args.layout))
    kit = _pick_kit(kits, layout.kit_guid)
    if kit:
        state.kit = kit
    problems = [
        p for p in __import__("blades_level_tool.generator", fromlist=["validate"]).validate(
            __import__("blades_level_tool.layout_io", fromlist=["parse_fixed_layout"]).parse_fixed_layout(Path(args.layout)),
            rooms.by_uid)
    ]
    print(f"{Path(args.layout).name}: {len(state.placements)} rooms, "
          f"{len(state.links)} links, kit={kit.name if kit else layout.kit_guid}")
    if problems:
        print("Problems:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: no structural problems found")
    return 0


def cmd_gui(args):
    from .editor_window import run_editor
    return run_editor(assets=args.assets, kit_name=args.kit, kit_guid=args.kit_guid,
                      layout=Path(args.layout) if args.layout else None)


def cmd_build_exe(args):
    """Build a Windows executable with PyInstaller (run on the target OS)."""
    import subprocess
    root = Path(__file__).resolve().parent.parent
    spec = root / "blades_level_editor.spec"
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if args.onefile:
        cmd.append("--onefile")
    cmd.append(str(spec))
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode == 0:
        print("\nBuilt: dist/BladesLevelEditor.exe (or dist/BladesLevelEditor/)")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="blades_level_tool",
                                     description="Elder Scrolls Blades layout editor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--assets", help="Path to ExportedProject/Assets (default: auto-discover)")
        p.add_argument("--kit", help="Kit name (e.g. Ayleid)")
        p.add_argument("--kit-guid", help="Kit asset guid")

    p_rooms = sub.add_parser("rooms", help="List rooms per kit")
    add_common(p_rooms)

    p_gui = sub.add_parser("gui", help="Open the 3D editor")
    add_common(p_gui)
    p_gui.add_argument("--layout", help="FixedLayout .asset to open")

    p_val = sub.add_parser("validate", help="Validate a FixedLayout .asset")
    add_common(p_val)
    p_val.add_argument("layout")

    p_exe = sub.add_parser("build-exe", help="Build a Windows .exe with PyInstaller")
    p_exe.add_argument("--onefile", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "rooms":
        return cmd_rooms(args)
    if args.cmd == "gui":
        return cmd_gui(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "build-exe":
        return cmd_build_exe(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
