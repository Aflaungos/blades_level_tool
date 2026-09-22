# Blades Level Editor

A Windows desktop tool for building **custom dungeon and town layouts** for
*The Elder Scrolls: Blades*, by visually assembling rooms from the game's
own data — with real 3D previews of every room, joint-snapped placement,
and one-click export to the game's `DungeonSettingsFixedLayout.asset`
format.

## How it works

The tool reads your exported game data (AssetRipper `ExportedProject`):

| Data | Used for |
|---|---|
| `export/gameplaymetadata/RoomsData.asset` | the room catalog: every room's UID, grid size, and its **joints** (connection points) |
| `BGS/scriptableobjects/kits/*.asset` | tilesets (Ayleid, Cave, Stone, Forest, ...) and the kit's **grid size** (1.828125 units/cell) |
| `export/resources/levels_*/..._Room.prefab` + `BGS/art/**` meshes | real room geometry for the 3D preview |

Layouts are written as Unity YAML assets that match the game's own
`MQ17_FixedLayout.asset` structure exactly: a `_roomInfos` list (room
placements) and a `_roomLinks` list (which joint connects to which).
Room/joint identities are the game's own UIDs, so the game can resolve
every reference at load time.

## Install (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python -m blades_level_tool gui
```

The tool auto-discovers the game data under `%USERPROFILE%\blades_game_data`.
You can also point it explicitly:

```
python -m blades_level_tool gui --assets C:\path\to\ExportedProject\Assets --kit Cave
python -m blades_level_tool gui --layout MyDungeon.asset
```

## Editor controls

| Input | Action |
|---|---|
| **Tab** | open the room list (↑/↓ + Enter to start placing, Esc closes) |
| **Mouse move** | move the room being placed |
| **F** | snap the placed/moving room onto the **nearest free joint** |
| **Left click** | drop the room / select a placed room |
| **G** | grab the selected room and move it |
| **R / Shift+R** | rotate 90° CW / CCW |
| **X / Delete** | delete the selected room |
| **V** | toggle joint markers (green = free, orange = already linked) |
| **Ctrl+S / Ctrl+Shift+S** | save / save-as the FixedLayout `.asset` |
| **Ctrl+L** | load a layout |
| **Right-drag / Middle-drag / Wheel** | orbit / pan / zoom camera (WASD+QE also pan) |
| **Esc** | cancel current action |

Rooms **snap together at joints**: press **F** while placing and the room
jumps onto the nearest open connection point, oriented so the two joints'
directions oppose each other — exactly the relationship the game expects
for a `_roomLinks` entry. Dropping a snapped room creates the link
automatically.

## Output

Saved `.asset` files can be dropped into an exported Unity project and
repacked, or used as reference for bundle patching. Each file contains:

- `_fixedLayoutUid` — a fresh UID for the layout
- `_kit` — the chosen kit's asset guid
- `_roomInfos` — one entry per placed room (fresh `_roomInfoUID` + the room's UID)
- `_roomLinks` — one entry per joint-to-joint connection

Use `python -m blades_level_tool validate MyDungeon.asset` to check a
layout for structural problems (unknown rooms, joints used twice,
disconnected rooms).

## Building the Windows executable

On a Windows machine with Python installed:

```
pip install -r requirements.txt
python -m blades_level_tool build-exe
```

This produces `dist\BladesLevelEditor\BladesLevelEditor.exe` (a folder with
the exe and dependencies). Share that folder; users just run the exe.

## Project layout

```
blades_level_tool/
  models.py        data classes mirroring the game's C# classes
  kit_io.py        Kit .asset parser (grid size, uid, guid)
  rooms_io.py      RoomsData.asset parser (rooms + joints)
  mesh_io.py       guid index + Unity Mesh parser + prefab geometry extraction
  joints.py        joint snapping math (matches Unity's yaw convention)
  layout_io.py     FixedLayout .asset reader/writer (reverse engineered)
  generator.py     layout validation
  editor_state.py  editor data model, snapping ops, save/load, picking math
  editor_window.py the pyglet 3D editor (OpenGL 3.3)
  cli.py           command line entry points
tests/test_smoke.py   smoke tests against the real game data
```

## Roadmap (later phases)

- NPC, quest, and interactable placement (the game's `SpawnGroup*`
  initializers in `BGS/Game/LevelGeneration/` are the reference)
- Per-room prop editing
- Direct bundle repacking
