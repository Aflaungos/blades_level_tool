"""GUI launcher for the Blades Level Editor (PyInstaller entry point).

Usage:
    BladesLevelEditor.exe                 (auto-discovers game data)
    BladesLevelEditor.exe --assets PATH   (explicit ExportedProject/Assets)
    BladesLevelEditor.exe --kit Cave      (pick a kit by name)
    BladesLevelEditor.exe --layout X.asset
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    # When frozen, --layout drag&drop args may arrive after the exe name;
    # parse everything after argv[0] as CLI args for the "gui" command.
    args = ["gui"] + sys.argv[1:]
    from blades_level_tool.cli import main as cli_main
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
