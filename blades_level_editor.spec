# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Blades Level Editor.
# Build on Windows with:  pyinstaller blades_level_editor.spec
# (cross-compiling from Linux is not supported by PyInstaller)

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("pyglet")

a = Analysis(
    ["blades_level_tool_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="BladesLevelEditor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
