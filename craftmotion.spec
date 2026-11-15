# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds CraftMotion.exe (one-file, windowed).

    pyinstaller craftmotion.spec            # on Windows
    pyinstaller craftmotion.spec --noconfirm

Bundles: bundled fonts, the icon, numpy/Pillow, and imageio-ffmpeg's ffmpeg
binary (so video decode/encode works on a clean Windows install).
"""
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(".")

datas = [
    ("craftmotion/data", "craftmotion/data"),
    ("assets/craftmotion.ico", "assets"),
    ("samples/am_presets", "samples/am_presets"),
]
# imageio-ffmpeg ships the ffmpeg executable as package data
try:
    datas += collect_data_files("imageio_ffmpeg", include_py_files=False)
except Exception:
    pass

binaries = []
try:
    binaries += collect_dynamic_libs("imageio_ffmpeg")
except Exception:
    pass

hiddenimports = [
    "imageio_ffmpeg",
    "PIL._tkinter_finder",
    "craftmotion.cli",
    "craftmotion.mcp",
    "craftmotion.gui.app",
]

a = Analysis(
    ["craftmotion_launcher.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CraftMotion",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="assets/craftmotion.ico",
)
