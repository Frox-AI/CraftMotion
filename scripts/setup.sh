#!/usr/bin/env bash
# Dev environment for CraftMotion on Linux/macOS (Windows: same pip command).
set -e
python3 -m pip install --upgrade pip
python3 -m pip install numpy pillow imageio-ffmpeg
echo "Ready. Run: python -m craftmotion"
