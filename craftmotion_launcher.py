"""Frozen-app entry point (PyInstaller).

    CraftMotion.exe                 → GUI
    CraftMotion.exe mcp             → MCP server on stdio (agent mode)
    CraftMotion.exe serve --port N  → headless HTTP bridge
    CraftMotion.exe run ...         → single engine command
"""
import sys

from craftmotion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
