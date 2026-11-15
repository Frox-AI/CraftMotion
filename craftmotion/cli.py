"""CraftMotion command line.

    craftmotion                  launch the GUI
    craftmotion mcp              MCP server over stdio (for AI agents)
    craftmotion serve            headless MCP over HTTP (for AI agents)
    craftmotion run <cmd> ...    execute one engine command
    craftmotion batch <file>     run a JSON list of commands
    craftmotion new|import-am|export|storyboard|doctor|version
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .core.registry import COMMANDS, CommandError, Registry, __version__


def _registry(project_path: Optional[str] = None) -> Registry:
    reg = Registry()
    if project_path:
        if not os.path.exists(project_path):
            raise SystemExit(f"project not found: {project_path}")
        reg.project = type(reg.project).load(project_path)
        reg.project.path = project_path
    return reg


def cmd_gui(args) -> int:
    from .gui.app import run
    return run(args.project)


def cmd_mcp(args) -> int:
    from .mcp import main as mcp_main
    return mcp_main()


def cmd_serve(args) -> int:
    from .core.agent_bridge import AgentBridge
    from .core.model import Project
    reg = Registry()
    if args.project:
        reg.project = Project.load(args.project)
        reg.project.path = args.project
    bridge = AgentBridge(reg, port=args.port)
    info = bridge.start()
    print(json.dumps({"ok": True, "url": info["url"], "token": info["token"],
                      "tools": len(COMMANDS), "pid": os.getpid()}, indent=2), flush=True)
    print("Headless bridge running — Ctrl+C to stop.", file=sys.stderr)
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
    return 0


def cmd_run(args) -> int:
    reg = _registry(args.project)
    payload = json.loads(args.args) if args.args else {}
    try:
        result = reg.execute(args.command, payload)
    except CommandError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, default=str))
    if args.save:
        reg.execute("project.save", {"path": args.save})
    return 0


def cmd_batch(args) -> int:
    reg = _registry(args.project)
    with open(args.file, "r", encoding="utf-8") as f:
        script = json.load(f)
    steps = script.get("commands", script) if isinstance(script, dict) else script
    for i, step in enumerate(steps):
        name = step.get("command") or step.get("cmd")
        payload = step.get("args") or {}
        try:
            reg.execute(name, payload)
            if args.verbose:
                print(f"[{i+1}/{len(steps)}] {name} ok")
        except CommandError as exc:
            print(f"[{i+1}] {name} failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                return 1
    out = script.get("save") if isinstance(script, dict) else None
    if args.save or out:
        reg.execute("project.save", {"path": args.save or out})
        print(f"saved {args.save or out}")
    print(f"done: {len(steps)} commands")
    return 0


def cmd_new(args) -> int:
    reg = Registry()
    reg.execute("project.new", {"title": args.title, "width": args.width, "height": args.height,
                                "fps": args.fps, "duration_ms": args.duration})
    reg.execute("project.save", {"path": args.path})
    print(args.path)
    return 0


def cmd_import_am(args) -> int:
    reg = _registry(args.project)
    merge = args.project is not None
    res = reg.execute("project.import_am_xml",
                      {"path": args.xml, "merge": merge, "offset_ms": args.offset})
    if args.out:
        reg.execute("project.save", {"path": args.out})
    print(json.dumps({"stats": res.get("stats"), "warnings": res.get("warnings", [])[:20]}, indent=2))
    return 0


def cmd_export(args) -> int:
    reg = _registry(args.project)
    job = reg.execute("render.video", {"path": args.out, "width": args.width or None,
                                       "height": args.height or None, "fps": args.fps,
                                       "quality": args.quality, "wait": True,
                                       "end_ms": args.duration, "audio": not args.no_audio})
    print(json.dumps(job, indent=2, default=str))
    return 0 if job.get("job", {}).get("status") == "done" else 1


def cmd_storyboard(args) -> int:
    reg = _registry(args.project)
    res = reg.execute("render.storyboard", {"count": args.count, "cols": args.cols,
                                            "tile": args.tile, "path": args.out})
    print(res["path"])
    return 0


def cmd_doctor(args) -> int:
    ok = True
    print(f"CraftMotion {__version__}")
    print(f"python           {sys.version.split()[0]}")
    try:
        import numpy
        print(f"numpy            {numpy.__version__}")
    except ImportError:
        ok = False
        print("numpy            MISSING (pip install numpy)")
    try:
        import PIL
        print(f"pillow           {PIL.__version__}")
    except ImportError:
        ok = False
        print("pillow           MISSING (pip install pillow)")
    from .core.media import ffmpeg_path
    try:
        print(f"ffmpeg           {ffmpeg_path()}")
    except Exception as exc:
        ok = False
        print(f"ffmpeg           MISSING ({exc})")
    try:
        import tkinter
        r = tkinter.Tk()
        r.destroy()
        print("tkinter          OK (display available)")
    except Exception as exc:
        print(f"tkinter          GUI unavailable here ({exc}) — CLI/MCP still work")
    print(f"commands         {len(COMMANDS)}")
    from .core import effects as FX
    print(f"effects          {len(FX.EFFECTS)}")
    print("status           " + ("READY" if ok else "NEEDS FIXES"))
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="craftmotion", description="CraftMotion Studio")
    parser.add_argument("--version", action="version", version=f"CraftMotion {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("gui", help="launch the editor window (default)")
    p.add_argument("project", nargs="?")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("mcp", help="MCP server over stdio")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("serve", help="headless MCP bridge over HTTP")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--project")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run", help="execute one engine command")
    p.add_argument("command")
    p.add_argument("--args", help="JSON object of arguments", default=None)
    p.add_argument("--project")
    p.add_argument("--save", help="save project to this path afterwards")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("batch", help="run a JSON batch script")
    p.add_argument("file")
    p.add_argument("--project")
    p.add_argument("--save")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("new", help="create a project file")
    p.add_argument("path")
    p.add_argument("--title", default="Untitled")
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--duration", type=int, default=15000)
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("import-am", help="import Alight Motion XML")
    p.add_argument("xml")
    p.add_argument("--project", help="merge into this project instead of creating a new one")
    p.add_argument("--out", help="save .cmproj here")
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=cmd_import_am)

    p = sub.add_parser("export", help="render a project to video")
    p.add_argument("project")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=float)
    p.add_argument("--quality", default="good", choices=["draft", "good", "high"])
    p.add_argument("--duration", type=int, help="limit to N ms")
    p.add_argument("--no-audio", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("storyboard", help="render a storyboard sheet")
    p.add_argument("project")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--count", type=int, default=9)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile", type=int, default=320)
    p.set_defaults(func=cmd_storyboard)

    p = sub.add_parser("doctor", help="environment check")
    p.set_defaults(func=cmd_doctor)

    argv = argv if argv is not None else sys.argv[1:]
    if any(a in ("--version", "-V", "-h", "--help") for a in argv):
        return parser.parse_args(argv) and 0
    if not argv or argv[0] not in ("gui", "mcp", "serve", "run", "batch", "new", "import-am",
                                    "export", "storyboard", "doctor"):
        # bare invocation (or project path) opens the GUI
        args = parser.parse_args(["gui"] + [a for a in argv if not a.startswith("-")])
    else:
        args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
