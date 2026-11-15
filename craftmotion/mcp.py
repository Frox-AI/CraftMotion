"""MCP (Model Context Protocol) server for CraftMotion — JSON-RPC 2.0 over stdio.

Every registry command is exposed as an MCP tool (same complete toolset the GUI
uses), plus resources for project JSON, commands, effects and the timeline, and
ready-made prompts that teach an agent a full human-style editing workflow.

Run standalone:   python -m craftmotion.mcp
Wire into an agent:  {"mcpServers": {"craftmotion": {"command": "craftmotion",
                      "args": ["mcp"]}}}
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from .core.registry import COMMANDS, CommandError, Registry, __version__, resolve_command

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "craftmotion"
SERVER_TITLE = "CraftMotion Studio"


class MCPServer:
    def __init__(self, registry: Registry):
        self.reg = registry
        self.tools = self._build_tools()
        self._out_lock = threading.Lock()

    # ---------------------------------------------------------- tools
    def _build_tools(self) -> List[dict]:
        tools = []
        for name, c in sorted(COMMANDS.items()):
            props: Dict[str, Any] = {}
            required = []
            for arg, spec in c.args.items():
                js = {"int": "integer", "float": "number", "bool": "boolean",
                      "dict": "object", "list": "array", "str": "string"}.get(spec.get("type", "str"), "string")
                prop: Dict[str, Any] = {"description": spec.get("desc", "")}
                if js == "array":
                    prop["type"] = "array"
                    prop["items"] = {"type": "number"}
                else:
                    prop["type"] = js
                if "default" in spec:
                    prop["default"] = spec["default"]
                if spec.get("enum"):
                    prop["enum"] = spec["enum"]
                props[arg] = prop
                if spec.get("required"):
                    required.append(arg)
            tools.append({
                "name": name.replace(".", "_"),
                "title": f"{c.group}: {name}",
                "description": c.description,
                "inputSchema": {"type": "object", "properties": props,
                                **({"required": required} if required else {})},
                "annotations": {"readOnlyHint": not c.mutates, "destructiveHint": c.name.endswith("delete")},
                "_command": name,
            })
        return tools

    def _tool_by_snake(self, name: str) -> Optional[dict]:
        for t in self.tools:
            if t["name"] == name:
                return t
        cmd = resolve_command(name)
        if cmd:
            for t in self.tools:
                if t["_command"] == cmd:
                    return t
        return None

    # ---------------------------------------------------------- resources
    def resources(self) -> List[dict]:
        return [
            {"uri": "craftmotion://project", "name": "Project", "mimeType": "application/json",
             "description": "Full project JSON (settings, media, layers, keyframes, effects)."},
            {"uri": "craftmotion://timeline", "name": "Timeline", "mimeType": "application/json",
             "description": "Compact timeline: layers with timing, transform, keyframes, effects."},
            {"uri": "craftmotion://commands", "name": "Command reference", "mimeType": "application/json",
             "description": "Every command/tool with its argument schema."},
            {"uri": "craftmotion://effects", "name": "Effect catalogue", "mimeType": "application/json",
             "description": "All effects with parameters, ranges and AM ids."},
            {"uri": "craftmotion://media", "name": "Media bin", "mimeType": "application/json",
             "description": "Imported media with metadata and missing-file flags."},
            {"uri": "craftmotion://am-format", "name": "Alight Motion XML format", "mimeType": "text/markdown",
             "description": "Summary of the AM scene/preset XML format supported by the importer."},
        ]

    def read_resource(self, uri: str) -> dict:
        reg = self.reg
        if uri == "craftmotion://project":
            body = json.dumps(reg.project.data, indent=1)
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": body}]}
        if uri == "craftmotion://timeline":
            layers = []
            for l in reg.project.layers:
                layers.append({
                    "id": l["id"], "name": l.get("name"), "kind": l.get("kind"),
                    "start": l.get("start"), "end": l.get("end"),
                    "transform": l.get("transform"), "keys": l.get("keys"),
                    "effects": [{"type": e.get("type"), "params": e.get("params"),
                                 "keys": e.get("keys")} for e in l.get("effects", [])],
                    "parent": l.get("parent"),
                })
            body = json.dumps({"settings": reg.project.settings, "layers": layers}, indent=1)
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": body}]}
        if uri == "craftmotion://commands":
            return {"contents": [{"uri": uri, "mimeType": "application/json",
                                  "text": json.dumps(self._command_catalogue(), indent=1)}]}
        if uri == "craftmotion://effects":
            from .core import effects as FX
            return {"contents": [{"uri": uri, "mimeType": "application/json",
                                  "text": json.dumps(FX.list_effects(), indent=1)}]}
        if uri == "craftmotion://media":
            return {"contents": [{"uri": uri, "mimeType": "application/json",
                                  "text": json.dumps(reg.project.media, indent=1)}]}
        if uri == "craftmotion://am-format":
            return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": AM_FORMAT_DOC}]}
        raise CommandError(f"unknown resource {uri}")

    def _command_catalogue(self) -> List[dict]:
        return [{"command": t["_command"], "tool": t["name"], "description": t["description"],
                 "args": t["inputSchema"].get("properties", {})} for t in self.tools]

    # ---------------------------------------------------------- prompts
    def prompts(self) -> List[dict]:
        return [
            {"name": "edit_like_a_human",
             "description": "Step-by-step workflow for producing an edit from footage to final MP4.",
             "arguments": [{"name": "goal", "description": "What the edit should achieve", "required": True}]},
            {"name": "apply_am_preset",
             "description": "Import an Alight Motion XML preset and merge it onto existing footage.",
             "arguments": [{"name": "xml_path", "description": "Path to the preset XML", "required": True},
                           {"name": "offset_ms", "description": "Where to place it", "required": False}]},
            {"name": "review_render",
             "description": "Render a storyboard, inspect it, and fix issues in the timeline.",
             "arguments": [{"name": "count", "description": "Frames to sample", "required": False}]},
        ]

    def get_prompt(self, name: str, args: dict) -> dict:
        if name == "edit_like_a_human":
            text = EDIT_PROMPT.format(goal=args.get("goal", "a punchy social edit"))
        elif name == "apply_am_preset":
            text = AM_PROMPT.format(xml_path=args.get("xml_path", "preset.xml"),
                                    offset_ms=args.get("offset_ms", "0"))
        elif name == "review_render":
            text = REVIEW_PROMPT.format(count=args.get("count", "9"))
        else:
            raise CommandError(f"unknown prompt {name}")
        return {"description": name, "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}

    # ---------------------------------------------------------- dispatch
    def handle(self, msg: dict) -> Optional[dict]:
        method = msg.get("method", "")
        mid = msg.get("id")
        params = msg.get("params") or {}
        is_notification = "id" not in msg

        def ok(result: Any) -> Optional[dict]:
            return None if is_notification else {"jsonrpc": "2.0", "id": mid, "result": result}

        def err(code: int, message: str) -> Optional[dict]:
            return None if is_notification else {"jsonrpc": "2.0", "id": mid,
                                                 "error": {"code": code, "message": message}}

        try:
            if method == "initialize":
                return ok({
                    "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {"name": SERVER_NAME, "title": SERVER_TITLE, "version": __version__},
                    "instructions": SERVER_INSTRUCTIONS,
                })
            if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
                return None
            if method == "ping":
                return ok({})
            if method == "tools/list":
                tools = [{k: v for k, v in t.items() if not k.startswith("_")} for t in self.tools]
                return ok({"tools": tools})
            if method == "tools/call":
                name = params.get("name", "")
                args = params.get("arguments") or {}
                tool = self._tool_by_snake(name)
                if tool is None:
                    return err(-32602, f"unknown tool '{name}'")
                try:
                    result = self.reg.execute(tool["_command"], args)
                    return ok({"content": [{"type": "text", "text": json.dumps(result, indent=1, default=str)}],
                               "structuredContent": result, "isError": False})
                except CommandError as exc:
                    return ok({"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True})
            if method == "resources/list":
                return ok({"resources": self.resources()})
            if method == "resources/read":
                return ok(self.read_resource(params.get("uri", "")))
            if method == "prompts/list":
                return ok({"prompts": self.prompts()})
            if method == "prompts/get":
                return ok(self.get_prompt(params.get("name", ""), params.get("arguments") or {}))
            if method == "logging/setLevel":
                return ok({})
            if method == "completion/complete":
                return ok({"completion": {"values": []}})
            return err(-32601, f"method not found: {method}")
        except CommandError as exc:
            return err(-32000, str(exc))
        except Exception as exc:  # keep the server alive
            return err(-32603, f"internal error: {exc}")

    # ---------------------------------------------------------- stdio loop
    def serve_stdio(self) -> None:
        stdin = sys.stdin.buffer
        stdout = sys.stdout.buffer
        while True:
            line = stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                with self._out_lock:
                    stdout.write((json.dumps({"jsonrpc": "2.0", "id": None,
                                              "error": {"code": -32700, "message": "parse error"}}) + "\n").encode())
                    stdout.flush()
                continue
            batch = msg if isinstance(msg, list) else [msg]
            responses = []
            for m in batch:
                resp = self.handle(m)
                if resp is not None:
                    responses.append(resp)
            if responses:
                payload = responses if isinstance(msg, list) else responses[0]
                with self._out_lock:
                    stdout.write((json.dumps(payload) + "\n").encode("utf-8"))
                    stdout.flush()


# ------------------------------------------------------------------ docs

SERVER_INSTRUCTIONS = """You are connected to CraftMotion Studio, a motion-graphics and video editor.
You have the same complete toolset a human editor has in the GUI: media, layers, transforms,
keyframes with easing, 60+ effects, transitions, track mattes, adjustment layers, Alight Motion
XML import/export, stills, storyboards and MP4/WebM/GIF/sequence export.

Workflow tips:
1. media_import or media_import_folder first; media_list tells you ids and durations.
2. layer_add with kind video/image/text/shape/solid/adjustment/audio/null.
3. keyframe_set animates pos/anchor/scale/rot/opacity/color/volume; easing names:
   linear, ease, ease_in, ease_out, ease_in_out, expo_out, back_out, hold, or
   "cubicBezier x1 y1 x2 y2".
4. effect_types → effect_add → effect_set_param (pass t to keyframe a param).
5. render_storyboard to check your work visually, render_preview for one frame,
   render_video for the final file (async job; poll job_status).
6. project_import_am_xml imports real Alight Motion presets/projects; merge=true
   applies a preset on top of the current timeline.
Always inspect project_info after big edits and keep layer counts sane."""

AM_FORMAT_DOC = """# Alight Motion XML — what CraftMotion imports

`<scene width height totalTime fps bgcolor>` containing `media*`, `bookmark*`, then elements.

| AM element | CraftMotion layer |
|---|---|
| `<shape s=".rect|.circle|.star|.poly|.triangle|.roundrect|.line|.quad|.teardrop">` | shape |
| `<text size font wrapWidth align><content>` | text |
| `<drawing><stroke points="x,y,p;...">` | shape (path) |
| `<embedScene><scene>` | flattened group under a null |
| `<audio src>` | audio |
| `<camera>` / `<nullobj>` | null (transform rig) |

- `<transform>` children `location(x,y,z) pivot scale(x,y) skew rotation opacity` are keyables:
  static `value="..."`, or `<kf t="seconds" v="..." e="cubicBezier 0.4 0 0.2 1"/>`.
- `<effect id="com.alightcreative.effects.*">` maps to CraftMotion effects where an
  equivalent exists (blur, glow, shake, colour grading, chromatic aberration, etc.).
  Unknown ids are kept as metadata and reported in warnings.
- `<gradient startColor endColor start end type>` becomes layer.color → color2.
- `<border>`, `<shadow>` become stroke and drop-shadow effects.
- `parent="<am id>"` is remapped to the new layer id automatically.
- Media: presets reference `am-internal:///<sig>.<ext>`; place the files next to the XML
  (any subfolder is scanned) or call media_relink afterwards.
"""

EDIT_PROMPT = """Produce {goal} using CraftMotion.

1. Inspect: app_capabilities, effect_types, project_info.
2. Import footage: media_import_folder (or media_import). Check fps/size via media_list.
3. Decide a canvas with project_update_settings (e.g. 1080x1920 @ 30fps for Reels).
4. Build the edit bottom-up: solid background → footage layers → titles → adjustment layer.
   Use layer_add and layer_set_transform; position via pos [x,y] in canvas pixels.
5. Animate: keyframe_set on pos/scale/rot/opacity, pick easing per segment. Keep motion
   motivated: ease_out for entrances, ease_in for exits.
6. Grade: add an adjustment layer on top, effect_apply_look cinematic_teal (or build a
   custom stack with effect_add + effect_set_param).
7. Check: render_storyboard count=9, inspect the posted image, fix anything off-centre,
   clipped or illegible. Re-check after fixes.
8. Export: render_video quality=high, then job_status until done. Report the output path.
"""

AM_PROMPT = """Import the Alight Motion XML preset at {xml_path}.

1. project_validate_am_xml on the path; report issues if invalid.
2. project_import_am_xml with merge=true and offset_ms={offset_ms} so it lands on the
   existing timeline (or merge=false to inspect it alone first).
3. Read the warnings: missing media (media_relink), substituted fonts, approximated shapes.
4. render_storyboard to see the preset, then re-parent/retime layers if the agent needs to
   retarget the preset onto different footage (layer_set_parent, layer_trim).
"""

REVIEW_PROMPT = """Review the current edit like a demanding reviewer.

1. render_storyboard count={count} and look at the image.
2. Check: composition balance, text legibility at phone size, safe margins, colour
   consistency between shots, pacing (anything static longer than ~4s needs motion or a cut).
3. Fix issues with layer_update / layer_set_transform / keyframe_set / effect_set_param.
4. Re-render the storyboard until it looks deliberate, then summarise what you changed.
"""


def main(argv: Optional[List[str]] = None) -> int:
    reg = Registry()
    server = MCPServer(reg)
    # Allow the GUI/host to point the server at a project file
    proj = os.environ.get("CRAFTMOTION_PROJECT")
    if proj and os.path.exists(proj):
        try:
            reg.project = type(reg.project).load(proj)
            reg.project.path = proj
        except Exception:
            pass
    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
