"""One command surface for everything.

The GUI, the CLI and the MCP server all execute the *same* registry, so a human
and an AI agent have literally every feature available, with the same
behaviour and the same undo history.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from . import effects as FX
from .am_xml import export_am_scene, import_am_scene, validate_am_xml
from .media import MediaManager, import_media_file, probe_media, ffmpeg_path
from .model import BLEND_MODES, LAYER_KINDS, Project, layer_summary
from .presets import BUILTIN_LOOKS
from .render import Renderer
from .util import (clamp, fmt_time, kf_add, kf_eval, kf_find, kf_remove, now_ms,
                   parse_color, parse_vec)

__version__ = "1.1.0"

APP_HOME = os.environ.get("CRAFTMOTION_HOME") or os.path.join(os.path.expanduser("~"), ".craftmotion")


class CommandError(Exception):
    pass


@dataclass
class Command:
    name: str
    group: str
    description: str
    args: Dict[str, dict]
    func: Callable
    mutates: bool = False


COMMANDS: Dict[str, Command] = {}


def cmd(name: str, group: str, description: str, args: Optional[Dict[str, dict]] = None,
        mutates: bool = False):
    def deco(fn):
        COMMANDS[name] = Command(name, group, description, args or {}, fn, mutates)
        return fn
    return deco


def A(type_: str, desc: str, required: bool = False, **extra) -> dict:
    d = {"type": type_, "desc": desc, "required": required}
    d.update(extra)
    return d


# ------------------------------------------------------------------ jobs

class Job:
    def __init__(self, jid: str, kind: str, label: str):
        self.id = jid
        self.kind = kind
        self.label = label
        self.status = "running"
        self.progress = 0.0
        self.message = "Starting"
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = now_ms()
        self.finished: Optional[int] = None
        self.cancel_flag = False

    def update(self, p: float, msg: str = "") -> None:
        self.progress = float(clamp(p, 0, 1))
        if msg:
            self.message = msg

    def to_dict(self) -> dict:
        return {"job_id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
                "progress": round(self.progress, 3), "message": self.message,
                "result": self.result, "error": self.error,
                "created": self.created, "finished": self.finished}


# ------------------------------------------------------------------ registry

class Registry:
    def __init__(self, project: Optional[Project] = None, media: Optional[MediaManager] = None):
        self.project = project or Project()
        self.media = media or MediaManager()
        self.lock = threading.RLock()
        self._undo: List[str] = []
        self._redo: List[str] = []
        self.jobs: Dict[str, Job] = {}
        self._job_seq = 0
        self.listeners: List[Callable[[dict], None]] = []
        self.user_presets_path = os.path.join(APP_HOME, "presets.json")

    # ---------------------------------------------------------- events
    def on_change(self, fn: Callable[[dict], None]) -> None:
        self.listeners.append(fn)

    def emit(self, event: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(event)
            except Exception:
                pass

    # ---------------------------------------------------------- undo
    def _snapshot(self) -> str:
        return json.dumps(self.project.data)

    def push_undo(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > 80:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self.project.data = json.loads(self._undo.pop())
        self.project._normalize()
        self.emit({"type": "changed", "command": "undo"})
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self.project.data = json.loads(self._redo.pop())
        self.project._normalize()
        self.emit({"type": "changed", "command": "redo"})
        return True

    # ---------------------------------------------------------- execute
    def execute(self, name: str, args: Optional[dict] = None) -> dict:
        args = dict(args or {})
        c = COMMANDS.get(name)
        if c is None:
            raise CommandError(f"unknown command '{name}' (see app.commands)")
        # coerce + validate
        coerced: Dict[str, Any] = {}
        for key, spec in c.args.items():
            if key in args and args[key] is not None:
                coerced[key] = _coerce(args[key], spec.get("type", "str"), key)
            elif spec.get("required"):
                raise CommandError(f"missing required argument '{key}' for {name}")
            elif "default" in spec:
                coerced[key] = spec["default"]
        extras = {k: v for k, v in args.items() if k not in c.args}
        if extras:
            coerced.update(extras)
        with self.lock:
            if c.mutates:
                self.push_undo()
            try:
                result = c.func(self, coerced) or {}
            except CommandError:
                raise
            except KeyError as exc:
                raise CommandError(str(exc).strip("'\"")) from exc
            except (ValueError, TypeError) as exc:
                raise CommandError(f"{name}: {exc}") from exc
            if c.mutates:
                self.project.touch()
                self.emit({"type": "changed", "command": name})
            out = {"ok": True, "command": name}
            out.update(result)
            return out

    def tool_definitions(self) -> List[dict]:
        tools = []
        for name, c in sorted(COMMANDS.items()):
            props = {}
            required = []
            for arg, spec in c.args.items():
                t = spec.get("type", "str")
                js = {"int": "integer", "float": "number", "bool": "boolean",
                      "dict": "object", "list": "array", "str": "string"}.get(t, "string")
                p: Dict[str, Any] = {"type": js, "description": spec.get("desc", "")}
                if "default" in spec:
                    p["default"] = spec["default"]
                if spec.get("enum"):
                    p["enum"] = spec["enum"]
                props[arg] = p
                if spec.get("required"):
                    required.append(arg)
            tools.append({
                "name": name.replace(".", "_"),
                "description": f"[{c.group}] {c.description}",
                "inputSchema": {"type": "object", "properties": props,
                                **({"required": required} if required else {}),
                                "additionalProperties": False},
                "_command": name,
            })
        return tools

    # ---------------------------------------------------------- jobs
    def start_job(self, kind: str, label: str, worker: Callable[[Job], Optional[dict]]) -> str:
        self._job_seq += 1
        jid = f"job{self._job_seq:04d}"
        job = Job(jid, kind, label)
        self.jobs[jid] = job

        def run():
            try:
                job.result = worker(job) or {}
                if job.status == "running":
                    job.status = "cancelled" if job.cancel_flag else "done"
            except Exception as exc:
                job.status = "error"
                job.error = str(exc)
                job.message = "Failed"
            finally:
                job.finished = now_ms()
                job.update(1.0 if job.status == "done" else job.progress, job.message)
                self.emit({"type": "job", "job": job.to_dict()})

        threading.Thread(target=run, daemon=True).start()
        return jid

    # ---------------------------------------------------------- helpers
    def _layer(self, lid) -> dict:
        return self.project.require_layer(int(lid))

    def _media(self, mid) -> dict:
        item = self.project.get_media(mid)
        if item is None:
            raise CommandError(f"media {mid} not found")
        return item

    def _media_manager(self) -> MediaManager:
        return self.media

    def export_dir(self) -> str:
        d = os.path.join(APP_HOME, "exports")
        os.makedirs(d, exist_ok=True)
        return d

    def renderer(self, mode: str = "preview") -> Renderer:
        return Renderer(self.project, self.media, mode=mode)

    def save_thumbnail(self, img: Image.Image, path: Optional[str] = None) -> str:
        if path is None:
            d = os.path.join(APP_HOME, "frames")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"frame_{int(time.time()*1000)}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        img.save(path)
        return os.path.abspath(path)


def _coerce(value: Any, type_: str, name: str) -> Any:
    try:
        if type_ == "int":
            return int(float(value))
        if type_ == "float":
            return float(value)
        if type_ == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if type_ == "dict":
            if isinstance(value, str):
                return json.loads(value)
            return dict(value)
        if type_ == "list":
            if isinstance(value, str):
                t = value.strip()
                if t.startswith("["):
                    return json.loads(t)
                parts = [x for x in t.replace(";", ",").split(",") if x.strip() != ""]
                try:
                    nums = [float(x) for x in parts]
                    return nums
                except ValueError:
                    return [t]
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]
        return str(value) if value is not None else ""
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CommandError(f"bad value for '{name}' ({type_}): {value!r}") from exc


# ================================================================== APP

@cmd("app.info", "App", "Version, engine status and environment details.")
def c_app_info(reg: Registry, a: dict):
    return {
        "name": "CraftMotion",
        "version": __version__,
        "engine": "python-compositor",
        "ffmpeg": _safe_ffmpeg(),
        "commands": len(COMMANDS),
        "effects": len(FX.EFFECTS),
        "project": reg.project.summary(include_layers=False),
        "capabilities": ["compositing", "keyframes", "effects", "am_xml_import", "am_xml_export",
                          "video_edit", "audio_mix", "mcp", "headless"],
    }


def _safe_ffmpeg() -> str:
    try:
        return ffmpeg_path()
    except Exception as exc:
        return f"missing ({exc})"


@cmd("app.commands", "App", "List every command with its argument schema (full feature surface).")
def c_app_commands(reg: Registry, a: dict):
    out = []
    for name, c in sorted(COMMANDS.items()):
        out.append({"command": name, "group": c.group, "description": c.description, "args": c.args})
    return {"count": len(out), "commands": out}


@cmd("app.capabilities", "App", "Machine-readable capability map for agents.")
def c_app_capabilities(reg: Registry, a: dict):
    groups: Dict[str, List[str]] = {}
    for name, c in COMMANDS.items():
        groups.setdefault(c.group, []).append(name)
    return {"groups": groups, "effects": FX.list_effects(), "looks": list(BUILTIN_LOOKS.keys()),
            "blend_modes": BLEND_MODES, "layer_kinds": LAYER_KINDS}


# ================================================================== PROJECT

@cmd("project.new", "Project", "Create a new empty project.", {
    "title": A("str", "Project title", default="Untitled"),
    "width": A("int", "Canvas width in px", default=1080),
    "height": A("int", "Canvas height in px", default=1920),
    "fps": A("float", "Frame rate", default=30.0),
    "duration_ms": A("int", "Duration in milliseconds", default=15000),
    "bg": A("str", "Background colour #AARRGGBB", default="#FF000000"),
}, mutates=True)
def c_project_new(reg: Registry, a: dict):
    p = Project()
    p.data["title"] = a["title"]
    p.settings.update({"width": a["width"], "height": a["height"], "fps": a["fps"],
                       "duration": a["duration_ms"], "bg": a["bg"]})
    p._normalize()
    reg.project = p
    return {"project": p.summary(include_layers=False)}


@cmd("project.info", "Project", "Project settings + layer list.", {
    "include_layers": A("bool", "Include layer summaries", default=True),
})
def c_project_info(reg: Registry, a: dict):
    return {"project": reg.project.summary(include_layers=a["include_layers"])}


@cmd("project.update_settings", "Project", "Change canvas size, fps, duration or background.", {
    "width": A("int", "Canvas width"),
    "height": A("int", "Canvas height"),
    "fps": A("float", "Frame rate"),
    "duration_ms": A("int", "Duration in ms"),
    "bg": A("str", "Background #AARRGGBB"),
    "title": A("str", "Project title"),
}, mutates=True)
def c_project_settings(reg: Registry, a: dict):
    s = reg.project.settings
    if a.get("width"): s["width"] = int(a["width"])
    if a.get("height"): s["height"] = int(a["height"])
    if a.get("fps"): s["fps"] = float(a["fps"])
    if a.get("duration_ms"): s["duration"] = int(a["duration_ms"])
    if a.get("bg"): s["bg"] = a["bg"]
    if a.get("title"): reg.project.data["title"] = a["title"]
    reg.project._normalize()
    return {"project": reg.project.summary(include_layers=False)}


@cmd("project.save", "Project", "Save project JSON (.cmproj).", {
    "path": A("str", "Destination file path"),
}, mutates=False)
def c_project_save(reg: Registry, a: dict):
    path = reg.project.save(a.get("path") or reg.project.path)
    return {"path": path}


@cmd("project.load", "Project", "Open a .cmproj project file.", {
    "path": A("str", "Project file path", required=True),
}, mutates=True)
def c_project_load(reg: Registry, a: dict):
    p = Project.load(a["path"])
    for item in p.media:
        item["project_dir"] = os.path.dirname(os.path.abspath(a["path"]))
        item["missing"] = MediaManager.resolve_path(item) is None
    reg.project = p
    return {"project": p.summary(include_layers=False)}


@cmd("project.undo", "Project", "Undo the last change.")
def c_undo(reg: Registry, a: dict):
    return {"undone": reg.undo()}


@cmd("project.redo", "Project", "Redo the last undone change.")
def c_redo(reg: Registry, a: dict):
    return {"redone": reg.redo()}


@cmd("project.import_am_xml", "Alight Motion", "Import an Alight Motion project or preset XML. "
     "merge=true applies the preset on top of the current project (offset_ms shifts it).", {
    "path": A("str", "XML file path", required=True),
    "merge": A("bool", "Merge into the current project instead of replacing it", default=False),
    "offset_ms": A("int", "Shift imported layers by N ms (for merging presets)", default=0),
    "title": A("str", "Project title when not merging"),
}, mutates=True)
def c_import_am(reg: Registry, a: dict):
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"file not found: {path}")
    existing = reg.project if a.get("merge") else None
    proj, res = import_am_scene(path, existing=existing, offset_ms=int(a.get("offset_ms") or 0))
    if not a.get("merge"):
        proj.data["title"] = a.get("title") or proj.data.get("title", "AM Import")
        reg.project = proj
    return {"stats": res.stats, "warnings": res.warnings[:40],
            "project": reg.project.summary(include_layers=True)}


@cmd("project.export_am_xml", "Alight Motion", "Export the project as Alight Motion scene XML.", {
    "path": A("str", "Destination .xml path", required=True),
    "title": A("str", "Scene title"),
}, mutates=False)
def c_export_am(reg: Registry, a: dict):
    path, warnings = export_am_scene(reg.project, a["path"], a.get("title"))
    return {"path": os.path.abspath(path), "warnings": warnings[:40]}


@cmd("project.validate_am_xml", "Alight Motion", "Validate an Alight Motion XML file without importing.", {
    "path": A("str", "XML file path", required=True),
})
def c_validate_am(reg: Registry, a: dict):
    return {"validation": validate_am_xml(a["path"])}


# ================================================================== MEDIA

@cmd("media.import", "Media", "Import a video, image or audio file into the project.", {
    "path": A("str", "Absolute path to the media file", required=True),
    "proxy": A("bool", "Also build a preview proxy in the background", default=True),
}, mutates=True)
def c_media_import(reg: Registry, a: dict):
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"file not found: {path}")
    item = import_media_file(path)
    item = reg.project.add_media(item)
    if a.get("proxy", True):
        reg.media.ensure_proxy(item)
        reg.media.ensure_audio_wav(item)
    reg.project.touch()
    return {"media": _media_public(item)}


@cmd("media.import_folder", "Media", "Import every supported file from a folder.", {
    "dir": A("str", "Folder path", required=True),
    "recursive": A("bool", "Scan subfolders", default=False),
}, mutates=True)
def c_media_import_folder(reg: Registry, a: dict):
    from .media import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT
    allowed = VIDEO_EXT | IMAGE_EXT | AUDIO_EXT
    root = a["dir"]
    if not os.path.isdir(root):
        raise CommandError(f"not a folder: {root}")
    files = []
    if a.get("recursive"):
        for base, _dirs, names in os.walk(root):
            files += [os.path.join(base, n) for n in names]
    else:
        files = [os.path.join(root, n) for n in os.listdir(root)]
    imported = []
    for f in files:
        if os.path.splitext(f)[1].lower() in allowed:
            try:
                item = reg.project.add_media(import_media_file(f))
                reg.media.ensure_proxy(item)
                imported.append(_media_public(item))
            except Exception:
                continue
    return {"imported": len(imported), "media": imported}


def _media_public(item: dict) -> dict:
    keys = ["id", "filename", "name", "kind", "width", "height", "duration_ms", "fps",
            "has_audio", "path", "missing"]
    out = {k: item.get(k) for k in keys}
    out["duration_s"] = round((item.get("duration_ms") or 0) / 1000.0, 2)
    return out


@cmd("media.list", "Media", "List all media in the project.")
def c_media_list(reg: Registry, a: dict):
    return {"media": [_media_public(m) for m in reg.project.media]}


@cmd("media.relink", "Media", "Point a media item at a new file path (fixes missing AM media).", {
    "media_id": A("int", "Media id", required=True),
    "path": A("str", "New file path", required=True),
}, mutates=True)
def c_media_relink(reg: Registry, a: dict):
    item = reg._media(a["media_id"])
    if not os.path.exists(a["path"]):
        raise CommandError(f"file not found: {a['path']}")
    info = probe_media(a["path"])
    item.update({"path": info["path"], "filename": info["filename"], "kind": info["kind"],
                 "width": info["width"], "height": info["height"],
                 "duration_ms": info["duration_ms"], "fps": info["fps"],
                 "has_audio": info["has_audio"], "missing": False})
    reg.media.ensure_proxy(item)
    return {"media": _media_public(item)}


@cmd("media.remove", "Media", "Remove a media item (layers using it stop rendering).", {
    "media_id": A("int", "Media id", required=True),
}, mutates=True)
def c_media_remove(reg: Registry, a: dict):
    mid = a["media_id"]
    before = len(reg.project.media)
    reg.project.data["media"] = [m for m in reg.project.media if m.get("id") != mid]
    return {"removed": before - len(reg.project.media)}


@cmd("media.waveform", "Media", "Peak waveform data for an audio-carrying media item.", {
    "media_id": A("int", "Media id", required=True),
    "buckets": A("int", "Number of peaks", default=200),
})
def c_media_waveform(reg: Registry, a: dict):
    item = reg._media(a["media_id"])
    return {"peaks": reg.media.waveform(item, int(a["buckets"]))}


@cmd("media.thumbnail", "Media", "Save a thumbnail PNG of a media item and return its path.", {
    "media_id": A("int", "Media id", required=True),
    "width": A("int", "Thumbnail width", default=320),
})
def c_media_thumb(reg: Registry, a: dict):
    item = reg._media(a["media_id"])
    img = reg.media.thumbnail(item, int(a["width"]))
    if img is None:
        raise CommandError("thumbnail unavailable (missing file?)")
    return {"path": reg.save_thumbnail(img)}


# ================================================================== LAYERS

LAYER_FIELDS = {
    "name", "start", "end", "hidden", "locked", "blend", "parent", "color", "color2",
    "gradient_angle", "shape", "w", "h", "radius", "points", "inner", "path", "closed",
    "stroke_color", "stroke_width", "text", "size", "font", "align", "wrap", "bold",
    "line_height", "shadow", "media_id", "fit", "speed", "source_in", "volume",
    "fade_in", "fade_out", "include_audio", "clip", "matte", "fill_type",
    # parity fields (precomps, 2.5D depth, lights, masks/styles, text, shape ops)
    "comp_id", "z", "masks", "styles", "text_animators", "trim", "repeater", "trail",
    "intensity", "lit", "expressions", "transition_in", "transition_out", "track", "solo",
}


@cmd("layer.add", "Layers", "Add a layer. kind: video|image|text|shape|solid|adjustment|audio|null. "
     "Anything not specified uses sensible defaults.", {
    "kind": A("str", "Layer kind", required=True, enum=LAYER_KINDS),
    "name": A("str", "Layer name"),
    "index": A("int", "Insert position (0 = bottom); default top"),
    "start": A("int", "Start time in ms", default=0),
    "end": A("int", "End time in ms"),
    "media_id": A("int", "Media id for video/image/audio layers"),
    "effects": A("list", "Optional initial effect list [{type, params}]"),
    "transform": A("dict", "Initial transform {pos,scale,rot,opacity,anchor,skew}"),
    "comp_id": A("int", "Source composition id for precomp layers"),
    "z": A("float", "Depth for 2.5D camera parallax", default=0.0),
    "w": A("float", "Layer width"), "h": A("float", "Layer height"),
    "color": A("str", "Fill colour (#AARRGGBB)"), "text": A("str", "Text content"),
    "masks": A("list", "Initial mask list"), "styles": A("list", "Initial layer-style list"),
    "text_animators": A("list", "Initial text animator list"),
    "trim": A("dict", "Shape trim {start,end,offset}"),
    "repeater": A("dict", "Shape repeater {copies,x,y,...}"),
    "intensity": A("float", "Light intensity", default=0.6),
    "radius": A("float", "Light radius"),
    "fit": A("str", "Media fit: cover|contain|stretch|none"),
}, mutates=True)
def c_layer_add(reg: Registry, a: dict):
    p = reg.project
    kind = a["kind"]
    if kind not in LAYER_KINDS:
        raise CommandError(f"kind must be one of {LAYER_KINDS}")
    layer: Dict[str, Any] = {"kind": kind}
    for key, val in a.items():
        if key in LAYER_FIELDS:
            layer[key] = val
    if a.get("transform"):
        tf = a["transform"]
        layer["transform"] = {k: v for k, v in tf.items()}
    if kind not in ("audio", "null"):
        tf = layer.setdefault("transform", {})
        tf.setdefault("pos", [p.width / 2.0, p.height / 2.0, 0.0])
    layer.setdefault("end", p.duration)
    if layer.get("end", 0) <= layer.get("start", 0):
        layer["end"] = layer["start"] + max(1000, p.duration)
    if kind == "precomp":
        layer.setdefault("fit", "stretch")
        comp = p.get_comp(layer.get("comp_id")) if layer.get("comp_id") is not None else None
        if comp is None:
            raise CommandError("precomp layers need a valid comp_id (see comp.create)")
        layer.setdefault("name", comp.get("name") or "Precomp")
        layer.setdefault("w", comp.get("width") or p.width)
        layer.setdefault("h", comp.get("height") or p.height)
        layer.setdefault("end", min(p.duration, int(comp.get("duration") or p.duration)))
    if kind == "light":
        layer.setdefault("color", "#FFFFCC88")
        layer.setdefault("radius", p.width * 0.6)
        layer.setdefault("intensity", 0.6)
        layer.setdefault("name", "Light")
    if kind in ("video", "image", "audio"):
        if layer.get("media_id") is None:
            raise CommandError("video/image/audio layers need a media_id (see media.import)")
        item = reg._media(layer["media_id"])
        if kind == "image":
            pass
        elif kind == "audio":
            layer.setdefault("name", item.get("name") or "Audio")
            layer.setdefault("volume", 1.0)
        else:
            layer.setdefault("fit", "cover")
        if not layer.get("name"):
            layer["name"] = item.get("name") or item.get("filename") or kind.title()
        if kind != "audio":
            natural_end = layer["start"] + (item.get("duration_ms") or 0)
            if kind == "video" and (item.get("duration_ms") or 0) > 0:
                layer["end"] = min(layer["end"], natural_end) if a.get("end") is None else layer["end"]
    if kind == "text":
        layer.setdefault("text", "Your text here")
        layer.setdefault("name", "Text")
        layer.setdefault("size", 96)
    if kind == "shape":
        layer.setdefault("shape", "rect")
        layer.setdefault("name", str(layer.get("shape", "shape")).title())
    if kind == "solid":
        layer.setdefault("name", "Solid")
    if kind == "adjustment":
        layer.setdefault("name", "Adjustment")
    new = p.add_layer(layer, index=a.get("index"))
    reg.emit({"type": "select", "layer_id": new["id"]})
    return {"layer": layer_summary(new)}


@cmd("layer.list", "Layers", "All layers, bottom → top.")
def c_layer_list(reg: Registry, a: dict):
    return {"layers": [layer_summary(l) for l in reg.project.layers]}


@cmd("layer.get", "Layers", "Full JSON for one layer.", {
    "layer_id": A("int", "Layer id", required=True),
})
def c_layer_get(reg: Registry, a: dict):
    return {"layer": reg._layer(a["layer_id"])}


@cmd("layer.update", "Layers", "Update any layer property (name, timing, blend, colors, text, media, ...).", {
    "layer_id": A("int", "Layer id", required=True),
    "props": A("dict", "Fields to change, e.g. {color:'#FFFF0000', text:'Hi', end:5000}", required=True),
}, mutates=True)
def c_layer_update(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    unknown = []
    for key, val in a["props"].items():
        if key in ("transform",):
            tf = dict(layer.get("transform") or {})
            tf.update(val)
            layer["transform"] = tf
        elif key in ("keys", "effects", "id", "kind"):
            continue
        elif key in LAYER_FIELDS:
            if key == "blend" and val not in BLEND_MODES:
                raise CommandError(f"blend must be one of {BLEND_MODES}")
            layer[key] = val
        else:
            unknown.append(key)
    reg.project._normalize_layer(layer)
    if unknown:
        return {"layer": layer_summary(layer), "ignored": unknown}
    return {"layer": layer_summary(layer)}


@cmd("layer.set_transform", "Layers", "Set static transform values (pos, anchor, scale, skew, rot, opacity).", {
    "layer_id": A("int", "Layer id", required=True),
    "pos": A("list", "Position [x,y] in px"),
    "anchor": A("list", "Anchor point [x,y]"),
    "scale": A("list", "Scale [x,y] or number"),
    "skew": A("list", "Skew [x,y] in degrees"),
    "rot": A("float", "Rotation in degrees"),
    "opacity": A("float", "Opacity 0-1"),
}, mutates=True)
def c_layer_transform(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    tf = layer.setdefault("transform", {})
    if a.get("pos") is not None: tf["pos"] = parse_vec(a["pos"], dims=3)
    if a.get("anchor") is not None: tf["anchor"] = parse_vec(a["anchor"], dims=2)
    if a.get("scale") is not None: tf["scale"] = parse_vec(a["scale"], dims=2)
    if a.get("skew") is not None: tf["skew"] = parse_vec(a["skew"], dims=2)
    if a.get("rot") is not None: tf["rot"] = [float(a["rot"])]
    if a.get("opacity") is not None: tf["opacity"] = [float(a["opacity"])]
    return {"layer": layer_summary(layer)}


@cmd("layer.delete", "Layers", "Delete a layer.", {
    "layer_id": A("int", "Layer id", required=True),
}, mutates=True)
def c_layer_delete(reg: Registry, a: dict):
    ok = reg.project.remove_layer(int(a["layer_id"]))
    return {"deleted": ok}


@cmd("layer.duplicate", "Layers", "Duplicate a layer (optionally shifted in time).", {
    "layer_id": A("int", "Layer id", required=True),
    "time_offset": A("int", "Shift the copy by N ms", default=0),
}, mutates=True)
def c_layer_dup(reg: Registry, a: dict):
    src = reg._layer(a["layer_id"])
    new = copy.deepcopy(src)
    new["id"] = reg.project.new_id()
    new["name"] = str(src.get("name", "Layer")) + " copy"
    off = int(a.get("time_offset") or 0)
    new["start"] = int(src.get("start", 0)) + off
    new["end"] = int(src.get("end", 0)) + off
    for prop, keys in (new.get("keys") or {}).items():
        for k in keys:
            k["t"] += off
    for eff in new.get("effects", []):
        for prop, keys in (eff.get("keys") or {}).items():
            for k in keys:
                k["t"] += off
    idx = reg.project.layer_index(src["id"]) + 1
    added = reg.project.add_layer(new, index=idx)
    return {"layer": layer_summary(added)}


@cmd("layer.reorder", "Layers", "Move a layer to a new index (0 = bottom, -1/last = top).", {
    "layer_id": A("int", "Layer id", required=True),
    "index": A("int", "New index", required=True),
}, mutates=True)
def c_layer_reorder(reg: Registry, a: dict):
    layers = reg.project.layers
    idx = reg.project.layer_index(int(a["layer_id"]))
    if idx < 0:
        raise CommandError("layer not found")
    layer = layers.pop(idx)
    target = int(a["index"])
    if target < 0:
        target = len(layers)
    layers.insert(max(0, min(target, len(layers))), layer)
    return {"order": [l["id"] for l in layers]}


@cmd("layer.set_parent", "Layers", "Parent a layer to another (nulls make handy control rigs).", {
    "layer_id": A("int", "Layer id", required=True),
    "parent_id": A("int", "Parent layer id, omit to clear"),
}, mutates=True)
def c_set_parent(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    pid = a.get("parent_id")
    if pid is not None:
        if int(pid) == int(layer["id"]):
            raise CommandError("a layer cannot parent itself")
        reg._layer(pid)
    layer["parent"] = int(pid) if pid is not None else None
    return {"layer": layer_summary(layer)}


@cmd("layer.group", "Layers", "Group layers under a new null (a control rig for moving them together).", {
    "layer_ids": A("list", "Layer ids to group", required=True),
    "name": A("str", "Group name", default="Group"),
}, mutates=True)
def c_group(reg: Registry, a: dict):
    ids = [int(x) for x in a["layer_ids"]]
    layers = [reg._layer(i) for i in ids]
    top = max(reg.project.layer_index(l["id"]) for l in layers)
    null = reg.project.add_layer({"kind": "null", "name": a["name"], "start": 0,
                                  "end": reg.project.duration}, index=top + 1)
    for l in layers:
        l["parent"] = null["id"]
    return {"group": layer_summary(null), "members": ids}


@cmd("layer.ungroup", "Layers", "Dissolve a group (children keep their world transform).", {
    "group_id": A("int", "Group/null layer id", required=True),
}, mutates=True)
def c_ungroup(reg: Registry, a: dict):
    gid = int(a["group_id"])
    children = [l for l in reg.project.layers if l.get("parent") == gid]
    for l in children:
        l["parent"] = None
    reg.project.remove_layer(gid)
    return {"released": [l["id"] for l in children]}


@cmd("layer.split", "Layers", "Split a layer at time t into two layers.", {
    "layer_id": A("int", "Layer id", required=True),
    "t": A("int", "Split time in ms", required=True),
}, mutates=True)
def c_split(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    t = int(a["t"])
    if not (int(layer["start"]) < t < int(layer["end"])):
        raise CommandError("split time must be inside the layer")
    right = copy.deepcopy(layer)
    right["id"] = reg.project.new_id()
    right["start"] = t
    right["name"] = str(layer.get("name", "Layer")) + " (R)"
    if layer.get("kind") in ("video", "image", "audio"):
        right["source_in"] = int(layer.get("source_in", 0)) + int((t - layer["start"]) * float(layer.get("speed", 1.0)))
    layer["end"] = t
    layer["name"] = str(layer.get("name", "Layer"))
    added = reg.project.add_layer(right, index=reg.project.layer_index(layer["id"]) + 1)
    return {"left": layer_summary(layer), "right": layer_summary(added)}


@cmd("layer.trim", "Layers", "Change a layer's start/end time.", {
    "layer_id": A("int", "Layer id", required=True),
    "start": A("int", "New start ms"),
    "end": A("int", "New end ms"),
}, mutates=True)
def c_trim(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    if a.get("start") is not None:
        layer["start"] = int(a["start"])
    if a.get("end") is not None:
        layer["end"] = int(a["end"])
    if layer["end"] <= layer["start"]:
        raise CommandError("end must be after start")
    return {"layer": layer_summary(layer)}


@cmd("layer.set_matte", "Layers", "Use another layer's alpha as this layer's track matte.", {
    "layer_id": A("int", "Layer to mask", required=True),
    "matte_layer_id": A("int", "Layer providing the alpha, omit to clear"),
}, mutates=True)
def c_matte(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    mid = a.get("matte_layer_id")
    if mid is not None:
        reg._layer(mid)
        layer["matte"] = int(mid)
    else:
        layer.pop("matte", None)
    return {"layer": layer_summary(layer)}


# ================================================================== KEYFRAMES

KEYABLE = {"pos": 3, "anchor": 2, "scale": 2, "skew": 2, "rot": 1, "opacity": 1, "color": 4, "volume": 1}


@cmd("keyframe.set", "Keyframes", "Set a keyframe on a layer property: pos, anchor, scale, skew, rot, opacity, color, volume.", {
    "layer_id": A("int", "Layer id", required=True),
    "prop": A("str", "Property", required=True),
    "t": A("int", "Time in ms", required=True),
    "value": A("list", "Value (number or [..] depending on prop)"),
    "easing": A("str", "linear|ease|ease_in|ease_out|ease_in_out|hold|expo_out|back_out|cubicBezier x1 y1 x2 y2", default="ease_in_out"),
}, mutates=True)
def c_kf_set(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    prop = a["prop"]
    if prop not in KEYABLE:
        raise CommandError(f"prop must be one of {list(KEYABLE)}")
    dims = KEYABLE[prop]
    if prop == "color":
        val = list(parse_color(a.get("value", "#FFFFFFFF")))
    else:
        val = parse_vec(a.get("value", 0), dims=dims)
    keys = layer.setdefault("keys", {}).setdefault(prop, [])
    kf_add(keys, int(a["t"]), val, a.get("easing"))
    return {"prop": prop, "keyframes": keys}


@cmd("keyframe.delete", "Keyframes", "Delete a keyframe (all keyframes of a prop when t is omitted).", {
    "layer_id": A("int", "Layer id", required=True),
    "prop": A("str", "Property", required=True),
    "t": A("int", "Time in ms (omit to clear the whole track)"),
}, mutates=True)
def c_kf_delete(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    keys = (layer.get("keys") or {}).get(a["prop"])
    if not keys:
        return {"deleted": False}
    if a.get("t") is None:
        layer["keys"].pop(a["prop"], None)
        return {"deleted": True, "cleared": True}
    return {"deleted": kf_remove(keys, int(a["t"]))}


@cmd("keyframe.list", "Keyframes", "List keyframe tracks for a layer.", {
    "layer_id": A("int", "Layer id", required=True),
})
def c_kf_list(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    out = {}
    for prop, keys in (layer.get("keys") or {}).items():
        out[prop] = keys
    for i, eff in enumerate(layer.get("effects") or []):
        for prop, keys in (eff.get("keys") or {}).items():
            out[f"effect[{i}].{eff.get('type')}.{prop}"] = keys
    return {"tracks": out}


@cmd("keyframe.evaluate", "Keyframes", "Evaluate a property at a time (handy for agents).", {
    "layer_id": A("int", "Layer id", required=True),
    "prop": A("str", "Property", required=True),
    "t": A("int", "Time in ms", required=True),
})
def c_kf_eval_cmd(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    prop = a["prop"]
    dims = KEYABLE.get(prop, 2)
    keys = (layer.get("keys") or {}).get(prop)
    if keys:
        return {"value": kf_eval(keys, int(a["t"]), [0.0] * dims)}
    base = (layer.get("transform") or {}).get(prop)
    if base is None:
        base = {"color": [1, 1, 1, 1], "volume": [1]}.get(prop, [0.0] * dims)
    if isinstance(base, str):
        return {"value": list(parse_color(base))}
    return {"value": parse_vec(base, dims=dims)}


@cmd("keyframe.offset", "Keyframes", "Shift all keyframes of a layer (or one prop) in time.", {
    "layer_id": A("int", "Layer id", required=True),
    "delta_ms": A("int", "Offset in ms", required=True),
    "prop": A("str", "Only this property"),
}, mutates=True)
def c_kf_offset(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    dt = int(a["delta_ms"])
    props = [a["prop"]] if a.get("prop") else list((layer.get("keys") or {}).keys())
    for prop in props:
        for k in (layer.get("keys") or {}).get(prop, []):
            k["t"] = max(0, k["t"] + dt)
    return {"shifted": props}


@cmd("keyframe.set_easing", "Keyframes", "Change the easing of a keyframe segment.", {
    "layer_id": A("int", "Layer id", required=True),
    "prop": A("str", "Property", required=True),
    "t": A("int", "Keyframe time in ms", required=True),
    "easing": A("str", "easing name", required=True),
}, mutates=True)
def c_kf_easing(reg: Registry, a: dict):
    from .util import parse_easing
    layer = reg._layer(a["layer_id"])
    keys = (layer.get("keys") or {}).get(a["prop"]) or []
    k = kf_find(keys, int(a["t"]), tol=1000)
    if k is None:
        raise CommandError("no keyframe near that time")
    k["e"] = parse_easing(a["easing"])
    return {"keyframe": k}


# ================================================================== EFFECTS

@cmd("effect.types", "Effects", "List every available effect with its parameters.", {
    "category": A("str", "Filter: Color|Blur|Stylize|Distort|Transition|Generate"),
})
def c_effect_types(reg: Registry, a: dict):
    return {"effects": FX.list_effects(a.get("category")), "categories": FX.categories()}


@cmd("effect.add", "Effects", "Add an effect to a layer's effect stack.", {
    "layer_id": A("int", "Layer id", required=True),
    "type": A("str", "Effect type (see effect.types)", required=True),
    "params": A("dict", "Parameter overrides"),
    "index": A("int", "Stack position (default append)"),
}, mutates=True)
def c_effect_add(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    etype = FX.resolve_type(a["type"])
    if etype is None:
        raise CommandError(f"unknown effect '{a['type']}' (see effect.types)")
    eff = {"type": etype, "enabled": True, "params": FX.defaults_for(etype), "keys": {}}
    given = dict(a.get("params") or {})
    spec = FX.EFFECTS.get(etype)
    if spec and given:
        unknown = sorted(k for k in given if k not in spec.params)
        if unknown:
            raise CommandError(
                f"effect '{etype}' has no parameter(s) {unknown}; valid: {sorted(spec.params)}")
        for key, val in given.items():
            prm = spec.params[key]
            if prm.kind == "float" and isinstance(val, (int, float)) and not isinstance(val, bool):
                val = max(float(prm.lo or 0.0), min(float(prm.hi or 1.0), float(val)))
            eff["params"][key] = val
    else:
        eff["params"].update(given)
    stack = layer.setdefault("effects", [])
    if a.get("index") is not None:
        stack.insert(int(a["index"]), eff)
    else:
        stack.append(eff)
    return {"effects": [e["type"] for e in stack], "index": stack.index(eff)}


@cmd("effect.update", "Effects", "Update effect params / enabled flag.", {
    "layer_id": A("int", "Layer id", required=True),
    "effect_index": A("int", "Position in the stack", required=True),
    "params": A("dict", "Params to merge"),
    "enabled": A("bool", "Enable or bypass"),
}, mutates=True)
def c_effect_update(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    stack = layer.get("effects") or []
    idx = int(a["effect_index"])
    if idx < 0 or idx >= len(stack):
        raise CommandError(f"effect_index out of range (stack has {len(stack)})")
    eff = stack[idx]
    if a.get("params"):
        eff.setdefault("params", {}).update(a["params"])
    if a.get("enabled") is not None:
        eff["enabled"] = bool(a["enabled"])
    return {"effect": {"type": eff["type"], "enabled": eff.get("enabled", True), "params": eff.get("params", {})}}


@cmd("effect.set_param", "Effects", "Set one effect parameter; pass t to make it a keyframe.", {
    "layer_id": A("int", "Layer id", required=True),
    "effect_index": A("int", "Position in the stack", required=True),
    "param": A("str", "Parameter name", required=True),
    "value": A("list", "Value"),
    "t": A("int", "Keyframe time in ms (omit for static value)"),
    "easing": A("str", "Easing for the keyframe", default="ease_in_out"),
}, mutates=True)
def c_effect_param(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    stack = layer.get("effects") or []
    idx = int(a["effect_index"])
    if idx < 0 or idx >= len(stack):
        raise CommandError(f"effect_index out of range (stack has {len(stack)})")
    eff = stack[idx]
    val = a.get("value")
    if a.get("t") is not None:
        num = float(parse_vec(val, dims=1)[0])
        keys = eff.setdefault("keys", {}).setdefault(a["param"], [])
        kf_add(keys, int(a["t"]), [num], a.get("easing"))
        return {"keyframes": keys}
    eff.setdefault("params", {})[a["param"]] = val
    return {"params": eff["params"]}


@cmd("effect.remove", "Effects", "Remove an effect from the stack.", {
    "layer_id": A("int", "Layer id", required=True),
    "effect_index": A("int", "Position in the stack", required=True),
}, mutates=True)
def c_effect_remove(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    stack = layer.get("effects") or []
    idx = int(a["effect_index"])
    if 0 <= idx < len(stack):
        stack.pop(idx)
    return {"effects": [e["type"] for e in stack]}


@cmd("effect.move", "Effects", "Reorder an effect in the stack.", {
    "layer_id": A("int", "Layer id", required=True),
    "from_index": A("int", "Current index", required=True),
    "to_index": A("int", "New index", required=True),
}, mutates=True)
def c_effect_move(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    stack = layer.get("effects") or []
    i, j = int(a["from_index"]), int(a["to_index"])
    if not (0 <= i < len(stack)):
        raise CommandError("from_index out of range")
    eff = stack.pop(i)
    stack.insert(max(0, min(j, len(stack))), eff)
    return {"effects": [e["type"] for e in stack]}


@cmd("effect.list", "Effects", "Effects on a layer, with params and keyframes.", {
    "layer_id": A("int", "Layer id", required=True),
})
def c_effect_list(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    out = []
    for i, eff in enumerate(layer.get("effects") or []):
        out.append({"index": i, "type": eff.get("type"), "src_id": eff.get("src_id"),
                    "enabled": eff.get("enabled", True), "params": eff.get("params", {}),
                    "keyed": sorted((eff.get("keys") or {}).keys())})
    return {"effects": out}


@cmd("effect.looks", "Effects", "List built-in look presets (effect stacks).")
def c_effect_looks(reg: Registry, a: dict):
    return {"looks": {k: {"label": v["label"], "stack": [e[0] for e in v["effects"]]}
                      for k, v in BUILTIN_LOOKS.items()}}


@cmd("effect.apply_look", "Effects", "Apply a built-in look preset to a layer.", {
    "layer_id": A("int", "Layer id", required=True),
    "look": A("str", "Look name (see effect.looks)", required=True),
    "replace": A("bool", "Replace the existing stack", default=False),
}, mutates=True)
def c_apply_look(reg: Registry, a: dict):
    look = BUILTIN_LOOKS.get(a["look"])
    if not look:
        raise CommandError(f"unknown look '{a['look']}' (see effect.looks)")
    layer = reg._layer(a["layer_id"])
    if a.get("replace"):
        layer["effects"] = []
    stack = layer.setdefault("effects", [])
    for etype, params in look["effects"]:
        eff = {"type": etype, "enabled": True, "params": FX.defaults_for(etype), "keys": {}}
        eff["params"].update(params)
        stack.append(eff)
    return {"effects": [e["type"] for e in stack]}


@cmd("transition.add", "Effects", "Add an animated transition (fade/wipe/iris/dissolve/slide) to a layer.", {
    "layer_id": A("int", "Layer id", required=True),
    "kind": A("str", "Transition kind", default="fade",
              enum=["fade", "wipe", "iris", "dissolve", "slide"]),
    "duration": A("int", "Transition duration in ms", default=600),
    "mode": A("str", "in or out", default="in", enum=["in", "out"]),
    "easing": A("str", "Easing", default="ease_in_out"),
}, mutates=True)
def c_transition_add(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    etype = a["kind"]
    eff = {"type": etype, "enabled": True, "params": FX.defaults_for(etype), "keys": {}}
    eff["params"]["progress"] = -1.0
    if a.get("mode") == "out":
        eff["params"].setdefault("mode", "out")
    start = int(layer["start"])
    end = int(layer["end"])
    dur = int(min(a["duration"], end - start))
    if a.get("mode") == "out":
        t0, t1 = end - dur, end
        vals = [1.0, 0.0] if etype == "fade" else [1.0, 0.0]
    else:
        t0, t1 = start, start + dur
        vals = [0.0, 1.0]
    eff["params"]["progress"] = 1.0
    eff["keys"]["progress"] = [
        {"t": t0, "v": [vals[0]], "e": None},
        {"t": t1, "v": [vals[1]], "e": None},
    ]
    stack = layer.setdefault("effects", [])
    stack.append(eff)
    return {"effects": [e["type"] for e in stack], "keyframes": eff["keys"]["progress"]}


# ---- user effect-stack presets -------------------------------------------

def _load_user_presets(reg: Registry) -> dict:
    try:
        with open(reg.user_presets_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_user_presets(reg: Registry, data: dict) -> None:
    os.makedirs(APP_HOME, exist_ok=True)
    with open(reg.user_presets_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


@cmd("preset.save", "Effects", "Save a layer's effect stack as a reusable preset.", {
    "name": A("str", "Preset name", required=True),
    "layer_id": A("int", "Layer id", required=True),
}, mutates=False)
def c_preset_save(reg: Registry, a: dict):
    layer = reg._layer(a["layer_id"])
    data = _load_user_presets(reg)
    data[a["name"]] = {"stack": copy.deepcopy(layer.get("effects") or []), "saved": now_ms()}
    _save_user_presets(reg, data)
    return {"saved": a["name"], "effects": [e.get("type") for e in layer.get("effects") or []]}


@cmd("preset.list", "Effects", "List saved effect presets.")
def c_preset_list(reg: Registry, a: dict):
    data = _load_user_presets(reg)
    return {"presets": {k: [e.get("type") for e in v.get("stack", [])] for k, v in data.items()}}


@cmd("preset.apply", "Effects", "Apply a saved effect preset to a layer.", {
    "name": A("str", "Preset name", required=True),
    "layer_id": A("int", "Layer id", required=True),
    "replace": A("bool", "Replace the stack", default=False),
}, mutates=True)
def c_preset_apply(reg: Registry, a: dict):
    data = _load_user_presets(reg)
    if a["name"] not in data:
        raise CommandError(f"preset '{a['name']}' not found")
    layer = reg._layer(a["layer_id"])
    if a.get("replace"):
        layer["effects"] = []
    stack = layer.setdefault("effects", [])
    stack.extend(copy.deepcopy(data[a["name"]]["stack"]))
    return {"effects": [e.get("type") for e in stack]}


@cmd("preset.delete", "Effects", "Delete a saved preset.", {
    "name": A("str", "Preset name", required=True),
})
def c_preset_delete(reg: Registry, a: dict):
    data = _load_user_presets(reg)
    existed = data.pop(a["name"], None) is not None
    _save_user_presets(reg, data)
    return {"deleted": existed}


# ================================================================== MARKERS

@cmd("marker.add", "Timeline", "Add a timeline marker.", {
    "t": A("int", "Time in ms", required=True),
    "label": A("str", "Marker label", default=""),
}, mutates=True)
def c_marker_add(reg: Registry, a: dict):
    reg.project.data.setdefault("markers", []).append({"t": int(a["t"]), "label": a.get("label", "")})
    reg.project.data["markers"].sort(key=lambda m: m["t"])
    return {"markers": reg.project.data["markers"]}


@cmd("marker.remove", "Timeline", "Remove markers near a time.", {
    "t": A("int", "Time in ms", required=True),
    "tolerance": A("int", "Tolerance ms", default=50),
}, mutates=True)
def c_marker_remove(reg: Registry, a: dict):
    before = len(reg.project.data.get("markers", []))
    reg.project.data["markers"] = [m for m in reg.project.data.get("markers", [])
                                   if abs(m["t"] - int(a["t"])) > int(a["tolerance"])]
    return {"removed": before - len(reg.project.data["markers"])}


# ================================================================== RENDER

@cmd("render.preview", "Render", "Render one frame at preview size; returns a PNG path.", {
    "t": A("int", "Time in ms", default=0),
    "max_dim": A("int", "Max preview dimension", default=640),
    "path": A("str", "Optional output path"),
})
def c_render_preview(reg: Registry, a: dict):
    r = reg.renderer("preview")
    try:
        W, H = reg.project.width, reg.project.height
        scale = min(1.0, float(a["max_dim"]) / max(W, H))
        img = r.render(int(a["t"]), size=(max(2, int(W * scale)), max(2, int(H * scale))))
    finally:
        r.close()
    return {"path": reg.save_thumbnail(img, a.get("path")), "t": int(a["t"]),
            "width": img.width, "height": img.height}


@cmd("render.storyboard", "Render", "Render a grid of frames as one PNG (great for agent review).", {
    "times": A("list", "List of times in ms", required=False),
    "count": A("int", "Number of evenly spaced frames if times omitted", default=9),
    "cols": A("int", "Grid columns", default=3),
    "tile": A("int", "Tile width px", default=320),
    "path": A("str", "Optional output path"),
})
def c_storyboard(reg: Registry, a: dict):
    p = reg.project
    times = a.get("times") or [int(i * p.duration / max(1, int(a["count"]))) for i in range(int(a["count"]))]
    times = [int(t) for t in times][:64]
    cols = max(1, int(a["cols"]))
    tile_w = int(a["tile"])
    tile_h = int(tile_w * p.height / p.width)
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (18, 18, 22))
    r = reg.renderer("preview")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    try:
        for i, t in enumerate(times):
            img = r.render(int(t), size=(tile_w, tile_h)).convert("RGB")
            x, y = (i % cols) * tile_w, (i // cols) * tile_h
            sheet.paste(img, (x, y))
            draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(60, 60, 70))
            draw.text((x + 6, y + 4), fmt_time(t, p.fps), fill=(255, 255, 255))
    finally:
        r.close()
    return {"path": reg.save_thumbnail(sheet, a.get("path")), "times": times}


@cmd("render.still", "Render", "Export a full-resolution still frame (PNG/JPG).", {
    "t": A("int", "Time in ms", default=0),
    "path": A("str", "Output path", required=True),
    "width": A("int", "Output width"),
    "height": A("int", "Output height"),
})
def c_render_still(reg: Registry, a: dict):
    from .export import render_still
    path = render_still(reg.project, reg.media, int(a["t"]), a["path"],
                        a.get("width"), a.get("height"))
    return {"path": path}


@cmd("render.video", "Render", "Render the timeline to MP4/WebM (async job; poll job.status).", {
    "path": A("str", "Output file path (.mp4/.webm)", required=True),
    "width": A("int", "Output width"),
    "height": A("int", "Output height"),
    "fps": A("float", "Frame rate"),
    "start_ms": A("int", "Range start"),
    "end_ms": A("int", "Range end"),
    "quality": A("str", "draft|good|high", default="good"),
    "audio": A("bool", "Include mixed audio", default=True),
    "wait": A("bool", "Block until finished", default=False),
})
def c_render_video(reg: Registry, a: dict):
    from .export import render_video
    project_copy = reg.project.clone()
    project_copy.path = None
    args = dict(a)
    path = args["path"]

    def worker(job: Job):
        return {"path": render_video(
            project_copy, reg.media, path, args.get("width"), args.get("height"),
            args.get("fps"), args.get("start_ms"), args.get("end_ms"),
            args.get("quality", "good"), bool(args.get("audio", True)),
            progress=lambda p, m: job.update(p, m), cancelled=lambda: job.cancel_flag)}

    jid = reg.start_job("video", f"Video → {os.path.basename(path)}", worker)
    if a.get("wait"):
        while reg.jobs[jid].status == "running":
            time.sleep(0.2)
        job = reg.jobs[jid]
        return {"job": job.to_dict()}
    return {"job_id": jid, "status": "running"}


@cmd("render.sequence", "Render", "Export a PNG/JPG image sequence (async job).", {
    "dir": A("str", "Output folder", required=True),
    "start_ms": A("int", "Range start"),
    "end_ms": A("int", "Range end"),
    "fps": A("float", "Frame rate"),
    "width": A("int", "Output width"),
    "height": A("int", "Output height"),
    "format": A("str", "png or jpg", default="png"),
    "wait": A("bool", "Block until finished", default=False),
})
def c_render_sequence(reg: Registry, a: dict):
    from .export import render_sequence
    project_copy = reg.project.clone()
    args = dict(a)

    def worker(job: Job):
        return {"dir": render_sequence(
            project_copy, reg.media, args["dir"], args.get("start_ms") or 0, args.get("end_ms"),
            args.get("fps"), args.get("width"), args.get("height"), args.get("format", "png"),
            progress=lambda p, m: job.update(p, m), cancelled=lambda: job.cancel_flag)}

    jid = reg.start_job("sequence", f"Sequence → {os.path.basename(args['dir'])}", worker)
    if a.get("wait"):
        while reg.jobs[jid].status == "running":
            time.sleep(0.15)
        return {"job": reg.jobs[jid].to_dict()}
    return {"job_id": jid, "status": "running"}


@cmd("render.gif", "Render", "Export an animated GIF (async job).", {
    "path": A("str", "Output .gif path", required=True),
    "start_ms": A("int", "Range start"),
    "end_ms": A("int", "Range end"),
    "fps": A("float", "Frame rate", default=15.0),
    "width": A("int", "Output width", default=480),
    "wait": A("bool", "Block until finished", default=False),
})
def c_render_gif(reg: Registry, a: dict):
    from .export import render_gif
    project_copy = reg.project.clone()
    args = dict(a)

    def worker(job: Job):
        return {"path": render_gif(
            project_copy, reg.media, args["path"], args.get("start_ms") or 0, args.get("end_ms"),
            float(args.get("fps") or 15), int(args.get("width") or 480),
            progress=lambda p, m: job.update(p, m), cancelled=lambda: job.cancel_flag)}

    jid = reg.start_job("gif", f"GIF → {os.path.basename(args['path'])}", worker)
    if a.get("wait"):
        while reg.jobs[jid].status == "running":
            time.sleep(0.15)
        return {"job": reg.jobs[jid].to_dict()}
    return {"job_id": jid, "status": "running"}


@cmd("render.audio", "Render", "Mix and export the project audio as WAV.", {
    "path": A("str", "Output .wav path", required=True),
    "start_ms": A("int", "Range start"),
    "end_ms": A("int", "Range end"),
})
def c_render_audio(reg: Registry, a: dict):
    from .export import mix_audio
    out = mix_audio(reg.project, reg.media, os.path.abspath(a["path"]),
                    a.get("start_ms") or 0, a.get("end_ms"))
    if out is None:
        raise CommandError("project has no audible layers")
    return {"path": os.path.abspath(out)}


@cmd("job.status", "Render", "Status of a render job.", {
    "job_id": A("str", "Job id", required=True),
})
def c_job_status(reg: Registry, a: dict):
    job = reg.jobs.get(a["job_id"])
    if job is None:
        raise CommandError(f"unknown job '{a['job_id']}'")
    return {"job": job.to_dict()}


@cmd("job.list", "Render", "List render jobs.")
def c_job_list(reg: Registry, a: dict):
    return {"jobs": [j.to_dict() for j in reg.jobs.values()]}


@cmd("job.cancel", "Render", "Cancel a running render job.", {
    "job_id": A("str", "Job id", required=True),
})
def c_job_cancel(reg: Registry, a: dict):
    job = reg.jobs.get(a["job_id"])
    if job is None:
        raise CommandError(f"unknown job '{a['job_id']}'")
    job.cancel_flag = True
    job.message = "Cancelling…"
    return {"job_id": job.id}


# ------------------------------------------------------------------ aliases

ALIASES = {
    "add_layer": "layer.add", "set_transform": "layer.set_transform",
    "keyframe": "keyframe.set", "add_effect": "effect.add",
    "render": "render.video", "import_am_xml": "project.import_am_xml",
    "export_am_xml": "project.export_am_xml", "import_media": "media.import",
}


def resolve_command(name: str) -> Optional[str]:
    if name in COMMANDS:
        return name
    if name in ALIASES:
        return ALIASES[name]

    def norm(s: str) -> str:
        return s.replace("_", "").replace(".", "").lower()

    for key in COMMANDS:
        if norm(key) == norm(name):
            return key
    for key in COMMANDS:
        if norm(key).endswith(norm(name)) and norm(name):
            return key
    return None


# Advanced feature commands (masks, grade, audio, captions, tracking, editing, Lottie).
from . import commands_extra  # noqa: E402,F401  (registers extra commands on import)
