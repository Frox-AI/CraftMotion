"""Project + layer model for CraftMotion.

Everything is plain JSON-able dicts so projects are human readable,
diffable and easy for AI agents to manipulate.
"""
from __future__ import annotations

import copy
import json
import threading
from typing import Any, Dict, List, Optional

from .util import now_ms, parse_vec

PROJECT_FORMAT = 1
DEFAULT_FPS = 30

LAYER_KINDS = ["video", "image", "text", "shape", "solid", "adjustment", "audio", "null",
               "precomp", "camera", "light"]

TRANSFORM_DEFAULTS = {
    "pos": [0.0, 0.0, 0.0],
    "anchor": [0.0, 0.0],
    "scale": [1.0, 1.0],
    "skew": [0.0, 0.0],
    "rot": [0.0],
    "opacity": [1.0],
}

# Which transform props are keyable / exposed in the UI and MCP tools
TRANSFORM_PROPS = {
    "pos": {"dims": 3, "label": "Position", "unit": "px"},
    "anchor": {"dims": 2, "label": "Anchor", "unit": "px"},
    "scale": {"dims": 2, "label": "Scale", "unit": "x"},
    "skew": {"dims": 2, "label": "Skew", "unit": "deg"},
    "rot": {"dims": 1, "label": "Rotation", "unit": "deg"},
    "opacity": {"dims": 1, "label": "Opacity", "unit": "0-1"},
}

BLEND_MODES = [
    "normal", "dissolve", "darken", "multiply", "color_burn", "linear_burn", "darker_color",
    "lighten", "screen", "color_dodge", "linear_dodge", "lighter_color", "overlay",
    "soft_light", "hard_light", "vivid_light", "linear_light", "pin_light", "hard_mix",
    "difference", "exclusion", "subtract", "divide", "average", "negation", "phoenix",
    "reflect", "glow", "freeze", "heat", "stamp", "xor", "hue", "saturation", "color",
    "luminosity", "grain_extract", "grain_merge",
]


class Project:
    """Container for the whole edit: settings, media references and layers."""

    def __init__(self, data: Optional[dict] = None):
        self.lock = threading.RLock()
        self.path: Optional[str] = None
        self.data: Dict[str, Any] = data if data is not None else self._new_data()
        self._normalize()

    # ------------------------------------------------------------ basics
    @staticmethod
    def _new_data(title: str = "Untitled", w: int = 1080, h: int = 1920) -> dict:
        return {
            "app": "CraftMotion",
            "format": PROJECT_FORMAT,
            "title": title,
            "created": now_ms(),
            "modified": now_ms(),
            "settings": {
                "width": w,
                "height": h,
                "fps": DEFAULT_FPS,
                "duration": 15000,
                "bg": "#FF000000",
            },
            "media": [],
            "layers": [],
            "comps": [],
            "markers": [],
            "motion_blur": {"enabled": False, "samples": 4, "shutter": 0.5},
            "camera": {"enabled": False, "x": None, "y": None, "z": 0, "zoom": 1.0,
                       "rotation": 0.0, "dof": {"enabled": False, "focus_z": 0, "strength": 6.0}},
            "meta": {"next_id": 1, "am_xml_warnings": []},
        }

    def _normalize(self) -> None:
        d = self.data
        d.setdefault("app", "CraftMotion")
        d.setdefault("format", PROJECT_FORMAT)
        d.setdefault("title", "Untitled")
        d.setdefault("settings", {})
        s = d["settings"]
        s.setdefault("width", 1080)
        s.setdefault("height", 1920)
        s.setdefault("fps", DEFAULT_FPS)
        s.setdefault("duration", 15000)
        s.setdefault("bg", "#FF000000")
        for key in ("width", "height", "duration"):
            s[key] = max(1, int(s[key]))
        s["fps"] = float(s["fps"])
        d.setdefault("media", [])
        d.setdefault("layers", [])
        d.setdefault("comps", [])
        d.setdefault("markers", [])
        d.setdefault("motion_blur", {"enabled": False, "samples": 4, "shutter": 0.5})
        d.setdefault("camera", {"enabled": False, "x": None, "y": None, "z": 0, "zoom": 1.0,
                                "rotation": 0.0, "dof": {"enabled": False, "focus_z": 0, "strength": 6.0}})
        d["camera"].setdefault("dof", {"enabled": False, "focus_z": 0, "strength": 6.0})
        for comp in d.get("comps", []):
            comp.setdefault("id", 0)
            comp.setdefault("name", "Comp")
            comp.setdefault("layers", [])
            comp.setdefault("width", s["width"])
            comp.setdefault("height", s["height"])
            comp.setdefault("duration", s["duration"])
            comp.setdefault("fps", s["fps"])
            comp.setdefault("bg", "#00000000")
            for layer in comp["layers"]:
                Project._normalize_layer(layer)
        d.setdefault("meta", {})
        d["meta"].setdefault("next_id", 1)
        d["meta"].setdefault("am_xml_warnings", [])
        # enforce sane layer defaults
        for layer in d["layers"]:
            self._normalize_layer(layer)
        # keep next_id ahead of everything
        max_id = max([int(l.get("id", 0)) for l in d["layers"]] + [0])
        d["meta"]["next_id"] = max(int(d["meta"].get("next_id", 1)), max_id + 1)

    @staticmethod
    def _normalize_layer(layer: dict) -> None:
        layer.setdefault("kind", "solid")
        layer.setdefault("name", f"{layer['kind'].title()} Layer")
        layer.setdefault("start", 0)
        layer.setdefault("end", 5000)
        layer.setdefault("hidden", False)
        layer.setdefault("locked", False)
        layer.setdefault("blend", "normal")
        layer.setdefault("parent", None)
        layer.setdefault("effects", [])
        layer.setdefault("keys", {})
        layer.setdefault("transform", {})
        layer.setdefault("masks", [])
        layer.setdefault("styles", [])
        layer.setdefault("expressions", {})
        layer.setdefault("audio_fx", [])
        t = layer["transform"]
        for k, dims in (("pos", 3), ("anchor", 2), ("scale", 2), ("skew", 2), ("rot", 1), ("opacity", 1)):
            t[k] = parse_vec(t.get(k, TRANSFORM_DEFAULTS[k]), dims=dims)
        layer.setdefault("ui", {})
        for eff in layer["effects"]:
            eff.setdefault("enabled", True)
            eff.setdefault("params", {})
            eff.setdefault("keys", {})
        # per-kind defaults
        kind = layer["kind"]
        if kind in ("video", "image"):
            layer.setdefault("media_id", None)
            layer.setdefault("source_in", 0)
            layer.setdefault("fit", "cover")
            layer.setdefault("speed", 1.0)
        elif kind == "audio":
            layer.setdefault("media_id", None)
            layer.setdefault("source_in", 0)
            layer.setdefault("volume", 1.0)
            layer.setdefault("fade_in", 0)
            layer.setdefault("fade_out", 0)
        elif kind == "text":
            layer.setdefault("text", "Your text here")
            layer.setdefault("size", 96)
            layer.setdefault("font", "DejaVuSans-Bold")
            layer.setdefault("color", "#FFFFFFFF")
            layer.setdefault("align", "center")
            layer.setdefault("wrap", 900)
            layer.setdefault("line_height", 1.15)
            layer.setdefault("stroke_color", "#00000000")
            layer.setdefault("stroke_width", 0)
            layer.setdefault("shadow", 0)
            layer.setdefault("text_animators", [])
        elif kind == "shape":
            layer.setdefault("shape", "rect")
            layer.setdefault("w", 400)
            layer.setdefault("h", 400)
            layer.setdefault("radius", 0)
            layer.setdefault("points", 5)
            layer.setdefault("inner", 0.5)
            layer.setdefault("color", "#FF4C8DFF")
            layer.setdefault("stroke_color", "#00000000")
            layer.setdefault("stroke_width", 0)
            layer.setdefault("path", [])
            layer.setdefault("trim", None)
            layer.setdefault("repeater", None)
        elif kind == "solid":
            layer.setdefault("color", "#FF202020")
            layer.setdefault("w", None)
            layer.setdefault("h", None)
        elif kind == "adjustment":
            layer.setdefault("color", "#00000000")
        elif kind == "precomp":
            layer.setdefault("comp_id", None)
            layer.setdefault("fit", "stretch")
        elif kind == "camera":
            layer.setdefault("zoom", 1.0)
            layer.setdefault("camera_fov", 50.0)

    def clone(self) -> "Project":
        p = Project(copy.deepcopy(self.data))
        p.path = None
        return p

    # ------------------------------------------------------------ access
    @property
    def settings(self) -> dict:
        return self.data["settings"]

    @property
    def layers(self) -> List[dict]:
        return self.data["layers"]

    @property
    def media(self) -> List[dict]:
        return self.data["media"]

    def slug(self) -> str:
        base = (self.data.get("title") or "craftmotion").strip().lower()
        keep = "".join(c if (c.isalnum() or c in "-_") else "-" for c in base)
        return keep.strip("-") or "craftmotion"

    @property
    def comps(self) -> List[dict]:
        return self.data.setdefault("comps", [])

    def get_comp(self, comp_id) -> Optional[dict]:
        for c in self.comps:
            if int(c.get("id", -1)) == int(comp_id if comp_id is not None else -999):
                return c
        return None

    def comp_as_project(self, comp: dict) -> "Project":
        """Lightweight Project view over a comp's layers (shares media)."""
        view = Project()
        view.data = {
            "app": "CraftMotion", "format": PROJECT_FORMAT,
            "title": comp.get("name", "Comp"), "created": now_ms(), "modified": now_ms(),
            "settings": {"width": int(comp.get("width") or self.width),
                         "height": int(comp.get("height") or self.height),
                         "fps": float(comp.get("fps") or self.fps),
                         "duration": int(comp.get("duration") or self.duration),
                         "bg": comp.get("bg", "#00000000")},
            "media": self.media, "layers": comp.get("layers", []),
            "comps": self.comps, "markers": [],
            "motion_blur": self.data.get("motion_blur"),
            "camera": self.data.get("camera"),
            "meta": {"next_id": self.data["meta"].get("next_id", 1), "am_xml_warnings": []},
        }
        view.path = self.path
        return view

    @property
    def width(self) -> int:
        return int(self.settings["width"])

    @property
    def height(self) -> int:
        return int(self.settings["height"])

    @property
    def fps(self) -> float:
        return float(self.settings["fps"])

    @property
    def duration(self) -> int:
        return int(self.settings["duration"])

    def new_id(self) -> int:
        nid = int(self.data["meta"].get("next_id", 1))
        while self.get_layer(nid) is not None or any(int(m.get("id", -1)) == nid for m in self.media):
            nid += 1
        self.data["meta"]["next_id"] = nid + 1
        return nid

    def get_layer(self, layer_id: int) -> Optional[dict]:
        for l in self.layers:
            if int(l.get("id", -1)) == int(layer_id):
                return l
        return None

    def require_layer(self, layer_id: int) -> dict:
        l = self.get_layer(layer_id)
        if l is None:
            raise KeyError(f"layer {layer_id} not found (existing ids: {[x['id'] for x in self.layers]})")
        return l

    def get_media(self, media_id) -> Optional[dict]:
        for m in self.media:
            if m.get("id") == media_id:
                return m
        return None

    def add_media(self, item: dict) -> dict:
        item = dict(item)
        if "id" not in item or item.get("id") is None:
            item["id"] = self.new_id()
        self.media.append(item)
        return item

    def add_layer(self, layer: dict, index: Optional[int] = None) -> dict:
        layer = dict(layer)
        if layer.get("id") is None:
            layer["id"] = self.new_id()
        self._normalize_layer(layer)
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(max(0, min(int(index), len(self.layers))), layer)
        self.touch()
        return layer

    def remove_layer(self, layer_id: int) -> bool:
        for i, l in enumerate(self.layers):
            if int(l.get("id", -1)) == int(layer_id):
                self.layers.pop(i)
                # orphan children become root
                for other in self.layers:
                    if other.get("parent") == layer_id:
                        other["parent"] = None
                self.touch()
                return True
        return False

    def layer_index(self, layer_id: int) -> int:
        for i, l in enumerate(self.layers):
            if int(l.get("id", -1)) == int(layer_id):
                return i
        return -1

    def touch(self) -> None:
        self.data["modified"] = now_ms()

    # ------------------------------------------------------------ io
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no save path given")
        if not path.lower().endswith(".cmproj"):
            path += ".cmproj"
        self.touch()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        self.path = path
        return path

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = Project(data)
        p.path = path
        return p

    # ------------------------------------------------------------ summary
    def summary(self, include_layers: bool = True) -> dict:
        out = {
            "title": self.data.get("title", "Untitled"),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_ms": self.duration,
            "duration_human": f"{self.duration/1000:.2f}s",
            "bg": self.settings.get("bg"),
            "media_count": len(self.media),
            "layer_count": len(self.layers),
            "path": self.path,
            "warnings": self.data["meta"].get("am_xml_warnings", [])[-8:],
        }
        if include_layers:
            out["layers"] = [layer_summary(l) for l in self.layers]
        return out


def layer_summary(layer: dict) -> dict:
    t = layer.get("transform", {})
    out = {
        "id": layer.get("id"),
        "name": layer.get("name"),
        "kind": layer.get("kind"),
        "start": layer.get("start"),
        "end": layer.get("end"),
        "hidden": layer.get("hidden", False),
        "blend": layer.get("blend", "normal"),
        "parent": layer.get("parent"),
        "opacity": round(float(t.get("opacity", [1])[0]), 3),
        "pos": [round(v, 1) for v in t.get("pos", [0, 0, 0])][:2],
        "scale": [round(v, 3) for v in t.get("scale", [1, 1])],
        "rot": round(float(t.get("rot", [0])[0]), 2),
        "effects": [e.get("type") for e in layer.get("effects", [])],
        "keyed": sorted(list((layer.get("keys") or {}).keys())),
        "masks": len(layer.get("masks") or []),
        "styles": [st.get("type") for st in layer.get("styles") or []],
        "expressions": sorted((layer.get("expressions") or {}).keys()),
    }
    kind = layer.get("kind")
    if kind in ("video", "image", "audio"):
        out["media_id"] = layer.get("media_id")
    if kind == "text":
        out["text"] = (layer.get("text") or "")[:80]
    if kind == "shape":
        out["shape"] = layer.get("shape")
    if kind == "precomp":
        out["comp_id"] = layer.get("comp_id")
    if kind == "camera":
        out["zoom"] = layer.get("zoom")
    if layer.get("text_animators"):
        out["text_animators"] = len(layer["text_animators"])
    return out
