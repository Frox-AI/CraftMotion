"""Alight Motion scene/preset XML import & export.

Implements the documented AM scene format (ffver 106/107/108):
  <scene> -> media*, bookmark*, elements* (shape|text|drawing|embedScene|audio|camera|nullobj)
Keyables write `value=` when static or <kf t="sec" v=".." e="cubicBezier .."/> children.
Effect ids are canonical `com.alightcreative.effects.*` ids, mapped onto
CraftMotion effect types where an equivalent exists.

Import is deliberately forgiving: unknown effects, shape templates, fonts and
media become recorded warnings instead of failures.
"""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from . import effects as FX
from .model import Project
from .util import now_ms, parse_color, parse_easing, to_argb

AM_VER = "1028425"
AM_BUILD = "com.alightcreative.motion/5.0.273.1028425"
MAX_FF = 106

TRANSFORM_MAP = {
    "location": ("pos", 3),
    "pivot": ("anchor", 2),
    "scale": ("scale", 2),
    "skew": ("skew", 2),
    "rotation": ("rot", 1),
    "opacity": ("opacity", 1),
}

SHAPE_MAP = {
    "rect": ("rect", {}),
    "quad": ("polygon", {"points": 4}),
    "roundrect": ("rect", {"_radius": True}),
    "circle": ("ellipse", {}),
    "ellipse": ("ellipse", {}),
    "star": ("star", {}),
    "poly": ("polygon", {}),
    "penta": ("polygon", {"points": 5}),
    "triangle": ("triangle", {}),
    "plus": ("plus", {}),
    "line": ("line", {}),
    "wideline": ("line", {}),
    "moon": ("ellipse", {"_approx": "moon"}),
    "pie": ("ellipse", {"_approx": "pie"}),
    "arc": ("ellipse", {"_approx": "arc"}),
    "teardrop": ("heart", {"_approx": "teardrop"}),
    "arrow": ("triangle", {"_approx": "arrow"}),
    "multifoil": ("star", {"_approx": "multifoil"}),
    "calloutrr": ("roundrect", {"_approx": "callout"}),
    "stamp": ("polygon", {"_approx": "stamp"}),
    "testshape": ("rect", {"_approx": "testshape"}),
}

BLEND_MAP = {
    "normal": "normal", "multiply": "multiply", "screen": "screen", "add": "add",
    "lighten": "lighten", "darken": "darken", "overlay": "overlay",
    "softlight": "soft_light", "hardlight": "hard_light",
    "colordodge": "color_dodge", "colorburn": "color_burn",
    "difference": "difference", "diff": "difference", "exclusion": "exclusion",
    "subtract": "subtract", "divide": "divide",
    "luminosity": "luminosity", "color": "color", "hue": "hue", "saturation": "saturation",
    "lightercolor": "lighten", "darkercolor": "darken",
    "colormultiply": "multiply", "softoverlay": "soft_light", "pinlight": "hard_light",
    "vividlight": "hard_light", "linearlight": "overlay", "linearburn": "color_burn",
    "lineardodge": "color_dodge", "colorburn ": "color_burn", "blend": "normal",
}
BLEND_EXPORT = {
    "normal": None, "multiply": "multiply", "screen": "screen", "add": "add",
    "overlay": "overlay", "soft_light": "softlight", "hard_light": "hardlight",
    "darken": "darken", "lighten": "lighten", "color_dodge": "colordodge",
    "color_burn": "colorburn", "difference": "difference", "exclusion": "exclusion",
    "subtract": "subtract", "divide": "divide", "luminosity": "luminosity",
    "color": "color", "hue": "hue", "saturation": "saturation",
}


class AMImportResult:
    def __init__(self):
        self.warnings: List[str] = []
        self.stats = {"layers": 0, "effects": 0, "media": 0, "keyframes": 0, "skipped": 0}

    def warn(self, msg: str) -> None:
        if len(self.warnings) < 500 and msg not in self.warnings:
            self.warnings.append(msg)


# ------------------------------------------------------------------ utilities

def _f(el: Optional[ET.Element], attr: str, default: float = 0.0) -> float:
    if el is None:
        return default
    try:
        return float(el.get(attr, default))
    except (TypeError, ValueError):
        return default


def _vec(s: Optional[str], dims: int = 2) -> List[float]:
    if not s:
        return [0.0] * dims
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return [0.0] * dims
    while len(vals) < dims:
        vals.append(vals[-1] if vals else 0.0)
    return vals[:dims]


def _kf_track(el: Optional[ET.Element], dims: int, res: AMImportResult) -> Optional[List[dict]]:
    """Parse a keyable element -> list of our keyframes (ms), or None if static."""
    if el is None:
        return None
    kfs = el.findall("kf")
    if not kfs:
        return None
    out = []
    for kf in kfs:
        try:
            t = int(round(float(kf.get("t", "0")) * 1000.0))
            v = _vec(kf.get("v", ""), dims)
            e = parse_easing(kf.get("e"))
            out.append({"t": t, "v": v, "e": e})
        except (TypeError, ValueError):
            continue
    if len(out) < 2:
        return None
    out.sort(key=lambda k: k["t"])
    res.stats["keyframes"] += len(out)
    return out


def _static_vec(el: Optional[ET.Element], dims: int, default: List[float]) -> List[float]:
    if el is None:
        return list(default)
    return _vec(el.get("value", ""), dims) if el.get("value") else list(default)


def _font_name(desc: str) -> str:
    if not desc:
        return "DejaVu Sans"
    m = re.search(r"name=([^&]+)", desc)
    if m:
        return unquote(m.group(1)).replace("+", " ")
    return desc


def _media_kind(mime: str, filename: str) -> str:
    mime = (mime or "").lower()
    ext = os.path.splitext(filename or "")[1].lower()
    if mime.startswith("video") or ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
        return "video"
    if mime.startswith("audio") or ext in (".mp3", ".wav", ".aac", ".m4a", ".ogg", ".opus"):
        return "audio"
    return "image"


def _resolve_media_path(xml_dir: str, media_el: ET.Element) -> Optional[str]:
    names = []
    for key in ("filename", "uri", "title"):
        v = media_el.get(key)
        if not v:
            continue
        if key == "uri" and ("://" in v):
            try:
                v = os.path.basename(unquote(urlparse(v).path))
            except Exception:
                continue
        names.append(os.path.basename(v))
    wanted = {n.lower() for n in names if n}
    if not wanted:
        return None
    try:
        for root, _dirs, files in os.walk(xml_dir):
            for fn in files:
                if fn.lower() in wanted:
                    return os.path.join(root, fn)
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ import

def import_am_scene(xml_path: str, existing: Optional[Project] = None,
                    offset_ms: int = 0) -> Tuple[Project, AMImportResult]:
    res = AMImportResult()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "scene":
        raise ValueError("not an Alight Motion scene: root element is <%s>" % root.tag)
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    proj = existing or Project()
    if existing is None:
        w = int(_f(root, "width", 1080))
        h = int(_f(root, "height", 1080))
        total = int(_f(root, "totalTime", 5000))
        fps = _f(root, "fps", 30.0) or 30.0
        bg = root.get("bgcolor", "#FFD8D8D8")
        proj.data["title"] = root.get("title", "AM Import")
        proj.settings.update({"width": w, "height": h, "duration": max(500, total), "fps": fps, "bg": bg})
        proj.data["meta"]["source"] = {"format": "alight-motion", "ffver": root.get("ffver", "106"),
                                       "file": os.path.basename(xml_path)}
    else:
        # merge preset: grow the timeline if needed
        shift = int(offset_ms)
        total = int(_f(root, "totalTime", 0)) + shift
        if total > proj.duration:
            proj.settings["duration"] = total

    media_by_sig: Dict[str, dict] = {}
    for m in root.findall("media"):
        filename = m.get("filename") or os.path.basename(m.get("uri", "") or "")
        path = _resolve_media_path(xml_dir, m)
        kind = _media_kind(m.get("type", ""), filename)
        item = {
            "id": None,
            "path": path or "",
            "filename": filename,
            "name": m.get("title") or filename,
            "kind": kind,
            "width": int(_f(m, "width", 0)),
            "height": int(_f(m, "height", 0)),
            "duration_ms": int(_f(m, "duration", 0)),
            "has_audio": kind in ("video", "audio"),
            "sig": m.get("sig") or os.path.splitext(filename)[0],
            "missing": path is None,
            "am_uri": m.get("uri"),
            "project_dir": xml_dir,
        }
        item = proj.add_media(item)
        media_by_sig[item["sig"].lower()] = item
        media_by_sig[filename.lower()] = item
        res.stats["media"] += 1
        if path is None:
            res.warn(f"media not found on disk: {filename} (re-link it in the Media panel)")

    def find_media(uri: Optional[str]) -> Optional[dict]:
        if not uri:
            return None
        base = os.path.basename(unquote(urlparse(uri).path)) if "://" in uri else os.path.basename(uri)
        stem = os.path.splitext(base)[0].lower()
        return media_by_sig.get(base.lower()) or media_by_sig.get(stem) or media_by_sig.get(uri.lower())

    scene_offset = int(offset_ms)
    if existing is not None:
        scene_offset += int(_f(root, "retimeIn", 0))

    elements = list(root)
    idmap: Dict[str, int] = {}
    imported: List[dict] = []
    # AM `location` values are absolute canvas pixels (its own minimal example
    # places a 1080-wide backdrop at 540,540), which matches our `pos` meaning.
    center = None
    for el in elements:
        if el.tag in ("media", "bookmark"):
            continue
        try:
            layer = _import_element(proj, el, find_media, res, scene_offset, xml_dir,
                                    existing is not None, center)
            if layer is not None:
                imported.append(layer)
                am_id = el.get("id")
                if am_id is not None:
                    idmap[str(am_id)] = int(layer["id"])
        except Exception as exc:
            res.warn(f"element <{el.tag} id={el.get('id')}> skipped: {exc}")
            res.stats["skipped"] += 1
    # resolve parent references (AM ids are per-scene; ours are remapped)
    for layer in imported:
        am_parent = layer.pop("_am_parent", None)
        if am_parent is not None:
            layer["parent"] = idmap.get(str(am_parent))

    for bm in root.findall("bookmark"):
        t = int(_f(bm, "t", 0)) + scene_offset
        proj.data.setdefault("markers", []).append({"t": t, "label": "AM bookmark"})

    proj.data["meta"].setdefault("am_xml_warnings", [])
    proj.data["meta"]["am_xml_warnings"].extend(res.warnings[:80])
    proj._normalize()
    proj.touch()
    return proj, res


def _import_element(proj: Project, el: ET.Element, find_media, res: AMImportResult,
                    offset: int, xml_dir: str, merging: bool,
                    center: Optional[List[float]] = None) -> Optional[dict]:
    tag = el.tag
    layer: dict = {
        "id": None,
        "name": el.get("label") or el.get("tag") or tag.title(),
        "start": int(_f(el, "startTime", 0)) + offset,
        "end": int(_f(el, "endTime", 5000)) + offset,
        "hidden": el.get("hidden", "false").lower() == "true",
        "blend": BLEND_MAP.get((el.get("blending") or "normal").lower(), "normal"),
        "effects": [],
        "keys": {},
        "transform": {},
        "ui": {"source": "alight-motion"},
    }
    parent = el.get("parent")
    if parent:
        layer["_am_parent"] = parent

    # ---- transform
    tf = el.find("transform")
    transform = {}
    keys: Dict[str, List[dict]] = {}
    for am_name, (our_name, dims) in TRANSFORM_MAP.items():
        node = tf.find(am_name) if tf is not None else None
        default = {"pos": [0, 0, 0], "anchor": [0, 0], "scale": [1, 1], "skew": [0, 0],
                   "rot": [0], "opacity": [1]}[our_name]
        track = _kf_track(node, dims, res)
        if track:
            keys[our_name] = track
            transform[our_name] = track[0]["v"]
        else:
            transform[our_name] = _static_vec(node, dims, default)
    layer["transform"] = transform
    layer["keys"] = keys

    # ---- fill colour / gradient
    fill_el = el.find("fillColor")
    if fill_el is not None:
        fill_kfs = _kf_track(fill_el, 4, res)
        if fill_kfs:
            layer["keys"]["color"] = []
            for k in fill_kfs:
                layer["keys"]["color"].append({"t": k["t"], "v": [round(c, 4) for c in k["v"]], "e": k["e"]})
        val = fill_el.get("value")
        if val:
            layer["color"] = to_argb(parse_color(val))
    grad = el.find("gradient")
    if grad is not None:
        # AM renders the gradient as the fill, so it wins over fillColor
        layer["color"] = to_argb(parse_color(grad.get("startColor", "#FF000000")))
        layer["color2"] = to_argb(parse_color(grad.get("endColor", "#FFFFFFFF")))
        start = _vec(grad.get("start", "0,0"), 2)
        end = _vec(grad.get("end", "1,1"), 2)
        dx, dy = end[0] - start[0], end[1] - start[1]
        layer["gradient_angle"] = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        layer["fill_type"] = grad.get("type", "linear")

    # ---- effects
    for eff_el in el.findall("effect"):
        eid = eff_el.get("id", "")
        etype = FX.resolve_type(eid)
        eff = {
            "type": etype or eid.split(".")[-1],
            "src_id": eid,
            "enabled": eff_el.get("hidden", "false").lower() != "true",
            "params": {},
            "keys": {},
        }
        am_suffix = eid.split(".")[-1] if "." in eid else eid
        # AM often spells the same knob differently ("strength" vs "amount"); a
        # straight rename keeps imported presets looking like their source.
        rename = dict(FX.AM_PARAM_MAP.get(am_suffix) or {})
        known = set((FX.EFFECTS.get(etype).params if etype in FX.EFFECTS else {}).keys())
        for prop in eff_el.findall("property"):
            pname = prop.get("name", "")
            pname = rename.get(pname, pname)
            if etype in FX.EFFECTS and pname not in known and rename.get(pname) is None:
                pass  # keep unknown names as metadata below
            ptype = prop.get("type", "float")
            pkeys = _kf_track(prop, 4 if ptype == "color" else 2 if ptype.startswith("vec") else 1, res)
            if pkeys:
                eff["keys"][pname] = pkeys
                continue
            val = prop.get("value")
            if val is None:
                val = (prop.text or "").strip()
            if ptype == "float":
                try:
                    pval = float(val)
                except (TypeError, ValueError):
                    eff["params"][pname] = val
                else:
                    if etype in FX.EFFECTS and pname in FX.EFFECTS[etype].params:
                        prm = FX.EFFECTS[etype].params[pname]
                        if prm.kind == "float" and prm.hi is not None:
                            # AM sometimes ships 0..100 where we expect 0..1
                            if 1.0 < prm.hi <= 2.5 and 1.0 < abs(pval) <= 100.0:
                                pval = pval / 100.0
                            pval = max(float(prm.lo), min(float(prm.hi), pval))
                        elif prm.kind == "int":
                            pval = int(round(pval))
                        elif prm.kind == "bool":
                            pval = bool(pval)
                    eff["params"][pname] = pval
            elif ptype in ("int",):
                try:
                    eff["params"][pname] = int(float(val))
                except (TypeError, ValueError):
                    eff["params"][pname] = val
            elif ptype == "bool":
                eff["params"][pname] = str(val).lower() == "true"
            elif ptype == "color":
                eff["params"][pname] = to_argb(parse_color(val))
            elif ptype.startswith("vec"):
                eff["params"][pname] = _vec(str(val), int(ptype[3:]) if ptype[3:].isdigit() else 2)
            else:
                eff["params"][pname] = val
        if not etype and am_suffix == "adjustment_layer":
            # AM models adjustment layers as a pass-through effect; we have a real kind
            layer["kind"] = "adjustment"
            res.stats["effects"] += 1
            continue
        if not etype and am_suffix == "textspacing":
            sel = {p.get("name"): p.get("value") for p in eff_el.findall("property")}
            try:
                layer["tracking"] = float(sel.get("letterspacing", 0) or 0) * float(layer.get("size", 96)) * 0.35
            except (TypeError, ValueError):
                pass
            try:
                layer["line_height"] = float(sel.get("linespacing", 1) or 1)
            except (TypeError, ValueError):
                pass
            res.stats["effects"] += 1
            continue
        if not etype:
            res.warn(f"effect not supported yet, kept as metadata: {eid}")
        else:
            res.stats["effects"] += 1
        layer["effects"].append(eff)

    # ---- per-tag specifics
    if tag == "shape":
        _shape_from_el(el, layer, res)
    elif tag == "text":
        _text_from_el(el, layer, res)
    elif tag == "drawing":
        _drawing_from_el(el, layer, res)
    elif tag == "audio":
        item = find_media(el.get("src"))
        layer["kind"] = "audio"
        layer["media_id"] = item["id"] if item else None
        layer["name"] = layer["name"] or "Audio"
        gain = el.find("gain")
        if gain is not None:
            layer["volume"] = _f(gain, "value", 1.0)
    elif tag == "camera":
        layer["kind"] = "null"
        layer["name"] = layer["name"] or "Camera"
        res.warn("camera approximated as a null transform (no 3D camera in v1)")
    elif tag == "nullobj":
        layer["kind"] = "null"
    elif tag == "embedScene":
        _import_embed(proj, el, find_media, res, offset, xml_dir)
        return None
    else:
        layer["kind"] = "solid"
        res.warn(f"unknown element <{tag}> imported as solid")

    # media fill on shapes/text
    if layer.get("kind") in (None, "shape", "text") and (el.get("fillImage") or (el.get("fillType") == "media")):
        item = find_media(el.get("fillImage") or el.get("src"))
        if item:
            layer["media_id"] = item["id"]
            layer["kind"] = "image" if item.get("kind") == "image" else "video"
            layer.setdefault("fit", "stretch" if el.get("mediaFillMode") in ("stretch",) else "cover")
            layer.setdefault("w", int(item.get("width") or 400))
            layer.setdefault("h", int(item.get("height") or 400))

    # speed / retime
    sp = el.get("speed")
    if sp:
        try:
            layer["speed"] = float(sp)
        except ValueError:
            pass
    if el.find("speedMap") is not None:
        res.warn(f"'{layer['name']}': speedMap retiming is approximated with speed=1")

    layer["kind"] = layer.get("kind") or "solid"
    if el.get("clip") or el.get("clippingMask", "false").lower() == "true":
        layer["clip"] = True

    added = proj.add_layer(layer)
    res.stats["layers"] += 1
    return added


def _shape_from_el(el: ET.Element, layer: dict, res: AMImportResult) -> None:
    layer["kind"] = "shape"
    slug = (el.get("s") or "").strip().lstrip(".")
    params: Dict[str, List[float]] = {}
    for prop in el.findall("property"):
        name = prop.get("name", "")
        if prop.findall("kf"):
            res.warn(f"shape property keyframes '{name}' imported as static")
            first = prop.find("kf")
            params[name] = _vec(first.get("v") if first is not None else prop.get("value"), 2)
        else:
            params[name] = _vec(prop.get("value"), 2)

    contour = None
    param_el = el.find("parameter")
    if param_el is not None:
        c = param_el.find("contour")
        if c is not None:
            contour = c.get("d")

    def size_of(default=(100.0, 100.0)):
        s = params.get("size") or params.get("radius") or []
        if len(s) >= 2 and s[0] > 0:
            return [s[0], s[1]]
        if len(s) == 1 and s[0] > 0:
            return [s[0] * 2, s[0] * 2]
        return [default[0], default[1]]

    if contour:
        layer["shape"] = "path"
        pts, size = _parse_svg_path_points(contour)
        layer["path"] = pts
        layer["w"], layer["h"] = size
        layer["name"] = layer["name"] or "Shape"
        layer["color"] = layer.get("color", "#FFFFFFFF")
    elif slug in SHAPE_MAP:
        our_shape, extra = SHAPE_MAP[slug]
        layer["shape"] = our_shape
        size = size_of()
        if extra.get("_radius"):
            r = params.get("cornerRadius") or params.get("radius") or [0]
            layer["radius"] = float(r[0]) * 2.0 if r else 0.0
        if slug in ("star",):
            pts = params.get("points") or params.get("numPoints") or [5]
            layer["points"] = int(round(pts[0])) or 5
            inner = params.get("innerRadius") or params.get("inner") or []
            outer = params.get("outerRadius") or size_of()
            if inner and outer and float(outer[0]) > 0:
                layer["inner"] = max(0.08, min(0.95, float(inner[0]) / float(outer[0])))
        if slug in ("poly", "penta"):
            pts = params.get("points") or params.get("numPoints") or [int(extra.get("points", 6)) if "points" in extra else 6]
            layer["points"] = int(round(pts[0])) or 6
        if slug in ("line", "wideline"):
            layer["stroke_width"] = float((params.get("thickness") or [8])[0])
        if extra.get("_approx"):
            res.warn(f"shape '{slug}' approximated with nearest CraftMotion shape ({our_shape})")
            layer["name"] = f"{layer['name']} ({slug})"
        layer["w"], layer["h"] = max(2.0, size[0]), max(2.0, size[1])
        # AM shape units are 100x100 by default; transform scale carries real size.
        if not params.get("size"):
            layer["w"] = layer["h"] = 100.0
    else:
        layer["shape"] = "rect"
        layer["w"] = layer["h"] = 100.0
        if slug:
            res.warn(f"unknown shape template '.{slug}' -> rectangle placeholder")
    layer.setdefault("w", 100)
    layer.setdefault("h", 100)
    if not layer.get("color"):
        layer["color"] = "#FFFFFFFF"

    border = el.find("border")
    if border is not None:
        layer["stroke_color"] = to_argb(parse_color(border.get("color", "#FFFFFFFF")))
        try:
            layer["stroke_width"] = float(border.get("size", "4"))
        except ValueError:
            layer["stroke_width"] = 4.0
    shadow = el.find("shadow")
    if shadow is not None and not any(e.get("src_id") == "com.alightcreative.effects.shadow" for e in layer.get("effects", [])):
        try:
            layer["effects"].append({
                "type": "drop_shadow", "src_id": "shadow",
                "params": {
                    "offset_x": _vec(shadow.get("offset", "0,8"), 2)[0],
                    "offset_y": _vec(shadow.get("offset", "0,8"), 2)[1],
                    "blur": float(shadow.get("size", "8")),
                    "opacity": float(shadow.get("opacity", "0.5")),
                    "color": to_argb(parse_color(shadow.get("color", "#FF000000"))),
                }, "keys": {}, "enabled": True,
            })
        except ValueError:
            pass


def _parse_svg_path_points(d: str) -> Tuple[List[List[float]], List[float]]:
    """Very small SVG path parser: samples M/L/C/Q/Z into a normalized polygon."""
    tokens = re.findall(r"[MLCQZmlcqz]|-?\d*\.?\d+(?:e-?\d+)?", d or "")
    pts: List[Tuple[float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    i = 0
    cmd = "M"

    def nxt():
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"[MLCQZmlcqz]", tok):
            cmd = tok
            i += 1
            if cmd.upper() == "Z":
                cur = start
                pts.append(cur)
                continue
        try:
            if cmd.upper() == "M":
                cur = (nxt(), nxt())
                start = cur
                pts.append(cur)
                cmd = "L" if cmd == "M" else "l"
            elif cmd.upper() == "L":
                cur = (nxt(), nxt())
                pts.append(cur)
            elif cmd.upper() == "C":
                x1, y1, x2, y2, x, y = nxt(), nxt(), nxt(), nxt(), nxt(), nxt()
                px, py = cur
                for s in range(1, 9):
                    t = s / 8.0
                    bx = (1 - t) ** 3 * px + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t * t * x2 + t ** 3 * x
                    by = (1 - t) ** 3 * py + 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t * t * y2 + t ** 3 * y
                    pts.append((bx, by))
                cur = (x, y)
            elif cmd.upper() == "Q":
                x1, y1, x, y = nxt(), nxt(), nxt(), nxt()
                px, py = cur
                for s in range(1, 9):
                    t = s / 8.0
                    bx = (1 - t) ** 2 * px + 2 * (1 - t) * t * x1 + t * t * x
                    by = (1 - t) ** 2 * py + 2 * (1 - t) * t * y1 + t * t * y
                    pts.append((bx, by))
                cur = (x, y)
            else:
                i += 1
        except IndexError:
            break
    if not pts:
        return [], [100.0, 100.0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w = max(1.0, maxx - minx)
    h = max(1.0, maxy - miny)
    norm = [[(x - minx) / w, (y - miny) / h] for x, y in pts]
    return norm, [w, h]


def _text_from_el(el: ET.Element, layer: dict, res: AMImportResult) -> None:
    layer["kind"] = "text"
    try:
        layer["size"] = float(el.get("size", "72"))
    except ValueError:
        layer["size"] = 72.0
    desc = el.get("font") or ""
    layer["font"] = _font_name(desc)
    if layer["font"] and layer["font"].lower() not in ("dejavu sans", "default"):
        if not re.match(r"^(arial|dejavu|helvetica|roboto|impact|verdana|times)", layer["font"].lower()):
            res.warn(f"font '{layer['font']}' substituted with bundled DejaVu Sans")
    try:
        layer["wrap"] = float(el.get("wrapWidth", "512"))
    except ValueError:
        layer["wrap"] = 512.0
    layer["align"] = (el.get("align") or "left").lower()
    content = el.find("content")
    layer["text"] = (content.text or "") if content is not None else (el.get("text") or "Text")
    if not layer.get("color"):
        layer["color"] = "#FFFFFFFF"
    border = el.find("border")
    if border is not None:
        layer["stroke_color"] = to_argb(parse_color(border.get("color", "#FF000000")))
        try:
            layer["stroke_width"] = float(border.get("size", "0"))
        except ValueError:
            layer["stroke_width"] = 0.0
    layer["name"] = layer["name"] or "Text"


def _drawing_from_el(el: ET.Element, layer: dict, res: AMImportResult) -> None:
    layer["kind"] = "shape"
    layer["shape"] = "path"
    strokes = el.findall("stroke")
    if not strokes:
        layer["w"] = layer["h"] = 100
        layer["path"] = []
        res.warn("drawing element had no strokes")
    else:
        st = strokes[0]
        raw = st.get("points", "")
        pts: List[Tuple[float, float]] = []
        for chunk in raw.split(";"):
            parts = chunk.split(",")
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        xs = [p[0] for p in pts] or [0]
        ys = [p[1] for p in pts] or [0]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        w, h = max(1.0, maxx - minx), max(1.0, maxy - miny)
        layer["path"] = [[(x - minx) / w, (y - miny) / h] for x, y in pts]
        layer["closed"] = False
        layer["w"], layer["h"] = w, h
        layer["color"] = to_argb(parse_color(st.get("color", "#FF000000")))
        try:
            layer["stroke_width"] = float(st.get("width", "4"))
        except ValueError:
            layer["stroke_width"] = 4.0
        layer["name"] = layer["name"] or "Drawing"
        if len(strokes) > 1:
            res.warn(f"drawing with {len(strokes)} strokes imported using the first stroke only")


def _import_embed(proj: Project, el: ET.Element, find_media, res: AMImportResult,
                  offset: int, xml_dir: str) -> None:
    """Nested scene: import its elements, parented to a null carrying the embed transform."""
    inner = el.find("scene")
    if inner is None:
        res.warn("embedScene without inner <scene> skipped")
        return
    null_layer = {
        "id": None, "name": (el.get("label") or "Nested Scene") + " [group]",
        "kind": "null",
        "start": int(_f(el, "startTime", 0)) + offset,
        "end": int(_f(el, "endTime", 5000)) + offset,
        "transform": {}, "keys": {}, "effects": [], "hidden": False,
    }
    tf = el.find("transform")
    for am_name, (our_name, dims) in TRANSFORM_MAP.items():
        node = tf.find(am_name) if tf is not None else None
        default = {"pos": [0, 0, 0], "anchor": [0, 0], "scale": [1, 1], "skew": [0, 0],
                   "rot": [0], "opacity": [1]}[our_name]
        track = _kf_track(node, dims, res)
        if track:
            null_layer["keys"][our_name] = track
            null_layer["transform"][our_name] = track[0]["v"]
        else:
            null_layer["transform"][our_name] = _static_vec(node, dims, default)
    null = proj.add_layer(null_layer)
    res.stats["layers"] += 1
    start = int(_f(el, "startTime", 0)) + offset
    before = {id(l) for l in proj.layers}
    for child in list(inner):
        if child.tag in ("media", "bookmark"):
            continue
        try:
            _import_element(proj, child, find_media, res, start, xml_dir, True)
        except Exception as exc:
            res.warn(f"nested element skipped: {exc}")
    for layer in proj.layers:
        if id(layer) not in before and layer.get("parent") is None:
            layer["parent"] = null["id"]
    res.warn(f"nested scene '{null_layer['name']}' flattened; inner layers follow the group transform")


# ------------------------------------------------------------------ export

def export_am_scene(project: Project, out_path: str, title: Optional[str] = None) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    root = ET.Element("scene", {
        "title": title or project.data.get("title", "CraftMotion Export"),
        "width": str(project.width),
        "height": str(project.height),
        "exportWidth": str(project.width),
        "exportHeight": str(project.height),
        "bgcolor": str(project.settings.get("bg", "#FF000000")),
        "totalTime": str(project.duration),
        "fps": _fmt_num(project.fps),
        "modifiedTime": str(now_ms()),
        "amver": AM_VER,
        "ffver": str(MAX_FF),
        "am": AM_BUILD,
        "amplatform": "android",
    })
    # media declarations
    for item in project.media:
        sig = item.get("sig") or re.sub(r"[^A-Za-z0-9]", "", str(item.get("filename") or "media"))[:12]
        ext = os.path.splitext(item.get("filename") or "")[1] or ".bin"
        ET.SubElement(root, "media", {
            "uri": f"am-internal:///{sig}{ext}",
            "filename": f"{sig}{ext.lstrip('.')}",
            "title": str(item.get("name") or item.get("filename") or "media"),
            "type": _mime_for(item),
            "sig": sig,
            "width": str(int(item.get("width") or 0)),
            "height": str(int(item.get("height") or 0)),
            "duration": str(int(item.get("duration_ms") or 0)),
        })
    for marker in project.data.get("markers", []):
        ET.SubElement(root, "bookmark", {"t": str(int(marker.get("t", 0)))})

    for layer in project.layers:
        _export_layer(root, project, layer, warnings)

    _indent(root)
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, warnings


def _mime_for(item: dict) -> str:
    kind = item.get("kind")
    ext = os.path.splitext(item.get("filename", ""))[1].lower().lstrip(".")
    if kind == "image":
        return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext, "image/png")
    if kind == "audio":
        return {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "aac": "audio/aac"}.get(ext, "audio/mpeg")
    return {"mov": "video/quicktime", "webm": "video/webm", "mkv": "video/x-matroska"}.get(ext, "video/mp4")


def _fmt_num(v: float) -> str:
    return ("%.6f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def _write_keyable(parent: ET.Element, tag: str, layer: dict, prop: str, dims: int,
                   default: List[float], warnings: List[str]) -> ET.Element:
    node = ET.SubElement(parent, tag)
    keys = (layer.get("keys") or {}).get(prop)
    if keys:
        for k in sorted(keys, key=lambda x: x["t"]):
            attrs = {"t": _fmt_num(k["t"] / 1000.0), "v": ",".join(_fmt_num(v) for v in _pad(k["v"], dims))}
            e = k.get("e")
            if e == "hold":
                attrs["e"] = "hold"
            elif isinstance(e, (tuple, list)) and len(e) == 4:
                attrs["e"] = "cubicBezier " + " ".join(_fmt_num(float(x)) for x in e)
            ET.SubElement(node, "kf", attrs)
    else:
        base = layer.get("transform", {}).get(prop, default)
        vals = _pad(base if isinstance(base, (list, tuple)) else [base], dims)
        node.set("value", ",".join(_fmt_num(float(v)) for v in vals))
    return node


def _pad(vals, dims: int) -> List[float]:
    vals = [float(v) for v in vals]
    while len(vals) < dims:
        vals.append(vals[-1] if vals else 0.0)
    return vals[:dims]


def _export_layer(root: ET.Element, project: Project, layer: dict, warnings: List[str]) -> None:
    kind = layer.get("kind")
    tag = {"video": "shape", "image": "shape", "solid": "shape", "shape": "shape",
           "text": "text", "adjustment": "shape", "audio": "audio", "null": "nullobj"}.get(kind, "shape")
    attrs = {
        "id": str(int(layer.get("id", 0))),
        "label": str(layer.get("name", "Layer")),
        "startTime": str(int(layer.get("start", 0))),
        "endTime": str(int(layer.get("end", project.duration))),
    }
    if layer.get("hidden"):
        attrs["hidden"] = "true"
    blend = BLEND_EXPORT.get(str(layer.get("blend", "normal")))
    if blend:
        attrs["blending"] = blend
    if layer.get("parent") is not None:
        attrs["parent"] = str(int(layer["parent"]))
    if kind in ("video", "image"):
        item = project.get_media(layer.get("media_id"))
        if item:
            sig = item.get("sig") or re.sub(r"[^A-Za-z0-9]", "", str(item.get("filename") or "media"))[:12]
            ext = os.path.splitext(item.get("filename") or "")[1] or ".bin"
            attrs["fillType"] = "media"
            attrs["fillVideo" if kind == "video" else "fillImage"] = f"am-internal:///{sig}{ext}"
            attrs["mediaFillMode"] = "stretch" if layer.get("fit") == "stretch" else "fit"
        else:
            attrs["fillType"] = "color"
            warnings.append(f"layer '{layer.get('name')}' references missing media; exported as rectangle")
    elif kind == "audio":
        item = project.get_media(layer.get("media_id"))
        attrs["src"] = f"am-internal:///{(item or {}).get('sig', 'audio.mp3')}"

    el = ET.SubElement(root, tag, attrs)

    # transform
    tf = ET.SubElement(el, "transform")
    _write_keyable(tf, "location", layer, "pos", 3, [0, 0, 0], warnings)
    _write_keyable(tf, "pivot", layer, "anchor", 2, [0, 0], warnings)
    _write_keyable(tf, "scale", layer, "scale", 2, [1, 1], warnings)
    _write_keyable(tf, "skew", layer, "skew", 2, [0, 0], warnings)
    _write_keyable(tf, "rotation", layer, "rot", 1, [0], warnings)
    _write_keyable(tf, "opacity", layer, "opacity", 1, [1], warnings)

    # content
    if kind == "text":
        el.set("size", _fmt_num(float(layer.get("size", 72))))
        el.set("font", "googlefonts?name=" + str(layer.get("font", "DejaVu Sans")).replace(" ", "+"))
        el.set("wrapWidth", _fmt_num(float(layer.get("wrap", 512))))
        el.set("align", str(layer.get("align", "left")))
        ET.SubElement(el, "fillColor", {"value": to_argb(parse_color(layer.get("color", "#FFFFFFFF")))})
        content = ET.SubElement(el, "content")
        content.text = str(layer.get("text", ""))
    elif kind == "audio":
        ET.SubElement(el, "gain", {"value": _fmt_num(float(layer.get("volume", 1.0)))})
    elif kind in ("shape", "solid", "adjustment", "video", "image"):
        slug = {("rect", "solid"): ".rect", "ellipse": ".circle", "circle": ".circle", "star": ".star",
                "polygon": ".poly", "triangle": ".triangle", "plus": ".plus", "heart": ".teardrop",
                "line": ".line", "diamond": ".quad", "ring": ".circle"}.get(layer.get("shape", "rect"), ".rect")
        if kind in ("solid", "adjustment"):
            slug = ".rect"
        el.set("fillType", "color")
        el.set("s", slug)
        if kind != "video" and kind != "image":
            ET.SubElement(el, "fillColor", {"value": to_argb(parse_color(layer.get("color", "#FFFFFFFF")))})
        if layer.get("color2"):
            ET.SubElement(el, "gradient", {
                "type": "linear",
                "startColor": to_argb(parse_color(layer.get("color", "#FF000000"))),
                "endColor": to_argb(parse_color(layer.get("color2"))),
                "start": "0,0", "end": "1,1",
            })
        if kind == "shape" and layer.get("shape") == "path":
            slim = {k: v for k, v in el.attrib.items() if k != "s"}
            el.attrib.clear()
            el.attrib.update(slim)
            param = ET.SubElement(el, "parameter")
            pts = layer.get("path") or []
            w = max(1.0, float(layer.get("w", 100)))
            h = max(1.0, float(layer.get("h", 100)))
            if pts:
                d = "M" + " L".join(f"{_fmt_num(p[0]*w)},{_fmt_num(p[1]*h)}" for p in pts) + " Z"
                ET.SubElement(param, "contour", {"d": d})
        elif kind in ("solid", "adjustment", "video", "image") and kind != "video":
            if kind in ("solid", "adjustment"):
                ET.SubElement(el, "property", {"name": "size", "type": "vec2",
                                               "value": f"{project.width},{project.height}"})
        else:
            ET.SubElement(el, "property", {"name": "size", "type": "vec2",
                                           "value": f"{_fmt_num(float(layer.get('w', 100)))},{_fmt_num(float(layer.get('h', 100)))}"})
            if layer.get("radius"):
                ET.SubElement(el, "property", {"name": "cornerRadius", "type": "float",
                                               "value": _fmt_num(float(layer['radius']))})
            if layer.get("shape") in ("star", "polygon"):
                ET.SubElement(el, "property", {"name": "points", "type": "int",
                                               "value": str(int(layer.get("points", 5)))})
                if layer.get("shape") == "star":
                    ET.SubElement(el, "property", {"name": "innerRadius", "type": "float",
                                                   "value": _fmt_num(float(layer.get("inner", 0.5)) * 50)})

    # effects
    if kind == "adjustment":
        lift = ET.SubElement(el, "effect", {"id": "com.alightcreative.effects.lift", "locallyApplied": "true"})
        ET.SubElement(lift, "property", {"name": "fill", "type": "float", "value": "0"})
    for eff in layer.get("effects", []):
        if not eff.get("enabled", True):
            continue
        etype = FX.resolve_type(eff.get("type", ""))
        spec = FX.EFFECTS.get(etype) if etype else None
        am_id = None
        if spec and spec.am_ids:
            am_id = spec.am_ids[0]
        if not am_id:
            warnings.append(f"effect '{eff.get('type')}' has no Alight Motion equivalent; skipped")
            continue
        e_el = ET.SubElement(el, "effect", {"id": am_id, "locallyApplied": "true"})
        params = eff.get("params", {})
        declared = spec.params if spec else {}
        for name, prm in declared.items():
            val = params.get(name, prm.default)
            keys = (eff.get("keys") or {}).get(name)
            if prm.kind == "color":
                ET.SubElement(e_el, "property", {"name": name, "type": "color", "value": to_argb(parse_color(val))})
            elif prm.kind == "bool":
                ET.SubElement(e_el, "property", {"name": name, "type": "bool", "value": "true" if val else "false"})
            elif keys:
                node = ET.SubElement(e_el, "property", {"name": name, "type": "float"})
                for k in keys:
                    ET.SubElement(node, "kf", {"t": _fmt_num(k["t"] / 1000.0), "v": _fmt_num(float(k["v"][0]))})
            else:
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                ET.SubElement(e_el, "property", {"name": name, "type": "float", "value": _fmt_num(fv)})


def _indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def validate_am_xml(path: str) -> dict:
    """Quick structural validation used by CLI + MCP."""
    root = ET.parse(path).getroot()
    issues = []
    if root.tag != "scene":
        issues.append("root element must be <scene>")
    for attr in ("width", "height", "totalTime", "fps"):
        if root.get(attr) is None:
            issues.append(f"missing scene attribute '{attr}'")
    ids = []
    for el in root:
        if el.tag in ("media", "bookmark"):
            continue
        if el.get("id") is None:
            issues.append(f"element <{el.tag}> missing id")
        else:
            ids.append(el.get("id"))
    if len(ids) != len(set(ids)):
        issues.append("duplicate element ids")
    return {"ok": not issues, "issues": issues, "ffver": root.get("ffver"),
            "elements": len(ids), "media": len(root.findall("media"))}
