"""Lottie (Bodymovin) import / export.

Supported on export: solid/shape/text layers, transforms with keyframes, opacity,
trim-path start/end for shapes, masks (as Lottie masks), and nested comps as
precomps. Effects are baked only where a matching Lottie effect exists; the
rest are reported as warnings.

Supported on import: solid, shape (rect/ellipse/path), text, image layers,
precomps and transform/property keyframes.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from .model import Project
from .util import parse_color, now_ms


def _hex_to_lottie(hex_color: str) -> List[float]:
    c = parse_color(hex_color)
    return [round(c[0], 4), round(c[1], 4), round(c[2], 4)]


def _lottie_color(arr) -> str:
    try:
        r, g, b = float(arr[0]), float(arr[1]), float(arr[2])
        a = float(arr[3]) if len(arr) > 3 else 1.0
    except Exception:
        r = g = b = 0.0
        a = 1.0
    return "#%02X%02X%02X%02X" % (int(round(a * 255)), int(round(r * 255)), int(round(g * 255)),
                                  int(round(b * 255)))


def _prop(static_value=None, keys: Optional[List[dict]] = None, dims: int = 1) -> dict:
    if keys:
        kfs = []
        for k in sorted(keys, key=lambda k: k.get("t", 0)):
            v = k.get("v") or [0.0] * dims
            if not isinstance(v, (list, tuple)):
                v = [v]
            kfs.append({"t": float(k.get("t", 0)), "s": [float(x) for x in v[:dims]],
                        "i": {"x": [0.4], "y": [1]}, "o": {"x": [0.6], "y": [0]}})
        return {"a": 1, "k": kfs}
    v = static_value if static_value is not None else [0.0] * dims
    if not isinstance(v, (list, tuple)):
        v = [v]
    return {"a": 0, "k": [float(x) for x in v[:dims]] if dims > 1 else float(v[0])}


def _scale_keys(layer: dict, prop: str, t_start: int) -> List[dict]:
    return (layer.get("keys") or {}).get(prop) or []


def export_lottie(project: Project) -> dict:
    w, h = project.width, project.height
    fps = float(project.fps)
    duration_frames = max(1, int(round(project.duration / 1000.0 * fps)))
    def _num(v, default=0.0):
        if isinstance(v, (list, tuple)):
            return float(v[0]) if v else float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _vec2(v, default=(0.0, 0.0)):
        if isinstance(v, (list, tuple)) and v:
            if len(v) >= 2:
                return [_num(v[0]), _num(v[1])]
            return [_num(v[0]), _num(v[0])]
        try:
            f = float(v)
            return [f, f]
        except (TypeError, ValueError):
            return [float(default[0]), float(default[1])]

    layers = []
    for idx, layer in enumerate(reversed(project.layers)):
        kind = layer.get("kind")
        st = int(layer.get("start", 0))
        et = int(layer.get("end", project.duration))
        ip = st / 1000.0 * fps
        op = max(ip + 1, et / 1000.0 * fps)
        transform = layer.get("transform") or {}
        pos_keys = _scale_keys(layer, "pos", st)
        scale_keys = _scale_keys(layer, "scale", st)
        rot_keys = _scale_keys(layer, "rot", st)
        op_keys = _scale_keys(layer, "opacity", st)
        ks = {
            "o": _prop(_num(transform.get("opacity", 1.0), 1.0) * 100,
                       [{"t": k["t"], "v": [float(k["v"][0]) * 100]} for k in op_keys], 1),
            "r": _prop(_num(transform.get("rot", 0), 0.0), rot_keys, 1),
            "p": _prop(_vec2(transform.get("pos"), (0.0, 0.0)) + [0.0],
                       [{"t": k["t"], "v": [float(k["v"][0]), float(k["v"][1]), 0.0]} for k in pos_keys], 3),
            "a": _prop(_vec2(transform.get("anchor"), (0.0, 0.0)) + [0.0], None, 3),
            "s": _prop([_num(transform.get("scale", [1, 1])[0] if isinstance(transform.get("scale"), (list, tuple)) else transform.get("scale"), 1.0) * 100,
                        _num(transform.get("scale", [1, 1])[1] if isinstance(transform.get("scale"), (list, tuple)) and len(transform.get("scale")) > 1 else 1.0, 1.0) * 100,
                        100.0],
                       [{"t": k["t"], "v": [float(k["v"][0]) * 100, float(k["v"][1]) * 100, 100.0]}
                        for k in scale_keys], 3),
        }
        base = {"ddd": 0, "ind": idx + 1, "nm": layer.get("name", kind or "layer"),
                "ks": ks, "ao": 0, "ip": ip, "op": op, "st": 0,
                "bm": 0, "sr": 1, "parent": None}
        if kind in ("solid", "adjustment"):
            base["ty"] = 1
            base["sw"] = float(layer.get("w") or w)
            base["sh"] = float(layer.get("h") or h)
            base["sc"] = _hex_to_lottie(layer.get("color", "#FF888888"))
        elif kind == "text":
            base["ty"] = 5
            base["t"] = {
                "d": {"k": [{
                    "s": {"s": float(layer.get("size", 64)), "f": str(layer.get("font", "DejaVuSans")),
                          "t": str(layer.get("text", "")), "j": 2, "tr": 0, "lh": float(layer.get("size", 64)) * 1.2,
                          "ls": 0, "fc": _hex_to_lottie(layer.get("color", "#FFFFFFFF")),
                          "sc": _hex_to_lottie(layer.get("stroke_color", "#FF000000")),
                          "sw": float(layer.get("stroke_width", 0) or 0)},
                    "t": 0}]}}
        elif kind in ("shape", "solid"):
            shape = (layer.get("shape") or "rect").lower()
            items = []
            if shape in ("ellipse", "circle", "oval"):
                items.append({"ty": "el", "p": {"a": 0, "k": [0, 0]},
                              "s": {"a": 0, "k": [float(layer.get("w", 200)), float(layer.get("h", 200))]}, "nm": "Ellipse"})
            elif shape in ("path", "freehand"):
                pts = layer.get("path") or []
                verts = [[float(p[0]) * float(layer.get("w", 200)) - float(layer.get("w", 200)) / 2,
                          float(p[1]) * float(layer.get("h", 200)) - float(layer.get("h", 200)) / 2] for p in pts]
                items.append({"ty": "sh", "ks": {"a": 0, "k": {"c": bool(layer.get("closed", True)),
                                                               "v": verts, "i": [[0, 0]] * len(verts),
                                                               "o": [[0, 0]] * len(verts)}},
                              "nm": "Path"})
            else:
                items.append({"ty": "rc", "p": {"a": 0, "k": [0, 0]},
                              "s": {"a": 0, "k": [float(layer.get("w", 200)), float(layer.get("h", 200))]},
                              "r": {"a": 0, "k": float(layer.get("radius", 0) or 0)}, "nm": "Rect"})
            trim = layer.get("trim")
            if trim:
                items.append({"ty": "tm", "s": {"a": 0, "k": float(trim.get("start", 0))},
                              "e": {"a": 0, "k": float(trim.get("end", 100))},
                              "o": {"a": 0, "k": float(trim.get("offset", 0))}, "m": 1, "nm": "Trim"})
            items.append({"ty": "fl", "c": {"a": 0, "k": _hex_to_lottie(layer.get("color", "#FF4C8DFF"))},
                          "o": {"a": 0, "k": 100}, "r": 1, "nm": "Fill"})
            sw = float(layer.get("stroke_width", 0) or 0)
            if sw > 0:
                items.append({"ty": "st", "c": {"a": 0, "k": _hex_to_lottie(layer.get("stroke_color", "#FF000000"))},
                              "o": {"a": 0, "k": 100}, "w": {"a": 0, "k": sw}, "nm": "Stroke"})
            base["ty"] = 4
            base["shapes"] = [{"ty": "gr", "it": items, "nm": "Group"}]
        elif kind == "precomp":
            base["ty"] = 0
            base["refId"] = f"comp_{layer.get('comp_id')}"
            base["w"] = float(layer.get("w") or w)
            base["h"] = float(layer.get("h") or h)
        else:  # media layers become image layers referencing the file
            base["ty"] = 2
            base["refId"] = f"asset_{layer.get('media_id')}"
            base["sw"] = float(w)
            base["sh"] = float(h)
        masks = layer.get("masks") or []
        if masks:
            base["hasMask"] = True
            base["masksProperties"] = []
            for m in masks:
                pts = m.get("points") or []
                verts = [[float(p[0]) * float(layer.get("w", w)), float(p[1]) * float(layer.get("h", h))]
                         for p in pts]
                base["masksProperties"].append({
                    "inv": bool(m.get("invert")), "mode": "a" if (m.get("mode") or "add") == "add" else "s",
                    "pt": {"a": 0, "k": {"c": True, "v": verts, "i": [[0, 0]] * len(verts),
                                         "o": [[0, 0]] * len(verts)}},
                    "o": {"a": 0, "k": float(m.get("opacity", 1.0)) * 100},
                    "x": {"a": 0, "k": float(m.get("expansion", 0) or 0)},
                    "nm": "Mask"})
        layers.append(base)

    assets = []
    for item in project.media:
        assets.append({"id": f"asset_{item.get('id')}", "w": int(item.get("width") or w),
                       "h": int(item.get("height") or h), "u": "", "p": os.path.basename(item.get("path", "")),
                       "e": 0})
    comps = []
    for comp in project.comps:
        comps.append({"id": f"comp_{comp.get('id')}", "nm": comp.get("name", "Comp")})
    return {
        "v": "5.7.4", "fr": fps, "ip": 0, "op": duration_frames,
        "w": w, "h": h, "nm": project.data.get("title") or "CraftMotion",
        "ddd": 0, "assets": assets + comps, "layers": layers,
        "markers": [{"tm": m.get("t", 0) / 1000.0 * fps, "cm": str(m.get("label", ""))}
                    for m in project.data.get("markers", [])],
    }


def export_lottie_file(project: Project, path: str) -> str:
    data = export_lottie(project)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return path


def _read_prop(prop, default, dims=1):
    """Return (static_value, keys)."""
    if not isinstance(prop, dict):
        return default, []
    if prop.get("a") == 1:
        keys = []
        for k in prop.get("k", []):
            if not isinstance(k, dict):
                continue
            v = k.get("s")
            if v is None:
                continue
            if not isinstance(v, (list, tuple)):
                v = [v]
            keys.append({"t": int(round(float(k.get("t", 0)))), "v": [float(x) for x in v[:dims]],
                         "e": "linear"})
        if keys:
            return keys[0]["v"], keys
    k = prop.get("k")
    if isinstance(k, (int, float)):
        return [float(k)], []
    if isinstance(k, list):
        if k and isinstance(k[0], list):
            return [float(x) for x in k[0][:dims]], []
        return [float(x) for x in k[:dims]], []
    return default, []


def import_lottie(data: dict) -> tuple:
    """Convert Lottie JSON into a Project. Returns (project, warnings)."""
    warnings: List[str] = []
    project = Project()
    fps = float(data.get("fr", 30))
    w = int(data.get("w", 1080))
    h = int(data.get("h", 1920))
    duration_frames = float(data.get("op", 90))
    project.data["settings"].update({"width": w, "height": h, "fps": fps,
                                     "duration": int(duration_frames / fps * 1000)})
    project.data["title"] = data.get("nm", "Lottie import")
    project.data["layers"] = []
    assets = {a.get("id"): a for a in data.get("assets", []) if isinstance(a, dict)}

    for lyr in data.get("layers", []):
        ty = lyr.get("ty")
        ks = lyr.get("ks") or {}
        pos, pos_keys = _read_prop(ks.get("p"), [w / 2, h / 2], 2)
        scale, scale_keys = _read_prop(ks.get("s"), [100, 100], 2)
        rot, rot_keys = _read_prop(ks.get("r"), [0], 1)
        opacity, op_keys = _read_prop(ks.get("o"), [100], 1)
        anchor, _ = _read_prop(ks.get("a"), [0, 0], 2)
        start = int(float(lyr.get("ip", 0)) / fps * 1000)
        end = int(float(lyr.get("op", duration_frames)) / fps * 1000)
        base = {
            "name": lyr.get("nm", "Layer"),
            "kind": "solid",
            "start": start, "end": max(start + 100, end),
            "transform": {"pos": [float(pos[0]), float(pos[1]), 0.0],
                          "anchor": [float(anchor[0]), float(anchor[1])],
                          "scale": [float(scale[0]) / 100.0, float(scale[1]) / 100.0],
                          "rot": float(rot[0]), "skew": [0, 0],
                          "opacity": float(opacity[0]) / 100.0},
        }
        keys = {}
        if pos_keys:
            keys["pos"] = pos_keys
        if scale_keys:
            keys["scale"] = [{"t": k["t"], "v": [v[0] / 100.0, v[1] / 100.0], "e": "linear"} for k in scale_keys]
        if rot_keys:
            keys["rot"] = rot_keys
        if op_keys:
            keys["opacity"] = [{"t": k["t"], "v": [v[0] / 100.0], "e": "linear"} for k in op_keys]
        if keys:
            base["keys"] = keys

        if ty == 1:  # solid
            base["kind"] = "solid"
            base["w"] = int(lyr.get("sw", w))
            base["h"] = int(lyr.get("sh", h))
            sc = lyr.get("sc")
            base["color"] = _lottie_color(sc) if sc else "#FF707070"
        elif ty == 5:  # text
            base["kind"] = "text"
            doc = (((lyr.get("t") or {}).get("d") or {}).get("k") or [{}])[0]
            s = (doc or {}).get("s") or {}
            base["text"] = s.get("t", "")
            base["size"] = float(s.get("s", 64))
            base["font"] = s.get("f", "")
            base["color"] = _lottie_color(s.get("fc", [1, 1, 1])) if s.get("fc") else "#FFFFFFFF"
            if s.get("sc"):
                base["stroke_color"] = _lottie_color(s.get("sc"))
            base["stroke_width"] = float(s.get("sw", 0) or 0)
        elif ty == 4:  # shape
            base["kind"] = "shape"
            items = []
            for group in lyr.get("shapes", []):
                items.extend(group.get("it", []) if group.get("ty") == "gr" else [group])
            size = [200.0, 200.0]
            color = "#FF4C8DFF"
            shape = "rect"
            trim = None
            for it in items:
                t = it.get("ty")
                if t == "el":
                    shape = "ellipse"
                    size = (it.get("s") or {}).get("k", size)
                elif t == "rc":
                    shape = "rect"
                    size = (it.get("s") or {}).get("k", size)
                    base["radius"] = float((it.get("r") or {}).get("k", 0) or 0)
                elif t == "sh":
                    k = (it.get("ks") or {}).get("k") or {}
                    verts = k.get("v") or []
                    if verts:
                        shape = "path"
                        xs = [float(v[0]) for v in verts]
                        ys = [float(v[1]) for v in verts]
                        ww = max(1.0, max(xs) - min(xs))
                        hh = max(1.0, max(ys) - min(ys))
                        base["path"] = [[(x - min(xs)) / ww, (y - min(ys)) / hh] for x, y in zip(xs, ys)]
                        base["closed"] = bool(k.get("c", True))
                        size = [ww, hh]
                elif t == "fl":
                    color = _lottie_color((it.get("c") or {}).get("k", [0, 0, 0]))
                elif t == "tm":
                    trim = {"start": float((it.get("s") or {}).get("k", 0)),
                            "end": float((it.get("e") or {}).get("k", 100)),
                            "offset": float((it.get("o") or {}).get("k", 0))}
            base["shape"] = shape
            base["w"] = float(size[0]) if not isinstance(size[0], str) else 200.0
            base["h"] = float(size[1]) if not isinstance(size[1], str) else 200.0
            base["color"] = color
            if trim:
                base["trim"] = trim
        elif ty in (0, 2):  # precomp / image asset
            ref = lyr.get("refId")
            asset = assets.get(ref) or {}
            if ty == 2 and asset:
                base["kind"] = "image"
                base["path"] = asset.get("p", "")
                base["w"] = int(asset.get("w", w))
                base["h"] = int(asset.get("h", h))
                warnings.append(f"image asset '{asset.get('p')}' needs relinking after import")
            else:
                base["kind"] = "precomp"
                warnings.append("precomp reference kept as a placeholder layer")
        else:
            warnings.append(f"unsupported Lottie layer type {ty} ('{lyr.get('nm')}') skipped")
            continue

        masks = lyr.get("masksProperties") or []
        if masks:
            base["masks"] = []
            for m in masks:
                k = ((m.get("pt") or {}).get("k") or {})
                verts = k.get("v") or []
                if not verts:
                    continue
                xs = [float(v[0]) for v in verts]
                ys = [float(v[1]) for v in verts]
                ww = max(1.0, max(xs) - min(xs))
                hh = max(1.0, max(ys) - min(ys))
                base["masks"].append({"shape": "polygon",
                                      "points": [[(x - min(xs)) / ww, (y - min(ys)) / hh] for x, y in zip(xs, ys)],
                                      "invert": bool(m.get("inv")), "mode": "add" if m.get("mode", "a") == "a" else "subtract",
                                      "feather": 0.0, "opacity": float(((m.get("o") or {}).get("k", 100) or 100)) / 100.0})
        project.data["layers"].append(base)

    for lyr in project.data["layers"]:
        Project._normalize_layer(lyr)
    project.data["meta"]["next_id"] = max([int(l.get("id", 0)) for l in project.data["layers"]] + [0]) + 1
    return project, warnings


def import_lottie_file(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return import_lottie(data)
