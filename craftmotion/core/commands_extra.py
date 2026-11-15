"""Registry commands for the advanced feature set (masks, styles, grade, audio,
captions, tracking, transitions, Lottie, three-point editing, comps, 3D).

Imported at the bottom of :mod:`registry` so every one of these is available to
the GUI, the CLI and MCP agents identically.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from . import audio as AUD
from . import grade as GRADE
from . import editing as EDIT
from . import tracking as TRACK
from . import captions as CAPS
from .captions import draw_captions
from .effects import EFFECTS, resolve_type
from .export import render_still
from .grade import render_scope_image
from .lottie import export_lottie_file, import_lottie_file
from .model import LAYER_KINDS, Project, layer_summary
from .registry import A, CommandError, Registry, cmd
from .render import Renderer
from .transitions import TRANSITIONS, list_transitions
from .util import clamp, kf_add, kf_eval, now_ms, parse_color


def _layer(reg: Registry, lid) -> dict:
    layer = reg.project.get_layer(int(lid))
    if layer is None:
        raise CommandError(f"no layer with id {lid}")
    return layer


def _media(reg: Registry, mid) -> dict:
    item = reg.project.get_media(int(mid))
    if item is None:
        raise CommandError(f"no media with id {mid}")
    return item


# ================================================================== masks
@cmd("mask.add", "Masks", "Add a mask to a layer (polygon/rect/ellipse) with feather, "
     "expansion, invert and add/subtract/intersect modes.", {
    "layer_id": A("int", "Layer id", required=True),
    "shape": A("str", "Mask shape", default="polygon",
               enum=["polygon", "rect", "ellipse"]),
    "points": A("list", "Normalised polygon points [[x,y], ...] (0..1 of the layer box)"),
    "feather": A("float", "Feather radius (0-100)", default=0.0),
    "expansion": A("float", "Expand (+) or contract (-) the mask", default=0.0),
    "opacity": A("float", "Mask opacity 0..1", default=1.0),
    "invert": A("bool", "Invert the mask", default=False),
    "mode": A("str", "Combine mode", default="add", enum=["add", "subtract", "intersect", "difference"]),
}, mutates=True)
def c_mask_add(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    points = a.get("points") or [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
    mask = {"shape": a.get("shape", "polygon"), "points": [[float(p[0]), float(p[1])] for p in points],
            "feather": float(a.get("feather", 0) or 0), "expansion": float(a.get("expansion", 0) or 0),
            "opacity": float(a.get("opacity", 1.0)), "invert": bool(a.get("invert")),
            "mode": a.get("mode", "add"), "enabled": True}
    layer.setdefault("masks", []).append(mask)
    return {"mask_index": len(layer["masks"]) - 1, "masks": len(layer["masks"])}


@cmd("mask.update", "Masks", "Update one mask on a layer.", {
    "layer_id": A("int", "Layer id", required=True),
    "index": A("int", "Mask index", required=True),
    "patch": A("dict", "Fields to change (shape, points, feather, expansion, invert, mode, opacity, enabled)",
               required=True),
}, mutates=True)
def c_mask_update(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    masks = layer.get("masks") or []
    if not masks or not (0 <= int(a["index"]) < len(masks)):
        raise CommandError("mask index out of range")
    masks[int(a["index"])].update(a["patch"] or {})
    return {"mask": masks[int(a["index"])]}


@cmd("mask.remove", "Masks", "Remove a mask from a layer.", {
    "layer_id": A("int", "Layer id", required=True), "index": A("int", "Mask index", required=True),
}, mutates=True)
def c_mask_remove(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    masks = layer.get("masks") or []
    if not masks or not (0 <= int(a["index"]) < len(masks)):
        raise CommandError("mask index out of range")
    masks.pop(int(a["index"]))
    return {"masks": len(masks)}


@cmd("mask.list", "Masks", "List masks on a layer.", {"layer_id": A("int", "Layer id", required=True)})
def c_mask_list(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    return {"masks": layer.get("masks") or []}


# ================================================================== layer styles
STYLE_TYPES = ["drop_shadow", "outer_glow", "inner_glow", "stroke", "color_overlay", "bevel", "satin"]


@cmd("style.add", "Layer styles", "Add a layer style (drop shadow, glow, stroke, colour overlay, bevel, satin).", {
    "layer_id": A("int", "Layer id", required=True),
    "type": A("str", "Style type", required=True, enum=STYLE_TYPES),
    "params": A("dict", "Style parameters (color, blur, distance, size, opacity, depth)"),
}, mutates=True)
def c_style_add(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    stype = a["type"]
    defaults = {
        "drop_shadow": {"color": "#AA000000", "blur": 12, "opacity": 0.65, "distance": 10},
        "outer_glow": {"color": "#CC66DDFF", "blur": 16, "opacity": 0.7},
        "inner_glow": {"color": "#CCFFFFFF", "blur": 10, "opacity": 0.55},
        "stroke": {"color": "#FFFFFFFF", "size": 3},
        "color_overlay": {"color": "#CCFF3366", "opacity": 0.85},
        "bevel": {"depth": 3, "highlight": "#FFFFFFFF"},
        "satin": {"color": "#88000000", "blur": 8, "opacity": 0.5},
    }
    params = dict(defaults.get(stype, {}))
    params.update(a.get("params") or {})
    layer.setdefault("styles", []).append({"type": stype, "params": params, "enabled": True})
    return {"styles": [s["type"] for s in layer["styles"]]}


@cmd("style.remove", "Layer styles", "Remove every style of a type (or all with index -1).", {
    "layer_id": A("int", "Layer id", required=True),
    "index": A("int", "Style index (-1 removes all)", default=-1),
}, mutates=True)
def c_style_remove(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    styles = layer.get("styles") or []
    idx = int(a.get("index", -1))
    if idx < 0:
        layer["styles"] = []
    else:
        if not (0 <= idx < len(styles)):
            raise CommandError("style index out of range")
        styles.pop(idx)
    return {"styles": [s["type"] for s in layer.get("styles", [])]}


@cmd("style.list", "Layer styles", "List layer styles.", {"layer_id": A("int", "Layer id", required=True)})
def c_style_list(reg: Registry, a: dict):
    return {"styles": _layer(reg, a["layer_id"]).get("styles") or []}


# ================================================================== text animators
@cmd("text.animator_add", "Text", "Add a per-character text animator with a range/word/line selector.", {
    "layer_id": A("int", "Text layer id", required=True),
    "selector": A("dict", "Selector {type: range|wipe|word|line, start, end, smooth}"),
    "params": A("dict", "Animator params {x, y, rotation, scale%, opacity%, blur, color}; "
                        "scale/opacity accept either percent (100) or a fraction (1.0)"),
}, mutates=True)
def c_text_anim_add(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    if layer.get("kind") != "text":
        raise CommandError("text animators need a text layer")
    params = dict(a.get("params") or {})
    # scale/opacity are stored as percentages; accept 0..1 fractions too so a
    # hand-written 0.4 means 40% instead of 0.4%.
    for key in ("scale", "opacity"):
        if key in params:
            try:
                val = float(params[key])
            except (TypeError, ValueError):
                continue
            if abs(val) <= 1.0:
                params[key] = val * 100.0
    animator = {"selector": {"type": "range", "start": 0, "end": 100, "smooth": True, **(a.get("selector") or {})},
                "params": {"x": 0, "y": 0, "rotation": 0, "scale": 100, "opacity": 100, "blur": 0,
                           **params},
                "enabled": True}
    layer.setdefault("text_animators", []).append(animator)
    return {"animators": len(layer["text_animators"])}


@cmd("text.animator_remove", "Text", "Remove a text animator.", {
    "layer_id": A("int", "Text layer id", required=True), "index": A("int", "Animator index", default=-1),
}, mutates=True)
def c_text_anim_remove(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    animators = layer.get("text_animators") or []
    idx = int(a.get("index", -1))
    if idx < 0:
        layer["text_animators"] = []
    elif 0 <= idx < len(animators):
        animators.pop(idx)
    else:
        raise CommandError("animator index out of range")
    return {"animators": len(layer.get("text_animators", []))}


@cmd("text.presets", "Text", "Built-in text animation presets (type-on, cascade, wave, pop, fade up).")
def c_text_presets(reg: Registry, a: dict):
    return {"presets": [
        {"name": "type_on", "selector": {"type": "range", "start": 0, "end": 100},
         "params": {"opacity": 0}, "hint": "animate selector end 0→100"},
        {"name": "cascade", "selector": {"type": "range", "start": 0, "end": 100},
         "params": {"y": -80, "opacity": 0}},
        {"name": "wave", "selector": {"type": "range", "start": 0, "end": 100},
         "params": {"y": -30, "rotation": 8}},
        {"name": "pop", "selector": {"type": "word", "start": 0, "end": 100},
         "params": {"scale": 140, "opacity": 0}},
        {"name": "fade_up", "selector": {"type": "line", "start": 0, "end": 100},
         "params": {"y": 40, "opacity": 0}},
    ]}


# ================================================================== shape operators
@cmd("shape.trim", "Shapes", "Set trim-path start/end/offset on a path shape (animatable).", {
    "layer_id": A("int", "Shape layer id", required=True),
    "start": A("float", "Start percent", default=0.0),
    "end": A("float", "End percent", default=100.0),
    "offset": A("float", "Offset percent", default=0.0),
}, mutates=True)
def c_shape_trim(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer["trim"] = {"start": float(a.get("start", 0)), "end": float(a.get("end", 100)),
                     "offset": float(a.get("offset", 0))}
    return {"trim": layer["trim"]}


@cmd("shape.repeater", "Shapes", "Repeat a shape with stepped offset, rotation, scale and fading opacity.", {
    "layer_id": A("int", "Shape layer id", required=True),
    "copies": A("int", "Number of copies", default=5),
    "x": A("float", "Per-copy X offset (px)", default=0.0),
    "y": A("float", "Per-copy Y offset (px)", default=0.0),
    "rotation": A("float", "Per-copy rotation (deg)", default=0.0),
    "scale": A("float", "Per-copy scale multiplier", default=1.0),
    "start_opacity": A("float", "First copy opacity", default=1.0),
    "end_opacity": A("float", "Last copy opacity", default=1.0),
}, mutates=True)
def c_shape_repeater(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer["repeater"] = {k: a[k] for k in ("copies", "x", "y", "rotation", "scale", "start_opacity", "end_opacity")
                         if a.get(k) is not None}
    return {"repeater": layer["repeater"]}


@cmd("shape.clear_ops", "Shapes", "Remove trim and repeater from a shape layer.", {
    "layer_id": A("int", "Shape layer id", required=True)}, mutates=True)
def c_shape_clear(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer["trim"] = None
    layer["repeater"] = None
    return {"ok": True}


# ================================================================== transitions
@cmd("transition.list", "Transitions", "Every available transition with its category and parameters.")
def c_transition_list(reg: Registry, a: dict):
    return {"count": len(TRANSITIONS), "transitions": list_transitions()}


@cmd("transition.set", "Transitions", "Set an in/out transition on a layer.", {
    "layer_id": A("int", "Layer id", required=True),
    "type": A("str", "Transition key (see transition.list)", required=True),
    "duration": A("int", "Transition length in ms", default=600),
    "when": A("str", "in or out", default="in", enum=["in", "out"]),
    "params": A("dict", "Transition parameters"),
}, mutates=True)
def c_transition_set(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    tkey = a["type"]
    if tkey not in TRANSITIONS:
        raise CommandError(f"unknown transition '{tkey}' (see transition.list)")
    payload = {"type": tkey, "duration": int(a.get("duration", 600)), "params": a.get("params") or {}}
    layer["transition_in" if a.get("when", "in") == "in" else "transition_out"] = payload
    return {"transition": payload}


@cmd("transition.clear", "Transitions", "Clear the in and out transitions of a layer.", {
    "layer_id": A("int", "Layer id", required=True)}, mutates=True)
def c_transition_clear(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer.pop("transition_in", None)
    layer.pop("transition_out", None)
    return {"ok": True}


# ================================================================== expressions
@cmd("expression.set", "Expressions", "Attach a JS-style expression to a layer property "
     "(pos, scale, rot, opacity, anchor, skew) or to an effect parameter.", {
    "layer_id": A("int", "Layer id", required=True),
    "property": A("str", "Property path, e.g. 'pos' or 'effect.1.slider'", required=True),
    "expression": A("str", "Expression source", required=True),
}, mutates=True)
def c_expression_set(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    prop = a["property"]
    if prop.startswith("effect."):
        parts = prop.split(".")
        if len(parts) < 3:
            raise CommandError("effect expressions look like effect.<index>.<param>")
        idx = int(parts[1])
        stack = layer.get("effects") or []
        if not (0 <= idx < len(stack)):
            raise CommandError("effect index out of range")
        stack[idx].setdefault("expressions", {})[parts[2]] = a["expression"]
    else:
        layer.setdefault("expressions", {})[prop] = a["expression"]
    return {"property": prop, "expression": a["expression"]}


@cmd("expression.clear", "Expressions", "Remove an expression (or all with property='*').", {
    "layer_id": A("int", "Layer id", required=True), "property": A("str", "Property or '*'", default="*"),
}, mutates=True)
def c_expression_clear(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    if a.get("property", "*") == "*":
        layer["expressions"] = {}
    else:
        layer.setdefault("expressions", {}).pop(a["property"], None)
    return {"expressions": sorted((layer.get("expressions") or {}).keys())}


@cmd("expression.list", "Expressions", "List expressions on a layer.", {"layer_id": A("int", "Layer id", required=True)})
def c_expression_list(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    return {"expressions": layer.get("expressions") or {},
            "effect_expressions": [{"index": i, "params": e.get("expressions") or {}}
                                   for i, e in enumerate(layer.get("effects") or []) if e.get("expressions")]}


@cmd("expression.eval", "Expressions", "Evaluate an expression at a time and return the value (dry run).", {
    "layer_id": A("int", "Layer id", required=True),
    "property": A("str", "Property name", required=True),
    "expression": A("str", "Expression source", required=True),
    "time": A("int", "Time in ms", default=0),
})
def c_expression_eval(reg: Registry, a: dict):
    from .expressions import ExpressionEngine
    layer = _layer(reg, a["layer_id"])
    engine = ExpressionEngine(reg.project, int(a.get("time", 0)), layer)
    value = engine.evaluate(a["expression"], [0.0], None)
    return {"value": value}


# ================================================================== comps / 3D
@cmd("comp.create", "Comps", "Create a nested composition (precomp) and return its id.", {
    "name": A("str", "Comp name", default="Comp"),
    "width": A("int", "Width"), "height": A("int", "Height"),
    "duration": A("int", "Duration in ms"), "fps": A("float", "Frame rate"),
}, mutates=True)
def c_comp_create(reg: Registry, a: dict):
    p = reg.project
    comp = {"id": int(p.data["meta"].get("next_comp_id", 1)),
            "name": a.get("name") or "Comp",
            "width": int(a.get("width") or p.width), "height": int(a.get("height") or p.height),
            "duration": int(a.get("duration") or p.duration), "fps": float(a.get("fps") or p.fps),
            "bg": "#00000000", "layers": []}
    p.data["meta"]["next_comp_id"] = comp["id"] + 1
    p.comps.append(comp)
    return {"comp": comp}


@cmd("comp.list", "Comps", "List nested comps and their layer counts.")
def c_comp_list(reg: Registry, a: dict):
    return {"comps": [{"id": c.get("id"), "name": c.get("name"), "layers": len(c.get("layers") or []),
                       "width": c.get("width"), "height": c.get("height"),
                       "duration": c.get("duration")} for c in reg.project.comps]}


@cmd("comp.add_layer", "Comps", "Add a layer inside a nested comp.", {
    "comp_id": A("int", "Comp id", required=True),
    "kind": A("str", "Layer kind", default="solid", enum=["solid", "text", "shape", "adjustment"]),
    "name": A("str", "Layer name"), "text": A("str", "Text for text layers"),
    "color": A("str", "Colour"), "w": A("float", "Width"), "h": A("float", "Height"),
}, mutates=True)
def c_comp_add_layer(reg: Registry, a: dict):
    comp = reg.project.get_comp(a["comp_id"])
    if comp is None:
        raise CommandError(f"no comp with id {a['comp_id']}")
    layer = {"kind": a.get("kind", "solid"), "name": a.get("name") or a.get("kind", "layer"),
             "start": 0, "end": comp.get("duration") or reg.project.duration,
             "color": a.get("color") or "#FF4C8DFF", "w": a.get("w"), "h": a.get("h"),
             "text": a.get("text") or "", "effects": [], "keys": {},
             "transform": {"pos": [comp["width"] / 2.0, comp["height"] / 2.0, 0.0],
                           "scale": [1, 1], "rot": 0, "opacity": 1}}
    Project._normalize_layer(layer)
    comp.setdefault("layers", []).append(layer)
    return {"comp_layers": len(comp["layers"])}


@cmd("comp.delete", "Comps", "Delete a nested comp.", {
    "comp_id": A("int", "Comp id", required=True)}, mutates=True)
def c_comp_delete(reg: Registry, a: dict):
    p = reg.project
    before = len(p.comps)
    p.comps[:] = [c for c in p.comps if int(c.get("id", -1)) != int(a["comp_id"])]
    return {"removed": before - len(p.comps)}


@cmd("camera.set", "3D", "Configure the 2.5D camera (position, zoom, rotation, depth of field).", {
    "enabled": A("bool", "Enable the camera", default=True),
    "x": A("float", "Camera X (project px)"), "y": A("float", "Camera Y (project px)"),
    "z": A("float", "Camera Z", default=0.0),
    "zoom": A("float", "Zoom factor", default=1.0),
    "rotation": A("float", "Roll in degrees", default=0.0),
    "dof": A("dict", "Depth of field {enabled, focus_z, strength}"),
}, mutates=True)
def c_camera_set(reg: Registry, a: dict):
    cam = reg.project.data.setdefault("camera", {})
    for key in ("enabled", "x", "y", "z", "zoom", "rotation"):
        if a.get(key) is not None:
            cam[key] = a[key]
    if a.get("dof"):
        cam["dof"] = {**(cam.get("dof") or {}), **(a["dof"] or {})}
    return {"camera": cam}


@cmd("light.add", "3D", "Add a light layer that illuminates lit layers.", {
    "color": A("str", "Light colour", default="#FFFFD9A0"),
    "radius": A("float", "Falloff radius in px", default=900),
    "intensity": A("float", "Intensity 0..2", default=0.6),
    "x": A("float", "X position"), "y": A("float", "Y position"),
}, mutates=True)
def c_light_add(reg: Registry, a: dict):
    p = reg.project
    layer = p.add_layer({"kind": "light", "name": "Light", "color": a.get("color", "#FFFFD9A0"),
                         "radius": float(a.get("radius", 900)), "intensity": float(a.get("intensity", 0.6)),
                         "start": 0, "end": p.duration,
                         "transform": {"pos": [float(a.get("x") if a.get("x") is not None else p.width / 2),
                                               float(a.get("y") if a.get("y") is not None else p.height / 2),
                                               0.0]}})
    return {"layer": layer_summary(layer)}


@cmd("project.set_depth", "3D", "Set a layer's Z depth for camera parallax.", {
    "layer_id": A("int", "Layer id", required=True), "z": A("float", "Depth (px, + = closer)", default=0.0),
}, mutates=True)
def c_set_depth(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer["z"] = float(a.get("z", 0))
    return {"z": layer["z"]}


@cmd("project.set_trail", "Time", "Give a layer a motion trail / echo.", {
    "layer_id": A("int", "Layer id", required=True),
    "count": A("int", "Ghost frames", default=5),
    "spacing": A("int", "Spacing in ms", default=60),
    "decay": A("float", "Opacity decay per ghost", default=0.7),
    "enabled": A("bool", "Enable", default=True),
}, mutates=True)
def c_set_trail(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer["trail"] = {"count": int(a.get("count", 5)), "spacing": int(a.get("spacing", 60)),
                      "decay": float(a.get("decay", 0.7)), "enabled": bool(a.get("enabled", True))}
    return {"trail": layer["trail"]}


@cmd("project.motion_blur", "Time", "Configure render motion blur (shutter angle sampling).", {
    "enabled": A("bool", "Enable motion blur", default=True),
    "samples": A("int", "Sub-frame samples", default=6),
    "shutter": A("float", "Shutter as a fraction of a frame", default=0.5),
}, mutates=True)
def c_motion_blur(reg: Registry, a: dict):
    mb = reg.project.data.setdefault("motion_blur", {})
    mb.update({"enabled": bool(a.get("enabled", True)), "samples": int(a.get("samples", 6)),
               "shutter": float(a.get("shutter", 0.5))})
    return {"motion_blur": mb}


# ================================================================== grade + scopes
@cmd("grade.get", "Colour", "Read the project grade (curves, wheels, HSL, LUT, HDR/log).")
def c_grade_get(reg: Registry, a: dict):
    return {"grade": GRADE.ensure_grade(reg.project)}


@cmd("grade.set", "Colour", "Update the project grade. Any subset of fields may be supplied.", {
    "enabled": A("bool", "Enable the grade", default=True),
    "exposure": A("float", "Exposure in stops"),
    "contrast": A("float", "Contrast multiplier"),
    "saturation": A("float", "Saturation multiplier"),
    "temperature": A("float", "Temperature -1..1"),
    "tint": A("float", "Tint -1..1"),
    "lift": A("float", "Black lift"), "gamma": A("float", "Gamma"), "gain": A("float", "Gain"),
    "wheels": A("dict", "Colour wheels {shadows:{hue,amount}, midtones:{...}, highlights:{...}}"),
    "hsl": A("dict", "{hue, sat, lum}"),
    "curves": A("dict", "{rgb:[[x,y],...], r:[], g:[], b:[]}"),
    "lut": A("dict", "{path, amount}"),
    "transform": A("str", "HDR/log transform",
                   enum=["none", "slog3_to_709", "logc_to_709", "to_linear", "aces"]),
    "linear_light": A("bool", "Composite in linear light"),
}, mutates=True)
def c_grade_set(reg: Registry, a: dict):
    grade = GRADE.ensure_grade(reg.project)
    for key in ("enabled", "exposure", "contrast", "saturation", "temperature", "tint",
                "lift", "gamma", "gain", "transform", "linear_light"):
        if a.get(key) is not None:
            grade[key] = a[key]
    for key in ("wheels", "hsl", "curves", "lut"):
        if a.get(key):
            cur = grade.get(key) or {}
            if isinstance(cur, dict):
                for k, v in a[key].items():
                    if isinstance(v, dict) and isinstance(cur.get(k), dict):
                        cur[k].update(v)
                    else:
                        cur[k] = v
            else:
                grade[key] = a[key]
    return {"grade": GRADE.grade_summary(grade)}


@cmd("grade.lut", "Colour", "Apply a .cube LUT as part of the grade.", {
    "path": A("str", "Path to the .cube file", required=True),
    "amount": A("float", "Mix amount 0..1", default=1.0),
}, mutates=True)
def c_grade_lut(reg: Registry, a: dict):
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"LUT not found: {path}")
    grade = GRADE.ensure_grade(reg.project)
    grade["lut"] = {"path": os.path.abspath(path), "amount": float(a.get("amount", 1.0))}
    grade["enabled"] = True
    return {"lut": grade["lut"]}


@cmd("scope.render", "Colour", "Render a scope (histogram, waveform or vectorscope) PNG for a frame.", {
    "kind": A("str", "Scope type", default="histogram", enum=["histogram", "waveform", "vectorscope"]),
    "time": A("int", "Frame time in ms", default=0),
    "width": A("int", "Output width", default=480), "height": A("int", "Output height", default=270),
    "path": A("str", "Output PNG path"),
})
def c_scope_render(reg: Registry, a: dict):
    img = Renderer(reg.project, reg._media_manager()).render(int(a.get("time", 0)))
    scope = render_scope_image(img, a.get("kind", "histogram"),
                               (int(a.get("width", 480)), int(a.get("height", 270))))
    out = a.get("path") or os.path.join(reg.export_dir(), f"scope_{a.get('kind','histogram')}_{now_ms()}.png")
    scope.save(out)
    return {"path": out, "kind": a.get("kind", "histogram")}


# ================================================================== audio
@cmd("audio.mixer", "Audio", "Set mixer state for an audio-carrying layer (volume, pan, mute, solo, fades).", {
    "layer_id": A("int", "Layer id", required=True),
    "volume": A("float", "Linear volume"),
    "pan": A("float", "Pan -1..1"),
    "mute": A("bool", "Mute"),
    "solo": A("bool", "Solo"),
    "fade_in": A("int", "Fade in ms"), "fade_out": A("int", "Fade out ms"),
}, mutates=True)
def c_audio_mixer(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    for key in ("volume", "pan", "mute", "solo", "fade_in", "fade_out"):
        if a.get(key) is not None:
            layer[key] = a[key]
    return {"mixer": {k: layer.get(k) for k in ("volume", "pan", "mute", "solo", "fade_in", "fade_out")}}


@cmd("audio.automation", "Audio", "Write a volume automation keyframe on an audio layer.", {
    "layer_id": A("int", "Layer id", required=True),
    "time": A("int", "Time in ms", required=True),
    "volume": A("float", "Gain at that time", required=True),
    "easing": A("str", "Keyframe easing", default="linear"),
}, mutates=True)
def c_audio_automation(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    keys = layer.setdefault("keys", {}).setdefault("volume", [])
    keys[:] = [k for k in keys if int(k.get("t", -1)) != int(a["time"])]
    keys.append({"t": int(a["time"]), "v": [float(a["volume"])], "e": a.get("easing", "linear")})
    keys.sort(key=lambda k: k["t"])
    return {"volume_keys": keys}


@cmd("audio.fx_add", "Audio", "Add an audio effect (eq, compressor, limiter, gain, normalize, gate, reverb, pan).", {
    "layer_id": A("int", "Layer id", required=True),
    "type": A("str", "Effect type", required=True,
              enum=["eq", "compressor", "limiter", "gain", "normalize", "gate", "reverb", "pan"]),
    "params": A("dict", "Effect parameters"),
}, mutates=True)
def c_audio_fx(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    layer.setdefault("audio_fx", []).append({"type": a["type"], "params": a.get("params") or {}, "enabled": True})
    return {"audio_fx": [f["type"] for f in layer["audio_fx"]]}


@cmd("audio.master", "Audio", "Set master bus gain, ceiling and effect chain.", {
    "gain_db": A("float", "Master gain in dB"), "ceiling_db": A("float", "True-peak ceiling in dB"),
    "chain": A("list", "Master effect chain"),
}, mutates=True)
def c_audio_master(reg: Registry, a: dict):
    master = reg.project.data.setdefault("audio_master", {"gain_db": 0.0, "ceiling_db": -0.3, "chain": []})
    for key in ("gain_db", "ceiling_db", "chain"):
        if a.get(key) is not None:
            master[key] = a[key]
    return {"master": master}


@cmd("audio.lufs", "Audio", "Measure integrated loudness (EBU R128), true peak and dynamics.")
def c_audio_lufs(reg: Registry, a: dict):
    return AUD.analyze(reg.project, reg._media_manager())


@cmd("audio.normalize", "Audio", "Normalise the master bus to a target LUFS level.", {
    "target": A("float", "Target integrated LUFS", default=-16.0),
}, mutates=True)
def c_audio_normalize(reg: Registry, a: dict):
    res = AUD.normalize_to(reg.project, reg._media_manager(), float(a.get("target", -16.0)))
    if not res.get("ok"):
        raise CommandError(res.get("reason", "normalise failed"))
    return res


@cmd("audio.export_wav", "Audio", "Render the mix to a 48 kHz stereo WAV.", {
    "path": A("str", "Output path"),
    "start": A("int", "Start ms", default=0), "end": A("int", "End ms"),
})
def c_audio_export(reg: Registry, a: dict):
    samples = AUD.mix(reg.project, reg._media_manager(), int(a.get("start", 0)),
                      int(a["end"]) if a.get("end") else None)
    if samples is None:
        raise CommandError("no audible layers to export")
    out = a.get("path") or os.path.join(reg.export_dir(), f"mix_{now_ms()}.wav")
    AUD.write_wav(samples, out)
    return {"path": out, "lufs": round(AUD.lufs(samples), 2)}


# ================================================================== captions
@cmd("caption.add", "Captions", "Add a caption cue.", {
    "start": A("int", "Start ms", required=True), "end": A("int", "End ms", required=True),
    "text": A("str", "Cue text", required=True),
}, mutates=True)
def c_caption_add(reg: Registry, a: dict):
    cues = reg.project.data.setdefault("captions", [])
    cues.append({"start": int(a["start"]), "end": int(a["end"]), "text": a["text"]})
    cues.sort(key=lambda c: c["start"])
    return {"captions": len(cues)}


@cmd("caption.list", "Captions", "List caption cues.")
def c_caption_list(reg: Registry, a: dict):
    return {"captions": reg.project.data.get("captions") or [],
            "style": reg.project.data.get("caption_style") or {}}


@cmd("caption.style", "Captions", "Set caption look (size, colours, position, alignment).", {
    "style": A("dict", "Style fields", required=True)}, mutates=True)
def c_caption_style(reg: Registry, a: dict):
    style = reg.project.data.setdefault("caption_style", {})
    style.update(a["style"] or {})
    return {"style": style}


@cmd("caption.import", "Captions", "Import captions from .srt or .vtt.", {
    "path": A("str", "Subtitle file", required=True),
    "replace": A("bool", "Replace existing cues", default=True),
}, mutates=True)
def c_caption_import(reg: Registry, a: dict):
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"file not found: {path}")
    cues = CAPS.import_vtt(path) if path.lower().endswith(".vtt") else CAPS.import_srt(path)
    if a.get("replace", True):
        reg.project.data["captions"] = cues
    else:
        reg.project.data.setdefault("captions", []).extend(cues)
    return {"captions": len(reg.project.data.get("captions") or [])}


@cmd("caption.export", "Captions", "Export captions to .srt or .vtt.", {
    "path": A("str", "Output path", required=True)})
def c_caption_export(reg: Registry, a: dict):
    cues = reg.project.data.get("captions") or []
    if not cues:
        raise CommandError("no captions to export")
    out = a["path"]
    if a["path"].lower().endswith(".vtt"):
        CAPS.export_vtt(cues, out)
    else:
        CAPS.export_srt(cues, out)
    return {"path": out, "cues": len(cues)}


@cmd("caption.auto", "Captions", "Generate evenly timed cues from a script.", {
    "text": A("str", "Script, one caption per line", required=True),
    "start": A("int", "Start ms", default=0),
}, mutates=True)
def c_caption_auto(reg: Registry, a: dict):
    cues = CAPS.auto_cues_from_script(a["text"], int(a.get("start", 0)))
    reg.project.data["captions"] = cues
    return {"captions": len(cues)}


# ================================================================== tracking
@cmd("track.point", "Tracking", "Track a feature point through a media clip (NCC tracker).", {
    "media_id": A("int", "Media id", required=True),
    "x": A("float", "Start X in source px", required=True),
    "y": A("float", "Start Y in source px", required=True),
    "start": A("int", "Start ms", default=0), "end": A("int", "End ms"),
    "step": A("int", "Sample step in ms", default=100),
    "patch": A("int", "Patch half-size in px", default=12),
    "search": A("int", "Search radius in px", default=24),
})
def c_track_point(reg: Registry, a: dict):
    item = _media(reg, a["media_id"])
    end = int(a["end"]) if a.get("end") else int(item.get("duration_ms") or reg.project.duration)
    points = TRACK.track_point(reg._media_manager(), item, float(a["x"]), float(a["y"]),
                               int(a.get("start", 0)), end, int(a.get("step", 100)),
                               int(a.get("patch", 12)), int(a.get("search", 24)))
    return {"points": points, "count": len(points)}


@cmd("track.apply", "Tracking", "Bake a track onto a layer as position keys, or stabilise it.", {
    "layer_id": A("int", "Layer id", required=True),
    "points": A("list", "Track points from track.point", required=True),
    "mode": A("str", "attach or stabilize", default="attach", enum=["attach", "stabilize"]),
    "strength": A("float", "Apply strength", default=1.0),
    "absolute": A("bool", "Place the layer on the tracked point (False = relative motion only)",
                  default=True),
    "append": A("bool", "Stack onto existing position keys instead of replacing them",
                default=False),
}, mutates=True)
def c_track_apply(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    TRACK.apply_track(layer, a["points"], a.get("mode", "attach"), float(a.get("strength", 1.0)),
                      bool(a.get("absolute", True)), bool(a.get("append", False)))
    return {"keys": len((layer.get("keys") or {}).get("pos") or []), "mode": a.get("mode", "attach"),
            "absolute": bool(a.get("absolute", True))}


# ================================================================== editing
@cmd("edit.three_point", "Editing", "Three-point edit: place a source clip at the playhead using in/out points.", {
    "layer_id": A("int", "Source layer id", required=True),
    "at": A("int", "Timeline position ms", required=True),
    "source_in": A("int", "Source in ms"),
    "source_out": A("int", "Source out ms"),
    "track": A("int", "Target track"),
    "mode": A("str", "overwrite or insert", default="overwrite", enum=["overwrite", "insert"]),
}, mutates=True)
def c_edit_three_point(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    res = EDIT.three_point_insert(reg.project, layer, int(a["at"]),
                                  a.get("source_in"), a.get("source_out"), a.get("track"),
                                  a.get("mode", "overwrite"), ripple=(a.get("mode") == "insert"))
    return {"start": res["start"], "end": res["end"], "track": res["track"], "mode": res["mode"]}


@cmd("edit.trim", "Editing", "Trim a clip: ripple, roll, slip, slide, set_in, set_out or move.", {
    "layer_id": A("int", "Layer id", required=True),
    "mode": A("str", "Trim mode", default="ripple_out",
              enum=["ripple_in", "ripple_out", "roll_in", "roll_out", "slip", "slide", "set_in", "set_out", "move"]),
    "amount": A("int", "Signed amount in ms", required=True),
}, mutates=True)
def c_edit_trim(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    return EDIT.trim_layer(reg.project, layer, a.get("mode", "ripple_out"), int(a["amount"]))


@cmd("edit.jkl", "Editing", "J/K/L transport maths: returns the new playback rate.", {
    "key": A("str", "j, k or l", required=True, enum=["j", "k", "l"]),
    "current": A("float", "Current rate", default=1.0),
})
def c_edit_jkl(reg: Registry, a: dict):
    return {"rate": EDIT.jkl_rate(a["key"], float(a.get("current", 1.0)))}


@cmd("edit.frame_step", "Editing", "Step exactly one frame forwards or backwards.", {
    "time": A("float", "Current time ms", default=0.0),
    "direction": A("int", "1 forward, -1 back", default=1),
})
def c_edit_frame_step(reg: Registry, a: dict):
    return {"time": EDIT.frame_step(reg.project, float(a.get("time", 0.0)), int(a.get("direction", 1)))}


@cmd("edit.razor", "Editing", "Split every clip that crosses the playhead on a track.", {
    "time": A("int", "Time ms", required=True), "track": A("int", "Track (omit for all tracks)"),
}, mutates=True)
def c_edit_razor(reg: Registry, a: dict):
    t = int(a["time"])
    track = a.get("track")
    new_layers = []
    for layer in list(reg.project.layers):
        if track is not None and int(layer.get("track", 0)) != int(track):
            continue
        s, e = int(layer.get("start", 0)), int(layer.get("end", reg.project.duration))
        if s < t < e:
            tail = json.loads(json.dumps(layer))
            tail["id"] = int(reg.project.data["meta"].get("next_id", 1))
            reg.project.data["meta"]["next_id"] = tail["id"] + 1
            tail["start"] = t
            tail["source_in"] = int(layer.get("source_in", 0) or 0) + (t - s)
            layer["end"] = t
            new_layers.append(tail)
    reg.project.layers.extend(new_layers)
    return {"splits": len(new_layers)}


# ================================================================== lottie
@cmd("project.import_lottie", "Lottie", "Import a Lottie / Bodymovin JSON animation as layers.", {
    "path": A("str", "Lottie .json path", required=True),
    "replace": A("bool", "Replace the current project", default=True),
}, mutates=True)
def c_import_lottie(reg: Registry, a: dict):
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"file not found: {path}")
    project, warnings = import_lottie_file(path)
    if a.get("replace", True):
        reg.project.data = project.data
        reg.project.path = path
    else:
        for layer in project.layers:
            layer["id"] = int(reg.project.data["meta"].get("next_id", 1))
            reg.project.data["meta"]["next_id"] = layer["id"] + 1
            reg.project.layers.append(layer)
    return {"layers": len(project.layers), "warnings": warnings}


@cmd("project.export_lottie", "Lottie", "Export the project as Lottie / Bodymovin JSON.", {
    "path": A("str", "Output .json path"),
})
def c_export_lottie(reg: Registry, a: dict):
    out = a.get("path") or os.path.join(reg.export_dir(), f"{reg.project.slug()}.json")
    export_lottie_file(reg.project, out)
    return {"path": out, "size": os.path.getsize(out)}


@cmd("edge.detect_all", "App", "Report the full feature surface with counts (for agent discovery).")
def c_edge_detect_all(reg: Registry, a: dict):
    from .effects import EFFECTS as FX_EFFECTS
    from .transitions import TRANSITIONS as TR
    from .expressions import EXPRESSION_HELP
    cats: Dict[str, int] = {}
    for eff in FX_EFFECTS.values():
        cats[eff.category] = cats.get(eff.category, 0) + 1
    return {"effects": len(FX_EFFECTS), "effect_categories": cats, "transitions": len(TR),
            "commands": len(__import__("craftmotion.core.registry", fromlist=["COMMANDS"]).COMMANDS),
            "blend_modes": len(__import__("craftmotion.core.model", fromlist=["BLEND_MODES"]).BLEND_MODES),
            "expressions": EXPRESSION_HELP, "layer_kinds": LAYER_KINDS}


# ================================================================== audio-reactive
def _audio_env(reg: Registry, hop_ms: float = 20.0) -> dict:
    media = reg._media_manager()
    return AUD.envelope(reg.project, media, hop_ms)


@cmd("audio.envelope", "Audio", "Analyse the mixed soundtrack into a level/band envelope "
     "(level, low, mid, high) sampled every hop_ms.", {
    "hop_ms": A("float", "Hop size in milliseconds", default=20.0),
    "limit": A("int", "Maximum samples returned (0 = all)", default=400),
})
def c_audio_envelope(reg: Registry, a: dict):
    env = _audio_env(reg, float(a.get("hop_ms") or 20.0))
    limit = int(a.get("limit") or 0)
    out = dict(env)
    if limit > 0 and len(env["level"]) > limit:
        step = max(1, len(env["level"]) // limit)
        for key in ("level", "low", "mid", "high", "times", "peaks"):
            seq = env.get(key) or []
            out[key] = seq[::step][:limit]
    return out


@cmd("audio.spectrum", "Audio", "Log-spaced spectrum per hop for analyser visualisers.", {
    "hop_ms": A("float", "Hop size in milliseconds", default=40.0),
    "bands": A("int", "Number of bands", default=24),
})
def c_audio_spectrum(reg: Registry, a: dict):
    spec = AUD.spectrogram(reg.project, reg._media_manager(), float(a.get("hop_ms") or 40.0),
                           int(a.get("bands") or 24))
    return {"hop_ms": spec["hop_ms"], "bands": spec["bands"],
            "frames": spec["frames"][:400], "level": spec["level"][:400]}


@cmd("audio.beats", "Audio", "Detect onsets and estimate tempo; optionally drop timeline markers.", {
    "threshold": A("float", "Onset sensitivity", default=0.35),
    "min_gap_ms": A("float", "Minimum gap between beats", default=180.0),
    "add_markers": A("bool", "Write the detected beats into the timeline as markers", default=False),
})
def c_audio_beats(reg: Registry, a: dict):
    env = _audio_env(reg, 20.0)
    beats = AUD.detect_beats(env, float(a.get("min_gap_ms") or 180.0), float(a.get("threshold") or 0.35))
    tempo = AUD.estimate_tempo(beats)
    if a.get("add_markers"):
        markers = reg.project.data.setdefault("markers", [])
        have = {int(m.get("t", -1)) for m in markers}
        for t in beats:
            if int(t) not in have:
                markers.append({"t": int(t), "label": "beat", "color": "#FFFFB440"})
        markers.sort(key=lambda m: m.get("t", 0))
    return {"beats": beats, "count": len(beats), "tempo": tempo}


def _react_curve(env: dict, source: str, lo: float, hi: float, smoothing: float,
                 threshold: float) -> List[float]:
    seq = env.get(source) or env.get("level") or []
    if not seq:
        return []
    arr = np.asarray(seq, np.float32)
    if threshold > 0:
        arr = np.clip((arr - threshold) / max(1e-3, 1 - threshold), 0, 1)
    if smoothing > 0:
        k = max(1, int(round(smoothing * 12)))
        kernel = np.ones(k, np.float32) / k
        arr = np.convolve(arr, kernel, mode="same")
    return [round(float(lo + (hi - lo) * v), 4) for v in arr]


@cmd("audio.react", "Audio", "Bake an audio envelope onto a layer property (scale, opacity, rot, "
     "pos_x, pos_y or an effect parameter) so motion follows the music.", {
    "layer_id": A("int", "Layer id", required=True),
    "source": A("str", "Envelope source", default="level",
                enum=["level", "low", "mid", "high"]),
    "property": A("str", "Target property", default="scale",
                  enum=["scale", "opacity", "rot", "pos_x", "pos_y", "effect"]),
    "lo": A("float", "Value at silence", default=100.0),
    "hi": A("float", "Value at full level", default=115.0),
    "smoothing": A("float", "Extra smoothing (0-1)", default=0.2),
    "threshold": A("float", "Noise gate (0-1)", default=0.15),
    "effect_index": A("int", "Effect index when property=effect", default=0),
    "param": A("str", "Effect parameter name when property=effect"),
    "every_ms": A("int", "Keyframe spacing in ms (0 = every envelope hop)", default=80),
})
def c_audio_react(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    env = _audio_env(reg, 20.0)
    values = _react_curve(env, a.get("source") or "level", float(a["lo"]), float(a["hi"]),
                          float(a.get("smoothing") or 0.0), float(a.get("threshold") or 0.0))
    if not values:
        raise CommandError("no audio in the project — import a clip with sound first")
    start = int(layer.get("start", 0))
    end = int(layer.get("end", reg.project.duration))
    hop = float(env.get("hop_ms") or 20.0)
    every = int(a.get("every_ms") or 0) or int(hop)
    prop = a.get("property") or "scale"
    times = env.get("times") or [i * hop for i in range(len(values))]
    keys: List[dict] = []
    last_t = -10 ** 9
    for t_ms, val in zip(times, values):
        if t_ms < start or t_ms > end:
            continue
        if t_ms - last_t < every - 1:
            continue
        last_t = t_ms
        keys.append({"t": int(round(t_ms)), "v": [float(val)], "e": "linear"})
    if not keys:
        raise CommandError("the envelope has no samples inside the layer's time range")
    if prop == "effect":
        idx = int(a.get("effect_index") or 0)
        stack = layer.get("effects") or []
        if not (0 <= idx < len(stack)):
            raise CommandError(f"effect index {idx} out of range ({len(stack)} effects)")
        name = a.get("param")
        if not name:
            raise CommandError("param is required when property=effect")
        stack[idx].setdefault("keys", {})[name] = keys
        return {"layer_id": layer["id"], "effect_index": idx, "param": name, "keys": len(keys)}
    if prop in ("pos_x", "pos_y"):
        base = layer.setdefault("transform", {}).setdefault("pos", [reg.project.width / 2.0,
                                                                    reg.project.height / 2.0, 0.0])
        axis = 0 if prop == "pos_x" else 1
        other = [float(base[1 - axis]) if axis == 0 else float(base[0])]
        kf = []
        for k in keys:
            vec = [0.0, 0.0]
            vec[axis] = k["v"][0]
            vec[1 - axis] = other[0]
            kf.append({"t": k["t"], "v": vec, "e": "linear"})
        layer.setdefault("keys", {})["pos"] = kf
        return {"layer_id": layer["id"], "property": "pos", "keys": len(kf)}
    if prop == "scale":
        layer.setdefault("keys", {})["scale"] = [{"t": k["t"], "v": [k["v"][0], k["v"][0]], "e": "linear"}
                                                 for k in keys]
    else:
        layer.setdefault("keys", {})[prop if prop != "rot" else "rot"] = keys
    return {"layer_id": layer["id"], "property": prop, "keys": len(keys),
            "min": min(k["v"][0] for k in keys), "max": max(k["v"][0] for k in keys)}


@cmd("audio.analyser", "Audio", "Add an audio-reactive effect (spectrum, waveform, pulse or zoom) "
     "to a layer so it reacts to the soundtrack live.", {
    "layer_id": A("int", "Layer id", required=True),
    "kind": A("str", "Analyser type", default="spectrum",
              enum=["spectrum", "waveform", "pulse", "zoom"]),
    "params": A("dict", "Optional parameter overrides"),
})
def c_audio_analyser(reg: Registry, a: dict):
    from .effects import defaults_for
    layer = _layer(reg, a["layer_id"])
    key = {"spectrum": "audio_spectrum", "waveform": "audio_waveform",
           "pulse": "audio_pulse", "zoom": "audio_scale"}[a.get("kind") or "spectrum"]
    params = defaults_for(key)
    params.update(a.get("params") or {})
    if layer.get("kind") == "text" and not str(layer.get("text", "")).strip():
        # the analyser draws into the (otherwise empty) sprite, so park it dead centre
        layer.setdefault("transform", {})["pos"] = [reg.project.width / 2.0,
                                                    reg.project.height / 2.0, 0.0]
    stack = layer.setdefault("effects", [])
    stack.append({"type": key, "params": params, "enabled": True})
    return {"layer_id": layer["id"], "effect": key, "effects": len(stack)}


# ================================================================== text presets
_TEXT_PRESETS = {
    "type_on": {
        "selector": {"type": "range", "start": 0, "end": 100,
                     "keys": {"start": [{"t": 0, "v": [0], "e": "linear"},
                                        {"t": 1200, "v": [100], "e": "linear"}]}},
        "params": {"opacity": 0, "y": 6}, "hint": "characters reveal left to right"},
    "cascade": {
        "selector": {"type": "range", "start": 0, "end": 100,
                     "keys": {"start": [{"t": 0, "v": [0], "e": "linear"},
                                        {"t": 700, "v": [100], "e": "ease_out"}]}},
        "params": {"y": -70, "opacity": 0, "rotation": -6}, "hint": "letters drop in one by one"},
    "wave": {
        "selector": {"type": "range", "start": 0, "end": 100},
        "params": {"y": -26, "rotation": 7, "scale": 104}, "hint": "static ribbon offset"},
    "pop": {
        "selector": {"type": "word", "start": 0, "end": 100,
                     "keys": {"start": [{"t": 0, "v": [0], "e": "linear"},
                                        {"t": 600, "v": [100], "e": "ease_out"}]}},
        "params": {"scale": 150, "opacity": 0}, "hint": "words pop in"},
    "blur_in": {
        "selector": {"type": "range", "start": 0, "end": 100,
                     "keys": {"start": [{"t": 0, "v": [0], "e": "linear"},
                                        {"t": 900, "v": [100], "e": "ease_in_out"}]}},
        "params": {"blur": 9, "opacity": 40}, "hint": "defocus to focus"},
    "tracking_in": {
        "selector": {"type": "line", "start": 0, "end": 100},
        "params": {"x": 14}, "hint": "line-by-line slide"},
}


@cmd("text.presets", "Text", "List the built-in text animation presets (install with text.preset_apply).")
def c_text_presets_v2(reg: Registry, a: dict):
    return {"presets": [{"name": k, **v} for k, v in _TEXT_PRESETS.items()]}


@cmd("text.preset_apply", "Text", "Apply a keyframed text animation preset (type_on, cascade, wave, "
     "pop, blur_in, tracking_in).", {
    "layer_id": A("int", "Text layer id", required=True),
    "name": A("str", "Preset name", default="cascade"),
    "speed": A("float", "Time multiplier for the preset (2 = twice as fast)", default=1.0),
}, mutates=True)
def c_text_preset_apply(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    if layer.get("kind") != "text":
        raise CommandError("text presets need a text layer")
    name = a.get("name") or "cascade"
    preset = _TEXT_PRESETS.get(name)
    if preset is None:
        raise CommandError(f"unknown preset '{name}' (choose from {', '.join(_TEXT_PRESETS)})")
    speed = max(0.1, float(a.get("speed") or 1.0))
    selector = json.loads(json.dumps(preset["selector"]))
    for keys in (selector.get("keys") or {}).values():
        for k in keys:
            k["t"] = int(round(float(k["t"]) / speed))
    animator = {"selector": selector, "params": dict(preset["params"]), "enabled": True}
    layer.setdefault("text_animators", []).append(animator)
    return {"preset": name, "animators": len(layer["text_animators"]),
            "hint": preset.get("hint", "")}


@cmd("text.animator_key", "Text", "Keyframe a text animator selector (start/end/offset) for "
     "typewriter and wipe reveals.", {
    "layer_id": A("int", "Text layer id", required=True),
    "index": A("int", "Animator index", default=0),
    "param": A("str", "Selector field", default="end", enum=["start", "end", "offset"]),
    "time": A("int", "Keyframe time in ms", required=True),
    "value": A("float", "Selector percentage", default=100.0),
    "easing": A("str", "Keyframe easing", default="linear"),
}, mutates=True)
def c_text_animator_key(reg: Registry, a: dict):
    layer = _layer(reg, a["layer_id"])
    animators = layer.get("text_animators") or []
    idx = int(a.get("index") or 0)
    if not (0 <= idx < len(animators)):
        raise CommandError(f"animator index {idx} out of range ({len(animators)})")
    sel = animators[idx].setdefault("selector", {})
    keys = sel.setdefault("keys", {}).setdefault(a.get("param") or "end", [])
    kf_add(keys, int(a["time"]), [float(a.get("value", 100.0))], a.get("easing"))
    return {"index": idx, "param": a.get("param"), "keys": len(keys)}


# ================================================================== AM preset looks
@cmd("preset.am_look", "Presets", "Extract the signature effect chain from an Alight Motion XML preset "
     "and apply it to a layer, so a community preset becomes a reusable look.", {
    "path": A("str", "Path to the .xml preset", required=True),
    "layer_id": A("int", "Target layer id", required=True),
    "mode": A("str", "signature = most common chain, heaviest = longest chain",
              default="signature", enum=["signature", "heaviest"]),
    "max_effects": A("int", "Cap on how many effects to copy", default=8),
    "scale": A("float", "Multiply numeric parameters (look strength)", default=1.0),
    "replace": A("bool", "Replace the layer's existing effect stack", default=True),
    "exclude": A("list", "Effect types to leave out (e.g. [\"invert\", \"blink\"])", default=None),
}, mutates=True)
def c_preset_am_look(reg: Registry, a: dict):
    from .am_xml import import_am_scene
    path = a["path"]
    if not os.path.exists(path):
        raise CommandError(f"file not found: {path}")
    layer = _layer(reg, a["layer_id"])
    proj, res = import_am_scene(path)
    skip = {str(x) for x in (a.get("exclude") or [])}
    chains: Dict[tuple, int] = {}
    by_chain: Dict[tuple, dict] = {}
    for src in proj.layers:
        effs = [e for e in (src.get("effects") or [])
                if e.get("type") and e["type"] not in skip]
        if len(effs) < 2:
            continue
        key = tuple(e["type"] for e in effs)
        chains[key] = chains.get(key, 0) + 1
        by_chain.setdefault(key, src)
    if not chains:
        raise CommandError("no multi-effect chains found in that preset")
    if (a.get("mode") or "signature") == "heaviest":
        pick = max(chains, key=lambda k: (len(k), chains[k]))
    else:
        pick = max(chains, key=lambda k: (chains[k], len(k)))
    src_layer = by_chain[pick]
    cap = max(1, int(a.get("max_effects") or 8))
    scale = float(a.get("scale") or 1.0)
    copied = []
    for eff in src_layer.get("effects") or []:
        if not eff.get("type") or len(copied) >= cap or eff["type"] in skip:
            continue
        params = {}
        spec = EFFECTS.get(eff["type"])
        for name, val in (eff.get("params") or {}).items():
            if spec and name in spec.params and isinstance(val, (int, float)) and not isinstance(val, bool):
                prm = spec.params[name]
                if prm.kind == "float" and scale != 1.0 and prm.hi and prm.lo is not None:
                    lo, hi = float(prm.lo), float(prm.hi)
                    val = max(lo, min(hi, lo + (float(val) - lo) * scale))
            params[name] = val
        copied.append({"type": eff["type"], "src_id": eff.get("src_id"),
                       "params": params, "keys": eff.get("keys") or {}, "enabled": True})
    if a.get("replace", True):
        layer["effects"] = copied
    else:
        layer.setdefault("effects", []).extend(copied)
    return {"layer_id": layer["id"], "preset": os.path.basename(path),
            "chain": [e["type"] for e in copied], "uses_in_preset": chains[pick],
            "source_layer": src_layer.get("name") or src_layer.get("kind"),
            "ignored_warnings": len(res.warnings), "effects": len(layer["effects"])}
