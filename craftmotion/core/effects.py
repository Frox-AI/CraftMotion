"""Effect registry: 65+ GPU-style effects implemented with numpy + PIL.

Every effect is `func(img_rgba, params, ctx)` -> RGBA PIL image where ctx is a
dict with: project, layer, t_local (ms inside the layer), extent (progress
0..1 across the layer), renderer. Parameter values are already resolved
(keyframes evaluated) by the renderer before calling.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .util import parse_color

# ------------------------------------------------------------------ helpers


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def _pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGBA")


def _rgb(img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    a = _np(img)
    return a[..., :3], a[..., 3:4]


def _merge(rgb: np.ndarray, a: np.ndarray) -> Image.Image:
    return _pil(np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]


def _noise(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((shape[0], shape[1], 1), dtype=np.float32)


def _vignette_mask(h: int, w: int, feather: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
    f = max(0.05, feather)
    return np.clip((r - (1.0 - f)) / f, 0, 1)[..., None]


def _remap(img: Image.Image, dx: np.ndarray, dy: np.ndarray) -> Image.Image:
    """Bilinear remap: out(x,y) = in(x+dx, y+dy)."""
    a = _np(img)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = np.clip(xx + dx, 0, w - 1)
    sy = np.clip(yy + dy, 0, h - 1)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    top = a[y0, x0] * (1 - fx) + a[y0, x1] * fx
    bot = a[y1, x0] * (1 - fx) + a[y1, x1] * fx
    return _pil(top * (1 - fy) + bot * fy)


def _blur(img: Image.Image, px: float) -> Image.Image:
    if px <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=float(px)))


def _clamped(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


# ------------------------------------------------------------------ registry

@dataclass
class Param:
    name: str
    default: Any
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    kind: str = "float"       # float | int | bool | color | choice | vec2
    options: List[str] = field(default_factory=list)
    label: str = ""


@dataclass
class Effect:
    key: str
    category: str
    label: str
    params: Dict[str, Param]
    func: Callable
    am_ids: List[str] = field(default_factory=list)
    description: str = ""


EFFECTS: Dict[str, Effect] = {}


def register(key: str, category: str, label: str, params: Dict[str, Param], am_ids=None, description=""):
    def deco(fn):
        EFFECTS[key] = Effect(key, category, label, params, fn, am_ids or [], description)
        return fn
    return deco


def F(name, default, lo=0.0, hi=1.0, step=0.01, label=""):
    return Param(name, float(default), lo, hi, step, "float", [], label)


def I(name, default, lo=0, hi=100, step=1, label=""):
    return Param(name, int(default), lo, hi, step, "int", [], label)


def B(name, default=False, label=""):
    return Param(name, bool(default), 0, 1, 1, "bool", [], label)


def C(name, default="#FFFFFFFF", label=""):
    return Param(name, default, 0, 1, 1, "color", [], label)


def CH(name, default, options, label=""):
    return Param(name, default, 0, 1, 1, "choice", list(options), label)


AM = "com.alightcreative.effects."

# ------------------------------------------------------------------ color


@register("brightness_contrast", "Color", "Brightness / Contrast",
          {"brightness": F("brightness", 0.0, -1, 1), "contrast": F("contrast", 0.0, -1, 1)},
          [AM + "brightcont", AM + "brightcont2"])
def fx_brightcont(img, p, ctx):
    rgb, a = _rgb(img)
    rgb = (rgb - 0.5) * (1.0 + p["contrast"]) + 0.5 + p["brightness"]
    return _merge(rgb, a)


@register("exposure", "Color", "Exposure",
          {"exposure": F("exposure", 0.0, -3, 3), "gamma": F("gamma", 1.0, 0.2, 3)},
          [AM + "exposure"])
def fx_exposure(img, p, ctx):
    rgb, a = _rgb(img)
    rgb = np.clip(rgb, 0, None) * (2.0 ** p["exposure"])
    rgb = np.sign(rgb) * np.abs(rgb) ** (1.0 / max(0.05, p["gamma"]))
    return _merge(rgb, a)


@register("saturation", "Color", "Saturation & Vibrance",
          {"saturation": F("saturation", 0.0, -1, 2), "vibrance": F("vibrance", 0.0, -1, 2)},
          [AM + "satvib", AM + "saturation"])
def fx_saturation(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    rgb = rgb + (rgb - l) * p["saturation"]
    # vibrance: boost low-saturation pixels more
    sat = (rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True))
    w = (1.0 - np.clip(sat * 1.6, 0, 1)) * p["vibrance"]
    l2 = _luma(rgb)
    rgb = rgb + (rgb - l2) * w
    return _merge(rgb, a)


@register("hue", "Color", "Hue Shift",
          {"shift": F("shift", 0.0, -180, 180, 1)}, [AM + "hue", AM + "hueshift"])
def fx_hue(img, p, ctx):
    hsv = np.asarray(img.convert("RGBA").convert("RGB").convert("HSV"), np.float32) / 255.0
    hsv[..., 0] = (hsv[..., 0] + p["shift"] / 360.0) % 1.0
    rgb = np.asarray(Image.fromarray((hsv * 255).astype(np.uint8), "HSV").convert("RGB"), np.float32) / 255.0
    a = _np(img)[..., 3:4]
    return _merge(rgb, a)


@register("temperature", "Color", "Color Temperature",
          {"temp": F("temp", 0.0, -1, 1), "tint": F("tint", 0.0, -1, 1)},
          [AM + "colortemperature", AM + "colortint"])
def fx_temperature(img, p, ctx):
    rgb, a = _rgb(img)
    rgb = rgb + np.array([p["temp"] * 0.18, p["tint"] * 0.08, -p["temp"] * 0.18], dtype=np.float32)
    return _merge(rgb, a)


@register("color_tint", "Color", "Tint (Two Color)",
          {"amount": F("amount", 1.0, 0, 1), "shadow": C("shadow", "#FF0033AA"), "highlight": C("highlight", "#FFFFAA33")},
          [AM + "colortune", AM + "colortune2"])
def fx_colortune(img, p, ctx):

    rgb, a = _rgb(img)
    l = _luma(rgb)
    s = np.array(parse_color(p["shadow"])[:3], np.float32)
    h = np.array(parse_color(p["highlight"])[:3], np.float32)
    tinted = s + (h - s) * l
    rgb = rgb * (1 - p["amount"]) + tinted * p["amount"] * (0.35 + l * 0.9)
    return _merge(rgb, a)


@register("color_balance", "Color", "Color Balance",
          {"shadows": F("shadows", 0.0, -1, 1), "midtones": F("midtones", 0.0, -1, 1), "highlights": F("highlights", 0.0, -1, 1)},
          [AM + "colorbalance"])
def fx_color_balance(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    warm = np.array([1.0, 0.55, -0.35], np.float32)
    cool = np.array([-0.35, 0.1, 1.0], np.float32)
    sw = np.clip(1.0 - l * 2.0, 0, 1) * p["shadows"]
    mw = (1.0 - np.abs(l - 0.5) * 2.0) * p["midtones"]
    hw = np.clip(l * 2.0 - 1.0, 0, 1) * p["highlights"]
    # cinematic split-tone: cool shadows, warm highlights
    rgb = rgb + cool * sw * 0.5 + warm * hw * 0.5 + (warm + cool) * 0.12 * mw
    return _merge(rgb, a)


@register("curves", "Color", "Curves (RGB)",
          {"lift": F("lift", 0.0, -0.5, 0.5), "gamma": F("gamma", 1.0, 0.2, 3), "gain": F("gain", 1.0, 0.0, 2)},
          [AM + "curves"])
def fx_curves(img, p, ctx):
    rgb, a = _rgb(img)
    rgb = np.clip(rgb + p["lift"], 0, 1) ** (1.0 / max(0.05, p["gamma"])) * p["gain"]
    return _merge(rgb, a)


@register("levels", "Color", "Levels",
          {"black": F("black", 0.0, 0, 0.5), "white": F("white", 1.0, 0.5, 1.0), "gamma": F("gamma", 1.0, 0.2, 3)},
          [AM + "levels"])
def fx_levels(img, p, ctx):
    rgb, a = _rgb(img)
    lo, hi = p["black"], max(p["black"] + 0.01, p["white"])
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** (1.0 / max(0.05, p["gamma"]))
    return _merge(rgb, a)


@register("posterize", "Color", "Posterize", {"levels": I("levels", 6, 2, 32)}, [AM + "posterize"])
def fx_posterize(img, p, ctx):
    rgb, a = _rgb(img)
    n = max(2, p["levels"])
    rgb = np.floor(rgb * n) / (n - 1)
    return _merge(rgb, a)


@register("threshold", "Color", "Threshold", {"level": F("level", 0.5, 0, 1)}, [AM + "threshold"])
def fx_threshold(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    v = (l > p["level"]).astype(np.float32)
    return _merge(np.repeat(v, 3, axis=-1), a)


@register("greyscale", "Color", "Black & White", {"mix": F("mix", 1.0, 0, 1)}, [AM + "greyscale", AM + "desaturate"])
def fx_greyscale(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    return _merge(rgb * (1 - p["mix"]) + np.repeat(l, 3, axis=-1) * p["mix"], a)


@register("duotone", "Color", "Duotone",
          {"amount": F("amount", 1.0, 0, 1), "dark": C("dark", "#FF101040"), "light": C("light", "#FFFFCC66")},
          [AM + "duotone"])
def fx_duotone(img, p, ctx):

    rgb, a = _rgb(img)
    l = _luma(rgb)
    d = np.array(parse_color(p["dark"])[:3], np.float32)
    li = np.array(parse_color(p["light"])[:3], np.float32)
    duo = d + (li - d) * l
    return _merge(rgb * (1 - p["amount"]) + duo * p["amount"], a)


@register("invert", "Color", "Invert", {"amount": F("amount", 1.0, 0, 1)}, [AM + "invert"])
def fx_invert(img, p, ctx):
    rgb, a = _rgb(img)
    return _merge(rgb * (1 - p["amount"]) + (1 - rgb) * p["amount"], a)


@register("sepia", "Color", "Sepia", {"amount": F("amount", 1.0, 0, 1)}, [AM + "sepia"])
def fx_sepia(img, p, ctx):
    rgb, a = _rgb(img)
    m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
    sep = rgb @ m.T
    return _merge(rgb * (1 - p["amount"]) + sep * p["amount"], a)


@register("channel_mixer", "Color", "Channel Mixer",
          {"red": F("red", 1.0, 0, 2), "green": F("green", 1.0, 0, 2), "blue": F("blue", 1.0, 0, 2)},
          [AM + "channelmap", AM + "channelmaphsv"])
def fx_channel_mixer(img, p, ctx):
    rgb, a = _rgb(img)
    rgb = rgb * np.array([p["red"], p["green"], p["blue"]], np.float32)
    return _merge(rgb, a)


@register("spot_color", "Color", "Spot Color",
          {"hue": F("hue", 0.0, 0, 360, 1), "range": F("range", 40.0, 1, 180), "amount": F("amount", 1.0, 0, 1)},
          [AM + "spotcolor", AM + "spotcolor2"])
def fx_spot_color(img, p, ctx):
    base = img.convert("RGB").convert("HSV")
    arr = np.asarray(base, np.float32)
    h = arr[..., 0] * (255.0 / 360.0)
    d = np.abs(h - (p["hue"] * 255.0 / 360.0))
    d = np.minimum(d, 255 - d) * (360.0 / 255.0)
    mask = np.clip(1.0 - d / max(1.0, p["range"]), 0, 1)[..., None]
    rgb, a = _rgb(img)
    grey = np.repeat(_luma(rgb), 3, axis=-1)
    out = grey + (rgb - grey) * mask
    return _merge(rgb * (1 - p["amount"]) + out * p["amount"], a)


@register("lut", "Color", "LUT (.cube)",
          {"file": Param("file", "", 0, 1, 1, "choice", []), "amount": F("amount", 1.0, 0, 1)},
          [AM + "lut"])
def fx_lut(img, p, ctx):
    path = p.get("file") or ""
    lut = _load_cube(path)
    if lut is None:
        return img
    rgb, a = _rgb(img)
    out = _apply_cube(rgb, lut)
    return _merge(rgb * (1 - p["amount"]) + out * p["amount"], a)


_CUBES: Dict[str, Any] = {}


def _load_cube(path: str):
    if not path or not os.path.exists(path):
        return None
    key = f"{path}:{os.path.getmtime(path)}"
    if key in _CUBES:
        return _CUBES[key]
    try:
        size = 0
        table = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                elif line.upper().startswith(("TITLE", "DOMAIN_", "LUT_1D")):
                    continue
                else:
                    parts = line.split()
                    if len(parts) == 3:
                        try:
                            table.append([float(x) for x in parts])
                        except ValueError:
                            pass
        if size and len(table) >= size ** 3:
            cube = np.array(table[: size ** 3], np.float32).reshape(size, size, size, 3)
            _CUBES[key] = cube
            return cube
    except Exception:
        pass
    _CUBES[key] = None
    return None


def _apply_cube(rgb: np.ndarray, cube: np.ndarray) -> np.ndarray:
    n = cube.shape[0]
    coords = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.floor(coords).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, n - 1)
    f = coords - i0
    c000 = cube[i0[..., 0], i0[..., 1], i0[..., 2]]
    c100 = cube[i1[..., 0], i0[..., 1], i0[..., 2]]
    c010 = cube[i0[..., 0], i1[..., 1], i0[..., 2]]
    c110 = cube[i1[..., 0], i1[..., 1], i0[..., 2]]
    c001 = cube[i0[..., 0], i0[..., 1], i1[..., 2]]
    c101 = cube[i1[..., 0], i0[..., 1], i1[..., 2]]
    c011 = cube[i0[..., 0], i1[..., 1], i1[..., 2]]
    c111 = cube[i1[..., 0], i1[..., 1], i1[..., 2]]
    fx_, fy, fz = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c00 = c000 * (1 - fx_) + c100 * fx_
    c10 = c010 * (1 - fx_) + c110 * fx_
    c01 = c001 * (1 - fx_) + c101 * fx_
    c11 = c011 * (1 - fx_) + c111 * fx_
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


# ------------------------------------------------------------------ blur


@register("blur", "Blur", "Gaussian Blur", {"radius": F("radius", 8.0, 0, 200, 0.5), "quality": I("quality", 1, 1, 3)},
          [AM + "gaussianblur", AM + "boxblur", AM + "boxblur2", AM + "boxblur3", AM + "dblur",
           AM + "fastboxblur", AM + "lensblur", AM + "lensblur2", AM + "maskblur", AM + "innerblur"])
def fx_blur(img, p, ctx):
    r = p["radius"]
    if r <= 0.25:
        return img
    return _blur(img, r)


@register("motion_blur", "Blur", "Motion Blur",
          {"distance": F("distance", 20.0, 0, 200), "angle": F("angle", 0.0, -180, 180, 1)},
          [AM + "motionblur", AM + "motionblur2", AM + "motionblur3", AM + "motionblur4"])
def fx_motion_blur(img, p, ctx):
    dist = p["distance"]
    if dist <= 0.5:
        return img
    rad = math.radians(p["angle"])
    dx, dy = math.cos(rad), math.sin(rad)
    acc = np.zeros_like(_np(img))
    steps = 9
    base = _np(img)
    h, w = base.shape[:2]
    for i in range(steps):
        off = (i / (steps - 1) - 0.5) * dist
        acc += np.roll(np.roll(base, int(off * dy), axis=0), int(off * dx), axis=1)
    return _pil(acc / steps)


@register("zoom_blur", "Blur", "Zoom Blur",
          {"strength": F("strength", 0.15, 0, 1), "samples": I("samples", 8, 2, 16)},
          [AM + "zoomblur", AM + "zoomblur2", AM + "zoomblur3"])
def fx_zoom_blur(img, p, ctx):
    n = p["samples"]
    base = _np(img)
    h, w = base.shape[:2]
    acc = np.zeros_like(base)
    for i in range(n):
        f = 1.0 + p["strength"] * (i / (n - 1) - 0.5)
        if abs(f - 1) < 1e-4:
            layer = Image.fromarray((base * 255).astype(np.uint8), "RGBA")
        else:
            nw, nh = max(2, int(w * f)), max(2, int(h * f))
            im = Image.fromarray((base * 255).astype(np.uint8), "RGBA").resize((nw, nh), Image.BILINEAR)
            ox, oy = (nw - w) // 2, (nh - h) // 2
            im = im.crop((ox, oy, ox + w, oy + h))
            layer = np.asarray(im, np.float32) / 255.0
        acc += layer
    return _pil(acc / n)


@register("spin_blur", "Blur", "Spin Blur",
          {"angle": F("angle", 20.0, 0, 360, 1), "samples": I("samples", 8, 2, 24)},
          [AM + "spinblur", AM + "spinblur2", AM + "vortexblur", AM + "warpblur",
           AM + "chromaticvortexblur"])
def fx_spin_blur(img, p, ctx):
    n = max(2, p["samples"])
    acc = np.zeros_like(_np(img))
    for i in range(n):
        ang = p["angle"] * (i / (n - 1) - 0.5)
        acc += _np(img.rotate(ang, resample=Image.BILINEAR))
    return _pil(acc / n)


# ------------------------------------------------------------------ stylize


@register("sharpen", "Stylize", "Sharpen", {"amount": F("amount", 0.5, 0, 3), "radius": F("radius", 2.0, 0.2, 10)},
          [AM + "sharpen", AM + "unsharp"])
def fx_sharpen(img, p, ctx):
    soft = img.filter(ImageFilter.GaussianBlur(p["radius"]))
    a = _np(img)
    s = _np(soft)
    out = a + (a - s) * p["amount"]
    return _pil(out)


@register("glow", "Stylize", "Glow",
          {"radius": F("radius", 12.0, 0, 100), "strength": F("strength", 1.0, 0, 3), "threshold": F("threshold", 0.55, 0, 1),
           "color": C("color", "#FFFFFFFF"), "use_color": B("use_color", False)},
          [AM + "glow", AM + "softglow", AM + "lightglow", AM + "darkglow", AM + "edgeglow",
           AM + "glow-inner", AM + "inner-glow"])
def fx_glow(img, p, ctx):

    rgb, a = _rgb(img)
    l = _luma(rgb)
    mask = np.clip((l - p["threshold"]) / max(0.05, 1 - p["threshold"]), 0, 1)
    bright = Image.fromarray(np.clip(rgb * mask * 255, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(max(0.5, p["radius"])))
    b = np.asarray(bright, np.float32) / 255.0
    if p["use_color"]:
        b = b * np.array(parse_color(p["color"])[:3], np.float32)
    out = 1 - (1 - rgb) * (1 - b * p["strength"])
    return _merge(out, a)


@register("bloom", "Stylize", "Bloom",
          {"radius": F("radius", 24.0, 0, 120), "strength": F("strength", 0.8, 0, 2), "threshold": F("threshold", 0.7, 0, 1)},
          [AM + "bloom", AM + "glowscan"])
def fx_bloom(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    mask = np.clip((l - p["threshold"]) / max(0.05, 1 - p["threshold"]), 0, 1) ** 1.5
    bright = Image.fromarray(np.clip(rgb * mask * 255, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(max(0.5, p["radius"])))
    b = np.asarray(bright, np.float32) / 255.0 * p["strength"]
    return _merge(rgb + b, a)


@register("vignette", "Stylize", "Vignette",
          {"amount": F("amount", 0.6, 0, 1), "feather": F("feather", 0.5, 0.05, 1), "roundness": F("roundness", 1.0, 0, 1)},
          [AM + "vignette"])
def fx_vignette(img, p, ctx):
    rgb, a = _rgb(img)
    h, w = rgb.shape[:2]
    m = _vignette_mask(h, w, p["feather"])
    out = rgb * (1 - m * p["amount"])
    return _merge(out, a)


@register("grain", "Stylize", "Film Grain",
          {"amount": F("amount", 0.12, 0, 1), "size": F("size", 1.0, 0.5, 4), "mono": B("mono", True)},
          [AM + "noise", AM + "noise2", AM + "noise3", AM + "filmgrain", AM + "grain"])
def fx_grain(img, p, ctx):
    rgb, a = _rgb(img)
    h, w = rgb.shape[:2]
    n = _noise((h, w), seed=int(ctx.get("t_local", 0) // 33))
    if p["size"] > 1.2:
        f = int(round(p["size"]))
        n = np.repeat(np.repeat(n[::f, ::f], f, axis=0), f, axis=1)[:h, :w]
    if not p["mono"]:
        n = np.concatenate([n, _noise((h, w), seed=99) if False else n, n], axis=-1)
    rgb = rgb + (n - 0.5) * p["amount"] * 0.6
    return _merge(rgb, a)


@register("scanlines", "Stylize", "Scanlines",
          {"spacing": I("spacing", 4, 2, 20), "opacity": F("opacity", 0.35, 0, 1)},
          [AM + "scanlines"])
def fx_scanlines(img, p, ctx):
    rgb, a = _rgb(img)
    h = rgb.shape[0]
    mask = np.ones((h, 1, 1), np.float32)
    mask[:: max(2, p["spacing"])] = 1.0 - p["opacity"]
    return _merge(rgb * mask, a)


@register("pixelate", "Stylize", "Pixelate", {"size": I("size", 12, 1, 200)}, [AM + "pixelate", AM + "pixelate2", AM + "mosaic"])
def fx_pixelate(img, p, ctx):
    s = max(1, p["size"])
    w, h = img.size
    small = img.resize((max(1, w // s), max(1, h // s)), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


@register("chromatic", "Stylize", "Chromatic Aberration",
          {"shift": F("shift", 4.0, 0, 40), "angle": F("angle", 0.0, -180, 180, 1)},
          [AM + "chromaticaberration", AM + "chromatic", AM + "chromaticzoomblur",
           AM + "chromaticvortexblur", AM + "streaks-zoom"])
def fx_chromatic(img, p, ctx):
    rgb, a = _rgb(img)
    dx = int(round(math.cos(math.radians(p["angle"])) * p["shift"]))
    dy = int(round(math.sin(math.radians(p["angle"])) * p["shift"]))
    r = np.roll(np.roll(rgb[..., 0], dy, axis=0), dx, axis=1)
    b = np.roll(np.roll(rgb[..., 2], -dy, axis=0), -dx, axis=1)
    return _merge(np.stack([r, rgb[..., 1], b], axis=-1), a)


@register("glitch", "Stylize", "Glitch",
          {"amount": F("amount", 0.5, 0, 1), "block": I("block", 12, 2, 80), "seed": I("seed", 1, 0, 9999)},
          [AM + "glitch", AM + "digitalglitch", AM + "blocknoise", AM + "blocknoise2", AM + "datamosh"])
def fx_glitch(img, p, ctx):
    arr = _np(img)
    h, w = arr.shape[:2]
    rng = np.random.default_rng(p["seed"] * 7919 + int(ctx.get("t_local", 0) // 66))
    out = arr.copy()
    block = max(2, p["block"])
    n_blocks = int(2 + p["amount"] * 8)
    q = float(ctx.get("scale", 1.0) or 1.0)
    for _ in range(n_blocks):
        y = rng.integers(0, max(1, h - block))
        bh = int(rng.integers(block // 2 + 1, max(block, block * 2)))
        shift = int(rng.normal(0, min(p["amount"], 1.5) * 22) * max(q, 0.2))
        out[y:y + bh] = np.roll(out[y:y + bh], shift, axis=1)
    # occasional channel split rows
    return _pil(out)


@register("edge_detect", "Stylize", "Edge Detect", {"amount": F("amount", 1.0, 0, 2)},
          [AM + "findedges", AM + "findedges-edgeglow", AM + "edges"])
def fx_edges(img, p, ctx):
    e = img.convert("RGB").filter(ImageFilter.FIND_EDGES)
    ea = np.asarray(e, np.float32) / 255.0 * p["amount"]
    a = _np(img)[..., 3:4]
    return _merge(1 - ea, a)


@register("emboss", "Stylize", "Emboss", {"amount": F("amount", 1.0, 0, 2)}, [AM + "emboss"])
def fx_emboss(img, p, ctx):
    e = img.convert("RGB").filter(ImageFilter.EMBOSS)
    ea = np.asarray(e, np.float32) / 255.0
    a = _np(img)[..., 3:4]
    return _merge(np.clip(ea * p["amount"], 0, 1), a)


@register("cartoon", "Stylize", "Cartoon",
          {"smooth": F("smooth", 3.0, 0, 10), "levels": I("levels", 6, 2, 24)},
          [AM + "cartoon", AM + "toon"])
def fx_cartoon(img, p, ctx):
    smooth = _blur(img, p["smooth"])
    rgb, a = _rgb(smooth)
    n = max(2, p["levels"])
    rgb = np.floor(rgb * n) / (n - 1)
    edges = np.asarray(smooth.convert("RGB").filter(ImageFilter.FIND_EDGES), np.float32) / 255.0
    rgb = rgb * (1 - np.clip(edges * 2.2, 0, 1))
    return _merge(rgb, a)


@register("vhs", "Stylize", "VHS / Retro",
          {"wobble": F("wobble", 1.0, 0, 5), "noise": F("noise", 0.15, 0, 1), "color_bleed": F("color_bleed", 6.0, 0, 30)},
          [AM + "vhs", AM + "oldfilm", AM + "retro", AM + "film"])
def fx_vhs(img, p, ctx):
    arr = _np(img)
    h, w = arr.shape[:2]
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    dx = np.sin(yy * 0.35 + int(ctx.get("t_local", 0) / 66.0)) * p["wobble"]
    out = _np(_remap(img, np.repeat(dx[..., None], w, axis=1) if dx.ndim == 2 else dx, np.zeros_like(dx)))
    shift = int(p["color_bleed"])
    if shift:
        out[..., 0] = np.roll(out[..., 0], shift, axis=1)
    n = _noise((h, w), seed=int(ctx.get("t_local", 0) // 33))
    out[..., :3] += (n - 0.5) * p["noise"] * 0.4
    return _pil(out)


@register("light_leak", "Stylize", "Light Leak",
          {"color": C("color", "#FFFFAA44"), "amount": F("amount", 0.5, 0, 1), "x": F("x", 0.8, 0, 1), "y": F("y", 0.2, 0, 1)},
          [AM + "lightleak", AM + "lensflare"])
def fx_light_leak(img, p, ctx):

    rgb, a = _rgb(img)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / max(1, w)) - p["x"]) ** 2 + ((yy / max(1, h)) - p["y"]) ** 2)
    glow = np.clip(1 - d * 1.6, 0, 1)[..., None] ** 2
    col = np.array(parse_color(p["color"])[:3], np.float32)
    return _merge(rgb + glow * col * p["amount"], a)


# ------------------------------------------------------------------ distort


@register("wave", "Distort", "Wave",
          {"amp": F("amp", 12.0, 0, 120), "freq": F("freq", 3.0, 0.2, 30), "speed": F("speed", 1.0, -6, 6)},
          [AM + "wavewarp", AM + "wavewarp2", AM + "wave"])
def fx_wave(img, p, ctx):
    h, w = img.size[1], img.size[0]
    t = ctx.get("t_local", 0) / 1000.0 * p["speed"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = np.sin(xx / max(1, w) * p["freq"] * 2 * math.pi + t * 2 * math.pi) * p["amp"]
    return _remap(img, np.zeros_like(dy), dy)


@register("ripple", "Distort", "Ripple",
          {"amp": F("amp", 14.0, 0, 150), "freq": F("freq", 12.0, 1, 80), "speed": F("speed", 1.0, -6, 6), "center_x": F("center_x", 0.5, 0, 1), "center_y": F("center_y", 0.5, 0, 1)},
          [AM + "circularripple", AM + "circularripple2", AM + "ripple"])
def fx_ripple(img, p, ctx):
    h, w = img.size[1], img.size[0]
    t = ctx.get("t_local", 0) / 1000.0 * p["speed"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = p["center_x"] * w, p["center_y"] * h
    dxp, dyp = xx - cx, yy - cy
    r = np.sqrt(dxp * dxp + dyp * dyp) + 1e-3
    f = np.sin(r / max(1, w) * p["freq"] * 2 * math.pi - t * 2 * math.pi) * p["amp"]
    return _remap(img, dxp / r * f, dyp / r * f)


@register("twirl", "Distort", "Twirl",
          {"angle": F("angle", 45.0, -360, 360, 1), "radius": F("radius", 0.5, 0.05, 1)},
          [AM + "twirl", AM + "swirl"])
def fx_twirl(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dxp, dyp = xx - cx, yy - cy
    r = np.sqrt(dxp * dxp + dyp * dyp) / (min(w, h) * 0.5)
    ang = np.radians(p["angle"]) * np.clip(1 - r / max(0.05, p["radius"]), 0, 1) ** 2
    ca, sa = np.cos(ang), np.sin(ang)
    sx = cx + dxp * ca - dyp * sa
    sy = cy + dxp * sa + dyp * ca
    return _remap(img, sx - xx, sy - yy)


@register("bulge", "Distort", "Bulge / Pinch",
          {"amount": F("amount", 0.35, -0.9, 0.9), "radius": F("radius", 0.6, 0.05, 1)},
          [AM + "pinchbulge", AM + "pinchbulge2", AM + "pinchbulgeinside", AM + "bulge", AM + "pinch"])
def fx_bulge(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    nx, ny = (xx - cx) / (w * 0.5), (yy - cy) / (h * 0.5)
    r = np.sqrt(nx * nx + ny * ny)
    f = np.clip(1 - r / max(0.05, p["radius"]), 0, 1) ** 2 * p["amount"]
    return _remap(img, -nx * f * w * 0.5, -ny * f * h * 0.5)


@register("mirror", "Distort", "Mirror", {"side": CH("side", "left", ["left", "right", "top", "bottom"])},
          [AM + "mirror", AM + "mirror2"])
def fx_mirror(img, p, ctx):
    arr = _np(img)
    h, w = arr.shape[:2]
    out = arr.copy()
    if p["side"] == "left":
        out[:, w // 2:] = arr[:, : w - w // 2][:, ::-1]
    elif p["side"] == "right":
        out[:, : w // 2] = arr[:, w - w // 2:][:, ::-1]
    elif p["side"] == "top":
        out[h // 2:] = arr[: h - h // 2][::-1]
    else:
        out[: h // 2] = arr[h - h // 2:][::-1]
    return _pil(out)


@register("kaleidoscope", "Distort", "Kaleidoscope", {"segments": I("segments", 6, 2, 24), "rotation": F("rotation", 0, -180, 180, 1)},
          [AM + "kaleidoscope", AM + "kaleido"])
def fx_kaleido(img, p, ctx):
    n = max(2, p["segments"])
    w, h = img.size
    ang = 360.0 / n
    if w > h:
        crop = img.crop(((w - h) // 2, 0, (w - h) // 2 + h, h))
    else:
        crop = img.crop((0, (h - w) // 2, w, (h - w) // 2 + w))
    crop = crop.resize((512, 512), Image.BILINEAR)
    wedge = crop.rotate(p["rotation"]).crop((0, 0, 256, 512)).resize((256, 512), Image.BILINEAR)
    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    for i in range(n):
        wimg = wedge.rotate(-ang * i, center=(512, 512)) if i % 2 == 0 else wedge.transpose(Image.FLIP_LEFT_RIGHT).rotate(-ang * i, center=(512, 512))
        canvas.alpha_composite(wimg, (0, 0))
    return canvas.resize((w, h), Image.BILINEAR)


@register("displace", "Distort", "Noise Displace",
          {"amount": F("amount", 20.0, 0, 200), "scale": F("scale", 60.0, 4, 400), "speed": F("speed", 1.0, 0, 6)},
          [AM + "displace", AM + "displacement", AM + "noisedisplace"])
def fx_displace(img, p, ctx):
    h, w = img.size[1], img.size[0]
    t = int(ctx.get("t_local", 0) / 100.0) % 10000
    rng = np.random.default_rng(t + int(p["scale"]))
    sh = (h // int(max(4, p["scale"])) + 2, w // int(max(4, p["scale"])) + 2)
    nx = rng.random(sh).astype(np.float32)
    ny = rng.random(sh).astype(np.float32)
    nx = np.asarray(Image.fromarray((nx * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 127.5 - 1
    ny = np.asarray(Image.fromarray((ny * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 127.5 - 1
    return _remap(img, nx * p["amount"], ny * p["amount"])


# ------------------------------------------------------------------ transitions


@register("fade", "Transition", "Fade",
          {"progress": F("progress", -1.0, -1, 1), "mode": CH("mode", "in", ["in", "out"])},
          [AM + "fade", AM + "fadein", AM + "fadeout"])
def fx_fade(img, p, ctx):
    prog = p["progress"]
    if prog < 0:
        prog = ctx.get("extent", 0.0)

    def ease(u):
        u = max(0.0, min(1.0, u))
        return u * u * (3 - 2 * u)

    x = ease(prog) if p["mode"] == "in" else 1.0 - ease(prog)
    a = _np(img)[..., 3:4] * x
    rgb = _np(img)[..., :3]
    return _merge(rgb, a)


@register("wipe", "Transition", "Wipe",
          {"progress": F("progress", -1.0, -1, 1), "angle": F("angle", 0.0, -180, 180, 1), "softness": F("softness", 0.08, 0.0, 0.6)},
          [AM + "wipe", AM + "wipetransition", AM + "checkerdissolve", AM + "checkerdissolve2"])
def fx_wipe(img, p, ctx):
    prog = p["progress"] if p["progress"] >= 0 else ctx.get("extent", 0.0)
    prog = max(0.0, min(1.0, prog))
    arr = _np(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    proj = (xx / w) * math.cos(rad) + (yy / h) * math.sin(rad)
    proj = (proj + 1.0) / 2.0
    soft = max(0.001, p["softness"])
    mask = np.clip((prog - proj) / soft + 0.5, 0, 1)[..., None]
    arr[..., 3:4] *= mask
    return _pil(arr)


@register("iris", "Transition", "Iris",
          {"progress": F("progress", -1.0, -1, 1), "feather": F("feather", 0.12, 0.01, 0.8)},
          [AM + "iris", AM + "iristransition"])
def fx_iris(img, p, ctx):
    prog = p["progress"] if p["progress"] >= 0 else ctx.get("extent", 0.0)
    prog = max(0.0, min(1.0, prog))
    arr = _np(img)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w * 0.55)) ** 2 + ((yy - h / 2) / (h * 0.55)) ** 2)
    mask = np.clip((prog * (1 + p["feather"]) - d) / p["feather"], 0, 1)[..., None]
    arr[..., 3:4] *= mask
    return _pil(arr)


@register("dissolve", "Transition", "Noise Dissolve",
          {"progress": F("progress", -1.0, -1, 1), "scale": F("scale", 1.0, 0.2, 4)},
          [AM + "dissolve", AM + "noisedissolve"])
def fx_dissolve(img, p, ctx):
    prog = p["progress"] if p["progress"] >= 0 else ctx.get("extent", 0.0)
    prog = max(0.0, min(1.0, prog))
    arr = _np(img)
    h, w = arr.shape[:2]
    n = _noise((h, w), seed=7)
    if p["scale"] > 1.2:
        f = max(1.0, float(p["scale"]))
        sw, sh = max(1, int(w / f)), max(1, int(h / f))
        n = np.asarray(Image.fromarray(((n[..., 0]) * 255).astype(np.uint8), "L")
                       .resize((sw, sh), Image.NEAREST).resize((w, h), Image.NEAREST), np.float32)[..., None] / 255.0
    mask = (n < prog).astype(np.float32)
    arr[..., 3:4] = arr[..., 3:4] * mask
    return _pil(arr)


@register("slide", "Transition", "Slide",
          {"progress": F("progress", -1.0, -1, 1), "direction": CH("direction", "left", ["left", "right", "up", "down"])},
          [AM + "slide", AM + "slidetransition"])
def fx_slide(img, p, ctx):
    prog = p["progress"] if p["progress"] >= 0 else ctx.get("extent", 0.0)
    prog = max(0.0, min(1.0, prog))
    w, h = img.size
    d = p["direction"]
    dx = -prog * w if d == "left" else prog * w if d == "right" else 0
    dy = -prog * h if d == "up" else prog * h if d == "down" else 0
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(img, (int(dx), int(dy)))
    return canvas


# ------------------------------------------------------------------ generate / misc


@register("drop_shadow", "Generate", "Drop Shadow",
          {"offset_x": F("offset_x", 8.0, -100, 100), "offset_y": F("offset_y", 8.0, -100, 100),
           "blur": F("blur", 8.0, 0, 80), "opacity": F("opacity", 0.5, 0, 1),
           "color": C("color", "#FF000000"), "grow": F("grow", 0.0, 0, 20)},
          [AM + "shadow", AM + "longshadow", AM + "longshadow2", AM + "radialshadow", AM + "radialshadow2",
           AM + "internal-shadow"])
def fx_drop_shadow(img, p, ctx):

    a = _np(img)[..., 3:4]
    alpha_img = Image.fromarray((a[..., 0] * 255).astype(np.uint8), "L")
    if p["grow"] > 0:
        alpha_img = alpha_img.filter(ImageFilter.MaxFilter(int(2 * round(p["grow"]) + 1)))
    if p["blur"] > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(p["blur"]))
    col = np.array(parse_color(p["color"]), np.float32)
    sh = np.zeros((*a.shape[:2], 4), np.float32)
    sh[..., :3] = col[:3]
    sh[..., 3] = (np.asarray(alpha_img, np.float32) / 255.0) * p["opacity"] * col[3]
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh_img = Image.fromarray((np.clip(sh, 0, 1) * 255).astype(np.uint8), "RGBA")
    out.alpha_composite(sh_img, (int(p["offset_x"]), int(p["offset_y"])))
    out.alpha_composite(img)
    return out


@register("gradient_map", "Generate", "Gradient Overlay",
          {"color1": C("color1", "#00000000"), "color2": C("color2", "#FF000000"),
           "angle": F("angle", 90.0, 0, 360, 1), "amount": F("amount", 0.6, 0, 1)},
          [AM + "gradient", AM + "gradientoverlay", AM + "fourcolorgradient"])
def fx_gradient_overlay(img, p, ctx):

    rgb, a = _rgb(img)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    t = (xx / max(1, w) * math.cos(rad) + yy / max(1, h) * math.sin(rad))
    t = (t - t.min()) / max(1e-5, (t.max() - t.min()))
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    grad = c1 + (c2 - c1) * t[..., None]
    return _merge(rgb * (1 - p["amount"]) + grad * p["amount"], a)


@register("vhs_jitter", "Distort", "Jitter / Shake",
          {"amount": F("amount", 12.0, 0, 80), "speed": F("speed", 20.0, 1, 60), "rotation": F("rotation", 1.0, 0, 10)},
          [AM + "shake", AM + "shake2", AM + "shakeparts", AM + "shake-parts", AM + "jitter"])
def fx_jitter(img, p, ctx):
    t = int(ctx.get("t_local", 0) / max(1.0, 1000.0 / p["speed"]))
    rng = np.random.default_rng(t + 17)
    dx = float(rng.normal(0, p["amount"] * 0.5))
    dy = float(rng.normal(0, p["amount"] * 0.5))
    rot = float(rng.normal(0, p["rotation"]))
    out = img.rotate(rot, resample=Image.BILINEAR, expand=False)
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    canvas.alpha_composite(out, (int(dx), int(dy)))
    return canvas


@register("mirror_repeat", "Distort", "Tile",
          {"tiles_x": I("tiles_x", 2, 1, 8), "tiles_y": I("tiles_y", 2, 1, 8), "mirror": B("mirror", True)},
          [AM + "tile", AM + "tiling"])
def fx_tile(img, p, ctx):
    tx, ty = max(1, p["tiles_x"]), max(1, p["tiles_y"])
    w, h = img.size
    tw, th = max(1, w // tx), max(1, h // ty)
    small = img.resize((tw, th), Image.BILINEAR)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for j in range(ty):
        for i in range(tx):
            t = small
            if p["mirror"] and (i + j) % 2 == 1:
                t = small.transpose(Image.FLIP_LEFT_RIGHT)
            out.alpha_composite(t, (i * tw, j * th))
    return out


# ------------------------------------------------------------------ lookup

AM_ALIASES: Dict[str, str] = {}
for _key, _eff in EFFECTS.items():
    for _aid in _eff.am_ids:
        AM_ALIASES[_aid] = _key


def rebuild_am_aliases() -> None:
    """Rebuild AM_ALIASES from every registered effect.

    Effects registered after this module finishes importing (effects_am,
    effects_audio) still carry AM ids, so the table is rebuilt once more at the
    bottom of the file.
    """
    for key, eff in EFFECTS.items():
        for aid in eff.am_ids:
            AM_ALIASES.setdefault(aid, key)

# Extra Alight Motion ids seen in real community presets. These map onto existing
# CraftMotion effects; the parameter names are translated by AM_PARAM_MAP below.
_AM_EXTRA = {
    "lift": "levels", "solidcolor": "solid_color", "colorhot": "thermal",
    "lensblurobs": "lens_blur", "swirl4": "twirl", "moblurplus": "motion_blur",
    "fractalwarp4": "jelly", "pagecurl": "page_curl", "rgbsep": "rgb_split",
    "stretch2": "pixel_stretch", "stretchsegment": "pixel_stretch",
    "chromakey": "chroma_key", "flip3": "mirror", "burningwipe": "wipe",
    "radialwipe": "wipe", "deepglow": "glow", "gaussianblurplus": "blur",
    "replacecolor": "color_replace", "bendwarp": "perspective", "smoothbevel": "emboss",
    "fourcolorgradient": "gradient_map", "gradientoverlay2": "gradient_overlay",
    "shaketransform": "oscillate", "shake2": "oscillate", "shake": "oscillate",
    "zoomblur3": "zoom_blur", "circularripple2": "ripple", "edgeglow": "neon_edges",
    "tile": "mirror_repeat", "displacemap3": "displacement", "texture2": "paper_texture",
}
for _aid, _key in _AM_EXTRA.items():
    # AM ships the same effect under effects.*, internal.* and bare namespaces
    for _ns in ("com.alightcreative.effects.", "com.alightcreative.internal.",
                "com.alightcreative."):
        AM_ALIASES.setdefault(_ns + _aid, _key)

# Short AM names ("oscillate3", "turbulentdisplace") are what people type and
# what AM_PARAM_MAP keys on; accept them anywhere a type string is accepted.
AM_SHORT_ALIASES: Dict[str, str] = {}


def rebuild_short_aliases() -> None:
    """Recompute both alias tables (namespace ids and bare AM names)."""
    rebuild_am_aliases()
    AM_SHORT_ALIASES.clear()
    for aid, key in AM_ALIASES.items():
        for ns in ("com.alightcreative.effects.", "com.alightcreative.internal.",
                   "com.alightcreative."):
            if aid.startswith(ns):
                AM_SHORT_ALIASES.setdefault(aid[len(ns):], key)
    for aid, key in _AM_EXTRA.items():
        AM_SHORT_ALIASES.setdefault(aid, key)


rebuild_short_aliases()

# AM parameter name -> our parameter name (per AM effect id suffix).
AM_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "lift": {"fill": "black"},
    "solidcolor": {"alpha": "opacity"},
    "colorhot": {"color": "tint_high"},
    "lensblurobs": {"RADIUS": "radius", "bright": "boost"},
    "swirl4": {"radius": "radius", "strength": "angle"},
    "moblurplus": {"shutter": "distance"},
    "rgbsep": {"strength": "offset", "angle": "angle"},
    "stretch2": {"scale": "amount", "angle": "angle"},
    "chromakey": {"keyColor": "color", "threshold": "tolerance", "feather": "softness"},
    "burningwipe": {"progressa": "progress", "deltab": "softness"},
    "radialwipe": {"progress": "progress", "feather": "softness"},
    "deepglow": {"strength": "strength", "rrr": "radius"},
    "gaussianblurplus": {"strength": "radius"},
    "replacecolor": {"oldcolor": "from", "newcolor": "to", "threshold": "tolerance",
                     "feather": "softness"},
    "bendwarp": {"amount": "tilt_y"},
    "smoothbevel": {"angle_d": "angle", "intensity_d": "depth"},
    "pagecurl": {"progress": "amount", "radius": "amount"},
    "turbulentdisplace3": {"intensity": "intensity", "scale": "scale",
                            "evolution": "evolution", "seed": "seed"},
    "turbulence2": {"intensity": "intensity", "scale": "scale", "octaves": "octaves",
                    "seed": "seed", "evolution": "evolution"},
    "randomdisplace": {"mag": "mag", "evolution": "evolution", "seed": "seed",
                        "scatter": "scatter"},
    "oscillate3": {"mag": "mag", "freq": "freq", "angle": "angle", "phase": "phase",
                    "direction": "direction"},
    "swing2": {"freq": "freq", "a1": "a1", "a2": "a2", "phase": "phase"},
    "blink2": {"freq": "freq"},
    "tile": {"scale": "scale"},
    "zoomblur3": {"strength": "amount", "center": "cx"},
    "circularripple2": {"amplitude": "amount", "frequency": "scale"},
    "edgeglow": {"width": "radius", "color": "color", "strength": "strength"},
}

# Values AM stores in *degrees* that our effects expect as radians/fraction.
AM_PARAM_SCALE: Dict[str, Dict[str, float]] = {
    "swing2": {"a1": 1.0, "a2": 1.0},   # already degrees
    "swirl4": {"angle": 1.0},
}


def resolve_type(name: str) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip()
    if n in EFFECTS:
        return n
    if n in AM_ALIASES:
        return AM_ALIASES[n]
    short = n.replace("com.alightcreative.internal.", "").replace(
        "com.alightcreative.effects.", "").replace("com.alightcreative.", "")
    if short in EFFECTS:
        return short
    if short in AM_SHORT_ALIASES:
        return AM_SHORT_ALIASES[short]
    for key in EFFECTS:
        if short.lower().startswith(key) or key.startswith(short.lower()):
            return key
    return None


def list_effects(category: Optional[str] = None) -> List[dict]:
    out = []
    for key, eff in EFFECTS.items():
        if category and eff.category.lower() != category.lower():
            continue
        out.append({
            "type": key,
            "label": eff.label,
            "category": eff.category,
            "params": {
                name: {
                    "default": prm.default, "min": prm.lo, "max": prm.hi,
                    "step": prm.step, "kind": prm.kind,
                    **({"options": prm.options} if prm.options else {}),
                }
                for name, prm in eff.params.items()
            },
            "am_ids": eff.am_ids,
        })
    return out


def categories() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for eff in EFFECTS.values():
        out[eff.category] = out.get(eff.category, 0) + 1
    return out


def defaults_for(key: str) -> dict:
    eff = EFFECTS.get(key)
    if not eff:
        return {}
    return {name: prm.default for name, prm in eff.params.items()}


def apply_effect(key: str, img: Image.Image, params: dict, ctx: dict) -> Image.Image:
    eff = EFFECTS.get(key)
    if eff is None:
        return img
    resolved = defaults_for(key)
    resolved.update({k: v for k, v in (params or {}).items() if v is not None})
    try:
        return eff.func(img, resolved, ctx)
    except Exception:
        return img


# ------------------------------------------------------------------ extra suite


@register("chroma_key", "Keying", "Chroma Key (green/blue screen)",
          {"color": C("color", "#FF00FF00"), "tolerance": F("tolerance", 0.35, 0.01, 1),
           "softness": F("softness", 0.12, 0.0, 0.5), "spill": F("spill", 0.6, 0, 1)},
          [AM + "chromakey", AM + "chromakey-adv", AM + "greenscreen"])
def fx_chroma(img, p, ctx):

    rgb, a = _rgb(img)
    key = np.array(parse_color(p["color"])[:3], np.float32)
    dist = np.sqrt(((rgb - key) ** 2).sum(axis=-1, keepdims=True)) / math.sqrt(3.0)
    tol = max(0.01, p["tolerance"])
    soft = max(0.001, p["softness"])
    mask = np.clip((dist - tol) / soft + 0.5, 0, 1)
    # spill suppression: pull channels toward key channel balance
    if p["spill"] > 0:
        kmax = key.argmax()
        others = [i for i in range(3) if i != kmax]
        spill_amt = np.clip(rgb[..., kmax] - np.maximum(rgb[..., others[0]], rgb[..., others[1]]), 0, 1)
        rgb2 = rgb.copy()
        rgb2[..., kmax] = rgb[..., kmax] - spill_amt * p["spill"]
        rgb = rgb2
    return _merge(rgb, a * mask)


@register("luma_key", "Keying", "Luma Key",
          {"threshold": F("threshold", 0.12, 0, 1), "softness": F("softness", 0.15, 0.01, 0.6), "invert": B("invert", False)},
          [AM + "lumakey", AM + "luminancekey"])
def fx_luma_key(img, p, ctx):
    rgb, a = _rgb(img)
    l = _luma(rgb)
    if p["invert"]:
        mask = np.clip((l - p["threshold"]) / max(0.01, p["softness"]), 0, 1)
    else:
        mask = 1.0 - np.clip((l - p["threshold"]) / max(0.01, p["softness"]), 0, 1)
    return _merge(rgb, a * mask)


@register("median_denoise", "Stylize", "Denoise (Median)",
          {"size": I("size", 3, 3, 9), "mix": F("mix", 1.0, 0, 1)}, [AM + "denoise", AM + "noisereduce"])
def fx_median(img, p, ctx):
    s = int(p["size"]) | 1
    med = img.filter(ImageFilter.MedianFilter(s))
    a = _np(img)
    m = _np(med)
    out = a * (1 - p["mix"]) + m * p["mix"]
    out[..., 3:4] = a[..., 3:4]
    return _pil(out)


@register("sketch", "Stylize", "Pencil Sketch",
          {"strength": F("strength", 1.0, 0, 2), "paper": F("paper", 0.92, 0.4, 1)}, [AM + "sketch", AM + "pencil"])
def fx_sketch(img, p, ctx):
    grey = img.convert("L")
    inv = Image.eval(grey, lambda x: 255 - x)
    blur = inv.filter(ImageFilter.GaussianBlur(6))
    a = np.asarray(grey, np.float32) / 255.0
    b = np.asarray(blur, np.float32) / 255.0
    dodge = np.clip(a / np.maximum(1e-3, 1 - b), 0, 1) * p["paper"]
    out = np.repeat(dodge[..., None], 3, axis=-1)
    alpha = _np(img)[..., 3:4]
    return _merge(out, alpha)


@register("border_frame", "Generate", "Border Frame",
          {"width": F("width", 12.0, 0, 120), "color": C("color", "#FFFFFFFF"),
           "radius": F("radius", 0.0, 0, 200), "softness": F("softness", 0.0, 0, 40)},
          [AM + "border", AM + "frame"])
def fx_border_frame(img, p, ctx):

    arr = _np(img)
    h, w = arr.shape[:2]
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    from .util import hex_to_rgb255
    col = hex_to_rgb255(p["color"])
    draw = ImageDraw.Draw(canvas)
    bw = int(max(0, p["width"]))
    if bw > 0:
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=p["radius"], outline=col, width=bw)
    base = Image.fromarray((arr * 255).astype(np.uint8), "RGBA")
    # keep picture inside the border
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    inset = bw + int(p["softness"])
    md.rounded_rectangle((inset, inset, w - 1 - inset, h - 1 - inset), radius=max(0, p["radius"] - inset), fill=255)
    if inset > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(max(0.0, p["softness"] / 2)))
    inner = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    inner.paste(base, (0, 0), mask)
    inner.alpha_composite(canvas)
    return inner


@register("film_flicker", "Stylize", "Film Flicker",
          {"amount": F("amount", 0.08, 0, 0.5), "speed": F("speed", 10.0, 1, 60)},
          [AM + "flicker", AM + "oldfilmflicker"])
def fx_flicker(img, p, ctx):
    rgb, a = _rgb(img)
    t = int(ctx.get("t_local", 0) / max(1.0, 1000.0 / p["speed"]))
    rng = np.random.default_rng(t * 131 + 3)
    k = 1.0 + float(rng.normal(0, p["amount"]))
    return _merge(rgb * k, a)


@register("zoom_punch", "Transition", "Zoom Punch",
          {"progress": F("progress", -1.0, -1, 1), "strength": F("strength", 0.25, 0, 1)},
          [AM + "zoomtransition", AM + "punch"])
def fx_zoom_punch(img, p, ctx):
    prog = p["progress"] if p["progress"] >= 0 else ctx.get("extent", 0.0)
    prog = max(0.0, min(1.0, prog))
    k = 1.0 + math.sin(prog * math.pi) * p["strength"]
    w, h = img.size
    nw, nh = int(w * k), int(h * k)
    big = img.resize((nw, nh), Image.BILINEAR)
    ox, oy = (nw - w) // 2, (nh - h) // 2
    return big.crop((ox, oy, ox + w, oy + h))


@register("outline", "Stylize", "Outline",
          {"width": F("width", 4.0, 0, 40), "color": C("color", "#FFFFFFFF")},
          [AM + "outline", AM + "stroke"])
def fx_outline(img, p, ctx):

    a = _np(img)[..., 3:4]
    alpha = Image.fromarray((a[..., 0] * 255).astype(np.uint8), "L")
    k = int(max(1, p["width"])) * 2 + 1
    grown = alpha.filter(ImageFilter.MaxFilter(k))
    edge = np.asarray(grown, np.float32) / 255.0 - np.asarray(alpha, np.float32) / 255.0
    edge = np.clip(edge, 0, 1)[..., None]
    col = np.array(parse_color(p["color"]), np.float32)
    base = _np(img)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ol = np.zeros_like(base)
    ol[..., :3] = col[:3]
    ol[..., 3] = edge[..., 0] * col[3]
    out.alpha_composite(Image.fromarray((ol * 255).astype(np.uint8), "RGBA"))
    out.alpha_composite(img)
    return out


@register("cross_process", "Color", "Cross Process",
          {"amount": F("amount", 1.0, 0, 1)}, [AM + "crossprocess", AM + "lomo"])
def fx_cross(img, p, ctx):
    rgb, a = _rgb(img)
    r = np.clip(rgb[..., 0] * 1.18 + 0.04, 0, 1)
    g = np.clip(rgb[..., 1] ** 0.92, 0, 1)
    b = np.clip(rgb[..., 2] * 0.82 + 0.06, 0, 1)
    out = np.stack([r, g, b], axis=-1)
    return _merge(rgb * (1 - p["amount"]) + out * p["amount"], a)


@register("neon_edges", "Stylize", "Neon Edges",
          {"color": C("color", "#FF00E5FF"), "strength": F("strength", 1.0, 0, 3), "glow": F("glow", 6.0, 0, 40)},
          [AM + "neon", AM + "glowedges"])
def fx_neon(img, p, ctx):

    edges = img.convert("RGB").filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges, np.float32) / 255.0
    e = np.clip(e.max(axis=-1, keepdims=True) * p["strength"], 0, 1)
    glow = np.asarray(Image.fromarray((e[..., 0] * 255).astype(np.uint8), "L")
                      .filter(ImageFilter.GaussianBlur(max(0.5, p["glow"]))), np.float32)[..., None] / 255.0
    col = np.array(parse_color(p["color"])[:3], np.float32)
    rgb = _np(img)[..., :3] * 0.25 + col * (e + glow * 0.8)
    alpha = _np(img)[..., 3:4]
    return _merge(rgb, alpha)


# Extended catalogue (imported last so `register` is available).
from . import effects_ext  # noqa: E402,F401  (populates the registry on import)


# Audio-reactive effects live in their own module (they need the mixed soundtrack);
# importing them last keeps the shared registration helpers available.
from . import effects_am  # noqa: E402,F401
from . import effects_audio  # noqa: E402,F401

rebuild_short_aliases()  # the late modules register AM ids too

# The add-on modules register more effects (and AM ids), so rebuild the alias map.
for _key, _eff in EFFECTS.items():
    for _aid in _eff.am_ids:
        AM_ALIASES[_aid] = _key
