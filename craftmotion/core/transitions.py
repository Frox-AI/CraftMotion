"""Transition catalogue — 30 film-style transitions between two frames.

Each transition is `fn(a, b, progress, params)` -> RGBA image, where `a` is the
outgoing frame and `b` the incoming frame (both canvas sized).

Transitions are used in two places:
  * layer transitions (`layer["transition_in"]` / `["transition_out"]`)
  * the timeline "apply transition between clips" command
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .util import parse_color

TRANSITIONS: Dict[str, dict] = {}


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def _pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGBA")


def register_transition(key: str, label: str, params: Optional[dict] = None,
                        categories: str = "Dissolve"):
    def deco(fn: Callable):
        TRANSITIONS[key] = {"key": key, "label": label, "func": fn,
                            "params": params or {}, "category": categories}
        return fn
    return deco


def _clamp01(v) -> float:
    return max(0.0, min(1.0, float(v)))


def _lerp(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
    return _pil(_np(a) * (1 - t) + _np(b) * t)


def _mask_transition(a: Image.Image, b: Image.Image, mask: np.ndarray) -> Image.Image:
    m = np.clip(mask, 0, 1)[..., None]
    return _pil(_np(a) * (1 - m) + _np(b) * m)


def _wipe_mask(size: Tuple[int, int], direction: str, t: float, softness: float = 0.0) -> np.ndarray:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if direction == "left":
        g = 1 - xx / max(1, w - 1)
    elif direction == "right":
        g = xx / max(1, w - 1)
    elif direction == "up":
        g = 1 - yy / max(1, h - 1)
    elif direction == "down":
        g = yy / max(1, h - 1)
    elif direction == "diagonal":
        g = (xx / max(1, w - 1) + yy / max(1, h - 1)) / 2.0
    elif direction == "diagonal2":
        g = (1 - xx / max(1, w - 1) + yy / max(1, h - 1)) / 2.0
    else:  # radial from centre
        r = np.sqrt(((xx - w / 2) / max(1, w / 2)) ** 2 + ((yy - h / 2) / max(1, h / 2)) ** 2)
        g = np.clip(r / 1.4142, 0, 1)
    edge = max(1e-3, softness)
    return np.clip((t * (1 + edge) - g) / edge, 0, 1)


# ------------------------------------------------------------------ dissolves
@register_transition("cross_dissolve", "Cross Dissolve", {"softness": 0.0})
def t_dissolve(a, b, p, params):
    return _lerp(a, b, p)


@register_transition("additive_dissolve", "Additive Dissolve")
def t_additive(a, b, p, params):
    arr = np.clip(_np(a) + _np(b) * p, 0, 1)
    return _pil(arr)


@register_transition("dip_to_black", "Dip to Black", {"color": None})
def t_dip_black(a, b, p, params):
    if p < 0.5:
        return _lerp(a, Image.new("RGBA", a.size, (0, 0, 0, 255)), p * 2)
    return _lerp(Image.new("RGBA", a.size, (0, 0, 0, 255)), b, (p - 0.5) * 2)


@register_transition("dip_to_white", "Dip to White")
def t_dip_white(a, b, p, params):
    if p < 0.5:
        return _lerp(a, Image.new("RGBA", a.size, (255, 255, 255, 255)), p * 2)
    return _lerp(Image.new("RGBA", a.size, (255, 255, 255, 255)), b, (p - 0.5) * 2)


@register_transition("dip_to_color", "Dip to Colour", {"color": "#FF2233AA"})
def t_dip_color(a, b, p, params):
    c = parse_color(params.get("color", "#FF2233AA"))
    col = Image.new("RGBA", a.size, (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255))
    if p < 0.5:
        return _lerp(a, col, p * 2)
    return _lerp(col, b, (p - 0.5) * 2)


@register_transition("flash", "Flash Cut")
def t_flash(a, b, p, params):
    arr = _np(_lerp(a, b, p))
    k = math.sin(p * math.pi) ** 0.5
    arr[..., :3] = np.clip(arr[..., :3] + k * 0.9, 0, 1)
    return _pil(arr)


@register_transition("luma_wipe", "Luma Wipe", {"softness": 0.15})
def t_luma(a, b, p, params):
    la = 0.2126 * _np(a)[..., 0] + 0.7152 * _np(a)[..., 1] + 0.0722 * _np(a)[..., 2]
    lb = 0.2126 * _np(b)[..., 0] + 0.7152 * _np(b)[..., 1] + 0.0722 * _np(b)[..., 2]
    soft = max(1e-3, float(params.get("softness", 0.15)))
    mask = np.clip((p * (1 + soft) * 2 - la), 0, 1)
    out = _mask_transition(a, b, mask)
    return out


@register_transition("gradient_wipe", "Gradient Wipe", {"angle": 45.0, "softness": 0.25})
def t_gradient(a, b, p, params):
    w, h = a.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(float(params.get("angle", 45.0)))
    g = (xx / max(1, w)) * math.cos(rad) + (yy / max(1, h)) * math.sin(rad)
    g = (g - g.min()) / max(1e-5, g.max() - g.min())
    soft = max(1e-3, float(params.get("softness", 0.25)))
    mask = np.clip((p * (1 + soft) - g) / soft, 0, 1)
    return _mask_transition(a, b, mask)


# ------------------------------------------------------------------ wipes
for _dir in ("left", "right", "up", "down", "diagonal", "diagonal2", "radial"):
    def _mk(direction=_dir):
        def apply(a, b, p, params):
            soft = float(params.get("softness", 0.02))
            return _mask_transition(a, b, _wipe_mask(a.size, direction, p, soft))
        return apply
    register_transition(f"wipe_{_dir}", f"Wipe — {_dir.replace('2', ' reverse').title()}",
                        {"softness": 0.02}, "Wipe")(_mk())


@register_transition("clock_wipe", "Clock Wipe")
def t_clock(a, b, p, params):
    w, h = a.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = (np.arctan2(xx - w / 2, -(yy - h / 2)) % (2 * math.pi)) / (2 * math.pi)
    mask = (ang <= p).astype(np.float32)
    return _mask_transition(a, b, mask)


@register_transition("blinds", "Blinds", {"count": 12, "vertical": False})
def t_blinds(a, b, p, params):
    w, h = a.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = max(2, int(params.get("count", 12)))
    coord = (xx / max(1, w)) if not params.get("vertical") else (yy / max(1, h))
    band = np.mod(coord * n, 1.0)
    mask = (band <= p).astype(np.float32)
    return _mask_transition(a, b, mask)


@register_transition("checker", "Checkerboard", {"count": 8})
def t_checker(a, b, p, params):
    w, h = a.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = max(2, int(params.get("count", 8)))
    pat = (np.floor(xx / max(1, w) * n) + np.floor(yy / max(1, h) * n)) % 2
    thresh = np.where(pat == 0, p * 2, max(0.0, p * 2 - 1))
    mask = (thresh >= 1).astype(np.float32)
    return _mask_transition(a, b, mask)


@register_transition("mosaic", "Mosaic", {"cells": 16})
def t_mosaic(a, b, p, params):
    w, h = a.size
    n = max(2, int(params.get("cells", 16)))
    cw, ch = max(1, int(w / n)), max(1, int(h / n))
    small_a = a.resize((n, n), Image.BILINEAR)
    small_b = b.resize((n, n), Image.BILINEAR)
    blend = Image.blend(small_a, small_b, p).resize((w, h), Image.NEAREST)
    # blend the mosaic with the full-res cross dissolve so the reveal is smooth
    return _lerp(_lerp(a, b, p * 0.6), blend, 1.0 - p * 0.4)


@register_transition("pixelize_dissolve", "Pixelize Dissolve", {"cells": 24})
def t_pixelize(a, b, p, params):
    w, h = a.size
    n = max(2, int(params.get("cells", 24) * max(0.2, 1 - p * 0.8)))
    small = Image.blend(a.resize((n, n), Image.BILINEAR), b.resize((n, n), Image.BILINEAR), p)
    return small.resize((w, h), Image.NEAREST)


@register_transition("iris", "Iris", {"softness": 0.2})
def t_iris(a, b, p, params):
    w, h = a.size
    soft = max(1e-3, float(params.get("softness", 0.2)))
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    r = p * math.hypot(w, h) * 1.05
    d.ellipse((w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(0.5, soft * min(w, h) * 0.05)))
    return Image.composite(b, a, mask)


@register_transition("shape_wipe", "Shape Wipe", {"shape": "star"})
def t_shape(a, b, p, params):
    w, h = a.size
    shape = params.get("shape", "star")
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    r = p * math.hypot(w, h) * 0.62
    cx, cy = w / 2, h / 2
    if shape == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.5
            pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
        d.polygon(pts, fill=255)
    elif shape == "heart":
        pts = []
        for i in range(120):
            ang = i / 119 * 2 * math.pi
            x = 16 * math.sin(ang) ** 3
            y = 13 * math.cos(ang) - 5 * math.cos(2 * ang) - 2 * math.cos(3 * ang) - math.cos(4 * ang)
            pts.append((cx + x / 18 * r, cy - y / 18 * r))
        d.polygon(pts, fill=255)
    else:
        d.rectangle((cx - r / 1.4, cy - r / 1.4, cx + r / 1.4, cy + r / 1.4), fill=255)
    return Image.composite(b, a, mask.filter(ImageFilter.GaussianBlur(2)))


# ------------------------------------------------------------------ movement
def _shift_wrap(img: Image.Image, dx: float, dy: float) -> Image.Image:
    a = _np(img)
    return _pil(np.roll(np.roll(a, int(round(dy)), axis=0), int(round(dx)), axis=1))


def _shift_pad(img: Image.Image, dx: float, dy: float, fill=(0, 0, 0, 255)) -> Image.Image:
    w, h = img.size
    out = Image.new("RGBA", (w, h), fill)
    out.paste(img, (int(round(dx)), int(round(dy))))
    return out


for _dir, (_dx, _dy) in {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.items():
    def _mkpush(dx=_dx, dy=_dy):
        def apply(a, b, p, params):
            w, h = a.size
            ea = _shift_pad(a, dx * w * p, dy * h * p)
            eb = _shift_pad(b, -dx * w * (1 - p), -dy * h * (1 - p))
            # incoming sits above outgoing where it has content
            arrb = _np(eb)
            mask = (arrb[..., 3:4] > 0.01).astype(np.float32)
            return _pil(_np(ea) * (1 - mask) + arrb * mask)
        return apply
    register_transition(f"push_{_dir}", f"Push — {_dir.title()}", {}, "Move")(_mkpush())


for _dir, (_dx, _dy) in {"left": (-1, 0), "up": (0, -1)}.items():
    def _mkslide(dx=_dx, dy=_dy):
        def apply(a, b, p, params):
            w, h = a.size
            ea = _shift_pad(a, dx * w * p, dy * h * p)
            eb = _shift_pad(b, dx * w * (1 - p), dy * h * (1 - p))
            out = eb.copy()
            out.alpha_composite(ea)
            return out
        return apply
    register_transition(f"slide_{_dir}", f"Slide — {_dir.title()}", {}, "Move")(_mkslide())


@register_transition("whip_pan", "Whip Pan")
def t_whip(a, b, p, params):
    w, h = a.size
    blur_amt = math.sin(p * math.pi) * 0.25
    ea = _shift_pad(a, -w * p * 1.2, 0)
    eb = _shift_pad(b, w * (1 - p) * 1.2, 0)
    arr = _np(ea) * (1 - p) + _np(eb) * p
    out = _pil(arr)
    if blur_amt > 0.01:
        out = out.filter(ImageFilter.GaussianBlur(blur_amt * w * 0.05))
    return out


@register_transition("zoom_transition", "Zoom", {"reverse": False})
def t_zoom(a, b, p, params):
    w, h = a.size
    if not params.get("reverse"):
        k = 1.0 + p * 0.6
        small_a = a.resize((max(1, int(w / k)), max(1, int(h / k))), Image.BILINEAR)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(small_a, ((w - small_a.size[0]) // 2, (h - small_a.size[1]) // 2))
        ca = canvas
    else:
        k = 1.0 + (1 - p) * 0.6
        small_a = a.resize((max(1, int(w / k)), max(1, int(h / k))), Image.BILINEAR)
        ca = small_a.resize((w, h), Image.BILINEAR)
    return _lerp(ca, b, p)


@register_transition("spin_transition", "Spin")
def t_spin(a, b, p, params):
    rotated = a.rotate(-p * 180, resample=Image.BILINEAR, expand=False)
    return _lerp(rotated, b, p)


@register_transition("squeeze_transition", "Squeeze", {"vertical": False})
def t_squeeze(a, b, p, params):
    w, h = a.size
    k = max(0.02, 1 - p)
    if not params.get("vertical"):
        small = a.resize((max(1, int(w * k)), h), Image.BILINEAR)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(small, ((w - small.size[0]) // 2, 0))
    else:
        small = a.resize((w, max(1, int(h * k))), Image.BILINEAR)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(small, (0, (h - small.size[1]) // 2))
    return _lerp(canvas, b, p)


@register_transition("cross_zoom", "Cross Zoom")
def t_cross_zoom(a, b, p, params):
    w, h = a.size
    if p < 0.5:
        k = 1.0 + p * 1.6
        small = a.resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)
        x = (small.size[0] - w) // 2
        y = (small.size[1] - h) // 2
        out = small.crop((x, y, x + w, y + h))
        return out.filter(ImageFilter.GaussianBlur(p * 8))
    k = max(0.05, 2.6 * (1 - p))
    small = b.resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(small, ((w - small.size[0]) // 2, (h - small.size[1]) // 2))
    return out.filter(ImageFilter.GaussianBlur((1 - p) * 8))


@register_transition("film_burn", "Film Burn")
def t_film_burn(a, b, p, params):
    w, h = a.size
    arr = _np(_lerp(a, b, p))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w * (0.3 + p * 0.4)) / max(1, w)) ** 2 + ((yy - h * 0.5) / max(1, h)) ** 2)
    burn = np.exp(-(r / max(0.02, 0.18 + 0.25 * math.sin(p * math.pi))) ** 2)
    col = np.array([1.0, 0.55, 0.18], np.float32)
    arr[..., :3] = np.clip(arr[..., :3] + burn[..., None] * col * 1.4 * math.sin(p * math.pi), 0, 1)
    return _pil(arr)


@register_transition("glitch_transition", "Glitch Cut")
def t_glitch(a, b, p, params):
    h, w = a.size[1], a.size[0]
    arr = _np(_lerp(a, b, p))
    rng = np.random.default_rng(int(p * 1000))
    n = int(24 * math.sin(p * math.pi))
    for _ in range(max(0, n)):
        y = int(rng.integers(0, h))
        bh = int(rng.integers(2, max(3, h // 20)))
        shift = int(rng.integers(-w // 8, w // 8))
        arr[y:min(h, y + bh)] = np.roll(arr[y:min(h, y + bh)], shift, axis=1)
    if 0.2 < p < 0.8:
        arr[..., 0] = np.roll(arr[..., 0], int(6 * math.sin(p * 20)), axis=1)
    return _pil(arr)


@register_transition("tv_power", "TV Power Off")
def t_tv_off(a, b, p, params):
    w, h = a.size
    if p < 0.5:
        k = 1 - p * 2
        squashed = a.resize((w, max(1, int(h * k))), Image.BILINEAR)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        canvas.paste(squashed, (0, (h - squashed.size[1]) // 2))
        arr = _np(canvas)
        arr[..., :3] = np.clip(arr[..., :3] * (0.4 + 0.6 * (1 - p)), 0, 1)
        return _pil(arr)
    k = (p - 0.5) * 2
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    squashed = b.resize((w, max(1, int(h * k))), Image.BILINEAR)
    canvas.paste(squashed, (0, (h - squashed.size[1]) // 2))
    arr = _np(canvas)
    arr[..., :3] = np.clip(arr[..., :3] * (0.4 + 0.6 * k), 0, 1)
    return _pil(arr)


@register_transition("morph_cut", "Morph Cut", {"hardness": 0.5})
def t_morph_cut(a, b, p, params):
    w, h = a.size
    blend = _lerp(a, b, p)
    arr = _np(blend)
    sharp = np.asarray(blend.filter(ImageFilter.UnsharpMask(radius=2, percent=int(120 * math.sin(p * math.pi)) + 20,
                                                            threshold=2)), np.float32) / 255.0
    k = math.sin(p * math.pi) * 0.8
    out = arr * (1 - k) + sharp * k
    return _pil(out)


@register_transition("split_slide", "Split Slide")
def t_split_slide(a, b, p, params):
    w, h = a.size
    half = w // 2
    top_a = a.crop((0, 0, half, h)).resize((half, h))
    top_b = b.crop((0, 0, half, h))
    merged = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ea = a.crop((0, 0, half, h))
    eb = b.crop((0, 0, half, h))
    merged.paste(_lerp(ea, eb, p), (0, 0))
    ea2 = a.crop((half, 0, w, h))
    eb2 = b.crop((half, 0, w, h))
    merged.paste(_lerp(ea2, eb2, 1 - p), (half, 0))
    return merged


@register_transition("burn_away", "Burn Away")
def t_burn_away(a, b, p, params):
    w, h = a.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(5)
    small = rng.random((max(2, h // 16), max(2, w // 16))).astype(np.float32)
    noise = np.asarray(Image.fromarray((small * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    g = np.clip(noise * 0.7 + (xx / max(1, w)) * 0.6, 0, 1)
    mask = np.clip((g - p) * 6 + 0.5, 0, 1)
    out = _np(_mask_transition(a, b, mask))
    edge = np.clip(1 - np.abs(mask - 0.5) * 2, 0, 1)[..., None]
    out[..., :3] = np.clip(out[..., :3] + edge * np.array([1.0, 0.45, 0.1], np.float32) * 0.9, 0, 1)
    return _pil(out)


def list_transitions() -> List[dict]:
    return [{"type": k, "label": v["label"], "category": v["category"],
             "params": {pk: {"default": pv} for pk, pv in (v["params"] or {}).items()}}
            for k, v in TRANSITIONS.items()]


def apply_transition(name: str, a: Image.Image, b: Image.Image, progress: float,
                     params: Optional[dict] = None) -> Image.Image:
    entry = TRANSITIONS.get(name) or TRANSITIONS.get("cross_dissolve")
    p = _clamp01(progress)
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    try:
        return entry["func"](a, b, p, params or {})
    except Exception:
        return _lerp(a, b, p)
