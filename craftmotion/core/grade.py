"""Project-level colour grade: curves, wheels, HSL, LUTs, HDR / log, linear light.

Everything here is applied once, at the end of the render pipeline, exactly like
a master grade node in a finishing app. Layer effects run before it.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .util import parse_color

DEFAULT_GRADE = {
    "enabled": False,
    "exposure": 0.0,          # stops
    "contrast": 1.0,
    "saturation": 1.0,
    "temperature": 0.0,       # -1..1 (cool -> warm)
    "tint": 0.0,              # -1..1 (green -> magenta)
    "lift": 0.0,              # black point
    "gamma": 1.0,
    "gain": 1.0,
    "wheels": {               # shadow / mid / highlight colour balance
        "shadows": {"hue": 220.0, "amount": 0.0},
        "midtones": {"hue": 40.0, "amount": 0.0},
        "highlights": {"hue": 45.0, "amount": 0.0},
    },
    "hsl": {"hue": 0.0, "sat": 0.0, "lum": 0.0},
    "curves": {"rgb": [], "r": [], "g": [], "b": []},   # [[x,y], ...] control points, 0..1
    "lut": {"path": "", "amount": 1.0},
    "transform": "none",      # none | slog3_to_709 | logc_to_709 | to_linear | aces
    "linear_light": False,    # composite in linear light
    "blend_linear": False,
}


def ensure_grade(project) -> dict:
    grade = project.data.setdefault("grade", {})
    for k, v in DEFAULT_GRADE.items():
        if k not in grade:
            grade[k] = v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, dict) else dict(vv))
                                                          for kk, vv in v.items()}
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if kk not in grade[k]:
                    grade[k][kk] = vv if not isinstance(vv, dict) else dict(vv)
    return grade


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), np.float32) / 255.0


def _pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGBA")


def srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _curve_apply(channel: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    if not points or len(points) < 2:
        return channel
    pts = sorted([[float(p[0]), float(p[1])] for p in points], key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], np.float32)
    ys = np.array([p[1] for p in pts], np.float32)
    x = np.clip(channel, 0, 1)
    return np.interp(x, xs, ys).astype(np.float32)


def _load_lut(path: str):
    from .effects_ext import _load_cube
    return _load_cube(path)


def _sample_lut(rgb: np.ndarray, size: int, table: np.ndarray) -> np.ndarray:
    from .effects_ext import _sample_lut as sample
    return sample(rgb, size, table)


def apply_grade(img: Image.Image, grade: Optional[dict], *, max_side: Optional[int] = None) -> Image.Image:
    if not grade or not grade.get("enabled"):
        return img
    a = _np(img)
    rgb = a[..., :3]
    alpha = a[..., 3:4]

    if grade.get("linear_light"):
        rgb = srgb_to_lin(rgb)

    tform = (grade.get("transform") or "none").lower()
    if tform in ("slog3_to_709", "slog3"):
        rgb = lin_to_srgb(np.clip((rgb - 0.09) / 0.6, 0, 1))
    elif tform in ("logc_to_709", "logc"):
        rgb = lin_to_srgb(np.clip((rgb - 0.15) / 0.6, 0, 1))
    elif tform == "to_linear":
        rgb = srgb_to_lin(rgb)
    elif tform == "aces":
        x = srgb_to_lin(rgb) * 1.6
        rgb = lin_to_srgb(np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1))

    ex = float(grade.get("exposure", 0.0) or 0.0)
    if abs(ex) > 1e-4:
        rgb = np.clip(rgb * (2.0 ** ex), 0, 1)

    lift = float(grade.get("lift", 0.0) or 0.0)
    gain = float(grade.get("gain", 1.0) or 1.0)
    gam = float(grade.get("gamma", 1.0) or 1.0)
    if abs(lift) > 1e-4 or abs(gain - 1) > 1e-4 or abs(gam - 1) > 1e-4:
        rgb = np.clip((rgb * gain + lift), 0, 1) ** (1.0 / max(0.05, gam))

    temp = float(grade.get("temperature", 0.0) or 0.0)
    tint = float(grade.get("tint", 0.0) or 0.0)
    if abs(temp) > 1e-4 or abs(tint) > 1e-4:
        rgb = rgb * np.array([1.0 + 0.22 * temp, 1.0 - 0.05 * abs(temp) + 0.1 * tint, 1.0 - 0.22 * temp],
                             np.float32)

    wheels = grade.get("wheels") or {}
    if any(abs(float(w.get("amount", 0) or 0)) > 1e-4 for w in wheels.values()):
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
        ws = 1.0 - np.clip((lum - 0.0) / 0.55, 0, 1)
        wh = np.clip((lum - 0.45) / 0.55, 0, 1)
        wm = np.clip(1.0 - ws - wh, 0, 1)
        for zone, weight in (("shadows", ws), ("midtones", wm), ("highlights", wh)):
            cfg = wheels.get(zone) or {}
            amt = float(cfg.get("amount", 0) or 0)
            if abs(amt) < 1e-4:
                continue
            rad = math.radians(float(cfg.get("hue", 0) or 0))
            col = np.array([math.cos(rad), math.cos(rad - 2.0944), math.cos(rad - 4.1888)], np.float32) * amt
            rgb = rgb + weight * col

    hsl = grade.get("hsl") or {}
    dh, ds, dl = float(hsl.get("hue", 0) or 0), float(hsl.get("sat", 0) or 0), float(hsl.get("lum", 0) or 0)
    if abs(dh) + abs(ds) + abs(dl) > 1e-4:
        from .effects_ext import _hsl
        rgb = _hsl(np.clip(rgb, 0, 1), dh, ds, dl)

    sat = float(grade.get("saturation", 1.0) or 1.0)
    if abs(sat - 1.0) > 1e-4:
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
        rgb = lum + (rgb - lum) * sat

    ctr = float(grade.get("contrast", 1.0) or 1.0)
    if abs(ctr - 1.0) > 1e-4:
        rgb = (rgb - 0.5) * ctr + 0.5

    curves = grade.get("curves") or {}
    if curves.get("rgb"):
        rgb = np.stack([_curve_apply(rgb[..., i], curves["rgb"]) for i in range(3)], -1)
    for i, ch in enumerate(("r", "g", "b")):
        if curves.get(ch):
            rgb[..., i] = _curve_apply(rgb[..., i], curves[ch])

    lut = grade.get("lut") or {}
    path = lut.get("path") or ""
    if path and os.path.exists(path):
        try:
            size, table = _load_lut(path)
            out = _sample_lut(np.clip(rgb, 0, 1), size, table)
            amt = float(lut.get("amount", 1.0) or 1.0)
            rgb = rgb * (1 - amt) + out * amt
        except Exception:
            pass

    rgb = np.clip(rgb, 0, 1)
    if grade.get("linear_light"):
        rgb = lin_to_srgb(rgb)
    return _pil(np.concatenate([rgb, alpha], -1))


def grade_summary(grade: dict) -> dict:
    g = ensure_grade_dict(grade)
    return {
        "enabled": bool(g.get("enabled")),
        "exposure": g.get("exposure"), "contrast": g.get("contrast"), "saturation": g.get("saturation"),
        "temperature": g.get("temperature"), "tint": g.get("tint"),
        "transform": g.get("transform"), "linear_light": g.get("linear_light"),
        "lut": (g.get("lut") or {}).get("path", ""),
        "curves": {k: len(v or []) for k, v in (g.get("curves") or {}).items()},
        "wheels": g.get("wheels"),
    }


def ensure_grade_dict(grade: dict) -> dict:
    out = dict(grade or {})
    for k, v in DEFAULT_GRADE.items():
        out.setdefault(k, v)
    return out


def scope_data(img: Image.Image) -> dict:
    """Histogram / waveform / vectorscope matrices for the scopes panel."""
    a = _np(img)
    rgb = a[..., :3]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    hist = {c: np.histogram(rgb[..., i], bins=64, range=(0, 1))[0].tolist() for i, c in enumerate("rgb")}
    hist["luma"] = np.histogram(lum, bins=64, range=(0, 1))[0].tolist()
    h, w = lum.shape
    cols = 128
    wave = np.zeros((64, cols), np.float32)
    for x in range(cols):
        x0, x1 = int(x * w / cols), max(int(x * w / cols) + 1, int((x + 1) * w / cols))
        col = lum[:, x0:x1].mean(axis=1)
        idx = np.clip((col * 63).astype(np.int32), 0, 63)
        np.add.at(wave, (idx, x), 1)
    if wave.max() > 0:
        wave = wave / wave.max()
    # vectorscope: downsample chroma
    step = max(1, min(h, w) // 96)
    small = rgb[::step, ::step]
    u = (small[..., 2] - (0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]) * 0)
    cb = small[..., 2] - (0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2])
    cr = small[..., 0] - (0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2])
    return {"hist": hist, "waveform": wave.tolist(), "cb": cb.ravel().tolist(), "cr": cr.ravel().tolist()}


def render_scope_image(img: Image.Image, kind: str = "histogram", size: Tuple[int, int] = (320, 180)) -> Image.Image:
    """Draw a scope as a stand-alone PNG (used by the GUI and the MCP tool)."""
    w, h = size
    out = Image.new("RGB", (w, h), (12, 14, 18))
    px = out.load()
    data = scope_data(img)
    if kind == "histogram":
        colors = {"r": (255, 90, 90), "g": (90, 255, 130), "b": (110, 150, 255)}
        for cname, col in colors.items():
            vals = data["hist"][cname]
            mx = max(1, max(vals))
            for x in range(w):
                v = vals[min(len(vals) - 1, int(x * len(vals) / w))]
                bar = int(v / mx * (h - 4))
                for y in range(h - 2, h - 2 - bar, -1):
                    r, g, b = px[x, y]
                    px[x, y] = (min(255, r + col[0] // 2), min(255, g + col[1] // 2), min(255, b + col[2] // 2))
    elif kind == "waveform":
        wave = np.array(data["waveform"], np.float32)
        for x in range(w):
            col = wave[:, min(wave.shape[1] - 1, int(x * wave.shape[1] / w))]
            for y in range(h):
                v = col[min(63, int((1 - y / h) * 63))]
                if v > 0.02:
                    g = min(255, int(40 + v * 215))
                    px[x, y] = (g, min(255, g + 20), g)
    else:  # vectorscope
        cb = np.array(data["cb"], np.float32)
        cr = np.array(data["cr"], np.float32)
        cx, cy, rad = w // 2, h // 2, min(w, h) // 2 - 4
        for i in range(0, len(cb), max(1, len(cb) // 4000)):
            x = int(cx + cb[i] * rad * 2.2)
            y = int(cy - cr[i] * rad * 2.2)
            if 0 <= x < w and 0 <= y < h:
                px[x, y] = (min(255, px[x, y][0] + 90), min(255, px[x, y][1] + 90), min(255, px[x, y][2] + 90))
        for ang, col in ((0, (255, 90, 90)), (60, (90, 255, 130)), (120, (110, 150, 255))):
            a = math.radians(ang)
            for r in range(rad):
                x, y = int(cx + math.cos(a) * r), int(cy - math.sin(a) * r)
                if 0 <= x < w and 0 <= y < h:
                    px[x, y] = col
    return out
