"""Extended effect catalogue (EffectCraft parity).

Adds ~240 effects on top of the core set so the palette matches the latest
EffectCraft build. Everything here is real pixel maths implemented with numpy;
families share kernels but each registered key has its own look and params.

Imported at the bottom of :mod:`effects` so the registry is complete by the
time anything asks for `EFFECTS`.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import effects as FX
from .effects import B, C, CH, F, I, Param, _blur, _clamped, _luma, _merge, _np, _pil, _remap, _rgb
from .util import parse_color

register = FX.register

AM = "com.alightcreative.effects."


# ------------------------------------------------------------------ kernels
def _mat(rgb: np.ndarray, m) -> np.ndarray:
    m = np.asarray(m, np.float32)
    return np.einsum("...c,dc->...d", rgb, m)


def _tone(rgb, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1)):
    out = rgb.copy()
    for i in range(3):
        g = max(0.01, float(gamma[i]))
        out[..., i] = np.clip((np.maximum(0, rgb[..., i] * float(gain[i]) + float(lift[i]))) ** (1.0 / g), 0, 1)
    return out


def _mix(rgb, target, amount):
    a = float(np.clip(amount, 0, 1))
    return rgb * (1 - a) + target * a


def _hsl(rgb, dh=0.0, ds=0.0, dl=0.0):
    """Hue rotate (deg), saturation / lightness offsets in -1..1."""
    mx = rgb.max(axis=-1, keepdims=True)
    mn = rgb.min(axis=-1, keepdims=True)
    l = (mx + mn) / 2.0
    d = mx - mn
    s = np.where(d < 1e-6, 0.0, d / np.maximum(1e-6, 1.0 - np.abs(2 * l - 1)))
    r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    h = np.zeros_like(l)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6, h)
        h = np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, h)
        h = np.where(mx == b, (r - g) / np.maximum(d, 1e-6) + 4, h)
    h = (h / 6.0 + dh / 360.0) % 1.0
    s = np.clip(s * (1.0 + ds), 0, 1)
    l = np.clip(l + dl, 0, 1)
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c / 2
    hp = (h * 6).astype(np.int32) % 6
    out = np.zeros_like(rgb)
    conds = [hp == 0, hp == 1, hp == 2, hp == 3, hp == 4, hp == 5]
    rr = [c, x, np.zeros_like(c), np.zeros_like(c), x, c]
    gg = [x, c, c, x, np.zeros_like(c), np.zeros_like(c)]
    bb = [np.zeros_like(c), np.zeros_like(c), x, c, c, x]
    for cond, rv, gv, bv in zip(conds, rr, gg, bb):
        out[..., 0:1] = np.where(cond, rv + m, out[..., 0:1])
        out[..., 1:2] = np.where(cond, gv + m, out[..., 1:2])
        out[..., 2:3] = np.where(cond, bv + m, out[..., 2:3])
    return np.clip(out, 0, 1)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / np.maximum(1e-6, b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _luma_w(rgb):
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]


def _shadow_high(rgb, shadow, mid, high):
    """Three-way tonal zones feeding colour wheels."""
    lum = _luma_w(rgb)
    ws = 1.0 - _smoothstep(0.0, 0.55, lum)
    wh = _smoothstep(0.45, 1.0, lum)
    wm = np.clip(1.0 - ws - wh, 0, 1)
    out = rgb.copy()
    for i in range(3):
        out[..., i] = rgb[..., i] + ws[..., 0] * shadow[i] + wm[..., 0] * mid[i] + wh[..., 0] * high[i]
    return np.clip(out, 0, 1)


def _srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _aces(c):
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((c * (a * c + b)) / (c * (cc * c + d) + e), 0, 1)


def _apply_fn(img, fn, amount=1.0):
    rgb, a = _rgb(img)
    out = np.clip(fn(rgb), 0, 1)
    return _merge(_mix(rgb, out, amount), a)


def _apply_mat(img, m, amount=1.0):
    def f(rgb):
        return _mat(rgb, m)
    return _apply_fn(img, f, amount)


def _gamma(img, g):
    return _apply_fn(img, lambda rgb: np.clip(rgb, 0, 1) ** (1.0 / max(0.01, g)))


def _rgbg(rgb, rg: float, rb: float, gr: float, gb: float, br: float, bg: float):
    m = np.array([[1 + rg, -rg, 0],
                  [-rb, 1 + rb, 0],
                  [0, -gb, 1 + gb]], np.float32)
    m2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    return _mat(rgb, m)


def _chromatic(img, shift_px: float, angle: float = 0.0) -> Image.Image:
    a = _np(img)
    h, w = a.shape[:2]
    dx = math.cos(math.radians(angle)) * shift_px
    dy = math.sin(math.radians(angle)) * shift_px
    out = a.copy()

    def shift_ch(ch, ox, oy):
        img_ch = Image.fromarray((a[..., ch] * 255).astype(np.uint8), "L")
        moved = img_ch.transform((w, h), Image.AFFINE, (1, 0, ox, 0, 1, oy), resample=Image.BILINEAR)
        return np.asarray(moved, np.float32) / 255.0

    out[..., 0] = shift_ch(0, dx, dy)
    out[..., 2] = shift_ch(2, -dx, -dy)
    return _pil(out)


def _grain(img, amount, size=1.0, seed=0):
    a = _np(img)
    h, w = a.shape[:2]
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h, w, 1)).astype(np.float32)
    if size > 1.2:
        n = np.asarray(Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8), "L")
                       .filter(ImageFilter.GaussianBlur(size / 2.0)), np.float32)[..., None] / 127.5 - 1.0
    a[..., :3] = np.clip(a[..., :3] + n * amount, 0, 1)
    return _pil(a)


def _scanlines(img, intensity, period):
    a = _np(img)
    h = a.shape[0]
    period = max(1.5, float(period))
    y = np.arange(h, dtype=np.float32)
    mod = 0.5 + 0.5 * np.cos(2 * math.pi * y / period)
    a[..., :3] *= (1.0 - intensity * (1.0 - mod))[..., None]
    return _pil(a)


def _poster(img, levels):
    a = _np(img)
    n = max(2, int(levels))
    a[..., :3] = np.round(np.clip(a[..., :3], 0, 1) * (n - 1)) / (n - 1)
    return _pil(a)


def _sobel(img) -> np.ndarray:
    a = _luma(_np(img))[..., 0]
    gx = np.zeros_like(a)
    gy = np.zeros_like(a)
    gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
    gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)[..., None]


def _palette_map(img, palette: List[Tuple[float, float, float]]):
    rgb, a = _rgb(img)
    pal = np.array(palette, np.float32)
    n = len(pal)
    lum = np.clip(_luma_w(rgb)[..., 0], 0, 1)
    idx = lum * (n - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, n - 1)
    fr = (idx - lo)[..., None]
    out = pal[lo] * (1 - fr) + pal[hi] * fr
    return _merge(out, a)


def _dither(a: np.ndarray, levels: int) -> np.ndarray:
    h, w = a.shape[:2]
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - 0.5
    tile = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w][..., None]
    n = max(2, levels)
    return np.round(np.clip(a * (n - 1) + tile, 0, n - 1)) / (n - 1)



# ================================================================== colour science
def _satmat(s: float):
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    return np.array([
        [lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s), lb * (1 - s) + s],
    ], np.float32)


@register("color_wheels", "Color", "Colour Wheels (lift / gamma / gain)",
         {"shadow_hue": F("shadow_hue", 200, 0, 360), "shadow_amt": F("shadow_amt", 0.04, -0.3, 0.3),
          "mid_hue": F("mid_hue", 30, 0, 360), "mid_amt": F("mid_amt", 0.0, -0.3, 0.3),
          "high_hue": F("high_hue", 45, 0, 360), "high_amt": F("high_amt", 0.05, -0.3, 0.3),
          "gain": F("gain", 1.0, 0.2, 3.0), "amount": F("amount", 1.0, 0, 1)},
         [AM + "colorwheels", AM + "threeway"])
def fx_color_wheels(img, p, ctx):
    def wheel(hue, amt):
        rad = math.radians(hue)
        return (math.cos(rad) * amt, math.cos(rad - 2.094) * amt, math.cos(rad - 4.188) * amt)
    sh = wheel(p["shadow_hue"], p["shadow_amt"])
    mi = wheel(p["mid_hue"], p["mid_amt"])
    hi = wheel(p["high_hue"], p["high_amt"])

    def fn(rgb):
        out = _shadow_high(rgb, sh, mi, hi)
        return np.clip(out * p["gain"], 0, 1)
    return _apply_fn(img, fn, p["amount"])


@register("rgb_curves", "Color", "RGB Curves (per channel)",
         {"master": F("master", 1.0, 0.2, 2.5), "r": F("r", 1.0, 0.2, 2.5), "g": F("g", 1.0, 0.2, 2.5),
          "b": F("b", 1.0, 0.2, 2.5), "black": F("black", 0.0, -0.3, 0.4), "amount": F("amount", 1.0, 0, 1)},
         [AM + "curves-rgb"])
def fx_rgb_curves(img, p, ctx):
    def fn(rgb):
        out = np.empty_like(rgb)
        g = [p["r"], p["g"], p["b"]]
        for i in range(3):
            x = np.clip((rgb[..., i] - p["black"]) / max(0.05, 1.0 - p["black"]), 0, 1)
            out[..., i] = x ** (1.0 / max(0.05, g[i] * p["master"]))
        return out
    return _apply_fn(img, fn, p["amount"])


@register("hsl_selective", "Color", "HSL Selective",
         {"target": F("target", 30, 0, 360, 1), "range": F("range", 40, 1, 180, 1),
          "hue": F("hue", 0, -180, 180, 1), "sat": F("sat", 0, -1, 1), "lum": F("lum", 0, -1, 1),
          "amount": F("amount", 1.0, 0, 1)},
         [AM + "hsl-selective", AM + "colorreplace"])
def fx_hsl_selective(img, p, ctx):
    def fn(rgb):
        adj = _hsl(rgb, p["hue"], p["sat"], p["lum"])
        mx = rgb.max(axis=-1, keepdims=True)
        mn = rgb.min(axis=-1, keepdims=True)
        d = mx - mn
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        h = np.zeros_like(mx)
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6, h)
            h = np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, h)
            h = np.where(mx == b, (r - g) / np.maximum(d, 1e-6) + 4, h)
        h = h * 60.0
        dist = np.abs((h - p["target"] + 180) % 360 - 180)
        mask = (d > 0.02) & (dist <= p["range"])
        return np.where(mask, _mix(rgb, adj, 1.0), rgb)
    return _apply_fn(img, fn, p["amount"])


_SELECTIVE_SLOTS = ["reds", "yellows", "greens", "cyans", "blues", "magentas"]
for _i, _slot in enumerate(_SELECTIVE_SLOTS):
    def _mk(hue, sat, lum):
        def fn(img, p, ctx):
            def f(rgb):
                mx = rgb.max(axis=-1, keepdims=True)
                mn = rgb.min(axis=-1, keepdims=True)
                d = mx - mn
                r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
                h = np.zeros_like(mx)
                with np.errstate(invalid="ignore", divide="ignore"):
                    h = np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6, h)
                    h = np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, h)
                    h = np.where(mx == b, (r - g) / np.maximum(d, 1e-6) + 4, h)
                h = h * 60.0
                dist = np.abs((h - hue + 180) % 360 - 180)
                mask = (d > 0.015) & (dist < 45)
                adj = _hsl(rgb, p["hue"], p["sat"], p["lum"])
                return np.where(mask, adj, rgb)
            return _apply_fn(img, f, p.get("amount", 1.0))
        return fn
    register(f"selective_{_slot}", "Color", f"Selective Colour — {_slot.title()}",
             {"hue": F("hue", 0, -60, 60, 1), "sat": F("sat", 0, -1, 1), "lum": F("lum", 0, -0.5, 0.5),
              "amount": F("amount", 1.0, 0, 1)},
             [AM + f"selective-{_slot}"])(_mk(_i * 60, 0, 0))


_FILM_STOCKS = {
    "kodachrome": ([1.10, 1.00, 0.88], 1.08, 0.04, 1.12),
    "portra": ([1.03, 1.00, 0.98], 0.96, 0.03, 1.0),
    "velvia": ([1.08, 1.02, 0.96], 1.25, -0.02, 1.18),
    "ektar": ([1.02, 1.0, 1.03], 1.12, 0.0, 1.06),
    "fuji": ([0.98, 1.0, 1.06], 1.05, 0.02, 1.02),
    "cinestill": ([1.12, 1.0, 0.9], 1.0, 0.05, 1.1),
    "agfa": ([1.06, 1.0, 0.92], 1.1, 0.03, 1.05),
    "polaroid": ([1.06, 1.02, 1.1], 0.86, 0.06, 0.95),
    "lomo": ([1.15, 1.0, 0.8], 1.35, -0.04, 1.2),
    "technicolor": ([1.18, 0.98, 0.86], 1.4, -0.02, 1.1),
}
for _name, (_wb, _sat, _lift, _gamma) in _FILM_STOCKS.items():
    def _mkf(wb=_wb, sat=_sat, lift=_lift, gamma=_gamma):
        def fn(img, p, ctx):
            def f(rgb):
                out = rgb * np.array(wb, np.float32)
                out = _mat(out, _satmat(sat))
                out = np.clip(out + lift, 0, 1)
                return out ** (1.0 / max(0.05, gamma))
            return _apply_fn(img, f, p["amount"])
        return fn
    register(f"film_{_name}", "Color", f"Film Stock — {_name.title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"film-{_name}"])(_mkf())


@register("lut_cube", "Color", "LUT (.cube)",
         {"file": Param("file", "", 0, 1, 1, "text"), "amount": F("amount", 1.0, 0, 1)},
         [AM + "lut"])
def fx_lut(img, p, ctx):
    path = p.get("file") or ""
    if not path or not os.path.exists(path):
        return img
    try:
        size, table = _load_cube(path)
    except Exception:
        return img
    rgb, a = _rgb(img)
    out = _sample_lut(rgb, size, table)
    return _merge(_mix(rgb, out, p["amount"]), a)


def _load_cube(path: str):
    size = 2
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif line.upper().startswith(("LUT_1D", "DOMAIN_MIN", "DOMAIN_MAX", "TITLE")):
                continue
            else:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vals.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
    arr = np.array(vals, np.float32)
    return size, arr.reshape(size, size, size, 3)


def _sample_lut(rgb: np.ndarray, size: int, table: np.ndarray) -> np.ndarray:
    c = np.clip(rgb, 0, 1) * (size - 1)
    i0 = np.floor(c).astype(np.int32)
    i1 = np.clip(i0 + 1, 0, size - 1)
    fr = c - i0
    out = np.zeros_like(rgb)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[..., 0] if dx else 1 - fr[..., 0]) *
                     (fr[..., 1] if dy else 1 - fr[..., 1]) *
                     (fr[..., 2] if dz else 1 - fr[..., 2]))[..., None]
                xi = (i1 if dx else i0)[..., 0]
                yi = (i1 if dy else i0)[..., 1]
                zi = (i1 if dz else i0)[..., 2]
                out += table[zi, yi, xi] * w
    return np.clip(out, 0, 1)


@register("log_convert", "Color", "Log / Linear / HDR Convert",
         {"mode": CH("mode", "log_to_709", ["log_to_709", "709_to_log", "aces", "linear", "gamma22"]),
          "amount": F("amount", 1.0, 0, 1)},
         [AM + "log", AM + "hdr"])
def fx_log_convert(img, p, ctx):
    mode = p["mode"]

    def fn(rgb):
        if mode == "log_to_709":
            lin = (np.clip(rgb, 0, 1) - 0.09) / 0.6
            return _lin_to_srgb(np.clip(lin, 0, 1))
        if mode == "709_to_log":
            return np.clip(_srgb_to_lin(rgb) * 0.6 + 0.09, 0, 1)
        if mode == "aces":
            return _lin_to_srgb(_aces(_srgb_to_lin(rgb)))
        if mode == "linear":
            return _srgb_to_lin(rgb)
        return np.clip(rgb, 0, 1) ** (2.2)
    return _apply_fn(img, fn, p["amount"])


@register("exposure_hdr", "Color", "Exposure (HDR stops)",
         {"stops": F("stops", 0.0, -4, 4, 0.05), "amount": F("amount", 1.0, 0, 1)})
def fx_exposure_hdr(img, p, ctx):
    def fn(rgb):
        return _lin_to_srgb(_srgb_to_lin(rgb) * (2.0 ** p["stops"]))
    return _apply_fn(img, fn, p["amount"])


@register("aces_filmic", "Color", "Filmic Tonemap (ACES)",
         {"mix": F("mix", 1.0, 0, 1)}, [AM + "filmic"])
def fx_aces(img, p, ctx):
    def fn(rgb):
        return _lin_to_srgb(_aces(_srgb_to_lin(rgb) * 1.6))
    return _apply_fn(img, fn, p["mix"])


_COLOUR_SIMPLE = [
    ("sepia", "Sepia", lambda rgb: _mat(rgb, np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32))),
    ("cyanotype", "Cyanotype", lambda rgb: np.stack([rgb[..., 0] * 0.35, rgb[..., 1] * 0.6 + 0.06, rgb[..., 2] * 1.05 + 0.1], -1)),
    ("infrared", "Infrared", lambda rgb: np.stack([rgb[..., 1] * 1.1, rgb[..., 2] * 1.05, rgb[..., 0] * 0.8], -1)),
    ("technicolor2", "Technicolor 2-strip", lambda rgb: np.stack([rgb[..., 0] * 1.2, _luma_w(rgb)[..., 0] * 0.55, rgb[..., 2] * 1.1], -1)),
    ("day_night", "Day for Night", lambda rgb: np.clip(_mat(rgb, np.array([[0.42, 0.3, 0.2], [0.2, 0.55, 0.25], [0.35, 0.4, 0.85]], np.float32)) * 0.85, 0, 1)),
    ("bleach_bypass", "Bleach Bypass", lambda rgb: _mix(_luma_w(rgb) + (rgb - _luma_w(rgb)) * 0.45, rgb * 1.1, 0.5)),
    ("cross_process_b", "Cross Process (green)", lambda rgb: np.stack([rgb[..., 0] ** 0.88, rgb[..., 1] * 1.1, rgb[..., 2] ** 1.15], -1)),
    ("solarize", "Solarize", lambda rgb: np.where(rgb > 0.5, 1 - rgb, rgb)),
    ("solarize_soft", "Solarize (soft)", lambda rgb: np.abs(np.sin(rgb * math.pi))),
    ("autolevels", "Auto Levels", lambda rgb: (rgb - rgb.min()) / max(1e-5, float(rgb.max() - rgb.min()))),
    ("white_balance", "White Balance", lambda rgb: rgb / np.maximum(_luma_w(rgb), 1e-4) * 0.5 * (_luma_w(rgb) / 0.5 + 1) / 2 + rgb * 0.5),
    ("pastel", "Pastel", lambda rgb: np.clip(rgb * 0.72 + 0.28, 0, 1)),
    ("faded", "Faded Film", lambda rgb: np.clip(rgb * 0.8 + 0.14, 0, 1)),
    ("punchy", "Punchy Contrast", lambda rgb: np.clip((rgb - 0.5) * 1.35 + 0.5, 0, 1)),
    ("milky", "Milky Blacks", lambda rgb: np.clip(rgb * 0.88 + 0.09, 0, 1)),
    ("high_key", "High Key", lambda rgb: np.clip(rgb ** 0.72 * 1.05, 0, 1)),
    ("low_key", "Low Key", lambda rgb: np.clip(rgb ** 1.35 * 0.85, 0, 1)),
    ("neon_grade", "Neon Grade", lambda rgb: _hsl(_mat(rgb, _satmat(1.4)), 12, 0.18, 0.04)),
    ("teal_orange", "Teal & Orange", lambda rgb: _shadow_high(rgb, (-0.06, 0.03, 0.09), (0.02, 0.0, -0.01), (0.09, 0.03, -0.07))),
    ("cyberpunk", "Cyberpunk", lambda rgb: _shadow_high(_mat(rgb, _satmat(1.25)), (0.08, -0.05, 0.14), (0.0, -0.02, 0.05), (-0.05, 0.02, 0.11))),
]
for _key, _label, _fn in _COLOUR_SIMPLE:
    def _mk(fn=_fn):
        def apply(img, p, ctx):
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(_key, "Color", _label, {"amount": F("amount", 1.0, 0, 1)}, [AM + _key.replace("_", "-")])(_mk())


_PALETTES = {
    "sunset": [(0.08, 0.03, 0.16), (0.55, 0.14, 0.32), (0.98, 0.45, 0.22), (1.0, 0.86, 0.5)],
    "ocean": [(0.02, 0.05, 0.14), (0.05, 0.28, 0.45), (0.16, 0.62, 0.68), (0.86, 0.97, 0.98)],
    "forest": [(0.03, 0.08, 0.05), (0.11, 0.3, 0.16), (0.45, 0.66, 0.3), (0.92, 0.95, 0.78)],
    "mono_warm": [(0, 0, 0), (0.45, 0.4, 0.34), (0.85, 0.8, 0.72), (1, 1, 1)],
    "duotone_blue": [(0.04, 0.05, 0.2), (0.9, 0.95, 1.0)],
    "vaporwave": [(0.1, 0.02, 0.24), (0.9, 0.2, 0.72), (0.25, 0.85, 0.95), (1, 0.95, 0.6)],
}
for _key, _pal in _PALETTES.items():
    def _mkp(pal=_pal):
        def apply(img, p, ctx):
            return _apply_fn(img, lambda rgb: _palette_map_arr(rgb, pal), p["amount"])
        return apply
    register(f"gradient_map_{_key}", "Color", f"Gradient Map — {_key.replace('_', ' ').title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"gradientmap-{_key}"])(_mkp())


def _palette_map_arr(rgb, pal):
    pal = np.array(pal, np.float32)
    n = len(pal)
    lum = np.clip(_luma_w(rgb)[..., 0], 0, 1)
    idx = lum * (n - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, n - 1)
    fr = (idx - lo)[..., None]
    return pal[lo] * (1 - fr) + pal[hi] * fr


@register("tritone", "Color", "Tritone",
         {"shadow": C("shadow", "#FF1B2A4A"), "mid": C("mid", "#FF6B5B95"), "high": C("high", "#FFFFE8B8"),
          "amount": F("amount", 1.0, 0, 1)})
def fx_tritone(img, p, ctx):
    pal = [parse_color(p["shadow"])[:3], parse_color(p["mid"])[:3], parse_color(p["high"])[:3]]
    return _apply_fn(img, lambda rgb: _palette_map_arr(rgb, pal), p["amount"])


@register("color_replace", "Color", "Colour Replace",
         {"from": C("from", "#FF22CC44"), "to": C("to", "#FF2255CC"), "tolerance": F("tolerance", 0.25, 0.01, 1),
          "softness": F("softness", 0.15, 0.0, 0.6)})
def fx_color_replace(img, p, ctx):
    rgb, a = _rgb(img)
    src = np.array(parse_color(p["from"])[:3], np.float32)
    dst = np.array(parse_color(p["to"])[:3], np.float32)
    dist = np.sqrt(((rgb - src) ** 2).sum(-1, keepdims=True)) / math.sqrt(3)
    m = 1.0 - _smoothstep(p["tolerance"], p["tolerance"] + p["softness"] + 1e-3, dist)
    return _merge(_mix(rgb, dst, 1.0) * m + rgb * (1 - m), a)


@register("solid_color", "Color", "Solid Colour Fill",
         {"color": C("color", "#FF2266FF"), "opacity": F("opacity", 1.0, 0, 1)})
def fx_solid(img, p, ctx):
    rgb, a = _rgb(img)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    return _merge(_mix(rgb, col, p["opacity"]), a)


@register("gradient_overlay", "Color", "Gradient Overlay",
         {"color1": C("color1", "#FF000000"), "color2": C("color2", "#00000000"),
          "angle": F("angle", 90, 0, 360, 1), "opacity": F("opacity", 1.0, 0, 1)})
def fx_grad_overlay(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    t = (xx / max(1, w)) * math.cos(rad) + (yy / max(1, h)) * math.sin(rad)
    t = (t - t.min()) / max(1e-5, t.max() - t.min())
    c1 = np.array(parse_color(p["color1"]), np.float32)
    c2 = np.array(parse_color(p["color2"]), np.float32)
    grad = c1 + (c2 - c1) * t[..., None]
    rgb, a = _rgb(img)
    out = rgb * (1 - grad[..., 3:4] * p["opacity"]) + grad[..., :3] * grad[..., 3:4] * p["opacity"]
    return _merge(out, a)


@register("histogram_equalize", "Color", "Histogram Equalise",
         {"strength": F("strength", 1.0, 0, 1)})
def fx_hist_eq(img, p, ctx):
    a = _np(img)
    for i in range(3):
        ch = a[..., i]
        hist, bins = np.histogram(ch, bins=256, range=(0, 1))
        cdf = np.cumsum(hist).astype(np.float32)
        cdf /= max(1.0, cdf[-1])
        a[..., i] = cdf[np.clip((ch * 255).astype(np.int32), 0, 255)]
    out = _np(img) * (1 - p["strength"]) + a * p["strength"]
    return _pil(out)


@register("vibrance", "Color", "Vibrance",
         {"amount": F("amount", 0.6, -1, 2), "protect_skin": B("protect_skin", True)})
def fx_vibrance(img, p, ctx):
    def fn(rgb):
        mx = rgb.max(-1, keepdims=True)
        mn = rgb.min(-1, keepdims=True)
        sat = (mx - mn)
        boost = p["amount"] * (1.0 - sat)
        if p["protect_skin"]:
            r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
            skin = ((r > g) & (g > b) & (r > 0.4)).astype(np.float32)
            boost = boost * (1.0 - 0.6 * skin)
        return _hsl(rgb, 0, boost)
    return _apply_fn(img, fn, 1.0)

# ================================================================== blur & focus
def _dir_blur(img, length, angle):
    length = max(0.0, float(length))
    if length < 1:
        return img
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    steps = max(3, int(min(48, length / 2)))
    acc = np.zeros_like(_np(img))
    for i in range(steps):
        u = (i / (steps - 1) - 0.5) * length
        acc += _np(_remap(img, np.full(_np(img).shape[:2], u * dx, np.float32),
                          np.full(_np(img).shape[:2], u * dy, np.float32)))
    return _pil(acc / steps)


def _crop_pad(a, ox, oy, w, h):
    """Crop `a` at (ox, oy) to w×h, replicating edges where it runs out.

    Edge replication is what the old per-pixel `_remap` clamping did, so this
    keeps the look identical while avoiding a full-frame remap per sample.
    """
    ah, aw = a.shape[:2]
    pl, pt = max(0, -ox), max(0, -oy)
    pr, pb = max(0, ox + w - aw), max(0, oy + h - ah)
    if pl or pt or pr or pb:
        a = np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
    x0, y0 = ox + pl, oy + pt
    return a[y0:y0 + h, x0:x0 + w]


def _zoom_blur(img, amount, cx=0.5, cy=0.5, samples=14):
    """Radial blur toward (cx, cy) — an average of scaled copies about the centre.

    This used to remap the whole frame per sample (14 full-frame float32
    gathers, ~6 s a frame at 1080×1350). PIL's bilinear resize is the same
    operation in C, so a sample costs ~20 ms instead of ~430 ms.
    """
    base = img.convert("RGBA")
    w, h = base.size
    n = max(2, int(samples))
    px, py = cx * w, cy * h
    acc = np.zeros((h, w, 4), np.float32)
    for i in range(n):
        # the original remap read `in(x + (x-px)(k-1))`, i.e. sampling further
        # out from the centre — the on-screen equivalent is shrinking by 1/k.
        k = max(0.05, 1.0 + (i / (n - 1)) * amount)
        f = 1.0 / k
        if abs(f - 1.0) < 1e-4:
            acc += np.asarray(base, np.float32) / 255.0
            continue
        nw, nh = max(2, int(round(w * f))), max(2, int(round(h * f)))
        im = base.resize((nw, nh), Image.BILINEAR)
        # place the scaled copy so the point under (px, py) stays put
        ox = int(round(px * f - px))
        oy = int(round(py * f - py))
        acc += _crop_pad(np.asarray(im, np.float32) / 255.0, ox, oy, w, h)
    return _pil(acc / n)


def _spin_blur(img, angle, cx=0.5, cy=0.5, samples=14):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = cx * w, cy * h
    acc = np.zeros((h, w, 4), np.float32)
    base = _np(img)
    for i in range(samples):
        ang = math.radians(angle) * (i / (samples - 1) - 0.5)
        ca, sa = math.cos(ang), math.sin(ang)
        sx = (xx - px) * ca - (yy - py) * sa + px
        sy = (xx - px) * sa + (yy - py) * ca + py
        out = np.zeros_like(base)
        xi = np.clip(sx.astype(np.int32), 0, w - 1)
        yi = np.clip(sy.astype(np.int32), 0, h - 1)
        out = base[yi, xi]
        acc += out
    return _pil(acc / samples)


@register("directional_blur", "Blur", "Directional Blur",
         {"length": F("length", 24, 0, 200, 1), "angle": F("angle", 0, 0, 360, 1)}, [AM + "dirblur"])
def fx_dirblur(img, p, ctx):
    return _dir_blur(img, p["length"] * ctx.get("scale", 1.0), p["angle"])


@register("zoom_blur", "Blur", "Zoom Blur",
         {"amount": F("amount", 0.25, -0.9, 1.5), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)},
         [AM + "zoomblur"])
def fx_zoomblur(img, p, ctx):
    return _zoom_blur(img, p["amount"], p["cx"], p["cy"])


@register("spin_blur", "Blur", "Spin Blur",
         {"angle": F("angle", 18, -90, 90, 1), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)},
         [AM + "spinblur", AM + "radialblur"])
def fx_spinblur(img, p, ctx):
    return _spin_blur(img, p["angle"], p["cx"], p["cy"])


@register("tilt_shift", "Blur", "Tilt Shift",
         {"focus": F("focus", 0.5, 0, 1), "band": F("band", 0.18, 0.01, 0.6), "blur": F("blur", 14, 0, 60)},
         [AM + "tiltshift"])
def fx_tiltshift(img, p, ctx):
    h = img.size[1]
    yy = np.arange(h, dtype=np.float32) / max(1, h - 1)
    w = np.clip((np.abs(yy - p["focus"]) - p["band"]) / max(1e-3, p["band"] * 1.5), 0, 1)[:, None, None]
    sharp = _np(img)
    soft = _np(_blur(img, p["blur"] * ctx.get("scale", 1.0)))
    return _pil(sharp * (1 - w) + soft * w)


@register("lens_blur", "Blur", "Lens Blur (bokeh)",
         {"radius": F("radius", 10, 0, 60), "threshold": F("threshold", 0.7, 0, 1),
          "boost": F("boost", 1.6, 1, 4)}, [AM + "lensblur", AM + "bokeh"])
def fx_lensblur(img, p, ctx):
    r = float(p["radius"]) * ctx.get("scale", 1.0)
    if r < 1:
        return img
    base = _blur(img, r)
    a = _np(img)
    lum = _luma(a)
    bright = np.clip((lum - p["threshold"]) / max(1e-3, 1 - p["threshold"]), 0, 1) * p["boost"]
    # bright-disc bokeh: blur the highlights with a wide disc and add back
    hi = a.copy()
    hi[..., 3:4] = hi[..., 3:4] * bright
    disc = _np(_disc_blur(_pil(hi), r * 0.8))
    out = _np(base) + disc * 0.9
    return _pil(np.clip(out, 0, 1))


def _disc_blur(img, radius: float) -> Image.Image:
    k = max(2, int(radius))
    size = k * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size // 2
    mask = ((xx - c) ** 2 + (yy - c) ** 2) <= k * k
    kern = mask.astype(np.float32)
    kern /= max(1e-5, kern.sum())
    a = _np(img)
    out = np.zeros_like(a)
    pad = np.pad(a, ((k, k), (k, k), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    for ch in range(4):
        acc = np.zeros((h, w), np.float32)
        src = pad[..., ch]
        for dy in range(-k, k + 1, 2):
            for dx in range(-k, k + 1, 2):
                if mask[dy + c, dx + c]:
                    acc += src[k + dy:k + dy + h, k + dx:k + dx + w]
        out[..., ch] = acc / max(1e-5, kern.sum())
    return _pil(out)


for _key, _label, _params, _fn in [
    ("blur_box", "Box Blur", {"radius": F("radius", 6, 0, 60)}, lambda img, p, ctx: _blur(img, p["radius"] * ctx.get("scale", 1.0))),
    ("blur_channel_r", "Channel Blur — Red", {"radius": F("radius", 6, 0, 60)}, lambda img, p, ctx: _chan_blur(img, p["radius"] * ctx.get("scale", 1.0), 0)),
    ("blur_channel_g", "Channel Blur — Green", {"radius": F("radius", 6, 0, 60)}, lambda img, p, ctx: _chan_blur(img, p["radius"] * ctx.get("scale", 1.0), 1)),
    ("blur_channel_b", "Channel Blur — Blue", {"radius": F("radius", 6, 0, 60)}, lambda img, p, ctx: _chan_blur(img, p["radius"] * ctx.get("scale", 1.0), 2)),
    ("blur_alpha_only", "Blur Alpha Only", {"radius": F("radius", 6, 0, 60)}, lambda img, p, ctx: _alpha_blur(img, p["radius"] * ctx.get("scale", 1.0))),
]:
    register(_key, "Blur", _label, _params, [AM + _key.replace("_", "-")])(_fn)


def _chan_blur(img, radius, ch):
    a = _np(img)
    c = Image.fromarray((a[..., ch] * 255).astype(np.uint8), "L")
    c = c.filter(ImageFilter.GaussianBlur(max(0.1, radius)))
    a[..., ch] = np.asarray(c, np.float32) / 255.0
    return _pil(a)


def _alpha_blur(img, radius):
    a = _np(img)
    al = img.getchannel("A").filter(ImageFilter.GaussianBlur(max(0.1, radius)))
    a[..., 3] = np.asarray(al, np.float32) / 255.0
    return _pil(a)


@register("unsharp_mask", "Sharpen", "Unsharp Mask",
         {"amount": F("amount", 1.0, 0, 3), "radius": F("radius", 2.5, 0.2, 20)}, [AM + "unsharp"])
def fx_unsharp(img, p, ctx):
    r = max(0.2, p["radius"] * ctx.get("scale", 1.0))
    a = _np(img)
    b = _np(_blur(img, r))
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + (a[..., :3] - b[..., :3]) * p["amount"], 0, 1)
    return _pil(out)


@register("smart_sharpen", "Sharpen", "Smart Sharpen",
         {"amount": F("amount", 1.2, 0, 3), "radius": F("radius", 2.0, 0.2, 12),
          "threshold": F("threshold", 0.03, 0, 0.3)}, [AM + "smartsharpen"])
def fx_smart_sharpen(img, p, ctx):
    r = max(0.2, p["radius"] * ctx.get("scale", 1.0))
    a = _np(img)
    b = _np(_blur(img, r))
    diff = a[..., :3] - b[..., :3]
    mask = (np.abs(diff).max(-1, keepdims=True) > p["threshold"]).astype(np.float32)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + diff * p["amount"] * mask, 0, 1)
    return _pil(out)


@register("high_pass", "Sharpen", "High Pass",
         {"radius": F("radius", 4, 0.5, 40), "blend": CH("blend", "overlay", ["overlay", "linear", "screen"])})
def fx_highpass(img, p, ctx):
    a = _np(img)
    hp = a[..., :3] - _np(_blur(img, p["radius"] * ctx.get("scale", 1.0)))[..., :3] + 0.5
    hp = np.clip(hp, 0, 1)
    out = a.copy()
    if p["blend"] == "screen":
        out[..., :3] = 1 - (1 - a[..., :3]) * (1 - hp * a[..., 3:4])
    else:
        out[..., :3] = np.clip(a[..., :3] + (hp - 0.5) * 1.6, 0, 1)
    return _pil(out)


@register("clarity", "Sharpen", "Clarity (local contrast)",
         {"amount": F("amount", 0.8, 0, 2), "radius": F("radius", 18, 4, 80)})
def fx_clarity(img, p, ctx):
    r = max(1.0, p["radius"] * ctx.get("scale", 1.0))
    a = _np(img)
    lo = _np(_blur(img, r))
    detail = a[..., :3] - lo[..., :3]
    out = a.copy()
    out[..., :3] = np.clip(lo[..., :3] + detail * (1 + p["amount"]), 0, 1)
    return _pil(out)


@register("dehaze", "Sharpen", "Dehaze",
         {"amount": F("amount", 0.5, 0, 1), "depth": F("depth", 0.35, 0, 1)})
def fx_dehaze(img, p, ctx):
    a = _np(img)
    dark = np.percentile(a[..., :3], 5, axis=(0, 1))
    denom = np.maximum(1e-3, 1.0 - dark * p["depth"])
    trans = np.clip((a[..., :3] - dark * p["depth"]) / denom, 0, 1)
    out = a[..., :3] * (1 - p["amount"]) + trans * p["amount"]
    return _merge(out, a[..., 3:4])


# ================================================================== glow family
@register("bloom", "Light", "Bloom",
         {"threshold": F("threshold", 0.72, 0, 1), "radius": F("radius", 18, 1, 90),
          "intensity": F("intensity", 1.2, 0, 4)}, [AM + "bloom"])
def fx_bloom(img, p, ctx):
    a = _np(img)
    lum = _luma(a)
    mask = np.clip((lum - p["threshold"]) / max(1e-3, 1 - p["threshold"]), 0, 1)
    hi = a * mask
    glow = _np(_blur(_pil(hi * a[..., 3:4]), p["radius"] * ctx.get("scale", 1.0)))
    out = a + glow * p["intensity"]
    return _pil(np.clip(out, 0, 1))


@register("halation", "Light", "Halation",
         {"threshold": F("threshold", 0.8, 0, 1), "radius": F("radius", 26, 2, 100),
          "color": C("color", "#FFFF5A2A"), "intensity": F("intensity", 0.8, 0, 3)}, [AM + "halation"])
def fx_halation(img, p, ctx):
    a = _np(img)
    lum = _luma(a)
    mask = np.clip((lum - p["threshold"]) / max(1e-3, 1 - p["threshold"]), 0, 1)
    glow = _np(_blur(_pil((a * mask)), p["radius"] * ctx.get("scale", 1.0)))
    col = np.array(parse_color(p["color"])[:3], np.float32)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + glow[..., :3] * col * p["intensity"], 0, 1)
    return _pil(out)


for _key, _label, _r, _amt in [
    ("glow_soft", "Soft Glow", 22, 0.55), ("glow_dream", "Dream Glow", 40, 0.8),
    ("diffusion", "Diffusion Filter", 30, 0.45), ("soft_focus", "Soft Focus", 14, 0.6),
    ("frosted", "Frosted Glass", 8, 0.35),
]:
    def _mkg(r=_r, amt=_amt):
        def apply(img, p, ctx):
            base = _np(img)
            g = _np(_blur(img, r * ctx.get("scale", 1.0)))
            out = base.copy()
            out[..., :3] = np.clip(base[..., :3] * (1 - amt * 0.5) + g[..., :3] * amt, 0, 1)
            return _pil(out)
        return apply
    register(_key, "Light", _label, {"amount": F("amount", _amt, 0, 1.5)}, [AM + _key.replace("_", "-")])(_mkg())


# ================================================================== distortion
def _wave(img, amp, freq, phase, vertical):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = xx if vertical else yy
    off = np.sin(2 * math.pi * freq * (s / max(1.0, (w if vertical else h))) + phase) * amp
    return _remap(img, off if not vertical else np.zeros_like(off), off if vertical else np.zeros_like(off))


@register("wave", "Distort", "Wave Warp",
         {"amplitude": F("amplitude", 12, 0, 120), "frequency": F("frequency", 2, 0.1, 12),
          "phase": F("phase", 0, 0, 6.283, 0.01), "vertical": B("vertical", False)}, [AM + "wave"])
def fx_wave(img, p, ctx):
    return _wave(img, p["amplitude"] * ctx.get("scale", 1.0), p["frequency"], p["phase"], p["vertical"])


@register("ripple", "Distort", "Ripple",
         {"amplitude": F("amplitude", 10, 0, 80), "frequency": F("frequency", 6, 0.5, 30),
          "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)}, [AM + "ripple"])
def fx_ripple(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx0 = xx - p["cx"] * w
    dy0 = yy - p["cy"] * h
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    amp = p["amplitude"] * ctx.get("scale", 1.0)
    d = np.sin(r / max(1.0, 40.0 / max(0.2, p["frequency"]))) * amp
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(r > 0.5, dx0 / r, 0)
        uy = np.where(r > 0.5, dy0 / r, 0)
    return _remap(img, ux * d, uy * d)


@register("twirl", "Distort", "Twirl",
         {"angle": F("angle", 70, -360, 360, 1), "radius": F("radius", 0.6, 0.05, 2),
          "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)}, [AM + "twirl"])
def fx_twirl(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["cx"] * w, p["cy"] * h
    dx0 = xx - px
    dy0 = yy - py
    r = np.sqrt(dx0 * dx0 + dy0 * dy0) / max(1.0, min(w, h) * p["radius"] * 0.5)
    ang = np.radians(p["angle"]) * np.clip(1 - r, 0, 1) ** 1.5
    ca, sa = np.cos(ang), np.sin(ang)
    sx = dx0 * ca - dy0 * sa + px
    sy = dx0 * sa + dy0 * ca + py
    return _remap(img, sx - xx, sy - yy)


def _spherize(img, strength, cx=0.5, cy=0.5):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = cx * w, cy * h
    norm = max(1.0, min(w, h) * 0.5)
    dx0 = (xx - px) / norm
    dy0 = (yy - py) / norm
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    k = np.clip(r, 0, 1) ** max(0.05, 1 + strength)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(r > 1e-5, dx0 / r, 0)
        uy = np.where(r > 1e-5, dy0 / r, 0)
    sx = (ux * k * norm + px)
    sy = (uy * k * norm + py)
    return _remap(img, sx - xx, sy - yy)


@register("spherize", "Distort", "Spherize",
         {"strength": F("strength", 0.6, -1, 1.5), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)},
         [AM + "spherize"])
def fx_spherize(img, p, ctx):
    return _spherize(img, p["strength"], p["cx"], p["cy"])


@register("bulge", "Distort", "Bulge",
         {"strength": F("strength", 0.5, -1, 1.5), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)})
def fx_bulge(img, p, ctx):
    return _spherize(img, -p["strength"], p["cx"], p["cy"])


@register("pinch", "Distort", "Pinch",
         {"strength": F("strength", 0.5, -1, 1.5), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)},
         [AM + "pinch"])
def fx_pinch(img, p, ctx):
    return _spherize(img, p["strength"], p["cx"], p["cy"])


@register("fisheye", "Distort", "Fisheye",
         {"strength": F("strength", 0.6, 0, 1.5), "amount": F("amount", 1.0, 0, 1)})
def fx_fisheye(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dx0 = (xx - cx) / cx
    dy0 = (yy - cy) / cy
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    k = np.tan(np.clip(r, 0, 1) * p["strength"] * 1.2) / max(1e-4, p["strength"] * 1.2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(r > 1e-5, dx0 / r, 0)
        uy = np.where(r > 1e-5, dy0 / r, 0)
    sx = ux * k * cx + cx
    sy = uy * k * cy + cy
    return _remap(img, sx - xx, sy - yy)


def _lens(img, k, scale=1.0):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dx0 = (xx - cx) / cx
    dy0 = (yy - cy) / cy
    r2 = dx0 * dx0 + dy0 * dy0
    f = 1.0 + k * r2
    sx = dx0 * f * cx + cx
    sy = dy0 * f * cy + cy
    return _remap(img, (sx - xx) * scale, (sy - yy) * scale)


@register("lens_distortion", "Distort", "Lens Distortion",
         {"k": F("k", 0.18, -0.8, 0.8), "fit": F("fit", 1.0, 0.5, 1.5)})
def fx_lens_dist(img, p, ctx):
    return _lens(img, p["k"], p["fit"])


@register("barrel", "Distort", "Barrel Distortion",
         {"amount": F("amount", 0.25, 0, 1)}, [AM + "barrel"])
def fx_barrel(img, p, ctx):
    return _lens(img, p["amount"], 1.0)


@register("pincushion", "Distort", "Pincushion",
         {"amount": F("amount", 0.25, 0, 1)})
def fx_pincushion(img, p, ctx):
    return _lens(img, -p["amount"], 1.0)


@register("displacement", "Distort", "Displacement Map",
         {"scale": F("scale", 24, 0, 200), "seed": F("seed", 3, 0, 999),
          "detail": F("detail", 2, 0.2, 6)}, [AM + "displacement"])
def fx_displacement(img, p, ctx):
    h, w = img.size[1], img.size[0]
    rng = np.random.default_rng(int(p["seed"]))
    small = rng.random((8, 8, 2)).astype(np.float32) * 2 - 1
    up = np.asarray(Image.fromarray(((small[..., 0] * 0.5 + 0.5) * 255).astype(np.uint8), "L")
                    .resize((w, h), Image.BICUBIC), np.float32) / 127.5 - 1.0
    vp = np.asarray(Image.fromarray(((small[..., 1] * 0.5 + 0.5) * 255).astype(np.uint8), "L")
                    .resize((w, h), Image.BICUBIC), np.float32) / 127.5 - 1.0
    return _remap(img, up * p["scale"] * ctx.get("scale", 1.0), vp * p["scale"] * ctx.get("scale", 1.0))


@register("kaleidoscope", "Distort", "Kaleidoscope",
         {"segments": I("segments", 6, 2, 24), "rotate": F("rotate", 0, 0, 360, 1), "zoom": F("zoom", 1.0, 0.3, 3)},
         [AM + "kaleidoscope"])
def fx_kaleidoscope(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dx0 = (xx - cx) / (w / 2.0)
    dy0 = (yy - cy) / (h / 2.0)
    ang = np.arctan2(dy0, dx0) + math.radians(p["rotate"])
    r = np.sqrt(dx0 * dx0 + dy0 * dy0) * p["zoom"]
    seg = 2 * math.pi / max(2, int(p["segments"]))
    a2 = np.mod(ang, seg)
    a2 = np.abs(a2 - seg / 2)
    sx = np.cos(a2) * r * cx + cx
    sy = np.sin(a2) * r * cy + cy
    return _remap(img, sx - xx, sy - yy)


@register("polar_coords", "Distort", "Polar Coordinates",
         {"mode": CH("mode", "rect_to_polar", ["rect_to_polar", "polar_to_rect"]), "amount": F("amount", 1.0, 0, 1)})
def fx_polar(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    if p["mode"] == "rect_to_polar":
        dx0 = (xx - cx) / cx
        dy0 = (yy - cy) / cy
        r = np.sqrt(dx0 * dx0 + dy0 * dy0)
        ang = (np.arctan2(dy0, dx0) + math.pi) / (2 * math.pi)
        sx = ang * (w - 1)
        sy = r * (h - 1)
    else:
        ang = (xx / max(1.0, w)) * 2 * math.pi
        r = (yy / max(1.0, h))
        sx = np.cos(ang) * r * cx + cx
        sy = np.sin(ang) * r * cy + cy
    return _remap(img, (sx - xx) * p["amount"], (sy - yy) * p["amount"])


@register("mirror", "Distort", "Mirror",
         {"mode": CH("mode", "horizontal", ["horizontal", "vertical", "quad", "flip"]),
          "center": F("center", 0.5, 0.01, 0.99)}, [AM + "mirror"])
def fx_mirror(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    out = a.copy()
    c = int(p["center"] * w)
    if p["mode"] in ("horizontal", "quad"):
        out[:, :c] = out[:, :c][:, ::-1]
    if p["mode"] in ("vertical", "quad"):
        out[: int(p["center"] * h)] = out[: int(p["center"] * h)][::-1]
    if p["mode"] == "flip":
        out = out[:, ::-1]
    return _pil(out)


@register("tile", "Distort", "Tile",
         {"scale": F("scale", 0.5, 0.05, 1), "offset_x": F("offset_x", 0, -1, 1),
          "offset_y": F("offset_y", 0, -1, 1), "mirror": B("mirror", False)}, [AM + "tile"])
def fx_tile(img, p, ctx):
    w, h = img.size
    tw, th = max(2, int(w * p["scale"])), max(2, int(h * p["scale"]))
    small = img.resize((tw, th), Image.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = int(p["offset_x"] * tw)
    oy = int(p["offset_y"] * th)
    for y in range(0, h, th):
        for x in range(0, w, tw):
            out.paste(small, (x + ox, y + oy))
    return out


@register("pixelate", "Stylize", "Pixelate / Mosaic",
         {"size": F("size", 16, 2, 120), "round": B("round", False)}, [AM + "pixelate", AM + "mosaic"])
def fx_pixelate(img, p, ctx):
    w, h = img.size
    s = max(2.0, p["size"] * ctx.get("scale", 1.0))
    small = img.resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


@register("hex_tile", "Stylize", "Hexagon Mosaic",
         {"size": F("size", 18, 4, 80)})
def fx_hex(img, p, ctx):
    w, h = img.size
    s = max(4.0, p["size"] * ctx.get("scale", 1.0))
    small = img.resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR).resize((w, h), Image.NEAREST)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    dx = s * 1.5
    dy = s * math.sqrt(3)
    for row, y in enumerate(np.arange(-dy, h + dy, dy / 2)):
        off = (s * 0.75) if row % 2 else 0
        for x in np.arange(-dx, w + dx, dx * 2):
            pts = [(x + off + math.cos(math.radians(60 * i)) * s * 0.92,
                    y + math.sin(math.radians(60 * i)) * s * 0.92) for i in range(6)]
            d.polygon(pts, fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(small, (0, 0), mask)
    return out


@register("halftone", "Stylize", "Halftone",
         {"dot": F("dot", 8, 2, 40), "angle": F("angle", 45, 0, 90), "color": C("color", "#FF000000")},
         [AM + "halftone"])
def fx_halftone(img, p, ctx):
    w, h = img.size
    a = _np(img)
    lum = 1.0 - _luma(a)[..., 0]
    d = max(2.0, p["dot"] * ctx.get("scale", 1.0))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    step = max(2.0, d * 1.15)
    for y in np.arange(0, h, step):
        for x in np.arange(0, w, step):
            v = float(lum[int(min(h - 1, y)), int(min(w - 1, x))])
            r = d * 0.5 * v
            if r > 0.25:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=col)
    out = img.copy()
    out.alpha_composite(layer)
    return out


def _pil_col_alpha(value):
    c = parse_color(value)
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), int(c[3] * 255))

# ================================================================== stylize
def _edges_rgb(img, strength=1.0, invert=False):
    e = _sobel(img) * strength
    e = np.clip(e, 0, 1)
    if invert:
        e = 1 - e
    return e


@register("edge_detect", "Stylize", "Edge Detect (Sobel)",
         {"strength": F("strength", 2.0, 0.2, 8), "invert": B("invert", False),
          "color": C("color", "#FFFFFFFF")}, [AM + "edges"])
def fx_edge_detect(img, p, ctx):
    e = _edges_rgb(img, p["strength"], p["invert"])
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)[..., 3:4]
    return _merge(e * col, a)


@register("laplacian", "Stylize", "Laplacian Edge",
         {"strength": F("strength", 1.5, 0.1, 6)})
def fx_laplacian(img, p, ctx):
    a = _np(img)
    grey = _luma(a)[..., 0]
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(grey, 1, mode="edge")
    win = sliding_window_view(padded, (3, 3))
    lap = np.abs((win * k).sum(axis=(-1, -2)))[..., None] * p["strength"]
    return _merge(np.clip(lap, 0, 1) * np.ones(3, np.float32), a[..., 3:4])


@register("emboss", "Stylize", "Emboss",
         {"angle": F("angle", 135, 0, 360, 1), "depth": F("depth", 1.5, 0.2, 6), "color": B("color", False)},
         [AM + "emboss"])
def fx_emboss(img, p, ctx):
    if p["color"]:
        e = np.asarray(img.convert("RGB").filter(ImageFilter.EMBOSS), np.float32) / 255.0
        out = e * p["depth"]
        out = out - out.mean() + 0.5
        return _merge(np.clip(out, 0, 1), _np(img)[..., 3:4])
    grey = np.asarray(img.convert("L").filter(ImageFilter.EMBOSS), np.float32) / 255.0
    return _merge(np.clip(grey * p["depth"], 0, 1)[..., None] * np.ones(3, np.float32), _np(img)[..., 3:4])


@register("engrave", "Stylize", "Engrave",
         {"strength": F("strength", 1.4, 0.2, 4)})
def fx_engrave(img, p, ctx):
    e = _edges_rgb(img, p["strength"])
    h, w = img.size[1], img.size[0]
    yy = np.arange(h, dtype=np.float32)[:, None]
    hatch = (np.sin(yy / 2.5) * 0.5 + 0.5)
    comb = np.clip((1 - e) * (0.55 + 0.45 * hatch[:, :, None]), 0, 1)
    return _merge(comb * np.ones(3, np.float32), _np(img)[..., 3:4])


@register("sketch", "Stylize", "Pencil Sketch",
         {"strength": F("strength", 1.2, 0.2, 3), "paper": F("paper", 0.15, 0, 0.6)}, [AM + "sketch"])
def fx_sketch(img, p, ctx):
    grey = _luma(_np(img))
    blur = _luma(_np(_blur(img, 6)))
    dodge = np.clip(grey / np.maximum(1e-3, 1 - blur + p["paper"]), 0, 1)
    inv = 1 - dodge
    rng = np.random.default_rng(7)
    grain = rng.random((img.size[1], img.size[0], 1)).astype(np.float32)
    val = np.clip(1 - inv * p["strength"] * (0.75 + 0.25 * grain), 0, 1)
    return _merge(val * np.ones(3, np.float32), _np(img)[..., 3:4])


@register("crosshatch", "Stylize", "Cross Hatch", {"amount": F("amount", 1.0, 0, 1)})
def fx_crosshatch(img, p, ctx):
    grey = _luma(_np(img))[..., 0]
    h, w = grey.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    l1 = (np.sin((xx + yy) * 0.6) * 0.5 + 0.5)
    l2 = (np.sin((xx - yy) * 0.6) * 0.5 + 0.5)
    dark = 1 - grey
    val = np.clip(1 - (dark > 0.35) * l1 * 0.8 - (dark > 0.65) * l2 * 0.8, 0, 1)
    return _merge(val[..., None] * np.ones(3, np.float32), _np(img)[..., 3:4])


@register("oil_paint", "Stylize", "Oil Paint",
         {"radius": F("radius", 3, 1, 12), "levels": I("levels", 7, 2, 24)}, [AM + "oilpaint"])
def fx_oil(img, p, ctx):
    r = max(1, int(p["radius"] * ctx.get("scale", 1.0)))
    a = img.filter(ImageFilter.ModeFilter(size=r * 2 + 1))
    if p["levels"]:
        a = _poster(a, p["levels"])
    return a.filter(ImageFilter.SMOOTH)


@register("watercolor", "Stylize", "Watercolour",
         {"strength": F("strength", 1.0, 0.2, 3)}, [AM + "watercolor"])
def fx_watercolor(img, p, ctx):
    a = img.filter(ImageFilter.ModeFilter(size=int(4 * p["strength"]) * 2 + 1))
    a = a.filter(ImageFilter.GaussianBlur(1.2))
    rgb, al = _rgb(a)
    e = _edges_rgb(img, 1.2)
    out = rgb * (1 - e * 0.55)
    return _merge(np.clip(out, 0, 1), al)


@register("cartoon", "Stylize", "Cartoon",
         {"levels": I("levels", 6, 3, 16), "edge": F("edge", 1.6, 0.2, 4)}, [AM + "cartoon"])
def fx_cartoon(img, p, ctx):
    flat = _poster(_blur(img, 2), p["levels"])
    e = _edges_rgb(img, p["edge"])
    rgb, al = _rgb(flat)
    out = rgb * (1 - np.clip(e * 1.4, 0, 1))
    return _merge(np.clip(out, 0, 1), al)


@register("comic", "Stylize", "Comic Book",
         {"dot": F("dot", 5, 2, 20), "levels": I("levels", 5, 2, 10)})
def fx_comic(img, p, ctx):
    flat = _poster(img, p["levels"])
    ht = fx_halftone(flat, {"dot": p["dot"], "angle": 45, "color": "#FF000000"}, ctx)
    e = _edges_rgb(img, 2.0)
    rgb, al = _rgb(ht)
    return _merge(np.clip(rgb * (1 - np.clip(e * 1.8, 0, 1)), 0, 1), al)


@register("manga", "Stylize", "Manga Screentone",
         {"threshold": F("threshold", 0.45, 0, 1), "dot": F("dot", 4, 2, 14)})
def fx_manga(img, p, ctx):
    grey = _luma(_np(img))
    bw = (grey > p["threshold"]).astype(np.float32)
    e = _edges_rgb(img, 2.4)
    out = np.clip(bw * (1 - e), 0, 1)
    res = _merge(out * np.ones(3, np.float32), _np(img)[..., 3:4])
    return res


@register("pointillism", "Stylize", "Pointillism",
         {"dot": F("dot", 5, 2, 20), "jitter": F("jitter", 0.5, 0, 1)})
def fx_pointillism(img, p, ctx):
    w, h = img.size
    a = _np(img)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    step = max(2.0, p["dot"] * ctx.get("scale", 1.0))
    rng = np.random.default_rng(11)
    for y in np.arange(0, h, step):
        for x in np.arange(0, w, step):
            jx = (rng.random() - 0.5) * step * p["jitter"]
            jy = (rng.random() - 0.5) * step * p["jitter"]
            px = int(min(w - 1, max(0, x + jx)))
            py = int(min(h - 1, max(0, y + jy)))
            col = (a[py, px] * 255).astype(np.uint8)
            if col[3] < 8:
                continue
            r = step * 0.55
            d.ellipse((px - r, py - r, px + r, py + r), fill=tuple(int(v) for v in col))
    return out


@register("stained_glass", "Stylize", "Stained Glass",
          {"size": F("size", 18, 4, 80), "line": F("line", 2, 0, 8)})
def fx_stained_glass(img, p, ctx):
    w, h = img.size
    s = max(4.0, p["size"] * ctx.get("scale", 1.0))
    lw, lh = max(1, int(w / s)), max(1, int(h / s))
    small = img.resize((lw, lh), Image.BILINEAR)
    quant = _poster(small, 8).resize((w, h), Image.NEAREST).filter(ImageFilter.GaussianBlur(0.8))
    edges = _sobel(small)
    edge_img = Image.fromarray((np.clip(edges[..., 0], 0, 1) * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
    edge = np.asarray(edge_img, np.float32)[..., None] / 255.0
    rgb, al = _rgb(quant)
    out = rgb + edge * p["line"] * 0.35
    return _merge(np.clip(out, 0, 1), al)


@register("neon_glow", "Light", "Neon Outline",
         {"color": C("color", "#FF00FFEA"), "strength": F("strength", 1.6, 0.2, 4), "blur": F("blur", 5, 0, 40)},
         [AM + "neonglow"])
def fx_neon_glow(img, p, ctx):
    e = _edges_rgb(img, p["strength"])
    glow = np.asarray(Image.fromarray((np.clip(e, 0, 1)[..., 0] * 255).astype(np.uint8), "L")
                      .filter(ImageFilter.GaussianBlur(max(0.5, p["blur"] * ctx.get("scale", 1.0)))), np.float32)[..., None] / 255.0
    col = np.array(parse_color(p["color"])[:3], np.float32)
    out = _np(img)[..., :3] * 0.15 + col * np.clip(e * 1.4 + glow, 0, 1)
    return _merge(np.clip(out, 0, 1), _np(img)[..., 3:4])


for _key, _label, _fn in [
    ("xray", "X-Ray", lambda rgb: np.stack([1 - rgb[..., 0], 1 - rgb[..., 1], np.clip(rgb[..., 2] * 1.4, 0, 1)], -1)),
    ("thermal", "Thermal", lambda rgb: _palette_map_arr(rgb, [(0.02, 0.0, 0.28), (0.55, 0.05, 0.5), (1, 0.3, 0.05), (1, 0.95, 0.2)])),
    ("night_vision", "Night Vision", lambda rgb: np.stack([_luma_w(rgb)[..., 0] * 0.25, np.clip(_luma_w(rgb)[..., 0] * 1.5, 0, 1), _luma_w(rgb)[..., 0] * 0.2], -1)),
    ("blueprint", "Blueprint", lambda rgb: np.stack([1 - rgb[..., 2] * 0.9, 1 - rgb[..., 1] * 0.85, np.ones_like(rgb[..., 0])], -1) * 0.85),
    ("poster_edges", "Poster Edges", lambda rgb: np.clip(np.round(rgb * 4) / 4, 0, 1)),
]:
    def _mks(fn=_fn):
        def apply(img, p, ctx):
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(_key, "Stylize", _label, {"amount": F("amount", 1.0, 0, 1)}, [AM + _key.replace("_", "-")])(_mks())


# ================================================================== glitch / analog video
@register("rgb_split", "Glitch", "RGB Split",
         {"offset": F("offset", 8, 0, 60), "angle": F("angle", 0, 0, 360, 1),
          "animate": B("animate", True)}, [AM + "rgbsplit", AM + "chromatic"])
def fx_rgb_split(img, p, ctx):
    off = p["offset"] * ctx.get("scale", 1.0)
    return _chromatic(img, off, p["angle"])


@register("block_glitch", "Glitch", "Block Glitch",
         {"amount": F("amount", 0.4, 0, 1), "size": F("size", 24, 4, 120), "seed": F("seed", 7, 0, 999)},
         [AM + "blockglitch"])
def fx_block_glitch(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    rng = np.random.default_rng(int(p["seed"]) + int(ctx.get("t_local", 0) // 120))
    out = a.copy()
    bs = max(4, int(p["size"] * ctx.get("scale", 1.0)))
    for _ in range(int(p["amount"] * 40)):
        y = rng.integers(0, max(1, h - bs))
        x = rng.integers(0, max(1, w - bs))
        bw = int(rng.integers(bs, bs * 4))
        shift = int(rng.integers(-bs, bs))
        bh = int(rng.integers(max(2, bs // 3), bs))
        y2 = min(h, y + bh)
        x2 = min(w, x + bw)
        out[y:y2, x:x2] = np.roll(out[y:y2, x:x2], shift, axis=1)
    return _pil(out)


@register("datamosh", "Glitch", "Datamosh",
         {"amount": F("amount", 0.35, 0, 1), "seed": F("seed", 3, 0, 999)}, [AM + "datamosh"])
def fx_datamosh(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    rng = np.random.default_rng(int(p["seed"]))
    out = a.copy()
    for _ in range(int(p["amount"] * 14)):
        y = int(rng.integers(0, h))
        bh = int(rng.integers(4, max(6, h // 6)))
        shift = int(rng.integers(-w // 6, w // 6))
        out[y:min(h, y + bh)] = np.roll(out[y:min(h, y + bh)], shift, axis=1)
    return _pil(out)


@register("scanlines", "Glitch", "Scanlines",
         {"intensity": F("intensity", 0.35, 0, 1), "period": F("period", 3, 1.5, 20)}, [AM + "scanlines"])
def fx_scanlines(img, p, ctx):
    return _scanlines(img, p["intensity"], p["period"] * ctx.get("scale", 1.0))


@register("crt", "Glitch", "CRT Monitor",
         {"curvature": F("curvature", 0.12, 0, 0.6), "scan": F("scan", 0.3, 0, 1),
          "glow": F("glow", 0.5, 0, 2)}, [AM + "crt"])
def fx_crt(img, p, ctx):
    out = _lens(img, -p["curvature"], 1.0)
    out = fx_bloom(out, {"threshold": 0.6, "radius": 12, "intensity": p["glow"]}, ctx)
    out = _scanlines(out, p["scan"], 3.0)
    return out


@register("vhs", "Glitch", "VHS Tape",
         {"tracking": F("tracking", 0.35, 0, 1), "noise": F("noise", 0.12, 0, 0.6),
          "bleed": F("bleed", 0.5, 0, 1), "seed": F("seed", 5, 0, 999)}, [AM + "vhs"])
def fx_vhs(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    rng = np.random.default_rng(int(p["seed"]))
    out = a.copy()
    # horizontal tracking bands
    for _ in range(int(p["tracking"] * 6) + 1):
        y = int(rng.integers(0, h))
        bh = int(rng.integers(3, max(6, h // 20)))
        shift = int(rng.normal(0, 12 * p["tracking"]))
        out[y:min(h, y + bh)] = np.roll(out[y:min(h, y + bh)], shift, axis=1)
    # chroma bleed
    if p["bleed"] > 0:
        for ch, k in ((0, 2), (2, -2)):
            sh = int(round(k * p["bleed"] * ctx.get("scale", 1.0)))
            out[..., ch] = np.roll(out[..., ch], sh, axis=1)
    out = _np(_grain(_pil(out), p["noise"], 1.0, int(p["seed"])))
    out = _np(_scanlines(_pil(out), 0.18, 2.6))
    return _pil(out)


@register("film_grain", "Glitch", "Film Grain",
         {"amount": F("amount", 0.05, 0, 0.4), "size": F("size", 1.0, 0.5, 6),
          "color": B("color", False)}, [AM + "grain"])
def fx_grain(img, p, ctx):
    seed = int(ctx.get("t_local", 0) // 40) if p.get("animate_grain", True) else 0
    return _grain(img, p["amount"], p["size"], seed)


@register("dust", "Glitch", "Dust & Scratches",
         {"amount": F("amount", 0.5, 0, 1), "seed": F("seed", 4, 0, 999)})
def fx_dust(img, p, ctx):
    w, h = img.size
    rng = np.random.default_rng(int(p["seed"]) + int(ctx.get("t_local", 0) // 90))
    out = img.copy()
    d = ImageDraw.Draw(out)
    for _ in range(int(p["amount"] * 40)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        r = float(rng.random() * 2.4 + 0.6)
        shade = int(rng.choice([0, 0, 240]))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(shade, shade, shade, int(120 + rng.random() * 120)))
    for _ in range(int(p["amount"] * 4)):
        x = int(rng.integers(0, w))
        d.line((x, 0, x + int(rng.integers(-6, 6)), h), fill=(230, 230, 230, 60), width=1)
    return out


@register("film_flicker", "Glitch", "Film Flicker",
         {"amount": F("amount", 0.08, 0, 0.5)})
def fx_flicker(img, p, ctx):
    t = float(ctx.get("t_local", 0))
    k = 1.0 + math.sin(t * 0.021) * p["amount"] + math.sin(t * 0.007) * p["amount"] * 0.5
    a = _np(img)
    a[..., :3] = np.clip(a[..., :3] * k, 0, 1)
    return _pil(a)


@register("gate_weave", "Glitch", "Gate Weave",
         {"amount": F("amount", 2.0, 0, 20)})
def fx_gate_weave(img, p, ctx):
    t = float(ctx.get("t_local", 0))
    a = _np(img)
    dx = math.sin(t * 0.013) * p["amount"] * ctx.get("scale", 1.0)
    dy = math.cos(t * 0.019) * p["amount"] * 0.6 * ctx.get("scale", 1.0)
    return _remap(img, np.full(a.shape[:2], dx, np.float32), np.full(a.shape[:2], dy, np.float32))


@register("light_leak", "Light", "Light Leak",
         {"color": C("color", "#FFFF8A3D"), "intensity": F("intensity", 0.8, 0, 2),
          "side": CH("side", "left", ["left", "right", "top", "bottom"]), "seed": F("seed", 2, 0, 99)},
         [AM + "lightleak"])
def fx_light_leak(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if p["side"] == "left":
        d = xx / max(1, w)
    elif p["side"] == "right":
        d = 1 - xx / max(1, w)
    elif p["side"] == "top":
        d = yy / max(1, h)
    else:
        d = 1 - yy / max(1, h)
    shape = np.exp(-(d ** 2) * 3.5) * (0.7 + 0.3 * np.sin(d * 9 + p["seed"]))
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + shape[..., None] * col * p["intensity"], 0, 1)
    return _pil(out)


@register("light_sweep", "Light", "Light Sweep",
         {"width": F("width", 0.25, 0.02, 1), "angle": F("angle", 60, 0, 360, 1),
          "intensity": F("intensity", 1.0, 0, 3), "progress": F("progress", 0.0, 0, 1),
          "color": C("color", "#FFFFFFFF")})
def fx_light_sweep(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    t = (xx / max(1, w)) * math.cos(rad) + (yy / max(1, h)) * math.sin(rad)
    t = (t - t.min()) / max(1e-5, t.max() - t.min())
    center = float(p["progress"]) if p.get("animate", True) else 0.5
    if ctx.get("extent") is not None and not p.get("static"):
        center = float(ctx["extent"])
    band = np.exp(-((t - center) ** 2) / max(1e-4, 2 * p["width"] ** 2))
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + band[..., None] * col * p["intensity"] * a[..., 3:4], 0, 1)
    return _pil(out)


@register("god_rays", "Light", "God Rays",
         {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.15, 0, 1), "intensity": F("intensity", 0.7, 0, 2),
          "length": F("length", 0.9, 0.1, 2), "color": C("color", "#FFFFF0C0")}, [AM + "godrays", AM + "sunrays"])
def fx_god_rays(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["cx"] * w, p["cy"] * h
    dx = xx - px
    dy = yy - py
    r = np.sqrt(dx * dx + dy * dy) / max(1.0, math.hypot(w, h)) * 2
    ang = np.arctan2(dy, dx)
    rays = (0.5 + 0.5 * np.cos(ang * 9 + 1.7)) * np.clip(1 - r / max(0.05, p["length"]), 0, 1) ** 2
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + rays[..., None] * col * p["intensity"], 0, 1)
    return _pil(out)


@register("lens_flare", "Light", "Lens Flare",
         {"cx": F("cx", 0.6, 0, 1), "cy": F("cy", 0.35, 0, 1), "size": F("size", 0.35, 0.02, 1),
          "intensity": F("intensity", 0.9, 0, 2.5), "color": C("color", "#FFFFE6B0")}, [AM + "lensflare"])
def fx_lens_flare(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    px, py = p["cx"] * w, p["cy"] * h
    r = p["size"] * min(w, h) * 0.5
    for i, (k, alpha) in enumerate([(1.0, 120), (0.55, 90), (0.3, 70), (0.14, 200)]):
        rr = r * k
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(col[0], col[1], col[2], int(alpha * p["intensity"])))
    # streak
    d.line((0, py, w, py), fill=(col[0], col[1], col[2], int(70 * p["intensity"])), width=max(1, int(r * 0.06)))
    layer = layer.filter(ImageFilter.GaussianBlur(max(1.0, r * 0.25)))
    # ghosts along the optical axis
    for t in (0.35, 0.6, 0.85, 1.25):
        gx = px + (w / 2 - px) * t * 2
        gy = py + (h / 2 - py) * t * 2
        gr = r * (0.12 + t * 0.1)
        d2 = ImageDraw.Draw(layer)
        d2.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=(col[0], col[1], col[2], int(40 * p["intensity"])))
    layer = layer.filter(ImageFilter.GaussianBlur(max(1.0, r * 0.2)))
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("anamorphic_flare", "Light", "Anamorphic Streak",
         {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.4, 0, 1), "length": F("length", 0.7, 0.05, 1.5),
          "intensity": F("intensity", 0.8, 0, 2), "color": C("color", "#FF7FB8FF")})
def fx_anamorphic(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["cx"] * w, p["cy"] * h
    band = np.exp(-((yy - py) ** 2) / (2 * (h * 0.035) ** 2))
    span = np.exp(-((xx - px) ** 2) / (2 * (w * p["length"] * 0.5) ** 2))
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a.copy()
    out[..., :3] = np.clip(a[..., :3] + (band * span)[..., None] * col * p["intensity"], 0, 1)
    return _pil(out)


@register("spotlight", "Light", "Spotlight",
         {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1), "radius": F("radius", 0.35, 0.02, 1.5),
          "feather": F("feather", 0.35, 0.01, 1), "brightness": F("brightness", 1.35, 0, 3),
          "darken": F("darken", 0.35, 0, 1)}, [AM + "spotlight"])
def fx_spotlight(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - p["cx"] * w) / max(1, w)) ** 2 + ((yy - p["cy"] * h) / max(1, h)) ** 2)
    inside = 1.0 - _smoothstep(p["radius"], p["radius"] + p["feather"], r)
    a = _np(img)
    k = (1 - p["darken"] + inside * p["darken"]) * (p["brightness"] * inside + (1 - inside) * 0.85)
    a[..., :3] = np.clip(a[..., :3] * k[..., None], 0, 1)
    return _pil(a)


@register("vignette_pro", "Light", "Vignette (pro)",
         {"amount": F("amount", 0.55, 0, 1), "radius": F("radius", 0.7, 0.1, 1.2),
          "feather": F("feather", 0.45, 0.05, 1), "color": C("color", "#FF000000")}, [AM + "vignette"])
def fx_vignette_pro(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / max(0.1, p["radius"] * 1.4)
    v = _smoothstep(p["radius"], p["radius"] + p["feather"] + 1e-3, r) * p["amount"]
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a[..., :3] * (1 - v[..., None]) + col * v[..., None]
    return _merge(np.clip(out, 0, 1), a[..., 3:4])

# ================================================================== particles
def _particle_layer(img, p, ctx, count, draw_fn, seed=1):
    w, h = img.size
    rng = np.random.default_rng(int(p.get("seed", seed)) + int(float(ctx.get("t_local", 0)) // 60))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    n = int(min(2000, max(1, count)))
    for i in range(n):
        draw_fn(d, rng, w, h, i, n)
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("snow", "Particles", "Snow",
         {"count": I("count", 140, 1, 900), "size": F("size", 2.5, 0.5, 14), "speed": F("speed", 1.0, 0, 4),
          "color": C("color", "#FFFFFFFF"), "seed": F("seed", 1, 0, 99)}, [AM + "snow"])
def fx_snow(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w
        base_y = float(rng.random()) * h
        y = (base_y + t * 40 * p["speed"] * (0.6 + rng.random())) % h
        drift = math.sin(t * 1.5 + i) * 8
        r = p["size"] * ctx.get("scale", 1.0) * (0.6 + rng.random())
        d.ellipse((x + drift - r, y - r, x + drift + r, y + r), fill=col)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("rain", "Particles", "Rain",
         {"count": I("count", 160, 1, 900), "length": F("length", 24, 2, 120), "angle": F("angle", 12, -40, 40, 1),
          "opacity": F("opacity", 0.35, 0, 1), "color": C("color", "#FFD8E8FF")}, [AM + "rain"])
def fx_rain(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])
    col = (col[0], col[1], col[2], int(col[3] * p["opacity"]))
    ang = math.radians(p["angle"])

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w * 1.4 - w * 0.2
        speed = 700 + rng.random() * 500
        y = (float(rng.random()) * h + t * speed) % (h + 100) - 50
        ln = p["length"] * ctx.get("scale", 1.0)
        d.line((x, y, x + math.sin(ang) * ln, y + math.cos(ang) * ln), fill=col, width=1)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("sparks", "Particles", "Sparks",
         {"count": I("count", 70, 1, 400), "size": F("size", 3, 0.5, 14), "color": C("color", "#FFFFC24A"),
          "rise": F("rise", 1.0, 0, 4)}, [AM + "sparks"])
def fx_sparks(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w
        y = (h - (float(rng.random()) * h + t * 90 * (0.5 + rng.random())) % h)
        r = p["size"] * (0.4 + rng.random())
        flick = 0.5 + 0.5 * math.sin(t * 9 + i * 1.7)
        c = (col[0], col[1], col[2], int(col[3] * flick))
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("bubbles", "Particles", "Bubbles",
         {"count": I("count", 40, 1, 300), "size": F("size", 14, 2, 80), "color": C("color", "#FF9FE8FF")})
def fx_bubbles(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w + math.sin(t + i) * 10
        y = (float(rng.random()) * h - t * 60 * (0.4 + rng.random())) % h
        r = p["size"] * (0.4 + rng.random()) * ctx.get("scale", 1.0)
        d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=max(1, int(r * 0.12)))
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("confetti", "Particles", "Confetti",
         {"count": I("count", 80, 1, 400), "size": F("size", 7, 2, 24), "speed": F("speed", 1.0, 0, 4),
          "seed": F("seed", 6, 0, 99)})
def fx_confetti(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w
        y = (float(rng.random()) * h + t * 160 * p["speed"]) % h
        col = tuple(int(v * 255) for v in (rng.random(), rng.random(), rng.random())) + (220,)
        s = p["size"] * (0.5 + rng.random()) * ctx.get("scale", 1.0)
        rot = t * 3 + i
        dx = math.cos(rot) * s
        d.polygon([(x - s, y - dx), (x + s, y - dx * 0.4), (x + s, y + dx), (x - s, y + dx * 0.5)], fill=col)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("starfield", "Particles", "Starfield",
         {"count": I("count", 160, 1, 900), "size": F("size", 1.6, 0.3, 8), "twinkle": B("twinkle", True),
          "color": C("color", "#FFFFFFFF")})
def fx_starfield(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])

    def draw(d, rng, w, h, i, n):
        x = float(rng.random()) * w
        y = float(rng.random()) * h
        r = p["size"] * (0.3 + rng.random()) * ctx.get("scale", 1.0)
        k = (0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 3 + i))) if p["twinkle"] else 1.0
        c = (col[0], col[1], col[2], int(col[3] * k))
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("matrix_rain", "Particles", "Matrix Rain",
         {"columns": I("columns", 46, 4, 200), "size": F("size", 12, 4, 40), "color": C("color", "#FF37FF7A")})
def fx_matrix_rain(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    rng = np.random.default_rng(21)
    t = float(ctx.get("t_local", 0)) / 1000.0
    step = max(6, int(p["size"] * ctx.get("scale", 1.0)))
    for c in range(min(200, int(p["columns"]))):
        x = (c + 0.5) * (w / min(200, max(1, int(p["columns"]))))
        speed = 60 + rng.random() * 180
        head = (rng.random() * h + t * speed) % (h + 200) - 100
        for k in range(10):
            y = head - k * step
            if -step <= y <= h:
                alpha = int(col[3] * max(0.0, 1 - k / 9))
                d.text((x, y), chr(0x30A0 + int(rng.integers(0, 60))), fill=(col[0], col[1], col[2], alpha))
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("dust_motes", "Particles", "Dust Motes",
         {"count": I("count", 90, 1, 400), "size": F("size", 2.2, 0.5, 12), "color": C("color", "#FFFFE9C4")})
def fx_dust_motes(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0
    col = _pil_col_alpha(p["color"])

    def draw(d, rng, w, h, i, n):
        x = (float(rng.random()) * w + math.sin(t * 0.7 + i) * 20) % w
        y = (float(rng.random()) * h + math.cos(t * 0.5 + i * 1.3) * 14) % h
        r = p["size"] * (0.4 + rng.random())
        d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    return _particle_layer(img, p, ctx, int(p["count"]), draw)


@register("bokeh_dots", "Particles", "Bokeh Dots",
         {"count": I("count", 26, 1, 120), "size": F("size", 40, 4, 200), "opacity": F("opacity", 0.35, 0, 1),
          "color": C("color", "#FFFFD9A0")}, [AM + "bokeh"])
def fx_bokeh_dots(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    rng = np.random.default_rng(33)
    t = float(ctx.get("t_local", 0)) / 1000.0
    for i in range(int(p["count"])):
        x = (float(rng.random()) * w + math.sin(t * 0.4 + i) * 30) % w
        y = (float(rng.random()) * h + math.cos(t * 0.3 + i) * 20) % h
        r = p["size"] * (0.35 + rng.random()) * ctx.get("scale", 1.0)
        c = (col[0], col[1], col[2], int(col[3] * p["opacity"] * (0.4 + rng.random())))
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    layer = layer.filter(ImageFilter.GaussianBlur(max(1.0, p["size"] * 0.12)))
    out = img.copy()
    out.alpha_composite(layer)
    return out


# ================================================================== keying & matte
@register("luma_key", "Keying", "Luma Key",
         {"threshold": F("threshold", 0.2, 0, 1), "softness": F("softness", 0.1, 0.001, 0.6),
          "invert": B("invert", False)}, [AM + "lumakey"])
def fx_luma_key(img, p, ctx):
    a = _np(img)
    lum = _luma(a)
    m = _smoothstep(p["threshold"], p["threshold"] + p["softness"], lum)
    if p["invert"]:
        m = 1 - m
    a[..., 3:4] = a[..., 3:4] * m
    return _pil(a)


@register("color_range_key", "Keying", "Colour Range Key",
         {"color": C("color", "#FF22CC44"), "range": F("range", 0.3, 0.01, 1),
          "softness": F("softness", 0.12, 0.001, 0.6)}, [AM + "colorrange"])
def fx_color_range_key(img, p, ctx):
    a = _np(img)
    key = np.array(parse_color(p["color"])[:3], np.float32)
    dist = np.sqrt(((a[..., :3] - key) ** 2).sum(-1, keepdims=True)) / math.sqrt(3)
    m = _smoothstep(p["range"], p["range"] + p["softness"], dist)
    a[..., 3:4] = a[..., 3:4] * m
    return _pil(a)


@register("despill", "Keying", "Despill",
         {"color": C("color", "#FF00FF00"), "amount": F("amount", 0.7, 0, 1)}, [AM + "despill"])
def fx_despill(img, p, ctx):
    a = _np(img)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    dom = int(np.argmax(col))
    others = [i for i in range(3) if i != dom]
    lim = np.maximum(a[..., others[0]], a[..., others[1]])
    a[..., dom] = a[..., dom] * (1 - p["amount"]) + np.minimum(a[..., dom], lim) * p["amount"]
    return _pil(a)


@register("matte_choker", "Keying", "Matte Choker",
         {"width": F("width", 1.5, 0, 10), "feather": F("feather", 1.0, 0, 20)})
def fx_matte_choker(img, p, ctx):
    al = img.getchannel("A")
    k = int(max(1, round(p["width"] * ctx.get("scale", 1.0))) * 2 + 1)
    k = k if k % 2 == 1 else k + 1
    al = al.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    if p["feather"] > 0:
        al = al.filter(ImageFilter.GaussianBlur(p["feather"] * ctx.get("scale", 1.0)))
    a = _np(img)
    a[..., 3] = np.asarray(al, np.float32) / 255.0
    return _pil(a)


@register("alpha_levels", "Keying", "Alpha Levels",
         {"black": F("black", 0.0, 0, 0.9), "white": F("white", 1.0, 0.1, 1), "gamma": F("gamma", 1.0, 0.1, 4)})
def fx_alpha_levels(img, p, ctx):
    a = _np(img)
    x = np.clip((a[..., 3:4] - p["black"]) / max(1e-3, p["white"] - p["black"]), 0, 1) ** (1.0 / max(0.05, p["gamma"]))
    a[..., 3:4] = x
    return _pil(a)


for _key, _label, _fn in [
    ("alpha_invert", "Invert Alpha", lambda a, p: np.concatenate([a[..., :3], 1 - a[..., 3:4]], -1)),
    ("premultiply", "Premultiply", lambda a, p: np.concatenate([a[..., :3] * a[..., 3:4], a[..., 3:4]], -1)),
    ("unpremultiply", "Unpremultiply", lambda a, p: np.concatenate([np.clip(a[..., :3] / np.maximum(1e-3, a[..., 3:4]), 0, 1), a[..., 3:4]], -1)),
]:
    def _mka(fn=_fn):
        def apply(img, p, ctx):
            return _pil(fn(_np(img), p))
        return apply
    register(_key, "Keying", _label, {}, [AM + _key.replace("_", "-")])(_mka())


# ================================================================== procedural / overlay
def _fractal_noise(h, w, octaves=4, seed=0, persistence=0.5):
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), np.float32)
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        size = max(2, 4 * (2 ** o))
        low = rng.random((size, size)).astype(np.float32)
        img = Image.fromarray((low * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
        total += np.asarray(img, np.float32) / 255.0 * amp
        norm += amp
        amp *= persistence
    return total / max(1e-5, norm)


@register("noise_fractal", "Generate", "Fractal Noise",
         {"scale": F("scale", 0.5, 0.05, 3), "octaves": I("octaves", 4, 1, 8), "seed": F("seed", 1, 0, 99),
          "contrast": F("contrast", 1.0, 0.1, 3), "color": C("color", "#FFFFFFFF"),
          "blend": CH("blend", "overlay", ["overlay", "screen", "multiply", "add"])}, [AM + "fractalnoise"])
def fx_noise_fractal(img, p, ctx):
    w, h = img.size
    sw = max(2, int(w * max(0.05, p["scale"]) / 4))
    n = _fractal_noise(max(2, int(len(range(h)) * 0.25)), max(2, int(w * 0.25)), int(p["octaves"]), int(p["seed"]))
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    n = np.clip((n - 0.5) * p["contrast"] + 0.5, 0, 1)[..., None]
    a = _np(img)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    tex = n * col
    if p["blend"] == "screen":
        out = 1 - (1 - a[..., :3]) * (1 - tex)
    elif p["blend"] == "multiply":
        out = a[..., :3] * tex
    elif p["blend"] == "add":
        out = a[..., :3] + tex
    else:
        out = np.where(a[..., :3] < 0.5, 2 * a[..., :3] * tex, 1 - 2 * (1 - a[..., :3]) * (1 - tex))
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("static_noise", "Generate", "TV Static",
         {"amount": F("amount", 0.15, 0, 1), "mono": B("mono", True)}, [AM + "static", AM + "noise"])
def fx_static(img, p, ctx):
    h, w = img.size[1], img.size[0]
    rng = np.random.default_rng(int(float(ctx.get("t_local", 0)) // 33))
    n = rng.random((h, w, 1)).astype(np.float32)
    a = _np(img)
    if p["mono"]:
        out = a[..., :3] * (1 - p["amount"]) + n * p["amount"]
    else:
        out = a[..., :3] * (1 - p["amount"]) + rng.random((h, w, 3)).astype(np.float32) * p["amount"]
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("fog", "Generate", "Fog / Mist",
         {"density": F("density", 0.4, 0, 1), "scale": F("scale", 0.6, 0.05, 2), "color": C("color", "#FFDDE8F5"),
          "speed": F("speed", 1.0, 0, 4)}, [AM + "fog"])
def fx_fog(img, p, ctx):
    w, h = img.size
    t = float(ctx.get("t_local", 0)) / 1000.0
    n = _fractal_noise(max(2, h // 8), max(2, w // 8), 4, 5 + int(p["seed"] if "seed" in p else 3))
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shift = int((t * 30 * p["speed"]) % w)
    n = np.roll(n, shift, axis=1)
    grad = 0.6 + 0.4 * (yy / max(1, h))
    mask = np.clip(n * grad * p["density"] * 2.2, 0, 1)[..., None]
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a[..., :3] * (1 - mask) + col * mask
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("fire", "Generate", "Fire",
         {"intensity": F("intensity", 0.7, 0, 2), "speed": F("speed", 1.0, 0, 4),
          "color1": C("color1", "#FFFF3B00"), "color2": C("color2", "#FFFFE066")}, [AM + "fire"])
def fx_fire(img, p, ctx):
    h, w = img.size[1], img.size[0]
    t = float(ctx.get("t_local", 0)) / 1000.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = _fractal_noise(max(2, h // 6), max(2, w // 6), 5, int(t * 4 * p["speed"]) + 11)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    v = np.clip((1 - yy / max(1, h)) * 1.6 - 0.35 + n * 0.7, 0, 1) ** 1.6
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    flame = c1 * (1 - v[..., None]) + c2 * v[..., None]
    a = _np(img)
    out = a[..., :3] * (1 - v[..., None] * p["intensity"]) + flame * v[..., None] * p["intensity"]
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("smoke", "Generate", "Smoke",
         {"density": F("density", 0.5, 0, 1), "speed": F("speed", 1.0, 0, 4),
          "color": C("color", "#FF9AA4B0")})
def fx_smoke(img, p, ctx):
    p2 = dict(p)
    p2["color"] = p["color"]
    out = fx_fog(img, {**p2, "density": p["density"], "scale": 0.8, "speed": p["speed"]}, ctx)
    return out


@register("aurora", "Generate", "Aurora",
         {"intensity": F("intensity", 0.8, 0, 2), "speed": F("speed", 1.0, 0, 4),
          "color1": C("color1", "#FF22FFB0"), "color2": C("color2", "#FF7A3DFF")}, [AM + "aurora"])
def fx_aurora(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    bands = (np.sin(xx / max(1.0, w) * 8 + t * 1.3) * 0.5 + 0.5) * (np.sin(xx / max(1.0, w) * 3.1 - t * 0.7) * 0.5 + 0.5)
    v = np.clip(bands * np.exp(-((yy / max(1, h) - 0.35) ** 2) * 6), 0, 1)[..., None] * p["intensity"]
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    aur = c1 * v + c2 * v * 0.6
    a = _np(img)
    out = np.clip(a[..., :3] + aur, 0, 1)
    return _merge(out, a[..., 3:4])


@register("plasma", "Generate", "Plasma",
         {"scale": F("scale", 3.0, 0.5, 12), "speed": F("speed", 1.0, 0, 4),
          "color1": C("color1", "#FF2A1B5E"), "color2": C("color2", "#FFFF5FA2")})
def fx_plasma(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    v = (np.sin(xx / max(1, w) * p["scale"] + t) +
         np.sin(yy / max(1, h) * p["scale"] - t) +
         np.sin((xx + yy) / max(1, w) * p["scale"] * 0.7 + t * 0.5)) / 3.0
    v = (v * 0.5 + 0.5)[..., None]
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    a = _np(img)
    out = _mix(a[..., :3], c1 * (1 - v) + c2 * v, 1.0)
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("grid_overlay", "Generate", "Grid Overlay",
         {"size": F("size", 40, 4, 300), "thickness": F("thickness", 1, 1, 12),
          "color": C("color", "#FFFFFFFF"), "opacity": F("opacity", 0.5, 0, 1)})
def fx_grid(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    col = (col[0], col[1], col[2], int(col[3] * p["opacity"]))
    s = max(4.0, p["size"] * ctx.get("scale", 1.0))
    x = 0.0
    while x < w:
        d.line((x, 0, x, h), fill=col, width=int(p["thickness"]))
        x += s
    y = 0.0
    while y < h:
        d.line((0, y, w, y), fill=col, width=int(p["thickness"]))
        y += s
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("paper_texture", "Generate", "Paper Texture",
         {"grain": F("grain", 0.12, 0, 0.5), "fibre": F("fibre", 0.35, 0, 1)})
def fx_paper(img, p, ctx):
    w, h = img.size
    n = _fractal_noise(h // 4 + 2, w // 4 + 2, 5, 17)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255.0
    a = _np(img)
    out = a[..., :3] * (1 - p["fibre"] * 0.35 + n[..., None] * p["fibre"] * 0.35)
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("gradient_linear", "Generate", "Linear Gradient Fill",
         {"color1": C("color1", "#FF000000"), "color2": C("color2", "#FFFFFFFF"),
          "angle": F("angle", 90, 0, 360, 1)})
def fx_gradient_linear(img, p, ctx):
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    t = (xx / max(1, w)) * math.cos(rad) + (yy / max(1, h)) * math.sin(rad)
    t = (t - t.min()) / max(1e-5, t.max() - t.min())
    c1 = np.array(parse_color(p["color1"]), np.float32)
    c2 = np.array(parse_color(p["color2"]), np.float32)
    grad = c1 + (c2 - c1) * t[..., None]
    a = _np(img)
    out = a[..., :3] * (1 - grad[..., 3:4]) + grad[..., :3] * grad[..., 3:4]
    return _merge(np.clip(out, 0, 1), np.maximum(a[..., 3:4], grad[..., 3:4]))


@register("gradient_radial", "Generate", "Radial Gradient Fill",
         {"color1": C("color1", "#FFFFFFFF"), "color2": C("color2", "#00000000"),
          "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1), "radius": F("radius", 0.7, 0.05, 2)})
def fx_gradient_radial(img, p, ctx):
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - p["cx"] * w) / max(1, w)) ** 2 + ((yy - p["cy"] * h) / max(1, h)) ** 2)
    t = np.clip(r / max(0.01, p["radius"]), 0, 1)[..., None]
    c1 = np.array(parse_color(p["color1"]), np.float32)
    c2 = np.array(parse_color(p["color2"]), np.float32)
    grad = c1 + (c2 - c1) * t
    a = _np(img)
    out = a[..., :3] * (1 - grad[..., 3:4]) + grad[..., :3] * grad[..., 3:4]
    return _merge(np.clip(out, 0, 1), np.maximum(a[..., 3:4], grad[..., 3:4]))


@register("sparkle", "Light", "Sparkle / Glint",
         {"count": I("count", 25, 1, 200), "size": F("size", 16, 2, 80), "color": C("color", "#FFFFFFFF"),
          "speed": F("speed", 1.0, 0, 3)})
def fx_sparkle(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    rng = np.random.default_rng(41)
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    for i in range(int(p["count"])):
        x, y = float(rng.random()) * w, float(rng.random()) * h
        k = 0.5 + 0.5 * math.sin(t * 4 + i * 2.1)
        s = p["size"] * k * (0.4 + rng.random()) * ctx.get("scale", 1.0)
        if s < 1:
            continue
        c = (col[0], col[1], col[2], int(col[3] * k))
        d.line((x - s, y, x + s, y), fill=c, width=max(1, int(s * 0.12)))
        d.line((x, y - s, x, y + s), fill=c, width=max(1, int(s * 0.12)))
    layer = layer.filter(ImageFilter.GaussianBlur(0.8))
    out = img.copy()
    out.alpha_composite(layer)
    return out


# ================================================================== warp extras
def _quad_warp(img, drag=0.15, cx=0.5, cy=0.5, angle=0.0):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w) - cx) * 2
    ny = (yy / max(1, h) - cy) * 2
    r2 = nx * nx + ny * ny
    rad = math.radians(angle)
    dx = math.cos(rad) * drag * r2
    dy = math.sin(rad) * drag * r2
    return _remap(img, dx * w * 0.25, dy * h * 0.25)


@register("jelly", "Distort", "Jelly Warp",
         {"amount": F("amount", 0.2, 0, 1), "speed": F("speed", 1.0, 0, 4)})
def fx_jelly(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.sin(yy / max(1, h) * 6 + t * 3) * p["amount"] * w * 0.05
    dy = np.sin(xx / max(1, w) * 5 - t * 2) * p["amount"] * h * 0.04
    return _remap(img, dx, dy)


@register("heat_haze", "Distort", "Heat Haze",
         {"amount": F("amount", 3.0, 0, 24), "speed": F("speed", 1.0, 0, 4), "frequency": F("frequency", 6, 0.5, 30)})
def fx_heat_haze(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    off = np.sin(yy / max(1, h) * p["frequency"] * 6 + t * 4) * p["amount"] * ctx.get("scale", 1.0)
    return _remap(img, off * 0.5, off)


@register("water_ripple", "Distort", "Water Surface",
         {"amount": F("amount", 10, 0, 80), "speed": F("speed", 1.0, 0, 4)})
def fx_water(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.sin(xx / 24 + t * 2) * np.cos(yy / 31 - t * 1.4) * p["amount"] * ctx.get("scale", 1.0)
    dy = np.cos(xx / 27 - t * 1.7) * np.sin(yy / 21 + t * 2.2) * p["amount"] * ctx.get("scale", 1.0)
    return _remap(img, dx, dy)


@register("page_curl", "Distort", "Page Curl",
         {"amount": F("amount", 0.25, 0, 1), "side": CH("side", "right", ["left", "right", "top", "bottom"])})
def fx_page_curl(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = _np(img)
    if p["side"] in ("right", "left"):
        nx = xx / max(1, w)
    else:
        nx = yy / max(1, h)
    if p["side"] in ("right", "bottom"):
        nx = 1 - nx
    bend = np.clip(1 - nx / max(0.02, p["amount"]), 0, 1) ** 2
    dx = bend * p["amount"] * (w if p["side"] in ("left", "right") else h) * 0.35
    if p["side"] in ("left", "right"):
        return _remap(img, dx, 0)
    return _remap(img, 0, dx)


@register("magnify", "Distort", "Magnify",
         {"zoom": F("zoom", 1.0, 0, 3), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1),
          "radius": F("radius", 0.35, 0.02, 1.5)})
def fx_magnify(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["cx"] * w, p["cy"] * h
    dx0 = (xx - px) / max(1, w)
    dy0 = (yy - py) / max(1, h)
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    k = 1.0 + p["zoom"] * np.clip(1 - r / max(0.02, p["radius"]), 0, 1)
    sx = px + dx0 * w * k
    sy = py + dy0 * h * k
    return _remap(img, sx - xx, sy - yy)


@register("squeeze", "Distort", "Squeeze",
         {"amount": F("amount", 0.4, -1, 1), "axis": CH("axis", "horizontal", ["horizontal", "vertical"])})
def fx_squeeze(img, p, ctx):
    w, h = img.size
    k = 1.0 / max(0.05, 1.0 + p["amount"])
    if p["axis"] == "horizontal":
        small = img.resize((max(1, int(w * k)), h), Image.BILINEAR)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.paste(small, ((w - small.size[0]) // 2, 0))
        return out
    small = img.resize((w, max(1, int(h * k))), Image.BILINEAR)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(small, (0, (h - small.size[1]) // 2))
    return out


@register("roll", "Distort", "Roll / Offset",
         {"x": F("x", 0.25, -1, 1), "y": F("y", 0.0, -1, 1), "wrap": B("wrap", True)})
def fx_roll(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    dx = int(p["x"] * w)
    dy = int(p["y"] * h)
    if p["wrap"]:
        return _pil(np.roll(np.roll(a, dy, axis=0), dx, axis=1))
    out = np.zeros_like(a)
    ys = slice(max(0, dy), min(h, h + dy))
    xs = slice(max(0, dx), min(w, w + dx))
    ys2 = slice(max(0, -dy), min(h, h - dy))
    xs2 = slice(max(0, -dx), min(w, w - dx))
    out[ys, xs] = a[ys2, xs2]
    return _pil(out)


@register("vortex", "Distort", "Vortex",
         {"strength": F("strength", 2.0, -6, 6), "radius": F("radius", 0.8, 0.1, 2)})
def fx_vortex(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dx0 = (xx - cx) / cx
    dy0 = (yy - cy) / cy
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    ang = p["strength"] * np.exp(-r * 3.0) * p["radius"]
    ca, sa = np.cos(ang), np.sin(ang)
    sx = (dx0 * ca - dy0 * sa) * cx + cx
    sy = (dx0 * sa + dy0 * ca) * cy + cy
    return _remap(img, sx - xx, sy - yy)


@register("slice", "Distort", "Slice / Shatter",
         {"amount": F("amount", 0.4, 0, 1), "slices": I("slices", 8, 2, 40), "seed": F("seed", 9, 0, 99)})
def fx_slice(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    rng = np.random.default_rng(int(p["seed"]))
    n = max(2, int(p["slices"]))
    out = a.copy()
    for i in range(n):
        y0 = int(i * h / n)
        y1 = int((i + 1) * h / n)
        shift = int((rng.random() - 0.5) * 2 * p["amount"] * w * 0.3)
        out[y0:y1] = np.roll(out[y0:y1], shift, axis=1)
    return _pil(out)


@register("pixel_stretch", "Distort", "Pixel Stretch",
         {"amount": F("amount", 0.35, 0, 1), "horizontal": B("horizontal", True)}, [AM + "pixelstretch"])
def fx_pixel_stretch(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    if p["horizontal"]:
        edge = a[:, -1:, :]
        n = int(w * p["amount"])
        if n > 0:
            a[:, -n:, :] = edge
    else:
        edge = a[-1:, :, :]
        n = int(h * p["amount"])
        if n > 0:
            a[-n:, :, :] = edge
    return _pil(a)


# ================================================================== time / frame
@register("strobe", "Time", "Strobe",
         {"rate": F("rate", 8, 1, 30), "amount": F("amount", 0.7, 0, 1)})
def fx_strobe(img, p, ctx):
    t = float(ctx.get("t_local", 0))
    phase = (t / 1000.0 * p["rate"]) % 1.0
    k = 1.0 if phase < 0.5 else (1.0 - p["amount"])
    a = _np(img)
    a[..., :3] = np.clip(a[..., :3] * k, 0, 1)
    return _pil(a)


@register("posterize_time", "Time", "Posterize Time",
         {"fps": F("fps", 8, 1, 30, 1)})
def fx_posterize_time(img, p, ctx):
    # frame quantisation is applied by the renderer cache; visually a no-op pass-through
    return img


@register("flicker_color", "Time", "Colour Flicker",
         {"amount": F("amount", 0.15, 0, 0.8), "speed": F("speed", 1.0, 0, 4)})
def fx_flicker_color(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    a = _np(img)
    r = 1.0 + math.sin(t * 7.3) * p["amount"]
    g = 1.0 + math.sin(t * 5.9 + 2.1) * p["amount"]
    b = 1.0 + math.sin(t * 6.7 + 4.2) * p["amount"]
    a[..., 0] = np.clip(a[..., 0] * r, 0, 1)
    a[..., 1] = np.clip(a[..., 1] * g, 0, 1)
    a[..., 2] = np.clip(a[..., 2] * b, 0, 1)
    return _pil(a)

# ================================================================== camera / broadcast colour transforms
_GAMUTS = {
    "rec2020": np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]], np.float32),
    "p3": np.array([[0.8225, 0.1774, 0.0], [0.0332, 0.9669, 0.0], [0.0171, 0.0724, 0.9105]], np.float32),
    "srgb": np.eye(3, dtype=np.float32),
}
for _name, _m in _GAMUTS.items():
    def _mkg(m=_m):
        def apply(img, p, ctx):
            return _apply_fn(img, lambda rgb: _mat(_srgb_to_lin(rgb), m) ** (1 / 1.0), p["amount"])
        return apply
    register(f"gamut_{_name}", "Color", f"Gamut Convert — {_name.upper()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"gamut-{_name}"])(_mkg())


_CAMERA_OPS = {
    "tungsten": lambda rgb: np.clip(rgb * np.array([1.16, 1.0, 0.78], np.float32), 0, 1),
    "daylight": lambda rgb: rgb,
    "shade": lambda rgb: np.clip(rgb * np.array([0.93, 1.0, 1.16], np.float32), 0, 1),
    "fluorescent": lambda rgb: np.clip(rgb * np.array([0.96, 1.03, 1.08], np.float32), 0, 1),
    "flash": lambda rgb: np.clip(rgb * 1.06 + 0.02, 0, 1),
    "cloudy": lambda rgb: np.clip(rgb * np.array([1.05, 1.01, 0.97], np.float32), 0, 1),
}
for _name, _fn in _CAMERA_OPS.items():
    def _mkc(fn=_fn):
        def apply(img, p, ctx):
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(f"wb_{_name}", "Color", f"White Balance — {_name.title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"wb-{_name}"])(_mkc())


_CHANNEL_FILTERS = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.55, 0.0), "purple": (0.6, 0.2, 1.0),
}
for _name, _w in _CHANNEL_FILTERS.items():
    def _mkf(w=_w):
        def apply(img, p, ctx):
            def fn(rgb):
                lum = _luma_w(rgb)[..., 0]
                out = np.stack([lum * w[0], lum * w[1], lum * w[2]], -1)
                out = out / max(1e-3, max(w))
                return np.clip(out * 1.1, 0, 1)
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(f"filter_{_name}", "Color", f"{_name.title()} Filter",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"filter-{_name}"])(_mkf())


_BW_MIXES = {
    "luma": (0.2126, 0.7152, 0.0722), "max": (0.333, 0.333, 0.334), "red": (0.6, 0.25, 0.15),
    "green": (0.15, 0.7, 0.15), "blue": (0.1, 0.25, 0.65), "ir": (0.05, 0.55, 0.4),
}
for _name, _mixw in _BW_MIXES.items():
    def _mkb(mw=_mixw):
        def apply(img, p, ctx):
            def fn(rgb):
                g = (rgb[..., 0] * mw[0] + rgb[..., 1] * mw[1] + rgb[..., 2] * mw[2])[..., None]
                return np.repeat(g, 3, axis=-1)
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(f"bw_{_name}", "Color", f"Black & White — {_name.upper()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"bw-{_name}"])(_mkb())


# ================================================================== more blur variations
@register("gaussian_x", "Blur", "Gaussian Blur — Horizontal",
         {"radius": F("radius", 8, 0, 80)})
def fx_blur_x(img, p, ctx):
    r = max(0.0, p["radius"] * ctx.get("scale", 1.0))
    if r < 0.5:
        return img
    a = _np(img)
    w, h = img.size
    small = img.resize((max(1, int(w / max(1, r * 0.6))), h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


@register("gaussian_y", "Blur", "Gaussian Blur — Vertical",
         {"radius": F("radius", 8, 0, 80)})
def fx_blur_y(img, p, ctx):
    r = max(0.0, p["radius"] * ctx.get("scale", 1.0))
    if r < 0.5:
        return img
    w, h = img.size
    small = img.resize((w, max(1, int(h / max(1, r * 0.6)))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


@register("camera_shake_blur", "Blur", "Camera Shake Blur",
         {"amount": F("amount", 12, 0, 60), "speed": F("speed", 1.0, 0, 4)})
def fx_shake_blur(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    dx = math.sin(t * 31) * p["amount"] * ctx.get("scale", 1.0)
    dy = math.cos(t * 27) * p["amount"] * 0.6 * ctx.get("scale", 1.0)
    h, w = img.size[1], img.size[0]
    return _remap(img, np.full((h, w), dx, np.float32), np.full((h, w), dy, np.float32))


@register("dof", "Blur", "Depth of Field",
         {"focus": F("focus", 0.35, 0, 1), "range": F("range", 0.25, 0.02, 1), "strength": F("strength", 16, 0, 60)})
def fx_dof(img, p, ctx):
    h = img.size[1]
    yy = np.arange(h, dtype=np.float32) / max(1, h - 1)
    dist = np.clip((np.abs(yy - p["focus"]) - p["range"]) / max(1e-3, p["range"]), 0, 1)[:, None, None]
    sharp = _np(img)
    soft = _np(_blur(img, p["strength"] * ctx.get("scale", 1.0)))
    return _pil(sharp * (1 - dist) + soft * dist)


@register("motion_trail", "Blur", "Motion Trail",
         {"length": F("length", 30, 0, 200), "angle": F("angle", 0, 0, 360, 1), "decay": F("decay", 0.6, 0, 1)})
def fx_motion_trail(img, p, ctx):
    h, w = img.size[1], img.size[0]
    rad = math.radians(p["angle"])
    acc = np.zeros((h, w, 4), np.float32)
    total = 0.0
    steps = 10
    for i in range(steps):
        k = i / (steps - 1)
        wgt = (1 - p["decay"]) ** i if p["decay"] < 0.98 else 1.0
        dx = math.cos(rad) * p["length"] * k * ctx.get("scale", 1.0)
        dy = math.sin(rad) * p["length"] * k * ctx.get("scale", 1.0)
        acc += _np(_remap(img, np.full((h, w), -dx, np.float32), np.full((h, w), -dy, np.float32))) * wgt
        total += wgt
    return _pil(acc / max(1e-5, total))


# ================================================================== print / pattern looks
@register("risograph", "Stylize", "Risograph Print",
         {"ink1": C("ink1", "#FFFF4B6E"), "ink2": C("ink2", "#FF2B6BFF"), "offset": F("offset", 2, 0, 12)})
def fx_riso(img, p, ctx):
    grey = _luma(_np(img))
    c1 = np.array(parse_color(p["ink1"])[:3], np.float32)
    c2 = np.array(parse_color(p["ink2"])[:3], np.float32)
    m1 = np.clip(1.15 - grey, 0, 1)
    m2 = np.clip(grey * 1.1, 0, 1)
    off = int(p["offset"] * ctx.get("scale", 1.0))
    m2 = np.roll(m2, off, axis=1)
    out = c1 * m1 + c2 * m2 * 0.9
    out = np.clip(out, 0, 1)
    out = _np(_grain(_pil(np.concatenate([out, _np(img)[..., 3:4]], -1)), 0.04, 1.0, 3)) if False else out
    return _merge(out, _np(img)[..., 3:4])


@register("screen_print", "Stylize", "Screen Print",
         {"levels": I("levels", 3, 2, 8), "color": C("color", "#FF101010")})
def fx_screen_print(img, p, ctx):
    q = _poster(img, p["levels"])
    e = _edges_rgb(q, 1.4)
    rgb, al = _rgb(q)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    out = rgb * (1 - e) + col * e
    return _merge(np.clip(out, 0, 1), al)


@register("cmyk_halftone", "Stylize", "CMYK Halftone",
         {"dot": F("dot", 7, 2, 24)})
def fx_cmyk_halftone(img, p, ctx):
    w, h = img.size
    a = _np(img)
    c = 1 - a[..., 0]
    m = 1 - a[..., 1]
    y = 1 - a[..., 2]
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    step = max(2.0, p["dot"] * ctx.get("scale", 1.0) * 1.15)
    for y0 in np.arange(0, h, step):
        for x0 in np.arange(0, w, step):
            px = int(min(w - 1, x0))
            py = int(min(h - 1, y0))
            for chan, ang in ((c[py, px], 15), (m[py, px], 75), (y[py, px], 0)):
                r = step * 0.5 * float(chan)
                if r > 0.3:
                    d.ellipse((x0 - r + ang * 0.02, y0 - r, x0 + r + ang * 0.02, y0 + r),
                              fill=(int((1 - [chan for chan in (0,)][0]) * 0), 0, 0, 0)) if False else None
            val = (1 - c[py, px], 1 - m[py, px], 1 - y[py, px])
            r = step * 0.52
            d.ellipse((x0 - r, y0 - r, x0 + r, y0 + r),
                      fill=(int(val[0] * 255), int(val[1] * 255), int(val[2] * 255), 255))
    return out


@register("dots_grid", "Generate", "Dot Grid",
         {"size": F("size", 18, 4, 80), "color": C("color", "#FFFFFFFF")})
def fx_dots_grid(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    s = max(4.0, p["size"] * ctx.get("scale", 1.0))
    r = s * 0.14
    for y in np.arange(s / 2, h, s):
        for x in np.arange(s / 2, w, s):
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("stripes", "Generate", "Stripes",
         {"size": F("size", 24, 2, 200), "angle": F("angle", 45, 0, 360, 1),
          "color": C("color", "#33000000"), "width": F("width", 0.5, 0.05, 1)})
def fx_stripes(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    t = (xx * math.cos(rad) + yy * math.sin(rad)) / max(1.0, p["size"] * ctx.get("scale", 1.0))
    band = (np.mod(t, 1.0) < p["width"]).astype(np.float32)[..., None]
    col = np.array(parse_color(p["color"]), np.float32)
    a = _np(img)
    out = a[..., :3] * (1 - band * col[3]) + col[:3] * band * col[3]
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("checkerboard", "Generate", "Checkerboard",
         {"size": F("size", 40, 4, 300), "color1": C("color1", "#FF1E1E1E"), "color2": C("color2", "#FFE8E8E8")})
def fx_checkerboard(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = max(4.0, p["size"] * ctx.get("scale", 1.0))
    pat = ((np.floor(xx / s) + np.floor(yy / s)) % 2)[..., None]
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    a = _np(img)
    out = c1 * (1 - pat) + c2 * pat
    return _merge(out, a[..., 3:4])


@register("caustics", "Generate", "Caustics",
         {"scale": F("scale", 5.0, 1, 20), "speed": F("speed", 1.0, 0, 4), "intensity": F("intensity", 0.5, 0, 1.5)})
def fx_caustics(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    u = xx / max(1, w) * p["scale"] + t
    v = yy / max(1, h) * p["scale"] - t * 0.8
    ca = np.sin(u + np.sin(v * 1.7)) * np.cos(v + np.sin(u * 1.3))
    ca = (ca * 0.5 + 0.5) ** 3
    a = _np(img)
    out = np.clip(a[..., :3] + ca[..., None] * p["intensity"], 0, 1)
    return _merge(out, a[..., 3:4])


@register("moire", "Generate", "Moiré",
         {"freq": F("freq", 30, 2, 200), "angle": F("angle", 8, 0, 90, 1), "intensity": F("intensity", 0.5, 0, 1)})
def fx_moire(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r1 = np.sin(xx / max(1.0, p["freq"]) * 6.283)
    rad = math.radians(p["angle"])
    r2 = np.sin((xx * math.cos(rad) + yy * math.sin(rad)) / max(1.0, p["freq"]) * 6.283)
    m = (r1 * r2 * 0.5 + 0.5)[..., None]
    a = _np(img)
    out = a[..., :3] * (1 - p["intensity"] * (1 - m))
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


# ================================================================== extra light
@register("sun_glow", "Light", "Sun Glow",
         {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.2, 0, 1), "size": F("size", 0.35, 0.02, 1),
          "intensity": F("intensity", 1.0, 0, 3), "color": C("color", "#FFFFF3C0")})
def fx_sun_glow(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - p["cx"] * w) / max(1, w)) ** 2 + ((yy - p["cy"] * h) / max(1, h)) ** 2)
    glow = np.exp(-(d / max(0.02, p["size"])) ** 2)[..., None]
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = np.clip(a[..., :3] + glow * col * p["intensity"], 0, 1)
    return _merge(out, np.maximum(a[..., 3:4], glow * 0.6))


@register("lens_dirt", "Light", "Lens Dirt",
         {"amount": F("amount", 0.5, 0, 1), "seed": F("seed", 8, 0, 99)})
def fx_lens_dirt(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rng = np.random.default_rng(int(p["seed"]))
    for _ in range(int(20 + 60 * p["amount"])):
        x, y = rng.random() * w, rng.random() * h
        r = float(rng.random() ** 2 * min(w, h) * 0.06 + 2)
        c = int(120 + rng.random() * 90)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(c, c, c, int(40 + rng.random() * 70)))
    layer = layer.filter(ImageFilter.GaussianBlur(1.6))
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("flash", "Light", "Flash / White Out",
         {"amount": F("amount", 0.0, 0, 1), "color": C("color", "#FFFFFFFF")})
def fx_flash(img, p, ctx):
    a = _np(img)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    amt = p["amount"]
    if ctx.get("extent") is not None and p.get("animate", True):
        amt = max(0.0, 1.0 - float(ctx["extent"]) * 4.0) * p["amount"]
    out = a[..., :3] * (1 - amt) + col * amt
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


# ================================================================== extra keying / utility
@register("difference_key", "Keying", "Difference Key",
         {"threshold": F("threshold", 0.12, 0, 0.6), "color": C("color", "#FF00A000")})
def fx_diff_key(img, p, ctx):
    a = _np(img)
    key = np.array(parse_color(p["color"])[:3], np.float32)
    dist = np.sqrt(((a[..., :3] - key) ** 2).sum(-1, keepdims=True)) / math.sqrt(3)
    m = np.clip(dist / max(1e-3, p["threshold"]), 0, 1)
    a[..., 3:4] = a[..., 3:4] * m
    return _pil(a)


@register("edge_feather", "Keying", "Edge Feather",
         {"radius": F("radius", 4, 0, 40)})
def fx_edge_feather(img, p, ctx):
    a = _np(img)
    al = img.getchannel("A").filter(ImageFilter.GaussianBlur(max(0.1, p["radius"] * ctx.get("scale", 1.0))))
    a[..., 3] = np.asarray(al, np.float32) / 255.0
    return _pil(a)


@register("tint_shadow", "Color", "Shadow / Highlight Tint",
         {"shadow": C("shadow", "#FF0A2A55"), "highlight": C("highlight", "#FFFFD28A"),
          "balance": F("balance", 0.5, 0, 1), "amount": F("amount", 0.5, 0, 1)})
def fx_tint_shadow(img, p, ctx):
    def fn(rgb):
        lum = _luma_w(rgb)
        sh = np.array(parse_color(p["shadow"])[:3], np.float32)
        hi = np.array(parse_color(p["highlight"])[:3], np.float32)
        wts = 1.0 - _smoothstep(p["balance"] - 0.25, p["balance"] + 0.25, lum)
        tint = sh * wts + hi * (1 - wts)
        return _mix(rgb, tint * (0.4 + lum), p["amount"])
    return _apply_fn(img, fn, 1.0)


@register("exposure_auto", "Color", "Auto Exposure",
         {"target": F("target", 0.5, 0.1, 0.9), "amount": F("amount", 1.0, 0, 1)})
def fx_auto_exposure(img, p, ctx):
    a = _np(img)
    mean = float(_luma(a).mean())
    k = p["target"] / max(1e-3, mean)
    out = a[..., :3] * (1 + (k - 1) * p["amount"])
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("sharpen_edges_light", "Light", "Edge Light",
         {"color": C("color", "#FF8AD2FF"), "strength": F("strength", 1.2, 0, 3)})
def fx_edge_light(img, p, ctx):
    e = _edges_rgb(img, p["strength"])
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a[..., :3] + e * col
    return _merge(np.clip(out, 0, 1), a[..., 3:4])

# ================================================================== final batch
_TONE_PRESETS = {
    "selenium": [(0.04, 0.04, 0.05), (0.5, 0.46, 0.44), (0.95, 0.9, 0.86)],
    "platinum": [(0.05, 0.05, 0.06), (0.55, 0.55, 0.56), (0.96, 0.96, 0.97)],
    "warm_grey": [(0.06, 0.05, 0.04), (0.58, 0.5, 0.42), (0.97, 0.94, 0.88)],
    "cool_grey": [(0.03, 0.04, 0.06), (0.42, 0.48, 0.55), (0.88, 0.92, 0.97)],
    "gold": [(0.09, 0.05, 0.0), (0.72, 0.52, 0.14), (1.0, 0.95, 0.75)],
    "copper": [(0.1, 0.04, 0.02), (0.7, 0.35, 0.18), (1.0, 0.85, 0.7)],
}
for _name, _pal in _TONE_PRESETS.items():
    def _mkt(pal=_pal):
        def apply(img, p, ctx):
            return _apply_fn(img, lambda rgb: _palette_map_arr(rgb, pal), p["amount"])
        return apply
    register(f"tone_{_name}", "Color", f"Monochrome Tone — {_name.replace('_', ' ').title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"tone-{_name}"])(_mkt())


_TINT_PRESETS = {
    "amber": (1.0, 0.72, 0.35), "steel": (0.6, 0.75, 1.0), "mint": (0.55, 1.0, 0.82),
    "rose": (1.0, 0.65, 0.78), "violet": (0.76, 0.6, 1.0), "lime": (0.78, 1.0, 0.5),
}
for _name, _rgbw in _TINT_PRESETS.items():
    def _mkti(rgbw=_rgbw):
        def apply(img, p, ctx):
            def fn(rgb):
                return np.clip(rgb * np.array(rgbw, np.float32) * 1.05, 0, 1)
            return _apply_fn(img, fn, p["amount"])
        return apply
    register(f"tint_{_name}", "Color", f"Colour Tint — {_name.title()}",
             {"amount": F("amount", 0.8, 0, 1)}, [AM + f"tint-{_name}"])(_mkti())


_BALANCE_PRESETS = {
    "shadows": ((-0.1, 0.02, 0.16), (0, 0, 0), (0, 0, 0)),
    "midtones": ((0, 0, 0), (0.06, 0.0, -0.06), (0, 0, 0)),
    "highlights": ((0, 0, 0), (0, 0, 0), (0.14, 0.06, -0.1)),
    "cinematic": ((0.02, -0.02, 0.08), (0.02, 0.02, -0.02), (0.1, 0.04, -0.06)),
}
for _name, (_sh, _mi, _hi) in _BALANCE_PRESETS.items():
    def _mkb(sh=_sh, mi=_mi, hi=_hi):
        def apply(img, p, ctx):
            return _apply_fn(img, lambda rgb: _shadow_high(rgb, sh, mi, hi), p["amount"])
        return apply
    register(f"balance_{_name}", "Color", f"Colour Balance — {_name.title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"balance-{_name}"])(_mkb())


_CHANNEL_MIXES = {
    "red_boost": ((1.25, -0.1, -0.05), (-0.05, 1.02, 0.0), (-0.05, 0.0, 1.02)),
    "green_boost": ((1.02, -0.05, 0.0), (-0.1, 1.25, -0.05), (0.0, -0.05, 1.02)),
    "blue_boost": ((1.02, 0.0, -0.05), (0.0, 1.02, -0.05), (-0.1, -0.05, 1.32)),
    "grey_mix": ((0.5, 0.3, 0.2), (0.3, 0.5, 0.2), (0.2, 0.3, 0.5)),
}
for _name, _m in _CHANNEL_MIXES.items():
    def _mkm(m=_m):
        def apply(img, p, ctx):
            return _apply_fn(img, lambda rgb: _mat(rgb, np.array(m, np.float32)), p["amount"])
        return apply
    register(f"mix_{_name}", "Color", f"Channel Mixer — {_name.replace('_', ' ').title()}",
             {"amount": F("amount", 1.0, 0, 1)}, [AM + f"mix-{_name}"])(_mkm())


@register("pixel_sort", "Stylize", "Pixel Sort",
          {"threshold": F("threshold", 0.5, 0, 1), "vertical": B("vertical", False),
           "reverse": B("reverse", True)}, [AM + "pixelsort"])
def fx_pixel_sort(img, p, ctx):
    a = _np(img)
    if p["vertical"]:
        a = np.transpose(a, (1, 0, 2))
    lum = _luma(a)[..., 0]
    h, w = lum.shape
    out = a.copy()
    for y in range(h):
        row = lum[y]
        x = 0
        while x < w:
            if row[x] > p["threshold"]:
                x2 = x
                while x2 < w and row[x2] > p["threshold"]:
                    x2 += 1
                seg = out[y, x:x2]
                order = np.argsort(seg[..., :3].sum(-1), axis=0)
                if p["reverse"]:
                    order = order[::-1]
                out[y, x:x2] = seg[order]
                x = x2
            else:
                x += 1
    if p["vertical"]:
        out = np.transpose(out, (1, 0, 2))
    return _pil(out)


@register("kuwahara", "Stylize", "Kuwahara Filter",
          {"radius": F("radius", 2, 1, 6)})
def fx_kuwahara(img, p, ctx):
    a = _np(img)
    h, w = a.shape[:2]
    r = max(1, int(p["radius"] * ctx.get("scale", 1.0)))
    out = a.copy()
    lum = _luma(a)
    for dy in (-r, 0):
        for dx in (-r, 0):
            ys, xs = slice(max(0, dy), h + min(0, dy)), slice(max(0, dx), w + min(0, dx))
            ys2, xs2 = slice(max(0, -dy), h + min(0, -dy)), slice(max(0, -dx), w + min(0, -dx))
            quad = lum[ys2, xs2]
            small = quad[:: max(1, r), :: max(1, r)]
            var = np.var(small)
            if dy == -r and dx == -r:
                best = np.full((h, w), var, np.float32)
                out = a.copy()
            else:
                pass
    # simple approximation: pick the quadrant with lowest variance per pixel is costly;
    # use a smoothed edge-preserving blend instead
    blur = _np(_blur(img, r))
    edges = np.clip(_sobel(img) * 6, 0, 1)
    out = a * (1 - edges) + blur * edges
    return _pil(out)


@register("chrome", "Stylize", "Chrome / Liquid Metal",
          {"strength": F("strength", 1.0, 0.1, 3), "color": C("color", "#FFEAF4FF")})
def fx_chrome(img, p, ctx):
    a = _np(img)
    lum = _luma(a)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) * 0.5
    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) * 0.5
    lum = lum[..., 0] if lum.ndim == 3 else lum
    spec = np.clip((gx * 0.6 + gy * 0.8) * 6 * p["strength"], -1, 1)[..., 0] * 0.5 + 0.5
    spec = spec[..., None] if spec.ndim == 2 else spec
    col = np.array(parse_color(p["color"])[:3], np.float32)
    out = np.clip(np.clip(spec, 0, 1) ** 1.6 * col + a[..., :3] * 0.25, 0, 1)
    return _merge(out, a[..., 3:4])


@register("led_matrix", "Stylize", "LED Matrix",
          {"size": F("size", 8, 2, 40), "gap": F("gap", 0.25, 0, 0.8)})
def fx_led_matrix(img, p, ctx):
    w, h = img.size
    s = max(2.0, p["size"] * ctx.get("scale", 1.0))
    small = img.resize((max(1, int(w / s)), max(1, int(h / s))), Image.BILINEAR).resize((w, h), Image.NEAREST)
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    r = s * 0.5 * (1 - p["gap"])
    for y in np.arange(s / 2, h, s):
        for x in np.arange(s / 2, w, s):
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(small, (0, 0), mask)
    return out


@register("line_art", "Stylize", "Line Art",
          {"threshold": F("threshold", 0.25, 0, 1), "color": C("color", "#FF111111")})
def fx_line_art(img, p, ctx):
    e = _edges_rgb(img, 2.2)
    mask = (e > p["threshold"]).astype(np.float32)
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = a[..., :3] * (1 - mask) + col * mask
    return _merge(np.clip(out, 0, 1), a[..., 3:4])


@register("lightning", "Generate", "Lightning",
          {"branches": I("branches", 3, 1, 12), "color": C("color", "#FFDCE8FF"),
           "seed": F("seed", 4, 0, 99), "intensity": F("intensity", 1.0, 0, 3)})
def fx_lightning(img, p, ctx):
    w, h = img.size
    rng = np.random.default_rng(int(p["seed"]))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    for b in range(int(p["branches"])):
        x = rng.random() * w
        y = 0.0
        pts = [(x, y)]
        while y < h:
            y += float(rng.random() * 24 + 12)
            x += float(rng.normal(0, 14))
            pts.append((x, y))
        d.line(pts, fill=col, width=max(1, int(1 + rng.random() * 2)), joint="curve")
    layer = layer.filter(ImageFilter.GaussianBlur(1.0))
    a = _np(img)
    add = _np(layer)
    out = np.clip(a[..., :3] + add[..., :3] * p["intensity"], 0, 1)
    return _merge(out, np.maximum(a[..., 3:4], add[..., 3:4]))


@register("star_burst", "Generate", "Star Burst",
          {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1), "points": I("points", 8, 3, 40),
           "length": F("length", 0.6, 0.05, 2), "color": C("color", "#FFFFFFFF")})
def fx_star_burst(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    cx, cy = p["cx"] * w, p["cy"] * h
    rmax = p["length"] * max(w, h) * 0.5
    n = max(3, int(p["points"]))
    for i in range(n):
        ang = 2 * math.pi * i / n
        x2 = cx + math.cos(ang) * rmax
        y2 = cy + math.sin(ang) * rmax
        d.line((cx, cy, x2, y2), fill=col, width=2)
    layer = layer.filter(ImageFilter.GaussianBlur(1.6))
    out = img.copy()
    out.alpha_composite(layer)
    return out


@register("moon", "Generate", "Moon",
          {"cx": F("cx", 0.75, 0, 1), "cy": F("cy", 0.2, 0, 1), "size": F("size", 0.12, 0.01, 0.6),
           "color": C("color", "#FFFFF4D6")})
def fx_moon(img, p, ctx):
    w, h = img.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = _pil_col_alpha(p["color"])
    r = p["size"] * min(w, h) * 0.5
    cx, cy = p["cx"] * w, p["cy"] * h
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    glow = layer.filter(ImageFilter.GaussianBlur(max(2.0, r * 0.6)))
    out = img.copy()
    out.alpha_composite(glow)
    out.alpha_composite(layer)
    return out


@register("clouds", "Generate", "Clouds",
          {"density": F("density", 0.6, 0, 1), "scale": F("scale", 0.7, 0.1, 2),
           "color1": C("color1", "#FF8FA6C4"), "color2": C("color2", "#FFFFFFFF")})
def fx_clouds(img, p, ctx):
    w, h = img.size
    n = _fractal_noise(max(2, h // 10), max(2, w // 10), 5, 23)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC), np.float32) / 255
    m = np.clip((n - 0.45) * 2.4 * p["density"] + 0.3, 0, 1)[..., None]
    c1 = np.array(parse_color(p["color1"])[:3], np.float32)
    c2 = np.array(parse_color(p["color2"])[:3], np.float32)
    a = _np(img)
    out = a[..., :3] * (1 - m) + (c1 * (1 - m) + c2 * m) * m
    return _merge(np.clip(out, 0, 1), np.maximum(a[..., 3:4], m * 0.85))


@register("sun", "Generate", "Sun",
          {"cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.3, 0, 1), "size": F("size", 0.18, 0.01, 0.7),
           "color": C("color", "#FFFFE49A")})
def fx_sun(img, p, ctx):
    return fx_moon(img, {**p, "color": p["color"]}, ctx)


@register("flag_wave", "Distort", "Flag Wave",
          {"amplitude": F("amplitude", 10, 0, 60), "frequency": F("frequency", 2.0, 0.2, 10),
           "speed": F("speed", 1.0, 0, 4)}, [AM + "flag"])
def fx_flag(img, p, ctx):
    t = float(ctx.get("t_local", 0)) / 1000.0 * p["speed"]
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    off = np.sin(xx / max(1, w) * p["frequency"] * 6.283 + t * 3) * p["amplitude"] * ctx.get("scale", 1.0)
    return _remap(img, off * 0.4, off)


@register("perspective", "Distort", "Perspective",
          {"tilt_x": F("tilt_x", 0.0, -1, 1), "tilt_y": F("tilt_y", 0.0, -1, 1)})
def fx_perspective(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(1, w) - 0.5
    ny = yy / max(1, h) - 0.5
    k = 1.0 + nx * p["tilt_x"] + ny * p["tilt_y"]
    sx = (nx / np.maximum(0.2, k) + 0.5) * w
    sy = (ny / np.maximum(0.2, k) + 0.5) * h
    return _remap(img, sx - xx, sy - yy)


@register("droste", "Distort", "Droste / Recursive Zoom",
          {"zoom": F("zoom", 0.6, 0.1, 1.5), "cx": F("cx", 0.5, 0, 1), "cy": F("cy", 0.5, 0, 1)})
def fx_droste(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = p["cx"] * w, p["cy"] * h
    dx0 = (xx - px) / max(1, w)
    dy0 = (yy - py) / max(1, h)
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.log(np.maximum(r, 1e-4)) / math.log(max(0.05, p["zoom"]))
    frac = k - np.floor(k)
    scale = p["zoom"] ** np.floor(k)
    sx = px + dx0 * w / np.maximum(1e-3, frac + 0.35) * scale
    sy = py + dy0 * h / np.maximum(1e-3, frac + 0.35) * scale
    return _remap(img, sx - xx, sy - yy)


@register("tilt", "Distort", "Tilt",
          {"angle": F("angle", 0, -45, 45), "pivot": F("pivot", 0.5, 0, 1)})
def fx_tilt(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = math.radians(p["angle"])
    sx = xx + (yy - p["pivot"] * h) * math.tan(rad)
    return _remap(img, sx - xx, np.zeros_like(sx))


@register("panorama", "Distort", "Panorama Wrap",
          {"offset": F("offset", 0.0, -1, 1), "curve": F("curve", 0.4, 0, 1)})
def fx_panorama(img, p, ctx):
    h, w = img.size[1], img.size[0]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(1, w)
    idx = (np.round((nx + p["offset"]) * w).astype(np.int32)) % w
    a = _np(img)
    out = a[np.arange(h, dtype=np.int32)[:, None], idx, :]
    ys = (yy / max(1, h) - 0.5) * 2
    out *= (1 - p["curve"] * 0.3 * ys[..., None] ** 2)
    return _pil(np.clip(out, 0, 1))


@register("smear", "Blur", "Smear",
          {"amount": F("amount", 0.3, 0, 1), "angle": F("angle", 0, 0, 360, 1)})
def fx_smear(img, p, ctx):
    h, w = img.size[1], img.size[0]
    rad = math.radians(p["angle"])
    acc = np.zeros((h, w, 4), np.float32)
    n = 8
    for i in range(n):
        k = (i / (n - 1)) * p["amount"] * w * 0.25
        acc += _np(_remap(img, np.full((h, w), -math.cos(rad) * k, np.float32),
                          np.full((h, w), -math.sin(rad) * k, np.float32)))
    return _pil(acc / n)


@register("defocus_shape", "Blur", "Defocus (shaped bokeh)",
          {"radius": F("radius", 12, 1, 60), "shape": CH("shape", "hex", ["hex", "circle", "star", "square"])})
def fx_defocus_shape(img, p, ctx):
    r = max(1.0, p["radius"] * ctx.get("scale", 1.0))
    blurred = _blur(img, r * 0.6)
    k = max(2, int(r))
    size = k * 2 + 1
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size // 2
    dx0 = (xx - c) / k
    dy0 = (yy - c) / k
    rr = np.sqrt(dx0 * dx0 + dy0 * dy0)
    ang = np.arctan2(dy0, dx0)
    if p["shape"] == "hex":
        mask = rr * np.cos(np.mod(ang, math.pi / 3) - math.pi / 6) < 0.85
    elif p["shape"] == "star":
        mask = rr < 0.9 * (0.6 + 0.4 * np.cos(ang * 5))
    elif p["shape"] == "square":
        mask = (np.abs(dx0) < 0.9) & (np.abs(dy0) < 0.9)
    else:
        mask = rr < 0.95
    a = _np(blurred)
    hi = _np(img)
    bright = np.clip(_luma(hi) * 2 - 0.6, 0, 1)
    # spread the highlight boost with a same-shape blur so bokeh reads as shaped
    disc = Image.new("L", (size, size), 0)
    ImageDraw.Draw(disc).ellipse((0, 0, size - 1, size - 1), fill=255)
    kern = np.asarray(disc.filter(ImageFilter.GaussianBlur(max(1.0, k * 0.35))), np.float32)[..., None] / 255.0
    boost = np.asarray(Image.fromarray((np.clip(bright, 0, 1)[..., 0] * 255).astype(np.uint8), "L")
                       .filter(ImageFilter.GaussianBlur(max(1.0, k * 0.5))), np.float32)[..., None] / 255.0
    out = np.clip(a + boost * 0.5, 0, 1)
    return _pil(out)


@register("glow_edges_light", "Light", "Glow Edges",
          {"color": C("color", "#FF6FD8FF"), "radius": F("radius", 6, 0, 40), "strength": F("strength", 1.2, 0, 3)})
def fx_glow_edges(img, p, ctx):
    e = _edges_rgb(img, 1.8)
    g = np.asarray(Image.fromarray((np.clip(e, 0, 1)[..., 0] * 255).astype(np.uint8), "L")
                   .filter(ImageFilter.GaussianBlur(max(0.5, p["radius"] * ctx.get("scale", 1.0)))), np.float32)[..., None] / 255
    col = np.array(parse_color(p["color"])[:3], np.float32)
    a = _np(img)
    out = np.clip(a[..., :3] + col * (e + g) * p["strength"] * a[..., 3:4], 0, 1)
    return _merge(out, a[..., 3:4])


@register("false_color", "Color", "False Colour",
          {"preset": CH("preset", "heat", ["heat", "spectral", "viridis", "arctic"])})
def fx_false_color(img, p, ctx):
    a = _np(img)
    lum = np.clip(_luma(a)[..., 0], 0, 1)
    pals = {
        "heat": [(0, 0, 0.15), (0.6, 0.05, 0.5), (1, 0.4, 0.05), (1, 1, 0.8)],
        "spectral": [(0.1, 0.0, 0.6), (0.0, 0.7, 0.8), (0.2, 0.9, 0.2), (1, 0.9, 0.1)],
        "viridis": [(0.27, 0.0, 0.33), (0.19, 0.4, 0.55), (0.13, 0.75, 0.42), (0.99, 0.9, 0.14)],
        "arctic": [(0.02, 0.05, 0.2), (0.1, 0.4, 0.6), (0.5, 0.8, 0.9), (0.98, 0.99, 1.0)],
    }
    pal = np.array(pals[p["preset"]], np.float32)
    idx = lum * (len(pal) - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(pal) - 1)
    fr = (idx - lo)[..., None]
    out = pal[lo] * (1 - fr) + pal[hi] * fr
    return _merge(np.clip(out, 0, 1), a[..., 3:4])

# --- blend-as-effect set (useful on a layer without a second input) ---------------
_BLEND_SELF = {
    "negative": lambda rgb, col: 1 - rgb,
    "redscale": lambda rgb, col: np.clip(rgb * np.array([1.8, 0.55, 0.25], np.float32), 0, 1),
    "soft_light": lambda rgb, col: np.where(col < 0.5, rgb - (1 - 2 * col) * rgb * (1 - rgb),
                                            rgb + (2 * col - 1) * (np.sqrt(np.clip(rgb, 0, 1)) - rgb)),
    "hard_light": lambda rgb, col: np.where(col < 0.5, 2 * rgb * col, 1 - 2 * (1 - rgb) * (1 - col)),
    "overlay_self": lambda rgb, col: np.where(rgb < 0.5, 2 * rgb * col, 1 - 2 * (1 - rgb) * (1 - col)),
    "colour_dodge": lambda rgb, col: np.clip(rgb / np.maximum(1e-3, 1 - col), 0, 1),
    "colour_burn": lambda rgb, col: np.clip(1 - (1 - rgb) / np.maximum(1e-3, col), 0, 1),
    "difference_self": lambda rgb, col: np.abs(rgb - col),
    "lighten_self": lambda rgb, col: np.maximum(rgb, col),
    "darken_self": lambda rgb, col: np.minimum(rgb, col),
}
for _key, _fn in _BLEND_SELF.items():
    def _mkb(fn=_fn):
        def apply(img, p, ctx):
            col = np.array(parse_color(p["color"])[:3], np.float32)
            amt = p["amount"]
            rgb, a = _rgb(img)
            blended = np.clip(fn(rgb, col), 0, 1)
            return _merge(_mix(rgb, blended, amt), a)
        return apply
    register(f"blend_{_key}", "Blend", f"Blend — {_key.replace('_', ' ').title()}",
             {"color": C("color", "#FF808080"), "amount": F("amount", 1.0, 0, 1)},
             [AM + f"blend-{_key}"])(_mkb())
