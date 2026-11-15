"""Alight Motion parity effects.

AM exposes a handful of effects that animate *geometry* rather than pixels
(oscillate, swing, blink) plus noise-family distortions whose parameter names
differ from ours. These implementations exist so imported AM presets keep their
motion instead of degrading to pass-through metadata.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .effects import F, I, B, C, CH, register, _np, _pil, _remap, _blur, AM


def _t(ctx: dict) -> float:
    """Seconds since the layer started (AM oscillators run on layer time)."""
    return float(ctx.get("t_local", 0) or 0) / 1000.0


@register("oscillate", "Distort", "Oscillate",
          {"mag": F("mag", 6.0, -100, 100), "freq": F("freq", 1.0, 0, 12),
           "angle": F("angle", 60.0, 0, 360), "phase": F("phase", 0.0, 0, 360),
           "direction": I("direction", 0, 0, 3)},
          [AM + "oscillate", AM + "oscillate2", AM + "oscillate3"])
def fx_oscillate(img, p, ctx):
    """Sine-wave positional oscillation (AM Oscillate)."""
    t = _t(ctx)
    wave = math.sin(2 * math.pi * p["freq"] * t + math.radians(p["phase"]))
    mag = p["mag"] * (ctx.get("scale", 1.0) or 1.0)
    ang = math.radians(p["angle"])
    dx = np.full((img.size[1], img.size[0]), mag * math.cos(ang) * wave, np.float32)
    dy = np.full((img.size[1], img.size[0]), mag * math.sin(ang) * wave, np.float32)
    return _remap(img, dx, dy)


@register("swing", "Distort", "Swing",
          {"freq": F("freq", 1.0, 0, 12), "a1": F("a1", -45.0, -180, 180),
           "a2": F("a2", 45.0, -180, 180), "phase": F("phase", 0.0, 0, 360)},
          [AM + "swing", AM + "swing2"])
def fx_swing(img, p, ctx):
    """Rock between two angles (AM Swing)."""
    t = _t(ctx)
    wave = 0.5 + 0.5 * math.sin(2 * math.pi * p["freq"] * t + math.radians(p["phase"]))
    angle = p["a1"] + (p["a2"] - p["a1"]) * wave
    if abs(angle) < 0.01:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


@register("blink", "Time", "Blink",
          {"freq": F("freq", 8.0, 0.2, 40), "duty": F("duty", 0.5, 0.05, 0.95),
           "amount": F("amount", 1.0, 0, 1)},
          [AM + "blink", AM + "blink2"])
def fx_blink(img, p, ctx):
    """Hard on/off strobing (AM Blink)."""
    t = _t(ctx)
    phase = (t * p["freq"]) % 1.0
    on = 1.0 if phase < p["duty"] else 0.0
    alpha = 1.0 - p["amount"] + p["amount"] * on
    if alpha >= 0.999:
        return img
    a = _np(img)
    a[..., 3:4] *= alpha
    return _pil(a)


@register("random_displace", "Distort", "Random Displace",
          {"mag": F("mag", 12.0, 0, 200), "evolution": F("evolution", 1.0, 0, 10),
           "seed": F("seed", 0.0, 0, 100), "scatter": F("scatter", 0.6, 0, 3)},
          [AM + "randomdisplace", AM + "randomdisplacement"])
def fx_random_displace(img, p, ctx):
    """Animated random offsets — the AM "Random Displace" glitch look."""
    h, w = img.size[1], img.size[0]
    t = _t(ctx) * p["evolution"]
    rng = np.random.default_rng(int(p["seed"] * 100) + 1)
    base = rng.random((h, w, 2), dtype=np.float32) - 0.5
    # scroll the field so it evolves over time without re-seeding
    shift = int((t * 37) % max(1, w))
    field = np.roll(base, shift, axis=1)
    field = np.roll(field, int((t * 23) % max(1, h)), axis=0)
    k = p["mag"] * (ctx.get("scale", 1.0) or 1.0) * p["scatter"]
    return _remap(img, field[..., 0] * k, field[..., 1] * k)


@register("turbulence", "Distort", "Turbulence Displace",
          {"intensity": F("intensity", 0.2, 0, 2), "scale": F("scale", 40.0, 1, 400),
           "evolution": F("evolution", 0.2, 0, 5), "octaves": F("octaves", 3.0, 1, 6),
           "seed": F("seed", 0.0, 0, 100)},
          [AM + "turbulence", AM + "turbulence2", AM + "turbulentdisplace",
           AM + "turbulentdisplace2", AM + "turbulentdisplace3"])
def fx_turbulence(img, p, ctx):
    """Smooth fbm displacement (AM Turbulent Displace)."""
    h, w = img.size[1], img.size[0]
    t = _t(ctx) * p["evolution"]
    k = p["intensity"] * (ctx.get("scale", 1.0) or 1.0)
    acc = np.zeros((h, w, 2), np.float32)
    amp = 1.0
    total = 0.0
    for o in range(max(1, int(p["octaves"]))):
        freq = (o + 1) * 6.28 / max(8.0, p["scale"])
        rng = np.random.default_rng(int(p["seed"]) + o * 17 + 5)
        ph = rng.random((2,), dtype=np.float32) * 6.28
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        acc[..., 0] += amp * np.sin(xx * freq + yy * freq * 0.6 + ph[0] + t * (o + 1))
        acc[..., 1] += amp * np.cos(yy * freq - xx * freq * 0.4 + ph[1] + t * (o + 1) * 0.8)
        total += amp
        amp *= 0.55
    acc /= max(total, 1e-6)
    return _remap(img, acc[..., 0] * 18.0 * k, acc[..., 1] * 18.0 * k)
