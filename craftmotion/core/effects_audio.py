"""Audio-reactive effects: spectrum bars, waveform ribbons and level pulses.

The Renderer analyses the mixed soundtrack once (``Renderer.audio_env``) and
exposes the value at the current frame as ``ctx["audio"]``; these effects read
that context so a 300-effect stack can still be audio-driven.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from .effects import F, I, B, C, CH, register, _np, _pil, _rgb, _merge, _luma, AM


def _audio_ctx(ctx: dict) -> dict:
    data = ctx.get("audio")
    if isinstance(data, dict) and data:
        return data
    return {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "wave": [], "spectrum": []}


@register("audio_spectrum", "Audio", "Audio Spectrum",
          {"bars": F("bars", 48, 1, 256, 1), "height": F("height", 0.45, 0.02, 1.5),
           "color": C("color", "#FF33E1FF"), "color2": C("color2", "#FFFF4FA3"),
           "thickness": F("thickness", 0.7, 0.1, 1.0), "mirror": B("mirror", True),
           "gain": F("gain", 1.0, 0.1, 6.0), "smooth": F("smooth", 0.35, 0, 0.95)},
          [AM + "audiospectrum", AM + "spectrum", AM + "audio-visualiser"])
def fx_audio_spectrum(img, p, ctx):
    """Draw an analyser-style spectrum over the image."""
    from .util import parse_color
    a = _audio_ctx(ctx)
    spec = a.get("spectrum") or []
    w, h = img.size
    bars = max(1, int(p["bars"]))
    base = _np(img)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(_pil(np.clip(base, 0, 1)))
    draw = ImageDraw.Draw(out)
    c1 = parse_color(p["color"]); c2 = parse_color(p["color2"])
    bw = w / float(bars)
    thick = max(1.0, bw * float(p["thickness"]))
    gain = float(p["gain"])
    smooth = float(p["smooth"])
    mirror = bool(p["mirror"])
    mid = h * 0.5 if mirror else float(h)
    prev = 0.0
    for i in range(bars):
        if spec:
            idx = int(i / bars * len(spec))
            v = float(spec[max(0, min(len(spec) - 1, idx))])
        else:
            v = float(a.get("level", 0.0))
        v = prev * smooth + v * (1 - smooth) * gain
        prev = v
        v = max(0.0, min(1.0, v))
        bh = max(1.0, v * h * float(p["height"]))
        x0 = i * bw + (bw - thick) / 2.0
        x1 = x0 + thick
        t = i / max(1, bars - 1)
        col = (int((c1[0] + (c2[0] - c1[0]) * t) * 255),
               int((c1[1] + (c2[1] - c1[1]) * t) * 255),
               int((c1[2] + (c2[2] - c1[2]) * t) * 255),
               int((c1[3] + (c2[3] - c1[3]) * t) * 255))
        if mirror:
            draw.rectangle((x0, mid - bh, x1, mid + bh), fill=col)
        else:
            draw.rectangle((x0, mid - bh, x1, mid), fill=col)
    return out


@register("audio_waveform", "Audio", "Audio Waveform",
          {"points": F("points", 96, 4, 512, 1), "thickness": F("thickness", 3.0, 1, 24),
           "color": C("color", "#FF7CFFB2"), "gain": F("gain", 1.0, 0.1, 6.0),
           "position": F("position", 0.5, 0, 1), "line": B("line", True)},
          [AM + "audiowaveform", AM + "waveform"])
def fx_audio_waveform(img, p, ctx):
    """Draw a oscilloscope ribbon across the frame."""
    from .util import parse_color
    a = _audio_ctx(ctx)
    wave = a.get("wave") or []
    w, h = img.size
    base = _np(img)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(_pil(np.clip(base, 0, 1)))
    draw = ImageDraw.Draw(out)
    col = parse_color(p["color"])
    fill = tuple(int(c * 255) for c in col)
    mid = h * float(p["position"])
    n = max(2, int(p["points"]))
    amp = h * 0.35 * float(p["gain"])
    pts = []
    for i in range(n):
        v = float(wave[int(i / (n - 1) * (len(wave) - 1))]) if wave else 0.0
        pts.append((i / (n - 1) * w, mid - v * amp))
    if p["line"]:
        draw.line(pts, fill=fill, width=int(max(1, p["thickness"])), joint="curve")
    else:
        poly = pts + [(w, mid), (0, mid)]
        draw.polygon(poly, fill=fill)
    return out


@register("audio_pulse", "Audio", "Audio Pulse",
          {"amount": F("amount", 0.35, 0, 1.5),
           "band": CH("band", "level", ["level", "low", "mid", "high"]),
           "color": C("color", "#FFFFFFFF"),
           "mode": CH("mode", "brighten", ["brighten", "tint", "saturate"])},
          [AM + "audiopulse", AM + "basspulse", AM + "beatflash"])
def fx_audio_pulse(img, p, ctx):
    """Flash / brighten the layer in time with a chosen band."""
    from .util import parse_color
    a = _audio_ctx(ctx)
    band = str(p.get("band") or "level").lower()
    v = float(a.get(band, a.get("level", 0.0)))
    v = max(0.0, min(1.0, v ** 0.8))
    amt = v * float(p["amount"])
    arr = _np(img)
    col = np.asarray(parse_color(p["color"])[:3], np.float32)
    mode = str(p.get("mode") or "brighten").lower()
    out = arr.copy()
    if mode == "tint":
        out[..., :3] = arr[..., :3] * (1 - amt) + col * amt * arr[..., 3:4]
    elif mode == "saturate":
        lum = _luma(arr)[..., 0:1] if _luma(arr).ndim == 3 else _luma(arr)
        out[..., :3] = np.clip(arr[..., :3] + (arr[..., :3] - lum) * amt * 1.6, 0, 1)
    else:
        out[..., :3] = np.clip(arr[..., :3] * (1 + amt * 1.2), 0, 1)
    return _pil(np.clip(out, 0, 1))


@register("audio_scale", "Audio", "Audio Zoom",
          {"amount": F("amount", 0.25, 0, 2.0),
           "band": CH("band", "level", ["level", "low", "mid", "high"]),
           "pivot_y": F("pivot_y", 0.5, 0, 1)},
          [AM + "audiozoom", AM + "beatshake"])
def fx_audio_scale(img, p, ctx):
    """Beat-driven punch-in."""
    a = _audio_ctx(ctx)
    band = str(p.get("band") or "level").lower()
    v = max(0.0, min(1.0, float(a.get(band, a.get("level", 0.0)))))
    k = 1.0 + v * float(p["amount"])
    w, h = img.size
    if abs(k - 1.0) < 1e-4:
        return img
    nw, nh = max(2, int(round(w * k))), max(2, int(round(h * k)))
    zoom = img.resize((nw, nh), Image.BICUBIC)
    cx, cy = nw // 2, int(nh * float(p["pivot_y"]))
    left = max(0, min(nw - w, cx - w // 2))
    top = max(0, min(nh - h, cy - h // 2))
    return zoom.crop((left, top, left + w, top + h))
