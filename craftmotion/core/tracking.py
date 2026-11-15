"""Point tracking and stabilisation (normalised cross-correlation on luma patches)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _luma_patch(img: Image.Image, cx: float, cy: float, half: int) -> Optional[np.ndarray]:
    w, h = img.size
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + half * 2 + 1, y0 + half * 2 + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    crop = img.convert("L").crop((x0, y0, x1, y1))
    a = np.asarray(crop, np.float32) / 255.0
    return a - a.mean()


def _search(img: Image.Image, template: np.ndarray, cx: float, cy: float, half: int,
            radius: int) -> Optional[Tuple[float, float, float]]:
    """NCC search around (cx, cy); returns (x, y, score)."""
    w, h = img.size
    gray = np.asarray(img.convert("L"), np.float32) / 255.0
    best = (-2.0, cx, cy)
    t = template
    tn = float(np.sqrt((t * t).sum()) + 1e-6)
    for dy in range(-radius, radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            x0 = int(round(cx + dx)) - half
            y0 = int(round(cy + dy)) - half
            if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
                continue
            patch = gray[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1]
            patch = patch - patch.mean()
            pn = float(np.sqrt((patch * patch).sum()) + 1e-6)
            score = float((t * patch).sum() / (tn * pn))
            if score > best[0]:
                best = (score, cx + dx, cy + dy)
    if best[0] < -1.5:
        return None
    return (best[1], best[2], best[0])


def track_point(media, item: dict, x: float, y: float, start_ms: int, end_ms: int,
                step_ms: int = 100, patch: int = 12, search: int = 24) -> List[dict]:
    """Track a feature from start_ms to end_ms, returning [{t, x, y, score}]."""
    frame = media.get_frame(item, start_ms)
    if frame is None:
        return []
    template = _luma_patch(frame, x, y, patch)
    if template is None:
        return []
    out = [{"t": start_ms, "x": float(x), "y": float(y), "score": 1.0}]
    cx, cy = x, y
    t = start_ms
    while t + step_ms <= end_ms:
        t += step_ms
        frame = media.get_frame(item, t)
        if frame is None:
            break
        fh, fw = frame.size[1], frame.size[0]
        tmpl = _luma_patch(frame, cx, cy, patch)
        res = _search(frame, template, cx, cy, patch, search)
        if res is None:
            # re-acquire the template from the last good position
            res = _search(frame, template, cx, cy, patch, search * 2)
            if res is None:
                continue
        nx, ny, score = res
        if score < 0.55:
            out.append({"t": t, "x": float(nx), "y": float(ny), "score": round(score, 3)})
            continue
        cx, cy = nx, ny
        if tmpl is not None:
            template = 0.7 * template + 0.3 * tmpl  # slow drift adaptation
        out.append({"t": t, "x": float(nx), "y": float(ny), "score": round(score, 3)})
    return out


def to_keyframes(points: List[dict], prop: str = "pos", projectile: str = "x") -> List[dict]:
    keys = []
    for p in points:
        v = [float(p["x"]), float(p["y"])]
        keys.append({"t": int(p["t"]), "v": v, "e": "linear"})
    return keys


def stabilize_patch(points: List[dict], smooth: int = 9, strength: float = 1.0) -> List[dict]:
    """Turn a track into corrective position offsets that cancel the motion."""
    if not points:
        return []
    xs = np.array([p["x"] for p in points], np.float32)
    ys = np.array([p["y"] for p in points], np.float32)
    k = max(1, int(smooth))
    kernel = np.ones(k, np.float32) / k
    sx = np.convolve(xs, kernel, mode="same")
    sy = np.convolve(ys, kernel, mode="same")
    # edge fix-up so the convolution doesn't pull in zeros
    sx[:k] = xs[:k].mean()
    sy[:k] = ys[:k].mean()
    sx[-k:] = xs[-k:].mean()
    sy[-k:] = ys[-k:].mean()
    out = []
    for i, p in enumerate(points):
        out.append({"t": int(p["t"]),
                    "v": [float((sx[i] - xs[i]) * strength), float((sy[i] - ys[i]) * strength)],
                    "e": "linear"})
    return out


def apply_track(layer: dict, track: List[dict], mode: str = "attach", strength: float = 1.0,
                absolute: bool = True, append: bool = False) -> dict:
    """Write a track into a layer as position keyframes (attach) or stabilise offsets.

    ``absolute=True`` makes the layer sit *on* the tracked feature (the usual
    "attach to point" behaviour); ``absolute=False`` keeps only the relative
    motion so an existing transform is preserved.
    """
    keys = (layer.setdefault("keys", {})).setdefault("pos", [])
    if not append:
        # a fresh track replaces the previous bake instead of stacking on top of it
        keys.clear()
    if mode == "stabilize":
        new_keys = stabilize_patch(track, 9, strength)
    else:
        base = (track[0] if track else {"x": 0.0, "y": 0.0}) if not absolute else {"x": 0.0, "y": 0.0}
        new_keys = [{"t": int(p["t"]), "v": [(p["x"] - base["x"]) * strength, (p["y"] - base["y"]) * strength],
                     "e": "linear"} for p in track]
    keys.extend(new_keys)
    keys.sort(key=lambda k: k["t"])
    layer["keys"]["pos"] = keys
    layer.setdefault("track", {})["data"] = track
    layer["track"]["mode"] = mode
    return layer
