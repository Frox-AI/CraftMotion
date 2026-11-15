"""CraftMotion core utilities: time, color, easing, keyframes.

Keyframes are stored AM-style: each keyframe carries the easing that applies
to the segment *after* it. Times are integer milliseconds everywhere in the
project model. AM XML uses seconds; conversion happens only at import/export.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------- generic

def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- colours
# Projects store colours as "#AARRGGBB" strings (same encoding as Alight Motion),
# which keeps round-tripping of AM XML lossless and makes JSON editable by hand.

def parse_color(value: Any, default: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> Tuple[float, float, float, float]:
    """Parse '#AARRGGBB' | '#RRGGBB' | '#RGB' | 'transparent' | [r,g,b,a] -> floats 0..1."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        vals = [float(x) for x in value]
        if len(vals) == 3:
            vals.append(1.0)
        if len(vals) == 4:
            # accept either 0..1 or 0..255 input
            if max(vals) > 1.0001:
                vals = [v / 255.0 for v in vals]
            return (vals[0], vals[1], vals[2], vals[3])
        return default
    s = str(value).strip()
    if s.lower() in ("", "none", "transparent"):
        return (0.0, 0.0, 0.0, 0.0)
    if s.startswith("#"):
        h = s[1:]
    elif s.lower().startswith("0x"):
        h = s[2:]
    else:
        # named colours, a few common ones
        named = {
            "black": "#FF000000", "white": "#FFFFFFFF", "red": "#FFFF0000",
            "green": "#FF00FF00", "blue": "#FF0000FF", "yellow": "#FFFFFF00",
            "cyan": "#FF00FFFF", "magenta": "#FFFF00FF", "orange": "#FFFF8800",
            "grey": "#FF808080", "gray": "#FF808080", "lightgrey": "#FFD8D8D8",
        }
        if s.lower() in named:
            h = named[s.lower()][1:]
        else:
            return default
    h = h.strip()
    if len(h) == 3:  # #RGB
        h = "".join(ch * 2 for ch in h)
    if len(h) == 6:  # RRGGBB -> opaque
        h = "FF" + h
    if len(h) == 4:  # ARGB short
        h = "".join(ch * 2 for ch in h)
    if len(h) != 8:
        return default
    try:
        a = int(h[0:2], 16) / 255.0
        r = int(h[2:4], 16) / 255.0
        g = int(h[4:6], 16) / 255.0
        b = int(h[6:8], 16) / 255.0
    except ValueError:
        return default
    return (r, g, b, a)


def to_argb(rgba: Sequence[float]) -> str:
    r, g, b, a = [clamp(float(x), 0.0, 1.0) for x in rgba]
    return "#%02X%02X%02X%02X" % (round(a * 255), round(r * 255), round(g * 255), round(b * 255))


def hex_to_rgb255(value: Any) -> Tuple[int, int, int, int]:
    r, g, b, a = parse_color(value)
    return (round(r * 255), round(g * 255), round(b * 255), round(a * 255))


def parse_vec(value: Any, default: Optional[List[float]] = None, dims: int = 2) -> List[float]:
    """'1.0,2.0' or [1,2] or 3 -> [3.0, 3.0] style vectors (scalar broadcasts)."""
    if default is None:
        default = [0.0] * dims
    if value is None:
        return list(default)
    if isinstance(value, (int, float)):
        return [float(value)] * dims
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").split(",") if p.strip() != ""]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return list(default)
        while len(vals) < dims:
            vals.append(vals[-1] if vals else 0.0)
        return vals[:dims]
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
        while len(vals) < dims:
            vals.append(vals[-1] if vals else 0.0)
        return vals[:dims]
    return list(default)


# ---------------------------------------------------------------- easing

Bezier = Tuple[float, float, float, float]

NAMED_EASINGS: Dict[str, Optional[Bezier]] = {
    "linear": None,
    "hold": "hold",  # type: ignore[dict-item]
    "step": "hold",  # type: ignore[dict-item]
    "ease": (0.25, 0.1, 0.25, 1.0),
    "ease_in": (0.42, 0.0, 1.0, 1.0),
    "ease_out": (0.0, 0.0, 0.58, 1.0),
    "ease_in_out": (0.42, 0.0, 0.58, 1.0),
    "sine_in": (0.12, 0.0, 0.39, 0.0),
    "sine_out": (0.61, 1.0, 0.88, 1.0),
    "sine_in_out": (0.37, 0.0, 0.63, 1.0),
    "quad_in": (0.11, 0.0, 0.5, 0.0),
    "quad_out": (0.5, 1.0, 0.89, 1.0),
    "cubic_in": (0.32, 0.0, 0.67, 0.0),
    "cubic_out": (0.33, 1.0, 0.68, 1.0),
    "cubic_in_out": (0.65, 0.0, 0.35, 1.0),
    "expo_out": (0.16, 1.0, 0.3, 1.0),
    "back_out": (0.34, 1.56, 0.64, 1.0),
    "back_in": (0.36, 0.0, 0.66, -0.56),
}


def parse_easing(value: Any):
    """Accepts None, 'ease_in', 'cubicBezier x1 y1 x2 y2', [x1,y1,x2,y2], 'hold'."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(float(x) for x in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_")
    if key in NAMED_EASINGS:
        return NAMED_EASINGS[key]
    if key.startswith("cubicbezier"):
        nums = []
        for tok in s.split()[1:]:
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        if len(nums) >= 4:
            return (nums[0], nums[1], nums[2], nums[3])
    return None


def bezier_ease(ease, u: float) -> float:
    """Cubic-bezier timing function (CSS semantics) evaluated at progress u.

    Accepts None (linear), the string names "linear"/"ease"/"ease_in"/"ease_out"/
    "ease_in_out"/"hold", or a 4-tuple of control points.
    """
    if ease is None:
        return u
    if isinstance(ease, str):
        name = ease.lower().replace("-", "_").replace(" ", "_")
        if name in ("hold", "step", "constant"):
            return 0.0
        if name == "linear":
            return u
        presets = {"ease": (0.25, 0.1, 0.25, 1.0), "ease_in": (0.42, 0.0, 1.0, 1.0),
                   "ease_out": (0.0, 0.0, 0.58, 1.0), "ease_in_out": (0.42, 0.0, 0.58, 1.0)}
        if name in presets:
            ease = presets[name]
        else:
            try:
                nums = [float(x) for x in name.replace("(", "").replace(")", "").split(",")]
            except ValueError:
                return u
            if len(nums) != 4:
                return u
            ease = tuple(nums)
    if not isinstance(ease, (tuple, list)) or len(ease) != 4:
        return u
    x1, y1, x2, y2 = ease
    if x1 == y1 and x2 == y2:
        return u
    # Newton-Raphson on x(t) then evaluate y(t).
    t = u
    for _ in range(8):
        cx = 3.0 * x1
        bx = 3.0 * (x2 - x1) - cx
        ax = 1.0 - cx - bx
        x = ((ax * t + bx) * t + cx) * t
        err = x - u
        if abs(err) < 1e-6:
            break
        dx = (3.0 * ax * t + 2.0 * bx) * t + cx
        if abs(dx) < 1e-6:
            break
        t -= err / dx
        t = clamp(t, 0.0, 1.0)
    cy = 3.0 * y1
    by = 3.0 * (y2 - y1) - cy
    ay = 1.0 - cy - by
    return ((ay * t + by) * t + cy) * t


# ---------------------------------------------------------------- keyframes
# A keyframe: {"t": int_ms, "v": [floats], "e": easing|None}
# "e" applies to the segment from this keyframe to the next one.

def normalize_kf(kf: dict, dims: int = 1) -> dict:
    return {
        "t": int(round(float(kf.get("t", 0)))),
        "v": parse_vec(kf.get("v", 0.0), dims=dims),
        "e": parse_easing(kf.get("e")),
    }


def kf_eval(keys: Optional[List[dict]], t_ms: float, default: Sequence[float]) -> List[float]:
    """Evaluate a keyframe track at t_ms, falling back to `default` when empty."""
    if not keys:
        return [float(x) for x in default]
    keys = sorted(keys, key=lambda k: k["t"])
    if t_ms <= keys[0]["t"]:
        return list(keys[0]["v"])
    if t_ms >= keys[-1]["t"]:
        return list(keys[-1]["v"])
    lo, hi = 0, len(keys) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keys[mid]["t"] <= t_ms:
            lo = mid
        else:
            hi = mid
    a, b = keys[lo], keys[hi]
    span = max(1, b["t"] - a["t"])
    u = (t_ms - a["t"]) / span
    ue = bezier_ease(a.get("e"), u)
    return [lerp(x, y, ue) for x, y in zip(a["v"], b["v"])]


def kf_add(keys: List[dict], t_ms: int, value: Sequence[float], easing=None) -> List[dict]:
    """Insert or replace a keyframe at t_ms (replaces near-duplicates)."""
    value = [float(v) for v in value]
    easing = parse_easing(easing)
    for k in keys:
        if abs(k["t"] - int(t_ms)) < 2:
            k["v"] = value
            k["e"] = easing
            return keys
    keys.append({"t": int(t_ms), "v": value, "e": easing})
    keys.sort(key=lambda k: k["t"])
    return keys


def kf_remove(keys: List[dict], t_ms: int, tol: int = 2) -> bool:
    for i, k in enumerate(keys):
        if abs(k["t"] - t_ms) <= tol:
            keys.pop(i)
            return True
    return False


def kf_find(keys: List[dict], t_ms: int, tol: int = 2) -> Optional[dict]:
    for k in keys:
        if abs(k["t"] - t_ms) <= tol:
            return k
    return None


def fmt_time(ms: float, fps: float = 30.0) -> str:
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, msec = divmod(rem, 1000)
    frames = int(round(msec / (1000.0 / max(fps, 1.0)))) % max(1, int(round(fps)))
    out = f"{m:02d}:{s:02d}.{frames:02d}"
    if h:
        out = f"{h:02d}:" + out
    return out


def pretty(value: Any) -> str:
    """Compact JSON-ish formatting for tool results."""
    if isinstance(value, float):
        return ("%.4f" % value).rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(pretty(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {pretty(v)}" for k, v in value.items()) + "}"
    return str(value)
