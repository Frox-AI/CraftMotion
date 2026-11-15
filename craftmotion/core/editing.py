"""Editing operations: three-point editing, trim modes, J/K/L transport maths."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .model import Project


def _layer_at(project: Project, track: int, t_ms: float, exclude: Optional[int] = None) -> Optional[dict]:
    for layer in project.layers:
        if exclude is not None and layer.get("id") == exclude:
            continue
        if int(layer.get("track", 0)) != int(track):
            continue
        if int(layer.get("start", 0)) <= t_ms < int(layer.get("end", project.duration)):
            return layer
    return None


def three_point_insert(project: Project, layer: dict, at_ms: float, source_in: Optional[float] = None,
                       source_out: Optional[float] = None, target_track: Optional[int] = None,
                       mode: str = "overwrite", ripple: bool = False) -> dict:
    """Insert `layer` at the playhead using source in/out and a target track."""
    s_in = int(source_in if source_in is not None else layer.get("source_in", 0) or 0)
    dur = int((source_out if source_out is not None else layer.get("end", 0)) - s_in)
    if dur <= 0:
        dur = int(layer.get("end", 2000)) - int(layer.get("start", 0))
    dur = max(200, dur)
    track = int(target_track if target_track is not None else layer.get("track", 0))
    start = int(at_ms)
    end = start + dur
    layer["source_in"] = s_in
    layer["start"] = start
    layer["end"] = end
    layer["track"] = track
    if layer not in project.layers:
        layer.setdefault("id", max([int(l.get("id", 0)) for l in project.layers] + [0]) + 1)
        project.layers.append(layer)

    overlaps = [l for l in project.layers
                if l.get("id") != layer.get("id") and int(l.get("track", 0)) == track
                and int(l.get("start", 0)) < end and int(l.get("end", 0)) > start]
    if mode == "overwrite":
        for other in overlaps:
            o_s, o_e = int(other.get("start", 0)), int(other.get("end", 0))
            if o_s >= start and o_e <= end:
                project.layers.remove(other)
                continue
            if o_e > end and o_s < start:
                # split around the insert
                tail = dict(other)
                tail["id"] = max([int(l.get("id", 0)) for l in project.layers] + [0]) + 1
                tail["start"] = end
                tail["source_in"] = int(other.get("source_in", 0)) + (end - o_s)
                other["end"] = start
                project.layers.append(tail)
                continue
            if o_s < start < o_e:
                other["end"] = start
            elif o_s < end < o_e:
                other["start"] = end
                other["source_in"] = int(other.get("source_in", 0)) + (end - o_s)
    elif ripple:
        for other in project.layers:
            if other is layer:
                continue
            if int(other.get("track", 0)) == track and int(other.get("start", 0)) >= start:
                shift = dur
                other["start"] = int(other.get("start", 0)) + shift
                other["end"] = int(other.get("end", 0)) + shift
    return {"layer": layer, "start": start, "end": end, "track": track, "mode": mode,
            "overwrote": len(overlaps)}


def trim_layer(project: Project, layer: dict, mode: str, amount_ms: int) -> dict:
    """Apply a professional trim mode. amount_ms is signed."""
    s, e = int(layer.get("start", 0)), int(layer.get("end", project.duration))
    src = int(layer.get("source_in", 0) or 0)
    amt = int(amount_ms)
    mode = (mode or "ripple").lower()
    if mode in ("ripple_in", "ripple-start"):
        layer["start"] = min(e - 100, max(0, s + amt))
        layer["source_in"] = src + (layer["start"] - s)
        _ripple_after(project, layer, layer["start"] - s)
    elif mode in ("ripple_out", "ripple-end"):
        layer["end"] = max(layer["start"] + 100, e + amt)
        _ripple_after(project, layer, layer["end"] - e, from_time=e)
    elif mode in ("roll", "roll_in"):
        layer["start"] = min(e - 100, max(0, s + amt))
        layer["source_in"] = src + (layer["start"] - s)
        prev = _prev_on_track(project, layer, s)
        if prev is not None:
            prev["end"] = layer["start"]
    elif mode in ("roll_out",):
        layer["end"] = max(layer["start"] + 100, e + amt)
        nxt = _next_on_track(project, layer, e)
        if nxt is not None:
            nxt["start"] = layer["end"]
            nxt["source_in"] = int(nxt.get("source_in", 0)) + (layer["end"] - e)
    elif mode == "slip":
        layer["source_in"] = max(0, src + amt)
    elif mode == "slide":
        layer["start"] = max(0, s + amt)
        layer["end"] = e + amt
        prev = _prev_on_track(project, layer, s)
        if prev is not None:
            prev["end"] = max(prev.get("start", 0) + 100, prev["end"] + amt)
        nxt = _next_on_track(project, layer, e)
        if nxt is not None:
            nxt["start"] = min(nxt.get("end", 0) - 100, nxt["start"] + amt)
            nxt["source_in"] = int(nxt.get("source_in", 0)) + amt
    elif mode == "set_in":
        layer["start"] = max(0, s + amt)
    elif mode == "set_out":
        layer["end"] = max(layer["start", 1] + 100, e + amt)
    else:  # simple move
        layer["start"] = max(0, s + amt)
        layer["end"] = e + amt
    return {"start": layer.get("start"), "end": layer.get("end"), "source_in": layer.get("source_in"),
            "mode": mode}


def _ripple_after(project: Project, layer: dict, delta: int, from_time: Optional[int] = None) -> None:
    if not delta:
        return
    cut = int(from_time if from_time is not None else layer.get("start", 0))
    for other in project.layers:
        if other is layer:
            continue
        if int(other.get("track", 0)) == int(layer.get("track", 0)) and int(other.get("start", 0)) >= cut:
            other["start"] = int(other.get("start", 0)) + delta
            other["end"] = int(other.get("end", 0)) + delta


def _prev_on_track(project: Project, layer: dict, before: int) -> Optional[dict]:
    best = None
    for other in project.layers:
        if other is layer or int(other.get("track", 0)) != int(layer.get("track", 0)):
            continue
        if int(other.get("end", 0)) <= before:
            if best is None or int(other.get("end", 0)) > int(best.get("end", 0)):
                best = other
    return best


def _next_on_track(project: Project, layer: dict, after: int) -> Optional[dict]:
    best = None
    for other in project.layers:
        if other is layer or int(other.get("track", 0)) != int(layer.get("track", 0)):
            continue
        if int(other.get("start", 0)) >= after:
            if best is None or int(other.get("start", 0)) < int(best.get("start", 0)):
                best = other
    return best


def jkl_rate(key: str, current: float = 1.0) -> float:
    """J/K/L transport: J reverse, K stop, L forward, repeats accelerate."""
    cur = float(current or 0.0)
    if key == "k":
        return 0.0
    if key == "l":
        if cur <= 0:
            return 1.0
        return min(8.0, cur * 2.0)
    if key == "j":
        if cur >= 0:
            return -1.0
        return max(-8.0, cur * 2.0)
    return cur


def frame_step(project: Project, t_ms: float, direction: int = 1) -> float:
    fps = max(1.0, float(project.fps))
    step = 1000.0 / fps
    return max(0.0, min(float(project.duration), round((t_ms + direction * step) / step) * step))
