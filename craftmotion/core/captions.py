"""Captions / subtitles: cue model, SRT + WebVTT import & export, burned-in render."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .util import parse_color

DEFAULT_STYLE = {
    "size": 44, "color": "#FFFFFFFF", "bg": "#B0000000", "stroke": "#FF000000",
    "stroke_width": 0, "pos": "bottom", "margin": 0.08, "font": "", "bold": True,
    "align": "center", "shadow": 0,
}


def _font(size: int, name: str = "", bold: bool = False):
    from .render import load_font
    return load_font(size, name, bold=bold)


def parse_timestamp(text: str) -> int:
    text = text.strip().replace(",", ".")
    m = re.match(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?", text)
    if not m:
        return 0
    hh = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = m.group(4) or "0"
    ms = int(float("0." + frac) * 1000)
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def fmt_timestamp(ms: float) -> str:
    ms = max(0, int(round(ms)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def import_srt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        raw = fh.read()
    return parse_srt_text(raw)


def parse_srt_text(raw: str) -> List[dict]:
    cues: List[dict] = []
    blocks = re.split(r"\r?\n\r?\n", raw.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        idx = 0
        if re.match(r"^\d+$", lines[0].strip()):
            idx = 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        times = lines[idx].split("-->")
        start = parse_timestamp(times[0])
        end = parse_timestamp(times[1].split()[0] if times[1].strip() else "0")
        text = "\n".join(lines[idx + 1:]).strip()
        if text:
            cues.append({"start": start, "end": max(start + 200, end), "text": text})
    return cues


def import_vtt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        raw = fh.read()
    raw = re.sub(r"^WEBVTT.*?\n\n", "", raw, flags=re.S)
    return parse_srt_text(raw.replace(".", ","))


def export_srt(cues: List[dict], path: str) -> str:
    lines = []
    for i, cue in enumerate(sorted(cues, key=lambda c: c.get("start", 0)), 1):
        lines.append(str(i))
        lines.append(f"{fmt_timestamp(cue.get('start', 0))} --> {fmt_timestamp(cue.get('end', 0))}")
        lines.append(str(cue.get("text", "")).strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def export_vtt(cues: List[dict], path: str) -> str:
    lines = ["WEBVTT", ""]
    for i, cue in enumerate(sorted(cues, key=lambda c: c.get("start", 0)), 1):
        def ts(ms):
            return fmt_timestamp(ms).replace(",", ".")
        lines.append(str(i))
        lines.append(f"{ts(cue.get('start', 0))} --> {ts(cue.get('end', 0))}")
        lines.append(str(cue.get("text", "")).strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def active_cues(cues: List[dict], t_ms: float) -> List[dict]:
    return [c for c in cues if float(c.get("start", 0)) <= t_ms < float(c.get("end", 0))]


def draw_captions(img: Image.Image, t_ms: float, cues: List[dict],
                  style: Optional[dict] = None) -> Image.Image:
    active = active_cues(cues, t_ms)
    if not active:
        return img
    st = dict(DEFAULT_STYLE)
    st.update(style or {})
    w, h = img.size
    scale = w / 1080.0
    size = max(8, int(float(st["size"]) * scale))
    font = _font(size, str(st.get("font", "")), bool(st.get("bold", True)))
    out = img.copy()
    draw = ImageDraw.Draw(out)
    color = parse_color(st["color"])
    fill = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(color[3] * 255))
    bg = parse_color(st["bg"])
    bgc = (int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255), int(bg[3] * 255))
    stroke_c = parse_color(st["stroke"])
    sc = (int(stroke_c[0] * 255), int(stroke_c[1] * 255), int(stroke_c[2] * 255), 255)

    # stack multiple simultaneous cues
    total_h = 0
    blocks = []
    for cue in active:
        for line in str(cue.get("text", "")).split("\n"):
            blocks.append(line)
    line_h = int(size * 1.32)
    total_h = line_h * len(blocks)
    margin = float(st.get("margin", 0.08)) * h
    pos = str(st.get("pos", "bottom")).lower()
    if pos == "top":
        y = margin
    elif pos == "center":
        y = (h - total_h) / 2
    else:
        y = h - margin - total_h
    for line in blocks:
        tw = draw.textlength(line, font=font) if hasattr(draw, "textlength") else font.getlength(line)
        if st["align"] == "left":
            x = margin
        elif st["align"] == "right":
            x = w - margin - tw
        else:
            x = (w - tw) / 2
        pad = max(2, size * 0.18)
        if bgc[3] > 0:
            draw.rounded_rectangle((x - pad, y - pad * 0.4, x + tw + pad, y + line_h - pad * 0.4),
                                   radius=pad * 0.6, fill=bgc)
        sw = int(st.get("stroke_width", 0) or 0)
        if sw > 0:
            draw.text((x, y), line, font=font, fill=fill, stroke_width=sw, stroke_fill=sc)
        else:
            draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return out


def auto_cues_from_script(text: str, start_ms: int = 0, per_line_ms: int = 2200) -> List[dict]:
    """Split a plain script into evenly-timed cues (handy for quick captioning)."""
    cues = []
    t = start_ms
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        # ~14 characters per second reading speed
        dur = max(1200, min(8000, int(len(line) / 14.0 * 1000)))
        cues.append({"start": t, "end": t + dur, "text": line})
        t += dur
    return cues
