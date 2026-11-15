"""Frame compositor: layers -> affine transform -> effects -> blend.

One entry point (`render_frame`) is shared by the GUI preview, the render
queue and the MCP server, so what agents see is byte-identical to what the
human sees.
"""
from __future__ import annotations

import math
import os
import subprocess
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import audio as audio_mod
from . import effects as FX
from .captions import draw_captions
from .expressions import ExpressionEngine
from .grade import apply_grade, ensure_grade
from .transitions import TRANSITIONS, apply_transition
from .media import MediaManager, ffmpeg_path
from .model import Project
from .util import clamp, hex_to_rgb255, kf_eval, parse_color, parse_vec, to_argb

# Params measured in pixels that must scale with the render resolution.
PIXEL_PARAMS: Dict[str, List[str]] = {
    "blur": ["radius"],
    "motion_blur": ["distance"],
    "sharpen": ["radius"],
    "glow": ["radius"],
    "bloom": ["radius"],
    "pixelate": ["size"],
    "chromatic": ["shift"],
    "glitch": ["block"],
    "wave": ["amp"],
    "ripple": ["amp"],
    "twirl": [],
    "bulge": [],
    "displace": ["amount", "scale"],
    "vignette": [],
    "drop_shadow": ["offset_x", "offset_y", "blur", "grow"],
    "zoom_blur": [],
    "spin_blur": [],
    "vhs": ["color_bleed"],
    "jitter": ["amount"],
    "grain": ["size"],
    "scanlines": [],
    "light_leak": [],
    "lut": [],
    "cartoon": ["smooth"],
}

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fonts")
_FONT_CACHE: Dict[Tuple[str, int, bool, bool], ImageFont.FreeTypeFont] = {}


def _font_file(name: str, bold: bool, italic: bool) -> str:
    n = (name or "").lower()
    if n.endswith((".ttf", ".otf")) and os.path.exists(name):
        return name
    table = {
        ("bold", False): "DejaVuSans-Bold.ttf",
        ("mono", False): "DejaVuSansMono.ttf",
        ("mono", True): "DejaVuSansMono-Bold.ttf",
    }
    filename = table.get(("mono" if "mono" in n else "bold" if bold else "sans", bold), "DejaVuSans.ttf")
    if "mono" in n:
        filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    path = os.path.join(FONT_DIR, filename)
    if os.path.exists(path):
        return path
    return os.path.join(FONT_DIR, "DejaVuSans.ttf")


def load_font(size: int, name: str = "", bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    size = max(4, int(size))
    key = (name or "", size, bold, italic)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _font_file(name, bold, italic)
    try:
        f = ImageFont.truetype(path, size)
    except Exception:
        try:
            f = ImageFont.truetype("arial.ttf", size)
        except Exception:
            f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f




def _pil_col(value) -> tuple:
    """parse_color -> 8-bit RGBA tuple for PIL drawing calls."""
    return hex_to_rgb255(value)

# ------------------------------------------------------------------ matrices


def mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def mat_translate(x: float, y: float) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[0, 2] = x
    m[1, 2] = y
    return m


def mat_rotate(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mat_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def mat_skew(sx_deg: float, sy_deg: float) -> np.ndarray:
    tx = math.tan(math.radians(clamp(sx_deg, -80, 80)))
    ty = math.tan(math.radians(clamp(sy_deg, -80, 80)))
    return np.array([[1, tx, 0], [ty, 1, 0], [0, 0, 1]], dtype=np.float64)


# ------------------------------------------------------------------ values


def eval_track(layer: dict, prop: str, t_ms: float, default: List[float],
               project: Optional[Project] = None) -> List[float]:
    keys = (layer.get("keys") or {}).get(prop)
    if keys:
        value = kf_eval(keys, t_ms, default)
    else:
        base = layer.get("transform", {}).get(prop, default)
        value = parse_vec(base, default, dims=len(default))
    expr = (layer.get("expressions") or {}).get(prop)
    if expr and project is not None:
        out = ExpressionEngine(project, t_ms, layer).evaluate(expr, value, keys)
        if out is not None:
            dims = len(default)
            while len(out) < dims:
                out.append(out[-1] if out else 0.0)
            return out[:dims]
    return value


def eval_param(project: Project, layer: dict, eff: dict, name: str, t_ms: float, default):
    expr = (eff.get("expressions") or {}).get(name)
    if expr and not isinstance(default, (str, bool)) and default is not None:
        val = ExpressionEngine(project, t_ms + int(layer.get("start", 0)), layer).eval_effect_param(
            layer, eff, name, default, t_ms)
        return val
    keys = (eff.get("keys") or {}).get(name)
    base = (eff.get("params") or {}).get(name, default)
    if keys:
        dims = len(default) if isinstance(default, (list, tuple)) else 1
        if isinstance(default, str) or isinstance(default, bool):
            return base
        return kf_eval(keys, t_ms, parse_vec(base if base is not None else default, dims=dims))[0] if dims == 1 \
            else kf_eval(keys, t_ms, parse_vec(base, dims=dims))
    return base


# ------------------------------------------------------------------ video source


class VideoFrameSource:
    """Sequential ffmpeg reader for fast export; falls back to seeks."""

    def __init__(self, mm: MediaManager, item: dict):
        self.mm = mm
        self.item = item
        self.proc: Optional[subprocess.Popen] = None
        self.size: Optional[Tuple[int, int]] = None
        self.path: Optional[str] = None
        self.pos_ms = 0
        self.fps = 30.0
        self.end_ms = int(item.get("duration_ms") or 0)

    def _open(self, path: str):
        from .media import probe_media
        self._close()
        info = probe_media(path)
        w = info.get("width") or 640
        h = info.get("height") or 360
        self.fps = float(info.get("fps") or 30.0)
        factor = min(1.0, 960.0 / max(1, w))
        self.size = (max(2, int(w * factor) // 2 * 2), max(2, int(h * factor) // 2 * 2))
        self.path = path
        return info

    def read(self, t_ms: float) -> Optional[Image.Image]:
        src = self.item.get("path")
        if not src or not os.path.exists(src):
            return None
        path = self.mm.proxy_path(self.item) or src
        t_ms = max(0.0, float(t_ms))
        if self.size is None or path != self.path:
            self._open(path)
        assert self.size is not None
        # restart if we need to go backwards or jump far
        if self.proc is None or t_ms < self.pos_ms - 40 or t_ms > self.pos_ms + 1500:
            self._start(path, t_ms)
        target = t_ms
        frame = None
        guard = 0
        while guard < 200:
            guard += 1
            if self.pos_ms > target + 45:
                break
            frame = self._read_one()
            if frame is None:
                break
            self.pos_ms += 1000.0 / self.fps
        return frame

    def _start(self, path: str, t_ms: float):
        self._close()
        assert self.size is not None
        w, h = self.size
        cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-v", "error",
               "-ss", f"{t_ms/1000.0:.3f}", "-i", path,
               "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={self.fps:.4f}",
               "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        flags = 0x08000000 if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         creationflags=flags)
            self.pos_ms = t_ms
        except Exception:
            self.proc = None

    def _read_one(self) -> Optional[Image.Image]:
        if not self.proc or not self.proc.stdout or not self.size:
            return None
        w, h = self.size
        n = w * h * 4
        try:
            raw = self.proc.stdout.read(n)
        except Exception:
            return None
        if not raw or len(raw) < n:
            self._close()
            return None
        return Image.frombytes("RGBA", (w, h), raw)

    def _close(self):
        if self.proc:
            try:
                self.proc.stdout and self.proc.stdout.close()
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def __del__(self):
        self._close()


# ------------------------------------------------------------------ renderer


class Renderer:
    def __init__(self, project: Project, media: Optional[MediaManager] = None, mode: str = "preview"):
        self.project = project
        self.media = media or MediaManager()
        self.mode = mode
        self.video_sources: "OrderedDict[int, VideoFrameSource]" = OrderedDict()
        self._last_media_cache: Dict[str, Image.Image] = {}
        self._mb_depth = 0
        self._in_subcomp = False
        self._cam_scale = 1.0
        self._lights: List[dict] = []
        self._precomp_stack: List[int] = []
        self._trail_depth = 0
        self._linear = False
        self._audio_cache = None

    # -------------------------------------------------------------- public
    def render(self, t_ms: float, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        mb = self.project.data.get("motion_blur") or {}
        samples = int(mb.get("samples", 4) or 4)
        if mb.get("enabled") and samples > 1 and self._mb_depth == 0 and not self._in_subcomp:
            samples = max(2, min(16, samples))
            shutter = clamp(float(mb.get("shutter", 0.5) or 0.5), 0.01, 2.0)
            span = (1000.0 / max(1.0, self.project.fps)) * shutter
            self._mb_depth += 1
            try:
                acc = None
                for i in range(samples):
                    off = (i / (samples - 1) - 0.5) * span
                    frame = self._render_frame(t_ms + off, size)
                    arr = _np_rgba(frame)
                    acc = arr if acc is None else acc + arr
                acc = acc / samples
                return Image.fromarray((np.clip(acc, 0, 1) * 255).astype(np.uint8), "RGBA")
            finally:
                self._mb_depth -= 1
        return self._render_frame(t_ms, size)

    def _render_frame(self, t_ms: float, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        proj = self.project
        out_w, out_h = size or (proj.width, proj.height)
        q = out_w / float(proj.width)
        bg = parse_color(proj.settings.get("bg", "#FF000000"))
        canvas = np.zeros((out_h, out_w, 4), dtype=np.float32)
        canvas[..., 0] = bg[0]
        canvas[..., 1] = bg[1]
        canvas[..., 2] = bg[2]
        canvas[..., 3] = bg[3]

        self._cam_scale = q
        cam_m = self._camera_matrix(t_ms, out_w, out_h)
        self._lights = self._collect_lights(t_ms)
        grade = ensure_grade(proj)
        self._linear = bool(proj.settings.get("linear_light") or grade.get("linear_light"))
        if self._linear:
            canvas[..., :3] = _srgb_to_lin(canvas[..., :3])
            # premultiplied-ish: keep alpha as coverage

        layers = proj.layers
        for idx, layer in enumerate(layers):
            if layer.get("hidden"):
                continue
            start = int(layer.get("start", 0))
            end = int(layer.get("end", proj.duration))
            if t_ms < start or t_ms >= end:
                continue
            if layer.get("kind") in ("audio", "null", "light", "camera"):
                continue
            trail = layer.get("trail") or {}
            trail_n = int(trail.get("count", 0) or 0) if trail.get("enabled", True) else 0
            if trail_n > 0 and self._trail_depth == 0:
                spacing = float(trail.get("spacing", 60) or 60)
                decay = float(trail.get("decay", 0.7) or 0.7)
                self._trail_depth += 1
                try:
                    for k in range(trail_n, 0, -1):
                        ghost = self._render_layer(np.zeros_like(canvas), layer, t_ms - spacing * k, q, cam_m)
                        weight = decay ** k
                        canvas = canvas + ghost * weight
                finally:
                    self._trail_depth -= 1
            try:
                canvas = self._render_layer(canvas, layer, t_ms, q, cam_m)
            except Exception:
                # A blanked layer is invisible damage: during an export, surface
                # the failure instead of quietly writing black frames.
                if self.mode == "export":
                    raise
                continue
        if self._linear:
            canvas[..., :3] = _lin_to_srgb(canvas[..., :3])
        img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA")
        captions = (proj.data.get("captions") or [])
        if captions and self.mode != "preview-thumb":
            try:
                img = draw_captions(img, t_ms, captions, proj.data.get("caption_style") or {})
            except Exception:
                pass
        try:
            img = apply_grade(img, grade)
        except Exception:
            pass
        return img

    # -------------------------------------------------------------- camera / lights
    def _camera_matrix(self, t_ms: float, out_w: int, out_h: int) -> np.ndarray:
        proj = self.project
        cam = proj.data.get("camera") or {}
        # A camera *layer* overrides the project camera when present and visible.
        cam_layer = None
        for layer in proj.layers:
            if layer.get("kind") == "camera" and not layer.get("hidden"):
                if int(layer.get("start", 0)) <= t_ms < int(layer.get("end", proj.duration)):
                    cam_layer = layer
                    break
        self._cam_center = (proj.width / 2.0, proj.height / 2.0)
        enabled = bool(cam.get("enabled")) or cam_layer is not None
        if not enabled:
            return np.eye(3)
        if cam_layer is not None:
            pos = eval_track(cam_layer, "pos", t_ms, [proj.width / 2.0, proj.height / 2.0], proj)
            cpos = [pos[0], pos[1]]
            zoom = float(eval_track(cam_layer, "scale", t_ms, [float(cam_layer.get("zoom", 1.0))], proj)[0])
            rot = float(eval_track(cam_layer, "rot", t_ms, [0], proj)[0])
        else:
            cx = cam.get("x")
            cy = cam.get("y")
            cpos = [proj.width / 2.0 if cx is None else float(cx),
                    proj.height / 2.0 if cy is None else float(cy)]
            zoom = float(cam.get("zoom", 1.0) or 1.0)
            rot = float(cam.get("rotation", 0.0) or 0.0)
        if zoom <= 0.0001:
            zoom = 0.0001
        q = self._cam_scale
        self._cam_center = (cpos[0], cpos[1])
        # world -> screen (project px), then to output px
        m = (mat_translate(proj.width / 2.0, proj.height / 2.0)
             @ mat_scale(zoom, zoom)
             @ mat_rotate(-rot)
             @ mat_translate(-cpos[0], -cpos[1]))
        return mat_scale(q, q) @ m @ mat_scale(1.0 / q, 1.0 / q)

    def _collect_lights(self, t_ms: float) -> List[dict]:
        proj = self.project
        lights = []
        for layer in proj.layers:
            if layer.get("kind") != "light" or layer.get("hidden"):
                continue
            if not (int(layer.get("start", 0)) <= t_ms < int(layer.get("end", proj.duration))):
                continue
            pos = eval_track(layer, "pos", t_ms, [proj.width / 2.0, proj.height / 2.0], proj)
            base_intensity = float(layer.get("intensity", 0.6) or 0.6)
            ikeys = (layer.get("keys") or {}).get("intensity")
            if ikeys:
                base_intensity = float(kf_eval(ikeys, t_ms, [base_intensity])[0])
            lights.append({
                "x": float(pos[0]), "y": float(pos[1]),
                "z": float(pos[2]) if len(pos) > 2 else float(layer.get("z", 0) or 0),
                "radius": float(layer.get("radius", 900) or 900),
                "intensity": clamp(base_intensity, 0.0, 2.0),
                "color": parse_color(layer.get("color", "#FFFFFFFF")),
            })
        return lights

    def _apply_lights(self, src: np.ndarray, layer: dict) -> np.ndarray:
        lights = getattr(self, "_lights", None)
        if not lights or layer.get("lit", True) is False:
            return src
        h, w = src.shape[:2]
        q = getattr(self, "_cam_scale", 1.0)
        out = src.copy()
        rr = None
        yy = xx = None
        for light in lights:
            if rr is None:
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                rr = None
            lx, ly = light["x"] * q, light["y"] * q
            radius = max(1.0, light["radius"] * q)
            d2 = (xx - lx) ** 2 + (yy - ly) ** 2
            fall = np.exp(-d2 / (2.0 * radius * radius)).astype(np.float32)
            k = fall * float(light["intensity"]) * 0.55
            col = light["color"]
            for i in range(3):
                out[..., i] += k * col[i] * src[..., 3]
            out[..., 3:4] = np.maximum(out[..., 3:4], src[..., 3:4])
        return np.clip(out, 0, 1)

    # -------------------------------------------------------------- layer
    def _matrix(self, layer: dict, t_ms: float, q: float, seen=None) -> np.ndarray:
        t = t_ms
        pos = eval_track(layer, "pos", t, [0, 0, 0], self.project)
        anchor = eval_track(layer, "anchor", t, [0, 0], self.project)
        scale = eval_track(layer, "scale", t, [1, 1], self.project)
        rot = eval_track(layer, "rot", t, [0], self.project)[0]
        skew = eval_track(layer, "skew", t, [0, 0], self.project)
        m = (mat_translate(pos[0], pos[1])
             @ mat_rotate(rot)
             @ mat_skew(skew[0], skew[1])
             @ mat_scale(scale[0], scale[1])
             @ mat_translate(-anchor[0], -anchor[1]))
        parent_id = layer.get("parent")
        if parent_id is not None:
            seen = seen or set()
            if parent_id not in seen:
                seen.add(parent_id)
                parent = self.project.get_layer(parent_id)
                if parent is not None:
                    m = self._matrix(parent, t_ms, q, seen) @ m
        return m

    def _layer_time(self, layer: dict, t_ms: float) -> float:
        return t_ms - int(layer.get("start", 0))

    def _extent(self, layer: dict, t_ms: float) -> float:
        start = int(layer.get("start", 0))
        end = int(layer.get("end", self.project.duration))
        span = max(1, end - start)
        return clamp((t_ms - start) / span, 0.0, 1.0)

    # -------------------------------------------------------------- sprite
    def _media_item(self, layer: dict) -> Optional[dict]:
        mid = layer.get("media_id")
        item = self.project.get_media(mid) if mid is not None else None
        if item is None:
            path = layer.get("path")
            if path:
                return {"kind": "video" if os.path.splitext(path)[1].lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi") else "image",
                        "path": path, "duration_ms": layer.get("end", 999999), "has_audio": False}
        return item

    def _fit_factor(self, mw: int, mh: int, layer: dict) -> float:
        W, H = self.project.width, self.project.height
        fit = layer.get("fit", "cover")
        if not mw or not mh:
            return 1.0
        if fit == "cover":
            return max(W / mw, H / mh)
        if fit == "contain":
            return min(W / mw, H / mh)
        if fit == "stretch":
            return -1.0
        return 1.0

    def _sprite(self, layer: dict, t_ms: float, q: float) -> Optional[Tuple[Image.Image, bool]]:
        """Returns (sprite, centered) in sprite pixels; centered=True draws center at origin."""
        kind = layer.get("kind")
        W, H = self.project.width, self.project.height

        if kind == "solid":
            w = int(layer.get("w") or W)
            h = int(layer.get("h") or H)
            return self._color_sprite(layer, w, h, q), True

        if kind == "shape":
            w = max(2, int(float(layer.get("w", 400))))
            h = max(2, int(float(layer.get("h", 400))))
            src = layer
            if (layer.get("keys") or {}).get("trim"):
                src = dict(layer)
                tr = dict(layer.get("trim") or {})
                vals = kf_eval(layer["keys"]["trim"], t_ms, [tr.get("start", 0), tr.get("end", 100)])
                tr["start"], tr["end"] = float(vals[0]), float(vals[1])
                src["trim"] = tr
            return self._shape_sprite(src, w, h, q), True

        if kind == "text":
            if not str(layer.get("text", "")).strip() and layer.get("effects"):
                # an empty text layer carrying effects (spectrum analysers, glows)
                # still needs a full-frame sprite to draw into
                w, h = max(2, int(round(W * q))), max(2, int(round(H * q)))
                return Image.new("RGBA", (w, h), (0, 0, 0, 0)), True
            animators = layer.get("text_animators") or []
            if animators:
                return self._text_sprite_animated(layer, q, self._layer_time(layer, t_ms)), True
            return self._text_sprite(layer, q), True

        if kind in ("video", "image"):
            item = self._media_item(layer)
            if item is None:
                return None
            mw = int(item.get("width") or 0)
            mh = int(item.get("height") or 0)
            if mw <= 0 or mh <= 0:
                return None
            factor = self._fit_factor(mw, mh, layer)
            if factor < 0:
                tw, th = W, H
            else:
                tw, th = max(2, int(mw * factor)), max(2, int(mh * factor))
            speed = float(layer.get("speed", 1.0) or 1.0)
            local = self._layer_time(layer, t_ms) * speed + float(layer.get("source_in", 0))
            disp_w = max(2, int(round(tw * q)))
            disp_h = max(2, int(round(th * q)))
            if kind == "video":
                frame = self._video_frame(layer, item, local, (disp_w, disp_h))
            else:
                frame = self.media.get_frame(item, 0)
            if frame is None:
                return None
            if frame.size != (disp_w, disp_h):
                frame = frame.resize((disp_w, disp_h), Image.LANCZOS)
            return frame.convert("RGBA"), True

        if kind == "adjustment":
            return None  # handled specially

        return None

    def _video_frame(self, layer: dict, item: dict, local_ms: float, size: Tuple[int, int]) -> Optional[Image.Image]:
        if self.mode == "export":
            lid = int(layer.get("id", 0))
            src = self.video_sources.get(lid)
            if src is None or src.path != (self.media.proxy_path(item) or item.get("path")):
                src = VideoFrameSource(self.media, item)
                self.video_sources[lid] = src
                # Every open source is a live ffmpeg process (~40 MB+). A montage
                # with a dozen clips would otherwise exhaust RAM, so keep only a
                # few decoders alive and close the least recently used one.
                max_sources = int(os.environ.get("CRAFTMOTION_MAX_VIDEO_SOURCES", "3"))
                while len(self.video_sources) > max_sources:
                    oldest = next(iter(self.video_sources))
                    self.video_sources.pop(oldest)._close()
            frame = src.read(local_ms)
            if lid in self.video_sources:
                self.video_sources.move_to_end(lid)
            if frame is not None:
                if frame.size != size:
                    frame = frame.resize(size, Image.LANCZOS)
                return frame
        return self.media.get_frame(item, local_ms, max_size=size)

    # -------------------------------------------------------------- sprites by kind
    def _color_sprite(self, layer: dict, w: int, h: int, q: float) -> Image.Image:
        sw = max(2, int(round(w * q)))
        sh = max(2, int(round(h * q)))
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        radius = float(layer.get("radius", 0) or 0)
        col = _pil_col(layer.get("color", "#FF888888"))
        if layer.get("shape") == "ellipse":
            draw.ellipse((0, 0, sw - 1, sh - 1), fill=col)
        elif radius > 0:
            draw.rounded_rectangle((0, 0, sw - 1, sh - 1), radius=radius * q, fill=col)
        else:
            draw.rectangle((0, 0, sw, sh), fill=col)
        return self._maybe_gradient(img, layer, q)

    def _maybe_gradient(self, img: Image.Image, layer: dict, q: float) -> Image.Image:
        c2 = layer.get("color2")
        if not c2:
            return img
        w, h = img.size
        c1 = parse_color(layer.get("color", "#FFFFFFFF"))
        cc2 = parse_color(c2)
        ang = float(layer.get("gradient_angle", 90) or 90)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rad = math.radians(ang)
        t = (xx / max(1, w)) * math.cos(rad) + (yy / max(1, h)) * math.sin(rad)
        t = (t - t.min()) / max(1e-5, t.max() - t.min())
        grad = np.zeros((h, w, 4), np.float32)
        for i in range(4):
            grad[..., i] = c1[i] + (cc2[i] - c1[i]) * t
        base = np.asarray(img, np.float32) / 255.0
        alpha = base[..., 3:4]
        out = grad.copy()
        out[..., 3:4] = alpha
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")

    def _shape_sprite(self, layer: dict, w: int, h: int, q: float) -> Image.Image:
        sw = max(2, int(round(w * q)))
        sh = max(2, int(round(h * q)))
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill = _pil_col(layer.get("color", "#FF4C8DFF"))
        stroke_c = _pil_col(layer.get("stroke_color", "#00000000"))
        stroke_w = float(layer.get("stroke_width", 0) or 0) * q
        shape = (layer.get("shape") or "rect").lower()
        pad = max(1.0, stroke_w / 2.0)
        box = (pad, pad, sw - 1 - pad, sh - 1 - pad)
        cx, cy = sw / 2.0, sh / 2.0
        rx, ry = (sw - 1) / 2.0 - pad, (sh - 1) / 2.0 - pad

        if shape in ("rect", "rectangle", "roundrect", "rounded_rect"):
            r = float(layer.get("radius", 0) or 0) * q
            if r > 0:
                draw.rounded_rectangle(box, radius=r, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
            else:
                draw.rectangle(box, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("ellipse", "circle", "oval"):
            draw.ellipse(box, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("triangle",):
            pts = [(cx, pad), (sw - 1 - pad, sh - 1 - pad), (pad, sh - 1 - pad)]
            draw.polygon(pts, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("star",):
            n = max(3, int(layer.get("points", 5)))
            inner = clamp(float(layer.get("inner", 0.5)), 0.05, 0.95)
            pts = []
            for i in range(n * 2):
                ang = -math.pi / 2 + i * math.pi / n
                r = 1.0 if i % 2 == 0 else inner
                pts.append((cx + math.cos(ang) * rx * r, cy + math.sin(ang) * ry * r))
            draw.polygon(pts, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("polygon", "poly"):
            n = max(3, int(layer.get("points", 6)))
            pts = [(cx + math.cos(-math.pi / 2 + i * 2 * math.pi / n) * rx,
                    cy + math.sin(-math.pi / 2 + i * 2 * math.pi / n) * ry) for i in range(n)]
            draw.polygon(pts, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("diamond",):
            draw.polygon([(cx, pad), (sw - 1 - pad, cy), (cx, sh - 1 - pad), (pad, cy)], fill=fill,
                         outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("heart",):
            pts = []
            for i in range(120):
                t = i / 119 * 2 * math.pi
                x = 16 * math.sin(t) ** 3
                y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
                pts.append((cx + x / 18 * rx, cy - y / 18 * ry))
            draw.polygon(pts, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
        elif shape in ("plus", "cross"):
            t = min(sw, sh) * 0.28
            pts = [(cx - t / 2, pad), (cx + t / 2, pad), (cx + t / 2, cy - t / 2), (sw - 1 - pad, cy - t / 2),
                   (sw - 1 - pad, cy + t / 2), (cx + t / 2, cy + t / 2), (cx + t / 2, sh - 1 - pad),
                   (cx - t / 2, sh - 1 - pad), (cx - t / 2, cy + t / 2), (pad, cy + t / 2),
                   (pad, cy - t / 2), (cx - t / 2, cy - t / 2)]
            draw.polygon(pts, fill=fill)
        elif shape in ("line",):
            draw.rectangle((pad, cy - max(1, stroke_w or 2) / 2, sw - 1 - pad, cy + max(1, stroke_w or 2) / 2), fill=fill)
        elif shape in ("ring",):
            draw.ellipse(box, outline=fill, width=max(1, int(stroke_w or max(2, min(sw, sh) * 0.08))))
        elif shape in ("path", "freehand"):
            pts = layer.get("path") or []
            trim = layer.get("trim") or None
            if trim and len(pts) >= 2:
                ts = float(trim.get("start", 0) or 0)
                te = float(trim.get("end", 100) or 100)
                off = float(trim.get("offset", 0) or 0)
                ts = (ts + off) % 100.0
                te = (te + off) % 100.0
                if te <= ts:
                    te = ts if abs(te - ts) < 1e-6 else te + 100.0
                pts, _closed = trim_path_points(pts, ts, min(100.0, te))
            if len(pts) >= 2:
                scaled = [(pad + float(px) * (sw - 2 * pad), pad + float(py) * (sh - 2 * pad)) for px, py in pts]
                if layer.get("closed", True) and not trim:
                    draw.polygon(scaled, fill=fill, outline=stroke_c if stroke_w > 0 else None, width=int(stroke_w))
                else:
                    draw.line(scaled, fill=fill if not trim else (stroke_c if stroke_w > 0 else fill),
                              width=max(1, int(stroke_w or max(2.0, min(sw, sh) * 0.03))), joint="curve")
        else:
            draw.rectangle(box, fill=fill)
        img = self._maybe_gradient(img, layer, q)
        rep = layer.get("repeater") or None
        if rep:
            img = self._apply_repeater(img, rep)
        return img

    def _apply_repeater(self, base: Image.Image, rep: dict) -> Image.Image:
        copies = int(rep.get("copies", 5) or 5)
        if copies <= 1:
            return base
        copies = min(copies, 200)
        ox = float(rep.get("x", 0) or 0)
        oy = float(rep.get("y", 0) or 0)
        rot = float(rep.get("rotation", 0) or 0)
        scale = float(rep.get("scale", 1.0) or 1.0)
        start_op = float(rep.get("start_opacity", 1.0) if rep.get("start_opacity") is not None else 1.0)
        end_op = float(rep.get("end_opacity", 1.0) if rep.get("end_opacity") is not None else 1.0)
        bw, bh = base.size
        maxd = max(abs(ox), abs(oy)) * (copies - 1)
        grow = max(0.0, abs(scale - 1.0)) * max(bw, bh) * copies
        pad = int(maxd + grow + 4)
        out = Image.new("RGBA", (bw + 2 * pad, bh + 2 * pad), (0, 0, 0, 0))
        for i in range(copies):
            op = start_op + (end_op - start_op) * (i / max(1, copies - 1))
            if op <= 0.001:
                continue
            im = base
            if abs(scale - 1.0) > 1e-6:
                k = scale ** i
                im = im.resize((max(1, int(bw * k)), max(1, int(bh * k))), Image.BILINEAR)
            if abs(rot) > 1e-6:
                im = im.rotate(rot * i, resample=Image.BICUBIC, expand=True)
            if op < 0.999:
                a = im.getchannel("A").point(lambda v: int(v * op))
                im = im.copy()
                im.putalpha(a)
            x = pad + (im.size[0] - bw) // 2 + int(round(ox * i))
            y = pad + (im.size[1] - bh) // 2 + int(round(oy * i))
            out.alpha_composite(im, (x, y))
        return out

    def _text_sprite(self, layer: dict, q: float) -> Image.Image:
        size = max(4.0, float(layer.get("size", 96))) * q
        bold = bool(layer.get("bold")) or "bold" in str(layer.get("font", "")).lower()
        font = load_font(int(round(size)), str(layer.get("font", "")), bold=bold)
        text = str(layer.get("text", ""))
        wrap = float(layer.get("wrap", 900)) * q
        line_h = float(layer.get("line_height", 1.15))
        align = str(layer.get("align", "center")).lower()

        lines: List[str] = []
        for raw in text.split("\n"):
            if not raw:
                lines.append("")
                continue
            words = raw.split(" ")
            cur = ""
            for wd in words:
                cand = (cur + " " + wd).strip()
                if _text_len(font, cand) <= wrap or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = wd
            lines.append(cur)
        if not lines:
            lines = [""]

        widths = [_text_len(font, ln) for ln in lines]
        line_h_px = size * line_h
        pad = size * 0.35 + float(layer.get("stroke_width", 0)) * q + 6
        w = max(widths + [1]) + pad * 2
        h = line_h_px * len(lines) + pad * 2
        sw, sh = max(2, int(math.ceil(w))), max(2, int(math.ceil(h)))
        img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        fill = _pil_col(layer.get("color", "#FFFFFFFF"))
        stroke_w = float(layer.get("stroke_width", 0) or 0) * q
        stroke_c = _pil_col(layer.get("stroke_color", "#FF000000"))
        shadow = float(layer.get("shadow", 0) or 0) * q
        y = pad
        for ln, lw in zip(lines, widths):
            if align == "left":
                x = pad
            elif align == "right":
                x = sw - pad - lw
            else:
                x = (sw - lw) / 2.0
            if shadow > 0:
                sh_img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
                sd = ImageDraw.Draw(sh_img)
                sd.text((x + shadow * 0.4, y + shadow * 0.4), ln, font=font, fill=(0, 0, 0, 170))
                sh_img = sh_img.filter(ImageFilter.GaussianBlur(shadow))
                img.alpha_composite(sh_img)
                draw = ImageDraw.Draw(img)
            if stroke_w > 0:
                draw.text((x, y), ln, font=font, fill=fill, stroke_width=int(stroke_w), stroke_fill=stroke_c)
            else:
                draw.text((x, y), ln, font=font, fill=fill)
            y += line_h_px
        return img

    # -------------------------------------------------------------- animated text
    def _animator_amount(self, animator: dict, name: str, t_local: float, default=0.0) -> float:
        keys = (animator.get("keys") or {}).get(name)
        if keys:
            try:
                return float(kf_eval(keys, t_local, [default])[0])
            except Exception:
                return float(default)
        val = (animator.get("params") or {}).get(name, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def _text_sprite_animated(self, layer: dict, q: float, t_local: float) -> Image.Image:
        size = max(4.0, float(layer.get("size", 96))) * q
        bold = bool(layer.get("bold")) or "bold" in str(layer.get("font", "")).lower()
        font = load_font(int(round(size)), str(layer.get("font", "")), bold=bold)
        text = str(layer.get("text", ""))
        wrap = float(layer.get("wrap", 900)) * q
        line_h = float(layer.get("line_height", 1.15))
        align = str(layer.get("align", "center")).lower()
        tracking = float(layer.get("tracking", 0) or 0) * q
        base_fill = _pil_col(layer.get("color", "#FFFFFFFF"))
        stroke_w = float(layer.get("stroke_width", 0) or 0) * q
        stroke_c = _pil_col(layer.get("stroke_color", "#FF000000"))
        animators = layer.get("text_animators") or []

        # split into wrapped lines
        lines: List[str] = []
        for raw in text.split("\n"):
            if not raw:
                lines.append("")
                continue
            cur = ""
            for wd in raw.split(" "):
                cand = (cur + " " + wd).strip()
                if _text_len(font, cand) + tracking * len(cand) <= wrap or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = wd
            lines.append(cur)
        if not lines:
            lines = [""]

        line_metrics = []
        for ln in lines:
            adv = []
            xpos = 0.0
            for ch in ln:
                adv.append(xpos)
                try:
                    xpos += float(font.getlength(ch))
                except Exception:
                    xpos += size * 0.6
                xpos += tracking
            total = xpos - tracking if ln else 0.0
            line_metrics.append({"adv": adv, "width": total})

        line_h_px = size * line_h
        max_w = max([m["width"] for m in line_metrics] + [1.0])
        # room for the largest animator displacement / scale so glyphs are never clipped
        extra = 0.0
        for animator in animators:
            for kname in ("x", "y"):
                keys = (animator.get("keys") or {}).get(kname)
                if keys:
                    try:
                        extra = max(extra, max(abs(float(k["v"][0])) for k in keys) * q)
                    except Exception:
                        pass
                else:
                    try:
                        extra = max(extra, abs(float((animator.get("params") or {}).get(kname, 0) or 0)) * q)
                    except Exception:
                        pass
        pad = size * 0.8 + stroke_w + 8 + extra
        width = int(math.ceil(max_w + pad * 2 + 2))
        height = int(math.ceil(line_h_px * len(lines) + pad * 2 + 2))
        img = Image.new("RGBA", (max(2, width), max(2, height)), (0, 0, 0, 0))

        # global character index for range selectors
        char_index = 0
        total_chars = sum(len(ln) for ln in lines)
        for li, (ln, met) in enumerate(zip(lines, line_metrics)):
            if align == "left":
                x0 = pad
            elif align == "right":
                x0 = img.size[0] - pad - met["width"]
            else:
                x0 = (img.size[0] - met["width"]) / 2.0
            y0 = pad + li * line_h_px
            word_idx = -1
            prev_space = True
            for ci, ch in enumerate(ln):
                if ch == " " or prev_space:
                    word_idx += 1
                prev_space = (ch == " ")
                if ch.strip() == "" and ch != " ":
                    char_index += 1
                    continue
                dx = dy = rot = 0.0
                sc = 1.0
                op = 1.0
                blur = 0.0
                col = base_fill
                for animator in animators:
                    if animator.get("enabled", True) is False:
                        continue
                    sel = dict(animator.get("selector") or {})
                    sel["_line_len"] = max(1, len(ln))
                    # selector start/end can be animated (typewriter, wipes)
                    sel_keys = sel.get("keys") or {}
                    for skey in ("start", "end", "offset"):
                        keys = sel_keys.get(skey)
                        if keys:
                            try:
                                sel[skey] = float(kf_eval(keys, t_local, [sel.get(skey, 0)])[0])
                            except Exception:
                                pass
                    stype = (sel.get("type") or "range").lower()
                    if stype in ("word", "words"):
                        frac = word_idx / max(1, max(1, ln.count(" ") + 1) - 1)
                        lo = float(sel.get("start", 0)) / 100.0
                        hi = float(sel.get("end", 100)) / 100.0
                        w = 1.0 if min(lo, hi) <= frac <= max(lo, hi) else 0.0
                    elif stype in ("line", "lines"):
                        frac = li / max(1, len(lines) - 1)
                        lo = float(sel.get("start", 0)) / 100.0
                        hi = float(sel.get("end", 100)) / 100.0
                        w = 1.0 if min(lo, hi) <= frac <= max(lo, hi) else 0.0
                    else:
                        w = selector_weight({**animator, "selector": sel}, char_index, total_chars,
                                            {"col": ci, "line": li})
                    if w <= 0.001:
                        continue
                    a = animator.get("animator") or animator.get("params") or {}
                    dx += w * self._animator_amount(animator, "x", t_local)
                    dy += w * self._animator_amount(animator, "y", t_local)
                    rot += w * self._animator_amount(animator, "rotation", t_local)
                    sc *= 1.0 + w * (self._animator_amount(animator, "scale", t_local, 100.0) / 100.0 - 1.0)
                    op *= max(0.0, 1.0 - w * (1.0 - self._animator_amount(animator, "opacity", t_local, 100.0) / 100.0))
                    blur += w * self._animator_amount(animator, "blur", t_local)
                    a_col = a.get("color")
                    if a_col:
                        cr = parse_color(a_col)
                        col = (
                            int(base_fill[0] + (cr[0] * 255 - base_fill[0]) * w),
                            int(base_fill[1] + (cr[1] * 255 - base_fill[1]) * w),
                            int(base_fill[2] + (cr[2] * 255 - base_fill[2]) * w),
                            int(base_fill[3] + (cr[3] * 255 - base_fill[3]) * w),
                        )
                try:
                    cw = int(math.ceil(float(font.getlength(ch)))) + int(stroke_w * 2) + 8
                except Exception:
                    cw = int(size) + int(stroke_w * 2) + 8
                ch_h = int(math.ceil(size * 1.6)) + int(stroke_w * 2) + 8
                glyph = Image.new("RGBA", (max(2, cw), max(2, ch_h)), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glyph)
                if stroke_w > 0:
                    gd.text((4 + stroke_w, 4 + stroke_w), ch, font=font, fill=col,
                            stroke_width=int(stroke_w), stroke_fill=stroke_c)
                else:
                    gd.text((4 + stroke_w, 4 + stroke_w), ch, font=font, fill=col)
                if blur > 0.1:
                    glyph = glyph.filter(ImageFilter.GaussianBlur(blur * q))
                if op < 0.999:
                    glyph.putalpha(glyph.getchannel("A").point(lambda v: int(v * op)))
                if abs(rot) > 1e-3 or abs(sc - 1.0) > 1e-3:
                    nw = max(1, int(glyph.size[0] * sc))
                    nh = max(1, int(glyph.size[1] * sc))
                    glyph = glyph.resize((nw, nh), Image.BICUBIC)
                    if abs(rot) > 1e-3:
                        glyph = glyph.rotate(rot, resample=Image.BICUBIC, expand=True)
                px = int(round(x0 + met["adv"][ci] + dx - 4)) - (glyph.size[0] - cw) // 2
                py = int(round(y0 + dy - 4)) - (glyph.size[1] - ch_h) // 2
                img.paste(glyph, (px, py), glyph)
                char_index += 1
        return img

    # -------------------------------------------------------------- audio
    def _audio_key(self) -> tuple:
        proj = self.project
        data = proj.data
        audio_layers = tuple(
            (l.get("id"), l.get("start"), l.get("end"), l.get("volume"), l.get("mute"),
             len((l.get("keys") or {}).get("volume") or []), len(l.get("audio_fx") or []))
            for l in proj.layers if l.get("kind") == "audio")
        master = data.get("audio_master") or {}
        return (tuple((m.get("id"), m.get("path")) for m in data.get("media") or []),
                audio_layers, tuple(sorted(master.items())), int(proj.duration))

    def audio_state(self, t_ms: float) -> dict:
        """Level / band / spectrum / waveform at t_ms (cached analysis)."""
        if self.media is None:
            return {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "wave": [], "spectrum": []}
        key = self._audio_key()
        if self._audio_cache is None or self._audio_cache[0] != key:
            try:
                env = audio_mod.envelope(self.project, self.media, 40.0)
                spec = audio_mod.spectrogram(self.project, self.media, 40.0, 24)
            except Exception:
                env, spec = {}, {}
            self._audio_cache = (key, env, spec)
        _, env, spec = self._audio_cache
        state = audio_mod.envelope_at(env, t_ms) if env else {
            "level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0}
        state["spectrum"] = audio_mod.spectrum_at(spec, t_ms) if spec else []
        if not state["spectrum"]:
            # fall back to the three bands so visualisers still respond
            state["spectrum"] = [state.get("low", 0.0), state.get("mid", 0.0),
                                 state.get("high", 0.0)] * 8
        try:
            state["wave"] = audio_mod.wave_slice(self.project, self.media, t_ms, 80.0, 96)
        except Exception:
            state["wave"] = []
        return state

    # -------------------------------------------------------------- effects
    def _apply_effect_stack(self, img: Image.Image, layer: dict, t_ms: float, q: float,
                            extent: float) -> Image.Image:
        stack = layer.get("effects") or []
        if not stack:
            return img
        local = self._layer_time(layer, t_ms)
        ctx = {"project": self.project, "layer": layer, "t_local": local,
               "extent": extent, "scale": q, "renderer": self,
               "audio": self.audio_state(t_ms)}
        out = img
        for eff in stack:
            if not eff.get("enabled", True):
                continue
            etype = FX.resolve_type(eff.get("type", ""))
            if not etype:
                continue
            spec = FX.EFFECTS.get(etype)
            params = {}
            if spec:
                for name, prm in spec.params.items():
                    val = eval_param(self.project, layer, eff, name, local, prm.default)
                    if prm.kind == "float":
                        try:
                            val = float(val)
                        except (TypeError, ValueError):
                            val = float(prm.default)
                        if name in PIXEL_PARAMS.get(etype, []):
                            val *= q
                    params[name] = val
            out = FX.apply_effect(etype, out, params, ctx)
        return out

    # -------------------------------------------------------------- compositing
    def _precomp_sprite(self, layer: dict, t_ms: float, q: float) -> Optional[Image.Image]:
        comp_id = layer.get("comp_id")
        comp = self.project.get_comp(comp_id) if comp_id is not None else None
        if comp is None:
            return None
        if int(comp_id) in self._precomp_stack or len(self._precomp_stack) > 6:
            return None
        local = float(self._layer_time(layer, t_ms)) * float(layer.get("speed", 1.0) or 1.0)
        local += float(layer.get("source_in", 0) or 0)
        if local < 0 or local >= float(comp.get("duration") or self.project.duration):
            return None
        tw = max(2, int(round(float(comp.get("width") or self.project.width) * q)))
        th = max(2, int(round(float(comp.get("height") or self.project.height) * q)))
        self._precomp_stack.append(int(comp_id))
        try:
            sub = Renderer(self.project.comp_as_project(comp), self.media, self.mode)
            sub._in_subcomp = True
            frame = sub.render(local, (tw, th))
            sub.close()
        finally:
            self._precomp_stack.pop()
        return frame.convert("RGBA")

    def _render_layer(self, canvas: np.ndarray, layer: dict, t_ms: float, q: float,
                      cam_m: Optional[np.ndarray] = None) -> np.ndarray:
        kind = layer.get("kind")
        opacity = clamp(float(eval_track(layer, "opacity", t_ms, [1.0], self.project)[0]), 0.0, 1.0)
        if opacity <= 0.001:
            return canvas
        out_h, out_w = canvas.shape[:2]
        m = self._matrix(layer, t_ms, q)
        # Sprites are rasterised at preview resolution, so sprite pixels are
        # q x project pixels: linear part unchanged, translation scaled by q.
        mq = mat_scale(q, q) @ m @ mat_scale(1.0 / q, 1.0 / q)

        # 2.5D: depth parallax under the camera (scale about the camera centre)
        if cam_m is not None and not np.allclose(cam_m, np.eye(3)):
            depth = float(layer.get("z", 0) or 0)
            dscale = 1.0 if abs(depth) < 1e-6 else 1.0 / (1.0 + max(-900.0, depth) / 1000.0)
            cx, cy = getattr(self, "_cam_center", (self.project.width / 2.0, self.project.height / 2.0))
            world = (mat_translate(cx, cy) @ mat_scale(dscale, dscale) @ mat_translate(-cx, -cy))
            mq = cam_m @ mat_scale(q, q) @ world @ m @ mat_scale(1.0 / q, 1.0 / q)

        extent = self._extent(layer, t_ms)

        if kind == "adjustment":
            base = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA")
            processed = self._apply_effect_stack(base, layer, t_ms, q, extent)
            src = _np_rgba(processed)
            return _composite(canvas, src, opacity, layer.get("blend", "normal"), 1.0)

        src_layer = layer
        if (layer.get("keys") or {}).get("color"):
            src_layer = dict(layer)
            src_layer["color"] = to_argb(eval_track(layer, "color", t_ms, [1, 1, 1, 1], self.project))

        if kind == "precomp":
            frame = self._precomp_sprite(layer, t_ms, q)
            if frame is None:
                return canvas
            img, centered = frame, True
            # fit into the layer box
            mw, mh = img.size
            if mw and mh:
                layer_box = (float(layer.get("w") or self.project.width), float(layer.get("h") or self.project.height))
                fit = layer.get("fit", "stretch")
                if fit == "cover":
                    k = max(layer_box[0] * q / mw, layer_box[1] * q / mh)
                elif fit == "contain":
                    k = min(layer_box[0] * q / mw, layer_box[1] * q / mh)
                elif fit == "none":
                    k = 1.0
                else:
                    img = img.resize((max(2, int(layer_box[0] * q)), max(2, int(layer_box[1] * q))), Image.LANCZOS)
                    k = 1.0
                if abs(k - 1.0) > 1e-6 and k > 0:
                    img = img.resize((max(2, int(mw * k)), max(2, int(mh * k))), Image.LANCZOS)
        else:
            sprite = self._sprite(src_layer, t_ms, q)
            if sprite is None:
                return canvas
            img, centered = sprite

        matte = layer.get("matte")
        processed = self._apply_effect_stack(img, layer, t_ms, q, extent)

        masks = layer.get("masks") or []
        if masks:
            processed = apply_masks(processed, masks)
        styles = layer.get("styles") or []
        if styles:
            processed = apply_layer_styles(processed, styles, q)

        if self._linear:
            arr = _np_rgba(processed)
            arr[..., :3] = _srgb_to_lin(arr[..., :3])
            processed = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")
        src = self._warp_to_canvas(processed, mq, (out_w, out_h), centered)
        src = self._apply_lights(src, layer)

        if matte is not None:
            m_layer = self.project.get_layer(matte)
            if m_layer is not None and not m_layer.get("hidden"):
                msprite = self._sprite(m_layer, t_ms, q)
                if msprite is not None:
                    mm_ = self._matrix(m_layer, t_ms, q)
                    mmq = mat_scale(q, q) @ mm_ @ mat_scale(1.0 / q, 1.0 / q)
                    if cam_m is not None and not np.allclose(cam_m, np.eye(3)):
                        mmq = cam_m @ mmq
                    mimg = self._warp_to_canvas(msprite[0], mmq, (out_w, out_h), msprite[1])
                    src[..., 3:4] *= mimg[..., 3:4]

        comp = _composite(canvas, src, opacity, layer.get("blend", "normal"), 1.0)

        tr_in = layer.get("transition_in") or None
        tr_out = layer.get("transition_out") or None
        start = int(layer.get("start", 0))
        end = int(layer.get("end", self.project.duration))
        if tr_in and tr_in.get("type"):
            dur = max(1.0, float(tr_in.get("duration", 500) or 500))
            if start <= t_ms < start + dur:
                prog = (t_ms - start) / dur
                base = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA")
                incoming = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8), "RGBA")
                comp = _np_rgba(apply_transition(tr_in["type"], base, incoming, prog,
                                                  tr_in.get("params") or {}))
        elif tr_out and tr_out.get("type"):
            dur = max(1.0, float(tr_out.get("duration", 500) or 500))
            if end - dur <= t_ms < end:
                prog = (t_ms - (end - dur)) / dur
                base = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA")
                outgoing = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8), "RGBA")
                comp = _np_rgba(apply_transition(tr_out["type"], outgoing, base, prog,
                                                  tr_out.get("params") or {}))
        return comp

    def _warp_to_canvas(self, img: Image.Image, mq: np.ndarray, size: Tuple[int, int],
                        centered: bool) -> np.ndarray:
        out_w, out_h = size
        sw, sh = img.size
        # mq maps layer-local coords -> output. The sprite's local origin is its
        # centre, which in sprite pixels sits at (sw/2, sh/2).
        try:
            inv_mq = np.linalg.inv(mq)
        except np.linalg.LinAlgError:
            return np.zeros((out_h, out_w, 4), np.float32)
        if centered:
            inv = mat_translate(sw / 2.0, sh / 2.0) @ inv_mq
        else:
            inv = inv_mq
        if not np.isfinite(inv).all():
            return np.zeros((out_h, out_w, 4), np.float32)
        a, b, c = inv[0, 0], inv[0, 1], inv[0, 2]
        d, e, f = inv[1, 0], inv[1, 1], inv[1, 2]
        warped = img.transform((out_w, out_h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        return _np_rgba(warped)

    # -------------------------------------------------------------- misc
    def close(self):
        for src in self.video_sources.values():
            src._close()
        self.video_sources.clear()


def _srgb_to_lin(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _np_rgba(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0


def _text_len(font, s: str) -> float:
    try:
        return float(font.getlength(s))
    except Exception:
        return float(font.getbbox(s)[2])


# ------------------------------------------------------------------ blending


def _lum(c: np.ndarray) -> np.ndarray:
    return 0.3 * c[..., 0:1] + 0.59 * c[..., 1:2] + 0.11 * c[..., 2:3]


def _sat(c: np.ndarray) -> np.ndarray:
    return c.max(axis=-1, keepdims=True) - c.min(axis=-1, keepdims=True)


def _clip_color(c: np.ndarray) -> np.ndarray:
    l = _lum(c)
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    lo = np.minimum(n, 0.0)
    hi = np.maximum(x - 1.0, 0.0)
    denom = np.maximum(x - n, 1e-6)
    c = np.where(l <= 1.0, l + (c - l) * l / np.maximum(l - lo, 1e-6),
                 l + (c - l) * (1.0 - l) / np.maximum(hi, 1e-6))
    return np.clip(c, 0.0, 1.0)


def _set_lum(c: np.ndarray, l: np.ndarray) -> np.ndarray:
    d = l - _lum(c)
    return _clip_color(c + d)


def _set_sat(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """W3C SetSat: scale the channel spread so max-min == s, keeping the order."""
    cmin = c.min(axis=-1, keepdims=True)
    cmax = c.max(axis=-1, keepdims=True)
    mid = c - cmin
    spread = np.maximum(cmax - cmin, 1e-6)
    return np.clip(mid * s / spread, 0.0, 1.0)


def _blend_rgb(cb: np.ndarray, cs: np.ndarray, mode: str) -> np.ndarray:
    """38 blend modes: 27 separable + 4 non-separable (W3C) + 7 extended.

    `cb` is the backdrop, `cs` the source; both float RGB in 0..1.
    """
    mode = (mode or "normal").lower().replace("-", "_").replace(" ", "_")
    if mode in ("normal", ""):
        return cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return 1.0 - (1.0 - cb) * (1.0 - cs)
    if mode in ("linear_dodge", "add", "plus"):
        return np.clip(cb + cs, 0, 1)
    if mode == "subtract":
        return np.clip(cb - cs, 0, 1)
    if mode == "divide":
        return np.clip(cb / np.maximum(cs, 1e-3), 0, 1)
    if mode == "overlay":
        return np.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))
    if mode == "soft_light":
        d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(np.clip(cb, 0, 1)))
        return np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb),
                        cb + (2 * cs - 1) * (d - cb))
    if mode == "hard_light":
        return np.where(cs <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))
    if mode == "vivid_light":
        return np.where(cs <= 0.5, np.clip(1 - (1 - cb) / np.maximum(2 * cs, 1e-3), 0, 1),
                        np.clip(cb / np.maximum(2 * (1 - cs), 1e-3), 0, 1))
    if mode == "linear_light":
        return np.clip(cb + 2 * cs - 1, 0, 1)
    if mode == "pin_light":
        return np.where(cs <= 0.5, np.minimum(cb, 2 * cs), np.maximum(cb, 2 * cs - 1))
    if mode == "hard_mix":
        return np.where(cb + cs >= 1.0, 1.0, 0.0)
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "darker_color":
        return np.where(_lum(cb) <= _lum(cs), cb, cs)
    if mode == "lighter_color":
        return np.where(_lum(cb) >= _lum(cs), cb, cs)
    if mode == "color_dodge":
        return np.where(cs >= 1.0, 1.0, np.clip(cb / np.maximum(1e-3, 1 - cs), 0, 1))
    if mode == "color_burn":
        return np.where(cs <= 0.0, 0.0, 1 - np.clip((1 - cb) / np.maximum(1e-3, cs), 0, 1))
    if mode == "linear_burn":
        return np.clip(cb + cs - 1, 0, 1)
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "exclusion":
        return cb + cs - 2 * cb * cs
    if mode == "average":
        return (cb + cs) * 0.5
    if mode == "negation":
        return 1.0 - np.abs(1.0 - cb - cs)
    if mode == "phoenix":
        return np.clip(np.minimum(cb, cs) - np.maximum(cb, cs) + 1.0, 0, 1)
    if mode == "reflect":
        return np.where(cs >= 1.0, 1.0, np.clip(cb * cb / np.maximum(1 - cs, 1e-3), 0, 1))
    if mode == "glow":
        return np.where(cb >= 1.0, 1.0, np.clip(cs * cs / np.maximum(1 - cb, 1e-3), 0, 1))
    if mode == "freeze":
        return np.clip(1 - (1 - cs) ** 2 / np.maximum(1e-3, cb), 0, 1)
    if mode == "heat":
        return np.clip(1 - (1 - cb) ** 2 / np.maximum(1e-3, cs), 0, 1)
    if mode == "stamp":
        return np.clip(cb + 2 * cs - 1, 0, 1)
    if mode == "xor":
        return np.bitwise_xor((cb * 255).astype(np.uint8), (cs * 255).astype(np.uint8)).astype(np.float32) / 255.0
    if mode == "grain_extract":
        return np.clip(cb - cs + 0.5, 0, 1)
    if mode == "grain_merge":
        return np.clip(cb + cs - 0.5, 0, 1)
    # --- non-separable (W3C compositing spec) -------------------------------
    if mode == "hue":
        return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))
    if mode == "saturation":
        return _set_lum(_set_sat(cb, _sat(cs)), _lum(cb))
    if mode == "color":
        return _set_lum(cs, _lum(cb))
    if mode == "luminosity":
        return _set_lum(cb, _lum(cs))
    if mode == "dissolve":
        # pseudo-random per-pixel threshold; a static threshold makes it a hard
        # dither that still reads correctly when blended at partial coverage.
        h, w = cb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rnd = (np.sin(xx * 12.9898 + yy * 78.233) * 43758.5453) % 1.0
        rnd = rnd[..., None]
        return np.where(rnd <= 0.5, cs, cb)
    return cs


def _composite(canvas: np.ndarray, src: np.ndarray, opacity: float, blend: str,
               alpha_mult: float = 1.0) -> np.ndarray:
    h, w = canvas.shape[:2]
    sh, sw = src.shape[:2]
    if (sh, sw) != (h, w):
        pad = np.zeros((h, w, 4), np.float32)
        pad[:min(h, sh), :min(w, sw)] = src[:min(h, sh), :min(w, sw)]
        src = pad
    ab = canvas[..., 3:4]
    as_ = src[..., 3:4] * opacity * alpha_mult
    cb = canvas[..., :3]
    cs = src[..., :3]
    bl = _blend_rgb(cb, cs, blend)
    ao = as_ + ab * (1 - as_)
    co = as_ * (1 - ab) * cs + as_ * ab * bl + (1 - as_) * ab * cb
    out = np.zeros_like(canvas)
    safe = np.maximum(ao, 1e-6)
    out[..., :3] = co / safe
    out[..., 3:4] = ao
    return out


# ------------------------------------------------------------------ helpers

def render_frame(project: Project, t_ms: float, media: Optional[MediaManager] = None,
                 size: Optional[Tuple[int, int]] = None, renderer: Optional[Renderer] = None) -> Image.Image:
    r = renderer or Renderer(project, media, mode="preview")
    img = r.render(t_ms, size=size)
    if renderer is None:
        r.close()
    return img


def render_preview(project: Project, t_ms: float, media: Optional[MediaManager] = None,
                   max_dim: int = 640) -> Image.Image:
    W, H = project.width, project.height
    scale = min(1.0, max_dim / float(max(W, H)))
    return render_frame(project, t_ms, media, size=(max(2, int(W * scale)), max(2, int(H * scale))))


# ================================================================== advanced layer features
# Masks, layer styles, trim paths, repeaters, text animators, motion blur,
# precompositions and the 2.5D camera all live here so the main pass stays small.


def _mask_raster(masks: List[dict], size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Build a combined mask (float 0..1) for a sprite of `size`."""
    if not masks:
        return None
    w, h = size
    combined = np.zeros((h, w), np.float32)
    has_any = False
    for mask in masks:
        if mask.get("enabled", True) is False:
            continue
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        shape = mask.get("shape", "polygon")
        pts = mask.get("points") or []
        if shape == "rect":
            draw.rectangle((0, 0, w - 1, h - 1), fill=255)
        elif shape == "ellipse":
            draw.ellipse((0, 0, w - 1, h - 1), fill=255)
        elif len(pts) >= 3:
            poly = [(float(p[0]) * w, float(p[1]) * h) for p in pts]
            draw.polygon(poly, fill=255)
        arr = np.asarray(m, np.float32) / 255.0
        feather = float(mask.get("feather", 0) or 0)
        if feather > 0:
            radius = max(0.5, feather * w / 400.0)
            arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8), "L")
                             .filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0
        expansion = float(mask.get("expansion", 0) or 0)
        if abs(expansion) > 0.01:
            k = int(abs(expansion) * w / 200.0) * 2 + 1
            img = Image.fromarray((arr * 255).astype(np.uint8), "L")
            img = img.filter(ImageFilter.MaxFilter(k) if expansion > 0 else ImageFilter.MinFilter(k))
            arr = np.asarray(img, np.float32) / 255.0
        if mask.get("invert"):
            arr = 1.0 - arr
        arr = arr * float(mask.get("opacity", 1.0) if mask.get("opacity") is not None else 1.0)
        mode = (mask.get("mode") or "add").lower()
        if not has_any and mode == "add":
            combined = arr
        elif mode == "add":
            combined = np.maximum(combined, arr)
        elif mode == "subtract":
            combined = np.clip(combined - arr, 0, 1)
        elif mode == "intersect":
            combined = combined * arr
        elif mode == "difference":
            combined = np.abs(combined - arr)
        has_any = True
    return combined if has_any else None


def apply_masks(img: Image.Image, masks: List[dict]) -> Image.Image:
    mask = _mask_raster(masks, img.size)
    if mask is None:
        return img
    arr = _np_rgba(img)
    arr[..., 3:4] *= mask[..., None]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")


def apply_layer_styles(img: Image.Image, styles: List[dict], q: float = 1.0) -> Image.Image:
    """Layer styles: drop shadow, outer/inner glow, bevel, satin, stroke, colour overlay."""
    if not styles:
        return img
    # Styles bleed outside the layer (shadows, outer glows, strokes), so give the
    # sprite enough transparent margin for them to actually be visible.
    pad = 0.0
    for style in styles:
        if style.get("enabled", True) is False:
            continue
        p_ = style.get("params") or {}
        blur = abs(float(p_.get("blur", p_.get("size", 10)) or 10))
        dist = abs(float(p_.get("distance", 8) or 8))
        size = abs(float(p_.get("size", max(blur, dist)) or 0))
        pad = max(pad, blur * 2.5 + dist, size * 2.5 + 4)
    pad = int(min(max(pad * max(q, 0.2), 4), 512))
    if pad:
        big = Image.new("RGBA", (img.size[0] + pad * 2, img.size[1] + pad * 2), (0, 0, 0, 0))
        big.paste(img, (pad, pad))
        out = big
        off = (pad, pad)
    else:
        out = img
        off = (0, 0)
    for style in styles:
        if style.get("enabled", True) is False:
            continue
        kind = (style.get("type") or "").lower()
        params = style.get("params") or {}
        try:
            if kind == "drop_shadow":
                out = _style_shadow(out, params, q)
            elif kind in ("outer_glow", "glow"):
                out = _style_glow(out, params, q, inner=False)
            elif kind in ("inner_glow",):
                out = _style_glow(out, params, q, inner=True)
            elif kind in ("stroke", "outline"):
                out = _style_stroke(out, params, q)
            elif kind in ("color_overlay", "color"):
                out = _style_overlay(out, params)
            elif kind in ("bevel", "bevel_emboss"):
                out = _style_bevel(out, params, q)
            elif kind == "satin":
                out = _style_satin(out, params, q)
        except Exception:
            continue
    return out


def _style_color(params: dict) -> tuple:
    rgba = parse_color(params.get("color", "#FF000000"))
    return int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(rgba[3] * 255)


def _alpha_of(img: Image.Image) -> Image.Image:
    return img.getchannel("A")


def _style_shadow(img: Image.Image, p: dict, q: float) -> Image.Image:
    w, h = img.size
    alpha = _alpha_of(img)
    blur = float(p.get("blur", 10)) * q
    if blur > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
    col = _style_color({**p, "color": p.get("color", "#FF000000")})
    op = float(p.get("opacity", 0.6))
    shadow = Image.new("RGBA", (w, h), (col[0], col[1], col[2], 0))
    sh_alpha = alpha.point(lambda v: int(v * op * col[3] / 255))
    shadow.putalpha(sh_alpha)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(shadow, (int(float(p.get("distance", 8)) * q * 0.7),
                                 int(float(p.get("distance", 8)) * q)))
    out.alpha_composite(img)
    return out


def _style_glow(img: Image.Image, p: dict, q: float, inner: bool) -> Image.Image:
    w, h = img.size
    alpha = _alpha_of(img)
    blur = max(0.5, float(p.get("blur", 10)) * q)
    if inner:
        inv = alpha.point(lambda v: 255 - v).filter(ImageFilter.GaussianBlur(blur))
        glow_mask = Image.fromarray(
            (np.clip((np.asarray(inv, np.float32) / 255.0) * float(p.get("opacity", 0.7)), 0, 1) * 255
             ).astype(np.uint8), "L").point(lambda v: v)
        glow_mask = Image.fromarray(
            (np.asarray(glow_mask, np.float32) * np.asarray(alpha, np.float32) / 255.0).astype(np.uint8), "L")
        out = img.copy()
    else:
        blur_alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
        glow_mask = Image.fromarray(
            (np.clip((np.asarray(blur_alpha, np.float32) / 255.0) * 1.4 * float(p.get("opacity", 0.7)), 0, 1) * 255
             ).astype(np.uint8), "L")
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    col = _style_color({**p, "color": p.get("color", "#FFFFFFFF")})
    layer = Image.new("RGBA", (w, h), (col[0], col[1], col[2], 0))
    layer.putalpha(glow_mask)
    if inner:
        out = img.copy()
        out.alpha_composite(layer)
        return out
    out.alpha_composite(layer)
    out.alpha_composite(img)
    return out


def _style_stroke(img: Image.Image, p: dict, q: float) -> Image.Image:
    w, h = img.size
    alpha = _alpha_of(img)
    k = int(max(1.0, float(p.get("size", 4)) * q)) * 2 + 1
    grown = alpha.filter(ImageFilter.MaxFilter(k))
    arr = (np.asarray(grown, np.float32) - np.asarray(alpha, np.float32)) / 255.0
    col = _style_color({**p, "color": p.get("color", "#FFFFFFFF")})
    stroke = Image.new("RGBA", (w, h), (col[0], col[1], col[2], 0))
    stroke.putalpha(np.clip(arr, 0, 1).astype(np.float32).__mul__(255).astype(np.uint8) if False
                    else Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L"))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(stroke)
    out.alpha_composite(img)
    return out


def _style_overlay(img: Image.Image, p: dict) -> Image.Image:
    col = _style_color({**p, "color": p.get("color", "#FFFF0000")})
    op = float(p.get("opacity", 0.85))
    overlay = Image.new("RGBA", img.size, (col[0], col[1], col[2], 255))
    overlay.putalpha(_alpha_of(img).point(lambda v: int(v * op)))
    out = img.copy()
    out.alpha_composite(overlay)
    return out


def _style_bevel(img: Image.Image, p: dict, q: float) -> Image.Image:
    col = _style_color({**p, "color": p.get("highlight", "#FFFFFFFF")})
    depth = float(p.get("depth", 4)) * q
    alpha = _alpha_of(img)
    emb = alpha.filter(ImageFilter.EMBOSS)
    arr = np.asarray(emb, np.float32) / 255.0
    shade = Image.new("RGBA", img.size, (col[0], col[1], col[2], 0))
    shade.putalpha(Image.fromarray((np.clip(arr * depth / 4.0, 0, 1) * 255).astype(np.uint8), "L"))
    out = img.copy()
    out.alpha_composite(shade)
    return out


def _style_satin(img: Image.Image, p: dict, q: float) -> Image.Image:
    alpha = _alpha_of(img)
    blur = max(0.5, float(p.get("blur", 8)) * q)
    a = np.asarray(alpha.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0
    a = np.clip(0.5 + (a - 0.5) * 1.6, 0, 1)
    col = _style_color({**p, "color": p.get("color", "#FF000000")})
    shade = Image.new("RGBA", img.size, (col[0], col[1], col[2], 0))
    shade.putalpha(Image.fromarray((a * float(p.get("opacity", 0.5)) * 255).astype(np.uint8), "L"))
    out = img.copy()
    out.alpha_composite(shade)
    return out


def trim_path_points(points: List[Sequence[float]], start_pct: float, end_pct: float
                     ) -> Tuple[List[List[float]], bool]:
    """Return the trimmed polyline and whether it is closed."""
    if not points:
        return [], False
    start = max(0.0, min(1.0, float(start_pct) / 100.0))
    end = max(0.0, min(1.0, float(end_pct) / 100.0))
    if end < start:
        start, end = end, start
    closed = abs(start) < 1e-6 and abs(end - 1) < 1e-6
    if closed:
        return [[float(p[0]), float(p[1])] for p in points], True
    n = len(points)
    segs = n if closed else n - 1
    if segs <= 0:
        return [[float(points[0][0]), float(points[0][1])]], False
    out: List[List[float]] = []
    for i in range(segs):
        a = points[i]
        b = points[(i + 1) % n]
        s0 = i / segs
        s1 = (i + 1) / segs
        lo = max(start, s0)
        hi = min(end, s1)
        if hi <= lo:
            continue
        u0 = (lo - s0) / (s1 - s0)
        u1 = (hi - s0) / (s1 - s0)
        p0 = [a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0]
        p1 = [a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1]
        if not out:
            out.append(p0)
        out.append(p1)
    return out, True


def text_runs(layer: dict) -> List[dict]:
    """Flatten text into positioned characters (for animators)."""
    text = str(layer.get("text", ""))
    runs = []
    for li, line in enumerate(text.split("\n")):
        runs.append({"char": "\n", "line": li, "x": 0.0, "y": li, "index": len(runs)})
        for ci, ch in enumerate(line):
            runs.append({"char": ch, "line": li, "col": ci, "index": len(runs)})
    return runs


def selector_weight(animator: dict, index: int, total: int, char: dict) -> float:
    sel = animator.get("selector") or {}
    kind = (sel.get("type") or "range").lower()
    if kind == "range":
        start = float(sel.get("start", 0)) / 100.0
        end = float(sel.get("end", 100)) / 100.0
        frac = index / max(1, total - 1)
        lo, hi = min(start, end), max(start, end)
        if frac < lo or frac > hi:
            return 0.0
        span = max(1e-6, hi - lo)
        u = (frac - lo) / span
        if sel.get("smooth", True):
            u = u * u * (3 - 2 * u)
        return float(u)
    if kind == "wipe":
        # weight by the character's horizontal position in the line
        col = float(char.get("col") or 0)
        line_len = max(1, int(sel.get("_line_len") or total))
        frac = col / max(1, line_len)
        start = float(sel.get("start", 0)) / 100.0
        end = float(sel.get("end", 100)) / 100.0
        if start <= frac <= end:
            return 1.0
        return 0.0
    return 1.0


