"""Media handling: probing, proxy generation, frame extraction, audio decoding.

Uses the ffmpeg binary bundled by imageio-ffmpeg when available, otherwise
falls back to a system ffmpeg on PATH. No ffprobe needed: metadata is parsed
from `ffmpeg -i` stderr.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

_FFMPEG: Optional[str] = None
_FFMPEG_LOCK = threading.Lock()

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".3gp"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".aac", ".m4a", ".ogg", ".opus", ".flac", ".wma"}


# ------------------------------------------------------------------ ffmpeg

def ffmpeg_path() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    with _FFMPEG_LOCK:
        if _FFMPEG:
            return _FFMPEG
        candidates = []
        try:
            import imageio_ffmpeg  # type: ignore
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                candidates.append(exe)
        except Exception:
            pass
        env = os.environ.get("CRAFTMOTION_FFMPEG")
        if env and os.path.exists(env):
            candidates.append(env)
        which = shutil.which("ffmpeg")
        if which:
            candidates.append(which)
        # bundled next to the executable (packaged builds)
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.argv[0])))
        for name in ("ffmpeg.exe", "ffmpeg"):
            p = os.path.join(base, name)
            if os.path.exists(p):
                candidates.append(p)
        if not candidates:
            raise RuntimeError(
                "ffmpeg not found. Install imageio-ffmpeg (pip install imageio-ffmpeg) "
                "or put ffmpeg on PATH / set CRAFTMOTION_FFMPEG."
            )
        _FFMPEG = candidates[0]
        return _FFMPEG


def _run(args: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
    creationflags = 0
    if os.name == "nt":
        creationflags = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        creationflags=creationflags,
    )


def cache_root() -> str:
    root = os.environ.get("CRAFTMOTION_CACHE")
    if not root:
        root = os.path.join(tempfile.gettempdir(), "craftmotion-cache")
    os.makedirs(root, exist_ok=True)
    return root


def _file_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]


# ------------------------------------------------------------------ probing

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)")
_AUD_RE = re.compile(r"Stream #\d+:\d+.*?Audio:")
_ROT_RE = re.compile(r"rotate\s*:\s*(-?\d+)")
_SAR_RE = re.compile(r"Stream #\d+:\d+.*?Video: ([^,]+)")


def ffmpeg_stderr(args: List[str]) -> str:
    try:
        cp = _run([ffmpeg_path(), "-hide_banner", "-nostdin"] + args, timeout=60)
        return cp.stderr.decode("utf-8", "ignore")
    except Exception as exc:  # pragma: no cover
        return f"error: {exc}"


def probe_media(path: str) -> dict:
    """Return a media descriptor dict for a file on disk."""
    ext = os.path.splitext(path)[1].lower()
    info: dict = {
        "path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "kind": "image" if ext in IMAGE_EXT else ("audio" if ext in AUDIO_EXT else "video"),
        "width": 0,
        "height": 0,
        "duration_ms": 0,
        "fps": 0.0,
        "has_audio": False,
    }
    if ext in IMAGE_EXT:
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                info["width"], info["height"] = im.size
            info["duration_ms"] = 0  # stills get the layer/project duration
            return info
        except Exception:
            pass
    text = ffmpeg_stderr(["-i", path])
    m = _DUR_RE.search(text)
    if m:
        h, mi, s = m.groups()
        info["duration_ms"] = int((int(h) * 3600 + int(mi) * 60 + float(s)) * 1000)
    mv = _VID_RE.search(text)
    if mv and ext not in IMAGE_EXT:
        w, h = int(mv.group(1)), int(mv.group(2))
        rot = _ROT_RE.search(text)
        if rot and abs(int(rot.group(1))) % 180 == 90:
            w, h = h, w
        info["width"], info["height"] = w, h
        mf = _FPS_RE.search(text)
        if mf:
            try:
                info["fps"] = float(mf.group(1))
            except ValueError:
                info["fps"] = 0.0
        info["has_audio"] = bool(_AUD_RE.search(text))
        if info["kind"] == "audio":
            info["has_audio"] = True
    elif ext in AUDIO_EXT:
        info["has_audio"] = True
        info["kind"] = "audio"
    return info


# ------------------------------------------------------------------ manager

class MediaManager:
    """Owns proxy files, extracted audio, thumbnails and a frame cache."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or cache_root()
        os.makedirs(self.cache_dir, exist_ok=True)
        self._frames: "OrderedDict[tuple, Image.Image]" = OrderedDict()
        # Frame cache is byte-bounded, not count-bounded: at 1080x1350 a single
        # RGB frame is ~4 MB, so a few hundred entries would OOM a render.
        self._frame_cache_bytes = 0
        self.frame_cache_limit = int(os.environ.get("CRAFTMOTION_FRAME_CACHE_MB", "64")) * 1024 * 1024
        self._frame_lock = threading.Lock()
        self._audio_wav: Dict[str, str] = {}
        self._proxy_state: Dict[str, dict] = {}
        self._thumbs: Dict[str, Tuple[float, Image.Image]] = {}
        self.on_status = None  # callable(str) set by GUI

    # ---------------------------------------------------------- status
    def _status(self, msg: str) -> None:
        try:
            if self.on_status:
                self.on_status(msg)
        except Exception:
            pass

    # ---------------------------------------------------------- paths
    @staticmethod
    def resolve_path(item: dict) -> Optional[str]:
        p = item.get("path")
        if p and os.path.exists(p):
            return p
        # try next to the project file
        proj = item.get("project_dir")
        if proj and p:
            cand = os.path.join(proj, os.path.basename(p))
            if os.path.exists(cand):
                return cand
        return None

    # ---------------------------------------------------------- proxies
    def proxy_path(self, item: dict) -> Optional[str]:
        p = self.resolve_path(item)
        if not p:
            return None
        ext = os.path.splitext(p)[1].lower()
        if ext in IMAGE_EXT:
            return p
        key = _file_key(p)
        out = os.path.join(self.cache_dir, f"proxy_{key}.mp4")
        return out if os.path.exists(out) else None

    def ensure_proxy(self, item: dict, synchronous: bool = False) -> Optional[str]:
        src = self.resolve_path(item)
        if not src:
            return None
        if os.path.splitext(src)[1].lower() in IMAGE_EXT:
            return src
        key = _file_key(src)
        out = os.path.join(self.cache_dir, f"proxy_{key}.mp4")
        if os.path.exists(out):
            return out
        existing = self._proxy_state.get(key)
        if existing and existing.get("running"):
            return None
        self._proxy_state[key] = {"running": True, "progress": 0.0}

        def work():
            try:
                self._status(f"Building preview proxy: {os.path.basename(src)}")
                dur = float(item.get("duration_ms") or 0) / 1000.0
                vf = "scale=-2:'min(720,ih)',fps=30"
                args = [
                    ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", src,
                    "-t", f"{max(0.1, dur):.3f}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "27", "-g", "30", "-pix_fmt", "yuv420p", "-an", out,
                ]
                _run(args, timeout=1800)
                self._proxy_state[key] = {"running": False, "done": os.path.exists(out)}
                self._status(f"Proxy ready: {os.path.basename(src)}")
            except Exception as exc:
                self._proxy_state[key] = {"running": False, "error": str(exc)}
                self._status(f"Proxy failed for {os.path.basename(src)}: {exc}")

        if synchronous:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()
        return out if os.path.exists(out) else None

    def proxy_progress(self, item: dict) -> float:
        p = self.resolve_path(item)
        if not p:
            return 1.0
        if self.proxy_path(item):
            return 1.0
        key = _file_key(p)
        st = self._proxy_state.get(key, {})
        return 0.0 if st.get("running") else (1.0 if st.get("done") else 0.5)

    # ---------------------------------------------------------- thumbnails
    def thumbnail(self, item: dict, width: int = 160) -> Optional[Image.Image]:
        src = self.resolve_path(item)
        if not src:
            return None
        now = time.time()
        cached = self._thumbs.get(src)
        if cached and now - cached[0] < 300 and cached[1].width >= width:
            return cached[1]
        img = None
        if item.get("kind") == "image":
            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                img = None
        elif item.get("kind") == "video":
            path = self.proxy_path(item) or src
            img = self._extract(path, 0.0, None)
        if img is None:
            img = Image.new("RGB", (width, int(width * 9 / 16)), (28, 30, 36))
        ratio = width / max(1, img.width)
        img = img.resize((width, max(1, int(img.height * ratio))), Image.LANCZOS)
        self._thumbs[src] = (now, img)
        return img

    # ---------------------------------------------------------- frames
    def _extract(self, path: str, seconds: float, out_size: Optional[Tuple[int, int]]) -> Optional[Image.Image]:
        vf_parts = []
        if out_size:
            vf_parts.append(f"scale={int(out_size[0])}:{int(out_size[1])}")
        args = [ffmpeg_path(), "-hide_banner", "-nostdin", "-v", "error"]
        args += ["-ss", f"{max(0.0, seconds):.4f}", "-i", path]
        if vf_parts:
            args += ["-vf", ",".join(vf_parts)]
        args += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        try:
            creationflags = 0x08000000 if os.name == "nt" else 0
            cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60, creationflags=creationflags)
            if cp.returncode == 0 and cp.stdout:
                return Image.open(io.BytesIO(cp.stdout)).convert("RGB")
        except Exception:
            pass
        return None

    def get_frame(self, item: dict, t_ms: float, max_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
        """Frame at media-local time t_ms (already includes source_in), RGB."""
        src = self.resolve_path(item)
        if not src:
            return None
        if item.get("kind") == "image":
            try:
                return Image.open(src).convert("RGB")
            except Exception:
                return None
        path = self.proxy_path(item) or src
        if not os.path.exists(path):
            # proxy still building (or never requested): read the source directly
            path = src
        bucket = int(round(float(t_ms) / 33.0))  # ~30fps cache buckets
        key = (path, bucket, max_size)
        with self._frame_lock:
            if key in self._frames:
                self._frames.move_to_end(key)
                return self._frames[key]
        img = self._extract(path, bucket * 0.033, max_size)
        if img is not None:
            with self._frame_lock:
                self._frames[key] = img
                self._frame_cache_bytes += img.size[0] * img.size[1] * len(img.getbands())
                while self._frames and (self._frame_cache_bytes > self.frame_cache_limit
                                        or len(self._frames) > 240):
                    old_key, old_img = self._frames.popitem(last=False)
                    self._frame_cache_bytes -= old_img.size[0] * old_img.size[1] * len(old_img.getbands())
                    self._frame_cache_bytes = max(0, self._frame_cache_bytes)
        return img

    # ---------------------------------------------------------- audio
    def ensure_audio_wav(self, item: dict, synchronous: bool = False) -> Optional[str]:
        src = self.resolve_path(item)
        if not src:
            return None
        key = _file_key(src)
        out = os.path.join(self.cache_dir, f"audio_{key}.wav")
        if os.path.exists(out):
            return out
        if not item.get("has_audio") and item.get("kind") != "audio":
            return None

        def work():
            try:
                self._status(f"Extracting audio: {os.path.basename(src)}")
                _run([
                    ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-i", src,
                    "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out,
                ], timeout=1800)
            except Exception:
                pass

        if synchronous:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()
        return out if os.path.exists(out) else None

    def read_audio(self, item: dict) -> Optional[np.ndarray]:
        """Return float32 stereo samples shaped (n, 2) at 48 kHz, or None."""
        wav = self.ensure_audio_wav(item, synchronous=True)
        if not wav or not os.path.exists(wav):
            return None
        try:
            with wave.open(wav, "rb") as w:
                ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                raw = w.readframes(n)
            if sw != 2:
                return None
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch == 2:
                arr = arr.reshape(-1, 2)
            else:
                arr = np.repeat(arr.reshape(-1, 1), 2, axis=1)
            if sr != 48000:
                idx = (np.arange(int(arr.shape[0] * 48000 / sr)) * sr / 48000).astype(np.int64)
                idx = np.clip(idx, 0, arr.shape[0] - 1)
                arr = arr[idx]
            return arr
        except Exception:
            return None

    def waveform(self, item: dict, buckets: int = 900) -> List[float]:
        arr = self.read_audio(item)
        if arr is None or arr.shape[0] == 0:
            return []
        mono = np.abs(arr).mean(axis=1)
        n = mono.shape[0]
        step = max(1, n // buckets)
        peaks = []
        for i in range(0, n, step):
            chunk = mono[i:i + step]
            if chunk.size:
                peaks.append(float(chunk.max()))
        mx = max(peaks) or 1.0
        return [min(1.0, p / mx) for p in peaks]


def import_media_file(path: str) -> dict:
    """Probe a file and return a media descriptor (id assigned by Project)."""
    info = probe_media(path)
    info["id"] = None
    info.setdefault("project_dir", os.path.dirname(os.path.abspath(path)))
    return info
