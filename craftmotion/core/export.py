"""Render/export pipeline: MP4/WebM, PNG sequences, GIF, stills and audio mixdown."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from io import BytesIO
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from .media import MediaManager, ffmpeg_path
from .model import Project
from .render import Renderer
from .util import kf_eval

SR = 48000


def _creationflags():
    return 0x08000000 if os.name == "nt" else 0


def mix_audio(project: Project, media: MediaManager, out_wav: str,
              start_ms: float = 0, end_ms: Optional[float] = None) -> Optional[str]:
    """Mix all audible layers into a 48k stereo WAV (mixer + automation + master chain)."""
    from . import audio as AUD
    samples = AUD.mix(project, media, start_ms, end_ms)
    if samples is None or not samples.size:
        return None
    return AUD.write_wav(samples, out_wav)


def _codec_args(out_path: str, quality: str):
    """Return (video args, audio args) for the requested container/quality."""
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".webm":
        video = ["-c:v", "libvpx-vp9", "-b:v", "0",
                 "-crf", "32" if quality == "draft" else "24", "-row-mt", "1"]
        audio = ["-c:a", "libopus", "-b:a", "160k"]
    else:
        crf = {"draft": "26", "good": "20", "high": "16"}.get(quality, "20")
        preset = {"draft": "ultrafast", "good": "veryfast", "high": "medium"}.get(quality, "veryfast")
        video = ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]
        audio = ["-c:a", "aac", "-b:a", "192k"]
    return video, audio


def _run_ffmpeg(args: List[str], what: str) -> None:
    """Run ffmpeg to completion, surfacing its stderr tail if it fails."""
    err_path = os.path.join(tempfile.gettempdir(), f"craftmotion_ffmpeg_{os.getpid()}.log")
    with open(err_path, "wb") as err_file:
        proc = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=err_file, creationflags=_creationflags())
    try:
        with open(err_path, "rb") as fh:
            err = fh.read()[-1200:]
        os.remove(err_path)
    except OSError:
        err = b""
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: " + err.decode("utf-8", "ignore"))


def render_video(project: Project, media: MediaManager, out_path: str,
                 width: Optional[int] = None, height: Optional[int] = None,
                 fps: Optional[float] = None, start_ms: Optional[float] = None,
                 end_ms: Optional[float] = None, quality: str = "good",
                 include_audio: bool = True,
                 progress: Optional[Callable[[float, str], None]] = None,
                 cancelled: Optional[Callable[[], bool]] = None) -> str:
    """Render the timeline to a video file.

    Frames are rendered in chunks to a temporary PNG sequence and each chunk is
    encoded before the next one starts. Streaming every frame straight into
    ffmpeg's stdin is faster, but ffmpeg reads a raw pipe ahead of the encoder
    and keeps those frames in native memory: on a 31 s 1080x1350 timeline that
    reached several GB and the render was OOM-killed mid-encode (leaving a short,
    still-playable file). Chunking bounds memory to one chunk of PNGs and makes
    the export resumable in the sense that a failed chunk never corrupts the
    already-encoded ones.
    """
    W = int(width or project.width)
    H = int(height or project.height)
    W -= W % 2
    H -= H % 2
    fps = float(fps or project.fps or 30.0)
    start = float(start_ms if start_ms is not None else 0)
    end = float(end_ms if end_ms is not None else project.duration)
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".mp4", ".webm", ".mov", ".mkv"):
        out_path += ".mp4"
        ext = ".mp4"

    audio_wav = None
    if include_audio:
        tmp = os.path.join(tempfile.gettempdir(), f"craftmotion_audio_{os.getpid()}.wav")
        try:
            audio_wav = mix_audio(project, media, tmp, start, end)
        except Exception:
            audio_wav = None

    total_frames = max(1, int(round((end - start) / 1000.0 * fps)))
    chunk = max(1, int(os.environ.get("CRAFTMOTION_RENDER_CHUNK", "90")))
    workdir = tempfile.mkdtemp(prefix="craftmotion_render_")
    frames_dir = os.path.join(workdir, "frames")
    seg_dir = os.path.join(workdir, "segments")
    os.makedirs(frames_dir)
    os.makedirs(seg_dir)
    video_args, audio_args = _codec_args(out_path, quality)

    renderer = Renderer(project, media, mode="export")
    segments: List[str] = []
    done = 0
    try:
        while done < total_frames:
            if cancelled and cancelled():
                break
            n = min(chunk, total_frames - done)
            for k in range(n):
                t = start + (done + k) / fps * 1000.0
                img = renderer.render(t, size=(W, H)).convert("RGB")
                # PNG (not raw) so a chunk stays a few hundred MB at most
                img.save(os.path.join(frames_dir, f"{k:05d}.png"), compress_level=1)
                if progress and ((done + k) % max(1, total_frames // 100) == 0):
                    progress((done + k) / total_frames,
                             f"Rendering frame {done + k + 1}/{total_frames}")
            seg = os.path.join(seg_dir, f"seg{len(segments):04d}{ext}")
            _run_ffmpeg([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                         "-framerate", f"{fps:.5f}", "-i", os.path.join(frames_dir, "%05d.png"),
                         *video_args, "-an", seg], "segment encode")
            for name in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, name))
            segments.append(seg)
            done += n
        if progress:
            progress(min(1.0, done / total_frames), "Assembling")
        if not segments:
            raise RuntimeError("nothing was rendered")
        joined = os.path.join(workdir, "video" + ext)
        if len(segments) == 1:
            shutil.move(segments[0], joined)
        else:
            listing = os.path.join(workdir, "segments.txt")
            with open(listing, "w") as fh:
                for seg in segments:
                    fh.write("file '%s'\n" % seg.replace("'", "'\\''"))
            _run_ffmpeg([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                         "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", joined],
                        "segment join")
        if audio_wav:
            _run_ffmpeg([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                         "-i", joined, "-i", audio_wav, "-c:v", "copy", *audio_args,
                         "-shortest", out_path], "audio mux")
        else:
            shutil.move(joined, out_path)
    finally:
        renderer.close()
        shutil.rmtree(workdir, ignore_errors=True)
        if audio_wav:
            try:
                os.remove(audio_wav)
            except OSError:
                pass

    # Never trust the encoder's exit code alone: a killed encoder still leaves a
    # playable file, so check what actually landed on disk.
    if not (cancelled and cancelled()):
        from .media import probe_media
        try:
            got = probe_media(out_path).get("duration_ms") or 0
        except Exception:
            got = 0
        want = end - start
        if got and got < want - 500:
            raise RuntimeError(
                f"encode truncated: {got / 1000:.2f}s written, {want / 1000:.2f}s expected")
    if progress:
        progress(1.0, "Finalising")
    return out_path


def render_still(project: Project, media: MediaManager, t_ms: float, out_path: str,
                 width: Optional[int] = None, height: Optional[int] = None) -> str:
    renderer = Renderer(project, media, mode="export")
    try:
        img = renderer.render(t_ms, size=(width or project.width, height or project.height))
    finally:
        renderer.close()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.splitext(out_path)[1].lower() in (".jpg", ".jpeg"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        bg.save(out_path, quality=92)
    else:
        img.save(out_path)
    return out_path


def render_sequence(project: Project, media: MediaManager, out_dir: str, start_ms: float = 0,
                    end_ms: Optional[float] = None, fps: Optional[float] = None,
                    width: Optional[int] = None, height: Optional[int] = None,
                    fmt: str = "png", progress: Optional[Callable] = None,
                    cancelled: Optional[Callable] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fps = float(fps or project.fps or 30.0)
    end = float(end_ms if end_ms is not None else project.duration)
    total = max(1, int(round((end - start_ms) / 1000.0 * fps)))
    renderer = Renderer(project, media, mode="export")
    try:
        for i in range(total):
            if cancelled and cancelled():
                break
            t = start_ms + i / fps * 1000.0
            img = renderer.render(t, size=(width or project.width, height or project.height))
            img.save(os.path.join(out_dir, f"frame_{i:05d}.{fmt}"))
            if progress and i % max(1, total // 50) == 0:
                progress(i / total, f"Frame {i+1}/{total}")
    finally:
        renderer.close()
    return out_dir


def render_gif(project: Project, media: MediaManager, out_path: str, start_ms: float = 0,
               end_ms: Optional[float] = None, fps: float = 15, width: int = 480,
               progress: Optional[Callable] = None, cancelled: Optional[Callable] = None) -> str:
    h = int(width * project.height / project.width)
    tmp_dir = tempfile.mkdtemp(prefix="cm_gif_")
    try:
        render_sequence(project, media, tmp_dir, start_ms, end_ms, fps, width, h, "png", progress, cancelled)
        out_path = os.path.abspath(out_path)
        cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-y", "-framerate", str(fps),
               "-i", os.path.join(tmp_dir, "frame_%05d.png"),
               "-vf", "split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=sierra2_4a",
               out_path]
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_creationflags())
        if cp.returncode != 0:
            raise RuntimeError("gif encode failed: " + cp.stderr.decode("utf-8", "ignore")[-500:])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
