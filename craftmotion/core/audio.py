"""Audio mixing, automation, effects and EBU R128 loudness.

The mixer feeds both the timeline meters and the video exporter so what you
hear in the app is exactly what lands in the file.
"""
from __future__ import annotations

import math
import os
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # scipy is optional (bundled in the Windows build, not required on Linux)
    from scipy.signal import lfilter as _lfilter
except Exception:  # pragma: no cover
    _lfilter = None

from .model import Project

SR = 48000


# ------------------------------------------------------------------ helpers
def _db(x: float) -> float:
    return 10.0 ** (float(x) / 20.0)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n, np.float32) / n
    return np.convolve(x, k, mode="same").astype(np.float32)


def _biquad(sig: np.ndarray, kind: str, freq: float, q: float = 0.707, gain_db: float = 0.0) -> np.ndarray:
    """RBJ cookbook biquad (per channel), zero-phase-ish forward only."""
    freq = float(np.clip(freq, 20.0, SR / 2.2))
    w0 = 2 * math.pi * freq / SR
    cs, sn = math.cos(w0), math.sin(w0)
    alpha = sn / (2 * max(0.05, q))
    A = 10 ** (gain_db / 40.0)
    if kind == "lowpass":
        b = [((1 - cs) / 2), (1 - cs), ((1 - cs) / 2)]
        a = [1 + alpha, -2 * cs, 1 - alpha]
    elif kind == "highpass":
        b = [((1 + cs) / 2), -(1 + cs), ((1 + cs) / 2)]
        a = [1 + alpha, -2 * cs, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cs, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cs, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cs + sq), 2 * A * ((A - 1) - (A + 1) * cs), A * ((A + 1) - (A - 1) * cs - sq)]
        a = [(A + 1) + (A - 1) * cs + sq, -2 * ((A - 1) + (A + 1) * cs), (A + 1) + (A - 1) * cs - sq]
    else:  # highshelf
        sq = 2 * math.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cs + sq), -2 * A * ((A - 1) + (A + 1) * cs), A * ((A + 1) + (A - 1) * cs - sq)]
        a = [(A + 1) - (A - 1) * cs + sq, 2 * ((A - 1) - (A + 1) * cs), (A + 1) - (A - 1) * cs - sq]
    b = np.array(b, np.float64) / a[0]
    a = np.array(a, np.float64) / a[0]
    mono = sig.ndim == 1
    x = sig[:, None] if mono else sig
    if _lfilter is not None:
        y = _lfilter(b, a, x, axis=0).astype(np.float32)
    else:
        y = np.empty_like(x, dtype=np.float32)
        for ch in range(x.shape[1]):
            y[:, ch] = _iir_slow(b, a, x[:, ch])
    return y[:, 0] if mono else y


def _iir_slow(b, a, x):
    y = np.zeros_like(x, dtype=np.float64)
    x0 = x1 = y0 = y1 = 0.0
    for i in range(x.shape[0]):
        v = b[0] * x[i] + b[1] * x1 + b[2] * x0 - a[1] * y1 - a[2] * y0
        x0, x1 = x1, x[i]
        y0, y1 = y1, v
        y[i] = v
    return y.astype(np.float32)


def _fft_filter(sig: np.ndarray, low: Optional[float], high: Optional[float]) -> np.ndarray:
    """Fast spectral band filter (used for gentle tone shaping)."""
    if low is None and high is None:
        return sig
    spec = np.fft.rfft(sig, axis=0)
    freqs = np.fft.rfftfreq(sig.shape[0], 1.0 / SR)
    mask = np.ones_like(freqs)
    if low:
        mask *= 1.0 / (1.0 + (low / np.maximum(freqs, 1e-3)) ** 4)
    if high:
        mask *= 1.0 / (1.0 + (freqs / high) ** 4)
    return np.fft.irfft(spec * mask[:, None], n=sig.shape[0], axis=0).astype(np.float32)


def _compress(sig: np.ndarray, threshold_db: float, ratio: float, attack_ms: float, release_ms: float,
              makeup_db: float = 0.0) -> np.ndarray:
    if ratio <= 1.001:
        return sig
    a = math.exp(-1.0 / max(1.0, attack_ms / 1000.0 * SR))
    r = math.exp(-1.0 / max(1.0, release_ms / 1000.0 * SR))
    level = 0.0
    out = np.empty_like(sig)
    th = _db(threshold_db)
    for i in range(sig.shape[0]):
        peak = float(np.max(np.abs(sig[i])))
        coef = a if peak > level else r
        level = coef * level + (1 - coef) * peak
        if level > th and level > 1e-9:
            over_db = 20 * math.log10(level / th)
            reduce_db = over_db * (1 - 1.0 / ratio)
            g = _db(-reduce_db) * _db(makeup_db)
        else:
            g = _db(makeup_db)
        out[i] = sig[i] * g
    return out


# ------------------------------------------------------------------ loudness
# ITU-R BS.1770-4 K-weighting coefficients for 48 kHz
_K_SHELF_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_K_SHELF_A = [1.0, -1.69065929318241, 0.73248077421585]
_K_HP_B = [1.0, -2.0, 1.0]
_K_HP_A = [1.0, -1.99004745483398, 0.99007225036621]


def _k_weight(sig: np.ndarray) -> np.ndarray:
    """ITU-R BS.1770 K-weighting: high-shelf + high-pass at 48 kHz."""
    x = np.asarray(sig, np.float32)
    if x.ndim == 3:
        x = x.squeeze()
    if x.shape[0] and _lfilter is not None:
        mono = x.ndim == 1
        yy = x[:, None] if mono else x
        yy = _lfilter(_K_SHELF_B, _K_SHELF_A, yy, axis=0)
        yy = _lfilter(_K_HP_B, _K_HP_A, yy, axis=0)
        return (yy[:, 0] if mono else yy).astype(np.float32)
    return _fft_filter(x, 38.0, 16000.0)


def lufs(samples: np.ndarray) -> float:
    """Integrated loudness in LUFS (BS.1770-4 with gating)."""
    if samples is None or samples.size == 0:
        return -70.0
    x = np.asarray(samples, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    x = _k_weight(x)
    block = int(0.4 * SR)
    hop = int(0.1 * SR)
    if x.shape[0] < block:
        x = np.pad(x, ((0, block - x.shape[0]), (0, 0)))
    energies = []
    for start in range(0, x.shape[0] - block + 1, hop):
        seg = x[start:start + block]
        energies.append(float(np.mean(np.sum(seg ** 2, axis=1))))
    energies = np.array(energies, np.float32)
    if energies.size == 0:
        return -70.0
    with np.errstate(divide="ignore"):
        l = -0.691 + 10 * np.log10(np.maximum(energies, 1e-12))
    # absolute gate at -70 LUFS then relative gate at -10 LU below the ungated mean
    keep = l > -70
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * math.log10(float(np.mean(energies[keep]))) - 10.0
    keep2 = l > rel
    if not keep2.any():
        keep2 = keep
    return float(-0.691 + 10 * math.log10(float(np.mean(energies[keep2]))))


def true_peak(samples: np.ndarray) -> float:
    if samples is None or samples.size == 0:
        return 0.0
    x = np.abs(np.asarray(samples, np.float32))
    if x.shape[0] > 4:
        # crude 4x oversample around peaks
        idx = np.argsort(x.reshape(-1))[-64:]
        up = np.interp(np.linspace(0, x.shape[0] - 1, x.shape[0] * 4), np.arange(x.shape[0]), x[:, 0] if x.ndim > 1 else x)
        return float(max(x.max(), up.max()))
    return float(x.max())


def short_term_stats(samples: np.ndarray, window_ms: int = 300) -> dict:
    if samples is None or samples.size == 0:
        return {"momentary": -70.0, "short_term": -70.0, "peak": 0.0, "crest": 0.0}
    x = np.asarray(samples, np.float32)
    n = max(1, int(window_ms / 1000.0 * SR))
    tail = x[-min(x.shape[0], n):]
    st = lufs(tail)
    peak = float(np.max(np.abs(x)))
    crest = 20 * math.log10(max(1e-6, peak) / max(1e-6, float(np.sqrt(np.mean(x ** 2)))))
    return {"momentary": round(lufs(x[-min(x.shape[0], n):]), 2), "short_term": round(st, 2),
            "peak": round(peak, 4), "crest": round(crest, 2)}


# ------------------------------------------------------------------ per-layer chain
def _layer_audio_fx(piece: np.ndarray, chain: List[dict]) -> np.ndarray:
    out = piece
    for fx in chain or []:
        if fx.get("enabled", True) is False:
            continue
        kind = (fx.get("type") or "").lower()
        p = fx.get("params") or {}
        try:
            if kind == "eq":
                out = _fft_filter(out, float(p.get("highpass", 0) or 0) or None,
                                  float(p.get("lowpass", 0) or 0) or None)
                if p.get("low_gain"):
                    out = _biquad(out, "lowshelf", float(p.get("low_freq", 120)), 0.7, float(p["low_gain"]))
                if p.get("high_gain"):
                    out = _biquad(out, "highshelf", float(p.get("high_freq", 6000)), 0.7, float(p["high_gain"]))
            elif kind == "compressor":
                out = _compress(out, float(p.get("threshold", -18)), float(p.get("ratio", 3)),
                                float(p.get("attack", 12)), float(p.get("release", 180)),
                                float(p.get("makeup", 0)))
            elif kind == "limiter":
                ceil = _db(float(p.get("ceiling", -1.0)))
                out = np.clip(out, -ceil, ceil)
            elif kind == "gain":
                out = out * _db(float(p.get("db", 0)))
            elif kind == "normalize":
                peak = float(np.max(np.abs(out))) if out.size else 0.0
                if peak > 1e-6:
                    out = out * (_db(float(p.get("target", -1.0))) / peak)
            elif kind == "gate":
                th = _db(float(p.get("threshold", -45)))
                mask = (np.abs(out) > th).astype(np.float32)
                out = out * _smooth(mask.mean(axis=1), 256)[:, None]
            elif kind == "reverb":
                wet = float(p.get("mix", 0.25))
                delay = int(SR * float(p.get("time", 0.18)))
                tail = np.zeros_like(out)
                if delay < out.shape[0]:
                    tail[delay:] = out[:-delay] * 0.6
                    tail[delay * 2:] += out[:-delay * 2] * 0.3 if delay * 2 < out.shape[0] else 0
                out = out * (1 - wet) + (out + tail) * wet
            elif kind == "pan":
                pan = float(np.clip(p.get("pan", 0.0), -1, 1))
                left = math.cos((pan + 1) * math.pi / 4)
                right = math.sin((pan + 1) * math.pi / 4)
                out = out * np.array([left, right], np.float32)[None, :]
        except Exception:
            continue
    return out


def automation_curve(layer: dict, start_ms: float, end_ms: float, name: str = "volume",
                     default: float = 1.0) -> np.ndarray:
    """Sample a layer automation track into a per-sample envelope."""
    n = max(1, int((end_ms - start_ms) / 1000.0 * SR))
    keys = (layer.get("keys") or {}).get(name)
    l_start = int(layer.get("start", 0))
    if not keys:
        base = float(layer.get("volume", 1.0) if name == "volume" else default) if name == "volume" else default
        return np.full(n, base, np.float32)
    t = (start_ms - l_start) + (np.arange(n) / SR) * 1000.0
    # vectorised linear interpolation over keyframe list
    ks = sorted(keys, key=lambda k: k["t"])
    times = np.array([k["t"] for k in ks], np.float32)
    vals = np.array([float(k["v"][0]) for k in ks], np.float32)
    return np.interp(t, times, vals).astype(np.float32)


# ------------------------------------------------------------------ main mixer
def mix(project: Project, media, start_ms: float = 0, end_ms: Optional[float] = None,
        use_mixer: bool = True) -> Optional[np.ndarray]:
    end = int(end_ms if end_ms is not None else project.duration)
    start = int(start_ms)
    n = max(1, int((end - start) / 1000.0 * SR))
    out = np.zeros((n, 2), np.float32)
    any_audio = False
    soloed = any(bool(l.get("solo")) for l in project.layers)
    master = (project.data.get("audio_master") or {})
    master_gain = _db(float(master.get("gain_db", 0.0) or 0.0))

    for layer in project.layers:
        kind = layer.get("kind")
        if kind not in ("audio", "video"):
            continue
        if not layer.get("include_audio", True):
            continue
        if use_mixer:
            if layer.get("mute"):
                continue
            if soloed and not layer.get("solo"):
                continue
        item = project.get_media(layer.get("media_id"))
        if not item:
            continue
        samples = media.read_audio(item)
        if samples is None or samples.size == 0:
            continue
        l_start = int(layer.get("start", 0))
        l_end = int(layer.get("end", project.duration))
        seg_start = max(l_start, start)
        seg_end = min(l_end, end)
        if seg_end <= seg_start:
            continue
        src_off = int(layer.get("source_in", 0) or 0)
        speed = float(layer.get("speed", 1.0) or 1.0)
        if abs(speed - 1.0) > 1e-3:
            idx = (np.arange(int((seg_end - seg_start) / 1000.0 * SR)) * speed).astype(np.int64)
            piece = samples[np.clip(idx + int(src_off / 1000.0 * SR), 0, samples.shape[0] - 1)]
        else:
            i0 = int((seg_start - l_start + src_off) / 1000.0 * SR)
            if i0 >= samples.shape[0]:
                continue
            count = int((seg_end - seg_start) / 1000.0 * SR)
            piece = samples[i0: i0 + count]
        if piece.shape[0] == 0:
            continue
        if piece.ndim == 1:
            piece = np.repeat(piece[:, None], 2, axis=1)
        if piece.shape[1] == 1:
            piece = np.repeat(piece, 2, axis=1)
        piece = piece[:, :2].astype(np.float32)
        env = automation_curve(layer, seg_start, seg_end, "volume",
                               float(layer.get("volume", 1.0) if kind == "audio" else 1.0))
        fi = float(layer.get("fade_in", 0) or 0)
        fo = float(layer.get("fade_out", 0) or 0)
        local = (seg_start - l_start) + (np.arange(piece.shape[0]) / SR) * 1000.0
        total = max(1.0, l_end - l_start)
        if fi > 0:
            env *= np.clip(local / fi, 0, 1)
        if fo > 0:
            env *= np.clip((total - local) / fo, 0, 1)
        piece = piece * np.interp(np.arange(piece.shape[0]), np.arange(env.shape[0]), env)[:, None]
        # pan automation
        pan = layer.get("pan")
        if pan is not None and abs(float(pan)) > 1e-3:
            ang = (float(pan) + 1) * math.pi / 4
            piece = piece * np.array([math.cos(ang), math.sin(ang)], np.float32)[None, :] * 1.414
        piece = _layer_audio_fx(piece, layer.get("audio_fx") or [])
        o0 = int((seg_start - start) / 1000.0 * SR)
        o1 = min(n, o0 + piece.shape[0])
        if o1 <= 0:
            continue
        out[o0:o1] += piece[: o1 - o0]
        any_audio = True

    if not any_audio:
        return None
    out *= master_gain
    # master chain
    for fx in master.get("chain") or []:
        out = _layer_audio_fx(out, [fx])
    ceiling = _db(float(master.get("ceiling_db", -0.3) or -0.3))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > ceiling:
        out = out * (ceiling / peak)
    return out


def write_wav(samples: np.ndarray, path: str) -> str:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(samples.shape[1] if samples.ndim > 1 else 1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def analyze(project: Project, media) -> dict:
    samples = mix(project, media)
    if samples is None:
        return {"integrated": -70.0, "peak": 0.0, "silent": True}
    return {"integrated": round(lufs(samples), 2), "peak": round(float(np.max(np.abs(samples))), 4),
            "true_peak": round(true_peak(samples), 4), "stats": short_term_stats(samples), "silent": False}


def normalize_to(project: Project, media, target_lufs: float = -16.0) -> dict:
    """Measure and store a master gain that hits `target_lufs`."""
    info = analyze(project, media)
    if info.get("silent"):
        return {"ok": False, "reason": "no audible layers"}
    gain = target_lufs - float(info["integrated"])
    master = project.data.setdefault("audio_master", {})
    master["gain_db"] = round(min(24.0, max(-24.0, gain)), 2)
    return {"ok": True, "gain_db": master["gain_db"], "integrated": info["integrated"],
            "target": target_lufs}


# ================================================================== reactive audio
# Envelope analysis powers both the audio-reactive effects (spectrum / waveform /
# pulse) and the `audio.react` command, which bakes an envelope onto any property.

_BANDS = ((20.0, 160.0), (160.0, 2000.0), (2000.0, 12000.0))


def _band_energy(power: "np.ndarray", freqs: "np.ndarray", lo: float, hi: float) -> float:
    sel = (freqs >= lo) & (freqs < hi)
    if not np.any(sel):
        return 0.0
    return float(np.sqrt(np.mean(power[sel])))


def envelope(project: Project, media, hop_ms: float = 20.0, bands: bool = True) -> dict:
    """Analyse the mixed soundtrack into a per-hop envelope.

    Returns ``{"hop_ms", "level", "low", "mid", "high", "times", "peaks"}`` with
    every channel normalised to 0..1 (peaks are the un-normalised maxima, in
    linear amplitude, so callers can decide their own scaling).
    """
    samples = mix(project, media)
    hop = max(1, int(SR * hop_ms / 1000.0))
    if samples is None or samples.size == 0:
        return {"hop_ms": hop_ms, "level": [], "low": [], "mid": [], "high": [],
                "times": [], "peaks": []}
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    n_hops = max(1, int(np.ceil(mono.shape[0] / hop)))
    win = 2048
    freqs = np.fft.rfftfreq(win, 1.0 / SR)
    level, low, mid, high, peaks = [], [], [], [], []
    for i in range(n_hops):
        seg = mono[i * hop: i * hop + hop]
        if seg.size == 0:
            seg = np.zeros(1, np.float32)
        rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
        peaks.append(rms)
        level.append(rms)
        if bands:
            chunk = mono[i * hop: i * hop + win]
            if chunk.size < win:
                chunk = np.pad(chunk, (0, win - chunk.size))
            spec = np.abs(np.fft.rfft(chunk * np.hanning(win))) / win
            power = spec ** 2
            low.append(_band_energy(power, freqs, *_BANDS[0]))
            mid.append(_band_energy(power, freqs, *_BANDS[1]))
            high.append(_band_energy(power, freqs, *_BANDS[2]))
    def _norm(seq):
        arr = np.asarray(seq, np.float32)
        top = float(arr.max()) if arr.size else 0.0
        if top <= 1e-9:
            return [0.0] * len(seq)
        return [round(float(v) / top, 5) for v in arr]
    return {"hop_ms": hop_ms, "level": _norm(level), "low": _norm(low), "mid": _norm(mid),
            "high": _norm(high), "times": [round(i * hop_ms, 1) for i in range(n_hops)],
            "peaks": [round(float(p), 5) for p in peaks]}


def envelope_at(env: dict, t_ms: float) -> Dict[str, float]:
    """Sample an envelope dict at a time (ms)."""
    hop = float(env.get("hop_ms") or 20.0)
    n = len(env.get("level") or [])
    if n == 0:
        return {"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0}
    i = int(max(0, min(n - 1, round(t_ms / hop))))
    out = {}
    for key in ("level", "low", "mid", "high"):
        seq = env.get(key) or []
        out[key] = float(seq[i]) if i < len(seq) else 0.0
    return out


def wave_slice(project: Project, media, t_ms: float, span_ms: float = 40.0,
               points: int = 64) -> List[float]:
    """A short window of the mixed waveform around t_ms, for waveform displays."""
    samples = mix(project, media, max(0.0, t_ms - span_ms / 2), t_ms + span_ms / 2)
    if samples is None or samples.size == 0:
        return [0.0] * points
    mono = samples.mean(axis=1)
    idx = np.linspace(0, max(0, mono.shape[0] - 1), points)
    return [round(float(np.clip(mono[int(i)], -1, 1)), 4) for i in idx]


def detect_beats(env: dict, min_gap_ms: float = 180.0, threshold: float = 0.35) -> List[float]:
    """Simple spectral-flux style onset detector over the envelope."""
    hop = float(env.get("hop_ms") or 20.0)
    level = env.get("level") or []
    if not level:
        return []
    flux = [0.0] + [max(0.0, level[i] - level[i - 1]) for i in range(1, len(level))]
    avg = float(np.mean(flux)) or 1e-6
    beats: List[float] = []
    last = -1e9
    for i, f in enumerate(flux):
        t = i * hop
        if f > avg * 1.6 and level[i] >= threshold and (t - last) >= min_gap_ms:
            beats.append(round(t, 1))
            last = t
    return beats


def estimate_tempo(beats: List[float]) -> float:
    """Median inter-onset interval → BPM (0.0 when there is nothing to measure)."""
    if len(beats) < 3:
        return 0.0
    gaps = np.diff(np.asarray(beats, np.float64))
    gaps = gaps[(gaps > 120) & (gaps < 2000)]
    if gaps.size == 0:
        return 0.0
    bpm = 60000.0 / float(np.median(gaps))
    while bpm < 70:
        bpm *= 2
    while bpm > 190:
        bpm /= 2
    return round(bpm, 1)


def spectrogram(project: Project, media, hop_ms: float = 40.0, bands: int = 24) -> dict:
    """Coarse log-spaced spectrum per hop, normalised 0..1 — feeds the analyser."""
    samples = mix(project, media)
    hop = max(1, int(SR * hop_ms / 1000.0))
    if samples is None or samples.size == 0:
        return {"hop_ms": hop_ms, "bands": bands, "frames": [], "level": []}
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    n_hops = max(1, int(np.ceil(mono.shape[0] / hop)))
    win = 1024
    freqs = np.fft.rfftfreq(win, 1.0 / SR)
    edges = np.geomspace(40.0, min(16000.0, SR / 2), bands + 1)
    frames, levels = [], []
    for i in range(n_hops):
        chunk = mono[i * hop: i * hop + win]
        if chunk.size < win:
            chunk = np.pad(chunk, (0, win - chunk.size))
        spec = np.abs(np.fft.rfft(chunk * np.hanning(win))) / win
        vals = []
        for b in range(bands):
            lo, hi = edges[b], edges[b + 1]
            sel = (freqs >= lo) & (freqs < hi)
            vals.append(float(spec[sel].mean()) if np.any(sel) else 0.0)
        frames.append(vals)
        levels.append(float(np.sqrt(np.mean(np.square(chunk[:hop], dtype=np.float64)))))
    flat = np.asarray(frames, np.float32)
    top = float(flat.max()) if flat.size else 0.0
    if top > 1e-9:
        flat = flat / top
    lv = np.asarray(levels, np.float32)
    ltop = float(lv.max()) if lv.size else 0.0
    if ltop > 1e-9:
        lv = lv / ltop
    return {"hop_ms": hop_ms, "bands": bands,
            "frames": [[round(float(v), 5) for v in row] for row in flat],
            "level": [round(float(v), 5) for v in lv]}


def spectrum_at(spec: dict, t_ms: float) -> List[float]:
    hop = float(spec.get("hop_ms") or 40.0)
    frames = spec.get("frames") or []
    if not frames:
        return []
    i = int(max(0, min(len(frames) - 1, round(t_ms / hop))))
    return list(frames[i])
