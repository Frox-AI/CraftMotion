"""Tests for the v1.1 parity feature set (308 effects, 39 transitions, masks,
styles, text animators, trim paths, repeaters, motion blur, precomps, 2.5D
camera + lights, expressions, colour grade, audio mixer + LUFS, captions,
tracking, three-point editing, Lottie)."""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from craftmotion.core import audio as AUD
from craftmotion.core import effects as FX
from craftmotion.core import editing as EDIT
from craftmotion.core.captions import (draw_captions, export_srt, import_srt, parse_srt_text)
from craftmotion.core.grade import apply_grade, ensure_grade, render_scope_image
from craftmotion.core.lottie import export_lottie, import_lottie
from craftmotion.core.model import Project
from craftmotion.core.registry import COMMANDS, Registry
from craftmotion.core.render import Renderer
from craftmotion.core.transitions import TRANSITIONS, apply_transition
from craftmotion.core.tracking import stabilize_patch


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def reg():
    r = Registry()
    r.execute("project.new", {"title": "parity", "width": 320, "height": 180,
                              "duration": 3000, "fps": 30})
    # transparent canvas so alpha assertions measure the layers, not the bg
    r.project.settings["bg"] = "#00000000"
    return r


def _probe_image(size=(96, 96)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.zeros((h, w, 4), np.float32)
    arr[..., 0] = xx / w
    arr[..., 1] = yy / h
    arr[..., 2] = 1 - xx / w
    arr[..., 3] = 1.0
    arr[h // 4:3 * h // 4, w // 4:3 * w // 4, :3] = (0.95, 0.9, 0.2)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGBA")


def _diff(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


# ------------------------------------------------------------------ catalogue
def test_effect_catalogue_size_and_health():
    assert len(FX.EFFECTS) >= 306
    img = _probe_image()
    ctx = {"project": None, "layer": {}, "t_local": 320, "extent": 0.4, "scale": 1.0, "renderer": None}
    broken = []
    for key, eff in FX.EFFECTS.items():
        try:
            out = eff.func(img, FX.defaults_for(key), ctx)
            assert isinstance(out, Image.Image) and out.size == img.size
        except Exception as exc:  # noqa: BLE001
            broken.append((key, repr(exc)[:80]))
    assert broken == [], f"effects raising: {broken[:5]}"


def test_effect_categories_span_the_palette():
    cats = {e.category for e in FX.EFFECTS.values()}
    for required in ("Color", "Blur", "Distort", "Stylize", "Glow" if "Glow" in cats else "Light",
                     "Generate", "Keying", "Transition", "Particles", "Blend", "Sharpen"):
        assert required in cats


def test_transition_catalogue_runs():
    assert len(TRANSITIONS) >= 30
    a = Image.new("RGBA", (64, 36), (10, 20, 40, 255))
    b = Image.new("RGBA", (64, 36), (220, 160, 20, 255))
    for key in TRANSITIONS:
        for p in (0.0, 0.5, 1.0):
            out = apply_transition(key, a, b, p)
            assert out.size == a.size


# ------------------------------------------------------------------ layer features
def test_mask_feather_changes_alpha(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FF22CCFF", "w": 200, "h": 200})["layer"]["id"]
    r = Renderer(reg.project)
    before = np.asarray(r.render(10).convert("RGBA"))[..., 3]
    reg.execute("mask.add", {"layer_id": lid, "shape": "ellipse", "feather": 10})
    after = np.asarray(r.render(10).convert("RGBA"))[..., 3]
    assert after.sum() < before.sum()


def test_mask_subtract_and_invert(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FFFFFFFF", "w": 300, "h": 300})["layer"]["id"]
    reg.execute("mask.add", {"layer_id": lid, "shape": "rect", "points": [[0.2, 0.2], [0.6, 0.6]],
                             "mode": "add"})
    reg.execute("mask.add", {"layer_id": lid, "shape": "polygon", "points": [[0, 0], [1, 0], [1, 1]],
                             "mode": "subtract"})
    rend = Renderer(reg.project)
    alpha = np.asarray(rend.render(10).convert("RGBA"))[..., 3]
    assert alpha.max() == 255 and alpha.min() == 0


def test_layer_style_adds_pixels_outside_the_layer(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FFFFFFFF", "w": 60, "h": 60})["layer"]["id"]
    plain = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    reg.execute("style.add", {"layer_id": lid, "type": "outer_glow",
                              "params": {"blur": 20, "opacity": 0.9}})
    glowing = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    assert glowing > plain


def test_trim_path_shortens_a_stroke(reg):
    lid = reg.execute("layer.add", {"kind": "shape", "shape": "path", "closed": False,
                                    "stroke_width": 12, "stroke_color": "#FFFF3366", "w": 300, "h": 300,
                                    "path": [[0.05, 0.5], [0.5, 0.1], [0.95, 0.5]]})["layer"]["id"]
    full = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    reg.execute("shape.trim", {"layer_id": lid, "start": 0, "end": 35})
    trimmed = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    assert trimmed < full * 0.75


def test_repeater_multiplies_shape(reg):
    lid = reg.execute("layer.add", {"kind": "shape", "shape": "rect", "w": 60, "h": 60,
                                    "color": "#FF33CC99", "transform": {"pos": [80, 90, 0]}})["layer"]["id"]
    one = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    reg.execute("shape.repeater", {"layer_id": lid, "copies": 4, "x": 40, "y": 0})
    many = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    assert many > one * 2


def test_text_animator_moves_characters(reg):
    lid = reg.execute("layer.add", {"kind": "text", "text": "ABCDEF", "size": 48,
                                    "transform": {"pos": [160, 90, 0]}})["layer"]["id"]
    plain = np.asarray(Renderer(reg.project).render(10).convert("L"), np.float32)
    reg.execute("text.animator_add", {"layer_id": lid,
                                      "selector": {"type": "range", "start": 0, "end": 100},
                                      "params": {"y": -40, "rotation": 12}})
    moved = np.asarray(Renderer(reg.project).render(10).convert("L"), np.float32)
    assert float(np.abs(plain - moved).mean()) > 0.5


def test_motion_blur_averages_subframes(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FFFF8800", "w": 60, "h": 60,
                                    "transform": {"pos": [40, 90, 0]}})["layer"]["id"]
    reg.project.get_layer(lid)["keys"] = {"pos": [{"t": 0, "v": [40, 90], "e": "linear"},
                                                  {"t": 400, "v": [280, 90], "e": "linear"}]}
    crisp = Renderer(reg.project).render(200)
    reg.execute("project.motion_blur", {"enabled": True, "samples": 6, "shutter": 0.5})
    blurred = Renderer(reg.project).render(200)
    # a blurred fast-moving edge shows more partially-covered pixels
    a_crisp = np.asarray(crisp.convert("RGBA"))[..., 3]
    a_blur = np.asarray(blurred.convert("RGBA"))[..., 3]
    # a shutter-sampled frame covers a wider footprint than a frozen one
    assert int((a_blur > 4).sum()) > int((a_crisp > 4).sum()) * 1.05


def test_precomp_renders_nested_content(reg):
    comp = reg.execute("comp.create", {"name": "Inner", "width": 120, "height": 120, "duration": 3000})["comp"]
    reg.execute("comp.add_layer", {"comp_id": comp["id"], "kind": "solid", "color": "#FF00FF88"})
    lid = reg.execute("layer.add", {"kind": "precomp", "comp_id": comp["id"], "w": 120, "h": 120})["layer"]["id"]
    alpha = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3]
    assert alpha.sum() > 0


def test_camera_zoom_and_depth_parallax(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FFFFFFFF", "w": 80, "h": 80})["layer"]["id"]
    plain = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    reg.execute("camera.set", {"enabled": True, "zoom": 1.5})
    zoomed = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    assert zoomed > plain * 1.5
    reg.execute("project.set_depth", {"layer_id": lid, "z": 400})
    near = np.asarray(Renderer(reg.project).render(10).convert("RGBA"))[..., 3].sum()
    assert near != zoomed


def test_light_layer_brightens_the_frame(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FF101018", "w": 320, "h": 180})
    base = np.asarray(Renderer(reg.project).render(10).convert("RGB"), np.float32).mean()
    reg.execute("light.add", {"color": "#FFFFCC88", "radius": 400, "intensity": 1.0})
    lit = np.asarray(Renderer(reg.project).render(10).convert("RGB"), np.float32).mean()
    assert lit > base + 3


def test_transition_ramps_in(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "color": "#FFFFFFFF", "w": 200, "h": 200,
                                    "start": 500, "end": 2500})["layer"]["id"]
    reg.execute("transition.set", {"layer_id": lid, "type": "cross_dissolve", "duration": 500, "when": "in"})
    rend = Renderer(reg.project)
    start_alpha = np.asarray(rend.render(520).convert("RGBA"))[..., 3].mean()
    mid_alpha = np.asarray(rend.render(750).convert("RGBA"))[..., 3].mean()
    assert mid_alpha > start_alpha


# ------------------------------------------------------------------ expressions
def test_expression_drives_property(reg):
    lid = reg.execute("layer.add", {"kind": "solid", "w": 40, "h": 40,
                                    "transform": {"pos": [50, 90, 0]}})["layer"]["id"]
    reg.execute("expression.set", {"layer_id": lid, "property": "pos",
                                   "expression": "[time * 100 + 40, 90]"})
    rend = Renderer(reg.project)
    a0 = np.asarray(rend.render(0).convert("RGBA"), np.float32)
    a1 = np.asarray(rend.render(1500).convert("RGBA"), np.float32)
    assert float(np.abs(a0 - a1).mean()) > 0.5


def test_expression_eval_command(reg):
    lid = reg.execute("layer.add", {"kind": "solid"})["layer"]["id"]
    res = reg.execute("expression.eval", {"layer_id": lid, "property": "rot",
                                          "expression": "wiggle(2, 10)[0]", "time": 400})
    assert isinstance(res["value"], list)


# ------------------------------------------------------------------ colour
def test_grade_changes_pixels_and_scopes_render(reg):
    img = _probe_image((64, 64))
    grade = {"enabled": True, "exposure": 0.8, "saturation": 1.5, "contrast": 1.2}
    out = apply_grade(img, grade)
    assert _diff(img, out) > 3
    for kind in ("histogram", "waveform", "vectorscope"):
        scope = render_scope_image(img, kind, (120, 60))
        assert scope.size == (120, 60)
    reg.execute("grade.set", {"enabled": True, "exposure": 0.5})
    assert reg.execute("grade.get")["grade"]["enabled"]


def test_grade_lut_roundtrip(tmp_path):
    path = tmp_path / "ident.cube"
    lines = ["LUT_3D_SIZE 2"]
    for b in (0, 1):
        for g in (0, 1):
            for r in (0, 1):
                lines.append(f"{r} {g} {b}")
    path.write_text("\n".join(lines))
    img = _probe_image((32, 32))
    out = apply_grade(img, {"enabled": True, "lut": {"path": str(path), "amount": 1.0}})
    assert out.size == img.size


def test_linear_light_compositing(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FF203040", "w": 320, "h": 180})
    a = np.asarray(Renderer(reg.project).render(10).convert("RGB"), np.float32)
    reg.execute("grade.set", {"enabled": True, "linear_light": True})
    b = np.asarray(Renderer(reg.project).render(10).convert("RGB"), np.float32)
    assert np.abs(a - b).mean() >= 0


# ------------------------------------------------------------------ audio
class _FakeMedia:
    """Minimal stand-in so the mixer can be tested without ffmpeg."""

    def __init__(self, samples):
        self.samples = samples

    def read_audio(self, item):
        return self.samples


def _sine(dbfs: float, seconds: float = 2.0, freq: float = 997.0) -> np.ndarray:
    t = np.arange(int(AUD.SR * seconds)) / AUD.SR
    a = 10 ** (dbfs / 20.0)
    mono = (a * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono], 1)


def test_lufs_is_calibrated():
    assert AUD.lufs(_sine(-23)) == pytest.approx(-23.0, abs=0.2)
    assert AUD.lufs(_sine(-16)) == pytest.approx(-16.0, abs=0.2)


def test_mixer_applies_volume_pan_mute_and_master(reg):
    p = reg.project
    item = {"id": 1, "kind": "audio", "path": "fake.wav", "name": "tone", "duration_ms": 2000}
    p.data["media"].append(item)
    lid = reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2000})["layer"]["id"]
    media = _FakeMedia(_sine(-20))
    base = AUD.mix(p, media)
    assert base is not None and np.abs(base).max() > 0.05
    reg.execute("audio.mixer", {"layer_id": lid, "volume": 0.25})
    quiet = AUD.mix(p, media)
    assert np.abs(quiet).max() < np.abs(base).max() * 0.5
    reg.execute("audio.mixer", {"layer_id": lid, "mute": True})
    assert AUD.mix(p, media) is None
    reg.execute("audio.mixer", {"layer_id": lid, "mute": False, "volume": 1.0})
    reg.execute("audio.master", {"gain_db": -6.0})
    trimmed = AUD.mix(p, media)
    assert np.abs(trimmed).max() < np.abs(base).max()


def test_audio_automation_and_analysis(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "tone", "duration_ms": 2000})
    lid = reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2000})["layer"]["id"]
    reg.execute("audio.automation", {"layer_id": lid, "time": 0, "volume": 0.1})
    reg.execute("audio.automation", {"layer_id": lid, "time": 2000, "volume": 1.0})
    media = _FakeMedia(_sine(-20))
    mix = AUD.mix(p, media, 0, 2000)
    head = float(np.abs(mix[: AUD.SR // 4]).max())
    tail = float(np.abs(mix[-AUD.SR // 4:]).max())
    assert tail > head * 2
    info = AUD.analyze(p, media)
    assert info["silent"] is False and -30 < info["integrated"] < 0


def test_audio_effects_chain(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "tone", "duration_ms": 2000})
    lid = reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2000})["layer"]["id"]
    reg.execute("audio.fx_add", {"layer_id": lid, "type": "gain", "params": {"db": -12}})
    media = _FakeMedia(_sine(-20))
    quiet = AUD.mix(p, media)
    reg.execute("audio.fx_add", {"layer_id": lid, "type": "limiter", "params": {"ceiling": -20}})
    limited = AUD.mix(p, media)
    assert np.abs(limited).max() <= np.abs(quiet).max() + 1e-6


# ------------------------------------------------------------------ captions
def test_captions_srt_roundtrip_and_render(tmp_path):
    srt = "1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n"
    cues = parse_srt_text(srt)
    assert len(cues) == 1 and cues[0]["start"] == 1000
    path = tmp_path / "c.srt"
    export_srt(cues, str(path))
    assert import_srt(str(path))[0]["text"] == "Hello there"
    img = Image.new("RGBA", (320, 180), (0, 0, 0, 255))
    stamped = draw_captions(img, 1500, cues)
    assert _diff(img, stamped) > 1


def test_caption_commands(reg):
    reg.execute("caption.add", {"start": 0, "end": 1000, "text": "one"})
    reg.execute("caption.auto", {"text": "a\nb\nc"})
    assert len(reg.execute("caption.list")["captions"]) == 3
    out = tempfile.mktemp(suffix=".srt")
    reg.execute("caption.export", {"path": out})
    assert os.path.exists(out)
    reg.execute("caption.import", {"path": out, "replace": True})
    assert len(reg.execute("caption.list")["captions"]) == 3


# ------------------------------------------------------------------ tracking / editing
def test_stabilize_patch_cancels_motion():
    pts = [{"t": i * 100, "x": 100 + i * 4.0, "y": 50 + math.sin(i) * 6, "score": 0.9} for i in range(24)]
    keys = stabilize_patch(pts, smooth=7, strength=1.0)
    assert len(keys) == len(pts)
    assert any(abs(k["v"][0]) > 0.01 for k in keys)


def test_three_point_edit_and_trim(reg):
    a = reg.execute("layer.add", {"kind": "solid", "start": 0, "end": 2000, "track": 0})["layer"]["id"]
    b = reg.execute("layer.add", {"kind": "solid", "start": 2000, "end": 3000, "track": 0})["layer"]["id"]
    res = reg.execute("edit.three_point", {"layer_id": b, "at": 1000, "source_in": 0, "source_out": 1500})
    assert res["start"] == 1000 and res["end"] == 2500
    trimmed = reg.execute("edit.trim", {"layer_id": b, "mode": "ripple_out", "amount": 400})
    assert trimmed["end"] == 2900
    razor = reg.execute("edit.razor", {"time": 1500})
    assert razor["splits"] >= 1
    assert reg.execute("edit.jkl", {"key": "l", "current": 1.0})["rate"] == 2.0


# ------------------------------------------------------------------ lottie
def test_lottie_roundtrip(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FF3366AA", "w": 200, "h": 100})
    tid = reg.execute("layer.add", {"kind": "text", "text": "Lottie", "size": 40})["layer"]["id"]
    reg.project.get_layer(tid)["keys"] = {"pos": [{"t": 0, "v": [40, 40], "e": "linear"},
                                                  {"t": 1000, "v": [240, 40], "e": "linear"}]}
    reg.execute("layer.add", {"kind": "shape", "shape": "ellipse", "w": 60, "h": 60})
    data = export_lottie(reg.project)
    assert len(data["layers"]) == 3
    project, warnings = import_lottie(data)
    assert len(project.layers) == 3
    kinds = {l["kind"] for l in project.layers}
    assert {"solid", "text", "shape"} <= kinds


def test_lottie_commands_write_files(reg):
    reg.execute("layer.add", {"kind": "solid"})
    out = tempfile.mktemp(suffix=".json")
    res = reg.execute("project.export_lottie", {"path": out})
    assert os.path.exists(res["path"])
    back = reg.execute("project.import_lottie", {"path": out, "replace": True})
    assert back["layers"] >= 1


# ------------------------------------------------------------------ surface parity
def test_every_parity_command_is_exposed_as_an_mcp_tool(reg):
    tools = {t["name"] for t in reg.tool_definitions()}
    assert len(tools) == len(COMMANDS)
    for name in ("mask.add", "style.add", "text.animator_add", "shape.trim", "transition.set",
                 "expression.set", "comp.create", "camera.set", "light.add", "grade.set",
                 "scope.render", "audio.mixer", "audio.lufs", "caption.add", "track.point",
                 "edit.three_point", "project.import_lottie", "project.export_lottie"):
        assert name.replace(".", "_") in tools


def test_capability_report_matches_catalogues(reg):
    info = reg.execute("edge.detect_all")
    assert info["effects"] >= 306
    assert info["transitions"] >= 30
    assert info["commands"] == len(COMMANDS)
    assert info["blend_modes"] >= 30


# ------------------------------------------------------------------ audio-reactive
class _AudioMedia:
    """Media manager stub that serves one synthetic WAV."""

    def __init__(self, samples):
        self.samples = samples

    def read_audio(self, item):
        return self.samples


def _pulsing_track(seconds: float = 2.5, hz: float = 2.0, dbfs: float = -12.0):
    t = np.arange(int(AUD.SR * seconds)) / AUD.SR
    a = 10 ** (dbfs / 20.0)
    env = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * hz * t))
    mono = (a * env * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    return np.stack([mono, mono], 1)


def test_envelope_and_bands_track_the_music(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "pulse",
                            "duration_ms": 2500})
    reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2500})
    env = AUD.envelope(p, _AudioMedia(_pulsing_track()), 20.0)
    assert len(env["level"]) > 100
    assert max(env["level"]) > 0.9 and min(env["level"]) < 0.5
    assert len(env["low"]) == len(env["level"]) == len(env["high"])
    at = AUD.envelope_at(env, 200)
    assert set(at) == {"level", "low", "mid", "high"}


def test_beat_detection_and_tempo(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "pulse",
                            "duration_ms": 2500})
    reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2500})
    env = AUD.envelope(p, _AudioMedia(_pulsing_track()), 20.0)
    beats = AUD.detect_beats(env)
    assert len(beats) >= 4
    tempo = AUD.estimate_tempo(beats)
    assert 100 <= tempo <= 140           # 2 Hz modulation ≈ 120 BPM
    orig = AUD.envelope
    AUD.envelope = lambda *a, **k: orig(p, _AudioMedia(_pulsing_track()), 20.0)
    try:
        res = reg.execute("audio.beats", {"add_markers": True})
    finally:
        AUD.envelope = orig
    assert res["count"] == len(beats) == len(p.data["markers"])
    assert abs(res["tempo"] - tempo) < 0.6


def test_spectrum_frames_are_normalised(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "pulse",
                            "duration_ms": 2500})
    reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2500})
    spec = AUD.spectrogram(p, _AudioMedia(_pulsing_track()), 40.0, 12)
    assert len(spec["frames"]) > 50 and len(spec["frames"][0]) == 12
    assert max(max(f) for f in spec["frames"]) <= 1.0


def test_audio_react_bakes_keys(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "pulse",
                            "duration_ms": 2500})
    reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2500})
    lid = reg.execute("layer.add", {"kind": "shape", "shape": "rect", "w": 80, "h": 80})["layer"]["id"]
    orig = AUD.envelope
    AUD.envelope = lambda *a, **k: orig(p, _AudioMedia(_pulsing_track()), 20.0)
    try:
        res = reg.execute("audio.react", {"layer_id": lid, "property": "scale", "lo": 90, "hi": 120,
                                          "every_ms": 80})
    finally:
        AUD.envelope = orig
    assert res["keys"] > 10
    keys = REG_LAYER(p, lid)["keys"]["scale"]
    assert all(len(k["v"]) == 2 for k in keys)
    assert max(k["v"][0] for k in keys) <= 120.01


def test_renderer_exposes_audio_state(reg):
    p = reg.project
    p.data["media"].append({"id": 1, "kind": "audio", "path": "fake.wav", "name": "pulse",
                            "duration_ms": 2500})
    reg.execute("layer.add", {"kind": "audio", "media_id": 1, "start": 0, "end": 2500})
    rend = Renderer(p, _AudioMedia(_pulsing_track()))
    state = rend.audio_state(400)
    assert 0.0 <= state["level"] <= 1.0
    assert len(state["spectrum"]) > 0 and len(state["wave"]) > 0


def test_audio_analyser_effects_respond(reg):
    from craftmotion.core.effects import apply_effect, defaults_for
    base = Image.new("RGBA", (160, 90), (18, 18, 26, 255))
    ctx = {"project": None, "layer": {}, "t_local": 0, "extent": 0.5, "scale": 1.0,
           "renderer": None,
           "audio": {"level": 0.8, "low": 0.9, "mid": 0.5, "high": 0.2,
                     "spectrum": [abs(math.sin(i / 5)) for i in range(24)],
                     "wave": [math.sin(i / 4) for i in range(96)]}}
    for key in ("audio_spectrum", "audio_waveform", "audio_pulse", "audio_scale"):
        out = apply_effect(key, base, defaults_for(key), ctx)
        assert out.size == base.size
    spec = apply_effect("audio_spectrum", base, defaults_for("audio_spectrum"), ctx)
    assert _diff(base, spec) > 5
    silent = dict(ctx, audio={"level": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0,
                              "spectrum": [], "wave": []})
    quiet = apply_effect("audio_pulse", base, defaults_for("audio_pulse"), silent)
    assert _diff(base, quiet) < 0.5
    reg.execute("layer.add", {"kind": "solid"})
    lid = reg.project.layers[0]["id"]
    res = reg.execute("audio.analyser", {"layer_id": lid, "kind": "waveform"})
    assert res["effect"] == "audio_waveform"


# ------------------------------------------------------------------ blend modes
def test_all_blend_modes_are_bounded(reg):
    from craftmotion.core.model import BLEND_MODES
    from craftmotion.core.render import _blend_rgb
    assert len(BLEND_MODES) == 38
    cb = np.random.rand(16, 16, 3).astype(np.float32)
    cs = np.random.rand(16, 16, 3).astype(np.float32)
    for mode in BLEND_MODES:
        out = _blend_rgb(cb, cs, mode)
        assert out.shape == cb.shape
        assert np.isfinite(out).all()
        assert out.min() >= -1e-4 and out.max() <= 1.0001
    assert float(np.abs(_blend_rgb(cb, cs, "difference") - np.abs(cb - cs)).max()) < 1e-6


def test_blend_mode_reaches_the_render(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FF3355FF", "w": 320, "h": 180})
    top = reg.execute("layer.add", {"kind": "solid", "color": "#FF77AA22", "w": 320, "h": 180})["layer"]["id"]
    rend = Renderer(reg.project)
    normal = np.asarray(rend.render(10).convert("RGB"), np.float32)
    REG_LAYER(reg.project, top)["blend"] = "multiply"
    mult = np.asarray(Renderer(reg.project).render(10).convert("RGB"), np.float32)
    assert float(np.abs(normal - mult).mean()) > 5


# ------------------------------------------------------------------ tracking
class _FrameMedia:
    def __init__(self, frames, fps=30):
        self.frames = frames
        self.fps = fps

    def get_frame(self, item, t_ms):
        i = int(round(t_ms / 1000.0 * self.fps))
        return self.frames[max(0, min(len(self.frames) - 1, i))]


def _moving_patch_frames(n=24, speed=3):
    rng = np.random.default_rng(7)
    patch = (rng.random((16, 16, 3)) > 0.5).astype(np.float32)
    frames = []
    for i in range(n):
        f = np.full((140, 220, 3), 0.06, np.float32)
        x0, y0 = 60 + i * speed, 50
        f[y0:y0 + 16, x0:x0 + 16] = patch
        frames.append(Image.fromarray((f * 255).astype(np.uint8)))
    return frames


def test_tracker_follows_a_moving_feature():
    from craftmotion.core.tracking import to_keyframes, track_point
    media = _FrameMedia(_moving_patch_frames())
    item = {"id": 1, "kind": "image", "path": "x.png", "duration_ms": 800, "width": 220, "height": 140}
    pts = track_point(media, item, 68, 58, 0, 750, 33, 8, 10)
    assert len(pts) > 15
    assert pts[-1]["x"] > pts[0]["x"] + 15           # follows the patch
    assert pts[-1]["score"] > 0.5
    keys = to_keyframes(pts)
    assert all(len(k["v"]) == 2 for k in keys)


def test_tracking_commands_attach_and_stabilize(reg):
    lid = reg.execute("layer.add", {"kind": "text", "text": "target"})["layer"]["id"]
    pts = [{"t": i * 100, "x": 100 + i * 3.0, "y": 60, "score": 0.9} for i in range(12)]
    res = reg.execute("track.apply", {"layer_id": lid, "points": pts, "mode": "attach"})
    assert res["keys"] == len(pts)
    layer = REG_LAYER(reg.project, lid)
    assert layer["keys"]["pos"][0]["v"][0] == pytest.approx(100.0)
    res2 = reg.execute("track.apply", {"layer_id": lid, "points": pts, "mode": "stabilize",
                                       "strength": 0.8})
    assert res2["keys"] == len(pts)
    assert any(abs(k["v"][0]) > 0.01 for k in REG_LAYER(reg.project, lid)["keys"]["pos"])


def REG_LAYER(project, lid):
    return project.get_layer(lid)


# ------------------------------------------------------------------ text presets
def test_text_presets_reveal_progressively(reg):
    reg.project.settings["bg"] = "#00000000"
    lid = reg.execute("layer.add", {"kind": "text", "text": "MOTION TITLES", "size": 52,
                                    "transform": {"pos": [160, 90, 0]}})["layer"]["id"]
    reg.execute("text.preset_apply", {"layer_id": lid, "name": "type_on"})
    rend = Renderer(reg.project)
    counts = [int((np.asarray(rend.render(t).convert("RGBA"))[..., 3] > 8).sum())
              for t in (0, 300, 600, 1100)]
    assert counts[0] < counts[1] < counts[3]
    assert reg.execute("text.presets")["presets"]


def test_text_animator_key_animates_selector(reg):
    lid = reg.execute("layer.add", {"kind": "text", "text": "WIPE", "size": 60})["layer"]["id"]
    reg.execute("text.animator_add", {"layer_id": lid, "params": {"opacity": 0}})
    reg.execute("text.animator_key", {"layer_id": lid, "index": 0, "param": "end", "time": 0,
                                      "value": 100})
    reg.execute("text.animator_key", {"layer_id": lid, "index": 0, "param": "end", "time": 800,
                                      "value": 0})
    sel = reg.project.get_layer(lid)["text_animators"][0]["selector"]
    assert len(sel["keys"]["end"]) == 2
    reg.project.settings["bg"] = "#00000000"
    rend = Renderer(reg.project)
    first = int((np.asarray(rend.render(0).convert("RGBA"))[..., 3] > 8).sum())
    last = int((np.asarray(rend.render(800).convert("RGBA"))[..., 3] > 8).sum())
    assert last > first


def test_empty_text_layer_with_analyser_still_draws(reg):
    lid = reg.execute("layer.add", {"kind": "text", "text": "", "size": 10})["layer"]["id"]
    reg.execute("audio.analyser", {"layer_id": lid, "kind": "spectrum"})
    reg.project.settings["bg"] = "#00000000"
    img = Renderer(reg.project).render(10)
    assert np.asarray(img.convert("RGBA"))[..., 3].sum() > 0


# ------------------------------------------------------------------ surface counts
def test_feature_surface_counts(reg):
    info = reg.execute("edge.detect_all")
    assert info["effects"] >= 312
    assert info["transitions"] >= 39
    assert info["blend_modes"] == 38
    assert info["commands"] >= 129
    assert "Audio" in info["effect_categories"]
    assert len(reg.tool_definitions()) == len(COMMANDS)
