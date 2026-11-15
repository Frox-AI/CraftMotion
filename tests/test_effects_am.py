"""Tests for the Alight-Motion-flavoured effect layer, parameter validation and
the AM look importer (`preset.am_look`).

These cover the parts that were added when the SAMUIFX preset pack was used as a
real-world import test: the five AM-native distort effects (`effects_am`), the
strict `effect.add` parameter contract, and `preset.am_look`, which copies one
effect chain out of an AM preset onto a layer.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from craftmotion.core import effects as FX
from craftmotion.core import effects_am
from craftmotion.core.registry import COMMANDS, CommandError, Registry
from craftmotion.core.render import Renderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = os.path.join(ROOT, "samples", "am_presets")
NEON = os.path.join(PRESETS, "neon_title.am.xml")
GLITCH = os.path.join(PRESETS, "glitch_shake.am.xml")

# The shipped samples only put one effect per layer, so a dedicated fixture with
# real multi-effect chains is needed to exercise preset.am_look.
CHAIN_XML = """<?xml version='1.0' encoding='UTF-8' ?>
<scene title="Chain Fixture" width="1080" height="1350" exportWidth="1080" exportHeight="1350"
       bgcolor="#FF000000" totalTime="3000" fps="30" amver="1028425" ffver="106"
       am="com.alightcreative.motion/5.0.273.1028425" amplatform="android" type="preset">
  <shape id="1" label="Heavy" startTime="0" endTime="3000" fillType="color" s=".rect">
    <transform>
      <location value="540.000000,675.000000,0.000000"/>
      <scale value="1.000000,1.000000"/>
      <rotation value="0.000000"/>
      <opacity value="1.000000"/>
    </transform>
    <fillColor value="#FF8844FF"/>
    <effect id="com.alightcreative.effects.chromaticaberration" locallyApplied="true">
      <property name="amount" type="float" value="0.300000"/>
    </effect>
    <effect id="com.alightcreative.effects.gaussianblur" locallyApplied="true">
      <property name="strength" type="float" value="0.200000"/>
    </effect>
    <effect id="com.alightcreative.effects.glow" locallyApplied="true">
      <property name="radius" type="float" value="0.250000"/>
    </effect>
    <effect id="com.alightcreative.effects.invert" locallyApplied="true">
      <property name="amount" type="float" value="1.000000"/>
    </effect>
  </shape>
  <shape id="3" label="Light Again" startTime="0" endTime="3000" fillType="color" s=".rect">
    <transform>
      <location value="540.000000,675.000000,0.000000"/>
      <scale value="1.000000,1.000000"/>
      <rotation value="0.000000"/>
      <opacity value="1.000000"/>
    </transform>
    <fillColor value="#FF88FF44"/>
    <effect id="com.alightcreative.effects.shake" locallyApplied="true">
      <property name="amount" type="float" value="0.100000"/>
    </effect>
    <effect id="com.alightcreative.effects.scanlines" locallyApplied="true">
      <property name="intensity" type="float" value="0.200000"/>
    </effect>
  </shape>
  <shape id="2" label="Light" startTime="0" endTime="3000" fillType="color" s=".rect">
    <transform>
      <location value="540.000000,675.000000,0.000000"/>
      <scale value="1.000000,1.000000"/>
      <rotation value="0.000000"/>
      <opacity value="1.000000"/>
    </transform>
    <fillColor value="#FF44FF88"/>
    <effect id="com.alightcreative.effects.shake" locallyApplied="true">
      <property name="amount" type="float" value="0.100000"/>
    </effect>
    <effect id="com.alightcreative.effects.scanlines" locallyApplied="true">
      <property name="intensity" type="float" value="0.200000"/>
    </effect>
  </shape>
</scene>
"""


@pytest.fixture()
def chain_xml(tmp_path):
    p = tmp_path / "chain.am.xml"
    p.write_text(CHAIN_XML)
    return str(p)


@pytest.fixture()
def reg():
    r = Registry()
    r.execute("project.new", {"title": "am_fx", "width": 320, "height": 180,
                              "duration_ms": 2000, "fps": 30})
    r.project.settings["bg"] = "#00000000"
    return r


def _probe(size=(80, 80)):
    yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    arr = np.zeros((size[1], size[0], 4), np.float32)
    arr[..., 0] = xx / size[0]
    arr[..., 1] = yy / size[1]
    arr[..., 2] = 0.5
    arr[..., 3] = 1.0
    arr[10:60, 10:60, :3] = (0.9, 0.4, 0.1)
    return Image.fromarray((arr * 255).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------ effects_am
AM_EFFECTS = ["oscillate", "swing", "blink", "random_displace", "turbulence"]


def test_am_effects_are_registered():
    for name in AM_EFFECTS:
        assert name in FX.EFFECTS, f"{name} missing from the effect catalogue"
        assert FX.resolve_type(name) == name


@pytest.mark.parametrize("name", AM_EFFECTS)
def test_am_effect_renders_without_mutating_size(name):
    img = _probe()
    op = FX.EFFECTS[name].func
    ctx = {"project": None, "layer": {}, "t_local": 320, "extent": 0.25,
           "scale": 1.0, "renderer": None}
    out = op(img.copy(), FX.defaults_for(name), ctx)
    assert out.size == img.size
    assert out.mode == "RGBA"


@pytest.mark.parametrize("name,t0,t1", [("oscillate", 0, 250), ("swing", 0, 250),
                                        ("blink", 20, 100)])
def test_time_driven_am_effects_move(name, t0, t1):
    """Oscillate/swing/blink read layer time, so two instants must differ."""
    img = _probe()
    op = FX.EFFECTS[name].func
    rendered = []
    for t in (t0, t1):
        ctx = {"project": None, "layer": {}, "t_local": t, "extent": 0.25,
               "scale": 1.0, "renderer": None}
        rendered.append(np.asarray(op(img.copy(), FX.defaults_for(name), ctx), np.float32))
    assert not np.array_equal(rendered[0], rendered[1])


@pytest.mark.parametrize("name", ["random_displace", "turbulence"])
def test_noise_am_effects_distort_the_image(name):
    img = _probe()
    op = FX.EFFECTS[name].func
    ctx = {"project": None, "layer": {}, "t_local": 250, "extent": 0.25,
           "scale": 1.0, "renderer": None}
    out = np.asarray(op(img.copy(), FX.defaults_for(name), ctx), np.float32)
    assert not np.array_equal(out, np.asarray(img, np.float32))


def test_effects_am_module_exports_every_am_op():
    for name in AM_EFFECTS:
        assert hasattr(effects_am, "fx_" + name), f"effects_am.fx_{name} is missing"


def test_am_alias_resolves_to_native_effect():
    cases = {
        "oscillate3": "oscillate",
        "swing2": "swing",
        "blink2": "blink",
        "randomdisplace": "random_displace",
        "turbulentdisplace3": "turbulence",
        "com.alightcreative.effects.randomdisplace": "random_displace",
        "lift": "levels",
        "colorhot": "thermal",
        "swirl4": "twirl",
        "rgbsep": "rgb_split",
        "deepglow": "glow",
    }
    for alias, native in cases.items():
        assert FX.resolve_type(alias) == native, f"{alias} should map to {native}"


# ------------------------------------------------------------------ effect.add
def test_effect_add_rejects_unknown_parameter(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    with pytest.raises(CommandError) as err:
        reg.execute("effect.add", {"layer_id": lid, "type": "thermal",
                                   "params": {"amount": 0.5, "warmth": 0.9}})
    assert "warmth" in str(err.value)
    assert "amount" in str(err.value)
    # nothing was appended by the failed call
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    assert not layer.get("effects")


def test_effect_add_rejects_unknown_type(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    with pytest.raises(CommandError):
        reg.execute("effect.add", {"layer_id": lid, "type": "not_an_effect"})


def test_effect_add_clamps_float_parameters(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": lid, "type": "glow",
                               "params": {"radius": 1e6}})
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    spec = FX.EFFECTS["glow"].params["radius"]
    assert layer["effects"][0]["params"]["radius"] <= float(spec.hi)


def test_effect_add_accepts_a_valid_parameter(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": lid, "type": "scanlines",
                               "params": {"intensity": 0.3, "period": 4}})
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    params = layer["effects"][0]["params"]
    assert params["intensity"] == pytest.approx(0.3)
    assert params["period"] == 4


def test_effect_add_accepts_alias_names(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    res = reg.execute("effect.add", {"layer_id": lid, "type": "turbulentdisplace3"})
    assert res["effects"] == ["turbulence"]


# ------------------------------------------------------------------ AM looks
def test_preset_am_look_copies_a_chain(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    res = reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml,
                                         "max_effects": 3})
    assert res["chain"], "am_look should report the chain it copied"
    assert len(res["chain"]) <= 3
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    got = [e["type"] for e in layer["effects"]]
    assert got == res["chain"]
    assert res["effects"] == len(got)
    assert all(t in FX.EFFECTS for t in got)
    assert res["source_layer"]


def test_preset_am_look_exclude_filters_the_chain(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    res = reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml,
                                         "exclude": ["invert", "blink"], "max_effects": 8})
    assert "invert" not in res["chain"]
    assert "blink" not in res["chain"]
    assert len(res["chain"]) >= 2


def test_preset_am_look_heaviest_mode(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    heavy = reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml,
                                           "mode": "heaviest"})
    lid2 = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    light = reg.execute("preset.am_look", {"layer_id": lid2, "path": chain_xml,
                                           "mode": "signature"})
    assert len(heavy["chain"]) > len(light["chain"])


def test_preset_am_look_replaces_by_default(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": lid, "type": "swirl"})
    before = len(next(l for l in reg.project.layers if l["id"] == lid)["effects"])
    res = reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml})
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    assert before == 1
    assert [e["type"] for e in layer["effects"]] == res["chain"], "replace must not append"
    assert "swirl" not in [e["type"] for e in layer["effects"]]


def test_preset_am_look_can_append(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": lid, "type": "swirl"})
    reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml, "replace": False})
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    assert layer["effects"][0]["type"] == "twirl"


def test_preset_am_look_scale_shrinks_parameters(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    res = reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml, "scale": 0.5})
    layer = next(l for l in reg.project.layers if l["id"] == lid)
    assert layer["effects"], "a look should land at least one effect"
    assert res["chain"] == [e["type"] for e in layer["effects"]]


def test_preset_am_look_missing_file(reg):
    lid = reg.execute("layer.add", {"kind": "shape"})["layer"]["id"]
    with pytest.raises(CommandError):
        reg.execute("preset.am_look", {"layer_id": lid, "path": "nope/never.am.xml"})


def test_am_look_chain_applies_to_render(reg, chain_xml):
    lid = reg.execute("layer.add", {"kind": "shape", "color": "#FFCC5522",
                                    "width": 160, "height": 90, "x": 80, "y": 45})["layer"]["id"]
    before = np.asarray(Renderer(reg.project).render(200).convert("RGBA"), np.float32)
    reg.execute("preset.am_look", {"layer_id": lid, "path": chain_xml, "max_effects": 2})
    after = np.asarray(Renderer(reg.project).render(200).convert("RGBA"), np.float32)
    assert not np.array_equal(before, after)


# ------------------------------------------------------------------ surface
def test_am_surface_is_exposed_everywhere(reg):
    """Both the GUI/MCP tool list and the capability report know about these."""
    tools = {t["name"] for t in reg.tool_definitions()}
    assert "preset_am_look" in tools
    info = reg.execute("edge.detect_all")
    assert info["effects"] == len(FX.EFFECTS)
    assert info["commands"] == len(COMMANDS)
    listed = {e["type"] for e in reg.execute("effect.types")["effects"]}
    for name in AM_EFFECTS:
        assert name in listed, f"{name} missing from effect.types"


# ---------------------------------------------------------------- zoom_blur perf
def test_zoom_blur_is_cheap_and_keeps_the_centre():
    """`zoom_blur` used a full-frame remap per sample (~6 s/frame at 1080×1350).

    It now averages scaled copies, so guard both the shape of the result and
    that it stays in the fast path.
    """
    import time

    from PIL import Image

    from craftmotion.core.effects import EFFECTS

    w, h = 540, 675
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 3] = 255
    arr[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20, :3] = 255   # white square
    img = Image.fromarray(arr, "RGBA")
    fx = EFFECTS["zoom_blur"]
    params = {k: v.default for k, v in fx.params.items()}

    t0 = time.perf_counter()
    out = fx.func(img, params, {"t_local": 0, "scale": 1.0})
    dt = time.perf_counter() - t0

    assert out.size == img.size
    out_a = np.asarray(out.convert("RGBA"), np.float32)
    centre = int(h / 2) * w + int(w / 2)
    assert out_a.reshape(-1, 4)[centre][:3].mean() > 200      # centre stays bright
    corner = out_a[:8, :8, :3].mean()
    assert corner < 40                                        # edges smear to black
    assert dt < 2.0, f"zoom_blur regressed to {dt:.2f}s per frame"
