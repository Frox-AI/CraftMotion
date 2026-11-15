"""Engine tests: keyframes, model, effects, render, registry."""
import math
import os
import tempfile

import pytest

from craftmotion.core.registry import Registry
from craftmotion.core.util import bezier_ease, kf_add, kf_eval, parse_color, parse_easing, to_argb


# ---------------------------------------------------------------- keyframes

def test_keyframe_linear_interpolation():
    keys = []
    kf_add(keys, 0, [0.0])
    kf_add(keys, 1000, [10.0])
    assert kf_eval(keys, 500, [0])[0] == pytest.approx(5.0)
    assert kf_eval(keys, -100, [0])[0] == 0.0
    assert kf_eval(keys, 99999, [0])[0] == 10.0


def test_keyframe_easing_and_hold():
    keys = []
    kf_add(keys, 0, [0.0], "hold")
    kf_add(keys, 1000, [1.0])
    assert kf_eval(keys, 500, [0])[0] == 0.0
    keys2 = []
    kf_add(keys2, 0, [0.0], "ease_in_out")
    kf_add(keys2, 1000, [1.0])
    mid = kf_eval(keys2, 500, [0])[0]
    assert 0.3 < mid < 0.7


def test_keyframe_vectors():
    keys = []
    kf_add(keys, 0, [0, 0])
    kf_add(keys, 1000, [100, 200])
    v = kf_eval(keys, 500, [0, 0])
    assert v[0] == pytest.approx(50) and v[1] == pytest.approx(100)


def test_bezier_ease_monotonic():
    e = parse_easing("cubicBezier 0.42 0 0.58 1")
    vals = [bezier_ease(e, u / 10) for u in range(11)]
    assert vals[0] == pytest.approx(0, abs=1e-6)
    assert vals[-1] == pytest.approx(1, abs=1e-6)
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:]))


def test_color_roundtrip():
    assert to_argb(parse_color("#FF7A1F")) == "#FFFF7A1F"
    assert to_argb(parse_color("#80112233")) == "#80112233"
    assert parse_color("nope", (1, 0, 0, 1)) == (1, 0, 0, 1)


# ---------------------------------------------------------------- registry

@pytest.fixture()
def reg():
    r = Registry()
    r.execute("project.new", {"title": "T", "width": 320, "height": 240, "duration_ms": 3000})
    return r


def test_project_layer_crud(reg):
    l = reg.execute("layer.add", {"kind": "shape", "shape": "star"})["layer"]
    assert reg.project.get_layer(l["id"]) is not None
    reg.execute("layer.set_transform", {"layer_id": l["id"], "pos": [10, 20], "rot": 45})
    tf = reg.project.get_layer(l["id"])["transform"]
    assert tf["pos"][:2] == [10, 20] and tf["rot"][0] == 45
    reg.execute("layer.delete", {"layer_id": l["id"]})
    assert reg.project.get_layer(l["id"]) is None


def test_undo_redo(reg):
    reg.execute("layer.add", {"kind": "solid"})
    n = len(reg.project.layers)
    reg.execute("layer.add", {"kind": "text"})
    assert len(reg.project.layers) == n + 1
    reg.execute("project.undo")
    assert len(reg.project.layers) == n
    reg.execute("project.redo")
    assert len(reg.project.layers) == n + 1


def test_effect_stack_ops(reg):
    lid = reg.execute("layer.add", {"kind": "solid"})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": lid, "type": "glow"})
    reg.execute("effect.add", {"layer_id": lid, "type": "vignette"})
    effs = reg.execute("effect.list", {"layer_id": lid})["effects"]
    assert [e["type"] for e in effs] == ["glow", "vignette"]
    reg.execute("effect.move", {"layer_id": lid, "from_index": 1, "to_index": 0})
    effs = reg.execute("effect.list", {"layer_id": lid})["effects"]
    assert [e["type"] for e in effs] == ["vignette", "glow"]
    reg.execute("effect.set_param", {"layer_id": lid, "effect_index": 1, "param": "radius", "value": 30})
    reg.execute("effect.set_param", {"layer_id": lid, "effect_index": 1, "param": "radius",
                                     "value": 5, "t": 0})
    assert reg.execute("effect.list", {"layer_id": lid})["effects"][1]["keyed"] == ["radius"]


def test_looks_and_presets(reg):
    lid = reg.execute("layer.add", {"kind": "solid"})["layer"]["id"]
    reg.execute("effect.apply_look", {"layer_id": lid, "look": "cinematic_teal"})
    assert len(reg.execute("effect.list", {"layer_id": lid})["effects"]) >= 3
    reg.execute("preset.save", {"name": "my look", "layer_id": lid})
    lid2 = reg.execute("layer.add", {"kind": "solid"})["layer"]["id"]
    reg.execute("preset.apply", {"name": "my look", "layer_id": lid2})
    assert len(reg.execute("effect.list", {"layer_id": lid2})["effects"]) >= 3


def test_unknown_command_errors(reg):
    from craftmotion.core.registry import CommandError
    with pytest.raises(CommandError):
        reg.execute("does.not.exist")
    with pytest.raises(CommandError):
        reg.execute("effect.add", {"layer_id": 1, "type": "not_an_effect"})


def test_tools_cover_commands():
    from craftmotion.core.registry import COMMANDS
    tools = Registry().tool_definitions()
    assert len(tools) == len(COMMANDS)
    names = {t["name"] for t in tools}
    assert "layer_add" in names and "project_import_am_xml" in names


# ---------------------------------------------------------------- render

def test_render_frame_size_and_content(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FFFF0000",
                              "w": 320, "h": 240})
    img = reg.execute("render.still", {"t": 0, "path": tempfile.mktemp(suffix=".png"),
                                       "width": 160, "height": 120})
    from PIL import Image
    im = Image.open(img["path"]).convert("RGB")
    assert im.size == (160, 120)
    r, g, b = im.getpixel((80, 60))
    assert r > 180 and g < 90


def test_every_effect_renders(reg):
    from craftmotion.core import effects as FX
    from craftmotion.core.render import Renderer
    lid = reg.execute("layer.add", {"kind": "shape", "shape": "star",
                                    "color": "#FF33AAFF", "w": 120, "h": 120})["layer"]["id"]
    r = Renderer(reg.project, reg.media)
    base = r.render(100, size=(160, 120))
    for key in FX.EFFECTS:
        stack = reg.project.get_layer(lid).setdefault("effects", [])
        stack.clear()
        stack.append({"type": key, "enabled": True, "params": {}, "keys": {}})
        out = r.render(100, size=(160, 120))
        assert out.size == (160, 120), key
    r.close()


# ---------------------------------------------------------------- export

def test_gif_export(reg):
    reg.execute("layer.add", {"kind": "solid", "color": "#FF2266FF"})
    out = tempfile.mktemp(suffix=".gif")
    res = reg.execute("render.gif", {"path": out, "end_ms": 400, "fps": 10, "width": 120, "wait": True})
    assert res["job"]["status"] == "done"
    assert os.path.exists(out) and os.path.getsize(out) > 100


def test_render_is_resolution_independent(reg):
    """A 540px layer must cover the same fraction of frame at any preview size."""
    import numpy as np
    from craftmotion.core.render import Renderer
    lid = reg.execute("layer.add", {"kind": "shape", "shape": "rect", "w": 120, "h": 120,
                                    "color": "#FFFF0000"})["layer"]["id"]
    r = Renderer(reg.project, reg.media)
    try:
        for size in ((160, 120), (320, 240)):
            arr = np.asarray(r.render(0, size=size).convert("RGB"))
            mask = (arr[..., 0] > 150) & (arr[..., 1] < 100)
            xs = np.where(mask)[1]
            width = xs.max() - xs.min() + 1
            expected = 120 * size[0] / reg.project.width
            assert abs(width - expected) <= 3, (size, width, expected)
    finally:
        r.close()
