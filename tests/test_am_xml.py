"""Alight Motion XML import/export tests."""
import os
import tempfile

import pytest

from craftmotion.core.am_xml import export_am_scene, import_am_scene, validate_am_xml
from craftmotion.core.registry import Registry

PRESET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "samples", "am_presets", "neon_title.am.xml")


def test_validate_sample():
    v = validate_am_xml(PRESET)
    assert v["ok"], v["issues"]
    assert v["elements"] == 5 and v["media"] == 0


def test_import_merges_keyframes_and_parents():
    proj, res = import_am_scene(PRESET)
    assert res.stats["layers"] == 5
    assert res.stats["keyframes"] >= 10
    ids = [l["id"] for l in proj.layers]
    null = proj.layers[0]
    assert null["kind"] == "null"
    children = [l for l in proj.layers if l.get("parent")]
    assert len(children) == 4, "AM parent ids must be remapped to new layer ids"
    assert all(c["parent"] == null["id"] for c in children)
    # gradient arrives as two colours
    backdrop = next(l for l in proj.layers if l["name"] == "Backdrop")
    assert backdrop.get("color") and backdrop.get("color2")


def test_import_preset_merge_offsets_layers():
    reg = Registry()
    reg.execute("project.new", {"title": "host", "width": 1080, "height": 1920, "duration_ms": 4000})
    res = reg.execute("project.import_am_xml", {"path": PRESET, "merge": True, "offset_ms": 2000})
    assert res["stats"]["layers"] >= 5
    starts = [l["start"] for l in reg.project.layers]
    assert any(s >= 2000 for s in starts)
    assert reg.project.duration >= 8000


def test_roundtrip_export_is_valid():
    reg = Registry()
    reg.execute("project.import_am_xml", {"path": PRESET})
    out = tempfile.mktemp(suffix=".xml")
    reg.execute("project.export_am_xml", {"path": out})
    v = validate_am_xml(out)
    assert v["ok"], v["issues"]
    proj2, res2 = import_am_scene(out)
    assert res2.stats["layers"] == 5


def test_unknown_effect_is_warned_not_fatal():
    xml = """<?xml version='1.0'?>
    <scene title="t" width="100" height="100" totalTime="1000" fps="30" bgcolor="#FF000000">
      <shape id="1" label="R" startTime="0" endTime="1000" fillType="color" s=".rect">
        <transform><location value="50,50,0"/><scale value="1,1"/></transform>
        <effect id="com.alightcreative.effects.doesnotexist" locallyApplied="true"/>
        <property name="size" type="vec2" value="50,50"/>
      </shape>
    </scene>"""
    path = tempfile.mktemp(suffix=".xml")
    open(path, "w").write(xml)
    proj, res = import_am_scene(path)
    assert res.stats["layers"] == 1
    assert any("doesnotexist" in w for w in res.warnings)


def test_media_missing_is_reported():
    xml = """<?xml version='1.0'?>
    <scene title="t" width="100" height="100" totalTime="1000" fps="30" bgcolor="#FF000000">
      <media uri="am-internal:///abc123.jpg" filename="abc123.JPG" type="image/jpeg" sig="abc123"/>
      <shape id="1" label="Pic" startTime="0" endTime="1000" fillType="media"
             fillImage="am-internal:///abc123.jpg">
        <transform><location value="50,50,0"/></transform>
      </shape>
    </scene>"""
    path = tempfile.mktemp(suffix=".xml")
    open(path, "w").write(xml)
    proj, res = import_am_scene(path)
    assert res.stats["media"] == 1
    assert proj.media[0]["missing"] is True
    assert any("not found" in w for w in res.warnings)
