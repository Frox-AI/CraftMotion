#!/usr/bin/env python3
"""Build "SIGNAL / NOISE" — an original edit made with CraftMotion.

The look is lifted from the five Alight Motion XML presets SAMUIFX ships with the
video this was made to test (AE INSPIRED 5 ANIME EDIT PRESETS), applied to our
own layers with `preset.am_look`. Cuts, timing, type and grade are ours.

Sources (public domain / CC0 — see edits/fetch_assets.py):
  · Superman: The Mechanical Monsters (Fleischer, 1941) — PD, Internet Archive
  · Superman: Electric Earthquake (Fleischer, 1942) — PD, Internet Archive
  · Solar flare (SDO 2016) + black-hole simulation (2018) — NASA, PD
  · "Midnight Club" — Alec Penn Radio, CC0 1.0, Internet Archive

    python edits/fetch_assets.py --dest /tmp/pd_assets      # sources + 720x480 cuts + presets
    python edits/samuifx_edit.py --assets /tmp/pd_assets --presets /tmp/pd_assets/samuifx \
           --out edits/render --render --width 1080 --height 1350

Note: /tmp does not survive a sandbox restart — re-run fetch_assets.py (it is
idempotent and rebuilds the cart_* cuts) before rendering again.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from craftmotion.core.registry import Registry, COMMANDS   # noqa: E402
from craftmotion.core.export import render_video           # noqa: E402
from craftmotion.core.render import Renderer               # noqa: E402

W, H, FPS = 1080, 1350, 30          # 4:5 vertical — same canvas as the presets
DUR = 31_000


# ---------------------------------------------------------------- helpers
def find(assets: str, *names: str) -> str:
    for name in names:
        hits = sorted(glob.glob(os.path.join(assets, "**", name), recursive=True))
        if hits:
            return hits[0]
    raise SystemExit(f"missing asset {names} under {assets}")


def preset(assets: str, needle: str):
    hits = [p for p in glob.glob(os.path.join(assets, "**", "*.xml"), recursive=True)
            if needle.lower() in os.path.basename(p).lower()]
    return hits[0] if hits else None


class Edit:
    def __init__(self, assets: str, out_dir: str):
        self.assets = assets
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.reg = Registry()
        self.reg.execute("project.new", {"title": "SIGNAL / NOISE", "width": W, "height": H,
                                         "duration_ms": DUR, "fps": FPS})
        self.reg.project.settings["bg"] = "#FF06060B"
        self.media = {}
        self.beats: list[float] = []
        self.log: list[str] = []

    # -- plumbing ---------------------------------------------------
    def cmd(self, command: str, **args):
        if command not in COMMANDS:
            raise SystemExit(f"unknown command {command}")
        return self.reg.execute(command, args)

    def import_media(self, key: str, path: str) -> int:
        item = self.cmd("media.import", path=path)["media"]
        self.media[key] = item["id"]
        return item["id"]

    def music(self, path: str) -> None:
        self.import_media("music", path)
        self.cmd("layer.add", kind="audio", media_id=self.media["music"], start=0, end=DUR,
                 name="Midnight Club · Alec Penn Radio (CC0)")
        res = self.cmd("audio.beats", threshold=0.28, add_markers=True)
        self.beats = [b for b in res["beats"] if 900 < b < DUR - 500]
        self.log.append(f"beats: {len(self.beats)} · tempo {res['tempo']} BPM")

    def clip(self, key: str, start: int, end: int, look: str | None = None,
             source_in: int = 0, max_fx: int = 5, look_scale: float = 1.0,
             name: str = "clip", exclude=("invert", "blink")) -> int:
        lid = self.cmd("layer.add", kind="video", media_id=self.media[key], start=start,
                       end=end, fit="cover", name=name)["layer"]["id"]
        if source_in:
            self.cmd("layer.update", layer_id=lid, props={"source_in": int(source_in)})
        if look and os.path.exists(look):
            res = self.cmd("preset.am_look", path=look, layer_id=lid, max_effects=max_fx,
                           scale=look_scale, exclude=list(exclude))
            return lid, res["chain"]
        return lid, []

    def fx(self, lid: int, **stack) -> None:
        for etype, params in stack.items():
            try:
                self.cmd("effect.add", layer_id=lid, type=etype, params=params)
            except Exception as exc:            # noqa: BLE001
                self.log.append(f"  ! {etype}: {exc}")

    def punch(self, lid: int, start: int, end: int, lo: float = 1.0, hi: float = 1.06,
              hold: int = 110, window: tuple[float, float] | None = None) -> None:
        """Beat-synced scale punch: snap to `hi` on the beat, settle to `lo`."""
        keys = [(start, lo)]
        for t in self.beats:
            if not (start + 60 < t < end - 80):
                continue
            if window and not (window[0] <= t <= window[1]):
                continue
            keys.append((int(t), hi))
            keys.append((int(t) + hold, lo))
        keys.append((end, lo))
        seen = set()
        for t, v in keys:
            if t in seen:
                continue
            seen.add(t)
            self.cmd("keyframe.set", layer_id=lid, prop="scale", t=t, value=[v, v],
                     easing="ease_out")

    def drift(self, lid: int, start: int, end: int, a=(0, 0), b=(0, 0), sa=None, sb=None):
        """Slide (and optionally scale) a layer over its duration.

        Pass sa=None to leave `scale` alone — mixing this with punch() on the
        same layer writes two sets of scale keys and overshoots.
        """
        self.cmd("keyframe.set", layer_id=lid, prop="pos", t=start,
                 value=[W / 2 + a[0], H / 2 + a[1], 0], easing="linear")
        self.cmd("keyframe.set", layer_id=lid, prop="pos", t=end,
                 value=[W / 2 + b[0], H / 2 + b[1], 0], easing="linear")
        if sa is None or sb is None:
            return
        self.cmd("keyframe.set", layer_id=lid, prop="scale", t=start, value=[sa, sa],
                 easing="linear")
        self.cmd("keyframe.set", layer_id=lid, prop="scale", t=end, value=[sb, sb],
                 easing="linear")

    def text(self, body: str, size: int, y: float, color: str = "#FFFFFFFF", start: int = 0,
             end: int = DUR, name: str = "", font: str = "", tracking: float = 0.0,
             preset_name: str | None = None) -> int:
        lid = self.cmd("layer.add", kind="text", text=body, size=size, color=color,
                       name=name or body[:18], start=start, end=end, font=font,
                       transform={"pos": [W / 2, y, 0]})["layer"]["id"]
        if tracking:
            self.cmd("layer.update", layer_id=lid, props={"tracking": tracking})
        if preset_name:
            self.cmd("text.preset_apply", layer_id=lid, name=preset_name)
        return lid

    def cap(self, start: int, end: int, body: str) -> None:
        self.cmd("caption.add", start=start, end=end, text=body)


# ---------------------------------------------------------------- the edit
def build(assets: str, out_dir: str, looks_root: str | None = None) -> Edit:
    ed = Edit(assets, out_dir)
    looks_root = looks_root or assets
    mm_jewels = find(assets, "cart_mm3.mp4")      # glowing jewel exhibit
    mm_paper = find(assets, "cart_mm2.mp4")       # "HOUSE OF JEWELS" headline
    mm_door = find(assets, "cart_mm.mp4")         # silhouette in a doorway
    ee_lab = find(assets, "cart_ee2.mp4")         # scientist, control panel
    ee_white = find(assets, "cart_ee3.mp4")       # white-out flash
    ee_machine = find(assets, "cart_ee.mp4")      # machinery
    flare = find(assets, "solarflare.mp4")
    hole = find(assets, "blackhole.mp4")

    for key, path in (("jewels", mm_jewels), ("paper", mm_paper), ("door", mm_door),
                      ("lab", ee_lab), ("whiteout", ee_white), ("machine", ee_machine),
                      ("flare", flare), ("hole", hole)):
        ed.import_media(key, path)
    ed.music(find(assets, "phonk.mp3"))

    looks = {
        "xray": preset(looks_root, "X-RAY"),            # colorhot → edgeglow
        "nagumo": preset(looks_root, "New Project.xml"),  # lift → turbulentdisplace → blink
        "reze": preset(looks_root, "REZE"),             # tile → lift → zoomblur
        "sukuna": preset(looks_root, "Sukuna"),         # ripple → swirl → stretch → swing
        "gojo": preset(looks_root, "Gojo"),             # tile → blur → motionblur → oscillate
    }
    ed.log.append("looks: " + ", ".join(f"{k}={'✓' if v else '—'}" for k, v in looks.items()))

    # ---- global finish
    ed.cmd("grade.set", enabled=True, contrast=1.10, saturation=1.16, exposure=0.04,
           temperature=0.05)

    # ================================================== 01 · title (0–2.30 s)
    t0, t1 = 0, 2300
    name = ed.text("SIGNAL", 196, H * 0.365, "#FFFFFFFF", t0, t1, "SIGNAL",
                   preset_name="cascade")
    slash = ed.text("/ NOISE", 92, H * 0.475, "#FF35E5FF", t0 + 250, t1, "NOISE",
                    preset_name="type_on")
    for lid in (name, slash):
        ed.punch(lid, t0, t1, 1.0, 1.09, 90)
        ed.fx(lid, rgb_split={"offset": 9.0, "animate": True},
              oscillate={"mag": 2.5, "freq": 4.0, "angle": 90.0})
    ed.text("public-domain footage · cc0 music · am presets by samuifx", 22, H * 0.535,
            "#FFA9B9CC", t0 + 600, t1, "tag")
    ed.cap(t0 + 400, t1, "xml-preset-driven edit · v1.1")
    ed.log.append("01 title 0.0–2.3 s")

    # ================================================== 02 · thermal jewels (2.3–7.0)
    t0, t1 = 2300, 7000
    a, chain_a = ed.clip("jewels", t0, t1, looks["xray"], 10_500, 4, 1.0, "jewels · thermal")
    ed.log.append(f"  look → {' → '.join(chain_a)}")
    ed.drift(a, t0, t1, a=(0, 40), b=(0, -40), sa=1.30, sb=1.46)
    ed.punch(a, t0, t1, 1.30, 1.42, 100)
    ed.fx(a, thermal={"amount": 1.0},
          glow={"radius": 26.0, "strength": 0.7, "threshold": 0.35},
          scanlines={"intensity": 0.18, "period": 3.0})
    ed.text("01", 260, H * 0.30, "#5535E5FF", t0, t1, "n01")
    ed.text("THE MECHANICAL\nMONSTERS", 70, H * 0.78, "#FFFFFFFF", t0 + 300, t1, "s02",
            preset_name="cascade")
    ed.cap(t0, t1, "fleischer, 1941 · public domain")
    ed.log.append("02 thermal 2.3–7.0 s")

    # ================================================== 03 · headline (7.0–10.3)
    # The newspaper is a 3:2 source on a 4:5 canvas, so a full-bleed cover crop
    # turns the type into unreadable slabs. Frame it as an editorial card
    # instead, with the actual headline above it.
    t0, t1 = 7000, 10300
    # `turbulence` is part of the Nagumo signature but at this scale it eats the
    # headline type, so the card keeps the colour half of the look only and gets
    # a short turbulence burst at the end instead (below).
    b, chain_b = ed.clip("paper", t0, t1, looks["nagumo"], 500, 4, 0.45, "headline",
                         exclude=("invert", "blink", "turbulence"))
    ed.log.append(f"  look → {' → '.join(chain_b)}")
    ed.cmd("layer.update", layer_id=b, props={"fit": "contain"})
    ed.drift(b, t0, t1, a=(-16, 78), b=(16, 88))
    ed.punch(b, t0, t1, 1.00, 1.06, 90)
    ed.fx(b, halftone={"dot": 4.0, "angle": 45.0}, paper_texture={"grain": 0.18, "fibre": 0.3})
    ed.text("EXHIBIT OPENS TO-DAY", 52, H * 0.115, "#FFFFE066", t0, t1, "ticker",
            preset_name="type_on")
    ed.text("THE MECHANICAL MONSTERS", 24, H * 0.165, "#B4FFFFFF", t0 + 220, t1, "subline",
            tracking=6.0)
    burst, _ = ed.clip("paper", t1 - 480, t1, None, 0, 0, 1.0, "headline burst")
    ed.cmd("layer.update", layer_id=burst, props={"fit": "contain"})
    ed.fx(burst, turbulence={"intensity": 0.22, "scale": 26.0}, rgb_split={"offset": 6.0})
    ed.cap(t0, t1, "same short, 1941 · pd")
    ed.log.append("03 headline 7.0–10.3 s")

    # ================================================== 04 · flash montage (10.3–15.0)
    t0, t1 = 10300, 15000
    cut_beats = [t for t in ed.beats if t0 <= t <= t1]
    bounds = [t0] + [int(t) for t in cut_beats][::2] + [t1]
    bounds = sorted(set(b for b in bounds if t0 <= b <= t1))
    sources = ["lab", "jewels", "machine", "whiteout", "door", "lab", "jewels", "machine"]
    offsets = {"lab": 1_000, "jewels": 11_000, "machine": 4_500, "whiteout": 2_000,
               "door": 3_000}
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 90:
            continue
        key = sources[i % len(sources)]
        look = looks["sukuna"] if i % 2 == 0 else looks["gojo"]
        c, _chain = ed.clip(key, s, e, look, offsets.get(key, 1_000) + i * 300, 3, 1.35,
                            f"cut{i:02d} · {key}")
        ed.fx(c, rgb_split={"offset": 14.0, "animate": True})
        if i % 3 == 0:
            ed.fx(c, random_displace={"mag": 16.0, "evolution": 2.0, "scatter": 0.8})
        ed.punch(c, s, e, 1.06, 1.16, 60)
    ed.text("FLASH", 120, H * 0.86, "#FFFFFFFF", t0, t1, "flash-word",
            preset_name="pop")
    ed.cap(t0, t1, "cuts land on detected onsets")
    ed.log.append(f"04 flash {t0/1000:.1f}–{t1/1000:.1f} s · {len(bounds)-1} cuts")

    # ================================================== 05 · deep field (15.0–19.5)
    t0, t1 = 15000, 19500
    c, chain_c = ed.clip("flare", t0, t1, looks["reze"], 70_000, 4, 1.2, "deep field")
    ed.log.append(f"  look → {' → '.join(chain_c)}")
    ed.drift(c, t0, t1, a=(-70, -40), b=(70, 40), sa=1.10, sb=1.42)
    ed.punch(c, t0, t1, 1.10, 1.24, 100)
    ed.fx(c, bloom={"threshold": 0.55, "radius": 24.0, "intensity": 0.55},
          halation={"threshold": 0.55, "radius": 30.0, "intensity": 0.45},
          starfield={"count": 220, "size": 1.8, "twinkle": True})
    ed.text("DEEP FIELD", 44, H * 0.885, "#CCFFFFFF", t0 + 400, t1, "s05",
            preset_name="type_on")
    ed.cap(t0, t1, "nasa sdo · deep field · pd")
    ed.log.append("05 deep field 15.0–19.5 s")

    # ================================================== 06 · event horizon (19.5–24.0)
    t0, t1 = 19500, 24000
    d, chain_d = ed.clip("hole", t0, t1, looks["gojo"], 95_000, 2, 0.5, "event horizon",
                         exclude=("invert", "blink", "tile"))
    ed.log.append(f"  look → {' → '.join(chain_d)}")
    ed.drift(d, t0, t1, a=(0, 30), b=(0, -30), sa=1.32, sb=1.60)
    ed.punch(d, t0, t1, 1.32, 1.44, 90)
    ed.fx(d, bloom={"threshold": 0.35, "radius": 34.0, "intensity": 1.1},
          rgb_split={"offset": 11.0, "animate": True},
          lens_flare={"cx": 0.62, "cy": 0.34, "size": 0.4, "intensity": 0.7})
    ed.text("EVENT\nHORIZON", 110, H * 0.24, "#FFB9F2FF", t0 + 200, t1, "s06",
            preset_name="blur_in")
    ed.cap(t0, t1, "nasa · black hole sim · pd")
    ed.log.append("06 black hole 19.5–24.0 s")

    # ================================================== 07 · control room (24.0–27.6)
    t0, t1 = 24000, 27600
    e, chain_e = ed.clip("lab", t0, t1, looks["sukuna"], 7_000, 2, 0.5, "control room",
                         exclude=("invert", "blink", "tile", "levels"))
    ed.log.append(f"  look → {' → '.join(chain_e)}")
    ed.drift(e, t0, t1, a=(-35, 0), b=(35, 0), sa=1.40, sb=1.26)
    ed.punch(e, t0, t1, 1.26, 1.40, 85)
    ed.fx(e, vignette_pro={"amount": 0.42, "radius": 0.64, "feather": 0.55},
          film_grain={"amount": 0.07, "size": 1.0},
          glow={"radius": 18.0, "strength": 0.35, "threshold": 0.62})
    ed.cap(t0, t1, "fleischer, 1942 · public domain")
    ed.log.append("07 control room 24.0–27.6 s")

    # ================================================== 08 · outro (27.6–31.0)
    t0, t1 = 27600, DUR
    big = ed.text("CRAFTMOTION", 116, H * 0.40, "#FFFFFFFF", t0, t1, "outro",
                  preset_name="cascade")
    sub = ed.text("317 effects · 39 transitions · 38 blends · 130 commands", 28, H * 0.50,
                  "#FF9FB4C8", t0 + 150, t1, "outro-cap", preset_name="type_on")
    cred = ed.text("footage: fleischer 1941/42 + nasa (public domain)\n"
                   "music: alec penn radio — midnight club (cc0)\n"
                   "looks: samuifx alight-motion xml presets · edit: craftmotion v1.1",
                   22, H * 0.585, "#FF7E93A8", t0 + 400, t1, "credits")
    for lid in (big, sub, cred):
        ed.cmd("keyframe.set", layer_id=lid, prop="opacity", t=t0, value=[0.0],
               easing="ease_out")
        ed.cmd("keyframe.set", layer_id=lid, prop="opacity", t=t0 + 450, value=[1.0])
    pan = ed.cmd("layer.add", kind="text", text="", size=10, start=t0, end=t1,
                 name="spectrum analyser")["layer"]["id"]
    ed.cmd("audio.analyser", layer_id=pan, kind="spectrum",
           params={"bars": 72, "height": 0.14, "thickness": 0.6, "mirror": False,
                   "color": "#FF35E5FF", "color2": "#FFFF3D8B"})
    ed.cmd("keyframe.set", layer_id=pan, prop="pos", t=t0, value=[W / 2, H * 0.955, 0],
           easing="linear")
    ed.cap(t0, t1, "made in craftmotion · no alight motion used")
    ed.log.append("08 outro 27.6–31.0 s")

    # ---- caption style (short, legible on a 4:5 canvas) + master + delivery
    ed.reg.project.data["caption_style"] = {"size": 30, "margin": 0.05, "color": "#FFFFFFFF",
                                            "bg": "#C0000000"}
    ed.cmd("audio.master", gain_db=2.5, ceiling_db=-1.0)
    ed.cmd("audio.normalize", target=-14.0)
    return ed


def render_stills(ed: Edit, moments: dict) -> list[str]:
    """Save one PNG per segment.

    Rendered in `export` mode on purpose: the stills are previews of what the MP4
    will contain, so they must exercise the same code path as the encoder.
    """
    from craftmotion.core.render import Renderer as _R
    paths = []
    for label, t in moments.items():
        rend = _R(ed.reg.project, ed.reg.media, mode="export")
        try:
            img = rend.render(t, (540, 675))
        finally:
            rend.close()
        path = os.path.join(ed.out, f"frame_{label}.png")
        img.convert("RGB").save(path)
        paths.append(path)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default="/tmp/pd_assets")
    # NOTE: never default this to a directory called `out`, `dist` or `build` —
    # those names are skipped by workspace snapshots, so the render disappears.
    ap.add_argument("--out", default="edits/render")
    ap.add_argument("--presets", default=None,
                    help="folder with the SAMUIFX .xml presets (default: <assets>)")
    ap.add_argument("--render", action="store_true", help="encode the MP4")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1350)
    ap.add_argument("--quality", default="good", choices=["draft", "good", "high"],
                    help="encode quality; 'good' (veryfast/CRF20) keeps x264 far "
                         "ahead of the frame feeder so RAM stays flat")
    ap.add_argument("--start", type=int, default=None,
                    help="render only this slice, e.g. two halves on two cores")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--suffix", default="", help="appended to the MP4 name")
    ap.add_argument("--no-stills", action="store_true", help="skip the storyboard stills")
    a = ap.parse_args(argv)

    ed = build(a.assets, a.out, looks_root=a.presets)
    for line in ed.log:
        print("·", line)
    proj_path = os.path.join(a.out, "signal_noise.cmproj")
    ed.cmd("project.save", path=proj_path)
    print(f"· saved {proj_path} ({os.path.getsize(proj_path) / 1024:.0f} KB)")

    if not a.no_stills:
        moments = {"title": 1900, "thermal": 4200, "headline": 8600, "flash": 12200,
                   "solar": 16600, "horizon": 21400, "control": 25800, "outro": 29600}
        stills = render_stills(ed, moments)
        print(f"· {len(stills)} stills → {a.out}")

    if a.render:
        out_mp4 = os.path.join(a.out, f"signal_noise{a.suffix}.mp4")
        state = {"last": ""}

        def progress(p, m):
            if m and m != state["last"]:
                state["last"] = m
                print(f"\r  {p * 100:5.1f}% {m[:60]}", end="", flush=True)

        render_video(ed.reg.project.clone(), ed.reg.media, out_mp4, a.width, a.height, FPS,
                     quality=a.quality, include_audio=True, progress=progress,
                     start_ms=a.start, end_ms=a.end)
        print(f"\n· rendered {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
