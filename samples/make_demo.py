"""Build the CraftMotion showcase project from synthetic media.

    python samples/make_demo.py

Creates samples/demo_assets/*.png, samples/craftmotion_demo.cmproj
and samples/demo_storyboard.png — a 9:16 social edit you can open in the GUI.
"""
from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from craftmotion.core.registry import Registry  # noqa: E402

ASSETS = os.path.join(ROOT, "samples", "demo_assets")
W, H, FPS, DUR = 1080, 1920, 30, 9000


def grad(w, h, stops, angle=90):
    img = Image.new("RGB", (w, h))
    px = img.load()
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    denom = abs(dx) * w + abs(dy) * h
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            t = ((x * dx + y * dy) / denom + 1) / 2 if denom else 0
            t = max(0.0, min(1.0, t))
            pos = t * (len(stops) - 1)
            i = int(pos)
            f = pos - i
            c1 = stops[min(i, len(stops) - 1)]
            c2 = stops[min(i + 1, len(stops) - 1)]
            c = tuple(int(c1[k] + (c2[k] - c1[k]) * f) for k in range(3))
            px[x, y] = c
            if x + 1 < w:
                px[x + 1, y] = c
            if y + 1 < h:
                px[x, y + 1] = c
    return img.filter(ImageFilter.GaussianBlur(1.2))


def make_assets():
    os.makedirs(ASSETS, exist_ok=True)
    # 1) sunset
    sunset = grad(W, H, [(255, 138, 61), (233, 69, 96), (58, 28, 113)], 100)
    d = ImageDraw.Draw(sunset, "RGBA")
    d.ellipse((W * 0.28, H * 0.32, W * 0.72, H * 0.62), fill=(255, 236, 170, 235))
    for i in range(7):
        y = H * (0.60 + i * 0.045)
        d.rectangle((0, y, W, y + H * 0.028), fill=(20, 10, 40, 30 + i * 22))
    sunset.save(os.path.join(ASSETS, "sunset.png"))

    # 2) city night
    city = grad(W, H, [(10, 14, 40), (26, 40, 92), (94, 40, 120)], 90)
    d = ImageDraw.Draw(city, "RGBA")
    import random
    random.seed(7)
    for i in range(26):
        bw = random.randint(60, 160)
        bh = random.randint(220, 780)
        x = random.randint(-40, W - 60)
        y = H - bh - random.randint(0, 120)
        d.rectangle((x, y, x + bw, H), fill=(12, 12, 28, 255))
        for wy in range(y + 24, H - 30, 46):
            for wx in range(x + 14, x + bw - 14, 34):
                if random.random() > 0.45:
                    d.rectangle((wx, wy, wx + 14, wy + 20),
                                fill=(255, 214, 120, random.randint(90, 220)))
    moon = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
    md = ImageDraw.Draw(moon)
    md.ellipse((10, 10, 250, 250), fill=(255, 245, 210, 240))
    city.paste(Image.new("RGB", (260, 260), (255, 246, 214)), (W - 330, 170), moon.split()[-1])
    city.convert("RGB").save(os.path.join(ASSETS, "city.png"))

    # 3) abstract waves
    waves = grad(W, H, [(88, 24, 164), (28, 148, 189), (16, 182, 140)], 35)
    d = ImageDraw.Draw(waves, "RGBA")
    for k in range(6):
        pts = []
        for x in range(0, W + 40, 40):
            y = H * (0.35 + k * 0.09) + math.sin(x / 130 + k) * 70
            pts.append((x, y))
        pts += [(W, H), (0, H)]
        d.polygon(pts, fill=(255, 255, 255, 12 + k * 9))
    waves.save(os.path.join(ASSETS, "waves.png"))
    print("assets:", os.listdir(ASSETS))


def build_project():
    reg = Registry()
    reg.execute("project.new", {"title": "CraftMotion Demo", "width": W, "height": H,
                                "fps": FPS, "duration_ms": DUR, "bg": "#FF0A0D16"})
    media = {}
    for name in ("sunset", "city", "waves"):
        item = reg.execute("media.import", {"path": os.path.join(ASSETS, f"{name}.png"),
                                            "proxy": False})["media"]
        media[name] = item["id"]

    # ---- shot 1: sunset with Ken Burns
    s1 = reg.execute("layer.add", {"kind": "image", "media_id": media["sunset"], "name": "Shot 1 · Sunset",
                                   "start": 0, "end": 3600, "fit": "cover"})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": s1, "pos": [W / 2, H / 2], "scale": [1.05, 1.05]})
    for t, sc in ((0, 1.05), (3600, 1.22)):
        reg.execute("keyframe.set", {"layer_id": s1, "prop": "scale", "t": t, "value": [sc, sc],
                                     "easing": "ease_in_out"})
    reg.execute("effect.add", {"layer_id": s1, "type": "bloom",
                               "params": {"radius": 40, "strength": 0.5, "threshold": 0.65}})

    # ---- shot 2: city with slide-in + grade
    s2 = reg.execute("layer.add", {"kind": "image", "media_id": media["city"], "name": "Shot 2 · City",
                                   "start": 3200, "end": 6400, "fit": "cover"})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": s2, "pos": [W / 2, H / 2]})
    reg.execute("effect.apply_look", {"layer_id": s2, "look": "neon_night"})
    reg.execute("transition.add", {"layer_id": s2, "kind": "slide", "duration": 550, "mode": "in"})

    # ---- shot 3: waves with zoom punch + glitch accent
    s3 = reg.execute("layer.add", {"kind": "image", "media_id": media["waves"], "name": "Shot 3 · Waves",
                                   "start": 6100, "end": 9000, "fit": "cover"})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": s3, "pos": [W / 2, H / 2]})
    reg.execute("transition.add", {"layer_id": s3, "kind": "zoom_punch", "duration": 500, "mode": "in"})

    # ---- titles
    t1 = reg.execute("layer.add", {"kind": "text", "name": "Title", "text": "CRAFT\nMOTION",
                                   "size": 150, "font": "DejaVu Sans Bold", "color": "#FFFFFFFF",
                                   "align": "center", "line_height": 0.95,
                                   "start": 400, "end": 3600})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": t1, "pos": [W / 2, H * 0.46]})
    for t, sc, op in ((300, 0.4, 0), (900, 1.0, 1), (3200, 1.06, 1), (3600, 1.15, 0)):
        reg.execute("keyframe.set", {"layer_id": t1, "prop": "scale", "t": t, "value": [sc, sc],
                                     "easing": "ease_in_out"})
        reg.execute("keyframe.set", {"layer_id": t1, "prop": "opacity", "t": t, "value": op,
                                     "easing": "ease_in_out"})
    reg.execute("effect.add", {"layer_id": t1, "type": "drop_shadow",
                               "params": {"offset_y": 10, "blur": 22, "opacity": 0.55}})

    t2 = reg.execute("layer.add", {"kind": "text", "name": "Subtitle", "text": "motion + video + alight motion xml",
                                   "size": 44, "color": "#FFB9C6FF", "align": "center",
                                   "start": 900, "end": 3600})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": t2, "pos": [W / 2, H * 0.62]})
    for t, op in ((900, 0), (1400, 1)):
        reg.execute("keyframe.set", {"layer_id": t2, "prop": "opacity", "t": t, "value": op})

    t3 = reg.execute("layer.add", {"kind": "text", "name": "End card", "text": "EDITED BY AN AGENT",
                                   "size": 78, "font": "DejaVu Sans Mono", "color": "#FF3DDC97",
                                   "align": "center", "start": 6500, "end": 9000})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": t3, "pos": [W / 2, H * 0.5]})
    for t, sc in ((6500, 0.7), (7100, 1.0)):
        reg.execute("keyframe.set", {"layer_id": t3, "prop": "scale", "t": t, "value": [sc, sc],
                                     "easing": "back_out"})
    reg.execute("effect.add", {"layer_id": t3, "type": "glow",
                               "params": {"radius": 26, "strength": 1.4, "threshold": 0.3}})

    # ---- star accents on shot 1
    st = reg.execute("layer.add", {"kind": "shape", "name": "Accent star", "shape": "star", "points": 5,
                                   "inner": 0.45, "w": 260, "h": 260, "color": "#FFFFC24B",
                                   "start": 300, "end": 3600})["layer"]["id"]
    reg.execute("layer.set_transform", {"layer_id": st, "pos": [W * 0.8, H * 0.22], "scale": [1, 1]})
    for t, rot in ((300, 0), (3600, 180)):
        reg.execute("keyframe.set", {"layer_id": st, "prop": "rot", "t": t, "value": rot,
                                     "easing": "ease_in_out"})
    reg.execute("effect.apply_look", {"layer_id": st, "look": "dreamy_glow"})

    # ---- global grade via adjustment layer
    adj = reg.execute("layer.add", {"kind": "adjustment", "name": "Grade", "start": 0, "end": DUR})["layer"]["id"]
    reg.execute("effect.add", {"layer_id": adj, "type": "brightness_contrast",
                               "params": {"contrast": 0.08, "brightness": 0.01}})
    reg.execute("effect.add", {"layer_id": adj, "type": "saturation",
                               "params": {"vibrance": 0.18}})
    reg.execute("effect.add", {"layer_id": adj, "type": "vignette",
                               "params": {"amount": 0.28, "feather": 0.6}})
    reg.execute("effect.add", {"layer_id": adj, "type": "grain", "params": {"amount": 0.03}})

    # markers
    for t, name in ((3200, "shot 2"), (6100, "shot 3"), (6500, "end card")):
        reg.execute("marker.add", {"t": t, "label": name})

    path = os.path.join(ROOT, "samples", "craftmotion_demo.cmproj")
    reg.execute("project.save", {"path": path})
    print("project:", path, "· layers:", len(reg.project.layers), "· media:", len(reg.project.media))
    return reg, path


def render_sheets(reg):
    board = os.path.join(ROOT, "samples", "demo_storyboard.png")
    reg.execute("render.storyboard", {"times": [500, 1200, 2400, 3600, 4800, 6200, 7000, 8200, 8800],
                                      "cols": 3, "tile": 300, "path": board})
    print("storyboard:", board)
    still = os.path.join(ROOT, "samples", "demo_still.png")
    reg.execute("render.still", {"t": 1200, "path": still,
                                 "width": 540, "height": 960})
    print("still:", still)


if __name__ == "__main__":
    make_assets()
    reg, _path = build_project()
    render_sheets(reg)
    print("done — open samples/craftmotion_demo.cmproj in the app")
