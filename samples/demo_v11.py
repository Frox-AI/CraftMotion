"""Render a v1.1 feature showcase: masks, styles, text animators, trim+repeater,
2.5D camera, lights, motion blur, grade, captions, audio-reactive spectrum."""
import sys; sys.path.insert(0, '.')
import numpy as np, wave, tempfile, os
from craftmotion.core.registry import Registry
from craftmotion.core.render import Renderer
from craftmotion.core import audio as AUD

# synth a beat so audio-reactive effects have something to chew on
tmp = tempfile.mktemp(suffix=".wav")
sr = 48000; dur = 3.0
t = np.arange(int(sr*dur))/sr

# kickish pulse + bass
env = 0.25 + 0.75*np.abs(np.sin(2*np.pi*2.0*t))
sig = (0.45*env*np.sin(2*np.pi*98*t) + 0.15*np.sin(2*np.pi*330*t)).astype(np.float32)
AUD.write_wav(np.stack([sig, sig], 1), tmp)

r = Registry()
r.execute("project.new", {"title": "v1.1 showcase", "width": 900, "height": 600,
                          "duration": 4000, "fps": 30})

media = r.execute("media.import", {"path": tmp})["media"]
r.execute("layer.add", {"kind": "audio", "media_id": media["id"], "start": 0, "end": 3000})

# background gradient solid
r.execute("layer.add", {"kind": "solid", "color": "#FF0B1020", "blend": "normal"})

# masked, styled panel
box = r.execute("layer.add", {"kind": "solid", "color": "#FF1E6FD9", "w": 420, "h": 300,
                              "transform": {"pos": [300, 300, 0]}})["layer"]["id"]
r.execute("mask.add", {"layer_id": box, "shape": "polygon",
                       "points": [[0.08, 0.1], [0.92, 0.06], [0.96, 0.9], [0.4, 0.97], [0.05, 0.6]],
                       "feather": 2})
r.execute("style.add", {"layer_id": box, "type": "drop_shadow",
                        "params": {"blur": 26, "distance": 18, "opacity": 0.75}})
r.execute("style.add", {"layer_id": box, "type": "stroke",
                        "params": {"size": 5, "color": "#FF8FE7FF"}})
r.execute("effect.add", {"layer_id": box, "type": "bloom", "params": {"threshold": 0.55, "intensity": 0.5}})

# trimmed + repeated shape
star = r.execute("layer.add", {"kind": "shape", "shape": "star", "w": 120, "h": 120,
                               "color": "#FFFFC24A", "transform": {"pos": [650, 220, 0]}})["layer"]["id"]
r.execute("shape.repeater", {"layer_id": star, "copies": 5, "x": 42, "y": 18, "scale": 0.86,
                             "rotation": 12, "end_opacity": 0.2})
r.execute("effect.add", {"layer_id": star, "type": "audio_scale", "params": {"amount": 0.18, "band": "low"}})

# trimmed stroke path
path = r.execute("layer.add", {"kind": "shape", "shape": "path", "closed": False,
                               "path": [[0.05, 0.75], [0.3, 0.62], [0.55, 0.8], [0.8, 0.66], [0.95, 0.72]],
                               "stroke_color": "#FFFF5A7A", "stroke_width": 9, "w": 900, "h": 600,
                               "transform": {"pos": [450, 300, 0]}})["layer"]["id"]
r.execute("shape.trim", {"layer_id": path, "start": 0, "end": 74})

# animated text
title = r.execute("layer.add", {"kind": "text", "text": "CRAFTMOTION v1.1",
                                "size": 64, "color": "#FFFFFFFF",
                                "transform": {"pos": [450, 120, 0]}})["layer"]["id"]
r.execute("text.preset_apply", {"layer_id": title, "name": "cascade", "speed": 0.6})

# audio spectrum analyser on a text layer at the bottom
analyser = r.execute("layer.add", {"kind": "text", "text": "", "size": 10})["layer"]["id"]
r.execute("audio.analyser", {"layer_id": analyser, "kind": "spectrum",
                             "params": {"bars": 64, "height": 0.22, "thickness": 0.55, "mirror": False,
                                        "color": "#FF38E8C8", "color2": "#FF8B5CF6"}})

# 2.5D camera + light
r.execute("camera.set", {"enabled": True, "zoom": 1.04, "rotation": -1.2})
r.execute("light.add", {"color": "#FFFFD9A0", "radius": 460, "intensity": 0.3,
                        "transform": {"pos": [700, 140, 0]}})

# grade + captions
r.execute("grade.set", {"exposure": 0.12, "contrast": 1.06, "saturation": 1.12,
                        "enabled": True})
r.execute("caption.add", {"start": 0, "end": 2200,
                          "text": "312 effects · 39 transitions · 38 blend modes"})
r.execute("caption.add", {"start": 2200, "end": 4000,
                          "text": "masks, styles, tracking, audio-reactive, Lottie"})

rend = Renderer(r.project, r._media_manager())
img = rend.render(1400)
img.convert("RGB").save("samples/showcase_v11.png")
img.convert("RGB").save("/tmp/showcase_v11.png")
print("rendered", img.size, "audio state:", {k: round(v, 2) if isinstance(v, float) else len(v)
                                            for k, v in rend.audio_state(1000).items()})
r.execute("project.save", {"path": "samples/showcase_v11.cmproj"})
