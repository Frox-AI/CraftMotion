"""Parity panels: graph editor, effect browser, colour/scopes, audio mixer,
captions and motion tracking. Every control calls the same registry command an
agent would call, so GUI and MCP stay in lock-step.
"""
from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional

from ..core import audio as AUD
from ..core import effects as FX
from ..core.grade import ensure_grade, render_scope_image
from ..core.transitions import TRANSITIONS
from ..core.util import kf_eval, parse_color
from .theme import C, font, label, mono

PROPS = ["pos", "scale", "rot", "opacity"]
COLORS = {"pos": "#3d8bfd", "scale": "#3ddc97", "rot": "#ffcc55", "opacity": "#c792ea"}


class PanelBase(tk.Frame):
    title = "PANEL"

    def __init__(self, master, app):
        super().__init__(master, bg=C.panel, bd=0, highlightthickness=0)
        self.app = app
        bar = tk.Frame(self, bg=C.panel, height=28)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        label(bar, self.title, C.panel, C.dim, 8, True).pack(side="left", padx=10)
        self.toolbar = bar

    # -- helpers ---------------------------------------------------------
    def cmd(self, name: str, args: Optional[dict] = None):
        try:
            return self.app.run_command(name, args or {}, quiet=True)
        except Exception as exc:  # pragma: no cover - UI feedback path
            self.app.log(f"{name}: {exc}")
            return None

    def selected_layer(self) -> Optional[dict]:
        return self.app.selected_layer()

    def number(self, parent, text: str, value, cb: Callable, width: int = 6,
               lo: Optional[float] = None, hi: Optional[float] = None, bg: str = C.panel):
        label(parent, text, bg, C.muted, 9).pack(side="left", padx=(6, 2))
        e = tk.Entry(parent, width=width, bg=C.panel2 if bg == C.panel else C.track, fg=C.text, bd=0,
                     font=mono(9), insertbackground=C.text, justify="center")
        e.insert(0, ("%.2f" % value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value))
        e.pack(side="left", ipady=2)

        def commit(_e=None):
            try:
                v = float(e.get().strip())
            except ValueError:
                return
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            cb(v)
        e.bind("<Return>", commit)
        e.bind("<FocusOut>", commit)
        return e

    def slider(self, parent, text: str, value: float, lo: float, hi: float, cb: Callable,
               bg: str = C.panel) -> tk.Scale:
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=8, pady=1)
        label(row, text, bg, C.muted, 9).pack(side="left", anchor="w")
        var = tk.DoubleVar(value=float(value))
        s = tk.Scale(row, from_=lo, to=hi, resolution=(hi - lo) / 200.0, orient="horizontal",
                     variable=var, bg=bg, fg=C.text, troughcolor=C.track, highlightthickness=0, bd=0,
                     showvalue=True, font=mono(7), length=150,
                     command=lambda v: cb(float(v)))
        s.pack(side="right", fill="x", expand=True)
        return s


# ====================================================================== graph editor
class GraphEditorPanel(PanelBase):
    title = "GRAPH EDITOR"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.prop = "pos"
        self.mode = "value"          # value | velocity
        self.canvas = tk.Canvas(self, bg=C.bg, highlightthickness=0, bd=0, height=200)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self._drag: Optional[dict] = None

        self.combo = ttk.Combobox(self.toolbar, values=PROPS, width=8, state="readonly", font=font(8))
        self.combo.set(self.prop)
        self.combo.pack(side="right", padx=6)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.set_prop(self.combo.get()))
        self.mode_btn = tk.Button(self.toolbar, text="VALUE", command=self.toggle_mode, bg=C.panel2,
                                  fg=C.muted, font=mono(7), bd=0, padx=6)
        self.mode_btn.pack(side="right", padx=4)
        self.app.reg.on_change(lambda ev: self.post_redraw())

    def post_redraw(self):
        try:
            self.after(10, self.redraw)
        except Exception:
            pass

    def set_prop(self, prop: str):
        self.prop = prop
        self.combo.set(prop)
        self.redraw()

    def toggle_mode(self):
        self.mode = "velocity" if self.mode == "value" else "value"
        self.mode_btn.configure(text=self.mode.upper())
        self.redraw()

    # -- geometry -------------------------------------------------------
    def _geom(self):
        w = max(50, self.canvas.winfo_width() or 600)
        h = max(40, self.canvas.winfo_height() or 200)
        return w, h

    def _layer(self):
        return self.selected_layer()

    def _keys(self):
        layer = self._layer()
        if not layer:
            return []
        return sorted(((layer.get("keys") or {}).get(self.prop) or []), key=lambda k: k.get("t", 0))

    def _sample(self, keys, t):
        try:
            return kf_eval(keys, t, [0.0])
        except Exception:
            return [0.0]

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = self._geom()
        proj = self.app.reg.project
        pad = 26
        c.create_rectangle(0, 0, w, h, fill=C.bg, outline="")
        keys = self._keys()
        dur = max(1, int(proj.duration))
        # grid
        for i in range(0, 6):
            y = pad + (h - 2 * pad) * i / 5.0
            c.create_line(pad, y, w - 8, y, fill=C.border)
            c.create_text(4, y, text="", fill=C.dim, anchor="w", font=mono(7))
        if not keys:
            c.create_text(w / 2, h / 2, text="No keyframes on this property — press K in the inspector",
                          fill=C.dim, font=font(9))
            return
        # value range
        vals = []
        for k in keys:
            for v in (k.get("v") or [0]):
                vals.append(float(v))
        if self.mode == "velocity":
            curve = []
            step = max(16, dur // 120)
            for t in range(0, dur, step):
                a = self._sample(keys, t)[0]
                b = self._sample(keys, min(dur, t + step))[0]
                curve.append((t, (b - a) / max(1e-3, step / 1000.0)))
            vals = [v for _, v in curve]
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-6:
            lo, hi = lo - 1, hi + 1
        span = hi - lo

        def X(t):
            return pad + (w - pad - 10) * (float(t) / dur)

        def Y(v):
            return pad + (h - 2 * pad) * (1.0 - (float(v) - lo) / span)

        if self.mode == "velocity":
            pts = []
            for t, v in curve:
                pts.extend([X(t), Y(v)])
            if len(pts) >= 4:
                c.create_line(*pts, fill=C.good, width=2, smooth=True)
            c.create_text(pad + 4, 10, text="velocity (units/s)", fill=C.good, anchor="w", font=mono(7))
        else:
            # eased sampling of the actual interpolation
            pts = []
            step = max(10, dur // 200)
            for t in range(0, dur + 1, step):
                pts.extend([X(t), Y(self._sample(keys, t)[0])])
            c.create_line(*pts, fill=COLORS.get(self.prop, C.accent2), width=2, smooth=True)
            for k in keys:
                v = float((k.get("v") or [0])[0])
                x, y = X(k.get("t", 0)), Y(v)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=C.kf, outline=C.bg)
                c.create_text(x, y - 10, text=("%.1f" % v), fill=C.dim, font=mono(6))
        # playhead
        px = X(self.app.t)
        c.create_line(px, 0, px, h, fill=C.playhead, width=1)
        c.create_text(w - 8, 10, text=f"{self.prop} · {self.mode}", fill=C.dim, anchor="e", font=mono(7))

    # -- dragging -------------------------------------------------------
    def _on_press(self, e):
        keys = self._keys()
        if not keys:
            return
        w, h = self._geom()
        dur = max(1, int(self.app.reg.project.duration))
        best = None
        for i, k in enumerate(keys):
            x = 26 + (w - 36) * (float(k.get("t", 0)) / dur)
            if abs(e.x - x) <= 8:
                best = i
        self._drag = {"index": best} if best is not None else None

    def _on_drag(self, e):
        if not self._drag or self._drag.get("index") is None:
            return
        layer = self._layer()
        if not layer:
            return
        keys = (layer.get("keys") or {}).get(self.prop) or []
        keys = sorted(keys, key=lambda k: k.get("t", 0))
        idx = self._drag["index"]
        if idx >= len(keys):
            return
        w, h = self._geom()
        dur = max(1, int(self.app.reg.project.duration))
        t = max(0, min(dur, (e.x - 26) / max(1, w - 36) * dur))
        vals = [float(v) for v in (keys[idx].get("v") or [0])]
        if self.mode != "velocity":
            keys[idx]["t"] = int(round(t))
            ynorm = 1.0 - max(0.0, min(1.0, (e.y - 26) / max(1, h - 52)))
            # derive value from the current visible range
            allv = [float(v) for k in keys for v in (k.get("v") or [0])]
            lo, hi = min(allv), max(allv)
            if abs(hi - lo) < 1e-6:
                lo, hi = lo - 1, hi + 1
            newv = lo + ynorm * (hi - lo)
            keys[idx]["v"] = [newv] + vals[1:]
            keys.sort(key=lambda k: k.get("t", 0))
            layer["keys"][self.prop] = keys
            self.app.request_preview_only()
            self.redraw()

    def _on_release(self, e):
        if self._drag:
            self._drag = None
            self.app.refresh_all()


# ====================================================================== effect browser
class EffectBrowserPanel(PanelBase):
    title = "EFFECTS & TRANSITIONS"

    def __init__(self, master, app):
        super().__init__(master, app)
        top = tk.Frame(self, bg=C.panel)
        top.pack(fill="x", padx=8, pady=4)
        self.search = tk.Entry(top, bg=C.panel2, fg=C.text, bd=0, font=font(9), insertbackground=C.text)
        self.search.pack(fill="x", ipady=3)
        self.search.bind("<KeyRelease>", lambda e: self.rebuild())
        self.count = label(top, "", C.panel, C.dim, 8)
        self.count.pack(anchor="w")
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=4)
        self.listbox = tk.Listbox(nb, bg=C.panel2, fg=C.text, bd=0, highlightthickness=0,
                                  selectbackground=C.sel, font=font(9), activestyle="none")
        sb = tk.Scrollbar(nb, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        nb.add(self.listbox, text=f"Effects ({len(FX.EFFECTS)})")
        self.tlist = tk.Listbox(nb, bg=C.panel2, fg=C.text, bd=0, highlightthickness=0,
                                selectbackground=C.sel, font=font(9), activestyle="none")
        for key, entry in TRANSITIONS.items():
            self.tlist.insert("end", f"{entry['label']}  ·  {entry['category']}")
        self._tkeys = list(TRANSITIONS.keys())
        nb.add(self.tlist, text=f"Transitions ({len(TRANSITIONS)})")
        self.nb = nb
        self.listbox.bind("<Double-Button-1>", lambda e: self.apply_effect())
        self.tlist.bind("<Double-Button-1>", lambda e: self.apply_transition())
        self._keys: List[str] = []
        self.rebuild()

    def rebuild(self):
        q = (self.search.get() or "").strip().lower()
        self.listbox.delete(0, "end")
        self._keys = []
        cats: Dict[str, List[str]] = {}
        for key, eff in FX.EFFECTS.items():
            hay = f"{key} {eff.label} {eff.category}".lower()
            if q and q not in hay:
                continue
            cats.setdefault(eff.category, []).append(key)
        for cat in sorted(cats):
            for key in sorted(cats[cat], key=lambda k: FX.EFFECTS[k].label):
                eff = FX.EFFECTS[key]
                self.listbox.insert("end", f"{eff.label}   ·  {cat}")
                self._keys.append(key)
        self.count.configure(text=f"{len(self._keys)} matches — double-click to apply")

    def apply_effect(self):
        layer = self.selected_layer()
        if not layer:
            self.app.log("select a layer first")
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        key = self._keys[int(sel[0])]
        self.cmd("effect.add", {"layer_id": layer["id"], "type": key})
        self.app.log(f"added effect {FX.EFFECTS[key].label}")

    def apply_transition(self):
        layer = self.selected_layer()
        sel = self.tlist.curselection()
        if not layer or not sel:
            return
        key = self._tkeys[int(sel[0])]
        self.cmd("transition.set", {"layer_id": layer["id"], "type": key, "duration": 600, "when": "in"})
        self.app.log(f"transition {key} set on {layer.get('name')}")


# ====================================================================== colour + scopes
class ColourPanel(PanelBase):
    title = "COLOUR · GRADE · SCOPES"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.scope_kind = "histogram"
        for kind in ("histogram", "waveform", "vectorscope"):
            tk.Button(self.toolbar, text=kind[:4].upper(), bd=0, bg=C.panel2, fg=C.muted, font=mono(7),
                      command=lambda k=kind: self.set_scope(k)).pack(side="right", padx=2)
        body = tk.Frame(self, bg=C.panel)
        body.pack(fill="both", expand=True)
        self.scope = tk.Label(body, bg=C.bg)
        self.scope.pack(fill="x", padx=8, pady=6)
        self.controls = tk.Frame(body, bg=C.panel)
        self.controls.pack(fill="both", expand=True)
        grade = ensure_grade(app.reg.project)
        self.slider(self.controls, "Exposure", grade.get("exposure", 0), -3, 3,
                    lambda v: self.set_grade({"exposure": v}))
        self.slider(self.controls, "Contrast", grade.get("contrast", 1), 0.3, 2.5,
                    lambda v: self.set_grade({"contrast": v}))
        self.slider(self.controls, "Saturation", grade.get("saturation", 1), 0, 2.5,
                    lambda v: self.set_grade({"saturation": v}))
        self.slider(self.controls, "Temperature", grade.get("temperature", 0), -1, 1,
                    lambda v: self.set_grade({"temperature": v}))
        self.slider(self.controls, "Tint", grade.get("tint", 0), -1, 1,
                    lambda v: self.set_grade({"tint": v}))
        self.slider(self.controls, "Hue", (grade.get("hsl") or {}).get("hue", 0), -180, 180,
                    lambda v: self.set_grade({"hsl": {"hue": v}}))
        self.slider(self.controls, "HSL Sat", (grade.get("hsl") or {}).get("sat", 0), -1, 1,
                    lambda v: self.set_grade({"hsl": {"sat": v}}))
        row = tk.Frame(self.controls, bg=C.panel)
        row.pack(fill="x", padx=8, pady=4)
        tk.Button(row, text="Enable grade", bd=0, bg=C.panel2, fg=C.text, font=font(8),
                  command=lambda: self.set_grade({"enabled": True})).pack(side="left")
        tk.Button(row, text="Bypass", bd=0, bg=C.panel2, fg=C.muted, font=font(8),
                  command=lambda: self.set_grade({"enabled": False})).pack(side="left", padx=4)
        tk.Button(row, text="Load .cube LUT", bd=0, bg=C.panel2, fg=C.text, font=font(8),
                  command=self.load_lut).pack(side="left", padx=4)
        tk.Button(row, text="Linear light", bd=0, bg=C.panel2, fg=C.muted, font=font(8),
                  command=lambda: self.set_grade({"linear_light": True})).pack(side="left")
        row2 = tk.Frame(self.controls, bg=C.panel)
        row2.pack(fill="x", padx=8, pady=2)
        for zname in ("shadows", "midtones", "highlights"):
            tk.Button(row2, text=zname[:5].upper(), bd=0, bg=C.panel2, fg=C.muted, font=mono(7),
                      command=lambda z=zname: self.tint_zone(z)).pack(side="left", padx=2)

    def set_scope(self, kind):
        self.scope_kind = kind
        self.refresh_scope()

    def refresh_scope(self):
        try:
            from PIL import ImageTk
            proj = self.app.reg.project
            renderer = self.app.reg.renderer("preview")
            img = renderer.render(min(self.app.t, proj.duration - 1), (320, 180))
            scope = render_scope_image(img, self.scope_kind, (300, 140))
            self._photo = ImageTk.PhotoImage(scope)
            self.scope.configure(image=self._photo)
        except Exception:
            pass

    def set_grade(self, patch: dict):
        self.cmd("grade.set", patch)
        self.app.request_preview_only()
        self.refresh_scope()

    def tint_zone(self, zone: str):
        from tkinter import colorchooser
        rgb, _ = colorchooser.askcolor(title=f"{zone} tint")
        if not rgb:
            return
        import colorsys
        h, l, s = colorsys.rgb_to_hls(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        self.set_grade({"wheels": {zone: {"hue": h * 360.0, "amount": 0.08}}})

    def load_lut(self):
        path = filedialog.askopenfilename(title="Choose LUT", filetypes=[("Cube LUT", "*.cube"), ("All", "*.*")])
        if path:
            self.cmd("grade.lut", {"path": path, "amount": 1.0})
            self.app.log(f"LUT loaded: {os.path.basename(path)}")


# ====================================================================== audio mixer
class AudioPanel(PanelBase):
    title = "AUDIO MIXER"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.body = tk.Frame(self, bg=C.panel)
        self.body.pack(fill="both", expand=True)
        self.meter = tk.Label(self.toolbar, text="— LUFS", bg=C.panel, fg=C.good, font=mono(8))
        self.meter.pack(side="right", padx=6)
        for txt, fn in (("Beats", self.detect_beats), ("Pulse→scale", self.react_scale),
                        ("Analyser", self.add_analyser),
                        ("Measure", self.measure), ("Normalise -16", lambda: self.normalise(-16.0)),
                        ("Export WAV", self.export_wav)):
            tk.Button(self.toolbar, text=txt, bd=0, bg=C.panel2, fg=C.muted, font=mono(7),
                      command=fn).pack(side="right", padx=2)
        self.build()

    def build(self):
        for child in list(self.body.winfo_children()):
            child.destroy()
        proj = self.app.reg.project
        rows = [l for l in proj.layers if l.get("kind") in ("audio", "video")]
        if not rows:
            label(self.body, "No audio layers yet — import a clip or add an audio layer.",
                  C.panel, C.dim, 9).pack(pady=12)
            return
        for layer in rows:
            row = tk.Frame(self.body, bg=C.track)
            row.pack(fill="x", padx=6, pady=2)
            label(row, (layer.get("name") or "Audio")[:16], C.track, C.text, 9).pack(side="left", padx=4)
            var = tk.DoubleVar(value=float(layer.get("volume", 1.0)))
            sc = tk.Scale(row, from_=0, to=2, resolution=0.01, orient="horizontal", variable=var,
                          bg=C.track, fg=C.text, troughcolor=C.panel2, highlightthickness=0, bd=0,
                          showvalue=False, length=110,
                          command=lambda v, lid=layer["id"]: self.cmd("audio.mixer", {"layer_id": lid, "volume": float(v)}))
            sc.pack(side="left", padx=4)
            for txt, key in (("M", "mute"), ("S", "solo")):
                active = layer.get(key)
                tk.Button(row, text=txt, bd=0, font=mono(7), width=2,
                          bg=C.bad if active else C.panel2, fg=C.text,
                          command=lambda lid=layer["id"], k=key, cur=bool(active): self.toggle(lid, k, cur)
                          ).pack(side="right", padx=2)
            p = tk.DoubleVar(value=float(layer.get("pan", 0.0) or 0.0))
            tk.Scale(row, from_=-1, to=1, resolution=0.05, orient="horizontal", variable=p,
                     bg=C.track, fg=C.text, troughcolor=C.panel2, highlightthickness=0, bd=0,
                     showvalue=False, length=70,
                     command=lambda v, lid=layer["id"]: self.cmd("audio.mixer", {"layer_id": lid, "pan": float(v)})
                     ).pack(side="right", padx=4)

    def toggle(self, lid, key, current):
        self.cmd("audio.mixer", {"layer_id": lid, key: (not current)})
        self.build()

    def measure(self):
        res = self.cmd("audio.lufs")
        if res:
            self.meter.configure(text=f"{res.get('integrated', -70):.1f} LUFS · peak {res.get('peak', 0):.2f}")
            self.app.log(f"integrated loudness {res.get('integrated')} LUFS")

    def normalise(self, target):
        self.cmd("audio.normalize", {"target": target})
        self.measure()

    def export_wav(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            self.cmd("audio.export_wav", {"path": path})

    # ---- reactive audio --------------------------------------------
    def detect_beats(self):
        res = self.cmd("audio.beats", {"add_markers": True})
        if res:
            self.app.log(f"{res.get('count', 0)} beats · {res.get('tempo', 0)} BPM")
            self.app.timeline.redraw()

    def react_scale(self):
        layer = self.app.selected_layer()
        if not layer:
            self.app.log("select a layer to drive with the music")
            return
        res = self.cmd("audio.react", {"layer_id": layer["id"], "source": "low",
                                       "property": "scale", "lo": 96.0, "hi": 116.0,
                                       "smoothing": 0.25, "every_ms": 80})
        if res:
            self.app.log(f"baked {res.get('keys')} scale keys from the audio")
            self.app.refresh_all()

    def add_analyser(self):
        layer = self.app.selected_layer()
        if not layer:
            self.app.log("select a layer for the analyser")
            return
        res = self.cmd("audio.analyser", {"layer_id": layer["id"], "kind": "spectrum"})
        if res:
            self.app.log(f"added {res.get('effect')} analyser")
            self.app.refresh_all()


# ====================================================================== captions
class CaptionsPanel(PanelBase):
    title = "CAPTIONS"

    def __init__(self, master, app):
        super().__init__(master, app)
        for txt, fn in (("＋Cue", self.add_cue), ("Import", self.import_cues), ("Export", self.export_cues)):
            tk.Button(self.toolbar, text=txt, bd=0, bg=C.panel2, fg=C.muted, font=mono(7),
                      command=fn).pack(side="right", padx=2)
        self.listbox = tk.Listbox(self, bg=C.panel2, fg=C.text, bd=0, highlightthickness=0,
                                  selectbackground=C.sel, font=font(8), activestyle="none")
        self.listbox.pack(fill="both", expand=True, padx=6, pady=4)
        entry_row = tk.Frame(self, bg=C.panel)
        entry_row.pack(fill="x", padx=6, pady=4)
        self.entry = tk.Entry(entry_row, bg=C.panel2, fg=C.text, bd=0, font=font(9), insertbackground=C.text)
        self.entry.pack(side="left", fill="x", expand=True, ipady=3)
        tk.Button(entry_row, text="Add at playhead", bd=0, bg=C.panel2, fg=C.text, font=font(8),
                  command=self.add_cue).pack(side="left", padx=4)
        self.rebuild()

    def rebuild(self):
        self.listbox.delete(0, "end")
        for cue in sorted(self.app.reg.project.data.get("captions") or [], key=lambda c: c.get("start", 0)):
            self.listbox.insert("end", f"{cue['start']/1000:6.2f}s  {cue['text'][:44]}")

    def add_cue(self):
        text = (self.entry.get() or "").strip()
        if not text:
            return
        t = int(self.app.t)
        self.cmd("caption.add", {"start": t, "end": t + 2200, "text": text})
        self.entry.delete(0, "end")
        self.rebuild()

    def import_cues(self):
        path = filedialog.askopenfilename(filetypes=[("Subtitles", "*.srt *.vtt"), ("All", "*.*")])
        if path:
            self.cmd("caption.import", {"path": path, "replace": True})
            self.rebuild()

    def export_cues(self):
        path = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SRT", "*.srt"), ("VTT", "*.vtt")])
        if path:
            self.cmd("caption.export", {"path": path})


# ====================================================================== tracking
class TrackerPanel(PanelBase):
    title = "MOTION TRACKING"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.points: List[dict] = []
        body = tk.Frame(self, bg=C.panel)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        row = tk.Frame(body, bg=C.panel)
        row.pack(fill="x")
        self.x = self.number(row, "X", 540, lambda v: None, 7)
        self.y = self.number(row, "Y", 960, lambda v: None, 7)
        row2 = tk.Frame(body, bg=C.panel)
        row2.pack(fill="x", pady=6)
        for txt, fn in (("Track forwards", self.track), ("Attach to layer", lambda: self.apply("attach")),
                        ("Stabilise layer", lambda: self.apply("stabilize"))):
            tk.Button(row2, text=txt, bd=0, bg=C.panel2, fg=C.text, font=font(8),
                      command=fn).pack(side="left", padx=3)
        self.status = label(body, "Pick a video layer, then track.", C.panel, C.dim, 8)
        self.status.pack(anchor="w", pady=4)
        self.canvas = tk.Canvas(body, bg=C.bg, height=90, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

    def _media_layer(self):
        layer = self.selected_layer()
        if not layer or layer.get("kind") not in ("video", "image") or layer.get("media_id") is None:
            self.app.log("select a video layer with media first")
            return None
        return layer

    def track(self):
        layer = self._media_layer()
        if not layer:
            return
        try:
            x = float(self.x.get() or 0)
            y = float(self.y.get() or 0)
        except ValueError:
            return
        res = self.cmd("track.point", {"media_id": layer["media_id"], "x": x, "y": y,
                                       "start": int(layer.get("start", 0)),
                                       "end": int(layer.get("end", self.app.reg.project.duration)),
                                       "step": 100})
        if res:
            self.points = res.get("points") or []
            self.status.configure(text=f"tracked {len(self.points)} points")
            self.draw()

    def apply(self, mode: str):
        layer = self.selected_layer()
        if not layer or not self.points:
            self.app.log("track a point first")
            return
        self.cmd("track.apply", {"layer_id": layer["id"], "points": self.points, "mode": mode})
        self.app.log(f"{mode} applied · {len(self.points)} keys")

    def draw(self):
        self.canvas.delete("all")
        w = max(60, self.canvas.winfo_width() or 240)
        h = max(40, self.canvas.winfo_height() or 90)
        if not self.points:
            return
        xs = [p["x"] for p in self.points]
        ys = [p["y"] for p in self.points]
        lo_x, hi_x = min(xs), max(xs)
        lo_y, hi_y = min(ys), max(ys)
        span_x = max(1e-3, hi_x - lo_x)
        span_y = max(1e-3, hi_y - lo_y)
        pts = []
        for p in self.points:
            pts.extend([8 + (w - 16) * (p["x"] - lo_x) / span_x, 8 + (h - 16) * (p["y"] - lo_y) / span_y])
        if len(pts) >= 4:
            self.canvas.create_line(*pts, fill=C.good, width=1)
        for i, p in enumerate(self.points):
            if i % 5 == 0:
                x, y = pts[i * 2], pts[i * 2 + 1]
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=C.accent, outline="")
