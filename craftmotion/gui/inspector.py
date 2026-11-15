"""Inspector: layer properties, transforms, content, effects and looks."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

from ..core import effects as FX
from ..core.model import BLEND_MODES
from ..core.presets import BUILTIN_LOOKS
from ..core.util import kf_find, parse_color, pretty, to_argb
from .theme import C, KeyButton, ScrollFrame, font, icon_button, label, mono, section

TRANSFORM_ROWS = [
    ("pos", "Position", 3, "px"),
    ("anchor", "Anchor", 2, "px"),
    ("scale", "Scale", 2, "×"),
    ("rot", "Rotation", 1, "°"),
    ("skew", "Skew", 2, "°"),
]


class InspectorPanel(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=C.panel)
        self.app = app
        self.layer_id: Optional[int] = None
        self._entries: Dict[str, tk.Entry] = {}
        self._kf_buttons: Dict[str, KeyButton] = {}
        self._effect_open: Dict[int, bool] = {}
        self._building = False

        head = tk.Frame(self, bg=C.panel, height=28)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="INSPECTOR", bg=C.panel, fg=C.dim, font=font(8, True)).pack(side="left", padx=10)
        self.kind_label = tk.Label(head, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.kind_label.pack(side="right", padx=8)

        self.scroll = ScrollFrame(self, bg=C.panel)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner
        self.rebuild()

    # ------------------------------------------------------------ public
    def focus_name(self):
        e = self._entries.get("name")
        if e:
            e.focus_set()
            e.select_range(0, "end")

    def rebuild(self):
        layer = self.app.selected_layer()
        self.layer_id = layer["id"] if layer else None
        self._entries.clear()
        self._kf_buttons.clear()
        for w in self.body.winfo_children():
            w.destroy()
        if layer is None:
            self._empty_state()
            self.kind_label.configure(text="")
            return
        self.kind_label.configure(text=str(layer.get("kind", "")).upper())
        self._build_layer_section(layer)
        self._build_timing(layer)
        self._build_transform(layer)
        self._build_content(layer)
        self._build_masks_styles(layer)
        self._build_expressions(layer)
        self._build_effects(layer)
        self._build_looks(layer)

    def refresh_values(self):
        """Update entries in place (called while scrubbing / animating)."""
        layer = self.app.selected_layer()
        if layer is None or layer["id"] != self.layer_id:
            return
        for prop, btn in self._kf_buttons.items():
            btn.set_state(self._kf_state(layer, prop))
        focus = self.focus_get()
        for key, e in self._entries.items():
            if e is focus:
                continue
            if key.startswith("tf."):
                prop, idx = key[3:].split(".")
                vals = layer.get("transform", {}).get(prop)
                if vals is None:
                    continue
                val = float(vals[int(idx)]) if int(idx) < len(vals) else 0.0
                e.delete(0, "end")
                e.insert(0, self._fmt(val))

    # ------------------------------------------------------------ sections
    def _empty_state(self):
        f = tk.Frame(self.body, bg=C.panel)
        f.pack(fill="both", expand=True, pady=40)
        label(f, "Nothing selected", C.panel, C.text, 11, True).pack()
        label(f, "Select a layer in the timeline\nor add one from the toolbar.", C.panel, C.dim, 9).pack(pady=6)
        for text, cmd in (("Add text", lambda: self.app.add_layer("text")),
                          ("Add shape", lambda: self.app.add_layer("shape")),
                          ("Add adjustment", lambda: self.app.add_layer("adjustment"))):
            icon_button(f, text, cmd, bg=C.panel2).pack(fill="x", padx=24, pady=2)

    def _build_layer_section(self, layer: dict):
        b = section(self.body, "Layer")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x")
        e = tk.Entry(row, bg=C.panel2, fg=C.text, bd=0, insertbackground=C.text, font=font(10))
        e.insert(0, str(layer.get("name", "")))
        e.pack(side="left", fill="x", expand=True, ipady=3)
        e.bind("<Return>", lambda ev: self._commit(layer, "name", e.get()))
        e.bind("<FocusOut>", lambda ev: self._commit(layer, "name", e.get()))
        self._entries["name"] = e

        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(6, 0))
        label(row2, "Blend", C.panel, C.muted, 9).pack(side="left")
        var = tk.StringVar(value=str(layer.get("blend", "normal")))
        combo = ttk.Combobox(row2, textvariable=var, values=BLEND_MODES, width=12, state="readonly",
                             font=font(9))
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>",
                   lambda ev: self._commit(layer, "blend", var.get(), quiet=True))
        hidden_var = tk.BooleanVar(value=bool(layer.get("hidden")))
        tk.Checkbutton(row2, text="Hidden", variable=hidden_var, bg=C.panel, fg=C.muted,
                       selectcolor=C.panel2, activebackground=C.panel, font=font(9), bd=0,
                       command=lambda: self._commit(layer, "hidden", hidden_var.get(), quiet=True)
                       ).pack(side="right")
        locked_var = tk.BooleanVar(value=bool(layer.get("locked")))
        tk.Checkbutton(row2, text="Locked", variable=locked_var, bg=C.panel, fg=C.muted,
                       selectcolor=C.panel2, activebackground=C.panel, font=font(9), bd=0,
                       command=lambda: self._commit(layer, "locked", locked_var.get(), quiet=True)
                       ).pack(side="right", padx=4)
        if layer.get("parent"):
            label(b, f"⛓ Parented to layer {layer['parent']}", C.panel, C.muted, 8).pack(anchor="w", pady=(4, 0))
        if layer.get("matte"):
            label(b, f"◐ Matte ← layer {layer['matte']}", C.panel, C.muted, 8).pack(anchor="w", pady=(2, 0))

    def _build_timing(self, layer: dict):
        b = section(self.body, "Timing")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x")
        label(row, "Start", C.panel, C.muted, 9).pack(side="left")
        e1 = tk.Entry(row, width=8, bg=C.panel2, fg=C.text, bd=0, font=mono(9), insertbackground=C.text)
        e1.insert(0, str(layer.get("start", 0)))
        e1.pack(side="left", padx=4, ipady=2)
        e1.bind("<Return>", lambda ev: self._commit(layer, "start", self._int(e1.get(), layer["start"])))
        label(row, "End", C.panel, C.muted, 9).pack(side="left", padx=(8, 0))
        e2 = tk.Entry(row, width=8, bg=C.panel2, fg=C.text, bd=0, font=mono(9), insertbackground=C.text)
        e2.insert(0, str(layer.get("end", 0)))
        e2.pack(side="left", padx=4, ipady=2)
        e2.bind("<Return>", lambda ev: self._commit(layer, "end", self._int(e2.get(), layer["end"])))
        self._entries["start"] = e1
        self._entries["end"] = e2
        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(4, 0))
        icon_button(row2, "Start › playhead", lambda: self._commit(layer, "start", int(self.app.t)),
                    "Set layer start to the playhead", bg=C.panel2, size=8).pack(side="left")
        icon_button(row2, "End › playhead", lambda: self._commit(layer, "end", int(self.app.t)),
                    "Set layer end to the playhead", bg=C.panel2, size=8).pack(side="left", padx=4)

    def _build_transform(self, layer: dict):
        b = section(self.body, "Transform")
        tf = layer.get("transform", {})
        for prop, title, dims, unit in TRANSFORM_ROWS:
            row = tk.Frame(b, bg=C.panel)
            row.pack(fill="x", pady=1)
            label(row, title, C.panel, C.muted, 9, width=None).pack(side="left")
            kb = KeyButton(row, lambda p=prop: self._toggle_key(layer, p))
            kb.pack(side="right")
            kb.set_state(self._kf_state(layer, prop))
            self._kf_buttons[prop] = kb
            vals = tf.get(prop, [0.0] * dims)
            for i in range(dims):
                e = tk.Entry(row, width=6, bg=C.panel2, fg=C.text, bd=0, font=mono(9),
                             insertbackground=C.text, justify="center")
                val = float(vals[i]) if i < len(vals) else 0.0
                e.insert(0, self._fmt(val))
                e.pack(side="left", padx=2, ipady=2)
                e.bind("<Return>", lambda ev, er=e, p=prop, ix=i: self._commit_vec(layer, p, ix, er.get()))
                e.bind("<FocusOut>", lambda ev, er=e, p=prop, ix=i: self._commit_vec(layer, p, ix, er.get()))
                self._entries[f"tf.{prop}.{i}"] = e
                if i < dims - 1:
                    label(row, ",", C.panel, C.dim, 8).pack(side="left")
            if unit:
                label(row, unit, C.panel, C.dim, 8).pack(side="left", padx=(2, 4))
        # opacity
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(4, 0))
        label(row, "Opacity", C.panel, C.muted, 9).pack(side="left")
        kb = KeyButton(row, lambda: self._toggle_key(layer, "opacity"))
        kb.pack(side="right")
        kb.set_state(self._kf_state(layer, "opacity"))
        self._kf_buttons["opacity"] = kb
        val = float(tf.get("opacity", [1.0])[0])
        s = tk.Scale(row, from_=0, to=1, resolution=0.01, orient="horizontal", showvalue=True,
                     bg=C.panel, fg=C.text, troughcolor=C.track, highlightthickness=0, bd=0,
                     sliderrelief="flat", width=8, font=mono(8),
                     command=lambda v, l=layer: self._set_static(l, "opacity", float(v)),
                     activebackground=C.accent)
        s.set(val)
        s.pack(side="left", fill="x", expand=True, padx=4)

    # ------------------------------------------------------------ parity
    def _build_masks_styles(self, layer: dict):
        b = section(self.body, "Masks & styles")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x")
        icon_button(row, "＋ Mask", lambda: self._add_mask(layer), bg=C.panel2, size=8).pack(side="left")
        icon_button(row, "＋ Style", lambda: self._add_style(layer), bg=C.panel2, size=8).pack(side="left", padx=4)
        masks = layer.get("masks") or []
        styles = layer.get("styles") or []
        for i, mask in enumerate(masks):
            r = tk.Frame(b, bg=C.panel)
            r.pack(fill="x", pady=1)
            label(r, f"◧ {mask.get('shape', 'polygon')} · {mask.get('mode', 'add')} · feather {mask.get('feather', 0):g}",
                  C.panel, C.muted, 8).pack(side="left")
            icon_button(r, "✕", lambda idx=i: self._rm_mask(layer, idx), bg=C.panel2, size=8).pack(side="right")
        for i, st in enumerate(styles):
            r = tk.Frame(b, bg=C.panel)
            r.pack(fill="x", pady=1)
            params = st.get("params") or {}
            label(r, f"◈ {st.get('type', '')} · {(params.get('color') or '')}",
                  C.panel, C.muted, 8).pack(side="left")
            icon_button(r, "✕", lambda idx=i: self._rm_style(layer, idx), bg=C.panel2, size=8).pack(side="right")
        if layer.get("kind") in ("shape", "solid"):
            r2 = tk.Frame(b, bg=C.panel)
            r2.pack(fill="x", pady=(4, 0))
            icon_button(r2, "Trim 0→70%", lambda: self.app.run_command(
                "shape.trim", {"layer_id": layer["id"], "start": 0, "end": 70}, quiet=True),
                bg=C.panel2, size=8).pack(side="left")
            icon_button(r2, "Repeater ×5", lambda: self.app.run_command(
                "shape.repeater", {"layer_id": layer["id"], "copies": 5, "x": 28, "y": 12, "scale": 0.92,
                                   "end_opacity": 0.25}, quiet=True), bg=C.panel2, size=8).pack(side="left", padx=4)
            icon_button(r2, "Clear ops", lambda: self.app.run_command(
                "shape.clear_ops", {"layer_id": layer["id"]}, quiet=True), bg=C.panel2, size=8).pack(side="left")
        if layer.get("kind") == "text":
            r3 = tk.Frame(b, bg=C.panel)
            r3.pack(fill="x", pady=(4, 0))
            icon_button(r3, "＋ Animator", lambda: self.app.run_command(
                "text.animator_add", {"layer_id": layer["id"],
                                      "selector": {"type": "range", "start": 0, "end": 100},
                                      "params": {"y": -40, "opacity": 0}}, quiet=True),
                bg=C.panel2, size=8).pack(side="left")

    def _add_mask(self, layer: dict):
        from tkinter import simpledialog
        shapes = ["polygon", "ellipse", "rect"]
        choice = simpledialog.askstring("Mask", "shape: polygon / ellipse / rect", initialvalue="polygon",
                                        parent=self.winfo_toplevel())
        if not choice:
            return
        choice = choice.strip().lower()
        if choice not in shapes:
            choice = "polygon"
        self.app.run_command("mask.add", {"layer_id": layer["id"], "shape": choice, "feather": 0.0}, quiet=True)
        self.app.refresh_all()

    def _rm_mask(self, layer: dict, index: int):
        self.app.run_command("mask.remove", {"layer_id": layer["id"], "index": index}, quiet=True)
        self.app.refresh_all()

    def _add_style(self, layer: dict):
        from tkinter import simpledialog
        kind = simpledialog.askstring("Layer style",
                                      "drop_shadow / outer_glow / inner_glow / stroke / color_overlay / bevel / satin",
                                      initialvalue="drop_shadow", parent=self.winfo_toplevel())
        if not kind:
            return
        kind = kind.strip().lower()
        self.app.run_command("style.add", {"layer_id": layer["id"], "type": kind}, quiet=True)
        self.app.refresh_all()

    def _rm_style(self, layer: dict, index: int):
        self.app.run_command("style.remove", {"layer_id": layer["id"], "index": index}, quiet=True)
        self.app.refresh_all()

    def _build_expressions(self, layer: dict):
        b = section(self.body, "Expressions")
        exprs = layer.get("expressions") or {}
        for prop in ("pos", "scale", "rot", "opacity"):
            if prop in exprs:
                r = tk.Frame(b, bg=C.panel)
                r.pack(fill="x", pady=1)
                label(r, f"{prop}: {exprs[prop][:34]}", C.panel, C.accent2, 8).pack(side="left")
                icon_button(r, "✕", lambda p=prop: self._clear_expr(layer, p), bg=C.panel2, size=8).pack(side="right")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(4, 0))
        self._expr_prop = tk.StringVar(value="pos")
        ttk.Combobox(row, textvariable=self._expr_prop, values=["pos", "scale", "rot", "opacity"],
                     width=7, state="readonly", font=font(8)).pack(side="left")
        e = tk.Entry(row, bg=C.panel2, fg=C.text, bd=0, font=mono(8), insertbackground=C.text)
        e.insert(0, exprs.get("pos", "wiggle(2, 20)"))
        e.pack(side="left", fill="x", expand=True, padx=4, ipady=2)

        def apply_expr(_e=None):
            src = e.get().strip()
            if not src:
                return
            self.app.run_command("expression.set", {"layer_id": layer["id"],
                                                    "property": self._expr_prop.get(),
                                                    "expression": src}, quiet=True)
            self.app.refresh_all()
        e.bind("<Return>", apply_expr)
        icon_button(row, "Set", apply_expr, bg=C.panel2, size=8).pack(side="right")

    def _clear_expr(self, layer: dict, prop: str):
        self.app.run_command("expression.clear", {"layer_id": layer["id"], "property": prop}, quiet=True)
        self.app.refresh_all()

    def _build_content(self, layer: dict):
        kind = layer.get("kind")
        if kind == "text":
            self._build_text(layer)
        elif kind == "shape":
            self._build_shape(layer)
        elif kind in ("video", "image"):
            self._build_media(layer)
        elif kind == "audio":
            self._build_audio(layer)
        elif kind == "solid":
            self._build_solid(layer)
        elif kind == "adjustment":
            b = section(self.body, "Adjustment")
            label(b, "Effects on this layer grade everything beneath it.", C.panel, C.dim, 8,
                  wraplength=260, justify="left").pack(anchor="w")

    # ---- text
    def _build_text(self, layer: dict):
        b = section(self.body, "Text")
        txt = tk.Text(b, height=3, bg=C.panel2, fg=C.text, bd=0, insertbackground=C.text,
                      font=font(10), wrap="word", padx=6, pady=4)
        txt.insert("1.0", str(layer.get("text", "")))
        txt.pack(fill="x")
        txt.bind("<KeyRelease>", lambda e: self._commit(layer, "text", txt.get("1.0", "end-1c"), quiet=True))
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(6, 0))
        self._num_field(row, "Size", layer.get("size", 96), lambda v: self._commit(layer, "size", v))
        self._combo_field(row, "Font", str(layer.get("font", "DejaVu Sans")),
                          ["DejaVu Sans", "DejaVu Sans Bold", "DejaVu Sans Mono"],
                          lambda v: self._commit(layer, "font", v))
        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(4, 0))
        self._combo_field(row2, "Align", str(layer.get("align", "center")),
                          ["left", "center", "right"], lambda v: self._commit(layer, "align", v))
        self._num_field(row2, "Wrap", layer.get("wrap", 900), lambda v: self._commit(layer, "wrap", v))
        row3 = tk.Frame(b, bg=C.panel)
        row3.pack(fill="x", pady=(4, 0))
        self._color_field(row3, "Fill", layer.get("color", "#FFFFFFFF"),
                          lambda v: self._commit(layer, "color", v))
        self._color_field(row3, "Stroke", layer.get("stroke_color", "#00000000"),
                          lambda v: self._commit(layer, "stroke_color", v))
        row4 = tk.Frame(b, bg=C.panel)
        row4.pack(fill="x", pady=(4, 0))
        self._num_field(row4, "Stroke px", layer.get("stroke_width", 0),
                        lambda v: self._commit(layer, "stroke_width", v))
        self._num_field(row4, "Shadow", layer.get("shadow", 0), lambda v: self._commit(layer, "shadow", v))

    # ---- shape
    def _build_shape(self, layer: dict):
        b = section(self.body, "Shape")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x")
        shapes = ["rect", "ellipse", "triangle", "star", "polygon", "diamond", "heart", "plus",
                  "line", "ring", "path"]
        self._combo_field(row, "Type", str(layer.get("shape", "rect")), shapes,
                          lambda v: self._commit(layer, "shape", v))
        self._num_field(row, "W", layer.get("w", 400), lambda v: self._commit(layer, "w", v), w=5)
        self._num_field(row, "H", layer.get("h", 400), lambda v: self._commit(layer, "h", v), w=5)
        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(4, 0))
        if layer.get("shape") == "rect":
            self._num_field(row2, "Radius", layer.get("radius", 0), lambda v: self._commit(layer, "radius", v))
        if layer.get("shape") in ("star", "polygon"):
            self._num_field(row2, "Points", layer.get("points", 5), lambda v: self._commit(layer, "points", int(v)))
            self._num_field(row2, "Inner", layer.get("inner", 0.5),
                            lambda v: self._commit(layer, "inner", v))
        row3 = tk.Frame(b, bg=C.panel)
        row3.pack(fill="x", pady=(4, 0))
        self._color_field(row3, "Fill", layer.get("color", "#FF4C8DFF"),
                          lambda v: self._commit(layer, "color", v))
        if layer.get("color2"):
            self._color_field(row3, "Grad", layer.get("color2"), lambda v: self._commit(layer, "color2", v))
        else:
            icon_button(row3, "＋grad", lambda: self._commit(layer, "color2", layer.get("color", "#FF4C8DFF")),
                        "Add a gradient end colour", bg=C.panel2, size=8).pack(side="left", padx=2)
        row4 = tk.Frame(b, bg=C.panel)
        row4.pack(fill="x", pady=(4, 0))
        self._color_field(row4, "Stroke", layer.get("stroke_color", "#00000000"),
                          lambda v: self._commit(layer, "stroke_color", v))
        self._num_field(row4, "px", layer.get("stroke_width", 0), lambda v: self._commit(layer, "stroke_width", v))

    # ---- media
    def _build_media(self, layer: dict):
        b = section(self.body, "Media")
        item = self.app.reg.project.get_media(layer.get("media_id"))
        if item:
            name = item.get("name") or item.get("filename")
            label(b, f"🎞 {name}", C.panel, C.text, 9).pack(anchor="w")
            meta = f"{item.get('width')}×{item.get('height')} · {(item.get('duration_ms') or 0)/1000:.1f}s"
            if item.get("missing"):
                meta += " · ⚠ missing file"
            label(b, meta, C.panel, C.warn if item.get("missing") else C.dim, 8).pack(anchor="w")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(6, 0))
        self._combo_field(row, "Fit", str(layer.get("fit", "cover")), ["cover", "contain", "stretch", "none"],
                          lambda v: self._commit(layer, "fit", v))
        self._num_field(row, "Speed", layer.get("speed", 1.0), lambda v: self._commit(layer, "speed", v))
        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(4, 0))
        self._num_field(row2, "Source in", layer.get("source_in", 0),
                        lambda v: self._commit(layer, "source_in", int(v)))
        if layer.get("kind") == "video":
            var = tk.BooleanVar(value=bool(layer.get("include_audio", True)))
            tk.Checkbutton(row2, text="Clip audio", variable=var, bg=C.panel, fg=C.muted,
                           selectcolor=C.panel2, activebackground=C.panel, font=font(9), bd=0,
                           command=lambda: self._commit(layer, "include_audio", var.get(), quiet=True)
                           ).pack(side="right")
        row3 = tk.Frame(b, bg=C.panel)
        row3.pack(fill="x", pady=(4, 0))
        icon_button(row3, "Replace media…", lambda: self.app.replace_media(layer), bg=C.panel2, size=8
                    ).pack(side="left")

    # ---- audio
    def _build_audio(self, layer: dict):
        b = section(self.body, "Audio")
        item = self.app.reg.project.get_media(layer.get("media_id"))
        if item:
            label(b, f"🎵 {item.get('name') or item.get('filename')}", C.panel, C.text, 9).pack(anchor="w")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(4, 0))
        label(row, "Volume", C.panel, C.muted, 9).pack(side="left")
        kb = KeyButton(row, lambda: self._toggle_key(layer, "volume"))
        kb.pack(side="right")
        kb.set_state(self._kf_state(layer, "volume"))
        self._kf_buttons["volume"] = kb
        s = tk.Scale(row, from_=0, to=2, resolution=0.01, orient="horizontal", showvalue=True,
                     bg=C.panel, fg=C.text, troughcolor=C.track, highlightthickness=0, bd=0,
                     sliderrelief="flat", width=8, font=mono(8),
                     command=lambda v, l=layer: self._set_static(l, "volume", float(v)),
                     activebackground=C.accent)
        s.set(float(layer.get("volume", 1.0)))
        s.pack(side="left", fill="x", expand=True, padx=4)
        row2 = tk.Frame(b, bg=C.panel)
        row2.pack(fill="x", pady=(4, 0))
        self._num_field(row2, "Fade in", layer.get("fade_in", 0),
                        lambda v: self._commit(layer, "fade_in", int(v)))
        self._num_field(row2, "Fade out", layer.get("fade_out", 0),
                        lambda v: self._commit(layer, "fade_out", int(v)))

    # ---- solid
    def _build_solid(self, layer: dict):
        b = section(self.body, "Solid")
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x")
        self._color_field(row, "Colour", layer.get("color", "#FF202020"),
                          lambda v: self._commit(layer, "color", v))
        icon_button(row, "gradient", lambda: self._commit(layer, "color2", "#FF3D2C8D"),
                    "Add gradient", bg=C.panel2, size=8).pack(side="left", padx=2)
        if layer.get("color2"):
            self._color_field(row, "Grad", layer.get("color2"), lambda v: self._commit(layer, "color2", v))

    # ---- effects
    def _build_effects(self, layer: dict):
        b = section(self.body, "Effects")
        stack = layer.get("effects") or []
        row = tk.Frame(b, bg=C.panel)
        row.pack(fill="x", pady=(0, 4))
        icon_button(row, "＋ Add effect", self.app.open_effect_dialog, bg=C.accent, fg="#141414",
                    size=9).pack(side="left")
        icon_button(row, "Save stack…", lambda: self.app.save_preset_from_layer(layer), bg=C.panel2,
                    size=8).pack(side="left", padx=4)
        for i, eff in enumerate(stack):
            self._effect_row(b, layer, i, eff)

    def _effect_row(self, parent, layer: dict, index: int, eff: dict):
        spec = FX.EFFECTS.get(eff.get("type", ""))
        title = spec.label if spec else str(eff.get("type"))
        frame = tk.Frame(parent, bg=C.panel2, bd=0)
        frame.pack(fill="x", pady=2)
        head = tk.Frame(frame, bg=C.panel2)
        head.pack(fill="x", padx=4, pady=3)
        open_state = self._effect_open.get(index, index == len(layer.get("effects", [])) - 1)
        arrow = "▾" if open_state else "▸"
        btn = tk.Label(head, text=f"{arrow} {title}", bg=C.panel2, fg=C.text, font=font(9, True),
                       anchor="w", cursor="hand2")
        btn.pack(side="left", fill="x", expand=True)
        btn.bind("<Button-1>", lambda e, i=index: self._toggle_effect_open(i))
        enabled_var = tk.BooleanVar(value=bool(eff.get("enabled", True)))
        tk.Checkbutton(head, variable=enabled_var, bg=C.panel2, fg=C.text, selectcolor=C.track,
                       activebackground=C.panel2, bd=0, highlightthickness=0, font=font(8),
                       command=lambda: self._toggle_effect(layer, index, enabled_var.get())
                       ).pack(side="left")
        icon_button(head, "✕", lambda: self._remove_effect(layer, index), bg=C.panel2, fg=C.muted,
                    size=8, pad=(4, 0)).pack(side="right")
        icon_button(head, "▼", lambda: self._move_effect(layer, index, 1), "Move down", bg=C.panel2,
                    fg=C.muted, size=7, pad=(3, 0)).pack(side="right")
        icon_button(head, "▲", lambda: self._move_effect(layer, index, -1), "Move up", bg=C.panel2,
                    fg=C.muted, size=7, pad=(3, 0)).pack(side="right")
        if not open_state or spec is None:
            return
        body = tk.Frame(frame, bg=C.panel2)
        body.pack(fill="x", padx=6, pady=(0, 5))
        for pname, prm in spec.params.items():
            row = tk.Frame(body, bg=C.panel2)
            row.pack(fill="x", pady=1)
            label(row, prm.label or pname, C.panel2, C.muted, 8, width=None).pack(side="left")
            kb = KeyButton(row, lambda p=pname, i=index: self._toggle_param_key(layer, i, p),
                           bg=C.panel2, size=14)
            kb.pack(side="right")
            kb.set_state(self._param_kf_state(layer, index, pname))
            cur = (eff.get("params") or {}).get(pname, prm.default)
            if prm.kind == "color":
                self._color_field(row, "", cur, lambda v, i=index, p=pname: self._set_param(layer, i, p, v),
                                  bg=C.panel2)
            elif prm.kind == "bool":
                var = tk.BooleanVar(value=bool(cur))
                tk.Checkbutton(row, variable=var, bg=C.panel2, selectcolor=C.track, bd=0,
                               highlightthickness=0,
                               command=lambda i=index, p=pname, v=var: self._set_param(layer, i, p, v.get())
                               ).pack(side="right", padx=4)
            elif prm.kind == "choice":
                self._combo_field(row, "", cur, prm.options,
                                  lambda v, i=index, p=pname: self._set_param(layer, i, p, v),
                                  bg=C.panel2, width=10)
            else:
                e = tk.Entry(row, width=7, bg=C.track, fg=C.text, bd=0, font=mono(8),
                             insertbackground=C.text, justify="center")
                e.insert(0, pretty(cur))
                e.pack(side="right", padx=2, ipady=1)
                e.bind("<Return>", lambda ev, i=index, p=pname, er=e: self._param_commit(layer, i, p, er.get()))
                e.bind("<FocusOut>", lambda ev, i=index, p=pname, er=e: self._param_commit(layer, i, p, er.get()))
                if prm.kind in ("float", "int"):
                    sl = tk.Scale(row, from_=prm.lo, to=prm.hi, resolution=prm.step,
                                  orient="horizontal", showvalue=False, bg=C.panel2,
                                  troughcolor=C.track, highlightthickness=0, bd=0, width=6,
                                  sliderrelief="flat", activebackground=C.accent,
                                  command=lambda v, i=index, p=pname: self._set_param(layer, i, p, float(v)))
                    sl.set(float(cur) if isinstance(cur, (int, float)) else float(prm.default))
                    sl.pack(side="left", fill="x", expand=True, padx=(4, 2))

    # ---- looks
    def _build_looks(self, layer: dict):
        b = section(self.body, "Looks")
        wrap = tk.Frame(b, bg=C.panel)
        wrap.pack(fill="x")
        for i, (key, look) in enumerate(BUILTIN_LOOKS.items()):
            btn = icon_button(wrap, look["label"], lambda k=key: self._apply_look(layer, k),
                              bg=C.panel2, fg=C.muted, size=8, pad=(6, 3))
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=1, pady=1)
            wrap.grid_columnconfigure(i % 2, weight=1)

    # ------------------------------------------------------------ field helpers
    def _num_field(self, parent, title: str, value, on_commit: Callable, w: int = 6):
        try:
            bg = parent.cget("bg")
        except Exception:
            bg = C.panel
        label(parent, title, bg, C.muted, 9).pack(side="left", padx=(6, 2))
        e = tk.Entry(parent, width=w, bg=C.panel2 if bg == C.panel else C.track, fg=C.text, bd=0,
                     font=mono(9), insertbackground=C.text, justify="center")
        e.insert(0, pretty(value))
        e.pack(side="left", ipady=2)
        e.bind("<Return>", lambda ev: self._num_commit(e, on_commit))
        e.bind("<FocusOut>", lambda ev: self._num_commit(e, on_commit))
        return e

    @staticmethod
    def _num_commit(entry: tk.Entry, on_commit: Callable):
        try:
            txt = entry.get().strip().replace("×", "").replace("°", "")
            val = float(txt)
        except ValueError:
            return
        on_commit(val)

    def _combo_field(self, parent, title: str, value: str, values: List[str], on_change: Callable,
                     bg: str = C.panel, width: int = 9):
        if title:
            label(parent, title, bg, C.muted, 9).pack(side="left", padx=(6, 2))
        var = tk.StringVar(value=str(value))
        c = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly",
                         font=font(8))
        c.pack(side="left")
        c.bind("<<ComboboxSelected>>", lambda e: on_change(var.get()))
        return c

    def _color_field(self, parent, title: str, value: str, on_change: Callable, bg: str = C.panel):
        if title:
            label(parent, title, bg, C.muted, 9).pack(side="left", padx=(6, 2))
        rgba = parse_color(value)
        swatch = tk.Label(parent, text="  ", bg=to_argb(rgba), width=3, bd=1, relief="flat",
                          cursor="hand2")
        swatch.pack(side="left", padx=2, ipady=3)

        def pick(_e=None):
            from tkinter import colorchooser
            r, g, b, a = parse_color(value)
            rgb, _ = colorchooser.askcolor(color="#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)),
                                           parent=self.winfo_toplevel(), title="Pick colour")
            if rgb:
                on_change(to_argb((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, a)))
        swatch.bind("<Button-1>", pick)

        def alpha(_e=None):
            from tkinter import simpledialog
            r, g, b, a = parse_color(value)
            v = simpledialog.askfloat("Alpha", "0–1", initialvalue=a, minvalue=0, maxvalue=1,
                                      parent=self.winfo_toplevel())
            if v is not None:
                on_change(to_argb((r, g, b, v)))
        swatch.bind("<Button-3>", alpha)
        return swatch

    # ------------------------------------------------------------ commits
    def _fmt(self, v: float) -> str:
        return ("%.2f" % v).rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-6 else str(int(round(v)))

    @staticmethod
    def _int(txt: str, fallback: int) -> int:
        try:
            return int(float(txt))
        except ValueError:
            return fallback

    def _commit(self, layer: dict, key: str, value, quiet: bool = False):
        if self._building:
            return
        if layer.get(key) == value:
            return
        self.app.reg.push_undo()
        layer[key] = value
        if key in ("start", "end") and layer["end"] <= layer["start"]:
            layer["end"] = layer["start"] + 500
        self.app.on_project_changed(rebuild=not quiet)

    def _commit_vec(self, layer: dict, prop: str, idx: int, txt: str):
        try:
            val = float(txt)
        except ValueError:
            return
        tf = layer.setdefault("transform", {})
        vals = list(tf.get(prop, [0.0] * 3))
        while len(vals) <= idx:
            vals.append(0.0)
        if abs(vals[idx] - val) < 1e-9:
            return
        vals[idx] = val
        tf[prop] = vals
        self.app.on_project_changed(rebuild=False)

    def _set_static(self, layer: dict, prop: str, value: float):
        tf = layer.setdefault("transform", {})
        if prop == "opacity":
            tf["opacity"] = [float(value)]
        elif prop == "scale":
            tf["scale"] = [float(value), float(value)]
        else:
            tf[prop] = [float(value)]
        self.app.request_preview_only()

    def _kf_state(self, layer: dict, prop: str) -> str:
        keys = (layer.get("keys") or {}).get(prop)
        if not keys:
            return "none"
        if kf_find(keys, int(self.app.t)) is not None:
            return "onkey"
        return "keyed"

    def _toggle_key(self, layer: dict, prop: str):
        from ..core.util import kf_add, kf_remove
        keys = layer.setdefault("keys", {}).setdefault(prop, [])
        existing = kf_find(keys, int(self.app.t))
        self.app.reg.push_undo()
        if existing is not None:
            kf_remove(keys, int(self.app.t))
        else:
            base = layer.get("transform", {}).get(prop, [0.0])
            dims = {"pos": 3, "anchor": 2, "scale": 2, "skew": 2, "rot": 1, "opacity": 1, "volume": 1}[prop]
            if prop == "volume":
                base = [float(layer.get("volume", 1.0))]
            from ..core.util import kf_eval
            vals = kf_eval(keys, int(self.app.t), list(base) + [0.0] * (dims - len(base))) if keys \
                else [float(x) for x in (list(base) + [0.0] * dims)[:dims]]
            kf_add(keys, int(self.app.t), vals, "ease_in_out")
        self.app.on_project_changed()

    def _toggle_param_key(self, layer: dict, index: int, param: str):
        from ..core.util import kf_add, kf_eval, kf_find, kf_remove
        eff = layer["effects"][index]
        spec = FX.EFFECTS.get(eff.get("type"))
        default = spec.params[param].default if spec and param in spec.params else 0.0
        keys = eff.setdefault("keys", {}).setdefault(param, [])
        self.app.reg.push_undo()
        existing = kf_find(keys, int(self.app.t))
        if existing is not None:
            kf_remove(keys, int(self.app.t))
        else:
            cur = (eff.get("params") or {}).get(param, default)
            try:
                cur_f = float(cur)
            except (TypeError, ValueError):
                cur_f = float(default) if isinstance(default, (int, float)) else 0.0
            vals = kf_eval(keys, int(self.app.t), [cur_f]) if keys else [cur_f]
            kf_add(keys, int(self.app.t), vals, "ease_in_out")
        self.app.on_project_changed()

    def _param_kf_state(self, layer: dict, index: int, param: str) -> str:
        keys = ((layer["effects"][index].get("keys") or {}).get(param))
        if not keys:
            return "none"
        return "onkey" if kf_find(keys, int(self.app.t)) is not None else "keyed"

    def _set_param(self, layer: dict, index: int, param: str, value):
        eff = layer["effects"][index]
        eff.setdefault("params", {})[param] = value
        self.app.request_preview_only()

    def _param_commit(self, layer: dict, index: int, param: str, txt: str):
        try:
            val = float(txt)
        except ValueError:
            return
        self._set_param(layer, index, param, val)
        self.app.on_project_changed(rebuild=False)

    def _toggle_effect_open(self, index: int):
        self._effect_open[index] = not self._effect_open.get(index, True)
        self.rebuild()

    def _toggle_effect(self, layer: dict, index: int, enabled: bool):
        layer["effects"][index]["enabled"] = enabled
        self.app.request_preview_only()

    def _remove_effect(self, layer: dict, index: int):
        self.app.reg.push_undo()
        layer["effects"].pop(index)
        self.app.on_project_changed()

    def _move_effect(self, layer: dict, index: int, direction: int):
        stack = layer["effects"]
        j = index + direction
        if 0 <= j < len(stack):
            stack[index], stack[j] = stack[j], stack[index]
            self.app.on_project_changed()

    def _apply_look(self, layer: dict, key: str):
        layer_id = layer["id"]
        self.app.reg.execute("effect.apply_look", {"layer_id": layer_id, "look": key})
        self.app.on_project_changed()
