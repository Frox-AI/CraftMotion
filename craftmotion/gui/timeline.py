"""Timeline: ruler, tracks, clip bars, keyframe diamonds, markers, playhead."""
from __future__ import annotations

import math
import tkinter as tk
from typing import Optional

from .theme import C, font, mono

HEADER_W = 196
RULER_H = 26
TRACK_H = 34
GAP = 2

PROP_COLORS = {
    "pos": "#3d8bfd", "scale": "#3ddc97", "rot": "#ffcc55", "opacity": "#c792ea",
    "anchor": "#7fd1ff", "skew": "#f78c6c", "color": "#ff9f6b", "volume": "#3ddc97",
}


class TimelinePanel(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=C.bg, bd=0, highlightthickness=0)
        self.app = app
        self.pps = 60.0          # pixels per second
        self.scroll_x = 0.0
        self._drag: Optional[dict] = None
        self._sel_bar: Optional[int] = None

        bar = tk.Frame(self, bg=C.panel, height=30)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="TIMELINE", bg=C.panel, fg=C.dim, font=font(8, True)).pack(side="left", padx=10)
        self.zoom_label = tk.Label(bar, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.zoom_label.pack(side="right", padx=8)
        self._btn(bar, "＋", lambda: self.zoom(1.25), "Zoom in (Ctrl+wheel)")
        self._btn(bar, "－", lambda: self.zoom(0.8), "Zoom out")
        self._btn(bar, "⇥", self.fit, "Fit timeline")

        self.canvas = tk.Canvas(self, bg=C.bg, highlightthickness=0, bd=0)
        self.hsb = tk.Scrollbar(self, orient="horizontal", command=self._on_hscroll)
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(xscrollcommand=self.hsb.set)

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<Double-Button-1>", self._on_double)

    # ------------------------------------------------------------ helpers
    def _btn(self, parent, text, cmd, tip=""):
        b = tk.Button(parent, text=text, command=cmd, bg=C.panel, fg=C.muted, font=font(10),
                      bd=0, relief="flat", padx=7, activebackground=C.panel2,
                      activeforeground=C.text, cursor="hand2", highlightthickness=0)
        b.pack(side="right")
        if tip:
            from .theme import Tooltip
            Tooltip(b, tip)
        return b

    def t2x(self, t_ms: float) -> float:
        return HEADER_W + (t_ms / 1000.0) * self.pps - self.scroll_x

    def x2t(self, x: float) -> float:
        return max(0.0, (x - HEADER_W + self.scroll_x) / self.pps * 1000.0)

    def zoom(self, factor: float, center_ms: Optional[float] = None):
        center = center_ms if center_ms is not None else self.app.t
        self.pps = max(8.0, min(600.0, self.pps * factor))
        w = max(1, self.canvas.winfo_width() - HEADER_W)
        self.scroll_x = max(0.0, (center / 1000.0) * self.pps - w / 2)
        self.redraw()

    def fit(self):
        dur = max(1000, self.app.reg.project.duration)
        w = max(100, self.canvas.winfo_width() - HEADER_W - 20)
        self.pps = max(8.0, min(600.0, w / (dur / 1000.0)))
        self.scroll_x = 0.0
        self.redraw()

    def ensure_visible(self, t_ms: float):
        x = self.t2x(t_ms)
        w = self.canvas.winfo_width()
        if x < HEADER_W + 40:
            self.scroll_x = max(0.0, (t_ms / 1000.0) * self.pps - 60)
        elif x > w - 60:
            self.scroll_x = max(0.0, (t_ms / 1000.0) * self.pps - (w - HEADER_W) + 120)
        self.redraw()

    def _on_hscroll(self, *args):
        if args[0] == "moveto":
            self.scroll_x = float(args[1]) * (self.app.reg.project.duration / 1000.0) * self.pps
        elif args[0] == "scroll":
            self.scroll_x += float(args[1]) * 40
        self.scroll_x = max(0.0, self.scroll_x)
        self.redraw()

    def _on_wheel(self, e):
        if getattr(e, "state", 0) & 0x0001:  # shift → horizontal scroll
            step = -getattr(e, "delta", 0) / 120.0 * 60.0
            if getattr(e, "num", None) == 4:
                step = -60.0
            elif getattr(e, "num", None) == 5:
                step = 60.0
            self.scroll_x = max(0.0, self.scroll_x + step)
            self.redraw()
            return
        delta = getattr(e, "delta", 0)
        if getattr(e, "num", None) == 4:
            delta = 120
        elif getattr(e, "num", None) == 5:
            delta = -120
        self.zoom(1.1 if delta > 0 else 0.9)

    def _on_ctrl_wheel(self, e):
        self.zoom(1.15 if getattr(e, "delta", 0) > 0 else 0.87)

    # ------------------------------------------------------------ drawing
    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        proj = self.app.reg.project
        w = max(1, cv.winfo_width())
        h = max(1, cv.winfo_height())

        cv.create_rectangle(0, 0, w, h, fill=C.bg, outline="")
        # ---- ruler
        cv.create_rectangle(0, 0, w, RULER_H, fill=C.panel, outline="")
        cv.create_line(0, RULER_H, w, RULER_H, fill=C.border)
        step = self._ruler_step()
        t = math.floor(self.x2t(HEADER_W) / step) * step
        while self.t2x(t) < w + 60:
            x = self.t2x(t)
            if x >= HEADER_W - 1:
                cv.create_line(x, RULER_H - 7, x, RULER_H, fill=C.dim)
                cv.create_text(x + 3, 8, text=self._label(t), anchor="w", fill=C.muted, font=mono(7))
            t += step
        cv.create_rectangle(0, 0, HEADER_W, RULER_H, fill=C.panel, outline="")
        cv.create_line(HEADER_W, 0, HEADER_W, h, fill=C.border)
        cv.create_text(10, RULER_H / 2, text="LAYERS  ▸  drag · trim · keys", anchor="w",
                       fill=C.dim, font=font(8, True))

        layers = list(reversed(proj.layers))  # top layer first
        y = RULER_H + 4
        for idx, layer in enumerate(layers):
            sel = layer.get("id") == self.app.sel
            row_bg = C.track_alt if idx % 2 else C.track
            if sel:
                row_bg = C.sel
            cv.create_rectangle(0, y, w, y + TRACK_H, fill=row_bg, outline="")
            self._draw_header(layer, y, sel)
            self._draw_bar(layer, y, w)
            y += TRACK_H + GAP
        # markers
        for m in proj.data.get("markers", []):
            x = self.t2x(m["t"])
            if HEADER_W < x < w:
                cv.create_polygon(x - 5, RULER_H - 2, x + 5, RULER_H - 2, x, RULER_H + 6,
                                  fill=C.accent, outline="")
        # playhead
        px = self.t2x(self.app.t)
        if px >= HEADER_W:
            cv.create_line(px, 0, px, h, fill=C.playhead, width=1)
            cv.create_polygon(px - 5, 0, px + 5, 0, px, 9, fill=C.playhead, outline="")
        # empty state
        if not proj.layers:
            cv.create_text(w / 2, h / 2, text="Import media and add a layer to begin",
                           fill=C.dim, font=font(10))
        self.zoom_label.configure(text=f"{int(self.pps)} px/s · {len(proj.layers)} layers")

    def _ruler_step(self) -> float:
        target_px = 80
        for step in (100, 250, 500, 1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000):
            if step / 1000.0 * self.pps >= target_px:
                return step
        return 600000

    @staticmethod
    def _label(t_ms: float) -> str:
        s = t_ms / 1000.0
        if s < 60:
            return f"{s:.1f}s" if s != int(s) else f"{int(s)}s"
        m, sec = divmod(int(s), 60)
        return f"{m}:{sec:02d}"

    def _draw_header(self, layer: dict, y: int, sel: bool):
        cv = self.canvas
        kind = layer.get("kind", "?")
        color = {"video": C.accent2, "image": C.good, "text": C.warn, "shape": C.accent,
                 "solid": C.muted, "adjustment": "#c792ea", "audio": "#3ddc97", "null": C.dim}.get(kind, C.muted)
        cv.create_rectangle(4, y + 5, 8, y + TRACK_H - 5, fill=color, outline="")
        name = str(layer.get("name", "Layer"))
        if len(name) > 20:
            name = name[:19] + "…"
        cv.create_text(16, y + TRACK_H / 2, text=name, anchor="w",
                       fill=C.text if sel else "#cdd3df", font=font(9, sel))
        cv.create_text(HEADER_W - 52, y + TRACK_H / 2, text=kind[:4].upper(), anchor="e",
                       fill=C.dim, font=mono(7))
        # visibility dot / lock
        eye = "◉" if not layer.get("hidden") else "◌"
        cv.create_text(HEADER_W - 32, y + TRACK_H / 2, text=eye, anchor="e",
                       fill=C.good if not layer.get("hidden") else C.dim, font=font(10),
                       tags=("eye", str(layer["id"])))
        cv.create_text(HEADER_W - 14, y + TRACK_H / 2, text="🔒" if layer.get("locked") else "·",
                       anchor="e", fill=C.warn if layer.get("locked") else C.dim, font=font(8),
                       tags=("lock", str(layer["id"])))

    def _draw_bar(self, layer: dict, y: int, w: int):
        cv = self.canvas
        x0 = max(HEADER_W, self.t2x(layer.get("start", 0)))
        x1 = min(w, self.t2x(layer.get("end", 0)))
        if x1 <= HEADER_W or x0 >= w:
            return
        kind = layer.get("kind", "solid")
        fill = {"video": "#2a4a7a", "image": "#245c48", "text": "#5c4a1f", "shape": "#5c3a1f",
                "solid": "#333a47", "adjustment": "#452f5c", "audio": "#1f4f3c", "null": "#2b303a"}.get(kind, "#333a47")
        edge = {"video": C.accent2, "image": C.good, "text": C.warn, "shape": C.accent,
                "solid": C.muted, "adjustment": "#c792ea", "audio": "#3ddc97", "null": C.dim}.get(kind, C.muted)
        if layer.get("parent"):
            cv.create_text(x0 + 6, y + 8, text="⛓", anchor="w", fill=C.muted, font=font(8))
        cv.create_rectangle(x0, y + 4, x1, y + TRACK_H - 4, fill=fill, outline=edge, width=1)
        if layer.get("matte") is not None:
            cv.create_text(x1 - 8, y + 8, text="◐", anchor="e", fill=C.warn, font=font(8))
        # keyframes
        for prop, keys in (layer.get("keys") or {}).items():
            col = PROP_COLORS.get(prop, C.kf)
            for k in keys:
                kx = self.t2x(k["t"])
                if HEADER_W < kx < w:
                    cy = y + TRACK_H - 9
                    cv.create_polygon(kx, cy - 4, kx + 4, cy, kx, cy + 4, kx - 4, cy,
                                      fill=col, outline="#0c0e12")
        for eff in layer.get("effects", []):
            for prop, keys in (eff.get("keys") or {}).items():
                for k in keys:
                    kx = self.t2x(k["t"])
                    if HEADER_W < kx < w:
                        cy = y + TRACK_H - 9
                        cv.create_polygon(kx, cy - 3, kx + 3, cy, kx, cy + 3, kx - 3, cy,
                                          fill="#8a8fa3", outline="")

    # ------------------------------------------------------------ interaction
    def _layer_at(self, y: float) -> Optional[dict]:
        layers = list(reversed(self.app.reg.project.layers))
        idx = int((y - RULER_H - 4) // (TRACK_H + GAP))
        if 0 <= idx < len(layers):
            return layers[idx]
        return None

    def _on_press(self, e):
        self.canvas.focus_set()
        proj = self.app.reg.project
        if e.y <= RULER_H:
            self.app.set_time(self.x2t(e.x))
            self._drag = {"mode": "scrub"}
            return
        layer = self._layer_at(e.y)
        if layer is None:
            self.app.select_layer(None)
            return
        # header eye/lock zones
        if e.x < HEADER_W:
            if e.x > HEADER_W - 40:
                layer["hidden"] = not layer.get("hidden")
                self.app.on_project_changed()
                return
            if e.x > HEADER_W - 20:
                layer["locked"] = not layer.get("locked")
                self.app.on_project_changed()
                return
            self.app.select_layer(layer["id"])
            return
        x0 = self.t2x(layer.get("start", 0))
        x1 = self.t2x(layer.get("end", 0))
        # keyframe hit?
        for prop, keys in (layer.get("keys") or {}).items():
            for k in keys:
                if abs(self.t2x(k["t"]) - e.x) <= 6:
                    self.app.select_layer(layer["id"])
                    self.app.set_time(k["t"])
                    return
        self.app.select_layer(layer["id"])
        if layer.get("locked"):
            return
        self.app.reg.push_undo()
        if abs(e.x - x0) <= 6:
            self._drag = {"mode": "trim_l", "layer": layer, "orig": (int(layer["start"]), int(layer["end"]))}
        elif abs(e.x - x1) <= 6:
            self._drag = {"mode": "trim_r", "layer": layer, "orig": (int(layer["start"]), int(layer["end"]))}
        else:
            self._drag = {"mode": "move", "layer": layer, "x0": e.x,
                          "orig": (int(layer["start"]), int(layer["end"]))}

    def _on_drag(self, e):
        if not self._drag:
            return
        mode = self._drag["mode"]
        if mode == "scrub":
            self.app.set_time(self.x2t(e.x))
            return
        layer = self._drag.get("layer")
        if not layer:
            return
        t = self.x2t(e.x)
        if mode == "trim_l":
            layer["start"] = int(max(0, min(t, layer["end"] - 100)))
        elif mode == "trim_r":
            layer["end"] = int(max(layer["start"] + 100, min(t, proj.duration + 60000)))
        elif mode == "move":
            dt = (e.x - self._drag["x0"]) / self.pps * 1000.0
            s0, e0 = self._drag["orig"]
            dur = e0 - s0
            new_s = int(max(0, s0 + dt))
            layer["start"] = new_s
            layer["end"] = new_s + dur
        self.redraw()

    def _on_release(self, _e):
        if self._drag and self._drag["mode"] != "scrub":
            nudge = 33
            layer = self._drag.get("layer")
            if layer:
                layer["start"] = int(round(layer["start"] / nudge) * nudge)
                layer["end"] = int(round(layer["end"] / nudge) * nudge)
            self.app.on_project_changed()
        self._drag = None

    def _on_double(self, e):
        layer = self._layer_at(e.y)
        if layer is not None:
            self.app.select_layer(layer["id"])
            self.app.inspector.focus_name()

    def _on_right(self, e):
        layer = self._layer_at(e.y)
        if layer is None:
            return
        self.app.select_layer(layer["id"])
        m = tk.Menu(self, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2,
                    activeforeground="#fff", bd=0, font=font(9))
        m.add_command(label="Split at playhead  (S)", command=lambda: self.app.split_selected())
        m.add_command(label="Duplicate  (Ctrl+D)", command=lambda: self.app.duplicate_selected())
        m.add_command(label="Delete", command=lambda: self.app.delete_selected())
        m.add_separator()
        m.add_command(label="Add effect…", command=lambda: self.app.open_effect_dialog())
        m.add_checkbutton(label="Hidden", command=lambda: self._toggle(layer, "hidden"))
        m.add_checkbutton(label="Locked", command=lambda: self._toggle(layer, "locked"))
        m.add_separator()
        m.add_command(label="Move start to playhead", command=lambda: self._move_to_playhead(layer))
        m.add_command(label="Rename…", command=lambda: self.app.inspector.focus_name())
        try:
            m.tk_popup(e.x_root, e.y_root)
        finally:
            m.grab_release()

    def _toggle(self, layer: dict, key: str):
        self.app.reg.push_undo()
        layer[key] = not layer.get(key)
        self.app.on_project_changed()

    def _move_to_playhead(self, layer: dict):
        self.app.reg.push_undo()
        dur = layer["end"] - layer["start"]
        layer["start"] = int(self.app.t)
        layer["end"] = int(self.app.t + dur)
        self.app.on_project_changed()
