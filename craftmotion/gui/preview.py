"""Program monitor: threaded preview rendering, transport, on-canvas manipulation."""
from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from typing import Optional, Tuple

from PIL import Image, ImageTk

from ..core.render import Renderer
from ..core.util import fmt_time
from .theme import C, font, icon_button, mono


class PreviewPanel(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=C.bg)
        self.app = app
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=4)
        self._busy = False
        self._gen = 0
        self._last_req: Optional[tuple] = None
        self._img_size: Tuple[int, int] = (1, 1)
        self._offset = (0, 0)
        self._drag: Optional[dict] = None
        self._renderer: Optional[Renderer] = None
        self._loop = tk.BooleanVar(value=True)

        # ---- top bar
        top = tk.Frame(self, bg=C.panel, height=28)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="PROGRAM", bg=C.panel, fg=C.dim, font=font(8, True)).pack(side="left", padx=10)
        self.res_label = tk.Label(top, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.res_label.pack(side="right", padx=8)
        self.fps_label = tk.Label(top, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.fps_label.pack(side="right", padx=8)

        # ---- canvas
        self.canvas = tk.Canvas(self, bg="#0a0b0e", highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.request_render(force=True))
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<MouseWheel>", lambda e: self.scrub_relative(-1 if getattr(e, "delta", 0) > 0 else 1))

        # ---- transport
        tr = tk.Frame(self, bg=C.panel, height=40)
        tr.pack(fill="x")
        tr.pack_propagate(False)
        icon_button(tr, "⏮", self.go_start, "Go to start (Home)").pack(side="left", padx=(10, 2), pady=8)
        icon_button(tr, "◀", lambda: self.scrub_relative(-1), "Previous frame (←)").pack(side="left", pady=8)
        self.play_btn = icon_button(tr, "▶", self.toggle_play, "Play / pause (Space)",
                                    bg=C.accent, fg="#141414")
        self.play_btn.configure(font=font(11, True), padx=12)
        self.play_btn.pack(side="left", padx=2, pady=8)
        icon_button(tr, "▶|", lambda: self.scrub_relative(1), "Next frame (→)").pack(side="left", pady=8)
        icon_button(tr, "⏭", self.go_end, "Go to end (End)").pack(side="left", padx=2, pady=8)

        self.time_var = tk.StringVar(value="00:00.00")
        entry = tk.Entry(tr, textvariable=self.time_var, width=9, bg=C.panel2, fg=C.text,
                         font=mono(10), bd=0, insertbackground=C.text, justify="center")
        entry.pack(side="left", padx=10, pady=8, ipady=3)
        entry.bind("<Return>", self._commit_time)
        self.dur_label = tk.Label(tr, text="", bg=C.panel, fg=C.dim, font=mono(9))
        self.dur_label.pack(side="left")
        tk.Checkbutton(tr, text="Loop", variable=self._loop, bg=C.panel, fg=C.muted,
                       selectcolor=C.panel2, activebackground=C.panel, activeforeground=C.text,
                       font=font(9), bd=0, highlightthickness=0).pack(side="right", padx=10)
        self.status = tk.Label(tr, text="", bg=C.panel, fg=C.dim, font=font(8))
        self.status.pack(side="right", padx=6)

        self.after(40, self._poll)

    # ------------------------------------------------------------ render
    def view_size(self) -> Tuple[int, int]:
        """Render size that fits the canvas while matching project aspect."""
        proj = self.app.reg.project
        cw = max(64, self.canvas.winfo_width())
        ch = max(64, self.canvas.winfo_height())
        ar = proj.width / max(1, proj.height)
        w = cw
        h = int(w / ar)
        if h > ch:
            h = ch
            w = int(h * ar)
        return max(16, w - 4), max(16, h - 4)

    def request_render(self, force: bool = False):
        proj = self.app.reg.project
        self.dur_label.configure(text="/ " + fmt_time(proj.duration, proj.fps))
        self.res_label.configure(text=f"{proj.width}×{proj.height} @ {proj.fps:g}fps")
        size = self.view_size()
        key = (int(self.app.t), size)
        if not force and key == self._last_req:
            return
        self._last_req = key
        self._gen += 1
        gen = self._gen
        t = int(self.app.t)
        project = proj
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._q.put((gen, t, size, project))

    def _poll(self):
        if not self._busy:
            try:
                while True:
                    gen, t, size, project = self._q.get_nowait()
                    threading.Thread(target=self._render_worker, args=(gen, t, size, project),
                                     daemon=True).start()
                    break
            except queue.Empty:
                pass
        if not self._q.empty() or self._busy:
            self.status.configure(text="rendering…")
        else:
            self.status.configure(text="")
        self.after(40, self._poll)

    def _render_worker(self, gen: int, t: int, size: Tuple[int, int], project):
        self._busy = True
        t0 = time.time()
        try:
            r = Renderer(project, self.app.reg.media, mode="preview")
            img = r.render(t, size=size)
            r.close()
            if gen == self._gen:
                self.app.post(lambda: self._show(img, t, time.time() - t0))
        except Exception as exc:
            self.app.log(f"preview render failed: {exc}")
        finally:
            self._busy = False

    def _show(self, img: Image.Image, t: int, dt: float):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.delete("all")
        self._img_size = img.size
        try:
            self._photo = ImageTk.PhotoImage(img)
        except Exception:
            return
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        self._offset = (x, y)
        self.canvas.create_rectangle(x - 1, y - 1, x + img.width, y + img.height,
                                     outline=C.border, fill="#000000")
        self.canvas.create_image(x, y, image=self._photo, anchor="nw")
        self.fps_label.configure(text=f"{1.0/max(dt,1e-3):.0f} fps preview")
        self.time_var.set(fmt_time(t, self.app.reg.project.fps))
        # selection outline
        layer = self.app.selected_layer()
        if layer and layer.get("kind") not in ("audio", "null", "adjustment"):
            self._draw_selection(layer)

    def _draw_selection(self, layer: dict):
        try:
            q = self._img_size[0] / float(self.app.reg.project.width)
            pos = layer.get("transform", {}).get("pos", [0, 0])
            x = self._offset[0] + float(pos[0]) * q
            y = self._offset[1] + float(pos[1]) * q
            self.canvas.create_line(x - 7, y, x + 7, y, fill=C.accent, width=1)
            self.canvas.create_line(x, y - 7, x, y + 7, fill=C.accent, width=1)
            self.canvas.create_oval(x - 9, y - 9, x + 9, y + 9, outline=C.accent)
        except Exception:
            pass

    # ------------------------------------------------------------ transport
    def toggle_play(self):
        self.app.toggle_play()

    def set_playing_visual(self, playing: bool):
        self.play_btn.configure(text="⏸" if playing else "▶", bg=C.accent2 if playing else C.accent)

    def scrub_relative(self, frames: int):
        fps = self.app.reg.project.fps
        self.app.set_time(self.app.t + frames * 1000.0 / fps)

    def go_start(self):
        self.app.set_time(0)

    def go_end(self):
        self.app.set_time(self.app.reg.project.duration - 1000.0 / self.app.reg.project.fps)

    def _commit_time(self, _e=None):
        txt = self.time_var.get().strip()
        try:
            if ":" in txt:
                parts = txt.replace(";", ":").split(":")
                sec = 0.0
                for p in parts:
                    sec = sec * 60 + float(p)
                self.app.set_time(sec * 1000.0)
            else:
                self.app.set_time(float(txt) * 1000.0)
        except ValueError:
            self.request_render(force=True)

    # ------------------------------------------------------------ on-canvas drag
    def _press(self, e):
        layer = self.app.selected_layer()
        if layer is None or layer.get("kind") in ("audio", "null"):
            return
        self.app.reg.push_undo()
        tf = layer.setdefault("transform", {})
        pos = tf.get("pos", [0, 0, 0])
        self._drag = {"layer": layer, "start": (e.x, e.y),
                      "pos": (float(pos[0]), float(pos[1])),
                      "rot": float(tf.get("rot", [0])[0]),
                      "scale": list(tf.get("scale", [1, 1]))}

    def _drag_move(self, e):
        if not self._drag:
            return
        layer = self._drag["layer"]
        q = self.app.reg.project.width / max(1, self._img_size[0])
        dx = (e.x - self._drag["start"][0]) * q
        dy = (e.y - self._drag["start"][1]) * q
        tf = layer.setdefault("transform", {})
        if e.state & 0x0008:  # alt → rotate
            angle = math.degrees(math.atan2(dy, dx))
            tf["rot"] = [self._drag["rot"] + angle * 0.25]
        elif e.state & 0x0004:  # ctrl → scale
            f = max(0.05, 1.0 + (e.x - self._drag["start"][0]) / 200.0)
            tf["scale"] = [self._drag["scale"][0] * f, self._drag["scale"][1] * f]
        else:
            tf["pos"] = [self._drag["pos"][0] + dx, self._drag["pos"][1] + dy, 0.0]
        self.request_render()

    def _release(self, _e):
        if self._drag:
            self._drag = None
            self.app.on_project_changed()
