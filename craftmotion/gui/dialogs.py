"""Modal dialogs: new project, effect picker, export, Alight Motion import."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

from ..core import effects as FX
from .theme import C, ScrollFrame, font, icon_button, label, mono


class BaseDialog(tk.Toplevel):
    def __init__(self, parent, title: str, width: int = 560, height: int = 520):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=C.panel)
        self.transient(parent)
        self.resizable(True, True)
        self.geometry(f"{width}x{height}")
        self.minsize(420, 320)
        self.result = None
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(50, self._center)

    def _center(self):
        try:
            self.update_idletasks()
            px = self.master.winfo_rootx() + self.master.winfo_width() // 2 - self.winfo_width() // 2
            py = self.master.winfo_rooty() + self.master.winfo_height() // 2 - self.winfo_height() // 2
            self.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

    def footer(self, parent, ok_text: str = "OK", on_ok: Optional[Callable] = None):
        bar = tk.Frame(parent, bg=C.panel)
        bar.pack(fill="x", side="bottom", pady=10, padx=12)
        cancel = icon_button(bar, "Cancel", self.destroy, bg=C.panel2)
        cancel.pack(side="right", padx=4)
        ok = icon_button(bar, ok_text, on_ok or (lambda: None), bg=C.accent, fg="#141414", size=9)
        ok.pack(side="right")
        return ok


# ------------------------------------------------------------------ new project

RES_PRESETS = [
    ("Reels / Shorts · 1080×1920", 1080, 1920),
    ("Square · 1080×1080", 1080, 1080),
    ("Landscape · 1920×1080", 1920, 1080),
    ("Portrait 4:5 · 1080×1350", 1080, 1350),
    ("720p · 1280×720", 1280, 720),
]


class NewProjectDialog(BaseDialog):
    def __init__(self, parent):
        super().__init__(parent, "New project", 460, 330)
        body = tk.Frame(self, bg=C.panel)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        label(body, "Title", C.panel, C.muted, 9).pack(anchor="w")
        self.title_var = tk.StringVar(value="Untitled")
        tk.Entry(body, textvariable=self.title_var, bg=C.panel2, fg=C.text, bd=0,
                 insertbackground=C.text, font=font(10)).pack(fill="x", ipady=4, pady=(2, 8))
        label(body, "Canvas", C.panel, C.muted, 9).pack(anchor="w")
        self.res_var = tk.StringVar(value=RES_PRESETS[0][0])
        ttk.Combobox(body, textvariable=self.res_var, values=[r[0] for r in RES_PRESETS],
                     state="readonly").pack(fill="x", pady=(2, 8))
        row = tk.Frame(body, bg=C.panel)
        row.pack(fill="x", pady=2)
        col = tk.Frame(row, bg=C.panel)
        col.pack(side="left", expand=True, fill="x")
        label(col, "FPS", C.panel, C.muted, 9).pack(anchor="w")
        self.fps_var = tk.StringVar(value="30")
        ttk.Combobox(col, textvariable=self.fps_var, values=["24", "25", "30", "50", "60"],
                     state="readonly", width=8).pack(anchor="w", pady=2)
        col2 = tk.Frame(row, bg=C.panel)
        col2.pack(side="left", expand=True, fill="x", padx=8)
        label(col2, "Duration (s)", C.panel, C.muted, 9).pack(anchor="w")
        self.dur_var = tk.StringVar(value="15")
        tk.Entry(col2, textvariable=self.dur_var, bg=C.panel2, fg=C.text, bd=0,
                 insertbackground=C.text, font=mono(9), width=8).pack(anchor="w", pady=2, ipady=3)
        col3 = tk.Frame(row, bg=C.panel)
        col3.pack(side="left", expand=True, fill="x")
        label(col3, "Background", C.panel, C.muted, 9).pack(anchor="w")
        self.bg_var = tk.StringVar(value="Black")
        ttk.Combobox(col3, textvariable=self.bg_var,
                     values=["Black", "White", "Light Grey", "Green", "Transparent"],
                     state="readonly", width=11).pack(anchor="w", pady=2)
        self.footer(body, "Create", self._ok)

    def _ok(self):
        w, h = next((r[1], r[2]) for r in RES_PRESETS if r[0] == self.res_var.get())
        bg = {"Black": "#FF000000", "White": "#FFFFFFFF", "Light Grey": "#FFD8D8D8",
              "Green": "#FF00FF00", "Transparent": "#00000000"}.get(self.bg_var.get(), "#FF000000")
        try:
            dur = int(float(self.dur_var.get()) * 1000)
        except ValueError:
            dur = 15000
        self.result = {"title": self.title_var.get() or "Untitled", "width": w, "height": h,
                       "fps": float(self.fps_var.get()), "duration_ms": max(500, dur), "bg": bg}
        self.destroy()


# ------------------------------------------------------------------ effect picker

class EffectPickerDialog(BaseDialog):
    def __init__(self, parent, on_pick: Callable[[str], None]):
        super().__init__(parent, "Add effect", 620, 560)
        self.on_pick = on_pick
        top = tk.Frame(self, bg=C.panel)
        top.pack(fill="x", padx=12, pady=(12, 4))
        label(top, "Search", C.panel, C.muted, 9).pack(side="left")
        self.q = tk.StringVar()
        entry = tk.Entry(top, textvariable=self.q, bg=C.panel2, fg=C.text, bd=0,
                         insertbackground=C.text, font=font(10))
        entry.pack(side="left", fill="x", expand=True, padx=6, ipady=3)
        entry.focus_set()
        self.q.trace_add("write", lambda *a: self._fill())

        main = tk.Frame(self, bg=C.panel)
        main.pack(fill="both", expand=True, padx=12)
        left = tk.Frame(main, bg=C.panel, width=130)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.cat_list = tk.Listbox(left, bg=C.panel2, fg=C.text, bd=0, font=font(9),
                                   selectbackground=C.accent2, highlightthickness=0,
                                   activestyle="none")
        self.cat_list.pack(fill="both", expand=True)
        self.cat_list.insert("end", "All")
        for cat in sorted(FX.categories()):
            self.cat_list.insert("end", cat)
        self.cat_list.selection_set(0)
        self.cat_list.bind("<<ListboxSelect>>", lambda e: self._fill())

        right = tk.Frame(main, bg=C.panel)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.scroll = ScrollFrame(right, bg=C.panel)
        self.scroll.pack(fill="both", expand=True)
        self.listbox = self.scroll.inner
        self.footer(self, "Add", self._add)
        self._items: List[dict] = []
        self._fill()

    def _cat(self) -> Optional[str]:
        sel = self.cat_list.curselection()
        if not sel or self.cat_list.get(sel[0]) == "All":
            return None
        return self.cat_list.get(sel[0])

    def _fill(self):
        for w in self.listbox.winfo_children():
            w.destroy()
        q = self.q.get().strip().lower()
        self._items = [e for e in FX.list_effects(self._cat())
                       if not q or q in e["label"].lower() or q in e["type"].lower()
                       or any(q in a for a in e.get("am_ids", []))]
        if not self._items:
            label(self.listbox, "No effects match", C.panel, C.dim, 9).pack(pady=20)
        for eff in self._items:
            row = tk.Frame(self.listbox, bg=C.panel2, cursor="hand2")
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text=f"{eff['label']}", bg=C.panel2, fg=C.text, font=font(9),
                           anchor="w", padx=8, pady=4, cursor="hand2")
            lbl.pack(fill="x")
            sub = tk.Label(row, text=f"{eff['category']} · {eff['type']} · {len(eff['params'])} params",
                           bg=C.panel2, fg=C.dim, font=mono(7), anchor="w", padx=8, cursor="hand2")
            sub.pack(fill="x")
            row.bind("<Button-1>", lambda e, t=eff["type"]: self._choose(t))
            lbl.bind("<Button-1>", lambda e, t=eff["type"]: self._choose(t))
            sub.bind("<Button-1>", lambda e, t=eff["type"]: self._choose(t))
        self._selected: Optional[str] = self._items[0]["type"] if self._items else None

    def _choose(self, etype: str):
        self._selected = etype
        self.on_pick(etype)
        self.destroy()

    def _add(self):
        if getattr(self, "_selected", None):
            self.on_pick(self._selected)
        self.destroy()


# ------------------------------------------------------------------ export

class ExportDialog(BaseDialog):
    def __init__(self, parent, app):
        super().__init__(parent, "Export", 520, 470)
        self.app = app
        proj = app.reg.project
        body = tk.Frame(self, bg=C.panel)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        label(body, "Format", C.panel, C.muted, 9).pack(anchor="w")
        self.fmt = tk.StringVar(value="MP4 (H.264)")
        ttk.Combobox(body, textvariable=self.fmt, state="readonly",
                     values=["MP4 (H.264)", "WebM (VP9)", "GIF", "PNG sequence", "Still PNG",
                             "Still JPG", "Audio only (WAV)"]).pack(fill="x", pady=(2, 8))

        row = tk.Frame(body, bg=C.panel)
        row.pack(fill="x", pady=2)
        c1 = tk.Frame(row, bg=C.panel)
        c1.pack(side="left", fill="x", expand=True)
        label(c1, "Resolution", C.panel, C.muted, 9).pack(anchor="w")
        self.size = tk.StringVar(value=f"Canvas · {proj.width}×{proj.height}")
        ttk.Combobox(c1, textvariable=self.size, state="readonly",
                     values=[f"Canvas · {proj.width}×{proj.height}", "1080p · 1920×1080",
                             "1080×1920", "720p · 1280×720", "480×854"]
                     ).pack(fill="x", pady=2)
        c2 = tk.Frame(row, bg=C.panel)
        c2.pack(side="left", fill="x", expand=True, padx=8)
        label(c2, "FPS", C.panel, C.muted, 9).pack(anchor="w")
        self.fps = tk.StringVar(value=f"{proj.fps:g}")
        ttk.Combobox(c2, textvariable=self.fps, values=["24", "25", "30", "50", "60"],
                     state="readonly").pack(fill="x", pady=2)
        c3 = tk.Frame(row, bg=C.panel)
        c3.pack(side="left", fill="x", expand=True)
        label(c3, "Quality", C.panel, C.muted, 9).pack(anchor="w")
        self.quality = tk.StringVar(value="good")
        ttk.Combobox(c3, textvariable=self.quality, values=["draft", "good", "high"],
                     state="readonly").pack(fill="x", pady=2)

        row2 = tk.Frame(body, bg=C.panel)
        row2.pack(fill="x", pady=(8, 2))
        c4 = tk.Frame(row2, bg=C.panel)
        c4.pack(side="left", fill="x", expand=True)
        label(c4, "Range", C.panel, C.muted, 9).pack(anchor="w")
        self.range = tk.StringVar(value="Whole timeline")
        ttk.Combobox(c4, textvariable=self.range, state="readonly",
                     values=["Whole timeline", "From playhead", "Custom"]).pack(fill="x", pady=2)
        c5 = tk.Frame(row2, bg=C.panel)
        c5.pack(side="left", fill="x", expand=True, padx=8)
        label(c5, "Start (ms)", C.panel, C.muted, 9).pack(anchor="w")
        self.start = tk.StringVar(value="0")
        tk.Entry(c5, textvariable=self.start, bg=C.panel2, fg=C.text, bd=0, font=mono(9),
                 insertbackground=C.text).pack(fill="x", pady=2, ipady=3)
        c6 = tk.Frame(row2, bg=C.panel)
        c6.pack(side="left", fill="x", expand=True)
        label(c6, "End (ms)", C.panel, C.muted, 9).pack(anchor="w")
        self.end = tk.StringVar(value=str(proj.duration))
        tk.Entry(c6, textvariable=self.end, bg=C.panel2, fg=C.text, bd=0, font=mono(9),
                 insertbackground=C.text).pack(fill="x", pady=2, ipady=3)

        self.audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="Include audio", variable=self.audio_var, bg=C.panel, fg=C.muted,
                       selectcolor=C.panel2, activebackground=C.panel, font=font(9), bd=0
                       ).pack(anchor="w", pady=(6, 0))
        label(body, "Destination", C.panel, C.muted, 9).pack(anchor="w", pady=(8, 0))
        prow = tk.Frame(body, bg=C.panel)
        prow.pack(fill="x", pady=2)
        self.path = tk.StringVar(value="")
        tk.Entry(prow, textvariable=self.path, bg=C.panel2, fg=C.text, bd=0, font=mono(8),
                 insertbackground=C.text).pack(side="left", fill="x", expand=True, ipady=3)
        icon_button(prow, "Browse…", self._browse, bg=C.panel2, size=8).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(10, 2))
        self.status = label(body, "", C.panel, C.dim, 8)
        self.status.pack(anchor="w")
        self._ok_btn = self.footer(body, "Export", self._export)
        app.reg.listeners.append(self._on_event)

    def _browse(self):
        fmt = self.fmt.get()
        if fmt.startswith("MP4"):
            types = [("MP4 video", "*.mp4")], "mp4"
        elif fmt.startswith("WebM"):
            types = [("WebM video", "*.webm")], "webm"
        elif fmt == "GIF":
            types = [("GIF", "*.gif")], "gif"
        elif fmt == "PNG sequence":
            d = filedialog.askdirectory(parent=self, title="Output folder")
            if d:
                self.path.set(d)
            return
        elif fmt == "Still PNG":
            types = [("PNG", "*.png")], "png"
        elif fmt == "Still JPG":
            types = [("JPEG", "*.jpg")], "jpg"
        else:
            types = [("WAV", "*.wav")], "wav"
        p = filedialog.asksaveasfilename(parent=self, defaultextension="." + types[1],
                                         filetypes=types, initialfile="craftmotion_export." + types[1])
        if p:
            self.path.set(p)

    def _dims(self):
        s = self.size.get()
        try:
            part = s.split("·")[-1].strip()
            w, h = part.split("×")
            return int(w), int(h)
        except Exception:
            proj = self.app.reg.project
            return proj.width, proj.height

    def _export(self):
        app = self.app
        path = self.path.get().strip()
        if not path:
            self._browse()
            path = self.path.get().strip()
            if not path:
                return
        w, h = self._dims()
        try:
            fps = float(self.fps.get())
        except ValueError:
            fps = app.reg.project.fps
        rng = self.range.get()
        if rng == "From playhead":
            start, end = int(app.t), app.reg.project.duration
        elif rng == "Custom":
            try:
                start = int(float(self.start.get()))
                end = int(float(self.end.get()))
            except ValueError:
                start, end = 0, app.reg.project.duration
        else:
            start, end = 0, app.reg.project.duration
        fmt = self.fmt.get()
        self._ok_btn.configure(state="disabled")
        try:
            if fmt.startswith("MP4") or fmt.startswith("WebM"):
                if fmt.startswith("WebM") and not path.lower().endswith(".webm"):
                    path += ".webm"
                app.reg.execute("render.video", {
                    "path": path, "width": w, "height": h, "fps": fps, "start_ms": start,
                    "end_ms": end, "quality": self.quality.get(), "audio": self.audio_var.get()})
                self.status.configure(text="Rendering in background — watch the status bar.")
            elif fmt == "GIF":
                app.reg.execute("render.gif", {"path": path, "start_ms": start, "end_ms": end,
                                               "fps": min(20, fps), "width": min(640, w)})
                self.status.configure(text="Rendering GIF…")
            elif fmt == "PNG sequence":
                app.reg.execute("render.sequence", {"dir": path, "start_ms": start, "end_ms": end,
                                                    "fps": fps, "width": w, "height": h, "format": "png"})
                self.status.configure(text="Rendering sequence…")
            elif fmt in ("Still PNG", "Still JPG"):
                app.reg.execute("render.still", {"t": int(app.t), "path": path, "width": w, "height": h})
                self.status.configure(text=f"Saved still → {path}")
            elif fmt.startswith("Audio"):
                app.reg.execute("render.audio", {"path": path, "start_ms": start, "end_ms": end})
                self.status.configure(text="Saved audio mixdown")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
            self._ok_btn.configure(state="normal")

    def _on_event(self, event: dict):
        if event.get("type") != "job":
            return
        job = event["job"]
        try:
            self.progress["value"] = job["progress"] * 100
            self.status.configure(text=f"{job['label']}: {job['message']}")
            if job["status"] in ("done", "error", "cancelled"):
                self._ok_btn.configure(state="normal")
        except tk.TclError:
            try:
                self.app.reg.listeners.remove(self._on_event)
            except ValueError:
                pass


# ------------------------------------------------------------------ AM import

class AMImportDialog(BaseDialog):
    def __init__(self, parent, default_path: Optional[str] = None):
        super().__init__(parent, "Import Alight Motion XML", 480, 300)
        body = tk.Frame(self, bg=C.panel)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        label(body, "Preset / project XML", C.panel, C.muted, 9).pack(anchor="w")
        row = tk.Frame(body, bg=C.panel)
        row.pack(fill="x", pady=(2, 8))
        self.path = tk.StringVar(value=default_path or "")
        tk.Entry(row, textvariable=self.path, bg=C.panel2, fg=C.text, bd=0, font=mono(8),
                 insertbackground=C.text).pack(side="left", fill="x", expand=True, ipady=3)
        icon_button(row, "Choose…", self._browse, bg=C.panel2, size=8).pack(side="left", padx=4)

        self.merge_var = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="Merge into the current timeline (preset mode)",
                       variable=self.merge_var, bg=C.panel, fg=C.text, selectcolor=C.panel2,
                       activebackground=C.panel, font=font(9), bd=0).pack(anchor="w")
        row2 = tk.Frame(body, bg=C.panel)
        row2.pack(fill="x", pady=(8, 0))
        label(row2, "Insert at (ms)", C.panel, C.muted, 9).pack(side="left")
        self.offset = tk.StringVar(value="0")
        tk.Entry(row2, textvariable=self.offset, width=9, bg=C.panel2, fg=C.text, bd=0,
                 font=mono(9), insertbackground=C.text).pack(side="left", padx=6, ipady=3)
        label(body, "Media (images/video) referenced by the preset is looked up next to the XML.\n"
                    "Missing files are listed after import so you can relink them.",
              C.panel, C.dim, 8, justify="left", wraplength=430).pack(anchor="w", pady=(10, 0))
        self.footer(body, "Import", self._ok)

    def _browse(self):
        p = filedialog.askopenfilename(parent=self, title="Alight Motion XML",
                                       filetypes=[("Alight Motion XML", "*.xml"), ("All files", "*.*")])
        if p:
            self.path.set(p)

    def _ok(self):
        p = self.path.get().strip()
        if not p or not os.path.exists(p):
            messagebox.showwarning("Import", "Choose a valid XML file first.", parent=self)
            return
        try:
            offset = int(float(self.offset.get() or 0))
        except ValueError:
            offset = 0
        self.result = {"path": p, "merge": self.merge_var.get(), "offset_ms": offset}
        self.destroy()


# ------------------------------------------------------------------ about

def show_about(parent, version: str):
    messagebox.showinfo(
        "CraftMotion Studio",
        f"CraftMotion Studio {version}\n\n"
        "Motion graphics + video editing + Alight Motion XML, in one app.\n\n"
        "• Human mode: full GUI editing\n"
        "• Agent mode: MCP server (stdio or live HTTP bridge) with the same 60+ tools\n\n"
        "Inspired by the open-source EffectCraft and FilmCraft projects (MIT/Apache-2.0).\n"
        "Imports and exports Alight Motion scene XML, the documented format used for\n"
        "presets and projects. Alight Motion is a product of Alight Creative, Inc.;\n"
        "this app is not affiliated with or endorsed by them.",
        parent=parent)
