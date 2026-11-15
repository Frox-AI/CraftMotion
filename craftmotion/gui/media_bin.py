"""Media bin: thumbnails, import, add-to-timeline, relink."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from typing import Dict, Optional

from PIL import ImageTk

from ..core.media import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT
from .theme import C, ScrollFrame, font, icon_button, label, mono


class MediaPanel(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=C.panel)
        self.app = app
        self._thumbs: Dict[int, ImageTk.PhotoImage] = {}
        self.selected: Optional[int] = None

        head = tk.Frame(self, bg=C.panel, height=28)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="MEDIA", bg=C.panel, fg=C.dim, font=font(8, True)).pack(side="left", padx=10)
        icon_button(head, "＋ Import", self.import_files, "Import video / image / audio",
                    bg=C.accent, fg="#141414", size=8).pack(side="right", padx=4, pady=3)

        self.scroll = ScrollFrame(self, bg=C.panel)
        self.scroll.pack(fill="both", expand=True)
        self.list = self.scroll.inner
        self.empty = label(self.list, "No media yet.\n\nImport files to start\nbuilding your edit.",
                           C.panel, C.dim, 9, justify="center")
        self.empty.pack(pady=30)
        self.refresh()

    # ------------------------------------------------------------ actions
    def import_files(self):
        exts = sorted(VIDEO_EXT | IMAGE_EXT | AUDIO_EXT)
        paths = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(), title="Import media",
            filetypes=[("Media", " ".join("*" + e for e in exts)),
                       ("Video", " ".join("*" + e for e in sorted(VIDEO_EXT))),
                       ("Images", " ".join("*" + e for e in sorted(IMAGE_EXT))),
                       ("Audio", " ".join("*" + e for e in sorted(AUDIO_EXT))),
                       ("All files", "*.*")])
        if not paths:
            return
        self.app.run_command("media.import", {"path": paths[0]})
        for p in paths[1:]:
            self.app.run_command("media.import", {"path": p}, quiet=True, rebuild=False)
        self.app.on_project_changed()

    def _add_to_timeline(self, item: dict, kind: Optional[str] = None):
        proj = self.app.reg.project
        kind = kind or ("audio" if item.get("kind") == "audio" else
                        "video" if item.get("kind") == "video" else "image")
        end = proj.duration
        if item.get("duration_ms"):
            end = min(proj.duration, int(item["duration_ms"])) if self.app.t == 0 else \
                int(self.app.t) + int(item["duration_ms"])
        self.app.run_command("layer.add", {
            "kind": kind, "media_id": item["id"], "start": int(self.app.t),
            "end": max(int(self.app.t) + 500, end),
        })

    def _relink(self, item: dict):
        path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Relink media",
                                          filetypes=[("Media", "*.*")])
        if path:
            self.app.run_command("media.relink", {"media_id": item["id"], "path": path})
            self.app.on_project_changed()

    # ------------------------------------------------------------ ui
    def refresh(self):
        for child in self.list.winfo_children():
            if child is not self.empty:
                child.destroy()
        self._thumbs.clear()
        media = self.app.reg.project.media
        if not media:
            self.empty.pack(pady=30)
            return
        self.empty.pack_forget()
        for item in media:
            self._row(item)

    def _row(self, item: dict):
        sel = item["id"] == self.selected
        row = tk.Frame(self.list, bg=C.sel if sel else C.panel, pady=3)
        row.pack(fill="x", padx=4, pady=1)
        thumb = self.app.reg.media.thumbnail(item, 64)
        holder = tk.Label(row, bg=C.track, width=64, height=36)
        holder.pack(side="left", padx=(2, 6))
        if thumb is not None:
            try:
                photo = ImageTk.PhotoImage(thumb.resize((64, max(20, int(64 * thumb.height / max(1, thumb.width))))))
                self._thumbs[item["id"]] = photo
                holder.configure(image=photo, width=photo.width(), height=photo.height())
            except Exception:
                pass
        info = tk.Frame(row, bg=row["bg"])
        info.pack(side="left", fill="x", expand=True)
        name = str(item.get("name") or item.get("filename") or "media")
        if len(name) > 22:
            name = name[:21] + "…"
        tk.Label(info, text=name, bg=row["bg"], fg=C.text, font=font(9), anchor="w").pack(fill="x")
        meta = item.get("kind", "?").upper()
        if item.get("duration_ms"):
            meta += f" · {item['duration_ms']/1000:.1f}s"
        if item.get("width"):
            meta += f" · {item['width']}×{item['height']}"
        if item.get("missing"):
            meta += " · ⚠ MISSING"
        tk.Label(info, text=meta, bg=row["bg"], fg=C.warn if item.get("missing") else C.dim,
                 font=mono(7), anchor="w").pack(fill="x")

        def on_press(_e, it=item):
            self.selected = it["id"]
            self.refresh()
        row.bind("<Button-1>", on_press)
        for w in (holder, info):
            w.bind("<Button-1>", on_press)
        row.bind("<Double-Button-1>", lambda e, it=item: self._add_to_timeline(it))
        row.configure(cursor="hand2")
        tk.Button(row, text="＋", command=lambda it=item: self._add_to_timeline(it), bg=row["bg"],
                  fg=C.muted, font=font(11), bd=0, relief="flat", activebackground=C.panel2,
                  cursor="hand2", highlightthickness=0).pack(side="right", padx=2)
        if item.get("missing"):
            tk.Button(row, text="relink", command=lambda it=item: self._relink(it), bg=row["bg"],
                      fg=C.warn, font=font(7), bd=0, relief="flat", highlightthickness=0,
                      cursor="hand2").pack(side="right")
