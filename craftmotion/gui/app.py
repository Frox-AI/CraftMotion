"""CraftMotion main window: menus, toolbar, panels, playback, dialogs."""
from __future__ import annotations

import os
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from ..core.registry import CommandError, Registry, __version__
from ..core.util import fmt_time
from .agent_panel import AgentPanel
from .dialogs import (AMImportDialog, EffectPickerDialog, ExportDialog, NewProjectDialog,
                      show_about)
from .advanced import (AudioPanel, CaptionsPanel, ColourPanel, EffectBrowserPanel,
                       GraphEditorPanel, TrackerPanel)
from .inspector import InspectorPanel
from .media_bin import MediaPanel
from .preview import PreviewPanel
from .theme import C, font, icon_button, label, mono, setup_styles
from .timeline import TimelinePanel

APP_NAME = "CraftMotion Studio"


def _log_path() -> str:
    base = os.environ.get("CRAFTMOTION_HOME") or os.path.join(os.path.expanduser("~"), ".craftmotion")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "gui.log")


class MainWindow(tk.Tk):
    def __init__(self, project=None):
        super().__init__()
        setup_styles(self)
        self.title(APP_NAME)
        self.configure(bg=C.bg)
        self.geometry("1440x900")
        self.minsize(1120, 680)
        self.reg = Registry(project) if project is not None else Registry()
        self.reg.on_change(self._on_registry_event)
        self.t: float = 0.0
        self.sel: Optional[int] = None
        self.playing = False
        self._play_start_wall = 0.0
        self._play_start_t = 0.0
        self._tick_count = 0
        self._title_dirty = False
        self.report_callback_exception = self._tk_error
        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._build_status()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self.refresh_all)
        from ..core.registry import COMMANDS
        self.log(f"Ready · {len(COMMANDS)} tools · human GUI + MCP agent mode")

    # ================================================================ layout
    def _build_menu(self):
        m = tk.Menu(self, bg=C.panel2, fg=C.text, activebackground=C.accent2, activeforeground="#fff",
                    bd=0, tearoff=0)
        fm = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        fm.add_command(label="New project…", command=self.new_project, accelerator="Ctrl+N")
        fm.add_command(label="Open…", command=self.open_project, accelerator="Ctrl+O")
        self._recent_menu = tk.Menu(fm, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        fm.add_cascade(label="Recent projects", menu=self._recent_menu)
        self._refresh_recent_menu()
        fm.add_command(label="Save", command=self.save_project, accelerator="Ctrl+S")
        fm.add_command(label="Save as…", command=lambda: self.save_project(True))
        fm.add_separator()
        fm.add_command(label="Import media…", command=lambda: self.media.import_files(),
                       accelerator="Ctrl+I")
        fm.add_command(label="Import Alight Motion XML…", command=self.import_am_xml,
                       accelerator="Ctrl+Shift+I")
        fm.add_command(label="Export Alight Motion XML…", command=self.export_am_xml)
        fm.add_separator()
        fm.add_command(label="Export video / images…", command=self.open_export, accelerator="Ctrl+E")
        fm.add_separator()
        fm.add_command(label="Quit", command=self._on_close)
        m.add_cascade(label="File", menu=fm)

        em = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        em.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        em.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Shift+Z")
        em.add_separator()
        em.add_command(label="Split at playhead", command=self.split_selected, accelerator="S")
        em.add_command(label="Duplicate layer", command=self.duplicate_selected, accelerator="Ctrl+D")
        em.add_command(label="Delete layer", command=self.delete_selected, accelerator="Del")
        em.add_separator()
        em.add_command(label="Add marker", command=self.add_marker, accelerator="M")
        m.add_cascade(label="Edit", menu=em)

        lm = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        for kind, name in (("text", "Text"), ("shape", "Shape"), ("solid", "Solid"),
                           ("adjustment", "Adjustment layer"), ("null", "Null / group")):
            lm.add_command(label=f"New {name} layer", command=lambda k=kind: self.add_layer(k))
        lm.add_separator()
        lm.add_command(label="Add effect…", command=self.open_effect_dialog, accelerator="A")
        lm.add_command(label="Add transition…", command=self.open_transition_dialog)
        lm.add_separator()
        lm.add_command(label="Group selected with null", command=self.group_selected)
        m.add_cascade(label="Layer", menu=lm)

        mm = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        mm.add_command(label="Alight Motion XML import…", command=self.import_am_xml)
        mm.add_command(label="Alight Motion XML export…", command=self.export_am_xml)
        mm.add_separator()
        mm.add_command(label="Validate XML file…", command=self.validate_am_xml)
        m.add_cascade(label="Alight Motion", menu=mm)

        am = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        am.add_command(label="Agent panel", command=lambda: self.right_tabs.select(self.agent))
        am.add_command(label="Start / stop live bridge", command=lambda: self.agent.toggle_bridge())
        am.add_command(label="Copy MCP config", command=lambda: self.agent._copy(self.agent._stdio_config()))
        m.add_cascade(label="Agents", menu=am)

        hm = tk.Menu(m, tearoff=0, bg=C.panel2, fg=C.text, activebackground=C.accent2)
        hm.add_command(label="Keyboard shortcuts", command=self.show_shortcuts)
        hm.add_command(label="About CraftMotion", command=lambda: show_about(self, __version__))
        m.add_cascade(label="Help", menu=hm)
        self.configure(menu=m)

    # ------------------------------------------------------------ recent files
    @staticmethod
    def _recent_path() -> str:
        base = os.environ.get("CRAFTMOTION_HOME") or os.path.join(os.path.expanduser("~"), ".craftmotion")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "recent.json")

    def _load_recent(self) -> list:
        try:
            import json
            with open(self._recent_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return [p for p in data if isinstance(p, str)]
        except Exception:
            return []

    def _remember_recent(self, path: str):
        import json
        items = [p for p in self._load_recent() if p != path]
        items.insert(0, path)
        try:
            with open(self._recent_path(), "w", encoding="utf-8") as f:
                json.dump(items[:10], f, indent=1)
        except Exception:
            pass
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        try:
            self._recent_menu.delete(0, "end")
        except Exception:
            pass
        items = self._load_recent()
        if not items:
            self._recent_menu.add_command(label="(none yet)", state="disabled")
            return
        for p in items:
            self._recent_menu.add_command(label=os.path.basename(p), command=lambda q=p: self.open_project_path(q))

    def _on_registry_event_persist(self, event):
        pass

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=C.panel, height=42)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        def sep():
            tk.Frame(bar, bg=C.border, width=1).pack(side="left", fill="y", padx=6, pady=8)
        b = lambda text, cmd, tip="", **kw: icon_button(bar, text, cmd, tip, **kw).pack(
            side="left", padx=1, pady=7)
        b("New", self.new_project, "New project (Ctrl+N)")
        b("Open", self.open_project, "Open project (Ctrl+O)")
        b("Save", self.save_project, "Save project (Ctrl+S)", bg=C.panel2)
        sep()
        b("↶", self.undo, "Undo (Ctrl+Z)")
        b("↷", self.redo, "Redo (Ctrl+Shift+Z)")
        sep()
        b("＋ Media", lambda: self.media.import_files(), "Import media (Ctrl+I)")
        b("＋ Text", lambda: self.add_layer("text"), "Add text layer")
        b("＋ Shape", lambda: self.add_layer("shape"), "Add shape layer")
        b("＋ Solid", lambda: self.add_layer("solid"), "Add solid")
        b("＋ Adjust", lambda: self.add_layer("adjustment"), "Add adjustment (grade/matte) layer")
        sep()
        b("AM XML ↓", self.import_am_xml, "Import Alight Motion preset/project XML",
          bg=C.accent2, fg="#fff")
        b("Export", self.open_export, "Export video / stills (Ctrl+E)", bg=C.accent, fg="#141414")
        self.agent_badge = tk.Label(bar, text="● agent-ready", bg=C.panel, fg=C.good, font=font(8, True))
        self.agent_badge.pack(side="right", padx=10)

    def _build_body(self):
        main = tk.PanedWindow(self, orient="horizontal", bg=C.bg, sashwidth=5, bd=0)
        main.pack(fill="both", expand=True)
        # left: media
        self.media = MediaPanel(main, self)
        main.add(self.media, minsize=200, width=250)
        # center: preview over timeline
        center = tk.PanedWindow(main, orient="vertical", bg=C.bg, sashwidth=5, bd=0)
        self.preview = PreviewPanel(center, self)
        center.add(self.preview, minsize=240, height=420)
        self.center_tabs = ttk.Notebook(center)
        self.timeline = TimelinePanel(self.center_tabs, self)
        self.graph = GraphEditorPanel(self.center_tabs, self)
        self.center_tabs.add(self.timeline, text="Timeline")
        self.center_tabs.add(self.graph, text="Graph editor")
        center.add(self.center_tabs, minsize=190, height=280)
        main.add(center, minsize=520)
        # right: inspector + agents
        self.right_tabs = ttk.Notebook(main, width=330)
        self.inspector = InspectorPanel(self.right_tabs, self)
        self.effects_browser = EffectBrowserPanel(self.right_tabs, self)
        self.colour = ColourPanel(self.right_tabs, self)
        self.audio_panel = AudioPanel(self.right_tabs, self)
        self.captions = CaptionsPanel(self.right_tabs, self)
        self.tracker = TrackerPanel(self.right_tabs, self)
        self.agent = AgentPanel(self.right_tabs, self)
        self.right_tabs.add(self.inspector, text="Inspector")
        self.right_tabs.add(self.effects_browser, text="Effects")
        self.right_tabs.add(self.colour, text="Colour")
        self.right_tabs.add(self.audio_panel, text="Audio")
        self.right_tabs.add(self.captions, text="Captions")
        self.right_tabs.add(self.tracker, text="Track")
        self.right_tabs.add(self.agent, text="AI Agents")
        main.add(self.right_tabs, minsize=300, width=340)
        self._main_pane = main

    def _build_status(self):
        bar = tk.Frame(self, bg=C.panel, height=24)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status = tk.Label(bar, text="", bg=C.panel, fg=C.muted, font=font(8), anchor="w")
        self.status.pack(side="left", padx=10)
        self.info_label = tk.Label(bar, text="", bg=C.panel, fg=C.dim, font=mono(8))
        self.info_label.pack(side="right", padx=10)

    def _bind_keys(self):
        self.bind("<space>", lambda e: self.toggle_play() if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<Control-n>", lambda e: self.new_project())
        self.bind("<Control-o>", lambda e: self.open_project())
        self.bind("<Control-s>", lambda e: self.save_project())
        self.bind("<Control-Shift-S>", lambda e: self.save_project(True))
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.bind("<Control-d>", lambda e: self.duplicate_selected())
        self.bind("<Control-i>", lambda e: self.media.import_files())
        self.bind("<Control-Shift-I>", lambda e: self.import_am_xml())
        self.bind("<Control-e>", lambda e: self.open_export())
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<BackSpace>", lambda e: None)
        self.bind("<s>", lambda e: self.split_selected() if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<a>", lambda e: self.open_effect_dialog() if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<m>", lambda e: self.add_marker())
        self.bind("<Home>", lambda e: self.set_time(0))
        self.bind("<End>", lambda e: self.set_time(self.reg.project.duration))
        self.bind("<Left>", lambda e: self.step(-1))
        self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Shift-Left>", lambda e: self.step(-10))
        self.bind("<Shift-Right>", lambda e: self.step(10))
        # pro transport: J/K/L with accelerating shuttle
        self.bind("<j>", lambda e: self.jkl("j") if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<k>", lambda e: self.jkl("k") if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<l>", lambda e: self.jkl("l") if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<o>", lambda e: self.open_lottie() if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<e>", lambda e: self.export_lottie() if not isinstance(e.widget, tk.Entry) else None)
        self.bind("<Control-Shift-T>", lambda e: self.center_tabs.select(1))
        self.bind("<Control-Shift-G>", lambda e: self.center_tabs.select(1))

    # ---- transport -------------------------------------------------
    def jkl(self, key: str):
        from ..core.editing import jkl_rate
        rate = jkl_rate(key, self.__dict__.get("shuttle_rate", 1.0))
        self.shuttle_rate = rate
        if key == "k" or abs(rate) < 1e-6:
            self.playing = False
            self.shuttle_rate = 1.0
            self.log("pause")
            return
        self.playing = True
        self.log(f"playback {rate:+.2f}x")
        if not getattr(self, "_play_running", False):
            self._play_running = True
            self._play_loop()

    def open_lottie(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Import Lottie JSON",
                                          filetypes=[("Lottie", "*.json"), ("All", "*.*")])
        if not path:
            return
        res = self.run_command("project.import_lottie", {"path": path, "replace": False})
        if res:
            self.log(f"Lottie imported · {res.get('layers')} layers")
            self.refresh_all()

    def export_lottie(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(title="Export Lottie JSON", defaultextension=".json",
                                            filetypes=[("Lottie", "*.json")])
        if path:
            self.run_command("project.export_lottie", {"path": path})
            self.log(f"Lottie exported → {os.path.basename(path)}")

    # ================================================================ state
    def selected_layer(self) -> Optional[dict]:
        if self.sel is None:
            return None
        return self.reg.project.get_layer(self.sel)

    def select_layer(self, layer_id: Optional[int]):
        self.sel = layer_id
        self.inspector.rebuild()
        self.timeline.redraw()
        self.request_preview_only()

    def set_time(self, t_ms: float):
        dur = self.reg.project.duration
        self.t = max(0.0, min(float(t_ms), dur - 1))
        self.preview.request_render()
        if self._tick_count % 2 == 0:
            self.timeline.redraw()
        self.inspector.refresh_values()
        self._update_info()

    def step(self, frames: int):
        self.set_time(self.t + frames * 1000.0 / self.reg.project.fps)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.shuttle_rate = 1.0
            self._play_start_wall = time.perf_counter()
            self._play_start_t = self.t
            self._play_loop()
            if hasattr(self.preview, "set_playing_visual"):
                self.preview.set_playing_visual(True)
        else:
            self.preview.set_playing_visual(False)

    def _play_loop(self):
        if not self.playing:
            self._play_running = False
            return
        rate = self.__dict__.get("shuttle_rate", 1.0) or 1.0
        elapsed = (time.perf_counter() - self._play_start_wall) * 1000.0 * rate
        t = self._play_start_t + elapsed
        dur = self.reg.project.duration
        if rate >= 0 and t >= dur:
            if self.preview._loop.get():
                self._play_start_wall = time.perf_counter()
                self._play_start_t = 0.0
                t = 0.0
            else:
                t = dur - 1
                self.playing = False
                self.shuttle_rate = 1.0
                self.preview.set_playing_visual(False)
        elif rate < 0 and t <= 0:
            if self.preview._loop.get():
                self._play_start_wall = time.perf_counter()
                self._play_start_t = dur
                t = dur - 1
            else:
                t = 0.0
                self.playing = False
                self.shuttle_rate = 1.0
                self.preview.set_playing_visual(False)
        self.t = t
        self.preview.request_render()
        self._tick_count += 1
        if self._tick_count % 2 == 0:
            self.timeline.redraw()
            self.inspector.refresh_values()
        self._update_info()
        if self.playing:
            self.after(33, self._play_loop)

    def _update_info(self):
        p = self.reg.project
        self.info_label.configure(
            text=f"{fmt_time(self.t, p.fps)} / {fmt_time(p.duration, p.fps)} · "
                 f"{p.width}×{p.height} @ {p.fps:g} · {len(p.layers)} layers · {len(p.media)} media")

    # ================================================================ refresh
    def refresh_panels(self):
        for attr in ("effects_browser", "audio_panel", "captions", "colour"):
            panel = getattr(self, attr, None)
            if panel is None:
                continue
            try:
                if attr in ("audio_panel", "captions"):
                    panel.build() if attr == "audio_panel" else panel.rebuild()
                elif attr == "colour":
                    panel.refresh_scope()
            except Exception:
                pass
        try:
            self.graph.redraw()
        except Exception:
            pass

    def refresh_all(self):
        self.timeline.redraw()
        self.media.refresh()
        self.inspector.rebuild()
        self.refresh_panels()
        self.request_preview_only()
        self._update_info()
        name = self.reg.project.data.get("title", "Untitled")
        star = " •" if self._title_dirty else ""
        self.title(f"{APP_NAME} — {name}{star}")

    def on_project_changed(self, rebuild: bool = True):
        self.timeline.redraw()
        if rebuild:
            self.inspector.rebuild()
        else:
            self.inspector.refresh_values()
        self.preview.request_render()
        self._update_info()
        self._title_dirty = True
        self.media.refresh()

    def request_preview_only(self):
        self.preview.request_render()

    def post(self, fn):
        def run():
            try:
                fn()
            except Exception:
                self.log(traceback.format_exc(limit=3))
        try:
            self.after(0, run)
        except Exception:
            pass

    def log(self, msg: str):
        try:
            self.status.configure(text=str(msg)[:160])
        except Exception:
            pass

    def _on_registry_event(self, event: dict):
        if event.get("type") == "job":
            job = event["job"]
            self.log(f"{job['label']}: {job['message']} ({job['progress']*100:.0f}%)")
            if job["status"] == "done" and job.get("result"):
                self.log(f"Done → {job['result'].get('path') or job['result'].get('dir')}")
            elif job["status"] == "error":
                self.log(f"Job failed: {job['error']}")

    # ================================================================ commands
    def run_command(self, name: str, args: Optional[dict] = None, quiet: bool = False,
                    rebuild: bool = True):
        try:
            result = self.reg.execute(name, args or {})
            if not quiet:
                self.log(f"{name} ok")
            if rebuild:
                self.on_project_changed()
            return result
        except CommandError as exc:
            self.log(f"{name}: {exc}")
            messagebox.showwarning("CraftMotion", str(exc), parent=self)
            return None
        except Exception as exc:
            self.log(f"{name} failed: {exc}")
            messagebox.showerror("CraftMotion", f"{name} failed:\n{exc}", parent=self)
            return None

    def undo(self):
        if self.reg.undo():
            self.refresh_all()
            self.log("Undo")
        else:
            self.log("Nothing to undo")

    def redo(self):
        if self.reg.redo():
            self.refresh_all()
            self.log("Redo")
        else:
            self.log("Nothing to redo")

    # ================================================================ file ops
    def new_project(self):
        dlg = NewProjectDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self.reg.execute("project.new", dlg.result)
            self.sel = None
            self.t = 0
            self._title_dirty = False
            self.refresh_all()
            self.log(f"New project · {dlg.result['width']}×{dlg.result['height']}")

    def open_project(self):
        path = filedialog.askopenfilename(parent=self, title="Open project",
                                          filetypes=[("CraftMotion project", "*.cmproj"),
                                                     ("All files", "*.*")])
        if path:
            self.open_project_path(path)

    def open_project_path(self, path: str):
        try:
            self.reg.execute("project.load", {"path": path})
            self.sel = None
            self.t = 0
            self._title_dirty = False
            self._remember_recent(path)
            self.refresh_all()
            self.log(f"Opened {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc), parent=self)

    def save_project(self, save_as: bool = False):
        path = self.reg.project.path
        if save_as or not path:
            path = filedialog.asksaveasfilename(parent=self, title="Save project",
                                                defaultextension=".cmproj",
                                                filetypes=[("CraftMotion project", "*.cmproj")],
                                                initialfile=f"{self.reg.project.data.get('title','untitled')}.cmproj")
            if not path:
                return
        try:
            self.reg.execute("project.save", {"path": path})
            self._title_dirty = False
            self._remember_recent(path)
            self.refresh_all()
            self.log(f"Saved → {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)

    def import_am_xml(self):
        dlg = AMImportDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        res = self.run_command("project.import_am_xml", dlg.result, quiet=True)
        if not res:
            return
        stats = res.get("stats", {})
        warns = res.get("warnings", [])
        self.sel = None
        self.refresh_all()
        msg = (f"Imported {stats.get('layers',0)} layers, {stats.get('effects',0)} effects, "
               f"{stats.get('keyframes',0)} keyframes.")
        if warns:
            shown = "\n• " + "\n• ".join(warns[:8])
            msg += "\n\nWarnings:" + shown
        messagebox.showinfo("Alight Motion import", msg, parent=self)

    def export_am_xml(self):
        path = filedialog.asksaveasfilename(parent=self, title="Export Alight Motion XML",
                                            defaultextension=".xml",
                                            filetypes=[("Alight Motion XML", "*.xml")],
                                            initialfile="craftmotion.xml")
        if not path:
            return
        res = self.run_command("project.export_am_xml", {"path": path}, quiet=True, rebuild=False)
        if res:
            warns = res.get("warnings", [])
            msg = f"Saved {path}"
            if warns:
                msg += "\n\nWarnings:\n• " + "\n• ".join(warns[:8])
            messagebox.showinfo("Alight Motion export", msg, parent=self)

    def validate_am_xml(self):
        path = filedialog.askopenfilename(parent=self, title="Validate Alight Motion XML",
                                          filetypes=[("XML", "*.xml")])
        if not path:
            return
        res = self.run_command("project.validate_am_xml", {"path": path}, quiet=True, rebuild=False)
        if res:
            v = res["validation"]
            messagebox.showinfo("Validation",
                                ("Valid ✓" if v["ok"] else "Problems found") +
                                f"\nElements: {v['elements']} · media: {v['media']}\n" +
                                "\n".join(v["issues"][:10]), parent=self)

    def open_export(self):
        ExportDialog(self, self)

    # ================================================================ layers
    def add_layer(self, kind: str):
        args = {"kind": kind, "start": int(self.t), "end": int(self.t + 5000)}
        if kind == "text":
            args.setdefault("name", "Text")
        res = self.run_command("layer.add", args)
        if res:
            self.select_layer(res["layer"]["id"])

    def add_marker(self):
        self.run_command("marker.add", {"t": int(self.t)}, quiet=True)

    def split_selected(self):
        if self.sel is None:
            return
        layer = self.selected_layer()
        if layer and int(layer["start"]) < int(self.t) < int(layer["end"]):
            self.run_command("layer.split", {"layer_id": self.sel, "t": int(self.t)}, quiet=True)

    def duplicate_selected(self):
        if self.sel is None:
            return
        res = self.run_command("layer.duplicate", {"layer_id": self.sel}, quiet=True)
        if res:
            self.select_layer(res["layer"]["id"])

    def delete_selected(self):
        if self.sel is None:
            return
        self.run_command("layer.delete", {"layer_id": self.sel}, quiet=True)
        self.select_layer(None)

    def group_selected(self):
        if self.sel is None:
            return
        res = self.run_command("layer.group", {"layer_ids": [self.sel], "name": "Group"}, quiet=True)
        if res:
            self.select_layer(res["group"]["id"])

    def open_effect_dialog(self):
        if self.sel is None:
            messagebox.showinfo("Effect", "Select a layer first.", parent=self)
            return

        def pick(etype: str):
            self.run_command("effect.add", {"layer_id": self.sel, "type": etype}, quiet=True)
        EffectPickerDialog(self, pick)

    def open_transition_dialog(self):
        if self.sel is None:
            messagebox.showinfo("Transition", "Select a layer first.", parent=self)
            return
        win = tk.Toplevel(self)
        win.title("Add transition")
        win.configure(bg=C.panel)
        win.transient(self)
        body = tk.Frame(win, bg=C.panel)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        label(body, "Kind", C.panel, C.muted, 9).grid(row=0, column=0, sticky="w")
        kind = tk.StringVar(value="fade")
        ttk.Combobox(body, textvariable=kind, values=["fade", "wipe", "iris", "dissolve", "slide",
                                                      "zoom_punch"], state="readonly", width=14
                     ).grid(row=0, column=1, pady=4)
        label(body, "Duration (ms)", C.panel, C.muted, 9).grid(row=1, column=0, sticky="w")
        dur = tk.StringVar(value="600")
        tk.Entry(body, textvariable=dur, bg=C.panel2, fg=C.text, bd=0, font=mono(9), width=10
                 ).grid(row=1, column=1, pady=4)

        def add():
            try:
                d = int(float(dur.get()))
            except ValueError:
                d = 600
            self.run_command("transition.add", {"layer_id": self.sel, "kind": kind.get(),
                                                "duration": d, "mode": "in"}, quiet=True)
            win.destroy()
        icon_button(body, "Add", add, bg=C.accent, fg="#141414").grid(row=2, column=1, sticky="e", pady=8)

    def save_preset_from_layer(self, layer: dict):
        from tkinter import simpledialog
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self)
        if name:
            self.reg.execute("preset.save", {"name": name, "layer_id": layer["id"]})
            self.log(f"Saved preset '{name}'")

    def replace_media(self, layer: dict):
        path = filedialog.askopenfilename(parent=self, title="Replace media")
        if not path:
            return
        res = self.run_command("media.import", {"path": path}, quiet=True, rebuild=False)
        if res:
            self.run_command("layer.update", {"layer_id": layer["id"],
                                              "props": {"media_id": res["media"]["id"]}},
                             quiet=True)

    # ================================================================ misc
    def show_shortcuts(self):
        messagebox.showinfo("Shortcuts",
                            "Space  Play / pause\nS  Split at playhead\nA  Add effect\n"
                            "Ctrl+D  Duplicate\nDel  Delete layer\n← →  Step frame\n"
                            "Shift+← →  10 frames\nHome / End  Jump\nM  Marker\n"
                            "Ctrl+I  Import media\nCtrl+Shift+I  Import AM XML\n"
                            "Ctrl+E  Export\nCtrl+Z / Ctrl+Shift+Z  Undo / redo", parent=self)

    def _tk_error(self, exc, val, tb):
        text = "".join(traceback.format_exception(exc, val, tb))
        try:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write("\n--- " + time.strftime("%Y-%m-%d %H:%M:%S") + " ---\n" + text)
        except Exception:
            pass
        self.log(f"Error: {val}")
        messagebox.showerror("CraftMotion error", f"{val}\n\nDetails written to {_log_path()}",
                             parent=self)

    def _on_close(self):
        if self.playing:
            self.playing = False
        try:
            self.agent.bridge.stop()
        except Exception:
            pass
        self.destroy()


def run(project_path: Optional[str] = None) -> int:
    project = None
    if not project_path:
        # fresh installs open the bundled demo so the window is never empty
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for cand in (os.path.join(here, "samples", "craftmotion_demo.cmproj"),
                     os.path.join(os.getcwd(), "samples", "craftmotion_demo.cmproj")):
            if os.path.exists(cand):
                project_path = cand
                break
    if project_path and os.path.exists(project_path):
        try:
            from ..core.model import Project
            project = Project.load(project_path)
        except Exception:
            project = None
    app = MainWindow(project)
    app.mainloop()
    return 0
