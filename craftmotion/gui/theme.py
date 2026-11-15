"""Dark UI palette, fonts and small reusable widgets."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class C:
    bg = "#12141a"
    panel = "#181b23"
    panel2 = "#1e222c"
    border = "#2a303d"
    border2 = "#374052"
    text = "#e8eaf0"
    muted = "#98a1b3"
    dim = "#5f6a7d"
    accent = "#ff7a1f"
    accent2 = "#3d8bfd"
    good = "#3ddc97"
    warn = "#ffcc55"
    bad = "#ff5d6c"
    track = "#20252f"
    track_alt = "#242a36"
    playhead = "#ff4d6d"
    kf = "#ffcc55"
    sel = "#2f3d57"


FONT = "Segoe UI"
FONT_MONO = "Consolas"


def font(size: int = 10, bold: bool = False) -> tuple:
    return (FONT, size, "bold") if bold else (FONT, size)


def mono(size: int = 9, bold: bool = False) -> tuple:
    return (FONT_MONO, size, "bold") if bold else (FONT_MONO, size)


def setup_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(".", background=C.bg, foreground=C.text, fieldbackground=C.panel2,
                    bordercolor=C.border, lightcolor=C.panel, darkcolor=C.panel,
                    troughcolor=C.track, font=font(10))
    style.configure("TFrame", background=C.panel)
    style.configure("Bg.TFrame", background=C.bg)
    style.configure("TLabel", background=C.panel, foreground=C.text)
    style.configure("Muted.TLabel", background=C.panel, foreground=C.muted, font=font(9))
    style.configure("Head.TLabel", background=C.panel, foreground=C.muted, font=font(9, True))
    style.configure("TButton", background=C.panel2, foreground=C.text, borderwidth=0,
                    focusthickness=0, padding=(8, 4), font=font(9))
    style.map("TButton", background=[("active", C.border2), ("pressed", C.border)],
              foreground=[("disabled", C.dim)])
    style.configure("Accent.TButton", background=C.accent, foreground="#141414", font=font(9, True))
    style.map("Accent.TButton", background=[("active", "#ff8f45"), ("pressed", "#e06a12")])
    style.configure("Ghost.TButton", background=C.panel, foreground=C.muted)
    style.map("Ghost.TButton", background=[("active", C.panel2)], foreground=[("active", C.text)])
    style.configure("TEntry", fieldbackground=C.panel2, foreground=C.text,
                    insertcolor=C.text, bordercolor=C.border, padding=3)
    style.map("TEntry", fieldbackground=[("focus", C.panel2)], bordercolor=[("focus", C.accent2)])
    style.configure("TCombobox", fieldbackground=C.panel2, background=C.panel2, foreground=C.text,
                    arrowcolor=C.muted, bordercolor=C.border, padding=3)
    style.map("TCombobox", fieldbackground=[("readonly", C.panel2)],
              foreground=[("readonly", C.text)])
    style.configure("TNotebook", background=C.bg, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=C.panel, foreground=C.muted, padding=(12, 5),
                    font=font(9))
    style.map("TNotebook.Tab", background=[("selected", C.panel2)],
              foreground=[("selected", C.text)])
    style.configure("TPanedwindow", background=C.bg)
    style.configure("Sash", sashthickness=6, background=C.bg)
    style.configure("Vertical.TScrollbar", background=C.panel2, troughcolor=C.bg,
                    arrowcolor=C.muted, bordercolor=C.bg)
    style.configure("TScale", background=C.panel, troughcolor=C.track)
    style.configure("TCheckbutton", background=C.panel, foreground=C.text, font=font(9))
    style.map("TCheckbutton", background=[("active", C.panel)])


class ScrollFrame(tk.Frame):
    """Vertically scrollable frame."""

    def __init__(self, master, bg: str = C.panel, **kw):
        super().__init__(master, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.inner)

    def _on_inner(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, widget):
        def on_wheel(e):
            delta = -1 if getattr(e, "delta", 0) > 0 else 1
            if getattr(e, "num", None) == 4:
                delta = -1
            elif getattr(e, "num", None) == 5:
                delta = 1
            self.canvas.yview_scroll(delta, "units")
        widget.bind("<MouseWheel>", on_wheel, add="+")
        widget.bind("<Button-4>", on_wheel, add="+")
        widget.bind("<Button-5>", on_wheel, add="+")


def section(parent, title: str, bg: str = C.panel) -> tk.Frame:
    """A titled block; returns the body frame to fill."""
    head = tk.Frame(parent, bg=bg)
    head.pack(fill="x", padx=10, pady=(10, 2))
    tk.Label(head, text=title.upper(), bg=bg, fg=C.dim, font=font(8, True)).pack(side="left")
    tk.Frame(head, bg=C.border, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))
    body = tk.Frame(parent, bg=bg)
    body.pack(fill="x", padx=10, pady=(0, 6))
    return body


def label(parent, text: str, bg: str = C.panel, fg: str = C.muted, size: int = 9,
          bold: bool = False, **kw) -> tk.Label:
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font(size, bold), **kw)


class KeyButton(tk.Canvas):
    """Diamond keyframe toggle."""

    def __init__(self, master, command: Optional[Callable] = None, bg: str = C.panel, size: int = 16):
        super().__init__(master, width=size, height=size, bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2")
        self._cmd = command
        self._size = size
        self._state = "none"  # none | keyed | onkey
        self._draw()
        self.bind("<Button-1>", self._click)

    def _draw(self):
        self.delete("all")
        s = self._size
        m = 3
        fill = {"none": C.track, "keyed": C.kf, "onkey": "#ffffff"}[self._state]
        outline = {"none": C.border2, "keyed": C.kf, "onkey": C.kf}[self._state]
        self.create_polygon(s / 2, m, s - m, s / 2, s / 2, s - m, m, s / 2,
                            fill=fill, outline=outline)

    def set_state(self, state: str):
        if state != self._state:
            self._state = state
            self._draw()

    def _click(self, _e):
        if self._cmd:
            self._cmd()


class Slider(tk.Frame):
    """Compact labelled slider with numeric readout."""

    def __init__(self, master, label_text: str, from_: float, to: float, value: float,
                 command: Optional[Callable[[float], None]] = None, fmt: str = "{:.2f}",
                 bg: str = C.panel, width: int = 10):
        super().__init__(master, bg=bg)
        self._fmt = fmt
        self._cmd = command
        self.var = tk.DoubleVar(value=value)
        top = tk.Frame(self, bg=bg)
        top.pack(fill="x")
        tk.Label(top, text=label_text, bg=bg, fg=C.muted, font=font(9)).pack(side="left")
        self.value_label = tk.Label(top, text=fmt.format(value), bg=bg, fg=C.text, font=mono(9))
        self.value_label.pack(side="right")
        self.scale = tk.Scale(self, from_=from_, to=to, orient="horizontal", showvalue=False,
                              variable=self.var, bg=bg, fg=C.text, troughcolor=C.track,
                              highlightthickness=0, bd=0, sliderrelief="flat",
                              activebackground=C.accent, command=self._on_change)
        self.scale.pack(fill="x")

    def _on_change(self, _val=None):
        v = float(self.var.get())
        self.value_label.configure(text=self._fmt.format(v))
        if self._cmd:
            self._cmd(v)

    def set(self, value: float):
        self.var.set(value)
        self.value_label.configure(text=self._fmt.format(value))


class Tooltip:
    def __init__(self, widget, text: str, delay: int = 450):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._id = self.widget.after(self.delay, self._show)

    def _show(self):
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except tk.TclError:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self._tip, bg=C.border2, bd=0)
        frame.pack()
        tk.Label(frame, text=self.text, bg="#0c0e12", fg=C.text, font=font(9),
                 padx=8, pady=4, justify="left").pack()

    def _hide(self, _e=None):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except tk.TclError:
                pass
            self._id = None
        if self._tip:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def hline(parent, bg: str = C.panel, pad: int = 10) -> tk.Frame:
    f = tk.Frame(parent, bg=bg)
    f.pack(fill="x", padx=pad, pady=3)
    tk.Frame(f, bg=C.border, height=1).pack(fill="x")
    return f


def icon_button(parent, text: str, command: Callable, tooltip: str = "", bg: str = C.panel2,
                fg: str = C.text, size: int = 9, pad: tuple = (8, 4)) -> tk.Button:
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, font=font(size),
                  bd=0, relief="flat", padx=pad[0], pady=pad[1], cursor="hand2",
                  activebackground=C.border2, activeforeground=C.text,
                  highlightthickness=0)
    if tooltip:
        Tooltip(b, tooltip)
    return b


def chip(parent, text: str, bg: str = C.panel2, fg: str = C.muted, size: int = 8) -> tk.Label:
    return tk.Label(parent, text=text, bg=bg, fg=fg, font=font(size, True), padx=6, pady=1)
