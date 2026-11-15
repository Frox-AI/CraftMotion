"""Minimal stub of the tkinter API so GUI code can be exercised headlessly.

Only used by the test suite (no display available in CI containers)."""
from __future__ import annotations

import sys
import types


class Dummy:
    _default_width = 900
    _default_height = 600

    def __init__(self, *args, **kwargs):
        self._children = []
        self._attrs = dict(kwargs)
        self._pending = []

    # -- lifetime
    def pack(self, *a, **k): return None
    def pack_forget(self, *a, **k): return None
    def grid(self, *a, **k): return None
    def grid_columnconfigure(self, *a, **k): return None
    def place(self, *a, **k): return None
    def destroy(self):
        for c in list(self._children):
            try:
                c.destroy()
            except Exception:
                pass
        self._children = []

    # -- geometry
    def winfo_width(self): return self._default_width
    def winfo_height(self): return self._default_height
    def winfo_rootx(self): return 0
    def winfo_rooty(self): return 0
    def winfo_children(self): return list(self._children)
    def winfo_toplevel(self): return self
    def winfo_ismapped(self): return True
    def geometry(self, *a, **k): return "900x600+0+0"
    def minsize(self, *a, **k): return None
    def resizable(self, *a, **k): return None
    def update_idletasks(self): return None
    def update(self): return None

    # -- configuration
    def configure(self, *a, **k):
        self._attrs.update(k)
        return None
    config = configure
    def cget(self, key):
        return self._attrs.get(key, "")
    def title(self, *a, **k): return "CraftMotion"
    def transient(self, *a): return None
    def protocol(self, *a): return None
    def attributes(self, *a): return None
    def state(self, *a): return "normal"
    def focus_set(self): return None
    def focus_get(self): return None
    def grab_release(self): return None
    def lift(self, *a): return None

    # -- events
    def bind(self, *a, **k): return None
    def unbind(self, *a, **k): return None
    def after(self, ms, func=None, *args):
        if func is None:
            return "after#0"
        QUEUE.append((int(ms), func, args))
        return "after#%d" % len(QUEUE)
    def after_cancel(self, _id): return None
    def mainloop(self, *a): return None

    # -- canvas-ish
    def delete(self, *a, **k): return None
    def create_rectangle(self, *a, **k): return 1
    def create_line(self, *a, **k): return 2
    def create_text(self, *a, **k): return 3
    def create_polygon(self, *a, **k): return 4
    def create_image(self, *a, **k): return 5
    def create_oval(self, *a, **k): return 6
    def create_window(self, *a, **k): return 7
    def itemconfigure(self, *a, **k): return None
    def yview(self, *a): return None
    def yview_scroll(self, *a, **k): return None
    def yview_moveto(self, *a): return None
    def xview(self, *a): return None
    def bbox(self, *a): return (0, 0, 100, 100)

    # -- text-ish
    def insert(self, *a, **k): return None
    def see(self, *a): return None
    def get(self, *a, **k): return ""
    def set(self, *a, **k): return None
    def curselection(self): return ()
    def selection_set(self, *a): return None
    def tk_popup(self, *a, **k): return None

    def __getitem__(self, key):
        return self._attrs.get(key, "")

    def __setitem__(self, key, value):
        self._attrs[key] = value

    def __getattr__(self, name):
        # unknown widget methods become no-ops
        def _noop(*a, **k):
            return None
        return _noop


QUEUE = []


class Var:
    def __init__(self, master=None, value=None, **k):
        self._value = value

    def get(self):
        return self._value if self._value is not None else self._default

    def set(self, v):
        self._value = v

    def trace_add(self, *a, **k):
        return "trace"


class StringVar(Var):
    _default = ""


class DoubleVar(Var):
    _default = 0.0


class IntVar(Var):
    _default = 0


class BooleanVar(Var):
    _default = False


class PhotoImage(Dummy):
    def __init__(self, *a, **k):
        super().__init__()
        self.width_v = k.get("width", 16)
        self.height_v = k.get("height", 16)

    def width(self): return self.width_v
    def height(self): return self.height_v


class TclError(Exception):
    pass


def install() -> None:
    """Register fake tkinter modules in sys.modules."""
    if getattr(install, "_done", False):
        return
    tk = types.ModuleType("tkinter")
    for name in ("Tk", "Toplevel", "Frame", "Label", "Button", "Entry", "Canvas", "Menu",
                 "Text", "Listbox", "Scale", "Checkbutton", "Radiobutton", "LabelFrame",
                 "PanedWindow", "Spinbox", "PhotoImage", "Message", "Scrollbar"):
        setattr(tk, name, type(name, (Dummy,), {}))
    tk.StringVar = StringVar
    tk.DoubleVar = DoubleVar
    tk.IntVar = IntVar
    tk.BooleanVar = BooleanVar
    tk.TclError = TclError
    tk.END = "end"
    tk.LEFT = "left"
    tk.RIGHT = "right"
    tk.BOTH = "both"
    tk.X = "x"
    tk.Y = "y"
    tk.VERTICAL = "vertical"
    tk.HORIZONTAL = "horizontal"
    tk.NW = "nw"
    tk.W = "w"
    tk.E = "e"
    tk.N = "n"
    tk.S = "s"

    ttk = types.ModuleType("tkinter.ttk")
    for name in ("Style", "Combobox", "Notebook", "Progressbar", "Scrollbar", "Treeview",
                 "Scale", "Frame", "Label", "Button", "Entry", "Checkbutton", "Separator"):
        setattr(ttk, name, type(name, (Dummy,), {}))
    tk.ttk = ttk

    def simple(name, ret=None):
        mod = types.ModuleType("tkinter." + name)
        def _noop(*a, **k):
            return ret
        setattr(mod, name.replace("box", "box") if False else "showinfo", _noop)
        setattr(mod, "showwarning", _noop)
        setattr(mod, "showerror", _noop)
        setattr(mod, "askstring", _noop)
        setattr(mod, "askfloat", _noop)
        setattr(mod, "askopenfilename", _noop)
        setattr(mod, "askopenfilenames", _noop)
        setattr(mod, "asksaveasfilename", _noop)
        setattr(mod, "askdirectory", _noop)
        setattr(mod, "askcolor", _noop)
        setattr(mod, "filedialog", mod)
        return mod

    tk.messagebox = simple("messagebox")
    tk.filedialog = simple("filedialog")
    tk.colorchooser = simple("colorchooser")
    tk.simpledialog = simple("simpledialog")
    tk.font = types.ModuleType("tkinter.font")

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.messagebox"] = tk.messagebox
    sys.modules["tkinter.filedialog"] = tk.filedialog
    sys.modules["tkinter.colorchooser"] = tk.colorchooser
    sys.modules["tkinter.simpledialog"] = tk.simpledialog
    sys.modules["tkinter.font"] = tk.font
    install._done = True
