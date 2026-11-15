"""Expression engine — AE-style expressions on any property or effect parameter.

Expressions are short scripts evaluated per frame, e.g.

    wiggle(2, 30)                       // organic shake
    value + [0, Math.sin(time*3)*40]    // combine with keyframed value
    loopOut("cycle")                    // repeat keyframes
    linear(time, 0, 2, 0, 360)          // remap time to a value
    thisComp.layer("Null 1").transform.position

Safety: a restricted namespace (no builtins, no imports) plus an AST guard that
rejects dunder attribute access and I/O. Expressions are a scripting feature the
user (or an agent) writes on purpose, not untrusted input.
"""
from __future__ import annotations

import ast
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BANNED_ATTRS = {"__globals__", "__builtins__", "__class__", "__subclasses__", "__code__",
                 "__dict__", "__getattribute__", "mro", "__mro__", "__base__", "__bases__",
                 "__import__", "system", "popen", "spawn", "fork"}
_BANNED_NAMES = {"open", "exec", "eval", "compile", "input", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "import", "breakpoint", "exit", "quit"}


class ExpressionError(Exception):
    pass


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ExpressionError("imports are not allowed in expressions")
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            raise ExpressionError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ExpressionError(f"name '{node.id}' is not allowed")


# ------------------------------------------------------------------ helpers

def _as_list(v: Any) -> List[float]:
    if isinstance(v, (int, float, bool)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [0.0]


def _vec(v: Any, dims: int) -> List[float]:
    vals = _as_list(v)
    while len(vals) < dims:
        vals.append(vals[-1] if vals else 0.0)
    return vals[:dims]


def _noise1(t: float, seed: float = 0.0) -> float:
    """Smooth 1-D value noise in -1..1."""
    x = t + seed * 13.37
    i = math.floor(x)
    f = x - i
    f = f * f * (3 - 2 * f)

    def h(n: int) -> float:
        n = (n * 374761393 + 668265263) & 0xFFFFFFFF
        n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
        return ((n ^ (n >> 16)) & 0xFFFF) / 32767.5 - 1.0

    return h(i) * (1 - f) + h(i + 1) * f


def wiggle(freq: float, amp: Any, octaves: int = 1, seed: float = 0.0, t: float = 0.0,
           dims: int = 2) -> List[float]:
    """AE-style wiggle: smooth random motion around zero."""
    out = []
    for d in range(dims):
        v = 0.0
        a = 1.0
        f = float(freq)
        total = 0.0
        for o in range(max(1, int(octaves))):
            v += _noise1(t * f, seed + d * 7.1 + o * 3.3) * a
            total += a
            a *= 0.5
            f *= 2.0
        out.append(v / max(total, 1e-6) * _as_list(amp)[min(d, len(_as_list(amp)) - 1)])
    return out


def value_at_time(keys: Optional[List[dict]], t_ms: float, default: Sequence[float]) -> List[float]:
    from .util import kf_eval
    if not keys:
        return [float(x) for x in default]
    return kf_eval(keys, t_ms, default)


def loop_out(keys: Optional[List[dict]], t_ms: float, mode: str, default: Sequence[float],
             pre: bool = False) -> List[float]:
    """Repeats keyframe animation before/after the key range (cycle/pingpong/offset)."""
    from .util import kf_eval
    if not keys or len(keys) < 2:
        return value_at_time(keys, t_ms, default)
    keys = sorted(keys, key=lambda k: k["t"])
    k0, k1 = keys[0], keys[-1]
    span = max(1, k1["t"] - k0["t"])
    if k0["t"] <= t_ms <= k1["t"]:
        return kf_eval(keys, t_ms, default)
    mode = (mode or "cycle").lower()
    if pre:
        if t_ms >= k0["t"]:
            return kf_eval(keys, t_ms, default)
        u = (k0["t"] - t_ms) / span
        base = [x - y for x, y in zip(k0["v"], k1["v"])] if mode == "offset" else None
        tt = k1["t"] - (u % 1.0) * span if mode != "pingpong" else \
            k0["t"] + (0.5 - abs((u % 1.0) - 0.5) * 2) * span
        v = kf_eval(keys, tt, default)
        if base:
            v = [a + b * math.floor(u + 1) for a, b in zip(v, base)]
        return v
    u = (t_ms - k1["t"]) / span
    if mode == "pingpong":
        phase = u % 2.0
        tt = k0["t"] + (phase if phase <= 1.0 else 2.0 - phase) * span
        return kf_eval(keys, tt, default)
    if mode == "offset":
        rep = math.floor(u) + 1
        v = kf_eval(keys, k1["t"] - (u % 1.0) * span, default)
        return [a + (b - c) * rep for a, b, c in zip(v, k1["v"], k0["v"])]
    return kf_eval(keys, k0["t"] + (u % 1.0) * span, default)


def linear(t: float, t_min: float, t_max: float, v_min: float, v_max: float) -> float:
    if t_max == t_min:
        return v_min
    u = max(0.0, min(1.0, (t - t_min) / (t_max - t_min)))
    return v_min + (v_max - v_min) * u


def ease(t: float, t_min: float, t_max: float, v_min: float, v_max: float) -> float:
    if t_max == t_min:
        return v_min
    u = max(0.0, min(1.0, (t - t_min) / (t_max - t_min)))
    u = u * u * (3 - 2 * u)
    return v_min + (v_max - v_min) * u


# ------------------------------------------------------------------ context

class LayerRef:
    """thisComp.layer("Name") / thisLayer — read-only property access."""

    def __init__(self, engine: "ExpressionEngine", layer: Optional[dict]):
        self.engine = engine
        self._layer = layer

    # convenience
    @property
    def transform(self) -> "TransformRef":
        return TransformRef(self.engine, self._layer)

    @property
    def name(self) -> str:
        return str(self._layer.get("name", "")) if self._layer else ""

    @property
    def index(self) -> int:
        return int(self._layer.get("id", 0)) if self._layer else 0

    @property
    def width(self) -> float:
        return float(self._layer.get("w", 0) or 0) if self._layer else 0.0

    @property
    def height(self) -> float:
        return float(self._layer.get("h", 0) or 0) if self._layer else 0.0

    def property(self, name: str) -> List[float]:
        if not self._layer:
            return [0.0]
        dims = {"position": 3, "pos": 3, "scale": 2, "rotation": 1, "opacity": 1,
                "anchor": 2, "anchorPoint": 2, "skew": 2}.get(name, 1)
        prop = {"position": "pos", "anchorPoint": "anchor"}.get(name, name)
        from .util import kf_eval
        keys = (self._layer.get("keys") or {}).get(prop)
        base = (self._layer.get("transform") or {}).get(prop, [0.0] * dims)
        if keys:
            return kf_eval(keys, self.engine.t_ms, base)
        return _vec(base, dims)

    def __getattr__(self, name: str):
        if name in ("position", "pos", "scale", "rotation", "opacity", "anchor", "anchorPoint", "skew"):
            return self.property(name)
        raise AttributeError(name)


class TransformRef:
    def __init__(self, engine: "ExpressionEngine", layer: Optional[dict]):
        self.engine = engine
        self._layer = layer

    @property
    def position(self) -> List[float]:
        return LayerRef(self.engine, self._layer).property("position")

    @property
    def scale(self) -> List[float]:
        return LayerRef(self.engine, self._layer).property("scale")

    @property
    def rotation(self) -> float:
        return LayerRef(self.engine, self._layer).property("rotation")[0]

    @property
    def opacity(self) -> float:
        return LayerRef(self.engine, self._layer).property("opacity")[0]

    @property
    def anchorPoint(self) -> List[float]:
        return LayerRef(self.engine, self._layer).property("anchorPoint")


class CompRef:
    def __init__(self, engine: "ExpressionEngine"):
        self.engine = engine

    @property
    def width(self) -> float:
        return float(self.engine.project.width)

    @property
    def height(self) -> float:
        return float(self.engine.project.height)

    @property
    def duration(self) -> float:
        return self.engine.project.duration / 1000.0

    @property
    def fps(self) -> float:
        return float(self.engine.project.fps)

    def layer(self, ref: Any) -> LayerRef:
        return self.engine.layer_ref(ref)

    @property
    def numLayers(self) -> int:
        return len(self.engine.project.layers)


class ExpressionEngine:
    def __init__(self, project, t_ms: float, layer: dict, depth: int = 0):
        self.project = project
        self.t_ms = float(t_ms)
        self.layer = layer
        self.depth = depth
        self._cache: Dict[str, Any] = {}

    # -- context helpers
    def layer_ref(self, ref: Any) -> LayerRef:
        if isinstance(ref, (int, float)):
            return LayerRef(self, self.project.get_layer(int(ref)))
        name = str(ref)
        for l in self.project.layers:
            if str(l.get("name", "")).lower() == name.lower():
                return LayerRef(self, l)
        return LayerRef(self, None)

    def _namespace(self, value: Sequence[float], keys: Optional[List[dict]]) -> Dict[str, Any]:
        eng = self
        ns: Dict[str, Any] = {
            "time": self.t_ms / 1000.0,
            "value": list(value) if len(value) > 1 else float(value[0]),
            "thisComp": CompRef(self),
            "thisLayer": LayerRef(self, self.layer),
            "Math": math,
            "math": math,
            "abs": abs, "min": min, "max": max, "round": round, "int": int, "float": float,
            "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan2": math.atan2,
            "sqrt": math.sqrt, "pow": pow, "floor": math.floor, "ceil": math.ceil,
            "pi": math.pi, "PI": math.pi, "random": lambda s=0: random.Random(int(s * 1000) + 7).random(),
            "noise": lambda t=0.0: _noise1(float(t)),
            "linear": linear, "ease": ease,
            "degrees": math.degrees, "radians": math.radians,
        }
        ns["wiggle"] = lambda f, amp, oct=1, seed=0.0: wiggle(f, amp, oct, seed, ns["time"], len(value))
        ns["loopOut"] = lambda mode="cycle": loop_out(keys, eng.t_ms, mode, value)
        ns["loopIn"] = lambda mode="cycle": loop_out(keys, eng.t_ms, mode, value, pre=True)
        ns["valueAtTime"] = lambda t: value_at_time(keys, float(t) * 1000.0, value)
        ns["velocity"] = lambda: self._velocity(keys, value)
        # lenient aliases so JS habits keep working
        ns["position"] = ns["value"]
        return ns

    def _velocity(self, keys: Optional[List[dict]], value: Sequence[float]) -> List[float]:
        dt = 1000.0 / max(1.0, self.project.fps)
        a = value_at_time(keys, self.t_ms - dt, value)
        b = value_at_time(keys, self.t_ms + dt, value)
        return [(y - x) / (2 * dt / 1000.0) for x, y in zip(a, b)]

    # -- main entry
    def evaluate(self, expr: str, value: Sequence[float], keys: Optional[List[dict]] = None) -> Optional[List[float]]:
        if not expr or not str(expr).strip():
            return None
        if self.depth > 4:
            return None
        code = self._translate(str(expr))
        try:
            tree = ast.parse(code, mode="eval")
            _check_ast(tree)
        except (SyntaxError, ExpressionError):
            return None
        ns = self._namespace(value, keys)
        try:
            result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
        except Exception:
            return None
        if isinstance(result, bool):
            return [1.0 if result else 0.0]
        if isinstance(result, (int, float)):
            return [float(result)]
        if isinstance(result, (list, tuple)):
            try:
                return [float(x) for x in result]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _translate(expr: str) -> str:
        s = expr.strip().rstrip(";")
        s = re.sub(r"\bMath\.", "math.", s)
        s = s.replace("thisComp.layer(", "thisComp.layer(")
        # JS-isms
        s = s.replace("===", "==").replace("!==", "!=").replace("&&", " and ").replace("||", " or ")
        s = s.replace("var ", "").replace("let ", "").replace("const ", "")
        s = s.replace("return ", "")
        s = re.sub(r"\btrue\b", "True", s)
        s = re.sub(r"\bfalse\b", "False", s)
        s = re.sub(r"\bnull\b", "None", s)
        s = s.replace(";", "\n")
        # vector literal helper: [a, b] works natively in python
        return s

    # -- convenience for renderer
    def eval_prop(self, layer: dict, prop: str, dims: int, default: Sequence[float]) -> List[float]:
        from .util import kf_eval
        keys = (layer.get("keys") or {}).get(prop)
        base = (layer.get("transform") or {}).get(prop, default)
        value = kf_eval(keys, self.t_ms, base) if keys else _vec(base if isinstance(base, (list, tuple)) else [base], dims)
        expr = (layer.get("expressions") or {}).get(prop)
        if expr:
            out = self.evaluate(expr, value, keys)
            if out is not None:
                return _vec(out, dims) if dims > 1 else [out[0]]
        return value

    def eval_effect_param(self, layer: dict, eff: dict, name: str, default: Any, t_local_ms: float) -> Any:
        from .util import kf_eval
        keys = (eff.get("keys") or {}).get(name)
        base = (eff.get("params") or {}).get(name, default)
        if isinstance(default, (str, bool)):
            return base
        if keys:
            value = kf_eval(keys, t_local_ms, _as_list(base if base is not None else default))
        else:
            value = _as_list(base if base is not None else default)
        expr = (eff.get("expressions") or {}).get(name)
        if expr:
            saved_t = self.t_ms
            self.t_ms = t_local_ms
            try:
                out = self.evaluate(expr, value, keys)
            finally:
                self.t_ms = saved_t
            if out is not None:
                return out[0] if len(out) == 1 else out
        return value[0] if len(value) == 1 else value


def expression_contexts_layer(expr: str) -> bool:
    return bool(re.search(r"thisComp|thisLayer|wiggle|loopOut|loopIn|valueAtTime", expr or ""))


EXPRESSION_HELP = {
    "properties": ["pos", "scale", "rot", "opacity", "anchor", "skew"],
    "global": ["time", "value", "fps", "width", "height", "duration"],
    "functions": ["wiggle(freq, amp, octaves, seed)", "loopOut(mode)", "loopIn(mode)",
                  "valueAtTime(t)", "velocity", "linear(t, t0, t1, v0, v1)",
                  "ease(t, t0, t1, v0, v1)", "noise(t, seed)", "Math.*"],
    "objects": ["thisComp", "thisLayer", "thisComp.layer(name).transform.position"],
    "example": "wiggle(2, 30) + thisComp.layer(\"Null 1\").transform.position",
}
