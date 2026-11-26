"""Reward terms of the library (policy_development/library/rewards/*.json), implemented over the observations
every cloud station provides.  Each term: f(prev, cur, ep, p) -> float for one env, where prev / cur are that env's
observation dicts, ep the per-episode memory (start centroids) and p the term's parameters.  The station's hidden judge
is never used here; the sparse success bonus comes from the station's episode-end flag in the learner."""
import numpy as np


def centroid(o, lab):
    m = np.asarray(o["labels"]) == lab
    if m.sum() < 15:
        return None
    return np.median(np.asarray(o["cloud"], np.float32)[m], axis=0)


class Episode:
    def __init__(self, obs0, label_ids):
        self.start = {n: centroid(obs0, i) for n, i in label_ids.items()} if "labels" in obs0 else {}
        self.palm0 = np.asarray(obs0["palm_pos"], float)


def _c(o, ep, ids, name):
    return centroid(o, ids[name]) if name in ids else None


def reach_palm_to_label(prev, cur, ep, p, ids):
    c = _c(cur, ep, ids, p["label"])
    if c is None:
        return 0.0
    return -float(np.linalg.norm(np.asarray(cur["palm_pos"]) - (c + np.asarray(p.get("offset", [0, 0, 0.1])))))


def grasp_contact(prev, cur, ep, p, ids):
    t = float(np.max(cur.get("tactile_n", [0])))
    return 1.0 if p.get("lo", 2.0) <= t <= p.get("hi", 25.0) else 0.0


def object_lift_height(prev, cur, ep, p, ids):
    c, s = _c(cur, ep, ids, p["label"]), ep.start.get(p["label"])
    if c is None or s is None:
        return 0.0
    h = p.get("h", 0.10)
    return float(np.clip(c[2] - s[2], 0, h) / h)


def hold_still(prev, cur, ep, p, ids):
    return -float(np.abs(np.asarray(cur["q29"])[:7] - np.asarray(prev["q29"])[:7]).sum())


def carry_xy_to_target(prev, cur, ep, p, ids):
    a, b = _c(cur, ep, ids, p["label_a"]), _c(cur, ep, ids, p["label_b"])
    if a is None or b is None:
        return 0.0
    return -float(np.linalg.norm(a[:2] - (b[:2] + np.asarray(p.get("offset", [0, 0]))[:2])))


def place_upright_release(prev, cur, ep, p, ids):
    a, b, s = _c(cur, ep, ids, p["label_a"]), _c(cur, ep, ids, p["label_b"]), ep.start.get(p["label_a"])
    if a is None or b is None or s is None:
        return 0.0
    on_table = abs(a[2] - s[2]) < p.get("z_tol", 0.015)
    released = float(np.max(cur.get("tactile_n", [0]))) < p.get("free_n", 1.0)
    d = np.linalg.norm(a[:2] - (b[:2] + np.asarray(p.get("offset", [0, -0.11]))[:2]))
    return 1.0 if on_table and released and d < p.get("tol", 0.03) else 0.0


def other_object_moved(prev, cur, ep, p, ids):
    c, s = _c(cur, ep, ids, p["label"]), ep.start.get(p["label"])
    if c is None or s is None:
        return 0.0
    return -float(max(0.0, np.linalg.norm(c[:2] - s[:2]) - p.get("deadband", 0.005)))


def insert_depth(prev, cur, ep, p, ids):
    m_a = np.asarray(cur["labels"]) == ids.get(p["label_a"], -1)
    m_b = np.asarray(cur["labels"]) == ids.get(p["label_b"], -1)
    if m_a.sum() < 15 or m_b.sum() < 15:
        return 0.0
    cl = np.asarray(cur["cloud"], np.float32)
    d = p.get("d", 0.03)
    return float(np.clip(cl[m_b][:, 2].max() - cl[m_a][:, 2].min(), 0, d) / d)


TERMS = {f.__name__: f for f in (reach_palm_to_label, grasp_contact, object_lift_height, hold_still, carry_xy_to_target,
                                 place_upright_release, other_object_moved, insert_depth)}
TERMS["expression"] = None          # defined below; compiled per term


def make(spec: list, labels: dict):
    """spec: [{"id", "weight", "params"}] (success_sparse is handled by the learner) -> r(prev, cur, ep)."""
    ids = {v: int(k) for k, v in labels.items()}
    use = []
    for t in spec:
        if t["id"] == "success_sparse":
            continue
        if t["id"] not in TERMS:
            raise ValueError(f"reward term {t['id']!r} has no implementation; available: {sorted(TERMS)} + success_sparse")
        fn = compile_expression(t.get("params", {}).get("expr", "")) if t["id"] == "expression" else TERMS[t["id"]]
        use.append((fn, float(t.get("weight", 1.0)), t.get("params", {})))

    def r(prev, cur, ep):
        return sum(w * f(prev, cur, ep, p, ids) for f, w, p in use)
    return r, ids


# ---------------------------------------------------------------------- expression: transcribed rewards (e.g. from the work library)
import ast  # noqa: E402
import math  # noqa: E402

_MATH = {"tanh": math.tanh, "exp": lambda x: math.exp(max(-50.0, min(50.0, x))), "abs": abs, "min": min, "max": max,
         "sqrt": lambda x: math.sqrt(max(0.0, x)), "clip": lambda x, lo, hi: max(lo, min(hi, x)),
         "ge": lambda x, y: 1.0 if x >= y else 0.0, "le": lambda x, y: 1.0 if x <= y else 0.0}


def _features(prev, cur, ep, ids):
    def c(lab):
        return _c(cur, ep, ids, lab)

    def palm_dist(lab, dx=0.0, dy=0.0, dz=0.0):
        x = c(lab)
        return 1.0 if x is None else float(np.linalg.norm(np.asarray(cur["palm_pos"]) - (x + np.array([dx, dy, dz]))))

    def rise(lab):
        x, s = c(lab), ep.start.get(lab)
        return 0.0 if x is None or s is None else float(x[2] - s[2])

    def xy_moved(lab):
        x, s = c(lab), ep.start.get(lab)
        return 0.0 if x is None or s is None else float(np.linalg.norm(x[:2] - s[:2]))

    def xy_dist(a, b, ox=0.0, oy=0.0):
        x, y = c(a), c(b)
        return 1.0 if x is None or y is None else float(np.linalg.norm(x[:2] - (y[:2] + np.array([ox, oy]))))

    t = np.asarray(cur.get("tactile_n", [0.0]), float)
    return {"palm_dist": palm_dist, "rise": rise, "xy_moved": xy_moved, "xy_dist": xy_dist,
            "label_z": lambda lab: 0.0 if c(lab) is None else float(c(lab)[2]),
            "palm_z": lambda: float(np.asarray(cur["palm_pos"])[2]),
            "palm_rise": lambda: float(np.asarray(cur["palm_pos"])[2] - ep.palm0[2]),
            "tactile_max": lambda: float(t.max()), "tactile_sum": lambda: float(t.sum()),
            "contact": lambda lo=2.0, hi=25.0: 1.0 if lo <= float(t.max()) <= hi else 0.0,
            "arm_speed": lambda: float(np.abs(np.asarray(cur["q29"])[:7] - np.asarray(prev["q29"])[:7]).sum())}


FEATURES = ("palm_dist(label, dx, dy, dz)", "rise(label)", "xy_moved(label)", "xy_dist(label_a, label_b, ox, oy)", "label_z(label)",
            "palm_z()", "palm_rise()", "tactile_max()", "tactile_sum()", "contact(lo_n, hi_n)", "arm_speed()")


def compile_expression(expr: str):
    """A bounded arithmetic expression over FEATURES and tanh/exp/abs/min/max/sqrt/clip/ge/le; no names, attributes or
    other calls (a transcription format, not code)."""
    tree = ast.parse(expr, mode="eval")
    allowed_calls = set(_MATH) | {f.split("(")[0] for f in FEATURES}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls or node.keywords:
                raise ValueError(f"call not allowed in a reward expression: {ast.dump(node.func)[:80]}")
        elif isinstance(node, ast.Name):
            if node.id not in allowed_calls:
                raise ValueError(f"name not allowed in a reward expression: {node.id}")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult,
                                   ast.Div, ast.USub, ast.UAdd)):
            raise ValueError(f"syntax not allowed in a reward expression: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str)):
            raise ValueError("only numbers and label names are allowed as constants")
    code = compile(tree, "<reward>", "eval")

    def f(prev, cur, ep, p, ids):
        env = {**_MATH, **_features(prev, cur, ep, ids), "__builtins__": {}}
        v = float(eval(code, env))  # noqa: S307 - validated AST above
        return v if math.isfinite(v) else 0.0
    return f


def expression(prev, cur, ep, p, ids):
    raise RuntimeError("expression terms are compiled by make()")
