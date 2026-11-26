"""Task frame: where a robot stands to work at the support, derived from the scene graph instead of world constants.

The approach edge is the side of the support box (task.support) that has free floor in front of it (no fixed furniture
box, not behind a wall, inside the floor's range) and is close to the target. The frame is
  origin   target xy projected onto that edge, z = support top
  forward  horizontal, from the free floor into the support (the direction the robot faces)
  left     up x forward
  up       +z
Movable furniture (stools) does not block an edge: it slides out of the robot's sweep (sim.sweep).
"""
import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def _boxes(sg, skip=()):
    return {k: np.asarray(v["box"], float) for k, v in sg.get("furniture", {}).items() if k not in skip}


def _walls(sg):
    """Vertical structure planes as (axis index, coordinate, range along the other horizontal axis)."""
    out = []
    for v in sg.get("structure", {}).values():
        if v.get("plane") in ("x", "y") and not str(v.get("role", "wall")).startswith("window"):
            ax = 0 if v["plane"] == "x" else 1
            out.append((ax, float(v["at"]), v["range"][0]))
    return out


def _floor(sg):
    for k, v in sg.get("structure", {}).items():
        if v.get("role", k) == "floor" or k == "floor":
            return np.asarray(v["range"], float)  # [[x0, x1], [y0, y1]]
    return None


def _free(pts, ref, boxes, walls, floor):
    """Fraction of floor points `pts` (N, 2) that are free: inside the floor, no fixed box below 1.5 m, and on the same
    side of every wall as the reference point (the target)."""
    ok = np.ones(len(pts), bool)
    if floor is not None:
        ok &= ((pts[:, 0] >= floor[0, 0]) & (pts[:, 0] <= floor[0, 1]) &
               (pts[:, 1] >= floor[1, 0]) & (pts[:, 1] <= floor[1, 1]))
    for b in boxes.values():
        if b[0, 2] < 1.5:
            ok &= ~((pts[:, 0] > b[0, 0]) & (pts[:, 0] < b[1, 0]) & (pts[:, 1] > b[0, 1]) & (pts[:, 1] < b[1, 1]))
    for ax, at, rng in walls:
        other = 1 - ax
        within = (pts[:, other] >= min(rng) - 0.05) & (pts[:, other] <= max(rng) + 0.05)
        ok &= ~(within & (np.sign(pts[:, ax] - at) != np.sign(ref[ax] - at)))
    return float(ok.mean())


def approach_edges(sg, p_target, depth=0.8, dist_weight=0.5):
    """Score the four sides of the support box; returns a list of dicts, best first."""
    task = sg["task"]
    box = np.asarray(sg["furniture"][task["support"]]["box"], float)
    boxes = _boxes(sg, skip={task["support"], task.get("support_body")})
    walls, floor = _walls(sg), _floor(sg)
    t = np.asarray(p_target, float)[:2]
    edges = []
    for ax in (0, 1):
        for sgn in (-1, 1):
            n = np.zeros(2)
            n[ax] = sgn
            edge = box[1 if sgn > 0 else 0, ax]
            other = 1 - ax
            lo, hi = box[0, other], box[1, other]
            u = np.linspace(lo, hi, 12)
            d = np.linspace(0.05, depth, 8)
            pts = np.zeros((len(u) * len(d), 2))
            pts[:, other] = np.repeat(u, len(d))
            pts[:, ax] = edge + sgn * np.tile(d, len(u))
            free = _free(pts, t, boxes, walls, floor)
            dist = abs(edge - t[ax])
            edges.append({"normal": n, "edge": float(edge), "axis": ax, "free": free, "dist": float(dist),
                          "score": free - dist_weight * dist})
    return sorted(edges, key=lambda e: -e["score"])


class TaskFrame:
    def __init__(self, sg, p_target, edge=None):
        self.sg = sg
        box = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
        self.support_top = float(box[1, 2])
        e = edge or approach_edges(sg, p_target)[0]
        self.edge_info = e
        n = np.array([e["normal"][0], e["normal"][1], 0.0])
        self.forward = -n
        self.left = np.cross(UP, self.forward)
        self.right = -self.left
        self.up = UP
        t = np.asarray(p_target, float)
        o = t.copy()
        o[e["axis"]] = e["edge"]
        o[2] = self.support_top
        self.origin = o
        self.yaw = float(np.arctan2(self.forward[1], self.forward[0]))

    def vec(self, right=0.0, forward=0.0, up=0.0):
        return right * self.right + forward * self.forward + up * self.up

    def sample_base(self, spec, rng):
        """Base (x, y, yaw) for a robot spec: `standoff` back from the edge, `lateral` to the robot's right of the
        target, heading within `yaw_tol_deg` of facing the support."""
        b = spec["base"]
        lat = float(rng.uniform(*b.get("lateral", (-0.3, 0.3))))
        st = float(rng.uniform(*b.get("standoff", (0.3, 0.6))))
        yaw = self.yaw + np.radians(float(rng.uniform(-b.get("yaw_tol_deg", 20), b.get("yaw_tol_deg", 20))))
        p = self.origin + lat * self.right - st * self.forward
        return float(p[0]), float(p[1]), float(yaw)

    def to_dict(self):
        return {"origin": self.origin.tolist(), "forward": self.forward.tolist(), "left": self.left.tolist(),
                "support_top": self.support_top, "edge": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                                          for k, v in self.edge_info.items()}}
