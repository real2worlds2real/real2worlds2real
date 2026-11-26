"""Robot-vs-scene penetration: the robot's swept volume over a motion, checked against everything placed.

Each robot link is the convex hull of its visual mesh, posed at every sampled physics state of a motion (IK
waypoints in apps/placements.py, every few control steps of the rollout in apps/episode.py). Then:
  * movable furniture (/World/Support) inside the sweep slides out of it: the shortest horizontal move that also
    keeps it clear of the room and of every other item (a stool slides along the counter or is pulled back), or it
    is removed when nothing within 1.5 m works;
  * the room (furniture boxes, walls) and background objects (/World/Static) cannot give way: robot points inside a
    box, or background points inside a link, are penetration (the base pose is rejected / the episode fails QA);
  * manipulable objects other than the target are dynamic: the arm pushes them, reported as contact.
Movable furniture should not collide while the motion is planned (`support_collisions(stage, False)` before the
first reset): the robot spawns where the photo had a stool. PhysX follows the USD edits made here at runtime.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.robots.sweep_geometry import Hull

from .clearance import MARGIN, TOL, box_push, room_obstacles, shift, world_points

DEPTH_OK = 0.01  # robot-vs-room contact up to 1 cm (fingertips brushing the counter) is not penetration




def children(stage, root):
    r = stage.GetPrimAtPath(root)
    return [p for p in r.GetChildren() if p.IsActive()] if r.IsValid() else []


def support_collisions(stage, enabled, root="/World/Support"):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(bool(enabled))


class RobotSweep:
    def __init__(self, stage, vega, path="/World/Robot"):
        """Link point clouds in their body frames, from the authored USD state (meshes and links read together)."""
        cache = UsdGeom.XformCache()
        self.vega, self.local, self.hulls = vega, {}, []
        for name in vega.body_names:
            prim = stage.GetPrimAtPath(f"{path}/{name}")
            P = world_points(prim, 20000) if prim.IsValid() else np.zeros((0, 3))
            if len(P) < 10:
                continue
            M = np.asarray(cache.GetLocalToWorldTransform(prim))  # row vectors: p_w = p @ M[:3, :3] + M[3, :3]
            R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=1, keepdims=True)  # scale removed: physics body frame
            L = (P - M[3, :3]) @ R.T
            try:
                L = L[ConvexHull(L, qhull_options="QJ").vertices]
            except Exception:  # flat link
                pass
            self.local[name] = L
        self._idx = {n: i for i, n in enumerate(vega.body_names)}

    def restart(self, vega):
        """Empty sweep on a (re-created) articulation handle."""
        self.vega, self.hulls, self._last = vega, [], {}
        return self

    def sample(self, tol=1e-3):
        """Every link's hull at the current physics state; a link that has not moved (by `tol`: m, and quaternion
        components) since its last sample adds nothing -- a base, the wheels, a tucked arm would otherwise repeat in
        every sample and slow every furniture slide (an hour per recorded batch)."""
        tf = self.vega.view._physics_view.get_link_transforms()
        tf = (tf.cpu().numpy() if hasattr(tf, "cpu") else np.asarray(tf))[0]
        last = getattr(self, "_last", {})
        for name, L in self.local.items():
            t = np.asarray(tf[self._idx[name]], float)
            if name in last and np.abs(t - last[name]).max() < tol:
                continue
            last[name] = t
            self.hulls.append((name, Hull(L @ Rotation.from_quat(t[3:7]).as_matrix().T + t[:3])))  # xyzw
        self._last = last

    def fork(self, vega):
        """An empty sweep of another articulation with the same links (a batch copy's robot)."""
        out = object.__new__(RobotSweep)
        out.vega, out.local, out._idx, out.hulls, out._last = vega, self.local, self._idx, [], {}
        return out

    def shifted(self, d):
        """A copy of the sweep translated by d: a batch copy's rollout checked against another copy's scene."""
        out = object.__new__(RobotSweep)
        out.vega, out.local, out._idx = self.vega, self.local, self._idx
        out.hulls = [(n, h.shifted(np.asarray(d, float))) for n, h in self.hulls]
        out._last = {}
        return out

    def bounds(self):
        P = np.concatenate([h.P for _, h in self.hulls])
        return P.min(0), P.max(0)

    def _boxes(self):
        """The sampled hulls' bounds as arrays (rebuilt when hulls were added): one vectorized prefilter instead of
        a Python test per hull (a stool slid out of a place sweep tested ~20000 hulls per candidate shift)."""
        key = (id(self.hulls), len(self.hulls))  # restart() starts a new list
        if getattr(self, "_key", None) != key:
            self._lo = np.array([h.lo for _, h in self.hulls]).reshape(-1, 3)
            self._hi = np.array([h.hi for _, h in self.hulls]).reshape(-1, 3)
            self._key = key
        return self._lo, self._hi

    def hits(self, P):
        """{link: points of P inside it at any sampled pose}."""
        lo, hi = P.min(0), P.max(0)
        LO, HI = self._boxes()
        out = {}
        for i in np.flatnonzero(np.all(LO <= hi, axis=1) & np.all(lo <= HI, axis=1)):
            name, h = self.hulls[i]
            n = int(h.contains(P).sum())
            if n:
                out[name] = max(out.get(name, 0), n)
        return out

    def in_boxes(self, obstacles):
        """[(link, box, depth)]: robot points inside room boxes (deepest per pair)."""
        deep = {}
        for name, h in self.hulls:
            for oname, (lo, hi) in obstacles.items():
                if not h.overlaps(lo + TOL, hi - TOL):
                    continue
                Q = h.P[np.all((h.P > lo + TOL) & (h.P < hi - TOL), axis=1)]
                if len(Q):
                    d = float(np.minimum(Q - lo, hi - Q).min(1).max())
                    deep[(name, oname)] = max(deep.get((name, oname), 0.0), d)
        return [(k[0], k[1], v) for k, v in deep.items()]


def slide(P, sweep, obstacles, others, n_dir=16, reach=1.5, step=0.02, hints=()):
    """Shortest horizontal shift that clears P of the sweep, the room boxes and the other items' hulls; a shift in
    `hints` that already does (one found for a similar rollout of the same scene) is taken without the search."""
    def free(Q):
        return (not sweep.hits(Q) and all(box_push(Q, lo, hi)[0] is None for lo, hi in obstacles.values())
                and not any(o.overlaps(Q.min(0), Q.max(0)) and o.contains(Q).any() for o in others))

    for h in hints:
        if free(P + h):
            return np.array(h, float)
    best = None
    for a in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        u = np.array([np.cos(a), np.sin(a), 0.0])
        for d in np.arange(step, reach if best is None else min(reach, best[0]), step):
            if free(P + d * u):
                best = (d, (d + MARGIN) * u)
                break
    return None if best is None else best[1]


def clear_robot(ts, sweep, target=None, move=True, root="/World", offset=None, hints=None):
    """Check (and with move=True resolve) the robot sweep against the scene. Returns
    {ok, moved: [{item, links, moved_m, action}], penetration: [{link, against, depth_m}], contact: [...]}.
    `root` / `offset`: a batch copy of the scene (its Static / Support / Objects under root, moved by offset).
    `hints` ({item name: [shift]}, updated): the shifts earlier rollouts of the scene gave each item, tried first."""
    stage = ts.stage
    off = np.zeros(3) if offset is None else np.asarray(offset, float)
    obstacles = {k: (lo + off, hi + off) for k, (lo, hi) in room_obstacles(ts).items()}
    rep = {"moved": [], "penetration": [], "contact": []}
    for link, box, depth in sweep.in_boxes(obstacles):
        if depth > DEPTH_OK:
            rep["penetration"].append({"link": link, "against": box, "depth_m": round(depth, 4)})
    statics = [(p.GetPath().pathString, world_points(p)) for p in children(stage, f"{root}/Static")]
    for path, P in statics:
        h = sweep.hits(P)
        if h:
            rep["penetration"].append({"link": max(h, key=h.get), "against": path, "points": sum(h.values())})
    support = children(stage, f"{root}/Support")
    for prim in support:
        P = world_points(prim)
        h = sweep.hits(P)
        if not h:
            continue
        name = prim.GetPath().pathString
        if not move:
            rep["penetration"].append({"link": max(h, key=h.get), "against": name, "points": sum(h.values())})
            continue
        others = [Hull(Q) for _, Q in statics if len(Q) > 3] + \
                 [Hull(world_points(q)) for q in support if q != prim and q.IsActive()]
        d = slide(P, sweep, obstacles, others, hints=(hints or {}).get(prim.GetName(), ()))
        if d is not None and hints is not None and not any(np.allclose(d, h) for h in hints.get(prim.GetName(), ())):
            hints.setdefault(prim.GetName(), []).append(d)
        if d is None:
            prim.SetActive(False)
            rep["moved"].append({"item": name, "links": sorted(h), "moved_m": None, "action": "removed"})
        else:
            shift(prim, d)
            rep["moved"].append({"item": name, "links": sorted(h), "moved_m": round(float(np.linalg.norm(d)), 3),
                                 "shift": np.round(d[:2], 3).tolist(), "action": "slid"})
    for prim in children(stage, f"{root}/Objects"):
        if prim.GetName() != target:
            h = sweep.hits(world_points(prim))
            if h:
                rep["contact"].append({"link": max(h, key=h.get), "against": prim.GetPath().pathString})
    rep["ok"] = not rep["penetration"]
    return rep


def summarize(rep):
    s = [f"{m['item'].split('/')[-1]} {m['action']} {m['moved_m'] or 0:.2f} m" for m in rep["moved"]]
    s += [f"{p['link']}~{p['against'].split('/')[-1]} " + (f"{p['depth_m'] * 100:.1f}cm" if "depth_m" in p else
                                                         f"{p['points']} pts") for p in rep["penetration"]]
    s += [f"contact {c['link']}~{c['against'].split('/')[-1]}" for c in rep["contact"]]
    return f"robot clearance: {'ok' if rep['ok'] else 'PENETRATION'}" + (" | " + "; ".join(s) if s else "")
