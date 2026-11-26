"""Penetration check and resolution for everything placed into a scene (twin or cousin).

Obstacles: the room's furniture collision boxes and its walls (structure planes thickened behind the surface).
Items, resolved in this order, each against the obstacles and every item placed before it:
  /World/Static   background objects (templated fixtures such as a faucet go through the counter by design: fixed)
  /World/Support  movable furniture (stools)
  /World/Objects  manipulable objects (the task target first, so distractors give way to it)
An item's world-space mesh points are tested against obstacle boxes and the convex hulls of earlier items. An item
sunk into the top of a box (all intruding points within SINK of its top: an object resting on a counter that the
reconstruction placed a little low) is lifted onto it; otherwise it is pushed out horizontally by the smallest xy
shift that clears it (things stand on a floor or counter, never pushed down), repeated until clear or `max_iter`.
An item standing on a furniture box (a cup on the counter) is only pushed along that box's top: the push direction
is searched (away from the other item first, then around) for the shortest one that keeps its footprint on the top;
when none does, the item is not cleared (a library item is then removed rather than left to fall off the edge).
Reconstructed background meshes come from the photo: they are never pushed sideways, and an overlap between two of
them (the convex hulls of a plant and the basket behind it) is the photo's layout, reported but kept. Their deeper
intrusions into the room are shape errors of the reconstruction (a trailing plant made too deep) and are conformed:
the intruding vertices are projected onto the nearest box face. A library item that cannot be cleared is removed.
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt
from scipy.spatial import Delaunay

TOL = 0.004  # contact (an object resting on a counter) is not penetration
MARGIN = 0.01
SINK = 0.05  # intrusion this close below a box top is an item resting too low on it: lift instead of push


def world_points(prim, max_pts=4000):
    cache = UsdGeom.XformCache()
    pts = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        P = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if P is None or len(P) == 0:
            continue
        M = np.asarray(cache.GetLocalToWorldTransform(p))
        pts.append(np.c_[np.asarray(P, float), np.ones(len(P))] @ M)
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts)[:, :3]
    return P[np.linspace(0, len(P) - 1, min(len(P), max_pts)).astype(int)]


def shift(prim, d):
    """Move an item's root by d (xyz; adds a translate op when the root has none, e.g. world-space meshes)."""
    xf = UsdGeom.Xformable(prim)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if ops:
        cur = ops[0].Get()
        ops[0].Set(type(cur)(cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]))
    else:
        order = xf.GetOrderedXformOps()
        op = xf.AddTranslateOp(opSuffix="clearance")
        op.Set(Gf.Vec3d(*[float(v) for v in d]))
        xf.SetXformOpOrder([op] + order)


def room_obstacles(scene):
    """{name: (lo, hi)}: furniture collision boxes + walls / windows thickened 0.2 m behind their surface."""
    boxes = {k: (np.asarray(lo, float), np.asarray(hi, float)) for k, (lo, hi) in scene.manifest["collision_boxes"].items()}
    for name, part in scene.manifest["parts"].items():
        if part.get("kind") != "structure" or name.startswith(("floor", "ceiling")):
            continue
        V = np.load(scene.s / "scene" / part["npz"])["V"]
        n = np.asarray(part["normal"], float)  # into the room
        lo, hi = V.min(0), V.max(0)
        back = V - 0.2 * n
        boxes[name] = (np.minimum(lo, back.min(0)), np.maximum(hi, back.max(0)))
    return boxes


def conform(prim, obstacles):
    """Project the vertices of a world-space mesh (no xform ops) that lie inside a box onto its nearest face
    (bottom face excluded). Returns the number of moved vertices."""
    mesh = UsdGeom.Mesh(prim)
    V = np.asarray(mesh.GetPointsAttr().Get(), float)
    n = 0
    for lo, hi in obstacles.values():
        lo_s, hi_s = lo + TOL, hi - TOL
        inside = np.all((V > lo_s) & (V < hi_s), axis=1)
        if not inside.any():
            continue
        Q = V[inside]
        # distance to the top, -x, +x, -y, +y faces; move to the closest
        d = np.stack([hi[2] - Q[:, 2], Q[:, 0] - lo[0], hi[0] - Q[:, 0], Q[:, 1] - lo[1], hi[1] - Q[:, 1]], 1)
        k = d.argmin(1)
        Q[k == 0, 2] = hi[2] + TOL
        Q[k == 1, 0] = lo[0] - TOL
        Q[k == 2, 0] = hi[0] + TOL
        Q[k == 3, 1] = lo[1] - TOL
        Q[k == 4, 1] = hi[1] + TOL
        V[inside] = Q
        n += int(inside.sum())
    if n:
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    return n


def box_push(P, lo, hi):
    """Shift (xyz) that moves every point of P out of the box (shrunk by TOL): a lift when P only sinks into the
    box top, else the smallest horizontal push. (None, 0) when nothing is inside."""
    lo_s, hi_s = lo + TOL, hi - TOL
    inside = np.all((P > lo_s) & (P < hi_s), axis=1)
    if not inside.any():
        return None, 0.0
    Q = P[inside]
    sink = hi_s[2] - Q[:, 2].min()
    if sink < SINK and P[:, 2].max() > hi[2]:
        return np.array([0.0, 0.0, sink + TOL]), float(sink)
    cands = [np.array([hi_s[0] - Q[:, 0].min(), 0, 0]), np.array([lo_s[0] - Q[:, 0].max(), 0, 0]),
             np.array([0, hi_s[1] - Q[:, 1].min(), 0]), np.array([0, lo_s[1] - Q[:, 1].max(), 0])]
    d = min(cands, key=np.linalg.norm)
    return d + MARGIN * np.sign(d), float(np.linalg.norm(d))


def support_of(P, obstacles, tol=0.03):
    """(lo, hi) of the furniture box whose top the points P stand on (bottom within `tol` of the top, centre over
    it), or None (on the floor, or hanging)."""
    z0, c = P[:, 2].min(), P[:, :2].mean(0)
    best = None
    for lo, hi in obstacles.values():
        if abs(hi[2] - z0) < tol and np.all(c >= lo[:2]) and np.all(c <= hi[:2]) and \
                (best is None or abs(hi[2] - z0) < abs(best[1][2] - z0)):
            best = (lo, hi)
    return best


def _on_top(P, keep, d):
    lo, hi = keep
    Q = P[:, :2] + d
    return bool(np.all(Q >= lo[:2] - TOL) and np.all(Q <= hi[:2] + TOL))


def hull_push(P, hull_pts, keep=None):
    """xy shift that moves P out of the convex hull of hull_pts: the step search runs along the centroid direction;
    with `keep` (the (lo, hi) box P stands on) the shift must keep P on its top, so 16 directions around are searched
    as well and the shortest valid one wins; a zero shift when no direction clears it on the top."""
    try:
        tri = Delaunay(hull_pts)
    except Exception:
        return None, 0.0
    if not (tri.find_simplex(P) >= 0).any():
        return None, 0.0
    d0 = P[:, :2].mean(0) - hull_pts[:, :2].mean(0)
    d0 = d0 / (np.linalg.norm(d0) + 1e-9) if np.linalg.norm(d0) > 1e-6 else np.array([0.0, -1.0])
    a0 = np.arctan2(d0[1], d0[0])
    dirs = [d0] if keep is None else [np.array([np.cos(a0 + a), np.sin(a0 + a)])
                                      for a in np.radians(np.r_[0, np.repeat(np.arange(22.5, 180.1, 22.5), 2)
                                                               * np.tile([1, -1], 8)])]
    best = None
    for d in dirs:
        for step in np.arange(0.01, 1.0, 0.01):
            if best is not None and step >= best[1]:
                break
            if not (tri.find_simplex(P + np.r_[d * step, 0]) >= 0).any():
                if keep is None or _on_top(P, keep, d * (step + MARGIN)):
                    best = (d, float(step))
                break
    if best is None:
        return (np.zeros(3), 1.0) if keep is not None else (np.r_[d0 * MARGIN, 0.0], 0.0)
    d, depth = best
    return np.r_[d * (depth + MARGIN), 0.0], depth


def chunks(P, cell=0.08, min_pts=8):
    """A measured structure's points in cells of `cell` m: each cell's points one small convex piece."""
    keys = np.floor(P / cell).astype(int)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return [P[inv.ravel() == k] for k in range(inv.max() + 1) if (inv.ravel() == k).sum() >= min_pts]


def others(placed, recon):
    """Items an item must not overlap: for a reconstructed instance only non-reconstructed ones."""
    return [x for x in placed if not (recon and x[2])]


def resolve(scene, roots=("/World/Static", "/World/Support", "/World/Objects"), first=None, max_iter=8):
    """Check and fix penetrations in place. `first`: item names placed (and kept fixed) first within their group.
    Returns a report [{item, against, depth_m, moved_m, clear}]."""
    stage = scene.stage
    obstacles = room_obstacles(scene)
    placed, report = [], []
    for root in roots:
        r = stage.GetPrimAtPath(root)
        if not r.IsValid():
            continue
        items = list(r.GetChildren())
        # `first` items, then photo reconstructions (they only move vertically), then swaps / props pushed around them
        items.sort(key=lambda p: (p.GetName() not in (first or ()),
                                  not (root == "/World/Static" and p.IsA(UsdGeom.Mesh) and
                                       not UsdGeom.Xformable(p).GetOrderedXformOps()),
                                  p.GetPath().pathString))
        for prim in items:
            # Decals are observed surface markings, without a physical solid to clear.
            if prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit") == "decal":
                continue
            P = world_points(prim)
            if not len(P):
                continue
            name = prim.GetPath().pathString
            fixture = root == "/World/Static" and scene.manifest["static_objects"].get(prim.GetName(), {}).get("template")
            if fixture:
                placed.append((name, P, True))
                continue
            if prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit") == "mesh":  # a structure measured from the frames: fixed where they
                # put it, and an obstacle in small pieces -- the convex hull of a frame's bars filled the desk inside it
                # and every cup on it was pushed off
                placed += [(f"{name}#{k}", Q, False) for k, Q in enumerate(chunks(world_points(prim, 40000)))]
                continue
            recon = root == "/World/Static" and prim.IsA(UsdGeom.Mesh) and not UsdGeom.Xformable(prim).GetOrderedXformOps()
            hits, moved, action = {}, np.zeros(3), None
            keep = None if recon else support_of(P, obstacles)
            for _ in range(max_iter):
                push = None
                for oname, (lo, hi) in obstacles.items():
                    d, depth = box_push(P, lo, hi)
                    if d is not None:
                        hits[oname] = max(hits.get(oname, 0.0), depth)
                        push = d
                        break
                if push is None:
                    for oname, H, _ in others(placed, recon):
                        d, depth = hull_push(P, H, keep)
                        if d is not None:
                            hits[oname] = max(hits.get(oname, 0.0), depth)
                            push = d
                            break
                if push is None or (recon and push[2] == 0) or not np.any(push):  # clear / photo layout / blocked
                    break
                P = P + push
                moved += push
            if np.any(moved):
                shift(prim, moved)
            if recon and any(box_push(P, lo, hi)[0] is not None for lo, hi in obstacles.values()):
                if np.any(moved):  # a lift alone did not clear it: undo, conform the shape instead
                    shift(prim, -moved)
                    moved[:] = 0
                n = conform(prim, obstacles)
                action = f"conformed {n} vertices"
                P = world_points(prim)
            clear = all(box_push(P, lo, hi)[0] is None for lo, hi in obstacles.values()) and \
                all(hull_push(P, H, keep)[0] is None for _, H, _ in others(placed, recon))
            if recon:  # overlaps between two photo instances (a plant in front of a basket): their layout is the photo's
                for oname, H, r in placed:
                    if r and hull_push(P, H)[0] is not None:
                        report.append({"item": name, "against": oname, "depth_m": None, "moved_m": 0.0,
                                       "action": "photo layout kept", "clear": True, "recon": True})
            if not clear and not recon:  # a library swap / stool / prop that does not fit here: leave it out
                prim.SetActive(False)
                action = "removed"
            for oname, depth in hits.items():
                report.append({"item": name, "against": oname, "depth_m": round(depth, 4),
                               "moved_m": round(float(np.linalg.norm(moved)), 4), "action": action,
                               "clear": bool(clear or action == "removed"), "recon": bool(recon)})
            if prim.IsActive():
                placed.append((name, P, recon))
    return report


def summarize(report):
    bad = [r for r in report if not r["clear"]]
    fixed = sorted({r["item"] for r in report if r["clear"] and not r["action"]})
    acted = sorted({f"{r['item'].split('/')[-1]} {r['action']}" for r in report if r["action"]})
    return (f"clearance: {len(fixed)} item(s) moved clear" + (f", {'; '.join(acted)}" if acted else "")
            + (f", {len({r['item'] for r in bad})} still penetrating: "
               + ", ".join(f"{r['item'].split('/')[-1]}~{r['against'].split('/')[-1]} {r['depth_m'] * 100:.1f}cm"
                           for r in bad) if bad else ", none left"))
