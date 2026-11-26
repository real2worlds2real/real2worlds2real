"""Functional grasps and in-hand skills from a demo: where the object is held, how it sits in the hand, pour / insert.

* hold: at a close event the TCP in the object frame (segment ee_rel[-1]) is measured against the object's mesh --
  the material between the fingers (rays from both fingertips toward the TCP: its width, the closing axis along which
  it is straddled), the TCP's offset from the object's vertical axis, its place along an elongated object's long axis,
  the mesh part (connected component) under the contact. classify() decides from these alone whether the demo held
  the object by a functional part (a thin wall / rim, a handle, a separate part, a tool's grip end) or by its body (a
  generic grasp): no part labels or hand-written regions. The region is the contact's part or a ball around it;
  seed_region() turns it into BODex seed constraints (grasping.synth `region`).
* in-hand pose: the demo's TCP in the object frame, in a canonical TCP frame (z approach, y closing / thumb: canon())
  so grippers and hands compare; rank_by_inhand() orders candidate grasps by their error to it; the object's
  rotational symmetry about its vertical axis is measured on the mesh (symmetry_z), not configured.
* candidates: BODex records -> TCP poses in the object frame (bodex_tcp), or, for a parallel gripper without a BODex
  asset, antipodal pairs sampled in the region (antipodal).
* held-object skills are re-targeted as OBJECT paths relative to the reference, then the TCP follows from the new
  in-hand pose: T_tcp = T_obj @ inv(T_tcp_obj_new); the approach before the close and the retreat after the release
  keep their motion relative to the TCP (retarget_segments). pour(): the object turns about a horizontal axis through
  its spout held over the target; insert(): the object slides along the hole's axis into its final pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .io import inv, make_T


def parts_of(V, F, min_main=0.5):
    """Per-face part label: connected components after merging coincident vertices (a handle built or reconstructed
    apart from the body). None when no component holds `min_main` of the area: a fragmented reconstruction (triangle
    soup) has no meaningful parts."""
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    m.merge_vertices()
    labels = np.zeros(len(F), int)
    for k, faces in enumerate(trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)))):
        labels[np.asarray(faces, int)] = k
    area = np.bincount(labels, weights=m.area_faces)
    return labels if area.max() >= min_main * area.sum() else None


def grasp_region(V, F, T_obj_tcp, radius=0.03, labels=None, probe=0.0):
    """Region of the object (mesh in its frame) the demo held it by. T_obj_tcp: TCP in the object frame at the close;
    `probe` moves the query point along the TCP's approach axis (z) onto the fingers' contact. -> {"contact", "normal",
    "part", "center", "radius", "T_obj_tcp"}; with labels, center / radius bound the contact's whole part."""
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    q = np.asarray(T_obj_tcp, float)[:3, 3] + probe * np.asarray(T_obj_tcp, float)[:3, 2]
    # one query point: against every triangle (no spatial index needed)
    C = trimesh.triangles.closest_point(m.triangles, np.repeat(q[None], len(m.faces), 0))
    f = int(np.argmin(np.linalg.norm(C - q, axis=1)))
    c = C[f]
    out = {"contact": c.tolist(), "normal": m.face_normals[f].tolist(), "part": None, "center": c.tolist(),
           "radius": float(radius), "T_obj_tcp": np.asarray(T_obj_tcp).tolist()}
    if labels is not None:
        part = int(labels[f])
        P = np.asarray(V)[np.unique(np.asarray(F)[labels == part])]
        ctr = (P.min(0) + P.max(0)) / 2
        out.update(part=part, center=ctr.tolist(), radius=float(np.linalg.norm(P - ctr, axis=1).max()))
    return out


def seed_region(region, margin=0.01, max_hull_depth=None):
    """BODex seed constraint (grasping.synth.synthesize `region`): object frame = the seated mesh's frame (bottom
    centre at the origin, as sim.scene.add_object). The hull-depth filter is off: a handle's inside is the grasp."""
    return {"center": [float(v) for v in region["center"]], "radius": float(region["radius"]) + margin,
            "max_hull_depth": max_hull_depth}


def in_region(points, region, margin=0.01):
    return np.linalg.norm(np.asarray(points) - np.asarray(region["center"]), axis=1) <= region["radius"] + margin


def inhand_error(T_obj_tcp, T_obj_tcp_ref, symmetry=None):
    """(position m, rotation rad) between two TCP poses in the object frame; symmetry "z": turns of the TCP about the
    object's z axis are free (a glass: any side; not a mug held by its handle)."""
    A, B = np.asarray(T_obj_tcp, float), np.asarray(T_obj_tcp_ref, float)
    if symmetry == "z":
        a = np.arctan2(B[1, 3], B[0, 3]) - np.arctan2(A[1, 3], A[0, 3])
        A = make_T(Rotation.from_euler("z", a).as_matrix()) @ A
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())


def rank_by_inhand(candidates, T_obj_tcp_ref, pos_tol=0.02, rot_tol_deg=25.0, symmetry=None):
    """Synthesized grasps (TCP poses in the object frame) -> indices within tolerance, best first (error in m +
    0.05 m per rad)."""
    errs = [inhand_error(T, T_obj_tcp_ref, symmetry) for T in candidates]
    ok = [i for i, (p, r) in enumerate(errs) if p <= pos_tol and np.degrees(r) <= rot_tol_deg]
    return sorted(ok, key=lambda i: errs[i][0] + 0.05 * errs[i][1])


# --- the demo's hold, measured on the mesh ---------------------------------------------------------------------------
#: classify() thresholds: held width / the object's width along the same axis (a wall, a handle), TCP offset from the
#: vertical axis / the object's radius (a rim, a handle), part area fraction (a separate part), place along the long
#: axis / half its length (a tool's grip end) for objects at least `elongated` times longer than wide
HOLD = {"thin": 0.35, "off_axis": 0.55, "part_frac": 0.5, "end": 0.45, "elongated": 2.5, "radius": 0.03}


def _mesh(V, F):
    import trimesh
    return trimesh.Trimesh(np.asarray(V, float), np.asarray(F), process=False)


def first_hits(m, origins, dirs, faces=False):
    """Distance along each ray to its first hit on the mesh (inf: none) [and the hit face, -1: none]. Moller-Trumbore
    against every triangle (no spatial index: a reconstructed object has a few 10k faces and the callers cast a few
    hundred rays)."""
    tri = np.asarray(m.triangles)
    a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out, face = np.full(len(origins), np.inf), np.full(len(origins), -1)
    for k, (o, d) in enumerate(zip(np.asarray(origins, float), np.asarray(dirs, float))):
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - a
        u = np.einsum("ij,ij->i", s, p) * inv_det
        q = np.cross(s, e1)
        v = (q @ d) * inv_det
        t = np.einsum("ij,ij->i", e2, q) * inv_det
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0)
        if hit.any():
            i = np.flatnonzero(hit)[np.argmin(t[hit])]
            out[k], face[k] = t[i], i
    return (out, face) if faces else out


def surface(m, n=20000):
    """Dense surface samples of the mesh with their face normals and a KD-tree (nearest-surface queries without an
    rtree index), cached on the mesh object."""
    if getattr(m, "_r2s2r_surface", None) is None:
        from scipy.spatial import cKDTree
        P, fid = m.sample(n, return_index=True)
        m._r2s2r_surface = (cKDTree(P), np.asarray(m.face_normals)[fid], fid)
    return m._r2s2r_surface


def nearest(m, points):
    """(distance to the surface ~ sample spacing, face normal, face index) per point."""
    tree, N, fid = surface(m)
    d, i = tree.query(np.asarray(points, float).reshape(-1, 3))
    return d, N[i], fid[i]


def held_width(m, p, axis, max_width=0.085):
    """Material between two fingertips at p +- max_width/2 * axis, closing toward p: (width m, hit a, hit b) or None
    when either finger closes on nothing."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    o = np.stack([p + a * max_width / 2, p - a * max_width / 2])
    t = first_hits(m, o, np.stack([-a, a]))
    if not np.all(t <= max_width):
        return None
    ha, hb = o[0] - a * t[0], o[1] + a * t[1]
    return float(max(0.0, (ha - hb) @ a)), ha, hb


def closing_axis(m, T_obj_tcp, max_width=0.085, n=36):
    """The direction (object frame, perpendicular to the TCP's approach z) along which the fingers straddle the least
    material: the demo's closing axis, measured (a dataset's gripper frame convention is not needed). None: no
    direction straddles material."""
    T = np.asarray(T_obj_tcp, float)
    best = None
    for th in np.linspace(0, np.pi, n, endpoint=False):
        a = T[:3, :3] @ [np.cos(th), np.sin(th), 0.0]
        w = held_width(m, T[:3, 3], a, max_width)
        if w is not None and (best is None or w[0] < best[0] - 1e-4):
            best = (w[0], a)
    return None if best is None else best[1]


def width_along(m, axis):
    """The object's extent along `axis` (its projection: a line through the contact may leave a tilted or open
    reconstruction early)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return float(np.ptp(np.asarray(m.vertices) @ a))


def symmetry_z(V, F, angles=(45, 90, 135, 180), tol=0.06, n=2000, n_ref=40000):
    """'z' when the mesh maps onto itself turned about its vertical axis through the origin (a glass, a bowl; not a
    mug with its handle): the 95th percentile of `n` turned surface samples' distance to `n_ref` surface samples, over
    the object's radius, stays below `tol` for every angle. -> (None | "z", worst ratio)."""
    import trimesh
    from scipy.spatial import cKDTree
    m = _mesh(V, F)
    R = trimesh.sample.sample_surface(m, n_ref, seed=0)[0]
    P = R[:n]
    tree = cKDTree(R)
    r = float(np.linalg.norm(R[:, :2], axis=1).max()) or 1.0
    worst = 0.0
    for a in angles:
        Q = P @ Rotation.from_euler("z", a, degrees=True).as_matrix().T
        worst = max(worst, float(np.percentile(tree.query(Q)[0], 95)) / r)
    return ("z" if worst <= tol else None), worst


def canon(T, tcp_axes, closing=None):
    """A TCP pose (4x4, link frame) in the canonical TCP frame: z = approach, y = closing (gripper) or thumb (hand),
    x = y x z. `closing` (a vector in the pose's parent frame) overrides the spec's closing axis (measured on the mesh:
    closing_axis)."""
    from calibrated_scene_reconstruction.robots.grasps import _axis
    T = np.asarray(T, float)

    def col(name):
        i, s = _axis(tcp_axes[name])
        return s * T[:3, i]

    z = col("approach")
    y = np.asarray(closing, float) if closing is not None else col(
        next(k for k in ("closing", "thumb", "fingers") if k in tcp_axes))
    y = y - (y @ z) * z
    y /= np.linalg.norm(y)
    return make_T(np.stack([np.cross(y, z), y, z], 1), T[:3, 3])


def classify(m, T_obj_tcp, closing=None, max_width=0.085, labels=None, th=None, opening=None):
    """Was the object held by a functional part? Measured, not configured: -> {"functional", "why", features}.
    `closing`: the closing axis (object frame), measured by closing_axis when None. `opening`: the gripper's opening
    while it held the object (m, from the demo's own gripper signal): what the fingers closed on, independent of the
    reconstruction; it decides "thin" when given, the mesh's held width otherwise."""
    th = {**HOLD, **(th or {})}
    T = np.asarray(T_obj_tcp, float)
    p = T[:3, 3]
    V = np.asarray(m.vertices)
    feats, why = {}, []
    a = closing if closing is not None else closing_axis(m, T, max_width)
    w = held_width(m, p, a, max_width) if a is not None else None
    if w is not None:
        feats["held_width_m"] = w[0]
    if w is not None or opening is not None:
        feats["object_width_m"] = width_along(m, a if a is not None else T[:3, 1])
        if opening is not None:
            feats["opening_m"] = float(opening)
        feats["thin"] = (feats["opening_m"] if opening is not None else w[0]) / max(feats["object_width_m"], 1e-6)
        if feats["thin"] <= th["thin"]:
            why.append("thin")
    r_obj = float(np.linalg.norm(V[:, :2], axis=1).max())
    feats["off_axis"] = float(np.linalg.norm(p[:2])) / max(r_obj, 1e-6)
    if feats["off_axis"] >= th["off_axis"]:
        why.append("off_axis")
    if labels is not None and len(set(np.asarray(labels).tolist())) > 1:
        f = int(nearest(m, p)[2][0])
        area = np.bincount(labels, weights=m.area_faces)
        feats["part_frac"] = float(area[labels[f]] / area.sum())
        if feats["part_frac"] <= th["part_frac"]:
            why.append("part")
    c = V.mean(0)
    _, U = np.linalg.eigh(np.cov((V - c).T))
    L = np.ptp((V - c) @ U, axis=0)
    if L[2] >= th["elongated"] * max(L[1], 1e-6):
        feats["along_long_axis"] = float(abs((p - c) @ U[:, 2]) / (L[2] / 2))
        if feats["along_long_axis"] >= th["end"]:
            why.append("end")
    return {"functional": bool(why), "why": why, "closing": None if a is None else np.asarray(a).tolist(), **feats}


def hold(V, F, T_obj_tcp, tcp_axes=None, max_width=0.085, grasp=None, opening=None):
    """The demo's hold of one object (mesh in its seated frame): region, measured closing axis, canonical in-hand
    pose, symmetry, functional decision. `grasp`: the task config's `grasp` ("type": "auto" lets the data decide,
    "functional" | "any" override it; "radius"; "symmetry": "auto" | null | "z"; thresholds "hold"); `opening`: the
    demo gripper's opening while holding (classify)."""
    grasp = grasp or {}
    m = _mesh(V, F)
    labels = parts_of(V, F)
    c = classify(m, T_obj_tcp, None, max_width, labels, grasp.get("hold"), opening)
    kind = grasp.get("type", "auto")
    functional = c["functional"] if kind == "auto" else kind == "functional"
    region = grasp_region(V, F, T_obj_tcp, grasp.get("radius", HOLD["radius"]), labels if "part" in c["why"] else None)
    T_c = canon(T_obj_tcp, tcp_axes or {"approach": "+z", "closing": "+y"}, c["closing"])
    sym = grasp.get("symmetry", "auto")
    sym, sym_err = symmetry_z(V, F) if sym == "auto" else (sym, None)
    return {**region, "T_obj_tcp": np.asarray(T_obj_tcp).tolist(), "T_obj_tcp_canon": T_c.tolist(),
            "functional": functional, "decided_by": "data" if kind == "auto" else "config", "measure": c,
            "symmetry": sym, "symmetry_err": sym_err}


# --- candidate grasps as TCP poses in the object frame ---------------------------------------------------------------
def bodex_tcp(rec, T_base_tcp, stage="grasp"):
    """A BODex record (hand base-link pose in the target's frame, as trajgen.bodex_target reads it) -> TCP pose."""
    st = rec["stages"][stage]
    return make_T(Rotation.from_quat(st["orientation"], scalar_first=True).as_matrix(), st["position"]) @ T_base_tcp


def hand_tcp(arm, base_link):
    """T_base_tcp of a hand whose BODex base link is the arm's TCP link (tcp_offset in it); a TCP on another link needs
    the articulation's FK (trajgen.bodex_target, Isaac Sim)."""
    if arm["tcp_link"] != base_link:
        raise NotImplementedError(f"TCP link {arm['tcp_link']} != BODex base link {base_link}: needs FK")
    return make_T(p=arm.get("tcp_offset", [0.0, 0.0, 0.0]))


def antipodal(V, F, region=None, max_width=0.08, n=400, approaches=12, max_up=0.3, palm=0.06, seed=0):
    """Parallel-gripper grasps sampled on the mesh (the region's surface when given): a surface point, the opposite
    surface along its inward normal within max_width (a reconstructed wall may be a millimetre thin) with a parallel
    normal (friction cone ~45 deg; a reconstruction's normals need not be oriented), TCP at the pair's midpoint,
    closing along the pair, approach turned about it (not from below: approach z <= max_up), the palm `palm` m behind
    the TCP clear of the mesh. -> canonical TCP poses (k, 4, 4) in the object frame."""
    m = _mesh(V, F)
    rng = np.random.default_rng(seed)
    P, fid = m.sample(n * 4, return_index=True)
    N = m.face_normals[fid]
    if region is not None:
        keep = in_region(P, region)
        P, N = P[keep], N[keep]
    if len(P) > n:
        k = rng.choice(len(P), n, replace=False)
        P, N = P[k], N[k]
    t, hit = first_hits(m, P - 1e-4 * N, -N, faces=True)
    out = []
    for p, nrm, d, f in zip(P, N, t, hit):
        if not (5e-4 < d <= max_width - 0.005) or abs(m.face_normals[f] @ nrm) < 0.7:
            continue
        q = p - nrm * (d + 1e-4)
        y, mid = -nrm, (p + q) / 2
        ref = np.array([0.0, 0.0, -1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z0 = ref - (ref @ y) * y
        z0 /= np.linalg.norm(z0)
        for th in np.linspace(0, 2 * np.pi, approaches, endpoint=False):
            z = Rotation.from_rotvec(y * th).as_matrix() @ z0
            if z[2] > max_up or nearest(m, mid - palm * z)[0][0] < 0.01:
                continue
            out.append(make_T(np.stack([np.cross(y, z), y, z], 1), mid))
    return np.stack(out) if out else np.zeros((0, 4, 4))


def flip(T):
    """The same parallel-gripper grasp with the fingers swapped (180 deg about the approach)."""
    return np.asarray(T, float) @ make_T(Rotation.from_euler("z", np.pi).as_matrix())


def rank(candidates, h, grasp=None, parallel=True):
    """Candidates (canonical TCP poses in the object frame) against a hold (hold()): functional -> those within
    tolerance of the demo's in-hand pose, best first; generic -> all, best first. -> [(index, pos err m, rot err rad)].
    A parallel gripper's grasp equals its finger-swapped twin."""
    grasp = grasp or {}
    ref = np.asarray(h["T_obj_tcp_canon"], float)
    errs = []
    for T in candidates:
        e = [inhand_error(T, ref, h["symmetry"])] + ([inhand_error(flip(T), ref, h["symmetry"])] if parallel else [])
        errs.append(min(e, key=lambda x: x[0] + 0.05 * x[1]))
    tol = (grasp.get("inhand_pos_tol", 0.02), grasp.get("inhand_rot_tol_deg", 25.0)) if h["functional"] else (
        np.inf, np.inf)
    ok = [i for i, (p, r) in enumerate(errs) if p <= tol[0] and np.degrees(r) <= tol[1]]
    return [(i, *errs[i]) for i in sorted(ok, key=lambda i: errs[i][0] + 0.05 * errs[i][1])]


# --- held-object skills ----------------------------------------------------------------------------------------------
def object_path(seg, T_tcp_obj):
    """A held segment's object poses in its reference's frame (the demo's in-hand pose T_tcp_obj)."""
    return seg.ee_rel @ np.asarray(T_tcp_obj, float)


def tcp_path(T_ref_new, obj_rel, T_tcp_obj_new):
    """Object path (reference frame) -> TCP poses in the world for a new reference pose and a new in-hand pose."""
    return np.asarray(T_ref_new, float) @ obj_rel @ inv(T_tcp_obj_new)


def link_map(tcp_axes):
    """C with T_canon = T_link @ C for a robot's TCP link (spec tcp_axes)."""
    return canon(np.eye(4), tcp_axes)


def retarget_segments(segs, grasps, skills=None, meshes=None):
    """Segments with new in-hand poses. grasps {close segment index: (T_old, T_new)}: the demo's and the new TCP pose
    (TCP link frame) in the grasped object's frame. The approach (the close segment) keeps its motion relative to the
    TCP: T_new @ inv(T_old) @ ee_rel; the held segments keep the OBJECT path (object_path / tcp_path), a `skills`
    entry {segment index: {"type": "pour" | "insert", ...}} replaces its object-centric part; the segment after the
    release keeps its motion relative to the release pose. meshes {object: vertices in its frame}: a pour's spout when
    the config has none."""
    from dataclasses import replace
    skills = {int(k): v for k, v in (skills or {}).items()}
    out = list(segs)
    for k, (T_old, T_new) in grasps.items():
        T_old, T_new = np.asarray(T_old, float), np.asarray(T_new, float)
        sg = out[k]
        out[k] = replace(sg, ee_rel=T_new @ inv(T_old) @ sg.ee_rel)
        j = k + 1
        while j < len(out) and out[j].held == sg.ref:
            h = out[j]
            O = object_path(h, inv(T_old))
            sk = skills.get(h.index, {})
            if sk.get("type") in ("pour", "insert"):
                O = O.copy()
                O[h.skill:] = skill_path(O[h.skill:], sk, (meshes or {}).get(h.held))
            new = O @ T_new  # tcp_path in the reference frame: O @ inv(T_tcp_obj_new)
            D = new[-1] @ inv(h.ee_rel[-1])
            out[j] = replace(h, ee_rel=new)
            j += 1
        if j < len(out) and out[j].held is None and out[j].ref == out[j - 1].ref and out[j - 1].held == sg.ref:
            out[j] = replace(out[j], ee_rel=D @ out[j].ee_rel)
    return out


def skill_path(O, sk, V=None):
    """A held segment's object-centric part (object poses in the reference frame) re-made by a skill: pour about the
    spout over the reference's origin (axis / spout / height from the config, else from the demo's own pour: the axis
    its z turned about, the rim vertex lowest at the deepest tilt) or insert along the axis into the demo's final
    pose."""
    if sk["type"] == "insert":
        return insert(np.eye(4), O[-1], sk.get("axis", (0.0, 0.0, 1.0)), sk.get("depth", 0.05), len(O))
    ang = pour_profile(O)
    k = int(np.argmax(ang))
    axis = sk.get("axis")
    if axis is None:  # the horizontal axis the object's z turned about at the deepest tilt
        axis = np.cross([0.0, 0.0, 1.0], O[k][:3, 2])
        axis = axis / max(np.linalg.norm(axis), 1e-9)
    spout = sk.get("spout")
    if spout is None and V is not None:
        V = np.asarray(V, float)
        rim = V[V[:, 2] >= V[:, 2].max() - 0.1 * np.ptp(V[:, 2])]
        spout = rim[np.argmin((rim @ O[k][:3, :3].T)[:, 2])]
    spout = np.asarray(spout if spout is not None else [0.0, 0.0, 0.0], float)
    height = sk.get("height", float((O[k][:3, :3] @ spout + O[k][:3, 3])[2]))
    return pour(np.eye(4), O[0], spout, axis, ang - ang[0], height)


def tilt(T_obj, up=(0.0, 0.0, 1.0)):
    return float(np.arccos(np.clip(np.asarray(T_obj)[:3, 2] @ np.asarray(up), -1, 1)))


def pour_profile(obj_rel):
    """Tilt angle (rad) of the object's z over a pour segment (object poses in the target's frame, z up)."""
    return np.array([tilt(T) for T in obj_rel])


def pour(T_target, T_obj0, spout, axis, angles, height=0.08):
    """Object poses of a pour: the object (upright pose T_obj0, world) turned by each angle about the horizontal
    `axis` (world) through its spout (object frame), the spout held `height` above the target's origin."""
    p_sp = np.asarray(T_target)[:3, 3] + [0.0, 0.0, height]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    out = []
    for a in angles:
        R = Rotation.from_rotvec(axis * a).as_matrix() @ np.asarray(T_obj0)[:3, :3]
        out.append(make_T(R, p_sp - R @ np.asarray(spout, float)))
    return np.stack(out)


def insert(T_hole, T_hole_obj, axis=(0.0, 0.0, 1.0), depth=0.05, n=30):
    """Object poses of an insertion: from `depth` back along the hole's axis (hole frame) to the final pose T_hole @
    T_hole_obj, orientation fixed (aligned before contact)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    out = []
    for s in np.linspace(0.0, 1.0, n):
        T = np.asarray(T_hole_obj, float).copy()
        T[:3, 3] += (1.0 - s) * depth * axis
        out.append(np.asarray(T_hole, float) @ T)
    return np.stack(out)


def aligned(T_obj, T_hole, T_hole_obj, axis=(0.0, 0.0, 1.0), lateral_tol=0.005, rot_tol_deg=3.0):
    """Insertion check: the object's offset from its final pose lies along the hole's axis (lateral error within
    tolerance) and its orientation matches."""
    D = inv(np.asarray(T_hole) @ np.asarray(T_hole_obj)) @ np.asarray(T_obj)
    axis_obj = np.asarray(T_hole_obj)[:3, :3].T @ (np.asarray(axis, float) / np.linalg.norm(axis))
    d = D[:3, 3]
    lateral = np.linalg.norm(d - (d @ axis_obj) * axis_obj)
    return lateral <= lateral_tol and np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()) <= rot_tol_deg
