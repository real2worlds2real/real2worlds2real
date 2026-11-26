"""S4b instance-level layout from the scene graph: every photographed thing becomes a 3D instance.

    python -m calibrated_scene_reconstruction.recon.layout sessions/<name>

structure  -> textured planes (floor / walls / ceiling / windows)
furniture  -> textured, collidable boxes (cabinets, fridge, counters), optional sink cut-out with a basin
objects    -> SAM 3D meshes / templates, instantiated in Isaac from the scene graph (sim.scene)

Each face is baked from the object-free plate twice: `baked` (photo as is) and, once apps/scene/irradiance.py has written
scene/irradiance_photo.npy, `albedo` = photo radiance / Isaac irradiance, which makes the twin relightable while it
still reproduces the photo from the photo camera. Writes scene/parts/* and scene/manifest.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.graph import furniture_role, structure_role
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.bake import PanoSource, PosedSource, ProjectionContext, bake_rect, smooth_fill

X, Y, Z = np.eye(3)
FACE_ORDER = ["front", "top", "left", "right", "bottom", "back"]


def plane_frame(axis, at, rng):
    (a0, a1), (b0, b1) = rng
    if axis == "z":
        return np.array([a0, b0, at]), X, Y, a1 - a0, b1 - b0
    if axis == "y":
        return np.array([a0, at, b1]), X, -Z, a1 - a0, b1 - b0
    return np.array([at, a0, b1]), Y, -Z, a1 - a0, b1 - b0


def box_faces(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return {  # name: (origin, U, V, size_u, size_v, outward normal)
        "top": (np.array([x0, y0, z1]), X, Y, x1 - x0, y1 - y0, Z),
        "bottom": (np.array([x0, y0, z0]), X, Y, x1 - x0, y1 - y0, -Z),
        "front": (np.array([x0, y0, z1]), X, -Z, x1 - x0, z1 - z0, -Y),
        "back": (np.array([x0, y1, z1]), X, -Z, x1 - x0, z1 - z0, Y),
        "left": (np.array([x0, y0, z1]), Y, -Z, y1 - y0, z1 - z0, -X),
        "right": (np.array([x1, y0, z1]), Y, -Z, y1 - y0, z1 - z0, X),
    }


def albedo_source(s: Session, plate):
    """photo / irradiance (per unit albedo), globally scaled to [0, 0.85] at the 99.5th percentile."""
    p = s / "scene" / "irradiance_photo.npy"
    if not p.exists():
        return None
    E = cv2.resize(np.load(p).mean(-1), (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
    E = cv2.GaussianBlur(E, (0, 0), 3)
    A = ((plate / 255.0) ** 2.2) / np.maximum(E, 1e-3)[..., None]
    A = np.clip(A * 0.85 / max(np.percentile(A, 99.5), 1e-6), 0, 1)
    out = (A ** (1 / 2.2) * 255).astype(np.float32)
    Image.fromarray(out.astype(np.uint8)).save(s / "scene" / "plate_albedo.png")
    return out


def measure_stools(s: Session, ctx: ProjectionContext, ids):
    """Seat centre / height / width of movable stools from their (possibly truncated) masks."""
    Pw = ctx.to_world(ctx.P)
    out = []
    for sid in ids:
        m = ctx.masks[sid] & ctx.valid
        q = Pw[m]
        x0, x1 = np.percentile(q[:, 0], [2, 98])
        y0, y1 = np.percentile(q[:, 1], [2, 98])
        cut_right, cut_left = ctx.masks[sid][:, -3:].any(), ctx.masks[sid][:, :3].any()
        width = 0.42 if (cut_right or cut_left or x1 - x0 > 0.5) else float(x1 - x0)
        cx = float(x0 + width / 2) if cut_right else float(x1 - width / 2) if cut_left else float((x0 + x1) / 2)
        out.append({"id": int(sid), "center": [cx, float((y0 + y1) / 2)], "seat_top": float(np.percentile(q[:, 2], 98)),
                    "width": width, "truncated": bool(cut_right or cut_left)})
    return out


def object_support_z(s: Session, ctx: ProjectionContext, sg):
    """Top of the slab each dynamic object rests on: the slab under its points (the task support by default)."""
    Pw = ctx.to_world(ctx.P)
    tops = {n: np.asarray(f["box"], float) for n, f in sg["furniture"].items()
            if furniture_role(sg, n, f) == "support_top"}
    out = {}
    for name, spec in sg.get("dynamic_objects", {}).items():
        m = np.zeros(ctx.masks.shape[1:], bool)
        for i in spec["ids"]:
            m |= ctx.masks[i]
        q = Pw[m & ctx.valid]
        if not len(q):
            continue
        c, bottom = np.median(q[:, :2], 0), np.percentile(q[:, 2], 2)
        under = [b for b in tops.values() if b[0, 0] - 0.05 < c[0] < b[1, 0] + 0.05 and
                 b[0, 1] - 0.05 < c[1] < b[1, 1] + 0.05 and b[1, 2] < bottom + 0.06]
        if under:
            out[name] = float(max(b[1, 2] for b in under))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--pano", default="", help="scene/pano/<tag> to fill unobserved texels from ('none': off)")
    ap.add_argument("--structure-only", default="",
                    help="bake only the structure planes into this directory, manifest untouched (a cousin's room)")
    args = ap.parse_args()
    s = Session(args.session)
    sg = s.scene_graph
    ctx = ProjectionContext(s)
    pdir = s / "scene" / "pano" / args.pano
    pano = PanoSource(pdir) if args.pano != "none" and (pdir / "views.npz").exists() else None
    posed = s / "parsing" / "posed_frames" / "frames.npz"
    if pano is None and args.pano != "none" and posed.exists():  # no completion: the walks' posed frames, where taken
        pano = PosedSource(posed)
        print(f"[layout] structure from {len(pano.names)} posed walk frames (recon.posed_frames)", flush=True)
    plate = np.asarray(Image.open(s / "scene" / "plate.png").convert("RGB")).astype(np.float32)
    sources = {"baked": plate}
    alb = albedo_source(s, plate)
    if alb is not None:
        sources["albedo"] = alb
    gains = {"baked": np.ones(3, np.float32)}  # pano views are radiance: the photo's albedo / radiance ratio
    if alb is not None:
        v = ctx.valid & ~ctx.removed & (plate.max(-1) > 20)
        gains["albedo"] = np.median(alb[v] / np.maximum(plate[v], 1.0), 0).astype(np.float32)
    out = Path(args.structure_only) if args.structure_only else s / "scene" / "parts"
    out.mkdir(parents=True, exist_ok=True)
    parts, boxes = {}, {}

    def bake(O, U, V, su, sv, res, ids, fill_color=None, use_pano=False):
        texs, obs, corners, cov = {}, None, None, None
        ctx.pano = pano if use_pano else None
        for k, src in sources.items():
            ctx.source, ctx.pano_gain = src, gains[k]
            texs[k], obs, corners, cov = bake_rect(ctx, O, U, V, su, sv, res, ids, fill_color=fill_color)
        covered[0] = cov
        return texs, obs, corners

    covered = [None]  # the last bake's pano-covered texels (emit writes them next to the observed mask)

    def emit(name, texs, obs, corners, normal, **extra):
        F = np.array([[0, 1, 2], [0, 2, 3]])
        if np.cross(corners[1] - corners[0], corners[2] - corners[0]) @ normal < 0:
            F = F[:, [0, 2, 1]]  # RTX does not emit/shade back faces the way we want: wind towards the normal
        np.savez_compressed(out / f"{name}.npz", V=corners, F=F, UV=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float))
        for k, t in texs.items():
            Image.fromarray(t).save(out / f"{name}_{k}.png")
        Image.fromarray((obs * 255).astype(np.uint8)).save(out / f"{name}_observed.png")
        cov = covered[0] if covered[0] is not None and covered[0].shape == obs.shape else np.zeros_like(obs)
        covered[0] = None
        if cov.any():
            extra["covered"] = f"parts/{name}_covered.png"
            extra["covered_fraction"] = float((obs | cov).mean())
            Image.fromarray(((obs | cov) * 255).astype(np.uint8)).save(out / f"{name}_covered.png")
        parts[name] = {"npz": f"parts/{name}.npz", "observed": f"parts/{name}_observed.png",
                       **{f"tex_{k}": f"parts/{name}_{k}.png" for k in texs},
                       "normal": normal.tolist(), "observed_fraction": float(obs.mean()), **extra}

    # structure planes -------------------------------------------------------------------------------------
    for name, st in sg["structure"].items():
        O, U, V, su, sv = plane_frame(st["plane"], st["at"], st["range"])
        normal = {"x": X, "y": Y, "z": Z}[st["plane"]] * (-1 if st.get("normal", "+") == "-" else 1)
        role = structure_role(name, st)
        res = 0.008 if role == "floor" else 0.006 if role == "window" else 0.01
        if isinstance(pano, PanoSource) and role != "window":  # completed views: ~5 mm detail at 3 m, 1 cm blurs it
            res = 0.005
        texs, obs, corners = bake(O, U, V, su, sv, res, st["ids"], use_pano=pano is not None)
        if isinstance(pano, PosedSource) and covered[0] is not None:
            # real frames, unlike a completion's generated views: what they see is observed (a look {"photo": true}
            # keeps it as the texture, the PBR fill only where no frame saw the plane)
            obs, covered[0] = obs | covered[0], None
        cov = covered[0]
        if cov is not None and role != "window" and (obs | cov).mean() >= 0.6 and not (obs | cov).all():
            # mostly photo + completed views: the holes (hidden behind things in both) continue the texture around
            # them instead of a PBR patch of another material
            hole = ~(obs | cov)
            texs = {k: cv2.inpaint(t, hole.astype(np.uint8), 9, cv2.INPAINT_TELEA) for k, t in texs.items()}
            covered[0] = cov | hole
        if role == "ceiling" and cov is not None and cov.any():
            # the completed views meet on the ceiling at steps of brightness (six ring views around the up view
            # draw a hexagon): a flat ceiling keeps only their low frequencies, the photo's own texels stay as seen
            sig = 0.12 / res
            texs = {k: np.where(obs[..., None], t, cv2.GaussianBlur(t.astype(np.float32), (0, 0), sig)).astype(np.uint8)
                    for k, t in texs.items()}
        if role == "window" and obs.any() and not obs.all():
            # a window's unseen texels (a lamp or a frame in front of it) continue the view around them; the
            # observed mean would paint them as flat white blots
            seen = obs | (covered[0] if covered[0] is not None else False)
            texs = {k: cv2.inpaint(t, (~seen).astype(np.uint8), 7, cv2.INPAINT_TELEA) for k, t in texs.items()}
        emit(name, texs, obs, corners, normal, kind="structure", collide=st.get("collide", False),
             emissive=st.get("emissive", 0.0))
    if args.structure_only:
        (out / "parts.json").write_text(json.dumps(parts, indent=1))
        print(f"{len(parts)} structure parts -> {out}")
        return

    # furniture boxes ----------------------------------------------------------------------------------------
    for name, f in sg["furniture"].items():
        lo, hi = f["box"]
        boxes[name] = [lo, hi]
        faces = box_faces(lo, hi)
        base = None
        material = f.get("material", "painted")
        seen = f.get("faces", {})  # a measured piece (a rig's posts from recon.object_fit) has no photo faces: its look
        for fname in FACE_ORDER:  # is the scene graph's (sim.setpieces)
            ids = seen.get(fname, [])
            O, U, V, su, sv, n = faces[fname]
            if not ids:
                continue
            if fname == "top" and f.get("sink"):
                (sx0, sy0), (sx1, sy1) = f["sink"]["xy"]
                z1 = hi[2]
                for k, (p0, p1) in {"a": ([lo[0], lo[1]], [hi[0], sy0]), "b": ([lo[0], sy1], [hi[0], hi[1]]),
                                    "c": ([lo[0], sy0], [sx0, sy1]), "d": ([sx1, sy0], [hi[0], sy1])}.items():
                    if min(p1[0] - p0[0], p1[1] - p0[1]) > 1e-3:
                        texs, obs, corners = bake(np.array([p0[0], p0[1], z1]), X, Y, p1[0] - p0[0], p1[1] - p0[1],
                                                  0.004, ids)
                        emit(f"{name}_top_{k}", texs, obs, corners, Z, kind="furniture", material=material)
                dz, bz = f["sink"]["depth"], z1 - f["sink"]["depth"]
                for k, (O2, U2, V2, su2, sv2, n2) in {
                        "floor": ([sx0, sy0, bz], X, Y, sx1 - sx0, sy1 - sy0, Z),
                        "wy0": ([sx0, sy0, z1], X, -Z, sx1 - sx0, dz, Y), "wy1": ([sx0, sy1, z1], X, -Z, sx1 - sx0, dz, -Y),
                        "wx0": ([sx0, sy0, z1], Y, -Z, sy1 - sy0, dz, X), "wx1": ([sx1, sy0, z1], Y, -Z, sy1 - sy0, dz, -X)}.items():
                    O2 = np.asarray(O2, float)
                    corners = np.array([O2, O2 + su2 * U2, O2 + su2 * U2 + sv2 * V2, O2 + sv2 * V2])
                    t = np.full((8, 8, 3), 215, np.uint8)
                    emit(f"{name}_basin_{k}", {k2: t for k2 in sources}, np.zeros((8, 8), bool), corners, n2,
                         kind="furniture", material="ceramic")
                continue
            texs, obs, corners = bake(O, U, V, su, sv, 0.004, ids)
            if fname == "front" and f.get("smooth_front"):
                texs = {k: smooth_fill(t, obs, f["smooth_front"]) for k, t in texs.items()}
            if base is None and obs.any():
                base = {k: t[obs].mean(0) for k, t in texs.items()}
            emit(f"{name}_{fname}", texs, obs, corners, n, kind="furniture", material=material)
        for fname in FACE_ORDER:  # unseen faces: the box's observed mean colour
            if seen.get(fname):
                continue
            O, U, V, su, sv, n = faces[fname]
            corners = np.array([O, O + su * U, O + su * U + sv * V, O + sv * V])
            texs = {k: np.tile(np.asarray(c, np.uint8), (8, 8, 1))
                    for k, c in (base or {k: [180, 170, 150] for k in sources}).items()}
            emit(f"{name}_{fname}", texs, np.zeros((8, 8), bool), corners, n, kind="furniture", material=material)

    # table legs: a support top standing free above the floor (a table, not a counter on cabinets) gets four legs
    # down to the floor, coloured like its top -- the photo rarely shows them well, the robot and the renders need them
    for name, f in sg["furniture"].items():
        lo, hi = np.asarray(f["box"], float)
        if furniture_role(sg, name, f) != "support_top" or lo[2] < 0.25 or f.get("legs") is False:
            continue
        under = [b for n2, b in boxes.items() if n2 != name and not n2.startswith(f"{name}_leg") and
                 b[1][2] <= lo[2] + 0.02 and b[0][0] < hi[0] and b[1][0] > lo[0] and b[0][1] < hi[1] and b[1][1] > lo[1]]
        if under:  # something already holds it up (a cabinet body)
            continue
        w, inset = float(f.get("leg_width", 0.05)), float(f.get("leg_inset", 0.04))
        col = {}
        for k in sources:  # the table's observed sides (its apron: legs share its finish), else a darker top
            seen = [out / f"{name}_{fn}_{k}.png" for fn in ("front", "left", "right", "back") if f["faces"].get(fn)]
            ref = [(np.asarray(Image.open(p_).convert("RGB"), float).reshape(-1, 3), 1.0) for p_ in seen if p_.exists()]
            if not ref and (out / f"{name}_top_{k}.png").exists():
                ref = [(np.asarray(Image.open(out / f"{name}_top_{k}.png").convert("RGB"), float).reshape(-1, 3), 0.7)]
            col[k] = np.median(ref[0][0], 0) * ref[0][1] if ref else np.array([120.0, 95.0, 70.0])
        for k, (x, y) in enumerate([(lo[0] + inset, lo[1] + inset), (hi[0] - inset - w, lo[1] + inset),
                                    (lo[0] + inset, hi[1] - inset - w), (hi[0] - inset - w, hi[1] - inset - w)]):
            llo, lhi = [x, y, 0.0], [x + w, y + w, float(lo[2])]
            boxes[f"{name}_leg{k}"] = [llo, lhi]
            faces = box_faces(llo, lhi)
            for fname in FACE_ORDER:
                if fname in ("top", "bottom"):
                    continue
                O, U, V, su, sv, n = faces[fname]
                corners = np.array([O, O + su * U, O + su * U + sv * V, O + sv * V])
                texs = {kk: np.tile(col[kk].astype(np.uint8), (8, 8, 1)) for kk in sources}
                emit(f"{name}_leg{k}_{fname}", texs, np.zeros((8, 8), bool), corners, n, kind="furniture",
                     material=f.get("material", "painted"))

    st, task = sg["structure"], sg["task"]
    sup = sg["furniture"][task["support"]]["box"]
    geo = s.geometry()
    floor = next(v for k, v in st.items() if structure_role(k, v) == "floor")
    ceiling = next((v for k, v in st.items() if structure_role(k, v) == "ceiling"), {"at": 2.5})
    movable_ids = [i for g in sg.get("movable_furniture", {}).values() for i in g["ids"]]
    manifest = {
        "parts": parts,
        "collision_boxes": boxes,
        "photo_camera": {"K": ctx.K.tolist(), "width": ctx.W, "height": ctx.H,
                         "T_world_cam_cv": ctx.T_wc.tolist(), "source": ctx.photo_pose_source},
        "surfaces": {"counter_top_z": sup[1][2], "ceiling_z": ceiling["at"],
                     "room_x": floor["range"][0], "room_y": floor["range"][1]},
        "support_z": object_support_z(s, ctx, sg),
        "stools": measure_stools(s, ctx, movable_ids),
        "static_objects": sg["static_objects"],
        "dynamic_objects": sg["dynamic_objects"],
        "textures": sorted(sources),
    }
    s.write_json("scene/manifest.json", manifest)
    print(f"{len(parts)} parts, {len(boxes)} collision boxes, textures: {sorted(sources)}")


if __name__ == "__main__":
    main()
