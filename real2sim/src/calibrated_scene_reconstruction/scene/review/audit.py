"""S3a per-object reconstruction audit: does each reconstructed mesh agree with the photo?

    python -m calibrated_scene_reconstruction.recon.audit sessions/<name> [--only cup_3] [--fix]

SAM 3D completes what the photo does not show from its shape prior, and a small, occluded object can come back as
another thing (a glass half hidden behind a glass rebuilt as a mug with a handle). Each reconstruction
(objects/<obj>/mesh_cam.glb) is compared with the photo evidence in the photo camera:
  visible IoU  projected silhouette vs the SAM 3 mask; pixels where the photo shows something in front of the mesh
               (MoGe depth nearer than the mesh's front surface) count for neither
  spill        share of the projected mesh, eroded by a few pixels, where the photo sees something farther away:
               the background shows through where the mesh claims to be
  size         mesh height and width against the masked points (the width only where the mask is not occluded)
  part         a protrusion from the body (radial profile about the vertical axis: a handle, a spout) that the photo
               does not show: most of its unoccluded pixels spill
Suspect: visible IoU < 0.5, spill > 0.2 or an unsupported part. The fix path (--fix, run by the pipeline after
align): library models of the object's class (YCB scans and Poly Haven props whose names match its assets.physprops
class) stood upright on the reconstruction's base, height from the photo, yaw searched, scored like the
reconstruction (visible IoU - spill - part); the best replaces mesh_cam.glb / albedo.png when it scores clearly
better (the reconstruction is kept as mesh_cam.recon.glb). Writes objects/<obj>/{audit.json, qa_audit.png} and
qa/audit.json.
"""
import argparse
import json
import re
import shutil

import cv2
import numpy as np
import trimesh
from PIL import Image

from calibrated_scene_reconstruction.assets.physprops import semantic_class
from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR, YCB_ROOT
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.reconstruct import object_specs

MIN_IOU, MAX_SPILL, PART_SPILL = 0.5, 0.2, 0.4
BETTER = 0.1  # a substitute must beat the reconstruction's score by this much


def silhouette(Vc, F, K, shape):
    """Projected mesh, one triangle at a time (cv2.fillPoly over all triangles at once fills by parity and leaves
    pinholes where an even number of them overlap)."""
    uv = np.round(Vc[:, :2] / Vc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(np.int32)
    canvas = np.zeros(shape, np.uint8)
    for tri in uv[F]:
        cv2.fillConvexPoly(canvas, tri, 1)
    return canvas.astype(bool)


def rasterize(Vc, F, K, shape, n=60000):
    """Silhouette and front-surface depth (z) of a camera-frame mesh."""
    S = silhouette(Vc, F, K, shape)
    pts, _ = trimesh.sample.sample_surface(trimesh.Trimesh(Vc, F, process=False), n, seed=0)
    P = np.vstack([pts, Vc])
    P = P[P[:, 2] > 1e-3]
    uv = np.round(P[:, :2] / P[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    Z = np.full(shape, np.inf, np.float32)
    np.minimum.at(Z, (uv[ok, 1], uv[ok, 0]), P[ok, 2].astype(np.float32))
    Z = cv2.erode(Z, np.ones((7, 7), np.uint8))  # splat holes take the front-most neighbour
    Z[~S] = np.inf
    return S, Z


def protrusion(Vw, F, bins=36, ratio=1.3):
    """Faces of a part sticking out of the body about the vertical axis through the base (a handle), or None."""
    z0, h = Vw[:, 2].min(), np.ptp(Vw[:, 2])
    base = Vw[Vw[:, 2] < z0 + 0.3 * h]
    c = np.median(base[:, :2], 0) if len(base) >= 10 else Vw[:, :2].mean(0)
    d = Vw[:, :2] - c
    r, a = np.linalg.norm(d, axis=1), np.arctan2(d[:, 1], d[:, 0])
    k = ((a + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    r95 = np.array([np.percentile(r[k == b], 95) if (k == b).sum() >= 5 else 0.0 for b in range(bins)])
    body = np.median(r95[r95 > 0]) if (r95 > 0).any() else 0.0
    out = (r95 > ratio * body)[k] & (r > 1.15 * body)
    faces = out[F].all(1)
    return F[faces] if body > 0 and faces.sum() >= 4 else None


class Evidence:
    """The photo side of the audit: masks, MoGe depth, the camera."""

    def __init__(self, s: Session):
        geo = s.geometry()
        self.K, self.valid = geo["K"], geo["mask"]
        self.Z = np.where(geo["mask"], geo["points"][..., 2], np.nan)
        self.P = geo["points"]
        self.masks = s.masks()
        self.shape = self.Z.shape
        _, _, self.T_wc = s.world_frame()

    def to_cam(self, Vw):
        return (Vw - self.T_wc[:3, 3]) @ self.T_wc[:3, :3]

    def to_world(self, Vc):
        return Vc @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]

    def mask(self, ids):
        m = np.zeros(self.shape, bool)
        for i in ids:
            m |= self.masks[i]
        return m

    def score(self, Vc, F, M, part_faces=None):
        S, Zm = rasterize(Vc, F, self.K, self.shape)
        tol = 0.02 + 0.02 * np.where(np.isfinite(Zm), Zm, 0)
        with np.errstate(invalid="ignore"):
            front = S & ~M & (self.Z < Zm - tol)  # the photo shows something in front of the mesh
            behind = S & ~M & (self.Z > Zm + tol)  # ... or sees past it
        inner = cv2.erode(S.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        vis_iou = float((S & M).sum() / max(((S & ~front) | M).sum(), 1))
        spill = float((behind & inner).sum() / max((inner & ~front).sum(), 1))
        out = {"visible_iou": round(vis_iou, 3), "spill": round(spill, 3),
               "occluded": round(float(front.sum() / max(S.sum(), 1)), 3)}
        if part_faces is not None:
            Sp = silhouette(Vc, part_faces, self.K, self.shape)
            seen = Sp & ~front
            out["part_px"] = int(seen.sum())
            out["part_spill"] = round(float((Sp & behind).sum() / max(seen.sum(), 1)), 3)
        out["score"] = round(vis_iou - spill - (out.get("part_spill", 0.0) if out.get("part_px", 0) > 30 else 0.0), 3)
        return out, S, behind


def observed_size(ev: Evidence, M):
    """Height above the support and horizontal extent (2-98 %) of the masked points, in the world frame."""
    m = cv2.erode(M.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ev.valid
    Pw = ev.to_world(ev.P[m])
    if len(Pw) < 20:
        return None
    lo, hi = np.percentile(Pw, 2, axis=0), np.percentile(Pw, 98, axis=0)
    return {"top_z": float(hi[2]), "width": float(max(hi[:2] - lo[:2]))}


def audit(s: Session, ev: Evidence, name, ids, label):
    d = s / "objects" / name
    mesh = trimesh.load(d / "mesh_cam.glb", force="mesh")
    Vc, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    Vw = ev.to_world(Vc)
    M = ev.mask(ids)
    part = protrusion(Vw, F)
    res, S, behind = ev.score(Vc, F, M, part)
    obs = observed_size(ev, M)
    res["class"] = semantic_class(label)[0] or "default"
    res["mesh_size_m"] = [round(float(v), 3) for v in np.ptp(Vw, axis=0)]
    if obs:
        h_obs = obs["top_z"] - Vw[:, 2].min()
        res["height_ratio"] = round(float(np.ptp(Vw[:, 2]) / max(h_obs, 1e-3)), 2)
        if res["occluded"] < 0.1:
            res["width_ratio"] = round(float(max(np.ptp(Vw[:, :2], axis=0)) / max(obs["width"], 1e-3)), 2)
    reasons = []
    if res["visible_iou"] < MIN_IOU:
        reasons.append(f"visible IoU {res['visible_iou']:.2f}")
    if res["spill"] > MAX_SPILL:
        reasons.append(f"spill {res['spill']:.2f}")
    if res.get("part_px", 0) > 30 and res["part_spill"] > PART_SPILL:
        reasons.append(f"a protruding part the photo does not show (spill {res['part_spill']:.2f})")
    res["suspect"], res["reasons"] = bool(reasons), reasons
    return res, (M, S, behind, part, Vc)


def candidates(label):
    """Library models of the object's class: YCB scans and Poly Haven props whose names match its class pattern."""
    pattern = semantic_class(label)[0]
    if not pattern:
        return []
    out = []
    if YCB_ROOT.exists():
        out += [("ycb", p.name) for p in sorted(YCB_ROOT.iterdir())
                if (p / "google_16k" / "textured.obj").exists() and re.search(pattern, p.name.replace("_", " "))]
    idx = POLYHAVEN_DIR / "index.json"
    if idx.exists():
        models = json.loads(idx.read_text()).get("models", {})
        out += [("polyhaven", m) for m in sorted(models) if re.search(pattern, m.replace("_", " "))]
    return out


def load_candidate(src, name):
    """(z-up mesh in metres, albedo image or None)."""
    if src == "ycb":
        mesh = trimesh.load(YCB_ROOT / name / "google_16k" / "textured.obj", force="mesh")
    else:
        from calibrated_scene_reconstruction.assets.library import model_file

        mesh = trimesh.load(model_file(name), force="mesh")
        mesh.vertices = np.stack([mesh.vertices[:, 0], -mesh.vertices[:, 2], mesh.vertices[:, 1]], 1)  # y-up glTF
    mat = getattr(mesh.visual, "material", None)
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    return mesh, img


def substitute(s: Session, ev: Evidence, name, ids, label, recon, n_yaw=12):
    """Best library model of the class where the photo shows the object: upright on the reconstruction's support
    height, its visible front at the masked points (centre pushed back by its radius along the view), the observed
    height, yaw searched where it has a part to turn; (score dict, mesh, vertices in the camera frame, albedo)."""
    d = s / "objects" / name
    Vw = ev.to_world(np.asarray(trimesh.load(d / "mesh_cam.glb", force="mesh").vertices))
    z0 = float(Vw[:, 2].min())
    M = ev.mask(ids)
    m = cv2.erode(M.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ev.valid
    Pw = ev.to_world(ev.P[m])
    if len(Pw) < 20:
        return None
    front, top = np.median(Pw[:, :2], 0), float(np.percentile(Pw[:, 2], 98))
    width = float(max(np.percentile(Pw[:, :2], 98, axis=0) - np.percentile(Pw[:, :2], 2, axis=0)))
    view = front - ev.T_wc[:2, 3]
    view /= max(np.linalg.norm(view), 1e-6)
    height = top - z0 if recon["occluded"] < 0.5 else float(np.ptp(Vw[:, 2]))
    best = None
    for src, cname in candidates(label):
        mesh, img = load_candidate(src, cname)
        F = np.asarray(mesh.faces)
        V0 = np.asarray(mesh.vertices, float)
        sc = height / max(np.ptp(V0[:, 2]), 1e-6)
        V0 = (V0 - [*(V0.min(0) + V0.max(0))[:2] / 2, V0[:, 2].min()]) * sc
        part = protrusion(V0, F)
        yaws = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False) if part is not None else [0.0]
        # footprint: as modelled, or the photo's width where the object is not hidden (a squatter cup, a slimmer glass)
        sxys = [1.0] + ([float(np.clip(width / max(min(np.ptp(V0[:, :2], axis=0)), 1e-6), 0.7, 1.3))]
                        if recon["occluded"] < 0.1 and part is None else [])
        for sxy in sxys:
            V1 = V0 * [sxy, sxy, 1.0]
            radius = 0.5 * float(min(np.ptp(V1[:, :2], axis=0)))
            for yaw in yaws:
                R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                for back in (0.6, 1.0):
                    c = front + back * radius * view
                    Vc = ev.to_cam(V1 @ R.T + [c[0], c[1], z0])
                    res, *_ = ev.score(Vc, F, M, protrusion(V1 @ R.T, F))
                    if best is None or res["score"] > best[0]["score"]:
                        best = ({**res, "source": src, "model": cname,
                                 "scale": [round(sc * sxy, 4)] * 2 + [round(sc, 4)], "yaw": round(float(yaw), 3),
                                 "xy": [round(float(v), 4) for v in c], "z": round(z0, 4)}, mesh, Vc, img)
    if best is not None and best[3] is not None:  # the model's texture detail in the photo's colour
        tex = np.asarray(best[3].convert("RGB"), np.float32)
        lum = tex.mean(-1, keepdims=True)
        photo = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"), np.float32)[m].mean(0)
        best = (*best[:3], Image.fromarray(np.clip(lum / max(lum.mean(), 1.0) * photo, 0, 255).astype(np.uint8)))
    return best


def qa_image(s, path, M, S, behind, Vc, part, ev):
    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB")).astype(np.float32)
    qa = img.copy()
    qa[S] = 0.5 * qa[S] + 0.5 * np.array([255, 0, 0])
    qa[M & ~S] = 0.5 * qa[M & ~S] + 0.5 * np.array([0, 0, 255])
    qa[behind] = 0.3 * qa[behind] + 0.7 * np.array([255, 230, 0])
    if part is not None:
        Sp = silhouette(Vc, part, ev.K, ev.shape)
        edge = Sp & ~cv2.erode(Sp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        qa[edge] = [0, 255, 0]
    ys, xs = np.nonzero(S | M)
    if len(ys):
        pad = 80
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, qa.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, qa.shape[1])
        crop = np.concatenate([img[y0:y1, x0:x1], qa[y0:y1, x0:x1]], 1)
        Image.fromarray(crop.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fix", action="store_true", help="replace suspect reconstructions by the best library model")
    args = ap.parse_args()
    s = Session(args.session)
    sg = s.scene_graph
    ev = Evidence(s)
    report = s.read_json("qa/audit.json") if args.only and (s / "qa" / "audit.json").exists() else {}
    specs = object_specs(sg)
    masks = {n: ev.mask(i) for n, i in specs.items()}
    for name, ids in specs.items():
        d = s / "objects" / name
        if (args.only and name not in args.only) or not (d / "mesh_cam.glb").exists():
            continue
        spec = {**sg.get("static_objects", {}), **sg.get("dynamic_objects", {})}[name]
        label = spec.get("_label") or name
        if (d / "mesh_cam.recon.glb").exists():  # audit the reconstruction, not an earlier substitute
            shutil.copy(d / "mesh_cam.recon.glb", d / "mesh_cam.glb")
            if (d / "albedo.recon.png").exists():
                shutil.copy(d / "albedo.recon.png", d / "albedo.png")
        res, (M, S, behind, part, Vc) = audit(s, ev, name, ids, label)
        for other, Mo in masks.items():  # one thing under two prompts ("cup" and "glass"): the review merges them
            if other != name and (M & Mo).sum() > 0.5 * (M | Mo).sum():
                res["reasons"].append(f"same photo pixels as {other}")
                res["suspect"] = True
        qa_image(s, d / "qa_audit.png", M, S, behind, Vc, part, ev)
        if (d / "retrieved.json").exists():  # recon.retrieve already stands a library model in for it
            res["retrieved"] = s.read_json(f"objects/{name}/retrieved.json")["model"]
        elif res["suspect"] and args.fix:
            best = substitute(s, ev, name, ids, label, res)
            if best is not None:
                res["substitute"] = dict(best[0])
                if best[0]["score"] >= res["score"] + BETTER:
                    _, mesh, Vc2, img = best
                    shutil.copy(d / "mesh_cam.glb", d / "mesh_cam.recon.glb")
                    if (d / "albedo.png").exists():
                        shutil.copy(d / "albedo.png", d / "albedo.recon.png")
                    out = mesh.copy()
                    out.vertices = Vc2
                    out.export(d / "mesh_cam.glb")
                    if img is not None:
                        img.convert("RGB").save(d / "albedo.png")
                    res["substituted"] = True
                    _, S2, behind2 = ev.score(Vc2, np.asarray(mesh.faces), M)
                    qa_image(s, d / "qa_audit_fix.png", M, S2, behind2, Vc2, None, ev)
        s.write_json(f"objects/{name}/audit.json", res)
        report[name] = res
        line = "SUSPECT: " + "; ".join(res["reasons"]) if res["suspect"] else "ok"
        sub = res.get("substitute")
        if res.get("retrieved"):
            line += f" -> replaced by retrieval: {res['retrieved']}"
        elif sub:
            line += (f" -> {'substituted' if res.get('substituted') else 'kept (no better substitute)'}: "
                     f"{sub['source']} {sub['model']} score {sub['score']:.2f} vs {res['score']:.2f}")
        print(f"[audit] {name} ({res['class']}): visible IoU {res['visible_iou']:.2f}, spill {res['spill']:.2f}, "
              f"occluded {res['occluded']:.2f}, size {res['mesh_size_m']} | {line}")
    s.write_json("qa/audit.json", report)


if __name__ == "__main__":
    main()
