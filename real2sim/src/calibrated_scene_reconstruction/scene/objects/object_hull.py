"""The task objects' size and place from the posed frames' silhouettes: a visual hull, no depth.

    envs/recon/bin/python -m calibrated_scene_reconstruction.recon.object_hull sessions/<name> [--prompts peg="white peg",...] [--apply]
    (after recon.posed_frames select: the frames' poses at the scene's scale)

A small, white, textureless thing is a few dozen
pixels in the photo and next to nothing in the splat (its Gaussians on the table mostly faint: a cluster of 1-3).
Its silhouette, though, SAM 3 finds in a frame's crop round where it projects. Per dynamic object: from up to
`--frames` posed frames that see its place within 2.5 m, a crop round its projection, the SAM 3 instance of its prompt
that holds the projected centre (else the nearest within 40 px); a 3 mm grid round its place carved by them -- a cell
stays where it falls in the silhouette in `--agree` of the frames that see it (a hand in front of it in a few) --, the
largest piece standing on the table its body: its box (1st-99th percentile) its size, the box's middle its place;
twice, the second time round the first's place. Writes splat/object_hull.json and qa/splat/object_hull.jpg (per
object: a few crops with the silhouette and the hull projected); `--apply` writes size and xy into the scene graph's
dynamic_objects.<name>.fix (sim.scene scales the reconstruction to the box) with `_object_hull`."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

from calibrated_scene_reconstruction.scene.session import Session

VOX = 0.003


def project(Xw, K, R, c):
    Xc = (Xw - c) @ R  # world -> camera (R: camera axes in the world)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return u, v, z


class CropSam3:
    """SAM 3 in a frame's crop round where a thing projects: the instance of the prompt that holds the projected
    centre, else the one whose middle is nearest it within `near` px (40); a full-frame mask, or None."""

    def __init__(self, crop=320, threshold=0.2, near=40.0):
        from transformers import Sam3Model, Sam3Processor

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.dev).eval()
        self.proc = Sam3Processor.from_pretrained("facebook/sam3")
        self.crop, self.threshold, self.near = crop, threshold, near

    def instances(self, img, prompt, uc, vc, crop):
        """Every instance of the prompt in a crop round (uc, vc): [(full-frame mask, score)]."""
        H, W = img.shape[:2]
        cw, ch = min(crop, W), min(crop, H)
        x0, y0 = int(np.clip(uc - cw // 2, 0, W - cw)), int(np.clip(vc - ch // 2, 0, H - ch))
        inp = self.proc(images=Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]), text=prompt, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            o = self.model(**inp)
        r = self.proc.post_process_instance_segmentation(o, threshold=self.threshold, mask_threshold=0.5,
                                                         target_sizes=inp.get("original_sizes").tolist())[0]
        out = []
        for m, sc in zip(r["masks"], r["scores"]):
            full = np.zeros((H, W), bool)
            full[y0:y0 + ch, x0:x0 + cw] = m.cpu().numpy().astype(bool)
            out.append((full, float(sc)))
        return out

    def __call__(self, img, prompt, uc, vc, union=None, exclude=None, near=None, with_score=False, crop=None):
        """`union` (px): every instance whose middle lies within it of the centre, together (a mark in pieces). A crop
        of None: the whole frame (a rig over the table spans it). `exclude` (a full-frame mask: another task object's
        silhouette in this frame): an instance half on it is that thing, passed over; `near` the search's reach then
        (px). `with_score`: (mask, SAM 3's score of the instance taken) -- None, 0.0 when none."""
        H, W = img.shape[:2]
        size = crop or self.crop  # `crop`: this call's (a thing filling a close-up overflows the default)
        cw, ch = (W, H) if size is None else (min(size, W), min(size, H))
        x0, y0 = int(np.clip(uc - cw // 2, 0, W - cw)), int(np.clip(vc - ch // 2, 0, H - ch))
        crop = Image.fromarray(img[y0:y0 + ch, x0:x0 + cw])
        inp = self.proc(images=crop, text=prompt, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            o = self.model(**inp)
        r = self.proc.post_process_instance_segmentation(o, threshold=self.threshold, mask_threshold=0.5,
                                                         target_sizes=inp.get("original_sizes").tolist())[0]
        inst = [(m.cpu().numpy().astype(bool), float(sc)) for m, sc in zip(r["masks"], r["scores"])]
        best, score, bd = None, 0.0, self.near if near is None else near
        iu, iv = int(round(uc - x0)), int(round(vc - y0))
        if exclude is not None:
            ex = exclude[y0:y0 + ch, x0:x0 + cw]
            inst = [(m, sc) for m, sc in inst if (m & ex).sum() < 0.5 * max(int(m.sum()), 1)]
        if union is not None:
            for m, sc in inst:
                ys, xs = np.nonzero(m)
                if len(xs) and np.hypot(xs.mean() - (uc - x0), ys.mean() - (vc - y0)) < union:
                    best, score = (m if best is None else best | m), max(score, sc)
            inst = []
        for m, sc in inst:
            if 0 <= iu < cw and 0 <= iv < ch and m[iv, iu]:
                best, score = m, sc
                break
            ys, xs = np.nonzero(m)
            if len(xs):
                d = float(np.hypot(xs.mean() - (uc - x0), ys.mean() - (vc - y0)))
                if d < bd:
                    best, score, bd = m, sc, d
        if best is None:
            return (None, 0.0) if with_score else None
        full = np.zeros((H, W), bool)
        full[y0:y0 + ch, x0:x0 + cw] = best
        return (full, score) if with_score else full


def photo_start(s, z, ids, z0):
    """An object's place from the photo's own posed camera (the photo is a frame of the walk: COLMAP knows it): its
    mask's lowest band -- where it touches the support -- cast onto the support's top, then half that band's width further
    along the view."""
    kf = s / "capture" / "keyframes.json"
    names = [str(n) for n in z["names"]]
    if not kf.exists():
        return None
    nm = f"0_{json.loads(kf.read_text())['photo']:05d}.jpg"
    if nm not in names or not ids:
        return None
    i = names.index(nm)
    m = np.any(s.masks()[ids], 0)
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return None
    # An area percentile is not a contact band: a broad canopy can put its
    # lowest3% of pixels well above the pot's foot. Use the actual lower
    # boundary for the contact ray; estimate foot width separately because a
    # rounded base has only a narrow tangent at its lowest image row.
    extent = float(np.ptp(ys))
    low = ys >= ys.max()-max(1.,.003*extent)
    foot = ys >= ys.max()-max(2.,.03*extent)
    u, v = xs[low].mean(), ys[low].mean()
    K, R, C = z["K"][i], z["R_world_cam"][i], z["centre"][i]
    # Perception commonly uses a downsampled primary image, whereas COLMAP
    # cameras retain full source resolution. Both pixel centres and widths
    # must be expressed in the camera's domain before casting a ray.
    camera_width, camera_height = z['size'][i]
    sx, sy = camera_width/m.shape[1], camera_height/m.shape[0]
    u, v = (u+.5)*sx-.5, (v+.5)*sy-.5
    d = R @ np.linalg.solve(K, [u, v, 1.0])
    if d[2] >= -1e-6:
        return None
    p = C + (z0 - C[2]) / d[2] * d
    # A crown, handle or lid can be much wider than the visible base. It must
    # not push the initial contact point behind the support.
    width = np.ptp(xs[foot]) * sx * float(((p-C)@R)[2]) / K[0][0]
    h = np.array([d[0], d[1]]) / np.hypot(d[0], d[1])
    return (p[:2] + 0.5 * width * h).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--prompts", default="", help="name=prompt,... (default: the object's label)")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--agree", type=float, default=0.85)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--seed", default="", help="name=x;y,...: where to look first (the twin's place is the default)")
    ap.add_argument("--only", default="", help="comma-separated names (default: every dynamic object)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--from-config", action="store_true",
                    help="the objects of the scene graph's fit requests not yet measured (the task_zone stage)")
    a = ap.parse_args()
    s = Session(a.session)
    z = np.load(s / "parsing" / "posed_frames" / "select.npz", allow_pickle=True)
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    names, Ks, Rs, Cs = [str(n) for n in z["names"]], z["K"], z["R_world_cam"], z["centre"]
    W, H = (int(v) for v in z["size"][0])
    sg = s.scene_graph
    prompts = dict(p.split("=", 1) for p in a.prompts.split(",") if "=" in p)
    seeds = {k: [float(x) for x in v.split(";")] for k, v in (p.split("=", 1) for p in a.seed.split(",") if "=" in p)}
    if a.from_config:  # what the review asked a kit for and nothing measured yet: its box first, the fit from it
        req = {n: e["fit"] for n, e in sg.get("dynamic_objects", {}).items()
               if (e.get("fit") or {}).get("kit") and not e["fit"].get("accepted") and not (e.get("fix") or {}).get("size")
               and (e.get("fix") or {}).get("kit") != e["fit"]["kit"]}
        if not req:
            print("done: no fit request left to measure")
            return
        a.only = ",".join(req)
        prompts = {**{n: f["prompt"] for n, f in req.items() if f.get("prompt")}, **prompts}
    cams = s / "parsing" / "posed_frames" / "cameras.npz"
    zc = np.load(cams) if cams.exists() else None
    top = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])
    old = s / "splat" / "object_hull.json"
    T_wc = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float)
    silhouette = CropSam3(a.crop)

    res, tiles = (json.loads(old.read_text()) if old.exists() and a.only else {}), []
    imgs = {}
    for name, spec in sg.get("dynamic_objects", {}).items():
        if a.only and name not in a.only.split(","):
            continue
        prompt = prompts.get(name, spec.get("_label", name))
        mesh = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
        V = np.asarray(mesh.vertices) @ T_wc[:3, :3].T + T_wc[:3, 3]
        c = (V.min(0) + V.max(0)) / 2
        ph = photo_start(s, zc, spec.get("ids", []), top) if zc is not None else None
        if name in seeds:  # the twin's place off by more than a crop's reach
            c[:2] = seeds[name]
        elif ph is not None:  # Use the posed camera's support-plane estimate.
            c[:2] = ph
        z_lo = float(V[:, 2].min())
        for rnd in range(2):
            seen = []
            for i in range(len(names)):
                u, v, d = project(c[None], Ks[i], Rs[i], Cs[i])
                if d[0] > 0.2 and np.linalg.norm(c - Cs[i]) < 2.5 and 20 < u[0] < W - 20 and 20 < v[0] < H - 20:
                    seen.append(i)
            pick = [seen[j] for j in np.unique(np.linspace(0, len(seen) - 1, min(a.frames, len(seen))).astype(int))] \
                if seen else []
            sil = {}
            for i in pick:
                if names[i] not in imgs:
                    imgs[names[i]] = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
                u, v, _ = project(c[None], Ks[i], Rs[i], Cs[i])
                m = silhouette(imgs[names[i]], prompt, u[0], v[0])
                if m is not None:
                    sil[i] = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if len(sil) < 6:
                res[name] = {"why": f"silhouettes in {len(sil)} of {len(pick)} frames"}
                break
            half = 0.12
            gx = np.arange(c[0] - half, c[0] + half, VOX)
            gy = np.arange(c[1] - half, c[1] + half, VOX)
            gz = np.arange(z_lo - 0.03, z_lo + 0.22, VOX)
            G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
            n_seen = np.zeros(len(G), np.int32)
            n_in = np.zeros(len(G), np.int32)
            for i, m in sil.items():
                u, v, d = project(G, Ks[i], Rs[i], Cs[i])
                ok = (d > 0.1) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
                ui, vi = u[ok].astype(int), v[ok].astype(int)
                n_seen[ok] += 1
                n_in[np.flatnonzero(ok)[m[vi, ui]]] += 1
            keep = (n_seen >= 6) & (n_in >= a.agree * n_seen)
            occ = keep.reshape(len(gx), len(gy), len(gz))
            lab, nl = ndimage.label(occ)
            if nl == 0:
                res[name] = {"why": "the hull is empty"}
                break
            sizes = ndimage.sum(occ, lab, range(1, nl + 1))
            body = lab == (1 + int(np.argmax(sizes)))
            Pb = G[body.reshape(-1)]
            lo, hi = np.percentile(Pb, 1, 0), np.percentile(Pb, 99, 0)
            c = np.r_[(lo[:2] + hi[:2]) / 2, (lo[2] + hi[2]) / 2]
            z_lo = float(lo[2])
            # the footprint's own axes (a peg lying across the axes: its box is the bounding one of a diagonal)
            Q2 = Pb[:, :2] - Pb[:, :2].mean(0)
            _, _, Vt = np.linalg.svd(Q2, full_matrices=False)
            tq = Q2 @ Vt.T
            ext = np.percentile(tq, 99, 0) - np.percentile(tq, 1, 0)
            oriented = {"length": round(float(ext[0]), 4), "width": round(float(ext[1]), 4),
                        "yaw_deg": round(float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))), 1)}
            res[name] = {"size": (hi - lo).round(4).tolist(), "xy": c[:2].round(4).tolist(), "bottom_z": round(z_lo, 4),
                         "oriented": oriented,
                         "frames": len(sil), "voxels": int(body.sum()), "prompt": prompt,
                         "twin_size": np.ptp(V, axis=0).round(4).tolist()}
        if "size" in res.get(name, {}):
            r = res[name]
            print(f"[hull] {name} ('{prompt}'): {r['size'][0] * 100:.1f} x {r['size'][1] * 100:.1f} x "
                  f"{r['size'][2] * 100:.1f} cm at ({r['xy'][0]:.3f}, {r['xy'][1]:.3f}), bottom {r['bottom_z']:.3f} m, "
                  f"along its axes {r['oriented']['length'] * 100:.1f} x {r['oriented']['width'] * 100:.1f} cm at "
                  f"{r['oriented']['yaw_deg']:.0f} deg, "
                  f"{r['frames']} silhouettes; the twin had {' x '.join(f'{x * 100:.1f}' for x in r['twin_size'])} cm",
                  flush=True)
            for i in list(sil)[:: max(1, len(sil) // 4)][:4]:  # QA crops: silhouette green, hull red
                img = imgs[names[i]].copy()
                u, v, _ = project(Pb, Ks[i], Rs[i], Cs[i])
                img[sil[i]] = (0.5 * img[sil[i]] + [0, 120, 0]).astype(np.uint8)
                ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                img[v[ok].astype(int), u[ok].astype(int)] = (230, 40, 40)
                uc, vc, _ = project(c[None], Ks[i], Rs[i], Cs[i])
                x0, y0 = int(np.clip(uc[0] - 120, 0, W - 240)), int(np.clip(vc[0] - 120, 0, H - 240))
                t = cv2.resize(img[y0:y0 + 240, x0:x0 + 240], (240, 240))
                cv2.putText(t, name, (4, 16), 0, 0.45, (255, 255, 255), 1)
                tiles.append(t)
        else:
            print(f"[hull] {name}: {res.get(name, {}).get('why')}", flush=True)
    (s / "splat").mkdir(exist_ok=True)
    (s / "splat" / "object_hull.json").write_text(json.dumps(res, indent=1) + "\n")
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        (s / "qa" / "splat").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(s / "qa" / "splat" / "object_hull.jpg"),
                    np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])[:, :, ::-1])
    if a.apply:
        g = s.config("scene_graph.json")
        for name, r in res.items():
            if "size" in r:
                o = r["oriented"]  # the box along the footprint's own axes (a peg lying across them)
                g["dynamic_objects"][name]["fix"] = {**(g["dynamic_objects"][name].get("fix") or {}),
                                                     "size": [o["length"], o["width"], r["size"][2]], "xy": r["xy"],
                                                     "size_yaw_deg": o["yaw_deg"]}
        g["_object_hull"] = {"note": "the task objects' boxes and places from the posed frames' silhouettes "
                                     "(recon.object_hull)", "measured": res}
        (s.config_dir / "scene_graph.json").write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}")


if __name__ == "__main__":
    main()
