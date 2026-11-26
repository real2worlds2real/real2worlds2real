"""The task objects' looks from the walk's frames: each fitted kit (recon.object_fit) unwrapped (xatlas) and its texture
baked from the frames its silhouettes were found in -- the real colours, print and wear in place of one flat colour.

    python -m calibrated_scene_reconstruction.recon.object_texture sessions/<name> [--objects peg,cup] [--res 512] [--apply]

Per object: its kit's visual parts as one mesh where the fit put it in the frames' world (on frames_support_z), unwrapped
into an atlas; per texel its point and normal; per frame of recon.object_fit's cache (splat/object_fit/<name>.npz:
the frames, SAM 3 silhouettes and their windows) the texels that face the camera, are the front-most of the object at
their pixel (a painter's triangle buffer) and fall inside the eroded silhouette (what stands in front of the object --
a rail, a hand -- is not the object) sample the frame; a texel's colour is the weighted mean of its samples nearest
their median (weights: the view's obliqueness, the frame's sharpness round the object). Colours are albedos as
recon.frame_colour reads a frame (a clip is a white of WHITE). Texels no frame saw (a cup's foot, a peg's underside)
are filled from the atlas round them.

Writes objects/<name>/look/{mesh.npz (V, F, UV in the kit's own frame), albedo.png, coverage.png} and
qa/object_texture/<name>.jpg (the frames' crops beside the textured mesh's projection into them); --apply sets
dynamic_objects.<name>.fix.look in the scene graph (sim.scene builds the textured mesh in place of the kit's parts;
the kit's colliders stay).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.assets import kits
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.frame_color import WHITE, to_linear
from calibrated_scene_reconstruction.scene.objects.object_fit import frames_support_z
from calibrated_scene_reconstruction.scene.objects.object_hull import project

ERODE = 3      # px the silhouette is eroded by before a texel inside it samples the frame
MIN_COS = 0.2  # a texel seen more obliquely than this (cos of the view to its normal) is not sampled
KEEP = 0.6     # share of a texel's samples nearest their median that are averaged


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def kit_mesh(fix):
    """The kit's visual parts as one mesh in its own frame (z up from its base)."""
    if fix.get("model"):
        from calibrated_scene_reconstruction.scene.objects.blender.export import asset
        kit = asset(fix)
    else:
        kit = kits.build(fix["kit"], fix.get("params", {}))
    V, F, n = [], [], 0
    for p in kit["parts"]:
        V.append(np.asarray(p["V"], float))
        F.append(np.asarray(p["F"], int) + n)
        n += len(p["V"])
    return np.concatenate(V), np.concatenate(F)


def pose(fix, z0):
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    return rotation(fix), np.array([*fix.get("xy", [0, 0]), fix["support_z"] if fix.get("model") else z0 + 0.002])


def unwrap(V, F):
    import xatlas

    # xatlas uses absolute geometric tolerances. Millimetre-scale facets can
    # otherwise silently receive UV=(0,0), even on a valid closed metric mesh.
    # The normalized copy is used only by the atlas; exported metres stay exact.
    extent = float(np.ptp(V, axis=0).max())
    if not np.isfinite(V).all() or not np.isfinite(extent) or extent <= 0:
        raise ValueError('Texture unwrap needs finite, nonzero geometry')
    normalized = (V - V.min(axis=0)) / extent
    # Some valid thin bevels survive unit normalization but still fall below
    # xatlas's absolute tolerance. Retry in a larger atlas-only coordinate
    # frame. Neither source vertices nor physical dimensions may be altered.
    for atlas_scale in (1., 1000.):
        vmap, Fn, UV = xatlas.parametrize((normalized * atlas_scale).astype(np.float32), F.astype(np.uint32))
        if Fn.shape != F.shape or not np.array_equal(V[vmap][Fn], V[F]):
            raise ValueError('Texture unwrap changed source triangles or their ordering')
        try:
            atlas_coverage(V[vmap], Fn, UV)
        except AtlasCollapseError:
            if atlas_scale == 1.:
                continue
            raise
        return V[vmap], Fn.astype(int), UV.astype(float)


class AtlasCollapseError(ValueError):
    """Finite UVs discarded a significant part of valid source geometry."""


def atlas_coverage(V, F, UV):
    """Reject silent UV collapse of geometrically significant source surfaces."""
    triangle = V[F]
    area = np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0],
                                   triangle[:, 2]-triangle[:, 0]), axis=1)
    chart = UV[F]
    a, b = chart[:, 1]-chart[:, 0], chart[:, 2]-chart[:, 0]
    uv_area = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    if not np.isfinite(UV).all() or area.sum() <= 0:
        raise ValueError('Texture atlas has invalid coordinates or zero source area')
    missing = uv_area <= 1e-14
    fraction = float(area[missing].sum()/area.sum())
    largest = float(area[missing].max(initial=0)/area.sum())
    if fraction > 1e-4 or largest > 1e-5:
        raise AtlasCollapseError(f'Texture atlas collapsed source surface area: {fraction:.6%}')
    return {'collapsed_surface_fraction': fraction,
            'largest_collapsed_face_fraction': largest, 'pass': True}


def texels(V, F, UV, res):
    """Per covered texel: its (row, col), triangle, point (barycentric) -- the atlas rasterised."""
    rows, cols, tris, P = [], [], [], []
    T = UV[F] * res - 0.5  # texel centres at integer + 0.5
    for t, (a, b, c) in enumerate(T):
        lo = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        xs, ys = np.meshgrid(np.arange(max(lo[0], 0), min(hi[0], res - 1) + 1),
                             np.arange(max(lo[1], 0), min(hi[1], res - 1) + 1))
        q = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
        M = np.array([b - a, c - a]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        w = np.linalg.solve(M, (q - a).T).T
        inside = (w[:, 0] >= -0.02) & (w[:, 1] >= -0.02) & (w.sum(1) <= 1.02)
        if not inside.any():
            continue
        w = w[inside]
        bary = np.c_[1 - w.sum(1), w]
        rows.append(q[inside, 1].astype(int)), cols.append(q[inside, 0].astype(int))
        tris.append(np.full(inside.sum(), t)), P.append(bary @ V[F[t]])
    r, c, t, P = (np.concatenate(x) for x in (rows, cols, tris, P))
    _, first = np.unique(r * res + c, return_index=True)  # a texel on two charts' seam: once
    return r[first], c[first], t[first], P[first]


def push_pull(img, known):
    """Every unknown texel from the known ones round it: averaged down a pyramid, filled back up."""
    if known.all() or not known.any():
        return img
    w = known.astype(np.float32)
    small = cv2.resize(img * w[..., None], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ws = cv2.resize(w, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    coarse = push_pull(small / np.maximum(ws[..., None], 1e-6), ws > 1e-6) if min(small.shape[:2]) > 1 else small
    up = cv2.resize(coarse, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return np.where(known[..., None], img, up)


def front_buffer(Vw, F, K, R, c, shape):
    """Perspective-correct per-pixel depth and triangle ids (visibility approach adapted from
    real2sim reference_texture.py). Intersecting or slanted triangles cannot be sorted by mean depth."""
    u, v, z = project(Vw, K, R, c)
    ids = np.full(shape, -1, np.int32)
    depth = np.full(shape, np.inf, np.float32)
    H, W = shape
    for t, tri in enumerate(F):
        if (z[tri] <= 0.05).any():
            continue
        points = np.c_[u[tri], v[tri]]
        lo = np.maximum(np.floor(points.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(points.max(0)).astype(int), [W-1, H-1])
        if (hi < lo).any():
            continue
        a, b, cc = points
        M = np.array([b-a, cc-a]).T
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        q = np.c_[xs.ravel(), ys.ravel()]
        w = (q-a) @ np.linalg.inv(M).T
        bary = np.c_[1-w.sum(1), w]
        inside = (bary >= -1e-7).all(1)
        inverse = bary @ (1/z[tri])
        values = np.where(inverse > 0, 1/np.maximum(inverse, 1e-12), np.inf)
        update = inside & (values < depth[ys.ravel(), xs.ravel()])
        yy, xx = ys.ravel()[update], xs.ravel()[update]
        depth[yy, xx], ids[yy, xx] = values[update], t
    return ids, depth


def sharpness(img, box):
    x0, y0, x1, y1 = (int(v) for v in box)
    g = cv2.cvtColor(img[max(y0, 0):y1, max(x0, 0):x1], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g.size else 0.0


def bake(s, name, fix, res, z0, *, normalize_exposure=True):
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    cache = np.load(s / "splat" / "object_fit" / f"{name}.npz", allow_pickle=True)
    V, F = kit_mesh(fix)
    V, F, UV = unwrap(V, F)
    R0, t0 = pose(fix, z0)
    Vw = V @ R0.T + t0
    N = np.cross(Vw[F[:, 1]] - Vw[F[:, 0]], Vw[F[:, 2]] - Vw[F[:, 0]])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    r, cidx, tri, P = texels(V, F, UV, res)
    Pw, Nt = P @ R0.T + t0, N[tri]
    from .mask_cache import read_masks
    source_masks = read_masks(cache)
    if fix.get('model'):
        # Appearance uses the same reviewed visibility as geometry: a source
        # occluder must not be baked onto an otherwise accurate object surface.
        from .blender.evidence import load as source_evidence
        _, observed_frames, source_masks, observed_windows = source_evidence(s, name)
        if (not np.array_equal(observed_frames, cache['frames'])
                or not np.array_equal(observed_windows, cache['windows'])):
            raise ValueError('Texture observations do not match the current geometry source cache')
    samples, weights, crops = [], [], []
    for k, i in enumerate(cache["frames"]):
        i = int(i)
        K, R, c = z["K"][i], z["R_world_cam"][i], z["centre"][i]
        img = np.asarray(Image.open(img_dir / str(z["names"][i])).convert("RGB"))
        H, W = img.shape[:2]
        x0, y0, cw, ch, f = cache["windows"][k]
        m = np.asarray(source_masks[k], np.uint8)
        m = cv2.erode(m, np.ones((2 * ERODE + 1, 2 * ERODE + 1), np.uint8))
        ids, depths = front_buffer(Vw, F, K, R, c, (H, W))
        u, v, d = project(Pw, K, R, c)
        view = (Pw - c) / np.linalg.norm(Pw - c, axis=1, keepdims=True)
        cos = -(view * Nt).sum(1)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (cos > MIN_COS) & (d > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        front = np.full(len(Pw), -1)
        front[ok] = ids[vi[ok], ui[ok]]
        visible_depth = np.full(len(Pw), np.inf)
        visible_depth[ok] = depths[vi[ok], ui[ok]]
        ok &= (front == tri) | (np.isfinite(visible_depth) & (visible_depth >= d - 0.001))
        mu, mv = ((u - x0) * f).astype(int), ((v - y0) * f).astype(int)
        inm = (mu >= 0) & (mu < m.shape[1]) & (mv >= 0) & (mv < m.shape[0])
        ok &= inm
        ok[ok] &= m[mv[ok], mu[ok]] > 0
        if ok.sum() < 20:
            continue
        box = (u[ok].min() - 8, v[ok].min() - 8, u[ok].max() + 8, v[ok].max() + 8)
        sh = sharpness(img, box)
        col = np.full((len(Pw), 3), np.nan, np.float32)
        col[ok] = WHITE * to_linear(img[vi[ok], ui[ok]] / 255.0)
        samples.append(col), weights.append(np.where(ok, cos ** 2, 0.0) * sh)
        crops.append((i, img, box))
    if not samples:
        return None
    S, Wt = np.stack(samples, 1), np.stack(weights, 1)  # texels x frames (x 3)
    Wt = Wt / max(Wt.max(), 1e-12)
    photometric = None
    if normalize_exposure:
        from .photometric import align
        S, photometric = align(S, Wt)
        photometric['source_frames'] = [int(row[0]) for row in crops]
    seen = Wt.sum(1) > 0
    med = np.nanmedian(np.where(Wt[..., None] > 0, S, np.nan)[seen], axis=1)
    dev = np.linalg.norm(S[seen] - med[:, None], axis=2)
    dev = np.where(Wt[seen] > 0, dev, np.inf)
    cut = np.nanquantile(np.where(np.isfinite(dev), dev, np.nan), KEEP, axis=1)
    w = Wt[seen] * (dev <= cut[:, None])
    col = (np.nan_to_num(S[seen]) * w[..., None]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-12)
    tex = np.zeros((res, res, 3), np.float32)
    cov = np.zeros((res, res), np.uint8)
    tex[r[seen], cidx[seen]] = col
    cov[r[seen], cidx[seen]] = 255
    srgb = (push_pull(to_srgb(tex), cov > 0) * 255).astype(np.uint8)  # unseen texels, gutters: from round them
    return {"V": V, "F": F, "UV": UV, "albedo": srgb[::-1], "coverage": cov[::-1],  # PNG rows run down, UV v up
            "seen": float(seen.mean()), "frames": len(samples), "Vw": Vw, "crops": crops,
            "photometric": photometric}


def textured_projection(V, F, UV, albedo, K, R, centre, background):
    """Unlit source-texture reprojection with depth-tested, perspective-correct UVs.

    This is an appearance diagnostic, not a relit scene render. In particular,
    back surfaces cannot paint over nearer surfaces on finely tessellated meshes.
    """
    ids, _ = front_buffer(V, F, K, R, centre, background.shape[:2])
    rows, cols = np.nonzero(ids >= 0)
    result = background.copy()
    if not len(rows):
        return result
    faces = F[ids[rows, cols]]
    u, v, depth = project(V, K, R, centre)
    screen = np.stack([u[faces], v[faces]], axis=-1)
    a, b, c = screen[:, 0], screen[:, 1], screen[:, 2]
    matrix = np.stack([b-a, c-a], axis=-1)
    w = np.linalg.solve(matrix, (np.c_[cols, rows]-a)[..., None])[..., 0]
    bary = np.c_[1-w.sum(1), w] / depth[faces]
    bary /= bary.sum(1, keepdims=True)
    uv = (UV[faces] * bary[..., None]).sum(1)
    height, width = albedo.shape[:2]
    # PNG rows run down, UV v up. Bilinear filtering matches native texture
    # sampling; face-centre colours lose detail on long triangles.
    x = np.clip(uv[:, 0]*width-.5, 0, width-1).astype(np.float32)
    y = np.clip((1-uv[:, 1])*height-.5, 0, height-1).astype(np.float32)
    # cv2.remap has a signed-short dimension limit; keep the sample map bounded.
    for start in range(0, len(rows), 16000):
        end = min(start+16000, len(rows))
        color = cv2.remap(albedo, x[start:end, None], y[start:end, None],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]
        result[rows[start:end], cols[start:end]] = color
    return result


def qa_sheet(out, name, b, z):
    """Source crops beside depth-tested unlit texture reprojections."""
    tiles = []
    for i, img, box in b["crops"][:: max(1, len(b["crops"]) // 6)][:6]:
        x0, y0, x1, y1 = (int(v) for v in box)
        x0, y0 = max(x0 - 20, 0), max(y0 - 20, 0)
        x1, y1 = min(x1 + 20, img.shape[1]), min(y1 + 20, img.shape[0])
        real = img[y0:y1, x0:x1]
        K, R, c = z["K"][i], z["R_world_cam"][i], z["centre"][i]
        crop_K = K.copy()
        crop_K[0, 2] -= x0
        crop_K[1, 2] -= y0
        ren = textured_projection(b['Vw'], b['F'], b['UV'], b['albedo'],
                                  crop_K, R, c, real.copy() // 3)
        h = 200
        sc = h / max(real.shape[0], 1)
        tiles.append(np.hstack([cv2.resize(real, (max(int(real.shape[1] * sc), 1), h)),
                                cv2.resize(ren, (max(int(real.shape[1] * sc), 1), h))]))
    if tiles:
        Wd = max(t.shape[1] for t in tiles)
        sheet = np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, Wd - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles])
        Image.fromarray(sheet).save(out / f"{name}.jpg", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--objects", default="", help="comma list (default: every dynamic object fitted as a kit)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    s = Session(a.session)
    sg = s.scene_graph
    z0 = frames_support_z(s)
    if z0 is None:
        z0 = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])
    want = [n for n in a.objects.split(",") if n]
    todo = {n: o["fix"] for n, o in sg.get("dynamic_objects", {}).items()
            if (o.get("fix") or {}).get("kit") and (not want or n in want)
            and (s / "splat" / "object_fit" / f"{n}.npz").exists()}
    if not todo:
        print("done: no fitted kit with frames to bake from")
        return
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    qa = s / "qa" / "object_texture"
    qa.mkdir(parents=True, exist_ok=True)
    looks = {}
    for name, fix in todo.items():
        b = bake(s, name, fix, a.res, z0)
        if b is None:
            print(f"[object_texture] {name}: no frame to sample", flush=True)
            continue
        d = s / "objects" / name / "look"
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "mesh.npz", V=b["V"], F=b["F"], UV=b["UV"])
        Image.fromarray(b["albedo"]).save(d / "albedo.png")
        Image.fromarray(b["coverage"]).save(d / "coverage.png")
        qa_sheet(qa, name, b, z)
        looks[name] = {"mesh": str((d / "mesh.npz").relative_to(s.dir)), "albedo": str((d / "albedo.png").relative_to(s.dir)),
                       "_object_texture": f"recon.object_texture: {b['frames']} frames, {100 * b['seen']:.0f} % of the "
                                          f"surface seen"}
        print(f"[object_texture] {name}: {b['frames']} frames, {100 * b['seen']:.0f} % of its surface seen", flush=True)
    if a.apply and looks:
        p = s.config_dir / "scene_graph.json"
        g = json.loads(p.read_text())
        for name, look in looks.items():
            g["dynamic_objects"][name].setdefault("fix", {})["look"] = look
        p.write_text(json.dumps(g, indent=1, ensure_ascii=False))
        print(f"done: looks -> {p}")


if __name__ == "__main__":
    main()
