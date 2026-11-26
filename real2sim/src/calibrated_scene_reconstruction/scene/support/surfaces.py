"""Surface review: what the photo puts on each surface the robot works at, for the agent to decide its material.

    python -m calibrated_scene_reconstruction.recon.surfaces sessions/<name> [--radius 2.5]

A photo texture is right from the photo camera only: its light, its cast shadows and highlights, whatever stood in
front of the surface (objects, the real robot, cables, the inpainted holes they left) are baked into the colour, and
from the robot's head camera -- across the table, looking down -- they read as wrong material. So every surface
within --radius (m, horizontally; default 4) of the task support (the support's own faces and the floor always; windows and the
ceiling never) is reviewed as a group (a furniture box's top / sides / legs, a structure plane):

  its region in the photo (the rectangle projected, depth-tested against the photo's point map) split into
    clean      the surface as the photo saw it
    front      something nearer the camera than the surface (an object, the real robot, a cable, a frame)
    removed    inpainted (recon.inpaint: the real robot and the reconstructed objects taken out of the plate)
    shadow     clean pixels much darker than the surface's local level (cast and contact shadows)
    highlight  near-saturated or much brighter than the local level (specular, glare, a lamp's reflection)
  the clean pixels' colour in the photo's albedo (scene/plate_albedo.png) and texture contrast -> candidates of the
  group's role (support top, furniture side / leg, wall, fabric, floor) matched in chromaticity, brightness and
  contrast, plus a plain finish of that colour

-> qa/surfaces/<group>.jpg (the photo crop with the classes overlaid, the albedo crop, the candidates as swatches at
real scale, the proposal) and qa/surfaces.json (fractions, flags, candidates, the proposal). recon.materials applies
the proposals once the agent has reviewed them (scene graph `_surface_review`; `--auto` accepts them): the agent
confirms or changes each group with a `look` (furniture: per group, sim.setpieces; structure planes: on the plane)
-- {"pbr", "tint": "match" | [r, g, b], "tile_m"}, {"color", "roughness"} or {"photo": true} to keep the photo
texture (a far surface seen from about the photo's direction).
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR
from calibrated_scene_reconstruction.scene.graph import face_group, structure_role
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.bake import ProjectionContext
from calibrated_scene_reconstruction.scene.background.materials import FLOOR_CANDS, WALL_CANDS, lin, local_std, tex_mean, tex_std

TOP_CANDS = ["oak_veneer_01", "ash_veneer", "american_walnut_veneer", "black_walnut_veneer_01", "cherry_veneer",
             "bamboo_veneer", "wood_table_001", "wood_table", "kitchen_wood", "plywood", "marble_01",
             "grey_cartago_01", "brushed_concrete", "terrazzo_tiles", "granite_tile", "proc_laminate_white",
             "proc_laminate_grey", "proc_laminate_black"]
FABRIC_CANDS = ["rough_linen", "cotton_jersey", "hessian_230", "waffle_pique_cotton"]
ROLE_CANDS = {"top": TOP_CANDS, "wall": WALL_CANDS + FABRIC_CANDS, "floor": FLOOR_CANDS}
PLAIN_ROLES = ("sides", "legs")  # box sides and legs: a plain finish of the clean colour unless the agent says more
OVERLAY = {"front": (230, 40, 40), "removed": (220, 40, 220), "shadow": (40, 90, 255), "highlight": (255, 220, 0),
           "second": (0, 210, 210)}
PLANE_TOL = 0.04  # m: a photo point this close to the rectangle's plane is the surface itself


def groups(s: Session, radius):
    """{group: {"parts", "role", "kind", "dist"}} of the surfaces within `radius` of the task support."""
    sg, parts = s.scene_graph, s.manifest["parts"]
    (sx0, sy0, _), (sx1, sy1, _) = sg["furniture"][sg["task"]["support"]]["box"]
    support = sg["task"]["support"]

    def dist(V):
        lo, hi = V[:, :2].min(0), V[:, :2].max(0)
        dx = max(sx0 - hi[0], lo[0] - sx1, 0.0)
        dy = max(sy0 - hi[1], lo[1] - sy1, 0.0)
        return float(np.hypot(dx, dy))

    out = {}
    for name, st in sg["structure"].items():
        role = structure_role(name, st)
        if role in ("window", "ceiling") or name not in parts:
            continue
        d = dist(np.load(s / "scene" / parts[name]["npz"])["V"])
        if role == "floor" or d <= radius:
            out[name] = {"parts": [name], "role": "floor" if role == "floor" else "wall", "kind": "structure",
                         "dist": d}
    for name, f in sg["furniture"].items():
        mine = [p for p in parts if p.startswith(f"{name}_") and p[len(name) + 1:].split("_")[0] in
                ("top", "front", "back", "left", "right", "bottom", "leg0", "leg1", "leg2", "leg3", "basin")]
        if not mine:
            continue
        d = min(dist(np.load(s / "scene" / parts[p]["npz"])["V"]) for p in mine)
        if name != support and d > radius:
            continue
        for g in ("top", "sides", "legs"):
            ps = [p for p in mine if face_group(p, name) == g and not p.endswith("_bottom")]
            if ps:
                out[f"{name}:{g}"] = {"parts": ps, "role": g, "kind": "furniture",
                                      "furniture": name, "group": g, "dist": d}
    return out


def split(shape, region, Pw, rgb, V, cam, own=None, removed=None, tol=PLANE_TOL):
    """The pixels of `region` (an image mask; Pw / rgb their world points and colours) of the rectangle V split into
    clean / front / removed / shadow / highlight / second material, as image masks."""
    n = np.cross(V[1] - V[0], V[3] - V[0])
    n = n / max(np.linalg.norm(n), 1e-9)
    sd = (Pw - V[0]) @ n  # signed distance of the image's points from the plane
    toward = np.sign((cam - V[0]) @ n) or 1.0  # the camera's side of the plane
    near = np.abs(sd) < tol
    front = (sd * toward) >= tol
    own = own if own is not None else np.zeros(len(Pw), bool)
    removed = removed if removed is not None else np.zeros(len(Pw), bool)
    onplane = near | (own & ~front)
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    clean0 = onplane & ~removed
    second = np.zeros_like(clean0)
    level = float(np.median(lum[clean0])) if clean0.sum() >= 30 else 128.0
    if clean0.sum() >= 200:  # two materials on one surface (a frame band, a mat, a stain) or a large shadow
        x = rgb.astype(np.float32)[clean0]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, lab, cen = cv2.kmeans(x, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = lab.ravel()
        minor = int(np.argmin(np.bincount(lab, minlength=2)))
        share = float((lab == minor).mean())
        l0, l1 = cen[1 - minor] @ [0.299, 0.587, 0.114], cen[minor] @ [0.299, 0.587, 0.114]
        ch = np.linalg.norm(cen[0] / max(cen[0].sum(), 1) - cen[1] / max(cen[1].sum(), 1))
        if 0.06 <= share <= 0.5 and (l1 / max(l0, 1) < 0.55 or l1 / max(l0, 1) > 1.8 or ch > 0.06):
            second[np.flatnonzero(clean0)[lab == minor]] = True
            level = float(np.median(lum[clean0 & ~second]))
    shadow = clean0 & ~second & (lum < 0.62 * level)
    highlight = clean0 & ~second & ((lum > 245) | (lum > 1.6 * level))
    clean = clean0 & ~shadow & ~highlight & ~second
    out = {"region": region}
    for k, m in (("clean", clean), ("front", front & ~removed), ("removed", removed), ("shadow", shadow),
                 ("highlight", highlight), ("second", second)):
        M = np.zeros(shape, bool)
        M[region] = m
        out[k] = M
    return out


def polygon(u, v, z, shape):
    """The image mask of a rectangle's projection (u, v, z per corner or per sample; behind-camera samples dropped)."""
    ok = z > 0.05
    if ok.sum() < 3:
        return None
    hull = cv2.convexHull(np.stack([u[ok], v[ok]], 1).astype(np.float32))
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1)
    return m.astype(bool)


def samples(V, n=40):
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return (V[0] + a[..., None] * (V[1] - V[0]) + b[..., None] * (V[3] - V[0])).reshape(-1, 3)


def classify(ctx: ProjectionContext, V, plate, own_ids, albedo):
    """The rectangle V (4 corners) in the photo: region and its split (split())."""
    region = polygon(*ctx.project(samples(V)), (ctx.H, ctx.W))
    if region is None:
        return None
    region &= ctx.valid
    if region.sum() < 30:
        return None
    own = ctx.union(own_ids)[region] if own_ids else None
    out = split((ctx.H, ctx.W), region, ctx.to_world(ctx.P[region]), plate[region], V, ctx.to_world(np.zeros(3)),
                own, ctx.removed[region])
    return out


class ViewSet:
    """The session's other photos (recon.multiview: DA3's depth and pose of each, corrected against the photo), in the
    world frame -- empty for a single-photo session. A surface hidden in the photo (behind the real robot, under the
    table) is often in plain view in another; its pixels add to the clean colour and the sheet shows each view."""

    def __init__(self, s: Session, ctx: ProjectionContext):
        self.views = []
        mv, da = s / "parsing" / "multiview.json", s / "parsing" / "multiview" / "da3.npz"
        if not (mv.exists() and da.exists()):
            return
        J, D = json.loads(mv.read_text()), np.load(da, allow_pickle=True)
        if not J.get("views") or not J.get("da3"):
            return
        self.ctx = ctx
        self.scale = float((J.get("scale") or {}).get("value", 1.0))  # the point map was rescaled after the views
        to_photo = float(J["da3"]["to_point_map"])
        names = [str(n) for n in D["names"]]
        for v in J["views"]:
            if v["name"] == "photo" or v["name"] not in names:
                continue
            i = names.index(v["name"])
            p = s / "capture" / "views" / f"{v['name']}.png"
            if not p.exists():
                continue
            depth = D["depth"][i] * to_photo
            img = self.prep(np.asarray(Image.open(p).convert("RGB")), D["prep"][i], depth.shape)
            ok = D["conf"][i] >= np.percentile(D["conf"][i], 30)
            self.views.append({"name": v["name"], "T": np.asarray(v["T_photo_view"], float), "K": D["K"][i].astype(float),
                               "depth": depth, "valid": ok, "img": img})

    @staticmethod
    def prep(a, prep, shape):  # an image as DA3 saw it (recon.multiview.Views.to_da3)
        rot, sc, x0, y0 = prep
        a = np.ascontiguousarray(np.rot90(a, int(rot)))
        a = cv2.resize(a, (round(a.shape[1] * sc), round(a.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        return a[int(y0):int(y0) + shape[0], int(x0):int(x0) + shape[1]]

    def project(self, view, Pw):
        Xp = (Pw @ self.ctx.R_wc + self.ctx.o) / self.scale  # photo camera frame, the views' units
        Ti = np.linalg.inv(view["T"])
        X = Xp @ Ti[:3, :3].T + Ti[:3, 3]
        z = X[:, 2]
        u = X[:, 0] / np.maximum(z, 1e-6) * view["K"][0, 0] + view["K"][0, 2]
        v = X[:, 1] / np.maximum(z, 1e-6) * view["K"][1, 1] + view["K"][1, 2]
        return u, v, z

    def classify(self, view, V):
        h, w = view["depth"].shape
        region = polygon(*self.project(view, samples(V)), (h, w))
        if region is None:
            return None
        region &= view["valid"]
        if region.sum() < 30:
            return None
        vv, uu = np.nonzero(region)
        z = view["depth"][vv, uu]
        X = (np.linalg.inv(view["K"]) @ np.stack([uu + 0.5, vv + 0.5, np.ones_like(uu, float)])).T * z[:, None]
        Xp = (X @ view["T"][:3, :3].T + view["T"][:3, 3]) * self.scale
        Pw = self.ctx.to_world(Xp)
        cam = self.ctx.to_world(view["T"][:3, 3] * self.scale)
        return split((h, w), region, Pw, view["img"][region], V, cam, tol=2 * PLANE_TOL)  # DA3's depth is coarser


def anisotropy(gray, mask):
    """How directional a surface's texture is (0 isotropic: plaster, concrete, carpet; toward 1: wood grain, planks,
    slats), from the structure tensor of its high-pass over the mask; None for fewer than 400 pixels."""
    if mask is not None and mask.sum() < 400:
        return None
    g = cv2.GaussianBlur(np.asarray(gray, np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = mask if mask is not None else np.ones(g.shape, bool)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    a, b, c = (gx * gx)[m].mean(), (gx * gy)[m].mean(), (gy * gy)[m].mean()
    tr, det = a + c, a * c - b * b
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    return float((l1 - l2) / max(l1 + l2, 1e-6))


_TEX_AN = {}


def tex_anisotropy(name):
    if name not in _TEX_AN:
        g = Image.open(POLYHAVEN_DIR / "textures" / name / "diff.jpg").convert("L").resize((512, 512))
        _TEX_AN[name] = anisotropy(np.asarray(g), None)
    return _TEX_AN[name]


def candidates(role, color, std, an=None, k=3):
    """[(name, tint, cost)] of the role's PBR candidates closest to the clean colour (chromaticity, brightness),
    texture contrast and directionality (anisotropy(): grain against stone), best first."""
    t = lin(color)
    out = []
    for n in ROLE_CANDS.get(role, []):
        if not (POLYHAVEN_DIR / "textures" / n / "diff.jpg").exists():
            continue
        m = lin(tex_mean(n))
        c = np.linalg.norm(m / m.sum() - t / t.sum()) + 0.3 * abs(m.mean() - t.mean())
        c += 0.15 * float(np.abs(np.log(np.clip(t.mean() / max(m.mean(), 1e-4), 0.05, 20))))  # heavy tinting reads false
        if std is not None:
            c += 0.01 * abs(tex_std(n) - std)
        if an is not None:
            c += 0.4 * abs(tex_anisotropy(n) - an)
        tint = [round(float(x), 3) for x in np.clip(t / np.maximum(m, 1e-4), 0.02, 3.0)]
        out.append((n, tint, round(float(c), 4)))
    return sorted(out, key=lambda x: x[2])[:k]


def swatch(name, tint, size=160, tile_m=0.6, span_m=0.6):
    """A candidate as it would look over span_m of surface (its texture at tile_m per repeat), tinted."""
    img = Image.open(POLYHAVEN_DIR / "textures" / name / "diff.jpg").convert("RGB")
    rep = max(int(round(size * tile_m / span_m)), 8)
    tile = np.asarray(img.resize((rep, rep), Image.LANCZOS), np.float32)
    a = np.tile(tile, (size // rep + 1, size // rep + 1, 1))[:size, :size]
    a = (((a / 255.0) ** 2.2 * np.asarray(tint)) ** (1 / 2.2) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


def overlay(img, cls, pad_frac=0.08):
    """The crop of an image round a surface's region, plain and with the pixel classes overlaid."""
    ys, xs = np.nonzero(cls["region"])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(pad_frac * max(y1 - y0, x1 - x0)) + 10
    y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
    y1, x1 = min(y1 + pad, img.shape[0]), min(x1 + pad, img.shape[1])
    crop = img[y0:y1, x0:x1].astype(np.float32)
    over = crop.copy()
    over[~cls["region"][y0:y1, x0:x1]] *= 0.35
    for k, c in OVERLAY.items():
        m = cls[k][y0:y1, x0:x1]
        over[m] = 0.45 * over[m] + 0.55 * np.asarray(c, np.float32)
    return crop, over, (y0, y1, x0, x1)


def sheet(path, title, plate, albedo, cls, cands, color, proposal, lines, views=()):
    """The agent's sheet: the photo (if it sees the surface), the classes, the photo albedo, each other view that sees
    it (multi-photo sessions: plain | classes), the candidates and the proposal."""
    panels, labels = [], []
    if cls is not None:
        crop, over, (y0, y1, x0, x1) = overlay(plate, cls)
        al = albedo[y0:y1, x0:x1].copy()
        al[~cls["region"][y0:y1, x0:x1]] = (al[~cls["region"][y0:y1, x0:x1]] * 0.35).astype(al.dtype)
        sc = min(1500 / crop.shape[1], 360 / crop.shape[0])
        W, H = max(int(crop.shape[1] * sc), 200), max(int(crop.shape[0] * sc), 60)
        panels += [Image.fromarray(x.astype(np.uint8)).resize((W, H)) for x in (crop, over, al)]
        labels += ["photo", "classes: red in front, magenta inpainted, blue shadow, yellow highlight, cyan a second material",
                   "photo albedo (what a photo texture would bake in)"]
    for name, img, vc in views:
        crop, over, _ = overlay(img, vc)
        sc = min(740 / crop.shape[1], 300 / crop.shape[0])
        w, h = max(int(crop.shape[1] * sc), 120), max(int(crop.shape[0] * sc), 60)
        pair = Image.new("RGB", (2 * w + 8, h), (24, 24, 24))
        pair.paste(Image.fromarray(crop.astype(np.uint8)).resize((w, h)), (0, 0))
        pair.paste(Image.fromarray(over.astype(np.uint8)).resize((w, h)), (w + 8, 0))
        panels.append(pair)
        labels.append(f"view {name} (plain | classes)")
    sw = [(f"plain {np.round(color).astype(int).tolist()}", Image.new("RGB", (160, 160), tuple(int(v) for v in color)))]
    sw += [(f"{n}  cost {cost}", swatch(n, tint)) for n, tint, cost in cands]
    width = max([p.width for p in panels] + [1100]) + 12
    out = Image.new("RGB", (width, sum(p.height + 22 for p in panels) + 320), (24, 24, 24))
    d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(255, 255, 255))
    y = 20
    for lab, p in zip(labels, panels):
        d.text((6, y), lab, fill=(200, 200, 200))
        out.paste(p, (6, y + 14))
        y += p.height + 22
    y += 4
    d.text((6, y), "proposal: " + json.dumps(proposal)[:160], fill=(120, 255, 120))
    for i, (lab, im) in enumerate(sw):
        out.paste(im, (6 + i * 175, y + 18))
        d.text((6 + i * 175, y + 182), lab[:28], fill=(220, 220, 220))
    for i, ln in enumerate(lines):
        d.text((6 + 4 * 175 + 20, y + 18 + 16 * i), ln[:70], fill=(255, 180, 120) if ln.startswith("!") else (220, 220, 220))
    out.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--radius", type=float, default=4.0,
                    help="m from the task support (horizontal): what the robot's head camera sees past the support")
    args = ap.parse_args()
    s = Session(args.session)
    sg, parts = s.scene_graph, s.manifest["parts"]
    ctx = ProjectionContext(s)
    plate = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB").resize((ctx.W, ctx.H)))
    ap_alb = s / "scene" / "plate_albedo.png"
    albedo = np.asarray(Image.open(ap_alb).convert("RGB").resize((ctx.W, ctx.H))) if ap_alb.exists() else plate
    ok = ctx.valid & ~ctx.removed & (plate.max(-1) > 20)  # the photo's albedo / radiance: the views are radiance only
    gain = np.median(albedo[ok].astype(np.float32) / np.maximum(plate[ok].astype(np.float32), 1.0), 0)
    vs = ViewSet(s, ctx)
    g_alb = cv2.cvtColor(albedo, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = g_alb - cv2.GaussianBlur(g_alb, (0, 0), 6)
    qa = s / "qa" / "surfaces"
    qa.mkdir(parents=True, exist_ok=True)
    report, measured = {}, []
    for gname, g in groups(s, args.radius).items():
        if g["kind"] == "structure":
            own = sg["structure"].get(g["parts"][0], {}).get("ids", [])
        else:
            own = sorted({i for v in sg["furniture"][g["furniture"]].get("faces", {}).values() for i in v})
        own = ["*"] if "*" in own else own
        acc, vacc = None, {}
        for p in g["parts"]:
            V = np.load(s / "scene" / parts[p]["npz"])["V"]
            c = classify(ctx, V, plate, own, albedo)
            if c is not None:
                acc = c if acc is None else {k: acc[k] | c[k] for k in acc}
            for view in vs.views:
                c = vs.classify(view, V)
                if c is not None:
                    a = vacc.get(view["name"])
                    vacc[view["name"]] = c if a is None else {k: a[k] | c[k] for k in a}
        if acc is not None and acc["region"].sum() < 30:
            acc = None
        measured.append((gname, g, acc, vacc))
    # each view's exposure and white balance against the photo's albedo, from the surfaces both see clean (a view's
    # radiance times the photo's global albedo / radiance gain is off by the views' own exposure)
    vgain = {}
    for view in vs.views:
        r = []
        for _, _, acc, vacc in measured:
            vc = vacc.get(view["name"])
            if acc is None or vc is None or acc["clean"].sum() < 300 or vc["clean"].sum() < 300:
                continue
            r.append(np.median(albedo[acc["clean"]], 0) / np.maximum(np.median(view["img"][vc["clean"]], 0), 1.0))
        vgain[view["name"]] = np.median(r, 0) if r else gain
    for gname, g, acc, vacc in measured:
        flags, entry = [], {"parts": g["parts"], "role": g["role"], "dist_m": round(g["dist"], 2)}
        if acc is None and not vacc:
            flags.append("not in any photo: no evidence, the scene graph's look or a theme decides")
            entry.update(flags=flags, proposal=None)
            report[gname] = entry
            continue
        frac, lines = {}, [f"role {g['role']}, {entry['dist_m']} m from the support"]
        pool = []  # clean colours (albedo-like) from every image that sees the surface
        if acc is not None:
            n_reg = int(acc["region"].sum())
            visible = acc["clean"] | acc["shadow"] | acc["highlight"] | acc["second"]
            frac = {k: round(float(acc[k].sum()) / n_reg, 3)
                    for k in ("clean", "front", "removed", "shadow", "highlight", "second")}
            vis = max(int(visible.sum()), 1)
            if frac["clean"] < 0.15:
                flags.append(f"! clean only {frac['clean']:.0%} of its photo region")
            if acc["shadow"].sum() / vis > 0.12:
                flags.append(f"! shadows {acc['shadow'].sum() / vis:.0%} of the visible surface (baked by a photo texture)")
            if acc["highlight"].sum() / vis > 0.05:
                flags.append(f"! highlights {acc['highlight'].sum() / vis:.0%} (glare / reflections)")
            if acc["second"].sum() / vis > 0.06:
                c2 = np.median(albedo[acc["second"]], 0).astype(int).tolist()
                flags.append(f"! a second material {acc['second'].sum() / vis:.0%} (rgb {c2}): a band, a mat, a stain or "
                             "a shadow -- the agent decides (a look per group, a set piece)")
            if frac["front"] + frac["removed"] > 0.25:
                flags.append(f"! {frac['front'] + frac['removed']:.0%} hidden by things in front or inpainted")
            pool.append(albedo[acc["clean"]].astype(np.float32))
            lines += [f"photo: " + ", ".join(f"{k} {v:.0%}" for k, v in frac.items())]
        vfrac = {}
        for name, vc in vacc.items():
            view = next(v for v in vs.views if v["name"] == name)
            n_reg = int(vc["region"].sum())
            vfrac[name] = {k: round(float(vc[k].sum()) / n_reg, 3) for k in ("clean", "front", "shadow", "second")}
            pool.append(view["img"][vc["clean"]].astype(np.float32) * vgain[name])
            lines.append(f"view {name}: " + ", ".join(f"{k} {v:.0%}" for k, v in vfrac[name].items()))
        pool = np.concatenate([p for p in pool if len(p)]) if any(len(p) for p in pool) else None
        if pool is not None and len(pool) >= 30:
            color = np.clip(np.median(pool, 0), 0, 255)
        elif acc is not None:
            color = np.median(albedo[acc["region"]], 0)
            flags.append("! no clean pixel in any photo: the colour is a guess")
        else:
            color = np.median(np.concatenate([next(v for v in vs.views if v["name"] == n)["img"][vc["region"]]
                                              for n, vc in vacc.items()]), 0) * np.mean([vgain[n] for n in vacc], 0)
            flags.append("! no clean pixel in any photo: the colour is a guess")
        er = cv2.erode(acc["clean"].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) if acc is not None else None
        std = float(hp[er].std()) if er is not None and er.sum() >= 200 else None
        best = (int(acc["clean"].sum()), g_alb, acc["clean"]) if acc is not None else (0, None, None)
        for name, vc in vacc.items():  # a view facing the surface resolves its grain better than a grazing photo
            if int(vc["clean"].sum()) > best[0]:
                img = next(v for v in vs.views if v["name"] == name)["img"]
                best = (int(vc["clean"].sum()), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), vc["clean"])
        an = anisotropy(best[1], best[2]) if best[1] is not None else None
        role = g["role"]
        cands = candidates(role, color, std, an) if role not in PLAIN_ROLES else []
        proposal = ({"pbr": cands[0][0], "tint": cands[0][1]} if cands else
                    {"color": [round(float(v) / 255.0, 3) for v in color], "roughness": 0.5})
        entry.update(fractions=frac, view_fractions=vfrac, color_srgb=[int(v) for v in color],
                     anisotropy=an and round(an, 3),
                     contrast=std and round(std, 2), candidates=[{"pbr": n, "tint": t, "cost": c} for n, t, c in cands],
                     proposal=proposal, flags=flags)
        sheet(qa / f"{gname.replace(':', '__')}.jpg", gname, plate, albedo, acc, cands, color, proposal, lines + flags,
              [(n, next(v for v in vs.views if v["name"] == n)["img"], vc) for n, vc in vacc.items()])
        report[gname] = entry
        print(f"[surfaces] {gname}: photo {json.dumps(frac)} views {len(vacc)} -> {json.dumps(proposal)}"
              + (" | " + "; ".join(flags) if flags else ""))
    (s / "qa" / "surfaces.json").write_text(json.dumps({"radius_m": args.radius, "views": [v["name"] for v in vs.views],
                                                        "view_gains": {k: np.round(v, 3).tolist() for k, v in vgain.items()},
                                                        "groups": report}, indent=1))
    print(f"[surfaces] {len(report)} groups, {1 + len(vs.views)} photo(s) -> {qa}, {s / 'qa' / 'surfaces.json'}")


if __name__ == "__main__":
    main()
