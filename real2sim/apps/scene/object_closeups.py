"""Each manipulable object of the twin beside the walk's own close-ups of it: the nearest frame from each side,
rendered with that frame's camera at its full resolution and cropped round the object.

    tools/run_isaac.sh apps/scene/object_closeups.py sessions/<name> [--views 8] [--only peg,hole_block] [--all]
        [--background splat] [--variant v.json] [--tag lit]

A box size and a place can be right while the shape is wrong: SAM 3D completes a thing of a few dozen pixels from its
prior, and every check that compares
boxes, depths or silhouettes in the photo passes it. Here the twin's object is put next to what the camera saw from
up to `--views` directions (the nearest frame in each 360/views degree sector round it): the agent compares the shape
-- holes, surface details, handles, a base plate, things beside it the twin lacks -- and replaces a wrong one with a measured
kit (assets.kits, fitted by recon.object_fit; docs/object_review.md). The frames are recon.posed_frames' cameras
(parsing/posed_frames/cameras.npz: every frame of the walks, in the twin's world) and their images. Writes
qa/object_closeups/<object>.jpg (per view: the frame's crop over the twin's, the twin object's outline drawn on the
frame) and qa/object_closeups/closeups.json. Per view also the look against the frame, independent of either
camera's exposure: the object's mean colour (its eroded mask) as chromaticity (r, g of r+g+b) and its brightness over
the ring round it (the support it stands on) -- the twin's minus the frame's (`look` in closeups.json: what still
reads as computer graphics in a colour or in a light that is not the room's). `--background splat` renders over the
splat background, `--variant` a variant's lights (or anything else), `--tag` writes qa/object_closeups_<tag>/. `--all`: every static object and piece of room furniture as well (the
frames that see most of it): the review of the room's things against the walk, not only the photo."""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--reach", type=float, default=3.0, help="frames within this distance, m")
ap.add_argument("--only", default="")
ap.add_argument("--all", action="store_true", help="every item: the static objects and the room's furniture too")
ap.add_argument("--tile", type=int, default=300)
ap.add_argument("--background", default="mesh", choices=["mesh", "splat", "zone", "blender"])
ap.add_argument("--variant", default="", help="a variant json layered over the twin (its lights, say)")
ap.add_argument("--tag", default="", help="writes qa/object_closeups_<tag>/")
ap.add_argument("--fixed-exposure", action="store_true", help="no auto exposure (the lights set the meshes' level)")
ap.add_argument('--renderer', choices=['rt', 'pt'], default='rt', help='pt: path-traced material/refraction review')
ap.add_argument('--spp', type=int, default=64)
ap.add_argument('--bounces', type=int, default=16)
args = ap.parse_args()
if args.spp < 1 or args.bounces < 1:
    ap.error('Positive samples and bounce count required')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": 'PathTracing' if args.renderer == 'pt' else "RaytracedLighting", "width": 1280, "height": 720})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.review.views import observation_frames  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, quat_from_matrix, set_pose  # noqa: E402

configure_render(mode=args.renderer, auto_exposure=not args.fixed_exposure, spp=args.spp, bounces=args.bounces)
s = Session(args.session)
sg = s.scene_graph
cams = s / "parsing" / "posed_frames" / "cameras.npz"
if not cams.exists():
    sys.exit(f"no {cams}: run recon.posed_frames select first")
z = np.load(cams)
img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
names = [str(n) for n in z["names"]]
W, H = (int(v) for v in z["size"][0])

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(
    s.dir, json.loads(Path(args.variant).read_text()) if args.variant else {})))
if args.background == "splat":
    from calibrated_scene_reconstruction.sim.splat_stage import add_background

    hidden = add_background(stage, prepare_background(s.dir), "splat")  # their pixels are the splat's (hidden from the camera alone)
elif args.background in ("zone", "blender"):  # the task zone alone, as delivered (sim.zone: its own light)
    from calibrated_scene_reconstruction.sim.splat_stage import add_background

    add_background(stage, prepare_background(s.dir), args.background)
cam = add_camera(stage, "/World/Cams/closeup", z["K"][0], W, H, T_world_cam_cv=np.eye(4), far=60.0)
rp = rep.create.render_product(str(cam.GetPath()), (W, H))
ann = {k: rep.AnnotatorRegistry.get_annotator(k, init_params=p) for k, p in
       (("rgb", None), ("instance_id_segmentation", {"colorize": False}))}
for a in ann.values():
    a.attach(rp)
world.reset()
for _ in range(10):
    world.step(render=False)  # the objects settle on their supports

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
objs = [f"/World/Objects/{n}" for n in sg.get("dynamic_objects", {})]
if args.all or args.only:  # the room's things too: SAM 3D laid one upside down, every box check passed
    for root in ("/World/Static", "/World/Synthetic", "/World/Support", '/World/Room'):
        scope = stage.GetPrimAtPath(root)
        objs += [str(c.GetPath()) for c in (scope.GetChildren() if scope.IsValid() else [])
                 if "_collider" not in c.GetName()
                 and (root != '/World/Room' or c.GetName() in sg.get('furniture', {}))]
objs = [o for o in objs if not args.only or o.rsplit("/", 1)[1] in args.only.split(",")]
out = s / "qa" / ("object_closeups" + (f"_{args.tag}" if args.tag else ""))
out.mkdir(parents=True, exist_ok=True)


def to_linear(im):
    x = im.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def look(real, twin, mask):
    """(chromaticity difference, log brightness-over-ring difference): the twin's minus the frame's, or None."""
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0)
    if inner.sum() < 30 or ring.sum() < 30:
        return None
    out = []
    for im in (real, twin):
        L = to_linear(im)
        o, r = L[inner].mean(0), L[ring].mean(0)
        out.append((o[:2] / max(o.sum(), 1e-6), np.log(max(o.mean(), 1e-6) / max(r.mean(), 1e-6))))
    return float(np.linalg.norm(out[1][0] - out[0][0])), float(out[1][1] - out[0][1])


def level(real, twin, mask, bg):
    """Over the splat: the meshes' brightness against the frame's (log of the linear means; 0 when the meshes look
    as the frames do -- the splat beside them is trained to the frames' pixels) -- the object's and the ring's round
    it (its support, the splat's pixels `bg` left out). None without enough of either. (The splat's own ratio in the
    same render is no reference in a close-up: a few far pixels.)"""
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0) & ~(cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0)
    ring &= ~bg
    if inner.sum() < 30 or ring.sum() < 30:
        return None
    R, T = to_linear(real).mean(-1), to_linear(twin).mean(-1)
    bg = bg & (cv2.erode(bg.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)  # the splat's own, off the meshes' edges
    return tuple(float(np.log(max(T[m].mean(), 1e-6) / max(R[m].mean(), 1e-6))) if m.sum() >= 30 else float("nan")
                 for m in (inner, ring, bg))


def project(X, i):
    Xc = (X - z["centre"][i]) @ z["R_world_cam"][i]
    K = z["K"][i]
    return Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], Xc[:, 2]


report = {}
for opath in objs:
    name = opath.rsplit("/", 1)[1]
    prim = stage.GetPrimAtPath(opath)
    if not prim.IsValid():
        continue
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    c = (lo + hi) / 2
    corners = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in (lo[2], hi[2])])
    sector = {}
    eligible=observation_frames(s,name,names)
    for i in eligible:
        uv, d = project(c[None], i)
        dist = float(np.linalg.norm(c - z["centre"][i]))
        reach = max(args.reach, 3.0 * float(np.linalg.norm(hi - lo)))  # a sofa is seen whole from farther
        if d[0] < 0.2 or dist > reach or not (60 < uv[0, 0] < W - 60 and 60 < uv[0, 1] < H - 60):
            continue
        cu_, cz_ = project(corners, i)
        if (cz_ < 0.1).any() or ((cu_ < 0) | (cu_ > [W, H])).any(1).mean() > 0.25:
            continue  # most of it in the frame
        v = c - z["centre"][i]
        k = int((np.degrees(np.arctan2(v[1], v[0])) + 180) // (360 / args.views)) % args.views
        if k not in sector or dist < sector[k][0]:
            sector[k] = (dist, i)
    tiles, used = [], []
    for k in sorted(sector):
        dist, i = sector[k]
        K = z["K"][i]
        cam.GetHorizontalApertureAttr().Set(float(W / K[0, 0] * 18.0))
        cam.GetVerticalApertureAttr().Set(float(H / K[1, 1] * 18.0))
        set_pose(cam.GetPrim(), z["centre"][i], quat_from_matrix(z["R_world_cam"][i] @ np.diag([1, -1, -1])))
        for _ in range(6):
            world.step(render=True)
        rgb = np.ascontiguousarray(np.asarray(ann["rgb"].get_data())[..., :3])
        seg = ann["instance_id_segmentation"].get_data()
        ids = np.asarray(seg["data"])
        mine = [int(a) for a, v in seg["info"]["idToLabels"].items() if f"{opath}/" in str(v) + "/"]
        mask = np.isin(ids, mine).astype(np.uint8)
        real = cv2.cvtColor(cv2.imread(str(img_dir / names[i])), cv2.COLOR_BGR2RGB)
        uv, _ = project(corners, i)
        u0, v0 = uv.min(0)
        u1, v1 = uv.max(0)
        half = min(max(48.0, 1.3 * max(u1 - u0, v1 - v0) / 2), H / 2)
        cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
        x0, y0 = int(np.clip(cu - half, 0, W - 2 * half)), int(np.clip(cv_ - half, 0, H - 2 * half))
        x1, y1 = x0 + int(2 * half), y0 + int(2 * half)
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        lined = real.copy()
        cv2.drawContours(lined, cnt, -1, (0, 255, 0), 1)
        col = [cv2.resize(im[y0:y1, x0:x1], (args.tile, args.tile), interpolation=cv2.INTER_CUBIC)
               for im in (real, rgb, lined)]
        cv2.putText(col[0], f"{names[i]} {dist:.2f} m", (4, 16), 0, 0.45, (255, 255, 0), 1)
        tiles.append(np.vstack(col))
        lk = look(real, rgb, mask)
        lv = None
        if args.background == "splat":  # the splat's pixels: the meshes it stands for (hidden from the camera)
            sp_ids = [int(a) for a, v in seg["info"]["idToLabels"].items()
                      if any(str(v) == h or str(v).startswith(h + "/") for h in hidden)]
            lv = level(real, rgb, mask, np.isin(ids, sp_ids))
        used.append({"frame": names[i], "distance_m": round(dist, 3), "crop_px": int(2 * half),
                     **({"chroma": round(lk[0], 4), "contrast": round(lk[1], 3)} if lk else {}),
                     **({"level": round(lv[0], 3), "level_ring": round(lv[1], 3), "level_splat": round(lv[2], 3)}
                        if lv else {})})
    if not tiles:
        report[name] = {"views": 0}
        print(f"[closeups] {name}: no frame within {args.reach} m sees it", flush=True)
        continue
    sheet = np.hstack(tiles)
    cv2.imwrite(str(out / f"{name}.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    ch = [u["chroma"] for u in used if "chroma" in u]
    co = [u["contrast"] for u in used if "contrast" in u]
    lv = [(u["level"], u["level_ring"], u["level_splat"]) for u in used if "level" in u]
    report[name] = {"views": len(used), "frames": used, "candidate_observation_frames":len(eligible),
                    "box_m": (hi - lo).round(3).tolist(),
                    **({"look": {"chroma": round(float(np.median(ch)), 4), "contrast": round(float(np.median(co)), 3)}}
                       if ch else {}),
                    **({"level": round(float(np.median([v[0] for v in lv])), 3),
                        "level_ring": round(float(np.median([v[1] for v in lv])), 3),
                        "level_splat": round(float(np.nanmedian([v[2] for v in lv])), 3)} if lv else {})}
    print(f"[closeups] {name}: {len(used)} views, nearest {min(u['distance_m'] for u in used):.2f} m, "
          f"crops {min(u['crop_px'] for u in used)}-{max(u['crop_px'] for u in used)} px"
          + (f", look: chroma {report[name]['look']['chroma']:.3f}, contrast {report[name]['look']['contrast']:+.2f}"
             if "look" in report[name] else "")
          + (f", level against the frames {report[name]['level']:+.2f} (its support {report[name]['level_ring']:+.2f}, "
             f"the splat round it {report[name]['level_splat']:+.2f})"
             if "level" in report[name] else "") + f" -> {out / name}.jpg",
          flush=True)
(out / "closeups.json").write_text(json.dumps(report, indent=1) + "\n")
print(f"done: {len(report)} objects -> {out} (rows: the frame, the twin, the twin's outline on the frame)", flush=True)
app.close()
