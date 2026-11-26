"""Agent review of every object and surface of a built twin / cousin: automatic checks (sim.object_review) and
close-ups next to the photo, for the agent to judge as a real photo would be judged and fix in the scene graph.

    tools/run_isaac.sh apps/scene/review_objects.py sessions/<name> --tag twin [--variant cousins/cousin_00.json]

Items: /World/Static (reconstructions, templates, library swaps, set pieces), /World/Support (movable furniture),
/World/Synthetic (furniture the completion invented) and /World/Objects (the manipulable ones). Per item:
  support   rays down from its lowest points: floating, sunk, or hung on a wall / from the ceiling
  collider  a room object's box collider against its visible footprint (an invisible ledge)
  balance   rays down over its whole footprint: overhanging its support, or its centre off it (it would tip)
  overlap   a ray that starts inside another collider: the item intersects it (a stool inside a table's box)
  size      height and footprint against its class's range
  material  glass or metal rebuilt as an opaque diffuse shell
and two close views (from the photo's side, from 70 deg round) beside the photo's crop of its instance mask.
Surfaces: every table / counter top from above at an angle beside the photo's crop of it (a projected texture smeared
by the objects that stood on it, stair-stepped mask edges, shadows baked in). Overview: the photo next to the render
from the photo camera.

What the agent looks for (docs/object_review.md): would this pass in a photograph -- standing on something, at its
real size, its real material (glass, metal, wood, fabric), complete (a row of books, not five of them), and a surface
that looks like the material it is. Fixes go into configs/<name>/scene_graph.json (sim.setpieces): an object's
`fix` (snap / swap / kit / drop, or accept: "why it is right as it is"), `set_pieces`, a furniture `look`, a dynamic
object's `look`; then rerun from views.
Writes qa/objects/<tag>/{<item>.jpg, sheet_<k>.jpg, surfaces.jpg, overview.jpg} and qa/objects_<tag>.json.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant")
ap.add_argument("--per-sheet", type=int, default=6)
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 640, 480
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.grasping.targets import object_label  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.review import objects as orv  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import room_obstacles, world_points  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, load_variant, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, pinhole, set_pose  # noqa: E402

configure_render()
s = Session(args.session)
sg = s.scene_graph
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = prepare_variant(s.dir, load_variant(args.variant))
ts = build_scene(stage, prepare_scene(s.dir), variant)
world.reset()
sqi = get_physx_scene_query_interface()
man = ts.manifest
cam = man["photo_camera"]
T_wc = np.asarray(cam["T_world_cam_cv"], float)
K_photo = np.asarray(cam["K"], float)
photo = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"))
masks = np.load(s / "parsing" / "masks.npz")["masks"]
walls = {k: v for k, v in room_obstacles(ts).items() if k in man["parts"]}
ceiling_z = man["surfaces"]["ceiling_z"]
out = s / "qa" / "objects" / args.tag
out.mkdir(parents=True, exist_ok=True)
try:
    FONT = ImageFont.load_default(15)
except TypeError:  # Pillow < 10.1
    FONT = ImageFont.load_default()

K = pinhole(W, H, 1.0)
cams = {k: add_camera(stage, f"/World/Cams/review_{k}", K, W, H, [0, 0, 2], [1, 0, 1]) for k in ("a", "b")}
annot = {k: rgb_annotator(c.GetPath(), W, H) for k, c in cams.items()}
pcam = add_camera(stage, "/World/Cams/review_photo", pinhole(W, H, K_photo[0, 0] / cam["width"]), W, H,
                  T_world_cam_cv=T_wc)
annot["photo"] = rgb_annotator(pcam.GetPath(), W, H)


def shoot(views, frames=10):
    """views: {camera: (eye, target)} -> {camera: image}."""
    for k, (eye, at) in views.items():
        set_pose(cams[k].GetPrim(), eye, look_at_quat(eye, at))
    for _ in range(frames):
        world.step(render=True)
    return {k: grab(annot[k]) for k in views}


def label(img, lines, bad=False):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((8, 6 + 18 * i), t, fill=(255, 110, 110) if bad and i == 0 else (255, 255, 255), font=FONT,
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def photo_crop(M=None, box_px=None):
    """The photo around a mask (outlined) or a pixel box, letterboxed to a tile."""
    if M is not None:
        ys, xs = np.where(M)
        if not len(xs):
            return None
        box_px = [xs.min(), ys.min(), xs.max(), ys.max()]
    x0, y0, x1, y1 = box_px
    pad = 0.35 * max(x1 - x0, y1 - y0) + 20
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(photo.shape[1], x1 + pad)), int(min(photo.shape[0], y1 + pad))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = photo[y0:y1, x0:x1].copy()
    if M is not None:
        import cv2
        cs, _ = cv2.findContours(M[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cs, -1, (255, 60, 60), max(1, int(0.004 * max(crop.shape[:2]))))
    im = Image.fromarray(crop)
    im.thumbnail((W, H))
    tile = Image.new("RGB", (W, H), (30, 30, 30))
    tile.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.asarray(tile)


def project(P):
    Pc = (P - T_wc[:3, 3]) @ T_wc[:3, :3]
    Pc = Pc[Pc[:, 2] > 0.05]
    uv = Pc[:, :2] / Pc[:, 2:3] * [K_photo[0, 0], K_photo[1, 1]] + K_photo[:2, 2]
    return uv


def box_dist(lo, hi, blo, bhi):
    d = np.maximum(0, np.maximum(blo - hi, lo - bhi))
    return float(np.linalg.norm(d))


def gaps_under(P, path):
    """Per ray from the item's lowest points: the gap to the first collider under it that is not the item."""
    zmin = P[:, 2].min()
    low = P[P[:, 2] < zmin + 0.01]
    low = low[np.linspace(0, len(low) - 1, min(len(low), 40)).astype(int)]
    out, under, inside = [], [], set()
    z0 = float(zmin + 0.03)
    for p in low:
        hits = []

        def report(h):
            c = str(h.collision)
            if c == path or c.startswith(path + "/") or c.startswith(path + "_collider"):
                return True
            if h.distance < 1e-4:  # the ray starts inside this collider: the item intersects it
                inside.add(c)
            else:
                hits.append((float(h.position[2]), c))
            return True
        sqi.raycast_all((float(p[0]), float(p[1]), z0), (0.0, 0.0, -1.0), 3.0, report)
        top = max(hits) if hits else None
        out.append(float(p[2] - top[0]) if top else None)
        under.append(top[1] if top else None)
    names = [u for u in under if u]
    return out, max(set(names), key=names.count) if names else None, sorted(inside)


def drops_under(P, path, n=8):
    """(x, y, drop) on an n x n grid over the item's footprint (cells its points cover): how far below its base the
    first other collider under that point lies, None when nothing is."""
    zmin = float(P[:, 2].min())
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    cell = np.maximum((hi - lo) / n, 1e-6)
    idx = np.unique(np.clip(((P[:, :2] - lo) / cell).astype(int), 0, n - 1), axis=0)
    out = []
    for i, j in idx:
        x, y = lo + (np.array([i, j]) + 0.5) * cell
        hits = []

        def report(h):
            c = str(h.collision)
            if not (c == path or c.startswith(path + "/") or c.startswith(path + "_collider")) and h.distance > 1e-4:
                hits.append(float(h.position[2]))
            return True
        sqi.raycast_all((float(x), float(y), zmin + 0.03), (0.0, 0.0, -1.0), 3.0, report)
        out.append((float(x), float(y), zmin - max(hits) if hits else None))
    return out


# --- items ---------------------------------------------------------------------------------------------------------
stool_groups = [st.get("group") or st.get("recon") or st.get("model") for st in variant.get("stool_models", [])]
items = []
for root in ("/World/Static", "/World/Support", "/World/Synthetic", "/World/Objects"):
    p = stage.GetPrimAtPath(root)
    for prim in (p.GetChildren() if p.IsValid() else []):
        name = prim.GetName()
        if "_collider" in name:
            continue
        spec, ids, kind = {}, [], "library"
        if root == "/World/Static":
            spec = sg.get("static_objects", {}).get(name) or sg.get("set_pieces", {}).get(name) or {}
            ids = spec.get("ids", [])
            kind = ("kit" if prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit") else "template" if spec.get("template")
                    else "recon" if prim.IsA(UsdGeom.Mesh) else "library")
            lab = spec.get("_label") or name
        elif root == "/World/Support":
            k = int(name.rsplit("_", 1)[1])
            grp = stool_groups[k] if k < len(stool_groups) else "stool"
            lab = grp
            mf = sg.get("movable_furniture", {}).get(grp, {})
            ids = mf.get("ids", [])
            kind = "recon" if prim.GetCustomDataByKey("calibrated_scene_reconstruction:reconstructed") else "library"
            lab = mf.get("category", grp)
        elif root == "/World/Synthetic":
            src = next((p for p in variant.get("synthetic_furniture", []) if p["name"] == name), {}).get("source", {})
            kind = "recon" if src.get("recon") else "library" if src.get("model") else "proxy"
            lab = name.replace("synthetic_", "").rstrip("_0123456789")
        else:
            spec = next((o for o in variant.get("objects", []) if o.get("as", o.get("name")) == name), {"name": name})
            dyn = sg.get("dynamic_objects", {}).get(name, {})
            ids = dyn.get("ids", []) if not (spec.get("mesh_path") or spec.get("ph_model")) else []
            kind = "library" if (spec.get("mesh_path") or spec.get("ph_model")) else "recon"
            if dyn.get("look", {}).get("glass") and kind == "recon":
                kind = "recon_glass"
            lab = object_label(s, spec)
        items.append({"item": name, "root": root.rsplit("/", 1)[1], "prim": prim, "label": lab, "kind": kind,
                      "ids": ids, "spec": spec})

labels = {str(it["prim"].GetPath()): it["label"] for it in items}


def label_of(collider):
    """The review label of the item a collider belongs to (its prim, a child of it or its _collider box)."""
    for path, lab in labels.items():
        if collider == path or collider.startswith(path + "/") or collider.startswith(path + "_collider"):
            return lab
    return collider.rsplit("/", 1)[1]


report = {"tag": args.tag, "items": [], "surfaces": []}
tiles = []
for it in items:
    path = str(it["prim"].GetPath())
    P = world_points(it["prim"], 20000)
    if not len(P):
        continue
    lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    wall = min([box_dist(lo, hi, np.asarray(a), np.asarray(b)) for a, b in walls.values()] or [9.0])
    gaps, on, inside = gaps_under(P, path)
    held = [c for c in inside if orv.contains(it["label"], label_of(c))]  # a plant in its basket
    inside = [c for c in inside if c not in held]
    names = ", ".join(c.rsplit("/", 1)[1] for c in inside)
    checks = {"support": {**orv.support_check(it["label"], gaps, wall, ceiling_z - hi[2]), "on": on},
              "overlap": {"ok": not inside, "inside": inside, "holds_or_held": held,
                          **({"problem": "inside " + names} if inside else {})},
              "size": orv.size_check(it["label"], ext),
              "material": orv.material_check(it["label"], "kit" if it["kind"] == "recon_glass" else it["kind"]),
              "upright": orv.upright_check(it["label"], P)}
    if it["root"] == "Synthetic" and it["prim"].GetCustomDataByKey("calibrated_scene_reconstruction:mesh_collider"):
        checks["collider"] = {"ok": True, "note": "its meshes are its collider"}
    elif it["root"] == "Synthetic":
        box = next((np.asarray(q["box"], float) for q in variant.get("synthetic_furniture", []) if q["name"] == it["item"]),
                   None)
        if box is not None:
            checks["collider"] = orv.collider_check(lo, hi, box[0], box[1], it["label"], P)
    on_floor = checks["support"].get("on") and "floor" in str(checks["support"]["on"])
    checks["balance"] = orv.balance_check(drops_under(P, path), P.mean(0)[:2],
                                          checks["support"].get("mount") or ("floor" if on_floor else None))
    if held and not checks["support"]["ok"]:
        checks["support"] = {**checks["support"], "ok": True, "mount": "in " + held[0].rsplit("/", 1)[1]}
    if it["kind"] == "kit":
        checks["size"] = {"ok": True, "note": "set piece: sized from the photo"}
    accept = (it["spec"].get("fix") or {}).get("accept") if it["root"] == "Static" else None
    if accept:  # the agent judged it against the photo: as it is there (frames leaning on each other)
        checks = {k: {**v, "ok": True, **({"accepted": accept} if not v["ok"] else {})} for k, v in checks.items()}
    failed = [k for k, v in checks.items() if not v["ok"]]
    c = (lo + hi) / 2
    r = max(0.5 * float(np.linalg.norm(ext)), 0.06)
    dist = max(2.6 * r, 0.3)
    toward = T_wc[:2, 3] - c[:2]
    a0 = np.arctan2(toward[1], toward[0])
    views = {}
    for k, (da, el) in {"a": (0.0, 0.5), "b": (np.radians(70), 0.4)}.items():
        d = np.array([np.cos(a0 + da) * np.cos(el), np.sin(a0 + da) * np.cos(el), np.sin(el)])
        views[k] = (c + dist * d, c)
    imgs = shoot(views)
    M = np.any(masks[it["ids"]], 0) if it["ids"] else None
    crop = photo_crop(M) if M is not None else None
    if crop is None:
        uv = project(P[::20])
        ok = (uv[:, 0] > 0) & (uv[:, 0] < cam["width"]) & (uv[:, 1] > 0) & (uv[:, 1] < cam["height"])
        crop = photo_crop(box_px=[*uv[ok].min(0), *uv[ok].max(0)]) if ok.sum() > 10 else None
    if crop is None:
        crop = np.full((H, W, 3), 30, np.uint8)
        crop = label(crop, ["", "not in the photo"])
    probs = [v.get("problem") or k for k, v in checks.items() if not v["ok"]]
    head = f"{it['item']} ({it['label']}, {it['kind']})" + (f"  FAIL: {'; '.join(probs)}" if failed else "  ok")
    sup = checks["support"]
    line2 = (f"h {ext[2]:.2f} m, footprint {max(ext[:2]):.2f} m, gap {sup['min_gap_m']} m"
             + (f", {sup['mount']}" if sup.get("mount") else ""))
    row = np.concatenate([label(crop, [head, line2], bool(failed)), imgs["a"], imgs["b"]], 1)
    Image.fromarray(row).save(out / f"{it['item']}.jpg", quality=88)
    tiles.append(row)
    report["items"].append({"item": it["item"], "group": it["root"], "label": it["label"], "kind": it["kind"],
                            "ids": it["ids"], "bbox": [lo.round(3).tolist(), hi.round(3).tolist()],
                            "ok": not failed, "failed": failed, "checks": checks,
                            "fixed": getattr(ts, "fixed", {}).get(it["item"])})
    print(f"[objects] {args.tag} {head} | {line2}", flush=True)

for k in range(0, len(tiles), args.per_sheet):
    sheet = np.concatenate(tiles[k:k + args.per_sheet], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] * 2 // 3, sheet.shape[0] * 2 // 3)).save(
        out / f"sheet_{k // args.per_sheet}.jpg", quality=85)

# --- surfaces: table and counter tops --------------------------------------------------------------------------------
rows = []
for name, (blo, bhi) in man["collision_boxes"].items():
    blo, bhi = np.asarray(blo, float), np.asarray(bhi, float)
    if not (0.3 < bhi[2] < 1.3) or "_leg" in name or (bhi - blo)[:2].min() < 0.25:
        continue
    c = np.array([(blo[0] + bhi[0]) / 2, (blo[1] + bhi[1]) / 2, bhi[2]])
    size = float(np.linalg.norm((bhi - blo)[:2]))
    toward = T_wc[:2, 3] - c[:2]
    toward /= np.linalg.norm(toward)
    up = min(0.55 * size + 0.3, ceiling_z - 0.2 - c[2])  # under the ceiling: further back instead
    eye = c + np.r_[(0.45 * size + max(0.55 * size + 0.3 - up, 0.0)) * toward, up]
    img = shoot({"a": (eye, c)})["a"]
    corners = np.array([[x, y, bhi[2]] for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1])])
    uv = project(corners)
    ok = (uv[:, 0] > -200) & (uv[:, 1] > -200) & (uv[:, 0] < cam["width"] + 200) & (uv[:, 1] < cam["height"] + 200)
    crop = photo_crop(box_px=[*np.clip(uv[ok].min(0), 0, None), *uv[ok].max(0)]) if ok.sum() >= 2 else None
    if crop is None:
        crop = label(np.full((H, W, 3), 30, np.uint8), ["", "not in the photo"])
    looks = [p for p in man["parts"] if p.startswith(name)]
    rows.append(np.concatenate([label(crop, [f"surface {name}", f"top {bhi[2]:.2f} m"]), img], 1))
    report["surfaces"].append({"surface": name, "top_z": float(bhi[2]), "parts": looks})
if rows:
    for k in range(0, len(rows), 4):
        sheet = np.concatenate(rows[k:k + 4], 0)
        Image.fromarray(sheet).save(out / f"surfaces_{k // 4}.jpg", quality=85)

for _ in range(10):
    world.step(render=True)
ph = Image.fromarray(photo)
ph.thumbnail((W, H))
over = np.concatenate([np.asarray(ph.resize((W, H))), grab(annot["photo"])], 1)
Image.fromarray(over).save(out / "overview.jpg", quality=88)
s.write_json(f"qa/objects_{args.tag}.json", report)
bad = [r for r in report["items"] if not r["ok"]]
print(f"[objects] {args.tag}: {len(report['items'])} items, {len(bad)} with automatic flags"
      + (" (" + ", ".join(f"{r['item']}: {'/'.join(r['failed'])}" for r in bad) + ")" if bad else "")
      + f", {len(report['surfaces'])} surfaces -> {out}", flush=True)
app.close()
