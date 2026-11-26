"""S6c room objects: what the room completion painted, reconstructed in 3D for the twin and for every cousin.

    python -m calibrated_scene_reconstruction.recon.room_objects sessions/<name> [--tags twin,cousin_00] [--reuse-masks] [--refit] [--apply]

A completion (recon.panorama; a cousin's: recon.cousin_rooms) only paints the part of the room the photo did not see;
reviewed by the agent (recon.completion_review), it says what stands there. Its pixels are never a texture: a desk
painted onto a wall is flat from every other viewpoint. The same holds for what the photo itself shows beyond what
perception parsed (a desk, a chair, a shelf in the background, which the bake used to flatten onto the wall behind
it): for the twin the photo is one more source. Here, per tag (the twin: scene/pano and the photo, a cousin:
scene/pano/<id>), in the ring views of its completion:
  1. SAM 3 instances of a room vocabulary (VOCAB: furniture, storage, equipment, boxes, plants, and flat things on a
     wall; completion.json "objects" adds the agent's labels), outside the photo's own pixels, duplicates merged
     (recon.pano_furniture.segment),
  2. each instance lifted to world points with the view's metric depth (recon.panorama's joint depth scales,
     recon.pano_furniture.lift) and boxed; dropped: the photo's own furniture continued by the completion, anything
     in the work zone (WORK_CLEAR round the task support: where the robot stands and reaches), frame-cut slivers,
     sizes its class cannot have; one piece seen from several views keeps its best view; at most MAX_PIECES,
     the nearer the task support the earlier (NEAR),
  3. every solid piece is reconstructed with SAM 3D Objects from its own view (the view, its mask and the view's point
     map: recon.objects worker) and placed by the better silhouette IoU of SAM 3D's own layout (tilted upright, stood
     on its support) and an upright fit at the height it is seen at (the floor, or the top of what it rests on: yaw,
     planar ICP and a footprint scale, recon.furniture.fit_upright); a flat
     piece (whiteboard, door, poster, a screen on a wall) becomes a thin panel on its wall, textured from the view
     (rectified onto the wall plane: a flat thing looks right from every side),
  4. a reconstruction whose silhouette IoU stays below MIN_IOU is replaced by the best library model of its class
     (DINOv2 on the crop) where the library has the class, at most its class's height (STAND_IN), else dropped;
     both are reported for the agent's review.
Writes scene/room_objects/<tag>/<piece>/{inputs.npz, raw.glb, mesh_local.glb, albedo.png | panel.png},
scene/room_objects/<tag>.json and qa/room_objects/<tag>.jpg (each piece: its view crop with the fitted model's
silhouette, the scores, the room from above). --apply writes the kept pieces as "synthetic_furniture": into the scene
graph for the twin (recon.materials carries them into scene/twin.json), into cousins/<id>.json for a cousin.

configs/<name>/completion.json `"source": "captured"`: the scene as it was captured, nothing generated -- no
completion, no wall bays, no cousins; the twin's room objects come from the photo and the session's other views (a
walk-around video's frames) only, and what they did not see stays bare wall.
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from calibrated_scene_reconstruction.scene.graph import dumps, structure_role
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.align import icp_yaw, iou, project_mask  # noqa: F401  (icp_yaw: fit_upright's)
from calibrated_scene_reconstruction.scene.support.furniture import fit_upright, load_zup, place_upright
from calibrated_scene_reconstruction.scene.objects.reconstruct import run_worker
from calibrated_scene_reconstruction.scene.background.pano_furniture import RING, Room, footprint_overlap, lift
from calibrated_scene_reconstruction.scene.background.panorama import N, load_views

# label: (class, minimum depth in m -- the far side is never seen --, kind) with kind floor (stands on the floor; a
# cabinet / shelf seen only high up hangs there), rests (on the floor or on whatever is under it: a box, a monitor)
# or flat (a thin panel on a wall)
VOCAB = {
    "office chair": ("chair", 0.5, "floor"), "chair": ("chair", 0.45, "floor"), "armchair": ("chair", 0.7, "floor"),
    "stool": ("stool", 0.35, "floor"), "bench": ("bench", 0.35, "floor"), "sofa": ("sofa", 0.8, "floor"),
    "desk": ("table", 0.6, "floor"), "table": ("table", 0.6, "floor"), "workbench": ("table", 0.6, "floor"),
    "counter": ("counter", 0.6, "floor"), "cabinet": ("cabinet", 0.4, "floor"),
    "filing cabinet": ("cabinet", 0.45, "floor"), "drawer unit": ("cabinet", 0.4, "floor"),
    "bookshelf": ("shelf", 0.3, "floor"), "shelving unit": ("shelf", 0.35, "floor"),
    "storage rack": ("shelf", 0.4, "floor"), "refrigerator": ("cabinet", 0.6, "floor"),
    "trash can": ("bin", 0.25, "floor"), "potted plant": ("plant", 0.3, "floor"),
    "cardboard box": ("box", 0.25, "rests"), "storage bin": ("box", 0.25, "rests"), "printer": ("box", 0.35, "rests"),
    "computer monitor": ("monitor", 0.15, "rests"), "lamp": ("lamp", 0.2, "rests"),
    "whiteboard": ("whiteboard", 0.0, "flat"), "door": ("door", 0.0, "flat"), "poster": ("picture", 0.0, "flat"),
    "picture frame": ("picture", 0.0, "flat"), "television": ("screen", 0.0, "flat"),
}
# plausible size per class: (min height, max height, max horizontal extent) in m
SIZE = {"chair": (0.6, 1.4, 1.0), "stool": (0.35, 1.0, 0.6), "bench": (0.3, 0.9, 2.2), "sofa": (0.5, 1.2, 2.6),
        "table": (0.45, 1.2, 3.2), "counter": (0.7, 1.2, 3.5), "cabinet": (0.4, 2.3, 2.4), "shelf": (0.3, 2.4, 3.0),
        "bin": (0.2, 1.0, 0.7), "plant": (0.25, 2.2, 1.2), "box": (0.08, 1.2, 1.2), "monitor": (0.2, 0.8, 1.2),
        "lamp": (0.2, 1.9, 0.8), "whiteboard": (0.4, 1.6, 3.5), "door": (1.7, 2.4, 1.8), "picture": (0.15, 1.5, 2.0),
        "screen": (0.3, 1.4, 2.2)}
LIBRARY = {"chair", "stool", "bench", "table"}  # classes the converted library holds (recon.retrieve)
STAND_IN = {"chair": (1.2, 0.95), "stool": (0.95, 0.65), "bench": (0.6, 0.45), "table": (1.0, 0.75)}  # m: the
# tallest a library stand-in of a class may be made, and its height when the failed fit's box is taller
NEAR = 3.0  # m from the task support over which a piece's rank halves: the robot's cameras see the near room
STRUCTURE_LABELS = {"floor", "wall", "ceiling", "window", "door"}  # perception labels of the room's shell
WORK_CLEAR = 1.3  # m round the task support kept free of invented objects: robot bases and arm sweeps
MIN_IOU = 0.4  # silhouette IoU of a reconstruction in its own view below which it is clearly wrong (the library's)
MIN_PX = 1500
THIN = 0.12  # m: a reconstruction thinner than this and taller than 0.8 m is a slab, not furniture (a cabinet side,
SLAB = 5.0   # a door seen edge-on), and so is one over SLAB times taller than its thinnest side
MIN_PANEL_IOU = 0.75  # a flat piece's rectangle on its wall against its mask (a skewed or partial one reads wrong)
MAX_PIECES = 20  # solid pieces per tag
MAX_FLAT = 4  # flat ones (SAM 3 calls a cabinet's doors "door" too: they must not crowd out the furniture)
SAM3D_WORKERS = 2  # SAM 3D processes at once
FIT_PROCS = 8  # pieces fitted at once


class ViewFrame:
    """A completed view as a camera (recon.furniture.PhotoFrame's interface): image, intrinsics, point map in the
    view camera (OpenCV), its world pose."""

    def __init__(self, img, K, depth, ok, R, cam):
        self.img, self.K, self.shape = img, np.asarray(K, float), img.shape[:2]
        H, W = self.shape
        v, u = np.mgrid[0:H, 0:W]
        d = np.stack([(u + 0.5 - self.K[0, 2]) / self.K[0, 0], (v + 0.5 - self.K[1, 2]) / self.K[1, 1],
                      np.ones((H, W))], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        self.points = d * depth[..., None]  # depth: distance along the ray
        self.valid = ok & np.isfinite(depth) & (depth > 0)
        self.R, self.cam = np.asarray(R, float), np.asarray(cam, float)
        self.T_wc = np.eye(4)
        self.T_wc[:3, :3], self.T_wc[:3, 3] = self.R, self.cam

    def to_world(self, v):
        return v @ self.R.T + self.cam

    def to_cam(self, v):
        return (v - self.cam) @ self.R

    def mask(self, m):
        return m

    def observed(self, m, erode=5):
        er = cv2.erode(m.astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool) & self.valid
        return self.to_world(self.points[er])

    def silhouette_iou(self, Vw, faces, m):
        return float(iou(project_mask(self.to_cam(Vw), faces, self.K, self.shape), m))


def tags(s: Session):
    return ["twin"] + sorted(p.stem for p in (s / "cousins").glob("cousin_*.json"))


def pano_dir(s: Session, tag):
    return s / "scene" / "pano" / ("" if tag == "twin" else tag)


def walls(s: Session):
    """Wall planes: (name, axis 0 | 1, at, inward normal sign, (lo, hi) along the wall)."""
    out = []
    for k, v in s.scene_graph["structure"].items():
        if structure_role(k, v) == "wall" and v.get("plane") in ("x", "y"):
            sign = 1.0 if v.get("normal", "+") == "+" else -1.0
            out.append((k, 0 if v["plane"] == "x" else 1, float(v["at"]), sign, tuple(map(float, v["range"][0]))))
    return out


def work_zone(s: Session):
    sg = s.scene_graph
    sup = sg.get("furniture", {}).get(sg.get("task", {}).get("support"))
    if not sup:
        return None
    b = np.asarray(sup["box"], float)[:, :2]
    return b + [[-WORK_CLEAR] * 2, [WORK_CLEAR] * 2]


def segment_images(images, vocab, thr=0.45):
    """SAM 3 instances of the vocabulary per image {name: rgb} (one model load); one piece answering several labels
    keeps its most confident one."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda").eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    out = {}
    for name, img in images.items():
        inst = []
        for label in vocab:
            inputs = proc(images=Image.fromarray(img), text=label, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model(**inputs)
            r = proc.post_process_instance_segmentation(o, threshold=thr, mask_threshold=0.5,
                                                        target_sizes=inputs.get("original_sizes").tolist())[0]
            for m, sc in zip(r["masks"], r["scores"]):
                m = m.cpu().numpy().astype(bool)
                if 0.0005 < m.mean() < 0.5:  # a "wall" answer covers the view
                    inst.append({"label": label, "score": float(sc), "mask": m})
        inst.sort(key=lambda i: -i["score"])
        keep = []
        for i in inst:
            if all((i["mask"] & k["mask"]).sum() < 0.6 * min(i["mask"].sum(), k["mask"].sum()) for k in keep):
                keep.append(i)
        out[name] = keep
        print(f"[room_objects] {name}: {sorted(i['label'] for i in keep)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


class Source:
    """An image the room's objects are taken from: a completed ring view (its pixels outside the photo's) or the photo
    itself (its pixels outside what the scene graph already holds); lift(mask, taken) -> (world points, depth
    rescale), frame(rescale) -> ViewFrame."""

    def __init__(self, name, img, exclude, lift_fn, frame_fn, cam, anchored=True, photo=False, bay=False):
        self.name, self.img, self.exclude, self.lift, self.frame = name, img, exclude, lift_fn, frame_fn
        self.cam, self.anchored, self.photo, self.bay = np.asarray(cam, float), anchored, photo, bay
        self.shape = img.shape[:2]
        self.min_px = MIN_PX * self.shape[0] * self.shape[1] / N ** 2


def pano_sources(s, views, deps):
    out = {}
    for name in RING:
        img, kv, R = views[name]
        k = deps["names"].index(name)

        def lift_fn(m, taken, k=k, kv=kv, R=R):
            return lift(deps["depth"][k], deps["ok"][k], R, deps["K"][k], deps["cam"], m, taken,
                        anchored=kv.mean() > 0.02)

        def frame_fn(scale, k=k, img=img, R=R):
            return ViewFrame(img, deps["K"][k], deps["depth"][k] * scale, deps["ok"][k], R, deps["cam"])

        out[name] = Source(name, img, kv, lift_fn, frame_fn, deps["cam"])
    return out


def wall_sources(s: Session, tag):
    """The wall completion's bays (recon.wall_completion): furniture painted straight on against a free wall, its depth
    scaled to the rendered room; only its painted region is a source."""
    out = {}
    for f in sorted((s / "scene" / "walls" / tag).glob("*.npz")):
        done = f.with_name(f.stem + "_completed.png")
        if not done.exists():
            continue
        z = np.load(f)
        img = np.asarray(Image.open(done).convert("RGB"))
        depth = np.load(f.with_name(f.stem + "_depth.npy")).astype(np.float32)
        paint = np.asarray(Image.open(f.with_name(f.stem + "_paint.png"))) > 127
        K, T = z["K"], z["T_world_cam"]
        H, W = depth.shape
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        X = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1) * depth[..., None]
        vf = ViewFrame(img, K, np.linalg.norm(X, axis=-1), np.isfinite(depth), T[:3, :3], T[:3, 3])
        vf.points = X
        src = Source(f"bay:{f.stem}", img, ~cv2.dilate(paint.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool),
                     lambda m, taken, vf=vf: (vf.observed(m), 1.0), lambda scale, vf=vf: vf, T[:3, 3], bay=True)
        src.wall = json.loads(str(z["wall"]))
        out[src.name] = src
    return out


def photo_source(s: Session):
    """The photo: what it shows beyond what the scene graph holds (its objects, furniture, set pieces, the photo
    robot) and beyond the structure (floor, walls, ceiling, windows) -- a desk, a chair, a shelf in the background,
    which a photo texture would flatten onto the wall behind it."""
    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"))
    geo, masks = s.geometry(), s.masks()
    _, _, T = s.world_frame()
    T = np.asarray(T)
    pf = ViewFrame(img, geo["K"], np.linalg.norm(geo["points"], axis=-1), geo["mask"], T[:3, :3], T[:3, 3])
    pf.points = geo["points"]  # the point map as it is (OpenCV camera)
    inst = s.read_json("parsing/instances.json") if (s / "parsing" / "instances.json").exists() else []
    exclude = np.zeros(img.shape[:2], bool)
    for i in inst:  # everything perception already parsed but the room's shell (the scene graph's objects, furniture,
        if i["label"] not in STRUCTURE_LABELS and 0 <= i["id"] < len(masks):  # set pieces, the photo robot)
            exclude |= masks[i["id"]]
    return Source("photo", img, exclude, lambda m, taken: (pf.observed(m), 1.0), lambda scale: pf, T[:3, 3],
                  photo=True)


def multiview_sources(s: Session):
    """The session's other photos (recon.multiview: DA3 depth and pose, corrected against the photo) as sources: they
    see more of the real room (the side the photo's edge cuts off). Their pixels carry no perception instances; what
    the scene graph holds is kept out by place (candidates: the support, the photo robot's base)."""
    from calibrated_scene_reconstruction.scene.background.bake import ProjectionContext
    from calibrated_scene_reconstruction.scene.support.surfaces import ViewSet

    ctx = ProjectionContext(s)
    out = {}
    for v in ViewSet(s, ctx).views:
        sc = float((s.read_json("parsing/multiview.json").get("scale") or {}).get("value", 1.0))
        H, W = v["depth"].shape
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        X = (np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(v["K"]).T) * v["depth"][..., None]
        T = v["T"]
        to_w = lambda x, T=T: ctx.to_world((x @ T[:3, :3].T + T[:3, 3]) * sc)  # noqa: E731  view units -> world
        cam = to_w(np.zeros((1, 3)))[0]
        R = np.stack([(to_w(np.eye(3)[i:i + 1])[0] - cam) / sc for i in range(3)], 1)  # view camera -> world axes
        R /= np.linalg.norm(R, axis=0, keepdims=True)  # T_photo_view carries the view's own scale (k R): normalised
        vf = ViewFrame(v["img"], v["K"], np.linalg.norm(X, axis=-1) * sc, v["valid"], R, cam)
        vf.points = (to_w(X.reshape(-1, 3)) - cam).reshape(X.shape) @ R  # metric, in the view camera (with k)
        out[v["name"]] = Source(v["name"], v["img"], np.zeros((H, W), bool),
                                lambda m, taken, vf=vf: (vf.observed(m), 1.0), lambda scale, vf=vf: vf, cam, photo=True)
    return out


def flat_overlap(a, b, depth=0.4):
    """Two flat pieces one thing: on the same plane (their thin horizontal axes alike, their centres within `depth` of
    each other along it) and overlapping over half the narrower along it. A door seen from four walk frames, each
    lifted a few cm off the others, has footprints a centimetre deep that hardly intersect: four doors on the wall."""
    ea, eb = a[1, :2] - a[0, :2], b[1, :2] - b[0, :2]
    t = int(np.argmin(ea))
    if int(np.argmin(eb)) != t:
        return False
    o = 1 - t
    if abs((a[0, t] + a[1, t]) - (b[0, t] + b[1, t])) / 2 > depth:
        return False
    ov = min(a[1, o], b[1, o]) - max(a[0, o], b[0, o])
    return ov > 0.5 * min(ea[o], eb[o])


def posed_sources(s: Session):
    """The walk frames recon.posed_frames posed (COLMAP at the board's scale, in the twin's world) with a MoGe-2 depth
    each (parsing/posed_frames/frames.npz): the whole room the walks went through, where the photo and its views saw
    the corner round the table. Like the views', their pixels carry no perception instances: what the scene graph
    holds is kept out by place."""
    p = s / "parsing" / "posed_frames" / "frames.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for i, name in enumerate(z["names"]):
        img, K, D = z["images"][i], z["K"][i], z["depth"][i].astype(np.float64)
        H, W = D.shape
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        X = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1) * D[..., None]
        vf = ViewFrame(img, K, np.linalg.norm(X, axis=-1), z["ok"][i] & (D > 0), z["R_world_cam"][i], z["centre"][i])
        vf.points = X
        key = f"walk:{str(name).rsplit('.', 1)[0]}"
        out[key] = Source(key, img, np.zeros((H, W), bool), lambda m, taken, vf=vf: (vf.observed(m), 1.0),
                          lambda scale, vf=vf: vf, z["centre"][i], photo=True)
    return out


UNIT_RANK = {"table": 0, "counter": 0, "cabinet": 0, "shelf": 0, "sofa": 0, "bench": 0}  # before chairs, boxes ...


def merge_units(cands, gap=0.05):
    """Parts of one piece of furniture seen in one view -- a cabinet's doors and drawers, the bins on a rack, a monitor
    on its desk -- merged into one candidate (their masks united, boxes joined): SAM 3 answers part by part, and SAM
    3D reconstructs each part as a thin slab otherwise. Solid kept candidates whose boxes touch (within `gap`) in the
    same view are merged, the largest naming the unit."""
    by_view = {}
    for c in cands:
        if not c["why"] and c["kind"] != "flat":
            by_view.setdefault(c["view"], []).append(c)
    for view, cs in by_view.items():  # the unit takes the name of its main piece: a desk with its chair is a desk
        cs.sort(key=lambda c: (UNIT_RANK.get(c["class"], 1), -c["pixels"]))
        groups = []
        for c in cs:
            lo, hi = c["box"][0] - gap, c["box"][1] + gap
            g = next((g for g in groups if np.all(lo < g["box"][1]) and np.all(hi > g["box"][0])), None)
            if g is None:
                groups.append(c)
                continue
            g["mask"] = g["mask"] | c["mask"]
            g["box"] = np.array([np.minimum(g["box"][0], c["box"][0]), np.maximum(g["box"][1], c["box"][1])])
            g["pixels"] = int(g["mask"].sum())
            g["rank"] = max(g["rank"], c["rank"]) * 1.05
            g.setdefault("parts", [g["label"]]).append(c["label"])
            if c.get("hangs") is None and g.get("hangs"):
                g["hangs"] = False  # standing parts under a wall cabinet: the unit stands
                g["box"][0, 2] = 0.0
            c["why"] = f"part of {g['label']} ({view})"
        for g in groups:  # a unit standing on the floor (any part does) is boxed from the floor
            if g.get("parts") and g["box"][0, 2] < 0.15:
                g["box"][0, 2] = 0.0
    return cands


def candidates(s, sources, inst, vocab, room):
    """Pieces of every source with a box, a kind and a rank (rejected ones keep their reason)."""
    zone = work_zone(s)
    wall_list = walls(s)
    sg = s.scene_graph
    sup = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float) if sg.get("task", {}).get("support") \
        in sg.get("furniture", {}) else None
    base = (sg.get("photo_robot") or {}).get("base") or {}
    robot_xy = np.asarray(base["xy"], float) if "xy" in base else None
    cands = []
    for name, src in sources.items():
        H, W = src.shape
        taken = np.any([j["mask"] for j in inst[name]], 0) if inst[name] else np.zeros((H, W), bool)
        for i in inst[name]:
            if i["label"] not in vocab:
                continue
            cls, min_depth, kind = vocab[i["label"]]
            m = i["mask"] & ~src.exclude
            c = {"label": i["label"], "class": cls, "kind": kind, "view": name, "score": round(i["score"], 3),
                 "pixels": int(m.sum()), "mask": m, "why": ""}
            cands.append(c)
            if m.sum() < src.min_px or (i["mask"] & src.exclude).sum() > 0.3 * i["mask"].sum():
                c["why"] = ("held by the scene graph" if src.photo else "photo pixels") if m.sum() >= src.min_px \
                    else "small"
                continue
            P, c["floor_scale"] = src.lift(m, taken)
            if len(P) < 300:
                c["why"] = "no depth"
                continue
            lo, hi = np.percentile(P, 3, axis=0), np.percentile(P, 97, axis=0)
            ys, xs = np.nonzero(m)
            c["frame_cut"] = bool(min(ys.min(), xs.min()) < 4 or ys.max() > H - 5 or xs.max() > W - 5)
            if kind == "floor" and not src.photo and m[H - 5:].sum() > 0.3 * 5 * (xs.max() - xs.min() + 1) \
                    and cls not in ("cabinet", "shelf"):
                c["why"] = "foot out of the view"
                continue
            if kind == "flat":
                box = np.array([lo, hi])
            else:
                if kind == "floor" and lo[2] < (0.5 if cls in ("cabinet", "shelf") else 10.0):
                    lo[2] = 0.0  # hidden legs do not lift it; a cabinet seen only high up hangs there
                if kind == "rests" and lo[2] < 0.15:
                    lo[2] = 0.0
                inside = np.mean((P[:, 0] > room.floor[0, 0]) & (P[:, 0] < room.floor[0, 1])
                                 & (P[:, 1] > room.floor[1, 0]) & (P[:, 1] < room.floor[1, 1]))
                if inside < 0.5:  # a cousin's completion may paint a larger room than the twin's walls
                    c["why"] = f"beyond the walls ({inside:.0%} inside)"
                    continue
                shift = np.zeros(3)  # mostly inside: moved in along the wall's normal, its size kept
                for a in (0, 1):
                    shift[a] = max(room.floor[a, 0] + 0.03 - lo[a], 0.0) + min(room.floor[a, 1] - 0.03 - hi[a], 0.0)
                lo, hi = lo + shift, hi + shift
                c["shift"] = shift
                if kind == "floor" and lo[2] >= 0.5:  # a wall cabinet / shelf: on a wall of the room, or not at all
                    gap = min(((abs(lo[w[1]] - w[2]) if w[3] > 0 else abs(hi[w[1]] - w[2]), w) for w in wall_list
                               if w[4][0] - 0.2 <= (lo[1 - w[1]] + hi[1 - w[1]]) / 2 <= w[4][1] + 0.2),
                              default=(np.inf, None), key=lambda g: g[0])
                    if gap[0] > 0.35:
                        c["why"] = f"hangs {gap[0]:.2f} m off a wall"
                        continue
                    w = gap[1]  # pushed against it
                    push = (w[2] - lo[w[1]]) if w[3] > 0 else (w[2] - hi[w[1]])
                    lo[w[1]] += push
                    hi[w[1]] += push
                    c["shift"][w[1]] += push
                    c["hangs"] = True
                box = room.clip(np.array([lo, hi]))
                cen = (box[0] + box[1]) / 2
                ax = int(np.argmax(np.abs(cen[:2] - src.cam[:2])))
                if box[1, ax] - box[0, ax] < min_depth:  # the hidden far side
                    if cen[ax] > src.cam[ax]:
                        box[1, ax] = box[0, ax] + min_depth
                    else:
                        box[0, ax] = box[1, ax] - min_depth
            c["box"] = box
            size = box[1] - box[0]
            h_lo, h_hi, w_hi = SIZE[cls]
            if not (h_lo * (0.6 if c["frame_cut"] else 1.0) <= size[2] <= h_hi and size[:2].max() <= w_hi):
                c["why"] = f"size {np.round(size, 2).tolist()}"
                continue
            if src.photo:  # real: kept unless it stands at the support (under the table: the robot's knees) or is
                if sup is not None and np.all(box[0, :2] < sup[1, :2] + 0.3) and np.all(box[1, :2] > sup[0, :2] - 0.3) \
                        and footprint_overlap(box, sup) > 0.3:  # the photo robot (another photo's pixels carry no
                    c["why"] = "at the support"  # instances); a desk standing beside the table is the room's
                    continue
                if robot_xy is not None and np.all(box[0, :2] < robot_xy + 0.6) and np.all(box[1, :2] > robot_xy - 0.6):
                    c["why"] = "the photo robot"
                    continue
                if any(footprint_overlap(box, b) > 0.3 and box[0, 2] < b[1, 2] for b in room.boxes):
                    c["why"] = "scene-graph furniture"
                    continue
            else:
                if kind != "flat" and room.is_photo_furniture(box):
                    c["why"] = "photo furniture"
                    continue
                if zone is not None and np.all(box[0, :2] < zone[1]) and np.all(box[1, :2] > zone[0]):
                    c["why"] = "work zone"
                    continue
            if c["frame_cut"] and m.sum() < 4 * src.min_px:
                c["why"] = "frame-cut sliver"
                continue
            c["rank"] = (2.0 if src.photo else 1.5 if src.bay else 1.0) * i["score"] * np.sqrt(m.sum() * N ** 2 / (H * W)) \
                * (0.4 if c["frame_cut"] else 1.0)
            if sup is not None:  # a desk beside the task table before one across the room (the PC on it floated)
                gap = np.maximum(0.0, np.maximum(sup[0, :2] - box[1, :2], box[0, :2] - sup[1, :2]))
                c["rank"] *= NEAR / (NEAR + float(np.linalg.norm(gap)))
    cands = merge_units(cands)
    ok = sorted((c for c in cands if not c["why"]), key=lambda c: -c["rank"])
    cfg = s.config("completion.json", {})  # a whole office walked through holds more than a corner's 20 pieces
    cap_pieces, cap_flat = int(cfg.get("max_pieces", MAX_PIECES)), int(cfg.get("max_flat", MAX_FLAT))
    keep = []
    for c in ok:
        dup = next((k for k in keep if footprint_overlap(c["box"], k["box"]) > 0.3
                    and (c["class"] == k["class"] or min(c["box"][1, 2], k["box"][1, 2]) > max(c["box"][0, 2],
                                                                                              k["box"][0, 2]) + 0.1)
                    and c["kind"] != "flat" and k["kind"] != "flat"), None)
        if dup is None and c["kind"] == "flat":
            dup = next((k for k in keep if k["kind"] == "flat" and (footprint_overlap(c["box"], k["box"]) > 0.3
                                                                     or flat_overlap(c["box"], k["box"]))), None)
        if dup is not None:
            c["why"] = f"seen better in {dup['view']} ({dup['label']})"
        elif sum((k["kind"] == "flat") == (c["kind"] == "flat") for k in keep) >= (cap_flat if c["kind"] == "flat"
                                                                                   else cap_pieces):
            c["why"] = "beyond MAX_FLAT" if c["kind"] == "flat" else "beyond MAX_PIECES"
        else:
            keep.append(c)
    return keep, cands


def piece_name(c):
    """A piece's name from its class, view and mask: the same piece keeps its name (and its SAM 3D mesh) across runs."""
    import hashlib

    h = hashlib.sha1(c["view"].encode() + np.packbits(c["mask"]).tobytes()).hexdigest()[:6]
    return f"room_{c['class']}_{h}"


def adopt(out, keep):
    """Folders of an earlier run named otherwise (by order) whose inputs hold the same mask: renamed to the piece's."""
    want = {c["name"]: c for c in keep}
    for d in sorted(out.glob("room_*")):
        if d.name in want or not (d / "inputs.npz").exists():
            continue
        z = np.load(d / "inputs.npz")
        for name, c in want.items():
            if not (out / name).exists() and z["mask"].shape == c["mask"].shape and np.array_equal(z["mask"],
                                                                                                    c["mask"]):
                d.rename(out / name)
                break


def support_under(c, keep):
    """The top of what a resting piece stands on: the floor or a kept piece under its footprint; a hanging cabinet or
    shelf: its own bottom (on its wall)."""
    if c.get("hangs"):
        return float(c["box"][0, 2])
    if c["kind"] != "rests" or c["box"][0, 2] < 0.15:
        return 0.0
    tops = [k["box"][1, 2] for k in keep if k is not c and k["kind"] != "flat"
            and footprint_overlap(c["box"], k["box"]) > 0.5 and abs(k["box"][1, 2] - c["box"][0, 2]) < 0.12]
    return float(max(tops)) if tops else float(c["box"][0, 2])


def sam3d_inputs(s, c, vf, out):
    """The SAM 3D worker's inputs for a piece: its view, mask and point map (recon.objects)."""
    d = out / c["name"]
    d.mkdir(parents=True, exist_ok=True)
    pts = vf.points.astype(np.float32).copy()
    pts[~vf.valid] = np.nan
    inp = d / "inputs.npz"
    np.savez_compressed(inp, rgb=vf.img, mask=c["mask"], points=pts, out_dir=str(d.relative_to(s.dir)))
    return inp


def sam3d_all(s, jobs, workers=SAM3D_WORKERS):
    """SAM 3D for every (name, inputs) in `workers` processes at once (one model each: ~12.5 GB host RAM, a third of
    the GPU); returns {name: exit code}."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max(1, workers)) as ex:
        codes = list(ex.map(lambda j: run_worker(s.dir, j[0], [0], j[1]), jobs))
    out = {j[0]: c for j, c in zip(jobs, codes)}
    for j in [j for j in jobs if out[j[0]] != 0]:  # two large pieces at once can run the GPU out of memory: alone
        out[j[0]] = run_worker(s.dir, j[0], [0], j[1])
    return out


def layout_upright(d, mesh, F, vf, mask, z0, max_tilt=30.0):
    """SAM 3D's own layout (pose.json: scale, rotation, translation in the view camera; the object frame's axis
    convention picked by silhouette IoU as recon.align does), tilted upright about its foot (its up axis onto world
    z, within max_tilt deg) and stood on z0. Returns (world vertices, IoU) or None."""
    import json

    from scipy.spatial.transform import Rotation

    from calibrated_scene_reconstruction.scene.objects.align import axis_rotations, local_to_cam_cv

    if not (d / "pose.json").exists():
        return None
    pose = json.loads((d / "pose.json").read_text())
    V = np.asarray(mesh.vertices, float)
    best = None
    for M in axis_rotations():
        v = local_to_cam_cv(V @ M.T, pose)
        if (v[:, 2] <= 0.05).any():
            continue
        s = iou(project_mask(v, F, vf.K, vf.shape), mask)
        if best is None or s > best[0]:
            best = (s, M, v)
    if best is None:
        return None
    _, M, v = best
    Vw = vf.to_world(v)
    o, *E = vf.to_world(local_to_cam_cv(np.vstack([np.zeros(3), np.eye(3)]) @ M.T, pose))
    ax = [(e - o) / max(np.linalg.norm(e - o), 1e-9) for e in E]
    up = max((sg * a for a in ax for sg in (1, -1)), key=lambda a: a[2])  # the object axis nearest world up
    ang = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
    if ang > max_tilt:
        return None
    axis = np.cross(up, [0, 0, 1.0])
    R = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-9) * np.radians(ang)).as_matrix() \
        if ang > 1e-3 else np.eye(3)
    foot = np.r_[(Vw.min(0) + Vw.max(0))[:2] / 2, Vw[:, 2].min()]
    Vw = (Vw - foot) @ R.T + foot
    Vw[:, 2] += z0 + 0.002 - Vw[:, 2].min()
    return Vw, float(iou(project_mask(vf.to_cam(Vw), F, vf.K, vf.shape), mask))


def reconstruct(s, c, vf, out):
    """A piece's SAM 3D mesh (sam3d_all) stood upright and fitted to its silhouette."""
    d = out / c["name"]
    if not (d / "raw.glb").exists():
        return {"why": "SAM 3D failed"}
    Vz, F, mesh = load_zup(d / "raw.glb")
    z0 = c["support_z"]
    height = float(np.percentile(vf.observed(c["mask"])[:, 2], 98)) - z0
    if height < 0.05:
        return {"why": f"height {height:.2f}"}
    f = fit_upright(vf, Vz, F, c["mask"], height, z0, fit_footprint=True, coarse=True)
    Vw = place_upright(Vz, f["yaw"], f["scale"], f["xy"], z0, f["scale_xy"])
    lay = layout_upright(d, mesh, F, vf, c["mask"], z0)  # SAM 3D's own pose, stood upright: often the better fit
    if lay is not None and lay[1] > f["iou"]:
        Vw, f = lay[0], {**f, "iou": lay[1]}
    proj = project_mask(vf.to_cam(Vw), F, vf.K, vf.shape)  # (as seen, before a move inside the walls)
    Vw = Vw + np.asarray(c.get("shift", np.zeros(3)))
    ctr = (Vw.min(0) + Vw.max(0))[:2] / 2
    local = mesh.copy()
    local.vertices = Vw - np.r_[ctr, 0.0]  # origin: footprint centre at z 0, axes: world (sim.scene)
    local.export(d / "mesh_local.glb")
    tex = getattr(mesh.visual.material, "baseColorTexture", None)
    if tex is not None:
        tex.convert("RGB").save(d / "albedo.png")
    return {"iou": round(f["iou"], 3), "box": np.array([Vw.min(0), Vw.max(0)]), "center": ctr, "proj": proj,
            "source": {"mesh": str(d.relative_to(s.dir))}}


def panel(s, c, vf, out, wall_list):
    """A flat piece as a 2 cm panel on its wall, its face the view's pixels rectified onto the wall plane."""
    P = vf.observed(c["mask"], erode=3)
    best = None
    for w in wall_list:  # the wall along the piece's line of sight: its points pushed onto the plane about the camera
        name, ax, at, sign, span = w  # (a completion's depth is least sure where nothing anchors it)
        k = (at - vf.cam[ax]) / (np.median(P[:, ax]) - vf.cam[ax] + 1e-9)
        if not 0.5 <= k <= 2.0:
            continue
        Q = vf.cam + (P - vf.cam) * k
        a = np.median(Q[:, 1 - ax])
        if span[0] - 0.2 <= a <= span[1] + 0.2 and (best is None or abs(np.log(k)) < abs(np.log(best[1]))):
            best = (w, k, Q)
    if best is None:
        return {"why": "no wall behind it"}
    (name, ax, at, sign, _), k, P = best
    o = 1 - ax  # along the wall
    a0, a1 = np.percentile(P[:, o], [2, 98])
    z0, z1 = np.percentile(P[:, 2], [2, 98])
    h_lo, h_hi, w_hi = SIZE[c["class"]]
    if a1 - a0 < 0.15 or not h_lo * (0.6 if c["frame_cut"] else 1.0) <= z1 - z0 <= h_hi or a1 - a0 > w_hi:
        return {"why": f"size on the wall {a1 - a0:.2f} x {z1 - z0:.2f}"}
    px = 0.004  # m per texel
    nw, nh = int(np.clip((a1 - a0) / px, 32, 1024)), int(np.clip((z1 - z0) / px, 32, 1024))
    A, Z = np.meshgrid(np.linspace(a0, a1, nw), np.linspace(z1, z0, nh))
    W = np.zeros((nh, nw, 3))
    W[..., o], W[..., 2], W[..., ax] = A, Z, at + sign * 0.02
    v = vf.to_cam(W.reshape(-1, 3))
    uv = v[:, :2] / v[:, 2:3] * np.array([vf.K[0, 0], vf.K[1, 1]]) + np.array([vf.K[0, 2], vf.K[1, 2]]) - 0.5
    tex = cv2.remap(vf.img, uv[:, 0].reshape(nh, nw).astype(np.float32), uv[:, 1].reshape(nh, nw).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = out / c["name"]
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tex).save(d / "panel.png")
    lo, hi = np.zeros(3), np.zeros(3)
    lo[o], hi[o], lo[2], hi[2] = a0, a1, z0, z1
    lo[ax], hi[ax] = sorted([at, at + sign * 0.02])
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]])
    proj = np.zeros(vf.shape, np.uint8)
    q = W[[0, 0, -1, -1], [0, -1, -1, 0]]
    qc = vf.to_cam(q)
    if (qc[:, 2] > 0).all():
        quv = qc[:, :2] / qc[:, 2:3] * np.array([vf.K[0, 0], vf.K[1, 1]]) + np.array([vf.K[0, 2], vf.K[1, 2]])
        cv2.fillPoly(proj, [np.round(quv).astype(np.int32)], 1)
    proj = proj.astype(bool)
    return {"iou": round(float(iou(proj, c["mask"])), 3), "box": corners, "center": (lo[:2] + hi[:2]) / 2,
            "proj": proj, "source": {"panel": str((d / "panel.png").relative_to(s.dir)), "wall": name, "axis": int(ax),
                                     "normal": float(sign)}}


def library_model(c, img):
    from calibrated_scene_reconstruction.assets.library import Embedder, crop_on_white, models, rank

    lib = set(models(c["class"]))
    if not lib:
        return None
    ranked = [(n, sim) for n, sim in rank(crop_on_white(img, c["mask"]), c["class"], k=400,
                                           embedder=Embedder()) if n in lib]
    return {"model": ranked[0][0], "sim": round(ranked[0][1], 3)} if ranked else None


def qa(s, tag, keep, sources, room, out_file):
    tiles = []
    for c in keep:
        img = sources[c["view"]].img.astype(np.float32).copy()
        if c.get("proj") is not None:
            img[c["proj"]] = 0.5 * img[c["proj"]] + 0.5 * np.array([255, 40, 40])
        edge = cv2.morphologyEx(c["mask"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        img[edge] = (60, 160, 255)
        ys, xs = np.nonzero(c["mask"])
        H, W = img.shape[:2]
        pad = int(0.06 * max(H, W))
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
        t = Image.fromarray(img[y0:y1, x0:x1].astype(np.uint8))
        t.thumbnail((300, 300))
        tile = Image.new("RGB", (300, 340), (24, 24, 24))
        tile.paste(t, ((300 - t.width) // 2, 0))
        d = ImageDraw.Draw(tile)
        src = c.get("source", {})
        what = "SAM 3D" if "mesh" in src else "panel" if "panel" in src else f"library {src.get('model', '')}"[:30]
        d.text((4, 302), f"{c['name']} {c['label']} ({c['view']})", fill=(255, 255, 0))
        d.text((4, 318), f"{what} IoU {c.get('iou', 0):.2f} h {c['box'][1][2] - c['box'][0][2]:.2f}", fill=(220, 220, 220))
        tiles.append(tile)
    f = room.floor
    sc = 296 / max(f[:, 1] - f[:, 0])
    top = Image.new("RGB", (300, 340), (30, 30, 30))
    d = ImageDraw.Draw(top)
    xy = lambda p: (2 + (p[0] - f[0, 0]) * sc, 300 - (p[1] - f[1, 0]) * sc)  # noqa: E731
    for p in room.photo[::60]:
        d.point(xy(p), fill=(110, 110, 110))
    zone = work_zone(s)
    if zone is not None:
        d.rectangle([*xy((zone[0, 0], zone[1, 1])), *xy((zone[1, 0], zone[0, 1]))], outline=(255, 140, 0))
    for c in keep:
        b = np.asarray(c["box"])
        d.rectangle([*xy((b[0, 0], b[1, 1])), *xy((b[1, 0], b[0, 1]))], outline=(0, 255, 0))
        d.text(xy((b[0, 0], b[1, 1])), c["name"].split("_")[-1], fill=(0, 255, 0))
    d.text((4, 318), f"{tag}: room from above, work zone orange", fill=(220, 220, 220))
    tiles.append(top)
    cols = 6
    S = Image.new("RGB", (cols * 300, ((len(tiles) + cols - 1) // cols) * 340), (12, 12, 12))
    for k, t in enumerate(tiles):
        S.paste(t, ((k % cols) * 300, (k // cols) * 340))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    S.save(out_file, quality=88)


_FIT = {}


def _fit_one(i):
    s, c, src = _FIT["s"], _FIT["keep"][i], _FIT["sources"][_FIT["keep"][i]["view"]]
    vf = src.frame(c.get("floor_scale", 1.0))
    return panel(s, c, vf, _FIT["out"], _FIT["wl"]) if c["kind"] == "flat" else reconstruct(s, c, vf, _FIT["out"])


def captured(s: Session):
    """The scene as captured, nothing generated (completion.json "source": "captured")."""
    return s.config("completion.json", {}).get("source") == "captured"


def run_tag(s, tag, args):
    pd = pano_dir(s, tag)
    cfg = s.config("completion.json", {})
    vocab = {**VOCAB, **{p: (p.split()[-1], 0.4, "floor") for p in cfg.get("objects", []) if p not in VOCAB}}
    room = Room(s)
    if captured(s):
        if tag != "twin":
            return None
        pd.mkdir(parents=True, exist_ok=True)
        sources = {"photo": photo_source(s), **multiview_sources(s), **posed_sources(s)}
    else:
        if not (pd / "views.npz").exists():
            print(f"[room_objects] {tag}: no completion in {pd}")
            return None
        views = load_views(s, pd)
        z = np.load(pd / "views.npz")
        deps = {"names": [str(n) for n in z["names"]], "depth": z["depth"], "ok": z["ok"], "K": z["K"],
                "cam": z["cam"]}
        sources = pano_sources(s, views, deps)
        sources.update(wall_sources(s, tag))
        if tag == "twin" and not args.no_photo:  # the photos' own background objects (the twin's room as it is)
            sources["photo"] = photo_source(s)
            sources.update(multiview_sources(s))
    cache = pd / "room_object_masks.npz"
    inst = None
    if args.reuse_masks and cache.exists():
        inst = np.load(cache, allow_pickle=True)["inst"].item()
        inst = {k: v for k, v in inst.items() if k in sources} if set(sources) <= set(inst) else None
    if inst is None:
        inst = segment_images({n: src.img for n, src in sources.items()}, vocab)
        np.savez_compressed(cache, inst=np.array(inst, dtype=object))
    keep, cands = candidates(s, sources, inst, vocab, room)
    out = s / "scene" / "room_objects" / tag
    out.mkdir(parents=True, exist_ok=True)
    wl = walls(s)
    for c in keep:
        c["name"] = piece_name(c)
        c["support_z"] = support_under(c, keep)
    adopt(out, keep)
    jobs = []
    for c in keep:  # SAM 3D for every solid piece first, a few at once
        if c["kind"] != "flat":
            inp = sam3d_inputs(s, c, sources[c["view"]].frame(c.get("floor_scale", 1.0)), out)
            if not (args.refit and (out / c["name"] / "raw.glb").exists()):
                (out / c["name"] / "raw.glb").unlink(missing_ok=True)
                jobs.append((c["name"], inp))
    sam3d_all(s, jobs, args.workers)
    from concurrent.futures import ThreadPoolExecutor

    _FIT.update(s=s, keep=keep, sources=sources, out=out, wl=wl)  # the fits in threads (numpy / OpenCV / the KD-tree
    with ThreadPoolExecutor(max(1, min(FIT_PROCS, len(keep)))) as ex:  # release the GIL; forked processes deadlock on
        results = list(ex.map(_fit_one, range(len(keep))))             # the parent's thread-pool locks)
    kept = []
    for c, r in zip(keep, results):
        if "why" not in r and c["kind"] == "flat" and r["iou"] < MIN_PANEL_IOU:
            r = {"why": f"panel IoU {r['iou']}"}
        if "why" not in r and c["kind"] != "flat" and r["iou"] < float(cfg.get("min_fallback_iou", 0.0)):
            # SAM 3D's fit this far off says the mask, not the reconstruction, is wrong: no library model stands in for it
            r = {"why": f"SAM 3D IoU {r['iou']} under min_fallback_iou"}
        if "why" not in r and (r["iou"] < MIN_IOU) and c["kind"] != "flat":
            lib = library_model(c, sources[c["view"]].img) if c["class"] in LIBRARY else None
            if lib:  # in the box its points span (the failed fit's is no measure), facing the camera it was seen
                b = np.asarray(c["box"], float)  # from (library fronts face -y)
                h_max, h_std = STAND_IN[c["class"]]
                if b[1, 2] - b[0, 2] > h_max:  # its box holds more than the piece: the class's own height
                    b = np.array([b[0], [*b[1, :2], b[0, 2] + h_std]])
                ctr = (b[0, :2] + b[1, :2]) / 2
                to_cam = sources[c["view"]].cam[:2] - ctr
                r = {"iou": r["iou"], "box": b, "center": ctr, "proj": None, "source": lib,
                     "fallback": f"SAM 3D IoU {r['iou']}", "yaw": float(np.arctan2(to_cam[0], -to_cam[1]))}
            else:
                r = {"why": f"SAM 3D IoU {r['iou']} and no library model of class {c['class']}"}
        if "why" not in r and "mesh" in r.get("source", {}):
            ext = np.ptp(np.asarray(r["box"], float), axis=0)
            if ext[2] > 0.8 and (ext[:2].min() < THIN or ext[2] > SLAB * ext[:2].min()):  # a part seen edge-on
                r = {"why": f"thin slab {np.round(ext, 2).tolist()}"}
        if "why" in r:
            c["why"] = r["why"]
            print(f"[room_objects] {tag} {c['name']} ({c['label']}, {c['view']}): dropped, {r['why']}", flush=True)
            continue
        c.update(r)
        kept.append(c)
        print(f"[room_objects] {tag} {c['name']} ({c['label']}, {c['view']}): {list(c['source'])[0]} IoU {c['iou']}"
              f" box {np.round(np.asarray(c['box']), 2).tolist()}", flush=True)
    for view in {c["view"] for c in kept if c["view"].startswith("bay:")}:  # a bay's furniture against its wall
        w = sources[view].wall  # (asked for so; the painted depth of its back is a guess): one move for all the bay
        ax, at, sign = w["axis"], w["at"], w["normal"]  # shows, so a chair stays at its desk, a monitor on it
        bay = [c for c in kept if c["view"] == view and c["kind"] != "flat"]
        main = [c for c in bay if c["class"] not in ("chair", "stool")] or bay
        if not main:
            continue
        gap = min(((np.asarray(c["box"])[0, ax] - at) if sign > 0 else (at - np.asarray(c["box"])[1, ax])) for c in main)
        if not 0.0 < gap < 1.2:
            continue
        d = np.zeros(3)
        d[ax] = -sign * (gap - 0.02)
        for c in bay:
            c["box"], c["center"] = np.asarray(c["box"], float) + d, np.asarray(c["center"], float) + d[:2]
    flats = sorted((c for c in kept if c["kind"] == "flat"), key=lambda c: -c["iou"])
    for i, c in enumerate(flats):  # one flat thing from several views, told apart before the fit only by their lifted
        if any(k in kept and flat_overlap(np.asarray(c["box"], float), np.asarray(k["box"], float))  # points
               for k in flats[:i]):
            kept.remove(c)
            print(f"[room_objects] {tag} {c['name']} ({c['label']}): dropped, the same panel as a better one", flush=True)
    for c in kept:  # a door stands on the floor: its mask stops at what stands in front of it
        b = np.asarray(c["box"], float)
        if c["kind"] == "flat" and c["class"] == "door" and 0.0 < b[0, 2] < 0.5:
            c["box"] = b - np.array([0.0, 0.0, b[0, 2]])
    for c in kept:  # a reconstruction reaching through a wall (its hidden back): moved in, its size kept
        if c["kind"] == "flat":
            continue
        b = np.asarray(c["box"], float)
        d = np.zeros(3)
        for a in (0, 1):
            d[a] = max(room.floor[a, 0] + 0.02 - b[0, a], 0.0) + min(room.floor[a, 1] - 0.02 - b[1, a], 0.0)
        c["box"], c["center"] = b + d, np.asarray(c["center"], float) + d[:2]
    for c in [c for c in kept if c["kind"] == "rests" and np.asarray(c["box"])[0, 2] > 0.15]:
        b = np.asarray(c["box"])  # a monitor or a box resting on a piece that was dropped would float
        if not any(k is not c and k["kind"] != "flat" and footprint_overlap(b, np.asarray(k["box"])) > 0.3
                   and abs(np.asarray(k["box"])[1, 2] - b[0, 2]) < 0.15 for k in kept):
            c["why"] = "nothing under it"
            print(f"[room_objects] {tag} {c['name']} ({c['label']}, {c['view']}): dropped, nothing under it", flush=True)
    kept = [c for c in kept if not c["why"]]
    qa(s, tag, kept, sources, room, s / "qa" / "room_objects" / f"{tag}.jpg")
    pieces = []
    for c in kept:
        b = np.asarray(c["box"], float)
        pieces.append({"name": c["name"], "label": c["label"], "class": c["class"], "kind": c["kind"],
                       "view": c["view"], "box": b.round(3).tolist(), "center": np.round(c["center"], 3).tolist(),
                       "height": round(float(b[1, 2] - b[0, 2]), 3), "yaw": round(c.get("yaw", 0.0), 3), "iou": c["iou"],
                       "source": c["source"], "synthetic": True, **({"fallback": c["fallback"]} if "fallback" in c
                                                                     else {})})
    import shutil

    for d in out.glob("room_*"):  # folders of pieces no longer kept (a panel of an earlier run, a dropped piece's
        if d.name not in {c["name"] for c in keep}:  # mesh is kept under its own name until the piece leaves keep)
            shutil.rmtree(d)
    rejected = [{k: c[k] for k in ("label", "view", "pixels", "why")} for c in cands if c["why"]]
    (s / "scene" / "room_objects" / f"{tag}.json").write_text(json.dumps({"pieces": pieces, "rejected": rejected},
                                                                          indent=1))
    print(f"[room_objects] {tag}: {len(pieces)} pieces "
          f"({sum('mesh' in p['source'] for p in pieces)} reconstructed, {sum('panel' in p['source'] for p in pieces)}"
          f" panels, {sum('model' in p['source'] for p in pieces)} library) of {len(cands)} instances", flush=True)
    return pieces


def apply(s: Session, tag, pieces):
    keys = ("label", "class", "kind", "view", "box", "center", "height", "yaw", "source", "synthetic")
    if tag == "twin":
        path = s.config_dir / "scene_graph.json"
        sg = json.loads(path.read_text())
        fixes = {k: v["fix"] for k, v in sg.get("synthetic_furniture", {}).items() if "fix" in v}
        sg["synthetic_furniture"] = {p["name"]: {**{k: p[k] for k in keys}, **({"fix": fixes[p["name"]]}
                                                                               if p["name"] in fixes else {})}
                                     for p in pieces}
        path.write_text(dumps(sg))
    else:
        path = s / "cousins" / f"{tag}.json"
        c = json.loads(path.read_text())
        fixes = {p["name"]: p["fix"] for p in c.get("synthetic_furniture", []) if "fix" in p}
        c["synthetic_furniture"] = [{"name": p["name"], **{k: p[k] for k in keys},
                                     **({"fix": fixes[p["name"]]} if p["name"] in fixes else {})} for p in pieces]
        path.write_text(json.dumps(c, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tags", default="", help="comma-separated (default: twin and every cousin)")
    ap.add_argument("--reuse-masks", action="store_true", help="keep the SAM 3 instances of an earlier run")
    ap.add_argument("--refit", action="store_true", help="keep SAM 3D meshes already there, fit them again")
    ap.add_argument("--apply", action="store_true", help="write the pieces into the scene graph / cousin files")
    ap.add_argument("--no-photo", action="store_true", help="the twin: only the completion (not the photo's objects)")
    ap.add_argument("--workers", type=int, default=SAM3D_WORKERS, help="SAM 3D processes at once")
    args = ap.parse_args()
    s = Session(args.session)
    for tag in [t for t in args.tags.split(",") if t] or tags(s):
        pieces = run_tag(s, tag, args)
        if pieces is not None and args.apply:
            apply(s, tag, pieces)
            print(f"[room_objects] {tag}: applied", flush=True)


if __name__ == "__main__":
    main()
