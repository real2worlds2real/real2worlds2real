"""The room's free walls, rendered for the wall completion (recon.wall_completion): one camera per bay -- a stretch of
wall with free floor in front of it, outside the work zone -- facing it from inside the room.

    tools/run_isaac.sh apps/scene/wall_views.py sessions/<name> [--tag twin | --tag cousin_00 --variant sessions/<name>/cousins/cousin_00.json]

A completion painted around the photo camera sees the room near the camera, where the work zone keeps furniture out,
and the far walls too small to reconstruct. A bay view looks at one wall straight on, at a known pose, over the
room's own structure, so what the completion paints there stands at a known place and scale (the rendered depth of
the bare wall and floor anchors it). Rendered without the room objects of an earlier pass (recon.room_objects
replaces them); bays avoid the work zone, the scene graph's furniture and set pieces, and the photo's own objects.

Writes scene/walls/<tag>/<bay>.npz (rgb, depth: distance to the image plane, K, T_world_cam OpenCV, wall: name,
axis, at, normal, along [lo, hi], depth of the strip) and <bay>.png, scene/walls/<tag>/bays.json.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 1024, 1024
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import trimesh  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402

from calibrated_scene_reconstruction.scene.graph import structure_role  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402

BAY_W = 2.4      # m of wall per bay
BAY_DEPTH = 1.0  # m of floor in front of the wall the completion may furnish
WORK_CLEAR = 1.3  # m round the task support (recon.room_objects)
CAM_D = 2.8      # m from the wall to the camera (less where the room is narrower)
CAM_Z, AIM_Z = 1.45, 0.95
FOCAL = 0.5 / np.tan(np.radians(40))  # 80 deg horizontal field of view

configure_render()
s = Session(args.session)
sg = s.scene_graph
st = sg["structure"]
floor = np.asarray(next(v for k, v in st.items() if structure_role(k, v) == "floor")["range"], float)
sup = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
blocked = [np.array([sup[0, :2] - WORK_CLEAR, sup[1, :2] + WORK_CLEAR])]  # footprints a bay strip must not touch
blocked += [np.asarray(f["box"], float)[:, :2] for f in sg.get("furniture", {}).values()]
T_wc = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"])
for name in s.manifest.get("static_objects", {}):  # the photo's own objects (boxes on the floor ...)
    f = s / "objects" / name / "mesh_cam.glb"
    if f.exists():
        V = np.asarray(trimesh.load(f, force="mesh").vertices) @ T_wc[:3, :3].T + T_wc[:3, 3]
        blocked.append(np.array([V.min(0)[:2], V.max(0)[:2]]))
robot = ((sg.get("photo_robot") or {}).get("base") or {}).get("xy")
if robot:
    blocked.append(np.array([np.asarray(robot) - 0.6, np.asarray(robot) + 0.6]))


def free(lo, hi):
    return not any(np.all(lo < b[1]) and np.all(hi > b[0]) for b in blocked)


bays = []
for wname, w in st.items():
    if structure_role(wname, w) != "wall" or w.get("plane") not in ("x", "y"):
        continue
    ax = 0 if w["plane"] == "x" else 1
    o, at, sign = 1 - ax, float(w["at"]), 1.0 if w.get("normal", "+") == "+" else -1.0
    a0, a1 = (float(v) for v in w["range"][0])
    a0, a1 = max(a0, floor[o, 0]), min(a1, floor[o, 1])
    n = int((a1 - a0) // BAY_W)
    if n < 1:
        continue
    pad = (a1 - a0 - n * BAY_W) / 2
    for k in range(n):
        c0 = a0 + pad + k * BAY_W
        lo, hi = np.zeros(2), np.zeros(2)
        lo[o], hi[o] = c0, c0 + BAY_W
        lo[ax], hi[ax] = sorted([at, at + sign * BAY_DEPTH])
        if not (floor[ax, 0] - 0.01 <= lo[ax] and hi[ax] <= floor[ax, 1] + 0.01) or not free(lo, hi):
            continue
        room_depth = (floor[ax, 1] - at) if sign > 0 else (at - floor[ax, 0])
        d = min(CAM_D, room_depth - 0.3)
        if d < 1.6:  # too narrow to see the bay whole
            continue
        mid = np.zeros(3)
        mid[o], mid[ax], mid[2] = c0 + BAY_W / 2, at, AIM_Z
        eye = mid + np.r_[sign * d * (np.arange(2) == ax), 0.0]
        eye[2] = CAM_Z
        f = (mid - eye) / np.linalg.norm(mid - eye)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.stack([r, -u, f], 1), eye  # OpenCV camera: x right, y down, z forward
        bays.append({"name": f"{wname}_{k}", "wall": wname, "axis": ax, "at": at, "normal": sign,
                     "along": [float(c0), float(c0 + BAY_W)], "depth": BAY_DEPTH, "T": T})
print(f"[wall_views] {args.tag}: {len(bays)} bays " + ", ".join(b["name"] for b in bays), flush=True)

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
variant = prepare_variant(s.dir, variant)
variant["synthetic_furniture"] = []  # the bare room (the completion's objects replace an earlier pass's)
variant["objects"] = []  # nothing on the support: only the room is looked at
build_scene(stage, prepare_scene(s.dir), variant)
world.reset()
K = np.array([[W * FOCAL, 0, W / 2], [0, W * FOCAL, H / 2], [0, 0, 1.0]])
anns = []
for i, b in enumerate(bays):
    cam = add_camera(stage, f"/World/Cams/bay_{i}", K, W, H, T_world_cam_cv=b["T"])
    rgb = rgb_annotator(cam.GetPath(), W, H)
    rp = rep.create.render_product(str(cam.GetPath()), (W, H))
    dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    dep.attach(rp)
    anns.append((rgb, dep))
for _ in range(12):
    world.step(render=True)
out = s / "scene" / "walls" / args.tag
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*"):
    if f.is_file():
        f.unlink()
for b, (rgb, dep) in zip(bays, anns):
    im = grab(rgb)
    d = np.asarray(dep.get_data(), np.float32)
    np.savez_compressed(out / f"{b['name']}.npz", rgb=im, depth=d, K=K, T_world_cam=b["T"],
                        wall=json.dumps({k: v for k, v in b.items() if k != "T"}))
    Image.fromarray(im).save(out / f"{b['name']}.png")
(out / "bays.json").write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
                                           for b in bays], indent=1))
print(f"[wall_views] {args.tag}: {len(bays)} bays -> {out}", flush=True)
app.close()
