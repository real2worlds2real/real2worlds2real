"""The twin with its room written as code (sim.code_scene) rendered from the walk's own cameras, beside the walk's
frames -- the check of recon.code_scene's scene against what the camera saw: qa/code/compare.jpg (per frame: the
frame | the render) and qa/code/compare.json (per frame the mean absolute difference, the render's and the frame's
mean brightness).

    tools/run_isaac.sh apps/scene/code_compare.py sessions/<name> [--frames 8] [--renderer pt] [--spp 64]
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--names", default="", help="comma-separated walk frames (default: spread over the walk, wide ones)")
ap.add_argument("--renderer", default="rt", choices=["rt", "pt"])
ap.add_argument("--spp", type=int, default=64)
ap.add_argument("--tag", default="")
ap.add_argument("--background", default="code", choices=["code", "splat"], help="splat: the twin's splat (its exposure against the frames)")
ap.add_argument("--tonemap", type=int, default=-1, help="RTX tone mapping operator (/rtx/post/tonemap/op), -1 Kit's default")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

PT = "--renderer" in sys.argv and sys.argv[sys.argv.index("--renderer") + 1] == "pt"  # before the app starts
app = SimulationApp({"headless": True, "renderer": "PathTracing" if PT else "RaytracedLighting", "width": 1280,
                     "height": 720})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.code_stage import add_code_background  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, quat_from_matrix, set_pose  # noqa: E402

configure_render(mode=args.renderer, spp=args.spp, auto_exposure=False)  # the frames' own exposure is the target
s = Session(args.session)
z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
names = [str(n) for n in z["names"]]
W, H = (int(v) for v in z["size"][0])
w, h = W // 2, H // 2

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(s.dir, {})))
if args.background == "splat":
    from calibrated_scene_reconstruction.sim.splat_stage import add_background  # noqa: E402

    add_background(stage, prepare_background(s.dir), "splat")
else:
    add_code_background(stage, prepare_background(s.dir))
if args.tonemap >= 0:
    import carb  # noqa: E402

    carb.settings.get_settings().set("/rtx/post/tonemap/op", args.tonemap)
cam = add_camera(stage, "/World/Cams/compare", z["K"][0] * np.array([[0.5], [0.5], [1]]), w, h,
                 T_world_cam_cv=np.eye(4), far=60.0)
rp = rep.create.render_product(str(cam.GetPath()), (w, h))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
world.reset()
for _ in range(10):
    world.step(render=False)

if args.names:
    pick = [names.index(n) for n in args.names.split(",")]
else:  # spread over the walk, the frames that see farthest (the room, not the desk's close-ups)
    sg = s.scene_graph
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    c = np.r_[(lo[:2] + hi[:2]) / 2, hi[2]]
    far = [i for i in range(len(names)) if np.linalg.norm(z["centre"][i] - c) > 1.2]
    pick = [far[k] for k in np.linspace(0, len(far) - 1, args.frames).astype(int)]
rows, rep_ = [], []
for i in pick:
    K = z["K"][i]
    cam.GetHorizontalApertureAttr().Set(float(W / K[0, 0] * 18.0))
    cam.GetVerticalApertureAttr().Set(float(H / K[1, 1] * 18.0))
    set_pose(cam.GetPrim(), z["centre"][i], quat_from_matrix(z["R_world_cam"][i] @ np.diag([1, -1, -1])))
    for _ in range(8):
        world.step(render=True)
    data = np.asarray(ann.get_data())
    for _ in range(40):  # (path tracing hands the first frames back empty)
        if data.ndim == 3 and data.shape[0] == h:
            break
        world.step(render=True)
        data = np.asarray(ann.get_data())
    rgb = np.ascontiguousarray(data[..., :3])
    real = cv2.resize(cv2.cvtColor(cv2.imread(str(img_dir / names[i])), cv2.COLOR_BGR2RGB), (w, h))
    rows.append(np.hstack([real, rgb]))
    rep_.append({"frame": names[i], "mad": round(float(np.abs(real.astype(float) - rgb).mean()), 1),
                 "mean_frame": round(float(real.mean()), 1), "mean_render": round(float(rgb.mean()), 1)})
    print(f"[compare] {names[i]}: mean |frame - render| {rep_[-1]['mad']}, brightness {rep_[-1]['mean_frame']} / "
          f"{rep_[-1]['mean_render']}", flush=True)
out = s / "qa" / "code"
out.mkdir(parents=True, exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""
sheet = np.vstack([cv2.resize(r, (2 * 640, 360)) for r in rows])
cv2.imwrite(str(out / f"compare{tag}.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
(out / f"compare{tag}.json").write_text(json.dumps(rep_, indent=1) + "\n")
print(f"done: {len(rows)} frames -> {out / f'compare{tag}.jpg'}", flush=True)
app.close()
