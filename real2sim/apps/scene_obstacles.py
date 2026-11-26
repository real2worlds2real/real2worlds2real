"""The environment around the task target, as the simulator builds it, for environment-aware grasp synthesis.

    tools/run_isaac.sh apps/scene_obstacles.py sessions/<name> [--tags twin cousin_00 ...] [--radius 0.45]

For each scene (the twin and cousins/<tag>.json) the stage is built exactly as the episodes build it (sim.scene:
penetration-resolved placements, library swaps, templates), then every collider near the target is written as an
box (world frame): the room's furniture collision boxes and walls / windows (sim.clearance.room_obstacles, walls
thickened behind their surface), and the static objects and other manipulable objects as height-field columns of their
surface (--cell wide, from the lowest to the highest surface point in the column; runs of columns of about the same
span merged): a loaded tray, a stack of books or a cup stays the shape it is, where one bounding box per item would
wall off its whole footprint (a tray's box swallows the space beside the cups on it). Stools are left out (episodes
slide them out of the robot's way). Boxes farther than --radius from the target are dropped. The target's pose is
the one it rests in after --settle seconds of physics, as the episodes find it: a reconstruction whose bottom is not
level in its own frame tips onto the face it can stand on, and the synthesis, which works
in the target's frame, must see the support and the neighbours tilted the same way (records stay in the prim frame;
apps/episode.py maps them with the settled pose).
Writes grasps/_env/<tag>.json: {"target": {"pos", "quat_wxyz"}, "obstacles": [...]},
read by calibrated_scene_reconstruction.grasping.synth (BODex scene constraint) and apps/episode.py (trajectory checks)."""
import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--tags", nargs="*", help="twin and/or cousin tags (default: twin and every cousin)")
ap.add_argument("--radius", type=float, default=0.45)
ap.add_argument("--out", help="output directory (default <session>/grasps/_env)")
ap.add_argument("--cell", type=float, default=0.03, help="height-field column width of object obstacles (m)")
ap.add_argument("--settle", type=float, default=1.5, help="seconds of physics before the target's pose is read (0: as built)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import trimesh  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import room_obstacles  # noqa: E402
from calibrated_scene_reconstruction.scene.cousins.randomization import variant_path  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import load_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import children  # noqa: E402

s = Session(args.session)
target = s.scene_graph["task"]["target"]
tags = args.tags or ["twin"] + sorted(f.stem for f in (s / "cousins").glob("cousin_*.json"))


def box_dist(c, half, R, p):
    """Distance from point p to an oriented box (0 inside)."""
    q = np.abs(R.T @ (p - c)) - half
    return float(np.linalg.norm(np.maximum(q, 0.0)))


def surface_points(prim, n=20000):
    """Points spread over the surface of every mesh under `prim` (world frame), by area."""
    cache = UsdGeom.XformCache()
    meshes = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p)
        V, counts, idx = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
        if not V or not counts:
            continue
        M = np.asarray(cache.GetLocalToWorldTransform(p))
        V = (np.c_[np.asarray(V, float), np.ones(len(V))] @ M)[:, :3]
        counts, idx = np.asarray(counts), np.asarray(idx)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        F = [(idx[s], idx[s + k], idx[s + k + 1]) for s, c in zip(starts, counts) for k in range(1, c - 1)]  # fans
        if F:
            meshes.append(trimesh.Trimesh(V, np.asarray(F), process=False))
    if not meshes:
        return np.zeros((0, 3))
    mesh = trimesh.util.concatenate(meshes)
    return np.concatenate([np.asarray(mesh.sample(n)), np.asarray(mesh.vertices)[:: max(1, len(mesh.vertices) // n)]])


def columns(P, name, cell, merge=0.01, min_half=0.005):
    """Height-field boxes of a point set: one per `cell` x `cell` column (lowest to highest point), consecutive columns
    of a row merged while their spans agree within `merge`."""
    if len(P) < 4:
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    cols = {}
    for (i, j), z in zip(map(tuple, ij), P[:, 2]):
        lo, hi = cols.get((i, j), (z, z))
        cols[(i, j)] = (min(lo, z), max(hi, z))
    boxes, rows = [], {}
    for (i, j), span in cols.items():
        rows.setdefault(j, []).append((i, *span))
    for j, row in rows.items():
        row.sort()
        run = [row[0]]
        for c in row[1:] + [None]:
            if c is not None and c[0] == run[-1][0] + 1 and abs(c[1] - run[0][1]) < merge and abs(c[2] - run[0][2]) < merge:
                run.append(c)
                continue
            i0, i1 = run[0][0], run[-1][0]
            lo, hi = min(r[1] for r in run), max(r[2] for r in run)
            boxes.append({"name": f"{name}#{len(boxes)}",
                          "center": [(i0 + i1 + 1) * cell / 2, (j + 0.5) * cell, (lo + hi) / 2],
                          "half": [(i1 - i0 + 1) * cell / 2, cell / 2, max((hi - lo) / 2, min_half)],
                          "R": np.eye(3).tolist()})
            run = [c]
    return boxes


out_dir = Path(args.out) if args.out else s / "grasps" / "_env"
out_dir.mkdir(parents=True, exist_ok=True)
for tag in tags:
    t0 = time.time()
    variant = load_variant(variant_path(s, tag))
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    ts = build_scene(stage, prepare_scene(s.dir), prepare_variant(s.dir, variant))
    tprim = stage.GetPrimAtPath(f"/World/Objects/{target}")
    M = np.asarray(UsdGeom.Xformable(tprim).ComputeLocalToWorldTransform(0)).T
    p_t = M[:3, 3]
    from scipy.spatial.transform import Rotation

    q_t = Rotation.from_matrix(M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)).as_quat(scalar_first=True)
    tilt = 0.0
    if args.settle > 0:  # the pose the target rests in, as apps/episode.py reads it after its own settle
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
        world.reset()
        for _ in range(int(args.settle * 120)):
            world.step(render=False)
        p_t, q_t = (np.asarray(v, float) for v in SingleRigidPrim(f"/World/Objects/{target}").get_world_pose())
        up = Rotation.from_quat(q_t, scalar_first=True).apply([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        world.stop()
        World.clear_instance()
    boxes = []
    for name, (lo, hi) in room_obstacles(ts).items():
        boxes.append({"name": f"room/{name}", "center": [float(v) for v in (lo + hi) / 2],
                      "half": [float(v) for v in np.maximum((hi - lo) / 2, 0.005)], "R": np.eye(3).tolist()})
    for root in ("/World/Static", "/World/Objects"):
        for prim in children(stage, root):
            if root == "/World/Objects" and prim.GetName() == target:
                continue
            boxes += columns(surface_points(prim), prim.GetPath().pathString, args.cell)
    near = [b for b in boxes if box_dist(np.array(b["center"]), np.array(b["half"]), np.array(b["R"]), p_t)
            < args.radius]
    (out_dir / f"{tag}.json").write_text(json.dumps({
        "target": {"name": target, "pos": [float(v) for v in p_t], "quat_wxyz": [float(v) for v in q_t],
                   "settled_s": args.settle, "tilt_deg": tilt},
        "radius": args.radius, "obstacles": near}, indent=1))
    items = sorted({b["name"].split("/")[-1].split("#")[0] for b in near})
    print(f"[{tag}] {len(near)} obstacle boxes of {len(items)} items within {args.radius} m of {target}: {items}, "
          f"target at rest tilted {tilt:.1f} deg [{time.time() - t0:.0f} s]", flush=True)
app.close()
