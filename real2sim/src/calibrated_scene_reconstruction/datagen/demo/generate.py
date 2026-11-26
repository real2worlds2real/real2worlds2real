"""Many TCP plans of a demonstrated task over the twin and its cousins (the CPU half of demo data generation).

    python -m calibrated_scene_reconstruction.demo.generate sessions/<name> --demo real/<demo> [--cfg configs/demo_tasks/<task>.json] [--n 200]
                                  [--seed 0] [--fps 30] [--thr 0.5]

Reads the demo (demo.io), segments it against the twin's objects (demo.segment), derives the success spec from its end
state, measures each grasp of the demo on the grasped object's mesh (demo.functional.hold: functional or generic,
region, closing axis, in-hand pose, symmetry) and ranks candidate grasps against it (BODex records of the robot's hand
when grasping.synth --demo wrote them, else antipodal pairs for a parallel gripper). Then per draw: a scene variant
(demo.sampler: twin / cousin, object jitter, target augmentation, distractors, lights, cameras, robot base), one of
the best ranked grasps per hold, the segments re-targeted with its in-hand pose (functional.retarget_segments: the
approach relative to the TCP, held segments as object paths, `skills` pour / insert) and the TCP plan
(demo.retarget). Writes sessions/<name>/demo/<demo>/
  demo.json                 task, robot, segments, holds (measurements, region, ranked candidates), success spec, base
  plans/plan_XXXX.{npz,json}  ee (N, 7: xyz + quat wxyz, world), gripper, phase, segment; the variant, object poses,
                            per hold the chosen grasp (candidate, source, region, in-hand error), skills
Isaac Sim (TODO, apps/episode.py --plans): build the variant's scene, IK the plan (retarget.to_joints with
trajgen.TrajectoryPlanner.ik_from), validate in PhysX in parallel scene copies (--batch), check retarget.check_success
on the final object poses, record successes with meta["task"] = the task text; calibrated_scene_reconstruction.datagen.export.lerobot then exports them.

Object frames follow sim.scene.add_object: origin at the bottom centre of the world-aligned twin mesh, a cousin's
`xy` / `yaw` move and turn it about the world z.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from calibrated_scene_reconstruction.core.paths import ROOT

import numpy as np
from scipy.spatial.transform import Rotation

from . import functional, io, retarget, sampler, segment


def twin_poses(s, names):
    from calibrated_scene_reconstruction.assets.mesh_frames import twin_world
    out = {}
    for name in names:
        if not (s / "objects" / name / "mesh_cam.glb").exists():
            continue
        V = twin_world(s, name)
        out[name] = io.make_T(p=[(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2,
                                 V[:, 2].min()])
    return out


def scene_poses(twin, cousin=None):
    """The objects' poses in a cousin: its `objects` entries ("as": twin name) set xy and yaw, the twin's height kept.
    TODO: static_swaps that move a static object (a swapped container) are not followed yet."""
    out = {k: v.copy() for k, v in twin.items()}
    for spec in (cousin or {}).get("objects", []):
        name = spec.get("as", spec.get("name"))
        if name in out and spec.get("xy") is not None:
            R = Rotation.from_euler("z", spec.get("yaw", 0.0)).as_matrix()
            out[name] = io.make_T(R, [*spec["xy"][:2], out[name][2, 3]])
    return out


def radii_of(s, names):
    from calibrated_scene_reconstruction.assets.mesh_frames import twin_world
    out = {}
    for n in names:
        if (s / "objects" / n / "mesh_cam.glb").exists():
            V = twin_world(s, n)
            out[n] = float(np.max(np.linalg.norm(V[:, :2] - (V.min(0) + V.max(0))[:2] / 2, axis=1)))
    return out


def seated_mesh(s, name):
    """(V, F) of a twin object in its sim frame (bottom centre of the world-aligned mesh at the origin)."""
    import trimesh  # cousins.twin_world without its imports (cv2 is not in the grasp env that also calls this)

    m = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"] if (s / "scene" / "manifest.json").exists()
                   else s.world_frame()[2])
    V = np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]
    bottom = np.array([(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()])
    return V - bottom, np.asarray(m.faces)


def bodex_candidates(s, arm, name, region=None):
    """Successful BODex records of an object (grasps/<hand>/<key>*, the region's set first) as canonical TCP poses in
    its frame, with their record paths."""
    import yaml

    from calibrated_scene_reconstruction.assets.grasp_records import GRASP_DIR
    asset_path = ROOT / arm["ee"]["bodex"]["asset"]
    asset = yaml.safe_load(asset_path.read_text())
    T_bt = functional.hand_tcp(arm, asset["base_link"])
    C = functional.link_map(arm["tcp_axes"])
    dirs = sorted((s / GRASP_DIR / asset_path.stem).glob(f"twin_{name}_*"), key=lambda d: "__region" not in d.name)
    out, src = [], []
    for d in dirs[:1]:
        for f in sorted(d.glob("grasp_*.json")):
            r = json.loads(f.read_text())
            if r.get("success"):
                out.append(functional.bodex_tcp(r, T_bt) @ C)
                src.append(str(f.relative_to(s.dir)))
    return (np.stack(out) if out else np.zeros((0, 4, 4))), src


def opening_of(demo, segs, k):
    """The demo gripper's opening (m) while it held what it closed on at segment k: (1 - median position over the
    next segment, its ends trimmed) * the gripper's stroke (demo.json `gripper_max_width`); None without a stroke."""
    w = demo.meta.get("gripper_max_width")
    if not w or k + 1 >= len(segs):
        return None
    nx = segs[k + 1]
    g = np.asarray(demo.gripper[nx.start:nx.end + 1], float)
    g = g[len(g) // 10: len(g) - len(g) // 10] if len(g) > 10 else g
    return float((1.0 - np.median(g)) * w) if len(g) else None


def holds_of(s, segs, robot, cfg, demo=None):
    """Per close segment on a twin object: the demo's hold measured on its mesh (functional.hold; with the demo
    gripper's own opening when it has one), candidate grasps (BODex records when the robot's hand has an asset and
    they exist, else antipodal pairs for a parallel gripper, in the region when the hold is functional) ranked against
    it. -> {segment index: dict}."""
    from calibrated_scene_reconstruction.robots.spec import load_spec
    arm = next(iter(load_spec(robot)["arms"].values()))
    parallel = arm["ee"].get("type") == "parallel"
    max_w = float(arm["ee"].get("max_width", 0.085))
    out = {}
    for sg in segs:
        if sg.event != "close" or not sg.ref or not (s / "objects" / sg.ref / "mesh_cam.glb").exists():
            continue
        V, F = seated_mesh(s, sg.ref)
        opening = opening_of(demo, segs, sg.index) if demo is not None else None
        h = functional.hold(V, F, sg.ee_rel[-1], arm["tcp_axes"], max_w, cfg["grasp"], opening)
        region = h if h["functional"] else None
        cands, src = (bodex_candidates(s, arm, sg.ref, region) if arm["ee"].get("bodex") else (None, []))
        if cands is None or not len(cands):
            cands = functional.antipodal(V, F, region, max_w) if parallel else np.zeros((0, 4, 4))
            src = ["antipodal"] * len(cands)
        ranked = functional.rank(cands, h, cfg["grasp"], parallel)
        out[sg.index] = {"object": sg.ref, "hold": h, "V": V, "candidates": cands, "source": src, "ranked": ranked,
                         "parallel": parallel, "C": functional.link_map(arm["tcp_axes"])}
        m = h["measure"]
        print(f"[hold {sg.index}] {sg.ref}: {'functional' if h['functional'] else 'generic'} ({h['decided_by']}: "
              f"{', '.join(m['why']) or '-'}; held width {m.get('held_width_m', float('nan')):.3f} m on the mesh, "
              f"{m.get('opening_m', float('nan')):.3f} m by the gripper, of {m.get('object_width_m', float('nan')):.3f};"
              f" off axis {m['off_axis']:.2f}), symmetry "
              f"{h['symmetry']}; {len(cands)} candidates ({src[0].split('/')[0] if src else '-'}), "
              f"{len(ranked)} ranked" + (f", best {ranked[0][1] * 100:.1f} cm / {np.degrees(ranked[0][2]):.0f} deg"
                                         if ranked else ""))
    return out


def frame_map(holds):
    """Right-multiplied onto the demo's TCP poses: its gripper frame turned about the approach so that its closing axis
    is the one measured on the first hold's mesh (a dataset's TCP frame convention need not match the robot spec)."""
    for H in holds.values():
        h = H["hold"]
        if h["measure"]["closing"] is not None:
            return io.inv(np.asarray(h["T_obj_tcp"])) @ np.asarray(h["T_obj_tcp_canon"]) @ io.inv(H["C"])
    return np.eye(4)


def hold_summary(H):
    h = H["hold"]
    return {"object": H["object"], "functional": h["functional"], "decided_by": h["decided_by"],
            "measure": h["measure"], "symmetry": h["symmetry"], "symmetry_err": h["symmetry_err"],
            "region": {"center": h["center"], "radius": h["radius"], "part": h["part"], "contact": h["contact"]},
            "T_obj_tcp": h["T_obj_tcp"], "T_obj_tcp_canon": h["T_obj_tcp_canon"],
            "candidates": len(H["candidates"]), "ranked": [[int(i), float(p), float(np.degrees(r))]
                                                         for i, p, r in H["ranked"][:32]]}


def pick_grasps(holds, rng, top_k=8):
    """One ranked grasp per hold: {segment index: (T_old, T_new) TCP link poses in the object frame} and the record.
    Without a ranked candidate the demo's own in-hand pose is kept."""
    grasps, info = {}, []
    for k, H in holds.items():
        h, C = H["hold"], H["C"]
        T_old = np.asarray(h["T_obj_tcp_canon"]) @ io.inv(C)  # the demo's TCP with its measured closing axis
        rec = {"segment": k, "object": H["object"], "functional": h["functional"], "decided_by": h["decided_by"],
               "why": h["measure"]["why"], "region": {"center": h["center"], "radius": h["radius"], "part": h["part"]},
               "symmetry": h["symmetry"]}
        if H["ranked"]:
            i, pe, re = H["ranked"][int(rng.integers(min(top_k, len(H["ranked"]))))]
            T = H["candidates"][i]
            if H["parallel"]:  # the finger-swapped twin rank() matched
                T = min((T, functional.flip(T)), key=lambda X: np.dot(functional.inhand_error(
                    X, h["T_obj_tcp_canon"], h["symmetry"]), [1.0, 0.05]))
            grasps[k] = (T_old, T @ io.inv(C))
            rec.update(candidate=int(i), source=H["source"][i], inhand_err_m=pe, inhand_err_deg=float(np.degrees(re)))
        else:
            grasps[k] = (T_old, T_old)
            rec.update(candidate=None, source="demo", inhand_err_m=0.0, inhand_err_deg=0.0)
        info.append(rec)
    return grasps, info


def main(argv=None, *, session_factory=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--demo", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--thr", type=float, default=0.5, help="gripper threshold (DROID: 0.3, it stops at the object)")
    a = ap.parse_args(argv)
    if session_factory is None:
        raise SystemExit("Use tools/datagen/generate.py to compose prepared scene inputs")
    s, cfg = session_factory(a.session), sampler.load_cfg(a.cfg)
    sg = s.scene_graph
    demo = io.resample(io.load(a.demo, s.world_frame()[2]), a.fps)
    names = list(sg.get("dynamic_objects", {})) + list(sg.get("static_objects", {}))
    twin = twin_poses(s, names)
    segs, final = segment.segment(demo, twin, near=cfg["near"], thr=a.thr)
    holds = holds_of(s, segs, demo.robot, cfg, demo)
    M = frame_map(holds)
    segs = [replace(x, ee_rel=x.ee_rel @ M) for x in segs]  # the demo's gripper frame -> the robot's TCP link frame
    spec = retarget.success_spec(segs, final, twin, **cfg["success"])
    task = cfg["task"] or demo.task
    out = s / "demo" / Path(a.demo).name
    (out / "plans").mkdir(parents=True, exist_ok=True)
    s.write_json(out.relative_to(s.dir) / "demo.json", {
        "task": task, "robot": demo.robot, "side": demo.side, "segments": [x.to_dict() for x in segs],
        "success": spec, "T_world_base": demo.T_world_base.tolist(),
        "calibration_rms_m": demo.meta.get("calibration_rms_m"),
        "tcp_frame_map_deg": float(np.degrees(Rotation.from_matrix(M[:3, :3]).magnitude())),
        "holds": {k: hold_summary(H) for k, H in holds.items()}})
    for x in segs:
        print(f"[segment {x.index}] frames {x.start}-{x.end} {x.event} ref={x.ref} held={x.held} skill from +{x.skill}")
    if spec is None:
        print("no grasp in the demo: success spec undefined (articulated / non-prehensile tasks are TODO)")

    scenes = ["twin"] + sorted(p.stem for p in (s / "cousins").glob("cousin_*.json"))
    cousins = {t: (None if t == "twin" else json.loads((s / "cousins" / f"{t}.json").read_text())) for t in scenes}
    support = sg.get("task", {}).get("support")
    support_box = sg["furniture"][support]["box"] if support in sg.get("furniture", {}) else None
    boxes = [v["box"] for v in sg.get("furniture", {}).values()]
    radii = radii_of(s, list(cfg["objects"]))
    T_home = demo.ee[0]  # the demo's start relative to its base, carried by the jittered base
    rng, ok = np.random.default_rng(a.seed), 0
    for i in range(a.n):
        v = sampler.sample(cfg, rng, scenes)
        poses = sampler.perturb_objects(scene_poses(twin, cousins[v["scene"]]), cfg, rng, radii, support_box)
        T_base = sampler.base_pose(demo.T_world_base, v)
        T_start = T_base @ io.inv(demo.T_world_base) @ T_home
        grasps, info = pick_grasps(holds, rng, cfg["grasp"].get("top_k", 8))
        segs_i = functional.retarget_segments(segs, grasps, cfg["skills"], {H["object"]: H["V"] for H in holds.values()})
        p = retarget.plan(segs_i, poses, T_start, a.fps, cfg["speed"], boxes)
        if p is None:
            continue
        np.savez_compressed(out / "plans" / f"plan_{i:04d}.npz", ee=io.T_to_pose7(p["ee"]), gripper=p["gripper"],
                            phase=p["phase"], segment=p["segment"])
        s.write_json(out.relative_to(s.dir) / "plans" / f"plan_{i:04d}.json", {
            "task": task, "robot": demo.robot, "variant": v, "T_world_base": T_base.tolist(),
            "objects": {k: T.tolist() for k, T in poses.items()}, "success": spec, "grasps": info,
            "skills": {str(x.index): cfg["skills"].get(str(x.index), {"type": "transport"}) for x in segs if x.held}})
        ok += 1
    print(f"{ok}/{a.n} plans -> {out}/plans (blocked bridges dropped); Isaac Sim playback: TODO (apps/episode.py)")


if __name__ == "__main__":
    main()
