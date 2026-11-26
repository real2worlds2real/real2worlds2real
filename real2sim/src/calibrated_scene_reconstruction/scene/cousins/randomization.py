"""Domain randomization for batch data collection: configurations of a scene (the twin or a cousin) that differ in
what a policy must not overfit to, each written as a variant beside the cousins and used by its tag like one
(apps/scene_obstacles.py, calibrated_scene_reconstruction.grasping.synth, apps/episode.py):

    python -m calibrated_scene_reconstruction.dr sessions/<name> --scenes twin cousin_00 ... --per-scene 8 [--seed 0]
    -> sessions/<name>/dr/<scene>__d<k>.json   (tag <scene>__d<k>)

Per configuration (DR ranges; tools/collect_place.py runs the stages over them):
  target       its start anywhere free on the support within a window of the task frame (right / forward of the
               scene's own position: the robot's placement reaches there and follows it), clear of the other objects
               and of keep-out areas, at a random yaw; --target-window: a window
               in the base frame of the photo's robot instead, for a lab whose robot stands at that one place
  distractors  each kept with p_keep, then jittered; a kept one that no longer fits stays where it was
  lights       the scene's light scale x [0.55, 1.6], colour temperature 3600-7200 K, the sun on or off
  camera       the head camera's mount: +-1 cm and +-1.5 deg (a calibration's error), recorded per frame
  physics      the target's mass x [0.7, 1.4], its friction 0.8-1.2 (PhysX's default average with the hand's and the
               support's)
  start on     (p_start_on, --start-on) the target stands on a cutting board, a plate or a tray (assets.receptacles,
               a static receptacle under it: `receptacles` role "start", the target's `support_z` on its surface)
  tray         (p_tray, --tray) a tray elsewhere on the support (role "surface"): apps/episode.py may put the goal
               marker on its floor
  a flat set piece on the support is part of the top: a target or
  a receptacle over it stands on it
Per episode apps/episode.py draws the rest: the robot base among the accepted placements, the arm's start pose, the
trajectory (Bezier transit, standoff, speed), the grasp, and the goal marker's place and size."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.graph import dumps
from calibrated_scene_reconstruction.scene.session import Session

RANGES = {
    "target_right": (-0.2, 0.2), "target_forward": (-0.35, 0.08), "gap": 0.02,
    "p_keep": 0.7, "jitter": 0.06,
    "light_scale": (0.55, 1.6), "light_temp": (3600.0, 7200.0), "p_sun": 0.5,
    "cam_pos": 0.01, "cam_rot_deg": 1.5,
    "mass_scale": (0.7, 1.4), "friction": (0.8, 1.2),
    "p_start_on": 0.0, "start_on": ("cutting_board", "plate", "tray"), "p_tray": 0.0,
    "tray_right": (-0.45, 0.45), "tray_forward": (-0.35, 0.10),
    "clear_target": 0.0,  # m: distractors kept at least this far from the target (0: as they stand)
    "target_window": None,  # (fwd lo, hi, left lo, hi) m in the photo robot's base frame: replaces the task window
}


def flat_pieces(sg, support):
    """[(centre xy, half (x, y), yaw, height above the top)] of the flat set pieces set into the support's top."""
    out = []
    for name, p in (sg.get("set_pieces") or {}).items():
        if p.get("on") != support or p.get("kit") != "cooktop":
            continue
        h = 0.006 - float(p.get("inset", 0.0))  # assets.kits.cooktop: a 6 mm panel
        out.append((np.asarray(p["center"], float), np.array([p["length"], p["depth"]]) / 2,
                    np.radians(p.get("yaw_deg", 0.0)), max(h, 0.0)))
    return out


def raised(pieces, xy, r):
    """Height of the flat pieces under a footprint (circle xy, r) over the top (0 off them)."""
    z = 0.0
    for c, half, yaw, h in pieces:
        d = np.asarray(xy, float) - c
        loc = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
        if np.all(np.abs(loc) < half + r):
            z = max(z, h)
    return z


def scene_variant(s: Session, scene: str):
    """The variant a configuration starts from: a cousin's file, or the twin's objects at their own places."""
    if scene != "twin":
        return json.loads((s / "cousins" / f"{scene}.json").read_text())
    from calibrated_scene_reconstruction.scene.cousins.variants import twin_xy

    dyn = s.scene_graph["dynamic_objects"]
    return {"objects": [{"name": n, "as": n, "mass": d.get("mass", 0.3), "xy": [float(v) for v in twin_xy(s, n)]}
                        for n, d in dyn.items()]}


def _radius(s, spec):
    if spec.get("ph_model"):  # a library model: its height is set, its footprint about as wide
        return 0.5 * float(spec.get("height", 0.1)) + 0.03
    from calibrated_scene_reconstruction.scene.cousins.variants import footprint_radius

    return footprint_radius(s, spec)


def _height(s, spec):
    if spec.get("ph_model"):
        return float(spec.get("height", 0.1))
    if spec.get("mesh_path"):
        import trimesh

        from calibrated_scene_reconstruction.core.paths import resolve

        V = np.asarray(trimesh.load(resolve(spec["mesh_path"]), force="mesh").vertices)
    else:
        from calibrated_scene_reconstruction.scene.cousins.variants import twin_world

        V = twin_world(s, spec["name"])
    return float(np.ptp(V[:, 2])) * float(spec.get("scale", 1.0))


def _rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def keep_out_boxes(sg, margin=0.0):
    areas = (((sg.get("task") or {}).get("place") or {}).get("keep_out") or {}).values()
    return [(np.asarray(a[0], float) - margin, np.asarray(a[1], float) + margin) for a in areas]


def _free(xy, r, lo, hi, others, keep_out, gap):
    xy = np.asarray(xy, float)
    if np.any(xy - r - gap < lo) or np.any(xy + r + gap > hi):
        return False
    if any(np.all(xy + r + gap > a) and np.all(xy - r - gap < b) for a, b in keep_out):
        return False
    return all(np.linalg.norm(xy - np.asarray(o)) > r + ro + gap for o, ro in others)


def sample_config(s: Session, scene: str, rng, ranges=None, same_target=False):
    """One randomized variant of `scene` (a dict; `dr` records what was drawn), or None if the target found no free
    place. same_target: the twin's own target in every scene (the cousins' augmented or rescaled ones replaced: one
    object to learn first)."""
    from calibrated_scene_reconstruction.scene.cousins.variants import neighbour_points
    from calibrated_scene_reconstruction.robots.taskframe import TaskFrame

    c = {**RANGES, **(ranges or {})}
    U = rng.uniform
    sg = s.scene_graph
    target = sg["task"]["target"]
    box = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
    lo, hi, top = box[0, :2], box[1, :2], float(box[1, 2])
    v = json.loads(json.dumps(scene_variant(s, scene)))
    objs = v["objects"]
    t = next(o for o in objs if o.get("as", o.get("name")) == target)
    if same_target:
        keep = {k: t[k] for k in ("xy", "yaw", "mass") if k in t}
        t.clear()
        t.update(name=target, **{"as": target}, **keep)
    t0 = np.asarray(t["xy"], float)
    tf = TaskFrame(sg, [t0[0], t0[1], top])
    keep_out = keep_out_boxes(sg)
    static = neighbour_points(s, set(sg["dynamic_objects"]), top)  # the static objects standing on the support
    static_r = [(p, 0.0) for p in static[:: max(1, len(static) // 300)]]

    # distractors: some removed, the rest jittered where they stay free of each other
    rest, removed = [], []
    for o in objs:
        if o is t:
            continue
        if U() > c["p_keep"]:
            removed.append(o.get("as", o.get("name")))
            continue
        r = _radius(s, o)
        xy0 = np.asarray(o["xy"], float)
        for _ in range(30):
            xy = xy0 + rng.normal(0.0, c["jitter"] / 2, 2)
            if _free(xy, r, lo, hi, static_r + [(p["xy"], p["_r"]) for p in rest], keep_out, c["gap"]):
                o["xy"] = [float(a) for a in xy]
                break
        rest.append({**o, "_r": r})
    # the target: free of the static objects, the kept distractors and the keep-out areas (with a receptacle under
    # it: the receptacle's footprint, the target at its accept centre)
    from calibrated_scene_reconstruction.assets import receptacles as rcp

    r_t = _radius(s, t)
    h_t = _height(s, t)
    others = static_r + [(p["xy"], p["_r"]) for p in rest]
    under = None
    if U() < c["p_start_on"]:
        kinds = list(c["start_on"])
        while kinds and under is None:
            kind = kinds.pop(int(rng.integers(len(kinds))))
            under = rcp.make(kind, r_t, h_t, int(rng.integers(2 ** 31)))
    r_need = max(r_t, float(np.hypot(*under.half))) if under is not None else r_t
    y_u = float(U(-np.pi, np.pi))
    win = c.get("target_window")
    if win:  # where the robot will stand (scene graph photo_robot.base): a cousin's own target position moves the
        pb = sg["photo_robot"]["base"]  # task window, and from that one base the arm reaches a narrow band only
        a = np.radians(float(pb["yaw_deg"]))
        b_xy = np.asarray(pb["xy"], float)
        b_fwd, b_left = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    for _ in range(400):
        if win:
            xy = b_xy + U(win[0], win[1]) * b_fwd + U(win[2], win[3]) * b_left
        else:
            xy = t0 + (U(*c["target_right"]) * tf.right + U(*c["target_forward"]) * tf.forward)[:2]
        o = xy if under is None else xy - _rot2(y_u) @ np.asarray(under.accept_xy[:2], float)
        if _free(o, r_need, lo, hi, others, keep_out, c["gap"]):
            break
    else:
        return None
    t["xy"], t["yaw"] = [float(a) for a in xy], float(U(-np.pi, np.pi))
    if c.get("clear_target", 0.0) > 0:  # distractors kept off the pick and place: beyond clear_target of the target
        t_xy, kept = np.asarray(t["xy"], float), []  # (moved to a free spot of the support, else removed)
        for p in rest:
            if np.linalg.norm(np.asarray(p["xy"], float) - t_xy) >= c["clear_target"] + p["_r"]:
                kept.append(p)
                continue
            others_ = static_r + [(q["xy"], q["_r"]) for q in kept] + [(t_xy, r_t)]
            for _ in range(200):
                xy_ = np.array([U(lo[0], hi[0]), U(lo[1], hi[1])])
                if np.linalg.norm(xy_ - t_xy) >= c["clear_target"] + p["_r"] and \
                        _free(xy_, p["_r"], lo, hi, others_, keep_out, c["gap"]):
                    kept.append({**p, "xy": [float(a) for a in xy_]})
                    break
            else:
                removed.append(p.get("as", p.get("name")))
        rest = kept
    pieces = flat_pieces(sg, sg["task"]["support"])
    recs = []
    if under is not None:
        z0 = top + raised(pieces, o, r_need)
        recs.append({**under.to_dict(), "origin": [float(o[0]), float(o[1]), z0], "yaw": y_u, "role": "start"})
        t["support_z"] = z0 + float(under.place_z)
        others = others + [(o, r_need)]
    else:
        dz = raised(pieces, xy, 0.0)
        if dz:
            t["support_z"] = top + dz
        others = others + [(xy, r_t)]
    if U() < c["p_tray"]:  # a tray elsewhere on the top: the goal marker may go on its floor
        tray = rcp.make("tray", 1.12 * r_t, 0.01, int(rng.integers(2 ** 31)))
        y_t = float(U(-np.pi, np.pi))
        r_tr = float(np.hypot(*tray.half))
        for _ in range(200):
            o_t = t0 + (U(*c["tray_right"]) * tf.right + U(*c["tray_forward"]) * tf.forward)[:2]
            if _free(o_t, r_tr, lo, hi, others, keep_out, c["gap"]):
                recs.append({**tray.to_dict(), "origin": [float(o_t[0]), float(o_t[1]),
                                                          top + raised(pieces, o_t, r_tr)], "yaw": y_t,
                             "role": "surface"})
                break
    if recs:
        v["receptacles"] = recs
    mass = float(t.get("mass", 0.3)) * U(*c["mass_scale"])
    t["mass"] = mass
    v["objects"] = [t] + [{k: val for k, val in o.items() if k != "_r"} for o in rest]

    li = dict(v.get("lights") or {})
    li["scale"] = float(li.get("scale", 1.0) * U(*c["light_scale"]))
    li["temp"] = float(U(*c["light_temp"]))
    li["sun"] = bool(U() < c["p_sun"])
    v["lights"] = li
    cam = {"pos": rng.normal(0.0, c["cam_pos"] / 2, 3).round(4).tolist(),
           "rot_deg": rng.normal(0.0, c["cam_rot_deg"] / 2, 3).round(3).tolist()}
    fr = float(U(*c["friction"]))
    v["dr"] = {"scene": scene, "camera": cam, "removed": removed, "same_target": bool(same_target),
               "physics": {"mass": mass, "friction": fr},
               "receptacles": [{"role": r["role"], "kind": r["kind"]} for r in recs]}
    return v


def write_configs(s: Session, scenes, per_scene, seed=0, ranges=None, same_target=False, letter="d"):
    """sessions/<name>/dr/<scene>__<letter><k>.json for every scene (another letter: another collection beside
    the first); returns the tags."""
    out = s / "dr"
    out.mkdir(exist_ok=True)
    tags = []
    for scene in scenes:
        rng = np.random.default_rng([seed, sum(map(ord, scene))])
        k = 0
        for _ in range(4 * per_scene):
            if k == per_scene:
                break
            v = sample_config(s, scene, rng, ranges, same_target)
            if v is None:
                continue
            tag = f"{scene}__{letter}{k:02d}"
            (out / f"{tag}.json").write_text(dumps(v))
            tags.append(tag)
            k += 1
    return tags


def variant_path(s: Session, tag: str):
    """The stored variant JSON for a tag; None for the twin."""
    from calibrated_scene_reconstruction.core.artifacts import tagged_json_path

    return tagged_json_path(s, tag, ("cousins", "dr"), identity_tag="twin", label="variant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--scenes", nargs="*", help="default: the twin and every cousin")
    ap.add_argument("--per-scene", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-target", action="store_true", help="the twin's target in every scene")
    ap.add_argument("--letter", default="d", help="tags <scene>__<letter><k>: a second collection beside the first")
    ap.add_argument("--start-on", type=float, default=0.0, help="p_start_on: the target on a board / plate / tray")
    ap.add_argument("--tray", type=float, default=0.0, help="p_tray: a tray on the support for the goal marker")
    ap.add_argument("--target-right", default=None, help="lo,hi m: the target's start window along the task right")
    ap.add_argument("--target-forward", default=None, help="lo,hi m: and along the task forward")
    ap.add_argument("--clear-target", type=float, default=0.0,
                    help="m: distractors moved at least this far from the target (off the pick and the place)")
    ap.add_argument("--target-window", default=None,
                    help="fwd_lo,fwd_hi,left_lo,left_hi m: the target's start window in the photo robot's base frame "
                         "(scene graph photo_robot.base; apps/episode.py --photo-base), in place of the task window")
    args = ap.parse_args()
    s = Session(args.session)
    scenes = args.scenes or ["twin"] + sorted(f.stem for f in (s / "cousins").glob("cousin_*.json"))
    ranges = {"p_start_on": args.start_on, "p_tray": args.tray, "clear_target": args.clear_target}
    for k in ("target_right", "target_forward", "target_window"):
        if getattr(args, k):
            ranges[k] = tuple(float(v) for v in getattr(args, k).split(","))
    tags = write_configs(s, scenes, args.per_scene, args.seed, ranges=ranges, same_target=args.same_target,
                         letter=args.letter)
    print(f"[dr] {len(tags)} configurations: {' '.join(tags)}")


if __name__ == "__main__":
    main()
