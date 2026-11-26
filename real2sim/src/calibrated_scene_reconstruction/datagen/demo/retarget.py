"""Segments onto a new scene: TCP plans in the world frame, joint trajectories, and the task's success from the demo.

Per segment: the skill part replays the demo's TCP poses in the reference object's new frame, a bridge (motion part)
joins the previous pose to the first skill pose -- a quadratic Bezier lifted until no frame enters a furniture box,
minimum-jerk timing, slerped orientation -- and the gripper switches at the segment's event, held for `actuate` s.
IK runs frame by frame, warm-started (in Isaac Sim: trajgen.TrajectoryPlanner.ik_from / rob.solve_ik).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .io import inv

from calibrated_scene_reconstruction.core.geometry import min_jerk as _min_jerk

PHASE = {"motion": 0, "skill": 1, "actuate": 2}


def _slerp(Ra, Rb, s):
    from scipy.spatial.transform import Slerp
    return Slerp([0.0, 1.0], Rotation.from_matrix([Ra, Rb]))(np.clip(s, 0.0, 1.0)).as_matrix()


def transform(seg, T_ref_new, skill_only=True):
    """The segment's TCP poses with its reference object at T_ref_new (world): T_ref_new @ ee_rel."""
    rel = seg.ee_rel[seg.skill:] if skill_only else seg.ee_rel
    return np.asarray(T_ref_new, float) @ rel


def in_boxes(P, boxes, margin=0.02):
    """Per point: inside any axis-aligned box [[x0, y0, z0], [x1, y1, z1]] grown by margin."""
    hit = np.zeros(len(P), bool)
    for b in boxes:
        b = np.asarray(b, float)
        hit |= np.all((P > b[0] - margin) & (P < b[1] + margin), axis=1)
    return hit


def bridge(Ta, Tb, fps=30, speed=0.25, boxes=(), lifts=(0.0, 0.05, 0.1, 0.2, 0.3), margin=0.02, min_frames=10):
    """Frames from Ta to Tb (Tb included), the first of the lifted Bezier arcs that keeps clear of the boxes (boxes
    holding an endpoint are skipped: the support under a grasp). None when no lift clears. Only the TCP is checked
    (TODO: the hand and an object in hand as points, static objects as apps/scene_obstacles.py height fields)."""
    pa, pb = Ta[:3, 3], Tb[:3, 3]
    if np.linalg.norm(pb - pa) < 1e-3 and Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude() < 1e-2:
        return np.asarray(Tb, float)[None]  # already there (a retreat continues from the release)
    ends = np.stack([pa, pb])
    boxes = [b for b in boxes if not in_boxes(ends, [b], margin).any()]
    n = max(min_frames, int(np.ceil(np.linalg.norm(pb - pa) / speed * fps)))
    s = _min_jerk(n)
    R = _slerp(Ta[:3, :3], Tb[:3, :3], s)
    for lift in lifts:
        c = (pa + pb) / 2
        c[2] = max(pa[2], pb[2]) + 2 * lift  # the arc's apex is `lift` above the higher end
        P = ((1 - s) ** 2)[:, None] * pa + (2 * s * (1 - s))[:, None] * c + (s ** 2)[:, None] * pb
        if not in_boxes(P[:-1], boxes, margin).any():
            T = np.tile(np.eye(4), (n, 1, 1))
            T[:, :3, :3], T[:, :3, 3] = R, P
            return T
    return None


def plan(segments, poses_new, T_start, fps=30, speed=0.25, boxes=(), actuate=0.4):
    """New scene -> {"ee": (N, 4, 4), "gripper": (N,), "phase": (N,), "segment": (N,)} or None (a bridge blocked).
    poses_new {name: 4x4}: the reference objects' poses in the new scene (an object picked up earlier is carried, its
    later references are whatever the demo referred to: containers do not move)."""
    ee, grip, phase, seg_id = [], [], [], []
    T_prev = np.asarray(T_start, float)
    for sg in segments:
        T_ref = np.asarray(poses_new[sg.ref], float) if sg.ref else np.eye(4)
        skill = transform(sg, T_ref)
        b = bridge(T_prev, skill[0], fps, speed, boxes)
        if b is None:
            return None
        for part, name in ((b[:-1], "motion"), (skill, "skill")):
            ee.append(part)
            grip += [float(sg.closed)] * len(part)
            phase += [PHASE[name]] * len(part)
            seg_id += [sg.index] * len(part)
        T_prev = skill[-1]
        if sg.event in ("close", "open"):
            n = max(1, int(actuate * fps))
            ee.append(np.repeat(T_prev[None], n, 0))
            grip += [float(sg.event == "close")] * n
            phase += [PHASE["actuate"]] * n
            seg_id += [sg.index] * n
    return {"ee": np.concatenate(ee), "gripper": np.asarray(grip), "phase": np.asarray(phase),
            "segment": np.asarray(seg_id)}


def to_joints(ee, ik, q0, tol=(0.01, 0.05)):
    """ik(p, R, q_seed) -> (q, (pos_err m, rot_err rad)) frame by frame, warm-started. -> (q (N, D), ok)."""
    q, out, worst = np.asarray(q0, float), [], (0.0, 0.0)
    for T in ee:
        q, err = ik(T[:3, 3], T[:3, :3], q)
        out.append(q)
        worst = (max(worst[0], err[0]), max(worst[1], err[1]))
    return np.asarray(out), worst[0] <= tol[0] and worst[1] <= tol[1]


# --- success ---------------------------------------------------------------------------------------------------------
def success_spec(segments, final, initial, pos_tol=0.03, rot="upright", rot_tol_deg=15.0, lift=0.05):
    """The task's success from the demo's end state: the last object released, relative to the reference of the
    segment that released it (a cup in the basket); an object still in hand must be lifted by `lift` m.
    rot: "none" | "upright" (tilt of the object's z axis vs. the demo's) | "full"."""
    moved = [sg for sg in segments if sg.held]
    if not moved:
        return None  # no grasp in the demo: articulated / pushing tasks (TODO)
    last = moved[-1]
    if last.event == "open":
        T_rel = inv(final[last.ref]) @ final[last.held] if last.ref else final[last.held]
        return {"object": last.held, "relative_to": last.ref, "T_rel": T_rel.tolist(), "pos_tol": pos_tol,
                "rot": rot, "rot_tol_deg": rot_tol_deg}
    return {"object": last.held, "lift": lift, "z0": float(np.asarray(initial[last.held])[2, 3])}


def check_success(spec, poses, z0=None):
    """poses {name: 4x4} at the episode's end (sim) -> bool."""
    T_o = np.asarray(poses[spec["object"]], float)
    if "lift" in spec:
        return T_o[2, 3] - (spec["z0"] if z0 is None else z0) >= spec["lift"]
    T = inv(np.asarray(poses[spec["relative_to"]], float)) @ T_o if spec["relative_to"] else T_o
    T_ref = np.asarray(spec["T_rel"], float)
    if np.linalg.norm(T[:3, 3] - T_ref[:3, 3]) > spec["pos_tol"]:
        return False
    if spec["rot"] == "upright":
        ang = np.arccos(np.clip(T[:3, 2] @ T_ref[:3, 2], -1, 1))
    elif spec["rot"] == "full":
        ang = Rotation.from_matrix(T[:3, :3].T @ T_ref[:3, :3]).magnitude()
    else:
        return True
    return np.degrees(ang) <= spec["rot_tol_deg"]
