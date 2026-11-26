"""A demo -> object-centric segments, one per gripper event (MimicGen subtasks), each split DemoGen style:

    motion   free-space part, replanned in a new scene (retarget.bridge)
    skill    the last frames near the reference object, replayed in its frame: T_ee = T_ref_new @ inv(T_ref) @ T_ee

A segment ends at a gripper event (close: a grasp, open: a release) or at the demo's end (the retreat). Its reference
object is the object nearest the TCP at its end, the object in hand excluded (a place segment refers to the container,
not to what it carries); the retreat keeps the previous segment's reference. Objects are rigid and only the one in hand
moves: at a close the nearest object is attached to the TCP, at an open it stays where it is. That bookkeeping also
gives the demo's final object state, from which retarget.success_spec defines the task's success.
"""
from dataclasses import dataclass

import numpy as np

from .io import inv


@dataclass
class Segment:
    index: int
    start: int
    end: int                 # inclusive: the event frame
    ref: str                 # reference object (None: world, a demo without objects nearby)
    held: str                # object in hand during the segment (None: empty hand)
    closed: bool             # gripper state during the segment
    event: str               # "close" | "open" | "end"
    T_ref: np.ndarray        # reference object's world pose during the segment (4x4)
    ee_rel: np.ndarray       # (n, 4, 4) TCP in the reference frame, frames start..end
    skill: int               # first frame (relative to start) of the object-centric part

    def to_dict(self):
        return {k: getattr(self, k) for k in ("index", "start", "end", "ref", "held", "closed", "event", "skill")}


def gripper_events(g, thr=0.5, min_hold=3):
    """[(frame, closed_after)] where the thresholded command changes and keeps the new state for >= min_hold frames
    (debounces a teleoperator's trigger)."""
    b = np.asarray(g, float) > thr
    events, state, k = [], bool(b[0]), 1
    while k < len(b):
        if b[k] != state and np.all(b[k:k + min_hold] == b[k]):
            state = bool(b[k])
            events.append((k, state))
        k += 1
    return events


def nearest(p, poses, exclude=(), max_dist=None):
    best, bd = None, np.inf
    for name, T in poses.items():
        if name in exclude:
            continue
        d = float(np.linalg.norm(np.asarray(T)[:3, 3] - p))
        if d < bd:
            best, bd = name, d
    return best if max_dist is None or bd <= max_dist else None


def skill_start(p_rel, near):
    """First frame of the trailing run of frames within `near` m of the reference origin (0 when all are, the last
    frame when none is)."""
    far = np.flatnonzero(np.linalg.norm(p_rel, axis=1) > near)
    return min(int(far[-1] + 1), len(p_rel) - 1) if len(far) else 0


def event_objects(demo, thr=0.5, min_hold=3):
    """Objects inferred from the demo alone (no twin, or a dataset episode): a world-aligned frame at the TCP at each
    close ("pick_k") and each open ("place_k")."""
    out, k = {}, {"pick": 0, "place": 0}
    for f, closed in gripper_events(demo.gripper, thr, min_hold):
        kind = "pick" if closed else "place"
        T = np.eye(4)
        T[:3, 3] = demo.ee[f][:3, 3]
        out[f"{kind}_{k[kind]}"] = T
        k[kind] += 1
    return out


def segment(demo, objects, near=0.15, max_dist=0.5, thr=0.5, min_hold=3):
    """demo (io.Demo in the world frame) + objects {name: 4x4 world pose at the demo's start} -> (segments, final
    object poses). `near`: radius of the object-centric part; `max_dist`: farthest a reference object may be."""
    poses = {k: np.asarray(v, float).copy() for k, v in objects.items()}
    events = gripper_events(demo.gripper, thr, min_hold)
    bounds = [(e, ("close" if c else "open")) for e, c in events] + [(len(demo.ee) - 1, "end")]
    segs, start, held, T_grip, prev_ref = [], 0, None, None, None
    closed = bool(np.asarray(demo.gripper)[0] > thr)
    for i, (end, ev) in enumerate(bounds):
        if end < start:
            continue
        p_end = demo.ee[end][:3, 3]
        if ev == "end" and prev_ref is not None:
            ref = prev_ref
        else:
            ref = nearest(p_end, poses, exclude={held} if held else (), max_dist=max_dist)
        T_ref = poses[ref].copy() if ref else np.eye(4)
        ee_rel = inv(T_ref) @ demo.ee[start:end + 1]
        # the retreat replays whole in its reference's frame (it leaves the object, it does not approach one)
        sk = 0 if ev == "end" else skill_start(ee_rel[:, :3, 3], near) if ref else len(ee_rel) - 1
        segs.append(Segment(len(segs), start, end, ref, held, closed, ev, T_ref, ee_rel, sk))
        # the object in hand follows the TCP through this segment
        if held:
            poses[held] = demo.ee[end] @ T_grip
        if ev == "close":
            held = ref
            T_grip = inv(demo.ee[end]) @ poses[ref] if ref else None
            closed = True
        elif ev == "open":
            held, T_grip, closed = None, None, False
        prev_ref, start = ref, end + 1
    return segs, poses
