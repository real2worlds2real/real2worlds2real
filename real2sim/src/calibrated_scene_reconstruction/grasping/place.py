"""Pick-and-place: a place goal per episode, the segments after the lift, and the success check. Pure numpy / scipy.

    ... lift / hold -> transport -> place -> release -> retreat
                       Bezier      lower     fingers    back along the approach
                       arc over    onto the  open       axis, then up
                       the goal    goal

* goal: a visible receptacle (assets.receptacles: coaster, placemat, tray, plate, cutting board, taped square, bowl,
  basket; the kinds that belong in the scene's room, weighted per scene by the scene graph's task.place) placed on the
  support in the task frame around the pick position -- its footprint inside the support minus `edge_margin`, clear
  of the other boxes on the support, of the scene's keep-out areas (a cooktop) and of the object's start -- and the
  object's base target on its place surface. Without receptacles (`receptacles: {}`): an invisible point, as before.
  Optionally reachable: `reach_fn(p)` is the planner's IK check.
* transport / place: the grasp holds the object rigidly, so the TCP flies the object's goal offset (TCP minus object
  base at the grasp frame, world, orientation fixed), the object's bottom carried over the receptacle's rim; the
  lowering ends `place_sink` below contact and physics stops it. Out of a container the hand rises before backing off.
* success: the object's base centre in the receptacle's accept region (else within `pos_tol` of the goal), `tilt_tol`
  of upright, resting on the place surface, the hand open and the TCP at least `clear_m` from the object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..assets import receptacles as rcp

from calibrated_scene_reconstruction.core.geometry import min_jerk as _min_jerk

TASK_TYPES = ("pick", "place")
# trajgen.SEG continues with these labels; pick-only trajectories never use them
PLACE_SEG = {"transport": 5, "place": 6, "release": 7, "retreat": 8}


@dataclass
class PlaceCfg:
    forward: tuple = (-0.10, 0.15)         # m, goal offset from the pick position along the task frame's forward
    right: tuple = (-0.30, 0.30)           # m, along its right
    move: tuple = (0.10, 0.35)             # m, horizontal distance from the pick position
    receptacle_move: tuple = (0.10, 0.60)  # m, the same for the object's target on a receptacle (reach_fn decides)
    receptacle_right: tuple = (-0.40, 0.40)  # m, receptacle centre along the task frame's right
    rim_clear: float = 0.03                # m, the object's bottom above a receptacle's rim while carried
    yaw_deg: float = 0.0                   # random goal yaw (0: the object keeps its yaw; TODO wrist yaw)
    edge_margin: float = 0.03              # m, kept between the object's footprint and the support's edge
    obstacle_margin: float = 0.02          # m, and between the footprint and other boxes on the support
    tries: int = 200
    transport_speed: tuple = (0.14, 0.22)  # m/s cruise
    transport_min_s: float = 1.2
    transport_up: tuple = (0.02, 0.08)     # m, the arc's bulge above the straight line
    hover: tuple = (0.04, 0.08)            # m, object bottom above the support when the lowering starts
    place_s: tuple = (0.8, 1.2)
    place_sink: float = 0.004              # m, the lowering ends this far below contact (deeper pushes the object over)
    release_s: tuple = (0.6, 0.9)
    release_hold_s: float = 0.7            # s, open and still before the retreat: a slow gripper still touching the
                                           # object carries it away (UR10e + 2F-85: 6-20 % closed after 0.3 s)
    retreat_m: tuple = (0.06, 0.12)        # m, back along the approach axis
    retreat_up: tuple = (0.04, 0.08)       # m, then up
    retreat_s: tuple = (0.8, 1.2)
    pos_tol: float = 0.04                  # success: m, object (x, y) from the goal
    tilt_tol: float = 15.0                 # deg from upright
    rest_tol: float = 0.02                 # m, object base above the support top (resting, not held)
    clear_m: float = 0.05                  # m, TCP from the object's centre after the retreat
    open_tol: float = 0.15                 # fraction of the closing range left on the hand joints
    direction: str = "task"                # receptacle centre: "task" (the task-frame window / anywhere on the top)
                                           #   or "base": a quadrant of the robot's base frame (front, left, back,
                                           #   right; one per plan, uniform) at rel_dist from the object
    rel_dist: tuple = (0.15, 0.40)         # m, "base": the object's target from its start
    pre_release_s: tuple = (0.0, 0.0)      # s, set down and still before the fingers open (labelled place)
    done_hold_s: tuple = (0.0, 0.0)        # s, still after the retreat before the episode ends (labelled retreat):
                                           #   what a finished task looks like
    max_reach: int = 0                     # give the goal up after this many failed reach checks (IK; 0: no limit)


STYLE = ("transport_speed", "transport_up", "hover", "place_s", "release_s", "retreat_m", "retreat_up", "retreat_s")


def fixed_style(cfg):
    """The configuration with one motion style: every STYLE range at its midpoint (what varies is the scene, the
    object's and the goal's places, the start pose; not the path's shape, speed or timing)."""
    import copy

    c = copy.copy(cfg)
    for f in STYLE:
        lo, hi = getattr(c, f)
        setattr(c, f, ((lo + hi) / 2, (lo + hi) / 2))
    return c


@dataclass
class PlaceGoal:
    position: list                         # world, object base (x, y on the support, z = support top + base offset)
    yaw_deg: float
    support: str
    tf_offset: list                        # (forward, right) from the pick position, task frame
    info: dict = field(default_factory=dict)
    receptacle: dict = None                # placed receptacle summary: kind, label, origin, yaw, accept, ... (or None)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return PlaceGoal(**d)

    def as_pose7(self):
        """xyz + quat wxyz (yaw about +z), the per-frame `observation.goal` of the dataset."""
        h = np.radians(self.yaw_deg) / 2
        return np.array([*self.position, np.cos(h), 0.0, 0.0, np.sin(h)], np.float32)


def task_type(sg, override=None):
    """The episode's task: --task-type, else scene_graph task.type, else pick."""
    t = override or (sg.get("task") or {}).get("type") or "pick"
    if t not in TASK_TYPES:
        raise ValueError(f"task type {t!r}: one of {TASK_TYPES}")
    return t


def task_string(target, goal=None, support=None):
    obj = rcp.noun(target)
    if goal is None:
        return f"pick up the {obj}"
    if goal.receptacle:
        return rcp.task_text(target, goal.receptacle)
    return f"place the {obj} at the marked spot on the {rcp.noun(support or goal.support)}"


def receptacle_weights(sg):
    """{kind: weight} of the scene's place targets: the scene graph's task.place.receptacles (the agent's review,
    written per scene; default every kind) restricted to the kinds that belong in its room (task.place.room,
    assets.receptacles.KINDS[kind]["rooms"]; no room: no restriction) or that the photo itself shows as the place
    target (task.place.match: also set as assets.receptacles.MATCH, so those are built to the photo's
    measurements); {} = an invisible goal point."""
    place = (sg.get("task") or {}).get("place") or {}
    w = place.get("receptacles")
    w = dict(rcp.DEFAULT_WEIGHTS) if w is None else w
    rcp.set_match(place.get("match"))  # the photo's own place target, measured: built to it wherever drawn
    room = set(rcp.kinds_for(place.get("room"))) | set(place.get("match") or {})  # the photo shows it: it belongs
    return {k: v for k, v in w.items() if k in room}


def keep_out(sg):
    """World AABBs of the support's areas no receptacle may cover (scene graph task.place.keep_out {name: [[x0, y0],
    [x1, y1]]}: a cooktop or a sink in the counter top), as goal-sampling obstacles."""
    areas = (((sg.get("task") or {}).get("place") or {}).get("keep_out") or {}).values()
    return [(np.array([*a[0], -np.inf]), np.array([*a[1], np.inf])) for a in areas]


def support_obstacles(top, env=(), boxes=(), band=0.3):
    """World AABBs of what stands on a support whose top is at `top`: oriented environment boxes (centre, half
    extents, rotation, ...) and world AABBs reaching into the band `band` above it."""
    out = []
    for c_, h_, R_, *_ in env:
        r_ = np.abs(np.asarray(R_, float)) @ np.asarray(h_, float)
        out.append((np.asarray(c_, float) - r_, np.asarray(c_, float) + r_))
    return [b for b in out + [tuple(np.asarray(v, float) for v in b) for b in boxes]
            if b[1][2] > top + 0.005 and b[0][2] < top + band]


# --- goal sampling ---------------------------------------------------------------------------------------------------
def _footprint_clear(xy, radius, box, margin):
    """Circle (xy, radius) outside the axis-aligned xy box ((x0, y0, ...), (x1, y1, ...)) by `margin`."""
    lo, hi = np.asarray(box[0], float)[:2], np.asarray(box[1], float)[:2]
    d = np.maximum(np.maximum(lo - xy, xy - hi), 0.0)
    return float(np.linalg.norm(d)) >= radius + margin


def _rect_clear(corners, box, margin):
    """Convex polygon (corners (4, 2)) and an axis-aligned xy box at least `margin` apart (separating axes)."""
    lo, hi = np.asarray(box[0], float)[:2] - margin, np.asarray(box[1], float)[:2] + margin
    B = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    for P, Q in ((corners, B), (B, corners)):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            n = np.array([-e[1], e[0]])
            a, b = P @ n, Q @ n
            if a.max() < b.min() or b.max() < a.min():
                return True
    return False


def _rect_circle_clear(corners, xy, radius):
    """A convex polygon (counter-clockwise or not) and a circle do not overlap."""
    inside, d = True, np.inf
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        t = np.clip((xy - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0, 1)
        d = min(d, float(np.linalg.norm(a + t * (b - a) - xy)))
        e, q, o = b - a, xy - a, corners[(i + 2) % len(corners)] - a
        inside &= (e[0] * q[1] - e[1] * q[0]) * (e[0] * o[1] - e[1] * o[0]) >= 0
    return not inside and d > radius


def sample_goal(tf, p_obj, extent, support_box, rng, cfg=None, obstacles=(), reach_fn=None, support="support",
                weights=None, stats=None, base=None, surfaces=(), quadrant=None):
    """A collision-free, reachable place goal on the support (`support_box` world AABB ((x0, y0, z0), (x1, y1, z1))),
    or None. `p_obj` is the target's base position, `extent` its size; `obstacles` are world AABBs of the other
    furniture / objects standing on the support; `reach_fn(p_obj_goal) -> bool` checks the arm can get there;
    `weights` the receptacle kinds ({}: an invisible goal point); `stats` (a dict) gets the draws each check
    rejected, also when no goal is found. `base` (x, y, yaw deg): the robot's base, for cfg.direction "base";
    `surfaces` [{"name", "inner": (n, 2) convex world polygon, "box": world AABB, "z": m above the support top}]:
    raised places a receptacle may lie on (a tray), its footprint inside `inner`, never across `box`. `quadrant`
    (direction "base"): this quadrant (QUADRANTS) on every draw instead of a random one."""
    c = cfg or PlaceCfg()
    p_obj = np.asarray(p_obj, float)
    radius = 0.5 * float(np.hypot(extent[0], extent[1]))
    w = rcp.DEFAULT_WEIGHTS if weights is None else weights
    if w:
        return _sample_receptacle_goal(tf, p_obj, extent, radius, support_box, rng, c, obstacles, reach_fn, support, w,
                                       {} if stats is None else stats, base, surfaces, quadrant)
    lo, hi = np.asarray(support_box[0], float)[:2], np.asarray(support_box[1], float)[:2]
    why = {} if stats is None else stats
    why.update({"edge": 0, "move": 0, "obstacle": 0, "reach": 0})
    for _ in range(c.tries):
        f, r = rng.uniform(*c.forward), rng.uniform(*c.right)
        p = p_obj + tf.vec(forward=f, right=r)
        xy = p[:2]
        if np.any(xy - radius - c.edge_margin < lo) or np.any(xy + radius + c.edge_margin > hi):
            why["edge"] += 1
            continue
        d = float(np.linalg.norm(xy - p_obj[:2]))
        if not c.move[0] <= d <= c.move[1]:
            why["move"] += 1
            continue
        if not all(_footprint_clear(xy, radius, b, c.obstacle_margin) for b in obstacles):
            why["obstacle"] += 1
            continue
        p[2] = p_obj[2]  # the same support: the base stays at the support height
        if reach_fn is not None and not reach_fn(p):
            why["reach"] += 1
            continue
        yaw = float(rng.uniform(-c.yaw_deg, c.yaw_deg)) if c.yaw_deg else 0.0
        return PlaceGoal(position=p.tolist(), yaw_deg=yaw, support=support, tf_offset=[f, r],
                         info={"move_m": d, "rejected": why})
    return None


def _inside(poly, pts):
    """Every point inside the convex polygon `poly` (n, 2), either winding."""
    poly, pts = np.asarray(poly, float), np.asarray(pts, float)
    e = np.roll(poly, -1, 0) - poly
    area = np.sum(poly[:, 0] * np.roll(poly, -1, 0)[:, 1] - np.roll(poly, -1, 0)[:, 0] * poly[:, 1])
    cr = e[None, :, 0] * (pts[:, None, 1] - poly[None, :, 1]) - e[None, :, 1] * (pts[:, None, 0] - poly[None, :, 0])
    return bool(np.all(cr * np.sign(area) >= 0))


QUADRANTS = ("front", "left", "back", "right")


def _sample_receptacle_goal(tf, p_obj, extent, radius, support_box, rng, c, obstacles, reach_fn, support, weights,
                            why, base=None, surfaces=(), quadrant=None):
    """sample_goal with a visible receptacle: a kind that fits the object, its centre in the task-frame window or (half
    of the draws) anywhere on the support top -- a crowded table may leave no room in the window -- or (direction
    "base") in one quadrant of the robot's base frame at rel_dist from the object (the quadrant drawn once; the
    second half of the draws any quadrant); its footprint on the support (or inside a raised surface) and clear of
    the obstacles and of the object's start, the object's target within `receptacle_move`; at a random yaw."""
    lo, hi = np.asarray(support_box[0], float)[:2], np.asarray(support_box[1], float)[:2]
    top = float(support_box[1][2])
    why.update({"kind": 0, "edge": 0, "move": 0, "obstacle": 0, "start": 0, "reach": 0})
    h_obj = float(extent[2])
    quad = int(rng.integers(4)) if c.direction == "base" and base is not None else None
    forced = quadrant is not None and quad is not None
    if forced:
        quad = QUADRANTS.index(quadrant) if isinstance(quadrant, str) else int(quadrant)
    for i in range(c.tries):
        rec = rcp.sample(radius, h_obj, rng, weights)
        if rec is None:
            why["kind"] += 1
            return None
        q_ = None
        if quad is not None:
            q_ = quad if forced or i < c.tries // 2 else int(rng.integers(4))
            th = np.radians(float(base[2])) + q_ * np.pi / 2 + rng.uniform(-np.pi / 4, np.pi / 4)
            gxy = p_obj[:2] + rng.uniform(*c.rel_dist) * np.array([np.cos(th), np.sin(th)])
            yaw = float(rng.uniform(-np.pi, np.pi))
            ax, ay = rec.accept_xy[:2]
            o = np.array([gxy[0] - (np.cos(yaw) * ax - np.sin(yaw) * ay),
                          gxy[1] - (np.sin(yaw) * ax + np.cos(yaw) * ay), p_obj[2]])
            f, r = float((o - p_obj) @ tf.forward), float((o - p_obj) @ tf.right)
        elif rng.random() < 0.5:
            f, r = rng.uniform(*c.forward), rng.uniform(*c.receptacle_right)
            o = p_obj + tf.vec(forward=f, right=r)
        else:  # the support top shrunk by the footprint's circumradius
            m = float(np.hypot(*rec.half)) + c.edge_margin
            if np.any(hi - lo <= 2 * m):
                why["edge"] += 1
                continue
            o = np.array([*rng.uniform(lo + m, hi - m), p_obj[2]])
            f, r = float((o - p_obj) @ tf.forward), float((o - p_obj) @ tf.right)
        if q_ is None:
            yaw = float(rng.uniform(-np.pi, np.pi))
        rd = {**rec.to_dict(), "origin": [float(o[0]), float(o[1]), top], "yaw": yaw}
        corners = rcp.footprint(rd)
        if np.any(corners < lo + c.edge_margin) or np.any(corners > hi - c.edge_margin):
            why["edge"] += 1
            continue
        g = rcp.to_world(rd, rd["accept_xy"])
        d = float(np.linalg.norm(g - p_obj[:2]))
        move = c.rel_dist if q_ is not None else c.receptacle_move
        if not move[0] - 1e-6 <= d <= move[1] + 1e-6:
            why["move"] += 1
            continue
        if not all(_rect_clear(corners, b, c.obstacle_margin) for b in obstacles):
            why["obstacle"] += 1
            continue
        on = next((s for s in surfaces if _inside(s["inner"], corners)), None)
        if any(s is not on and not _rect_clear(corners, s["box"], c.obstacle_margin) for s in surfaces):
            why["obstacle"] += 1
            continue
        if not _rect_circle_clear(corners, p_obj[:2], radius + c.obstacle_margin):
            why["start"] += 1
            continue
        z0 = top + (float(on["z"]) if on else 0.0)  # the receptacle's bottom: the support top, or a raised surface
        rd["origin"][2] = z0
        p = np.array([g[0], g[1], z0 + rec.place_z])  # the object's base on the place surface
        if reach_fn is not None and not reach_fn(p):
            why["reach"] += 1
            if c.max_reach and why["reach"] >= c.max_reach:  # out of the arm's reach this way: no goal
                return None
            continue
        rd["clear_z"] = float(z0 + rec.rim_z)
        info = {"move_m": d, "rejected": why}
        if q_ is not None:
            info["quadrant"] = QUADRANTS[q_]
        if on:
            info["on"] = on["name"]
        return PlaceGoal(position=p.tolist(), yaw_deg=0.0, support=support, tf_offset=[f, r], info=info,
                         receptacle=rd)
    return None


# --- segments --------------------------------------------------------------------------------------------------------


def _bezier(p0, p1, p2, p3, s):
    s = s[:, None]
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * p3


def place_segments(p_lift, R_g, obj_offset, goal, hand_closed, hand_open, approach_world, rng, cfg=None, fps=30):
    """Cartesian keyframes after the lift: ([p], [R], [hand {dof: q}], [label]) for transport, place, release, retreat.
    `obj_offset` = TCP - object base at the grasp (world, the grasp's orientation is held), `approach_world` the hand's
    approach direction (the retreat backs off against it)."""
    c, U = cfg or PlaceCfg(), rng.uniform
    P, R, H, L = [], [], [], []

    def seg(ps, hs, label):
        P.extend(ps), R.extend([R_g] * len(ps)), H.extend(hs), L.extend([label] * len(ps))

    def mix(ha, hb, s):
        return [{i: (1 - w) * ha[i] + w * hb[i] for i in ha} for w in s]

    p_lift, off = np.asarray(p_lift, float), np.asarray(obj_offset, float)
    g = np.asarray(goal.position, float)
    p_contact = g + off
    # the object's bottom (g at the goal) clears the receptacle's rim at the hover pose and on the way there
    rim = (goal.receptacle or {}).get("clear_z", g[2]) + c.rim_clear - g[2]
    p_hover = p_contact + np.array([0.0, 0.0, max(U(*c.hover), rim)])
    # transport: a Bezier arc from the lift to the hover pose, minimum-jerk at a random cruise speed, arriving from
    # above (the last control point over the hover pose)
    chord = p_hover - p_lift
    dist = float(np.linalg.norm(chord))
    up = np.array([0.0, 0.0, U(*c.transport_up)])
    n = max(int(max(dist / U(*c.transport_speed), c.transport_min_s) * fps), 2)
    s = _min_jerk(n)
    c1, c2 = p_lift + chord / 3 + up, p_lift + 2 * chord / 3 + up
    c2[2] = max(c2[2], p_hover[2] + up[2])
    seg(list(_bezier(p_lift, c1, c2, p_hover, s)), [dict(hand_closed)] * n, "transport")
    # place: straight down until `place_sink` below contact (physics stops the object on the support)
    n = int(U(*c.place_s) * fps)
    s = _min_jerk(n)
    p_down = p_contact - np.array([0.0, 0.0, c.place_sink])
    seg(list(p_hover + s[:, None] * (p_down - p_hover)), [dict(hand_closed)] * n, "place")
    n = int(U(*c.pre_release_s) * fps)  # set down, still, before letting go
    seg([p_down] * n, [dict(hand_closed)] * n, "place")
    # release: fingers open, wrist still
    n = int(U(*c.release_s) * fps)
    seg([p_down] * n, mix(hand_closed, hand_open, _min_jerk(n)), "release")
    n = int(c.release_hold_s * fps)
    seg([p_down] * n, [dict(hand_open)] * n, "release")
    # retreat: back against the approach axis, then up; out of a container (rim above the place surface) up first
    a = np.asarray(approach_world, float)
    a = a / max(np.linalg.norm(a), 1e-9)
    n = max(int(U(*c.retreat_s) * fps) // 2, 1)
    if rim > c.rim_clear + 0.015:
        p_up = p_down + np.array([0.0, 0.0, rim + U(*c.retreat_up)])
        legs = ((p_down, p_up), (p_up, p_up - U(*c.retreat_m) * a))
    else:
        p_back = p_down - U(*c.retreat_m) * a
        legs = ((p_down, p_back), (p_back, p_back + np.array([0.0, 0.0, U(*c.retreat_up)])))
    for a_, b_ in legs:
        s = _min_jerk(n)
        seg(list(a_ + s[:, None] * (b_ - a_)), [dict(hand_open)] * n, "retreat")
    n = int(U(*c.done_hold_s) * fps)  # done: the hand still, away from the object
    seg([legs[-1][1]] * n, [dict(hand_open)] * n, "retreat")
    return P, R, H, L


def phase_spans(seg, labels):
    """{label: (first frame, end frame)} of a per-frame label array; `labels` maps name -> id (trajgen.SEG)."""
    seg = np.asarray(seg)
    out = {}
    for name, k in labels.items():
        idx = np.flatnonzero(seg == k)
        if len(idx):
            out[name] = (int(idx[0]), int(idx[-1]) + 1)
    return out


# --- success ---------------------------------------------------------------------------------------------------------
def tilt_deg(quat_wxyz):
    _w, x, y, _z = np.asarray(quat_wxyz, float) / np.linalg.norm(quat_wxyz)
    zz = 1 - 2 * (x * x + y * y)  # world z of the body's +z
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def place_outcome(obj_pos, obj_quat, centre, goal, support_top, base_dz, tcp_pos, hand_q, hand_open, hand_closed,
                  arrived=None):
    """Measurements of a finished place episode: goal error (x, y), tilt, base height above rest, TCP distance from the
    object's centre and how far the hand is from open (fraction of the open -> closed range over all joints together: a
    joint that barely moves between the two postures must not blow the fraction up). `base_dz` is the object base's
    height above the support top when resting upright. `arrived`: the object's position when the fingers start to
    open -- it must have been carried there (an object dropped on the way may still land in a big tray)."""
    p = np.asarray(obj_pos, float)
    err = float(np.linalg.norm(p[:2] - np.asarray(goal.position, float)[:2]))
    rest = float(p[2] - (support_top + base_dz))
    span = sum(abs(hand_closed[i] - hand_open[i]) for i in hand_open)
    closed = sum(abs(hand_q[i] - hand_open[i]) for i in hand_open) / span if span > 1e-6 else 0.0
    return {"goal_err_m": err, "tilt_deg": tilt_deg(obj_quat), "rest_dz_m": rest,
            "tcp_dist_m": float(np.linalg.norm(np.asarray(tcp_pos, float) - np.asarray(centre, float))),
            "hand_closed_frac": float(closed),
            "inside": rcp.inside(goal.receptacle, p[:2]) if goal.receptacle else None,
            "arrive_err_m": None if arrived is None else
            float(np.linalg.norm(np.asarray(arrived, float)[:2] - np.asarray(goal.position, float)[:2]))}


def place_succeeded(o, cfg=None):
    c = cfg or PlaceCfg()
    at_goal = o["inside"] if o.get("inside") is not None else o["goal_err_m"] <= c.pos_tol
    carried = o.get("arrive_err_m") is None or o["arrive_err_m"] <= c.pos_tol
    return (at_goal and carried and o["tilt_deg"] <= c.tilt_tol and abs(o["rest_dz_m"]) <= c.rest_tol
            and o["tcp_dist_m"] >= c.clear_m and o["hand_closed_frac"] <= c.open_tol)
