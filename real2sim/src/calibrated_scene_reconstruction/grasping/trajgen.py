"""Arm grasp trajectories around a synthesized (or analytic) grasp, randomized in where and how the hand arrives.

    start  -> transit -> standoff -> approach -> close -> squeeze / settle -> lift / hold
    random    Bezier     pregrasp    straight    wrist +   fingers past      world +z (+ a little
    pose      bulge      backed off  + slerp     fingers   the grasp         back toward the robot)

* start: the TCP somewhere between the robot and the target above the support (task frame ranges), its orientation
  a random blend of the home and standoff orientations plus a random rotation; IK gives the initial joint state.
* standoff: the pregrasp pulled back along the hand's approach axis by a random distance, held above the support by
  a floor (a retreat that would dip toward the support bends to a level slide, as in the original tabletop
  generator), and rotated off the pregrasp orientation (roll about the approach axis, tilt about the two others): the
  approach then turns the wrist back into the grasp while it closes in.
* transit: a cubic Bezier from the start to the standoff whose inner control points bulge up and sideways at random,
  orientation slerped, minimum-jerk timing at a random cruise speed.
* approach / close / squeeze / settle / lift follow the grasp record's stages (pregrasp, grasp, squeeze postures).

Every frame (30 fps) is an IK target for the arm (warm-started damped least squares on the live articulation), the
hand joints are interpolated between the record's postures. A kinematic check keeps the hand above the support and,
before the arrival, outside the target's bounding cylinder; physics (apps/episode.py) decides the rest.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from calibrated_scene_reconstruction.core.geometry import min_jerk as _min_jerk

AXES = {"x": 0, "y": 1, "z": 2}
SEG = {"transit": 0, "approach": 1, "close": 2, "squeeze": 3, "lift": 4,
       "transport": 5, "place": 6, "release": 7, "retreat": 8}  # 5.. pick-and-place only (grasping.place)


@dataclass
class TrajCfg:
    fps: int = 30
    start_back: tuple = (0.20, 0.40)       # m behind the target (toward the robot), task frame
    start_lateral: tuple = (-0.20, 0.20)   # m along the task frame's right
    start_up: tuple = (0.14, 0.32)         # m above the support top
    start_blend: tuple = (0.2, 0.6)        # orientation: home -> standoff slerp fraction
    start_rot_deg: float = 30.0            # plus a random rotation of at most this
    standoff: tuple = (0.06, 0.16)         # m, pregrasp backed off along the approach axis (at least)
    standoff_max: float = 0.30             # m, scanned outward until the open hand clears the target
    standoff_roll_deg: float = 25.0        # about the approach axis
    standoff_tilt_deg: float = 15.0        # about the two other hand axes
    standoff_up_deg: float = 50.0          # most the back-off tips toward the vertical when straight back is blocked
    pre_sink_m: float = 0.012              # most the pregrasp hand's collision hull may sit below the support top
    pre_gap_m: float = 0.008               # the hand flown into the pregrasp keeps this clear of the target (its
                                           #   points are a 400-point sample: 4 mm still slid a cup 19 mm)
    pre_back_m: tuple = (0.0, 0.01, 0.02, 0.03)  # and backs off this far along the approach axis if it must
    floor_m: float = 0.012                 # hand clearance above the support while not grasping
    bulge_up: tuple = (0.0, 0.08)
    bulge_side: float = 0.06
    transit_speed: tuple = (0.16, 0.26)    # m/s cruise
    transit_min_s: float = 1.2
    approach_s: tuple = (0.8, 1.3)
    close_s: tuple = (0.6, 1.0)
    squeeze_s: float = 0.5
    settle_s: float = 0.3
    lift_h: tuple = (0.10, 0.16)
    lift_back: tuple = (0.0, 0.04)
    lift_s: tuple = (1.5, 2.0)
    hold_s: float = 0.8
    obj_margin: float = 0.01
    final_m: tuple = (0.04, 0.07)          # the approach's last leg: straight along the approach axis, the hand
    final_max: float = 0.12                #   already in its pregrasp orientation and posture (backed off further
    final_s: tuple = (0.4, 0.6)            #   until that posture clears the target), exempt from the target check
    ik_tol_free: float = 0.03              # m, transit / transport / retreat: the arm may lag the curve in free space
                                           #   (a Vega torso held to its working range trails a high transport arc by
                                           #   15-18 mm); the recorded commands are what the IK reached, physics
                                           #   validates them
    ik_tol_contact: float = 0.006          # m, approach .. lift


STYLE = ("bulge_up", "transit_speed", "approach_s", "close_s", "lift_h", "lift_back", "lift_s", "final_m",
         "final_s", "standoff")


def fixed_style(cfg):
    """The configuration with one motion style: the path's shape, timing and pregrasp offset at their midpoints
    (STYLE), no sideways bulge, no wrist roll / tilt at the standoff; the start pose stays random (where a rollout may
    begin)."""
    import copy

    c = copy.copy(cfg)
    for f in STYLE:
        lo, hi = getattr(c, f)
        setattr(c, f, ((lo + hi) / 2, (lo + hi) / 2))
    c.bulge_side = c.standoff_roll_deg = c.standoff_tilt_deg = 0.0
    return c


@dataclass
class GraspTarget:
    """One grasp in world coordinates: TCP poses of the pregrasp and grasp, hand postures as {dof index: value}."""
    source: str
    params: dict
    pre: tuple
    grasp: tuple
    hand_pre: dict
    hand_grasp: dict
    hand_squeeze: dict
    approach_tcp: np.ndarray               # the hand's approach direction in the TCP frame
    hand_open: dict = None                 # posture flown during the transit, closing to hand_pre on the approach
    info: dict = field(default_factory=dict)


@dataclass
class Traj:
    q: np.ndarray                          # (T, D) joint targets
    seg: np.ndarray                        # (T,) SEG labels
    info: dict


def axis_vec(spec_axis):
    v = np.zeros(3)
    v[AXES[spec_axis[-1]]] = -1.0 if spec_axis.startswith("-") else 1.0
    return v


def _slerp(Ra, Rb, s):
    return Slerp([0.0, 1.0], Rotation.from_matrix([Ra, Rb]))(np.clip(s, 0.0, 1.0)).as_matrix()


def _random_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * np.radians(rng.uniform(0.0, max_deg))).as_matrix()


# --- grasp targets ---------------------------------------------------------------------------------------------------
def flexion_open(pre: dict, asset: dict) -> dict:
    """The original tabletop generator's `flexion_open` approach posture by joint name: the flexion channels
    (retarget.relax_joint_names) at the open posture, every other channel and planning.transit_hold_joint_names at the
    pregrasp angle, then planning.transit_start_joint_overrides."""
    r, pl = asset.get("retarget", {}), asset.get("planning") or {}
    opened = dict(pre)
    hold = set(pl.get("transit_hold_joint_names") or ())
    for n in r.get("relax_joint_names") or ():
        if n in opened and n not in hold:
            opened[n] = float(r.get("open_posture", {}).get(n, 0.0))
    for n, v in (pl.get("transit_start_joint_overrides") or {}).items():
        if n in opened:
            opened[n] = float(v)
    return opened


def bodex_target(rec, T_obj, rob, side, asset):
    """A BODex record (hand base-link poses in the target's prim frame) as a world GraspTarget for `rob`; `asset` is
    the hand's BODex YAML (base link, approach axis, open posture)."""
    approach_axis_base, base_link = asset["retarget"]["approach_axis"], asset["base_link"]
    arm = rob.spec["arms"][side]
    pb, Rb = rob.link_pose(base_link)
    pt, Rt = rob.tcp_pose(side)
    T_base_tcp = np.eye(4)  # fixed: the hand base link and the arm's TCP are rigidly attached
    T_base_tcp[:3, :3], T_base_tcp[:3, 3] = Rb.T @ Rt, Rb.T @ (pt - pb)

    def world(stage):
        st = rec["stages"][stage]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(st["orientation"], scalar_first=True).as_matrix()
        T[:3, 3] = st["position"]
        W = T_obj @ T @ T_base_tcp
        return W[:3, 3].copy(), W[:3, :3].copy()

    def by_index(joints):
        return {rob.idx[n]: float(v) for n, v in joints.items() if n in rob.idx}

    def hand(stage):
        return by_index(rec["stages"][stage]["joints"])

    stages = rec["stages"]
    return GraspTarget(
        source=f"bodex:{rec.get('_path', '')}", params={"rank": rec.get("rank"), "seed_index": rec.get("seed_index")},
        pre=world("pregrasp"), grasp=world("grasp"), hand_pre=hand("pregrasp"), hand_grasp=hand("grasp"),
        hand_squeeze=hand("squeeze" if "squeeze" in stages else "grasp"),
        approach_tcp=T_base_tcp[:3, :3].T @ np.asarray(approach_axis_base, float),
        hand_open=by_index(flexion_open(stages["pregrasp"]["joints"], asset)),
        info={"rank_score": rec.get("rank_score"), "arm": arm.get("tcp_link")})


def analytic_target(c, rob, side, pre_back=0.03):
    """A robots.grasps candidate as a GraspTarget: its grasp pose and closure; the pregrasp is the grasp `pre_back`
    back along the approach axis (the candidate's own pre waypoint was a standoff, which the planner now draws)."""
    q0 = rob.home()
    ee = [rob.idx[n] for n in rob.ee_joints(side)] + [rob.idx[n] for n in rob.mimic]
    opened, closed = rob.with_ee(q0, 0.0, side), rob.with_ee(q0, c["close"], side)
    approach = axis_vec(rob.spec["arms"][side]["tcp_axes"]["approach"])
    g = np.asarray(c["grasp"], float)
    return GraspTarget(
        source=f"analytic:{c['family']}", params={"family": c["family"], "params": c["params"]},
        pre=(g - pre_back * (c["R"] @ approach), c["R"]), grasp=(g, c["R"]),
        hand_pre={i: float(opened[i]) for i in ee}, hand_grasp={i: float(closed[i]) for i in ee},
        hand_squeeze={i: float(closed[i]) for i in ee}, approach_tcp=approach)


# --- geometry --------------------------------------------------------------------------------------------------------
def floored_standoff(p_pre, back, d, hand_min_z, z_floor):
    """The pregrasp backed off by `d` along `back`; a retreat that would take the hand's lowest point below `z_floor`
    stops descending there and continues level (as the original tabletop generator's floored_retreat_positions does)."""
    p = p_pre + d * back
    if back[2] >= 0 or p[2] + hand_min_z >= z_floor:
        return p
    knee = max((p_pre[2] + hand_min_z - z_floor) / -back[2], 0.0)
    flat = np.array([back[0], back[1], 0.0])
    n = np.linalg.norm(flat)
    if n < 1e-9:
        return np.array([p_pre[0], p_pre[1], z_floor - hand_min_z])
    q = p_pre + knee * back
    q[2] = max(min(q[2], p_pre[2]), z_floor - hand_min_z)
    return q + (d - knee) * flat / n


def hull_planes(V):
    """Plane equations (normals, offsets) of the convex hull of world points: signed distance = max(N x - d)."""
    import trimesh

    V = np.asarray(V, float)
    if len(V) > 600:  # hull of a subsample: tens of planes instead of hundreds, the check runs per hand point
        V = V[np.linspace(0, len(V) - 1, 600).astype(int)]
    hull = trimesh.convex.convex_hull(V)
    N = np.asarray(hull.face_normals)
    return N, np.einsum("ij,ij->i", N, np.asarray(hull.triangles)[:, 0])


def hand_clash(pts_tcp, p, R, support_top, floor, obj, margin=0.006, env=()):
    """What the hand hits, None when clear: hand points above the support's floor ("support"), (when `obj`, the
    target's hull planes or a list of its convex parts' planes, is given) at least `margin` outside the target
    ("target"; per hull, max over the planes: exact inside, a lower bound of the distance outside), and outside the
    environment boxes `env`
    [(centre, half, R[, name])] by `margin` (the box's name)."""
    W = pts_tcp @ R.T + p
    if W[:, 2].min() < support_top + floor:
        return "support"
    if obj is not None:
        for N, d in (obj if isinstance(obj, list) else [obj]):  # one convex hull, or the target's convex parts
            if ((W @ N.T - d).max(axis=1) < margin).any():
                return "target"
    for c, h, Rb, *name in env:
        if (np.abs((W - c) @ Rb) <= h + margin).all(axis=1).any():
            return name[0].split("/")[-1] if name else "environment"
    return None


def target_gap(pts_tcp, p, R, obj):
    """Signed gap (m) between the hand points and the target (its hull, or the nearest of its convex parts): negative
    inside, a lower bound of the distance outside."""
    W = pts_tcp @ R.T + p
    return min(float((W @ N.T - d).max(axis=1).min()) for N, d in (obj if isinstance(obj, list) else [obj]))


def hand_ok(*a, **k):
    return hand_clash(*a, **k) is None


# --- planner ---------------------------------------------------------------------------------------------------------
class TrajectoryPlanner:
    def __init__(self, rob, side, tf, q_home, pts_fn, obj_planes, cfg=None, body_links=None, support_xy=None,
                 env=()):
        self.rob, self.side, self.tf, self.q_home = rob, side, tf, np.asarray(q_home, float)
        # environment boxes [(centre, half, R)] other than the support (whose top is the floor check):
        # walls, furniture, static and other objects near the target (apps/scene_obstacles.py)
        self.env = list(env)
        # pts_fn(posture {dof: q}) -> hand hull points in the TCP frame with the hand in that posture
        self.pts_fn, self.obj = pts_fn, obj_planes  # obj_planes: the target's convex hull (hull_planes)
        self.cfg = cfg or TrajCfg()
        # whole-body check: hull points of the links other than the hand ({name: local points}) must stay out of the
        # support's top slab inside its footprint ((x0, y0), (x1, y1)) -- e.g. a humanoid's idle arm swung into the
        # counter by a torso bend, which stalls the reach
        self.body_links, self.support_xy = body_links or {}, support_xy

    def body_clear(self, slab=0.06):
        if not self.body_links or self.support_xy is None:
            return None
        tfs = self.rob.link_transforms()
        (x0, y0), (x1, y1) = self.support_xy
        top = self.tf.support_top
        for name, L in self.body_links.items():
            t = tfs[self.rob.body_names.index(name)]
            W = L @ Rotation.from_quat(t[3:7]).as_matrix().T + t[:3]
            hit = (W[:, 0] > x0) & (W[:, 0] < x1) & (W[:, 1] > y0) & (W[:, 1] < y1) & \
                (W[:, 2] < top - 0.005) & (W[:, 2] > top - slab)
            if hit.any():
                return name
        return None

    def _ik(self, p, R, q0, iters=120, low_weight_scale=1.0):
        q, err = self.rob.solve_ik(self.side, p, R, q0, iters=iters, low_weight_scale=low_weight_scale)
        return q, err

    def ik_from(self, p, R, q0, steps=6):
        """IK by continuation: the target slides from the TCP pose at `q0` to (p, R) in `steps`, each solve warm-started
        by the last (a single damped least-squares solve diverges on large wrist rotations)."""
        p0, R0 = self._fk(q0)
        q, err = np.asarray(q0, float), (np.inf, np.inf)
        for k, w in enumerate(np.linspace(0.0, 1.0, steps + 1)[1:]):
            q, err = self._ik(p0 + w * (np.asarray(p) - p0), _slerp(R0, R, w), q, iters=250 if k == steps - 1 else 60)
        return q, err

    def feasible(self, target, rng, restarts=3):
        """IK of the pregrasp and the grasp (quick filter before any sampling): from home, then from random arm
        configurations around it -- a synthesized grasp may need the wrist far from its home orientation."""
        ci = [self.rob.idx[c] for c in self.rob.spec["arms"][self.side]["ik_joints"]]
        best = None
        for k in range(restarts + 1):
            q0 = self.q_home.copy()
            if k:
                q0[ci] = np.clip(q0[ci] + rng.normal(scale=0.6, size=len(ci)), self.rob.lower[ci], self.rob.upper[ci])
            q_pre, e1 = self.ik_from(*target.pre, q0)
            if e1[0] > 0.15:  # hopeless from this start: skip the grasp solve
                e2 = e1
                q_g = q_pre
            else:
                q_g, e2 = self._ik(*target.grasp, q_pre, iters=250)
            ok = e1[0] < 0.01 and e1[1] < 0.08 and e2[0] < 0.01 and e2[1] < 0.08
            if best is None or e1[0] + e2[0] < best[2][0][0] + best[2][1][0]:
                best = (ok, (q_pre, q_g), (e1, e2))
            if ok:
                break
        return best

    def plan(self, target, rng, q_pre_seed, place=None, q_start=None, backoff=None):
        """`place` = (PlaceGoal, object base position at rest, PlaceCfg or None): pick-and-place, the lift continues
        with grasping.place.place_segments; None: pick only. `q_start`: the joint state the motion starts from
        instead of a random start pose (a state a policy reached: the expert's way on from there, DAgger-style).
        `backoff` (up, back) m with q_start: first the fingers open and the hand rises and draws back toward the robot
        (a policy's failed grasp may start with the fingers on the cup: those frames are not checked against it)."""
        c, tf = self.cfg, self.tf
        U = rng.uniform
        p_pre, R_pre = target.pre
        p_g, R_g = target.grasp
        top = tf.support_top
        info = {"source": target.source, "params": target.params}
        hand_open = target.hand_open or target.hand_pre
        pts_open, pts_pre = self.pts_fn(hand_open), self.pts_fn(target.hand_pre)
        # the support floor of the approach: the synthesis cleared the pregrasp against the support with its own
        # (sphere) hand model, which the simulator's collision hull may undercut by a few mm -- the approach may come
        # as low as the pregrasp itself sits, but no lower
        pre_low = float((pts_pre @ R_pre.T)[:, 2].min() + p_pre[2] - top) if len(pts_pre) else 0.0
        if pre_low < -c.pre_sink_m:
            return None, {**info, "fail": f"pregrasp hand {-1000 * pre_low:.0f} mm into the support"}
        floor_a = min(-0.003, pre_low - 0.001)
        # the posture flown in: as open as the support lets the hand be at the pregrasp (a hand lying low beside the
        # target cannot spread its fingers through the table), the pregrasp posture itself at worst
        for s_open in (0.0, 0.35, 0.7):
            h = {i: (1 - s_open) * hand_open[i] + s_open * target.hand_pre[i] for i in hand_open}
            P_ = pts_open if s_open == 0.0 else self.pts_fn(h)
            if not len(P_) or (P_ @ R_pre.T)[:, 2].min() + p_pre[2] - top >= floor_a:
                hand_open, pts_open = h, P_
                break
        else:
            s_open, hand_open, pts_open = 1.0, dict(target.hand_pre), pts_pre
        info["open_fraction"] = 1.0 - s_open

        # the posture flown into the pregrasp: the synthesized pregrasp, or -- where it already reaches into the
        # target (a pool grasp's pregrasp sits 6 mm inside a cup at the median, and the final leg slid the cup 4-6 cm
        # away before the fingers closed) -- the least-open blend toward the open hand that keeps pre_gap_m clear;
        # the close then shapes the fingers from it into the grasp
        # (a palm already in contact cannot open its way out: the pregrasp then backs off along the approach axis,
        # and the close advances the palm into the grasp while the fingers shape)
        hand_in, pts_in, p_in = dict(target.hand_pre), pts_pre, p_pre
        back_in = -(R_pre @ target.approach_tcp)
        back_in /= np.linalg.norm(back_in)
        gap = target_gap(pts_pre, p_pre, R_pre, self.obj) if len(pts_pre) else 1.0
        if gap < c.pre_gap_m:
            postures = [(0.0, dict(target.hand_pre), pts_pre)]
            for w_open in (0.25, 0.5, 0.75, 1.0):
                h = {i: (1 - w_open) * target.hand_pre[i] + w_open * hand_open[i] for i in target.hand_pre}
                postures.append((w_open, h, self.pts_fn(h)))
            for b, (w_open, h, P_) in ((b, x) for b in c.pre_back_m for x in postures):
                q_ = p_pre + b * back_in
                if (b or w_open) and target_gap(P_, q_, R_pre, self.obj) >= c.pre_gap_m and \
                        (P_ @ R_pre.T)[:, 2].min() + q_[2] - top >= floor_a:
                    hand_in, pts_in, p_in = h, P_, q_
                    info.update(pregrasp_opened=w_open, pregrasp_backed_m=b)
                    break
            else:
                return None, {**info, "fail": f"the pregrasp hand reaches {-1000 * gap:.0f} mm into the target and "
                                              f"neither opening nor {1000 * c.pre_back_m[-1]:.0f} mm of back-off "
                                              "clears it"}
        info["pregrasp_gap_mm"] = round(1000 * gap, 1)

        def pts_at(w):  # the approach closes the hand from open to the posture flown into the pregrasp
            return pts_open if w < 0.6 else pts_in

        # the final leg: the pregrasp backed off straight along the approach axis (the synthesis' own approach), far
        # enough that the hand in its pregrasp posture clears the target -- a grasp reaching into a cup's cavity
        # enters it along this line with the fingers already shaped, rather than sweeping them in while turning
        back0 = -(R_pre @ target.approach_tcp)
        back0 /= np.linalg.norm(back0)
        lead = back0.copy()  # the final leg's direction: the approach axis, levelled when it would dive into the support
        if lead[2] < 0.0:
            lead[2] = 0.0
            if np.linalg.norm(lead) < 0.5:
                return None, {**info, "fail": "the approach axis comes up from under the target"}
            lead /= np.linalg.norm(lead)
        d_near = U(*c.final_m)
        while hit := hand_clash(pts_in, p_in + d_near * lead, R_pre, top, floor_a, self.obj):
            d_near += 0.01
            if d_near > c.final_max:
                return None, {**info, "fail": f"the pregrasp posture does not clear the {hit} within "
                                              f"{c.final_max:.2f} m along the approach axis (target gap "
                                              f"{1000 * target_gap(pts_in, p_in + c.final_max * lead, R_pre, self.obj):.0f}"
                                              f" mm there)"}
        p_near = p_in + d_near * lead
        final_leg = [p_in + w * d_near * lead for w in np.linspace(0.0, 1.0, 6)]
        if hit := next((h for q_ in final_leg if (h := hand_clash(pts_in, q_, R_pre, top, floor_a, None,
                                                                    env=self.env))), None):
            return None, {**info, "fail": f"final approach leg blocked ({hit})"}

        # standoff: the start of the final leg backed off further (floored at the support; tipped toward the vertical
        # when a neighbour blocks the axis) and the wrist turned off the pregrasp orientation; the open hand's
        # straight, slerped way to the final leg must clear the target -- otherwise back off further, or draw again
        a = target.approach_tcp / np.linalg.norm(target.approach_tcp)
        o1 = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        o1 /= np.linalg.norm(o1)
        o2 = np.cross(a, o1)
        found, why = None, Counter()
        for attempt in range(6):
            k = 1.0 - attempt / 6  # later draws turn the wrist less
            # the back-off: along the approach axis; when that is blocked (a neighbour behind the hand), later draws
            # tip it toward the vertical, so the hand comes down onto the grasp rather than across the clutter
            back, up = lead, 0.0
            if attempt >= 2 and lead[2] < 0.95:
                up = np.radians(U(15, c.standoff_up_deg))
                side = np.cross(lead, [0.0, 0.0, 1.0])
                back = Rotation.from_rotvec(up * side / np.linalg.norm(side)).apply(lead)
            roll, t1, t2 = (U(-1, 1) * k * c.standoff_roll_deg, U(-1, 1) * k * c.standoff_tilt_deg,
                            U(-1, 1) * k * c.standoff_tilt_deg)
            dR = Rotation.from_rotvec(np.radians(roll) * a) * Rotation.from_rotvec(np.radians(t1) * o1) * \
                Rotation.from_rotvec(np.radians(t2) * o2)
            R_s = R_pre @ dR.as_matrix()
            hand_min_z = float((pts_open @ R_s.T)[:, 2].min()) if len(pts_open) else 0.0
            d0 = U(*c.standoff)
            for d in np.arange(max(d0, d_near + 0.02), c.standoff_max + 1e-9, 0.01):  # at least d0 back, further
                p_s = floored_standoff(p_near, back, d - d_near, hand_min_z, top + c.floor_m)  # if the hand needs it
                ws = np.linspace(0.0, 1.0, 12)
                Rs = _slerp(R_s, R_pre, ws)
                hit = next((h for h in (hand_clash(pts_at(w), p_s + w * (p_near - p_s), Rw, top, floor_a, self.obj,
                                                   env=self.env)
                                        for w, Rw in zip(ws, Rs)) if h), None)
                if hit is None:
                    found = d
                    break
                why[hit] += 1
            if found is not None:
                break
        if found is None:
            hits = ", ".join(f"{k} {v}" for k, v in why.most_common(3))
            return None, {**info, "fail": f"no standoff up to {c.standoff_max:.2f} m clears the target ({hits})"}
        d = found
        info.update(standoff_m=d, standoff_rot_deg=[roll, t1, t2], standoff_up_deg=float(np.degrees(up)),
                    final_leg_m=float(d_near))

        # start: a random TCP pose between the robot and the target, above the support
        if q_start is None:
            p0, R_home = self._home_tcp()
            p_start = (p_g + tf.vec(forward=-U(*c.start_back), right=U(*c.start_lateral)))
            p_start[2] = top + U(*c.start_up)
            R_start = _random_rotation(rng, c.start_rot_deg) @ _slerp(R_home, R_s, U(*c.start_blend))
            q_start, e0 = self.ik_from(p_start, R_start, self.q_home if q_pre_seed is None else q_pre_seed)
            if e0[0] > 0.02 or e0[1] > 0.15:
                return None, {**info, "fail": "start IK"}
        else:
            q_start = np.asarray(q_start, float).copy()
            info["start_given"] = True
        p_start, R_start = self._fk(q_start)
        info.update(start_tcp=p_start.tolist())

        # Cartesian keyframes, 30 fps
        P, R, H, S = [], [], [], []

        def seg(ps, Rs, hs, label):
            P.extend(ps), R.extend(Rs), H.extend(hs), S.extend([SEG[label]] * len(ps))

        hand_start = {i: float(q_start[i]) for i in hand_open}

        def hand_mix(ha, hb, s):
            return [{i: (1 - w) * ha[i] + w * hb[i] for i in ha} for w in s]

        n_backoff = 0
        if backoff is not None and info.get("start_given"):  # open and back off before anything else
            p_b = p_start + np.array([0.0, 0.0, backoff[0]]) + tf.vec(forward=-backoff[1])
            n = int(0.7 * c.fps)
            s = _min_jerk(n)
            seg(list(p_start + s[:, None] * (p_b - p_start)), [R_start] * n, hand_mix(hand_start, hand_open, s),
                "transit")
            n_backoff, p_start, hand_start = len(P), p_b, dict(hand_open)
            info.update(backoff_m=[float(backoff[0]), float(backoff[1])])

        # transit: Bezier with a random bulge
        chord = p_s - p_start
        L = float(np.linalg.norm(chord))
        side_dir = np.cross([0, 0, 1.0], chord / max(L, 1e-9))
        side_dir /= max(np.linalg.norm(side_dir), 1e-9)
        b1 = side_dir * U(-1, 1) * c.bulge_side + np.array([0, 0, U(*c.bulge_up)])
        b2 = side_dir * U(-1, 1) * c.bulge_side + np.array([0, 0, U(*c.bulge_up)])
        P1, P2 = p_start + chord / 3 + b1, p_start + 2 * chord / 3 + b2
        n = max(int(max(c.transit_min_s, 1.875 * L / U(*c.transit_speed)) * c.fps), 2)  # min-jerk peak 1.875 x mean
        if info.get("start_given"):  # a policy's start may hold the wrist far from the standoff's: turn it slowly
            turn = float(Rotation.from_matrix(R_start.T @ R_s).magnitude())
            n = max(n, int(1.875 * turn / 1.0 * c.fps))
        s = _min_jerk(n)
        bez = ((1 - s) ** 3)[:, None] * p_start + (3 * (1 - s) ** 2 * s)[:, None] * P1 + \
            (3 * (1 - s) * s ** 2)[:, None] * P2 + (s ** 3)[:, None] * p_s
        seg(list(bez), list(_slerp(R_start, R_s, s)), hand_mix(hand_start, hand_open, s), "transit")
        # approach: straight to the final leg, wrist turning back into the grasp orientation and the fingers shaping
        # the pregrasp posture; then the final leg, straight along the approach axis into the pregrasp
        n = int(U(*c.approach_s) * c.fps)
        s = _min_jerk(n)
        seg(list(p_s + s[:, None] * (p_near - p_s)), list(_slerp(R_s, R_pre, s)),
            hand_mix(hand_open, hand_in, s), "approach")
        n_final = len(P)
        n = int(U(*c.final_s) * c.fps)
        s = _min_jerk(n)
        seg(list(p_near + s[:, None] * (p_in - p_near)), [R_pre] * n, [dict(hand_in)] * n, "approach")
        n_approach_end = len(P)
        n_transit = S.count(SEG["transit"])
        # close: pregrasp -> grasp, wrist and fingers together
        n = int(U(*c.close_s) * c.fps)
        s = _min_jerk(n)
        seg(list(p_in + s[:, None] * (p_g - p_in)), list(_slerp(R_pre, R_g, s)),
            hand_mix(hand_in, target.hand_grasp, s), "close")
        # squeeze + settle
        n = int(c.squeeze_s * c.fps)
        s = _min_jerk(n)
        seg([p_g] * n, [R_g] * n, hand_mix(target.hand_grasp, target.hand_squeeze, s), "squeeze")
        n = int(c.settle_s * c.fps)
        seg([p_g] * n, [R_g] * n, [dict(target.hand_squeeze)] * n, "squeeze")
        # lift (+ a little back toward the robot), hold
        h, bk = U(*c.lift_h), U(*c.lift_back)
        p_l = p_g + np.array([0, 0, h]) + tf.vec(forward=-bk)
        n = int(U(*c.lift_s) * c.fps)
        s = _min_jerk(n)
        seg(list(p_g + s[:, None] * (p_l - p_g)), [R_g] * n, [dict(target.hand_squeeze)] * n, "lift")
        n = int(c.hold_s * c.fps)
        seg([p_l] * n, [R_g] * n, [dict(target.hand_squeeze)] * n, "lift")
        info.update(lift_m=h, lift_back_m=bk, frames=len(P))
        if place is not None:
            from calibrated_scene_reconstruction.grasping.place import place_segments

            goal, p_obj, pcfg = place
            ps, Rs, hs, ls = place_segments(p_l, R_g, p_g - np.asarray(p_obj, float), goal, target.hand_squeeze,
                                            hand_open, R_g @ target.approach_tcp, rng, pcfg, c.fps)
            P.extend(ps), R.extend(Rs), H.extend(hs), S.extend(SEG[k] for k in ls)
            info.update(goal=goal.to_dict(), frames=len(P))

        # kinematic check: support floor and environment everywhere before the grasp, the target before the final leg
        for k in range(n_backoff, n_approach_end, 2):
            floor = c.floor_m * 0.5 if S[k] == SEG["transit"] else floor_a  # a pregrasp may sit in the table band
            near = k >= n_final
            w = 0.0 if S[k] == SEG["transit"] else min((k - n_transit) / max(n_final - n_transit, 1), 1.0)
            hit = hand_clash(pts_at(w), P[k], R[k], top, floor, None if near else self.obj, env=self.env)
            if hit:
                return None, {**info, "fail": f"hand clearance at frame {k} ({'transit' if S[k] == 0 else 'approach'}"
                                             f", {hit})"}

        # IK along the path, warm-started. A link other than the hand swung into the support (a humanoid's idle
        # hand carried along by a torso bend) gets more tries with the low-weight joints (the torso) moving a
        # quarter as much, then not at all, so the arm does more of the reach
        Q, worst, fail = self._ik_path(P, R, H, S, q_start, 1.0)
        if fail and " in the support at frame " in fail and any(v < 1 for v in self.rob.spec["arms"][self.side].get(
                "ik_weights", {}).values()):
            for scale in (0.25, 0.0):
                Q2, worst2, fail2 = self._ik_path(P, R, H, S, q_start, scale)
                if fail2 is None:
                    Q, worst, fail = Q2, worst2, None
                    info["low_weight_scale"] = scale
                    break
        if fail:
            return None, {**info, "fail": fail}
        jump = float(np.abs(np.diff(Q, axis=0)).max()) if len(Q) > 1 else 0.0
        if jump > 0.12:
            return None, {**info, "fail": f"joint jump {jump:.3f} rad/frame"}
        info.update(ik_worst_mm=worst * 1000, max_joint_step=jump, grasp_frame=int(S.index(SEG["squeeze"]) - 1),
                    grasp_tcp=[float(v) for v in p_g])
        return Traj(Q, np.asarray(S, np.int8), info), info

    # --- helpers
    def _ik_path(self, P, R, H, S, q_start, scale):
        """(joint rows, worst position error, failure or None) of the warm-started IK along the keyframes."""
        c = self.cfg
        Q = np.zeros((len(P), len(self.q_home)))
        q = q_start.copy()
        worst = 0.0
        for k in range(len(P)):
            q, err = self._ik(P[k], R[k], q, iters=60, low_weight_scale=scale)
            if k % 3 == 0 and (hit := self.body_clear()):
                return Q, worst, f"{hit} in the support at frame {k}"
            # free-space legs (transit, the transport of a held object, the retreat) may lag; contact legs may not
            tol = c.ik_tol_free if S[k] in (SEG["transit"], SEG["transport"], SEG["retreat"]) else c.ik_tol_contact
            worst = max(worst, err[0])
            if err[0] > tol or err[1] > 0.12:
                seg = next((n for n, v in SEG.items() if v == S[k]), S[k])
                return Q, worst, f"IK at frame {k} ({seg}, err {err[0]*1000:.1f} mm, {np.degrees(err[1]):.1f} deg)"
            row = q.copy()
            for i, v in H[k].items():
                row[i] = v
            Q[k] = self._mimic(row)
        return Q, worst, None

    def _home_tcp(self):
        self.rob.teleport(self.q_home)
        self.rob.sync()
        return self.rob.tcp_pose(self.side)

    def _fk(self, q):
        self.rob.teleport(q)
        self.rob.sync()
        return self.rob.tcp_pose(self.side)

    def _mimic(self, q):
        for n, (master, mult, off) in self.rob.mimic.items():
            i = self.rob.idx[n]
            q[i] = float(np.clip(q[self.rob.idx[master]] * mult + off, self.rob.lower[i], self.rob.upper[i]))
        return q
