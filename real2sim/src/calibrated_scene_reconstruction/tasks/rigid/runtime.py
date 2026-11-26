"""Rigid episode environment. Construct only after the simulation backend starts."""
import copy
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor
from calibrated_scene_reconstruction.assets import receptacles as rcp
from calibrated_scene_reconstruction.assets.physprops import mass_props
from calibrated_scene_reconstruction.robots.grasps import candidates
from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame, approach_edges
from calibrated_scene_reconstruction.core.paths import ROOT

from typing import Any
from .services import EpisodeServices
from .cli import W, H

DEFAULT_ROBOT = "vega_1p_sharpa"

class EpisodeRuntime:
    BATCH: Any
    BATCH_ANNOTS: Any
    BATCH_CAMS: Any
    BATCH_FURNITURE: Any
    BATCH_LOGS: Any
    COPY_BASE: Any
    COPY_INFEASIBLE: Any
    COPY_PLANS: Any
    CUP_START: Any
    ENCODER: Any
    ENCODES: Any
    FIXED_DOFS: Any
    FLAT: Any
    FLAT_PIECES: Any
    GOAL_STATS: Any
    GRASP_PHYS: Any
    H: Any
    HOME_PTS: Any
    IK_OFF: Any
    IK_PATH: Any
    INFEASIBLE: Any
    NOISE_W: Any
    OBJ_BOXES: Any
    PCFG: Any
    PHOTO_BASE: Any
    POLICY_QUAD: Any
    QUAD_DONE: Any
    RECORDED: Any
    RESUME: Any
    SINKS: Any
    SLIDE_HINTS: Any
    START_BASES: Any
    START_STATES: Any
    SURFACES: Any
    SWEEPS: Any
    TARGETS: Any
    TASK: Any
    T_obj: Any
    UNUSABLE: Any
    VISITED: Any
    VISITED_BASE: Any
    W: Any
    WEIGHTS: Any
    _jitter: Any
    _keep: Any
    _pts_cache: Any
    _rs: Any
    acc: Any
    annots: Any
    arm: Any
    asset: Any
    base_check: Any
    base_order: Any
    best: Any
    best_failed: Any
    body: Any
    bx: Any
    by: Any
    byaw: Any
    c_local: Any
    cam_dr: Any
    chosen: Any
    etag0: Any
    ext: Any
    frame: Any
    grasp_source: Any
    hand: Any
    hl: Any
    ik_rob: Any
    info: Any
    log: Any
    n_batches: Any
    n_phys: Any
    obj: Any
    out: Any
    p_obj: Any
    parts: Any
    pending: Any
    planner: Any
    prepared: Any
    pts: Any
    q_home: Any
    q_obj: Any
    records: Any
    rng: Any
    rob: Any
    robot_cams: Any
    s: Any
    sbox: Any
    shift: Any
    side: Any
    spec: Any
    stage: Any
    substitute: Any
    support_top: Any
    sweep: Any
    target: Any
    tf: Any
    third: Any
    to_b: Any
    trials: Any
    ts: Any
    world: Any
    z_com: Any

    def __init__(self, args, app, services: EpisodeServices):
        self.args = args
        self.app = app
        self.services = services
        self.W, self.H = W, H

    def phys_steps(self, n):
        """Physics steps that last as long as `n` steps at 120 Hz: the stepping below was written for a 120 Hz scene
        (4 steps per 30 Hz command), and a session's simulation.json (sim.simulation.apply) can set another rate."""
        return max(1, int(round(n / 120.0 / float(self.world.get_physics_dt()))))

    def placements_file(self):
        p = self.s / "placements" / f"{self.args.robot}.json"
        return p if p.exists() or self.args.robot != DEFAULT_ROBOT else self.s / "placements" / "placements.json"

    def hand_pts_for(self, posture):
        """The hand proper's hull points (TCP frame) with its joints in `posture` ({dof index: q}): trajectory checks."""
        k = tuple(sorted((i, round(v, 3)) for i, v in posture.items()))
        if k not in self._pts_cache:
            q = self.q_home.copy()
            for i, v in posture.items():
                q[i] = v
            self.rob.teleport(q)
            self.rob.sync()
            P = self.services.hand_points(self.rob, self.side, self.body, links=self.hl)
            self._pts_cache[k] = P[np.linspace(0, len(P) - 1, min(len(P), 400)).astype(int)] if len(P) else P
        return self._pts_cache[k]

    def _object_boxes(self):
        """World AABBs of the other movable objects (distractors, cousins' extra objects) and the static objects: a
        receptacle must not be placed on them -- except a flat set piece set into the top (a cooktop, < 1 cm proud: part
        of the top) and a configuration's tray for the goal (calibrated_scene_reconstruction.dr role "surface": SURFACES)."""
        cache = self.services.UsdGeom.BBoxCache(self.services.Usd.TimeCode.Default(), [self.services.UsdGeom.Tokens.default_])
        out = []
        for root in ("/World/Objects", "/World/Static"):
            for prim in (self.stage.GetPrimAtPath(root).GetChildren() if self.stage.GetPrimAtPath(root) else []):
                if prim.GetName() == self.target or prim.GetCustomDataByKey("calibrated_scene_reconstruction:role") == "surface":
                    continue
                r_ = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lo_, hi_ = np.array(r_.GetMin()), np.array(r_.GetMax())
                if prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit") and hi_[2] < self.support_top + self.FLAT:
                    self.FLAT_PIECES.append((lo_, hi_))  # a surface (SURFACES), not an obstacle
                    continue
                out.append((lo_, hi_))
        return out

    def _surfaces(self):
        """sample_goal's raised surfaces: a configuration's trays (calibrated_scene_reconstruction.dr role "surface"): the floor's accept region
        (world polygon), the footprint's box, the floor's height over the top."""
        out = []
        for k, rd in enumerate(self.prepared.get("receptacles") or []):
            if rd.get("role") != "surface":
                continue
            a = rd["accept"]
            if a[0] == "circle":
                loc = [(a[1] * np.cos(w), a[1] * np.sin(w)) for w in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
            else:
                loc = [(sx * a[1], sy * a[2]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            inner = np.array([rcp.to_world(rd, np.asarray(rd["accept_xy"][:2]) + v) for v in loc])
            fp = rcp.footprint(rd)
            out.append({"name": f"dr_surface_{k}", "inner": inner, "z": float(rd["origin"][2] - self.support_top + rd["place_z"]),
                        "box": (np.r_[fp.min(0), -np.inf], np.r_[fp.max(0), np.inf])})
        for k, (lo_, hi_) in enumerate(self.FLAT_PIECES):  # a cooktop: the marker lies on its panel, wholly, at its height
            inner = np.array([[lo_[0], lo_[1]], [hi_[0], lo_[1]], [hi_[0], hi_[1]], [lo_[0], hi_[1]]])
            out.append({"name": f"flat_{k}", "inner": inner, "z": float(hi_[2] - self.support_top),
                        "box": (np.r_[lo_[:2], -np.inf], np.r_[hi_[:2], np.inf])})
        return out

    def new_goal(self, tgt, q_grasp, pl=None, rng_=None, stats=None, base=None, quadrant=None):
        """A place goal for the next plan (--task-type place): a receptacle on the support, clear of the boxes and objects
        standing on it and of its keep-out areas, where the arm holding this grasp reaches the object's place pose and the
        pose 8 cm above it (IK from the grasp configuration, grasp orientation kept). TODO: goals on another free
        surface. `pl`, `rng_`, `stats`: a batch planning slot's planner, generator and rejection counts (default the
        main scene's); `base` the robot's base the plan stands on (default the main one), `quadrant` the goal's."""
        pl = pl or self.planner
        rng_ = self.rng if rng_ is None else rng_
        stats = self.GOAL_STATS if stats is None else stats
        p_g, R_g = tgt.grasp
        off = np.asarray(p_g, float) - self.p_obj

        def reach(p):
            q1, e1 = pl.ik_from(p + off + [0.0, 0.0, 0.08], R_g, q_grasp, steps=3)
            if e1[0] > 0.01 or e1[1] > 0.08:
                return False
            _, e2 = pl._ik(p + off, R_g, q1, iters=120)
            return e2[0] < pl.cfg.ik_tol_contact and e2[1] < 0.08

        # the environment boxes without the flat pieces' columns (a cooktop is a surface: SURFACES)
        env_ = [b for b in pl.env if float(b[0][2] + (np.abs(b[2]) @ b[1])[2]) >= self.support_top + self.FLAT]
        obst = self.services.support_obstacles(self.tf.support_top, env_, self.OBJ_BOXES) + self.services.keep_out(self.s.scene_graph)
        stats.clear()
        return self.services.sample_goal(self.tf, self.p_obj, self.ext, self.sbox, rng_, self.PCFG, obstacles=obst, support=self.s.scene_graph["task"]["support"],
                           weights=self.WEIGHTS, reach_fn=reach, stats=stats, base=base or (self.bx, self.by, self.byaw), surfaces=self.SURFACES,
                           quadrant=quadrant)

    def show_goal(self, traj, root="/World/Goal"):
        """Build the plan's receptacle under `root` (a batch copy's root carries its offset); none for a pick plan."""
        rd = (traj.info.get("goal") or {}).get("receptacle") if traj is not None else None
        if rd is None:
            self.services.remove_receptacle(self.stage, f"{root}/receptacle")
            return
        self.services.UsdGeom.Xform.Define(self.stage, root)
        self.services.add_receptacle(self.stage, f"{root}/receptacle", rcp.rebuild(rd), rd["origin"], rd["yaw"])

    def record_step(self, q_cmd):
        return self.services.record_step(self, q_cmd)

    def outcome(self):
        """(lift, tilt_deg, distance from the TCP to the object's centre): a grasp succeeds when the object is lifted,
        upright and in the hand."""
        p, q = self.obj.get_world_pose()
        Rq = Rotation.from_quat(np.asarray(q, float), scalar_first=True)
        tilt = float(np.degrees(np.arccos(np.clip(Rq.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0))))
        centre = np.asarray(p) + Rq.apply(self.c_local)
        return float(np.asarray(p)[2] - self.p_obj[2]), tilt, float(np.linalg.norm(self.rob.tcp_pose(self.side)[0] - centre))

    def lift_frames(self, traj):
        """(first, last) command index of the lift and hold (before any transport): the in-hand rotation window."""
        k = np.flatnonzero(traj.seg == self.services.SEG["lift"])
        return (int(k[0]), int(k[-1])) if len(k) else (None, None)

    def in_hand(self, R_tcp, q_obj):
        """The object's orientation in the TCP frame (object quaternion wxyz)."""
        return Rotation.from_matrix(R_tcp).inv() * Rotation.from_quat(np.asarray(q_obj, float), scalar_first=True)

    def rot_deg(self, a, b):
        return float(np.degrees((a.inv() * b).magnitude())) if a is not None and b is not None else None

    def place_meas(self, traj, p, q, tcp, q_rob, z_rest, off=np.zeros(3), arrived=None):
        """grasping.place.place_outcome of a finished place rollout (object pose p, q wxyz) in the scene copy at `off`: the
        hand's open / closed postures are the trajectory's last command and its first transport command."""
        goal = self.services.PlaceGoal.from_dict(traj.info["goal"])
        goal.position = (np.asarray(goal.position) + off).tolist()
        if goal.receptacle:
            goal.receptacle = {**goal.receptacle, "origin": (np.asarray(goal.receptacle["origin"]) + off).tolist()}
            # resting on the receptacle's surface: its bottom (the top, or a raised surface such as a tray) + place_z --
            # not the object's start height (it may start on a board)
            z_rest = goal.receptacle["origin"][2] + goal.receptacle["place_z"]
        q_open, q_closed = traj.q[-1], traj.q[int(np.flatnonzero(traj.seg == self.services.PLACE_SEG["transport"])[0])]
        dofs = np.flatnonzero(np.abs(q_closed - q_open) > 1e-6)
        centre = np.asarray(p, float) + Rotation.from_quat(np.asarray(q, float), scalar_first=True).apply(self.c_local)
        return self.services.place_outcome(p, q, centre, goal, z_rest, 0.0, tcp, {i: float(q_rob[i]) for i in dofs},
                             {i: float(q_open[i]) for i in dofs}, {i: float(q_closed[i]) for i in dofs}, arrived)

    def release_frame(self, traj):
        """The command index where the fingers start to open (place plans), else None."""
        k = np.flatnonzero(traj.seg == self.services.PLACE_SEG["release"])
        return int(k[0]) if len(k) else None

    def exec_noise(self, traj, rng_):
        """(T, D) offsets added to a plan's executed commands: per IK joint (the torso's at a third) a low-passed
        Gaussian (time constant ~0.25 s) scaled to an amplitude drawn up to --action-noise, faded by segment (NOISE_W,
        smoothed over half a second). The recorded action is the plan's command: where the arm is pushed off it, the
        label points back."""
        T = len(traj.q)
        cols = [self.rob.idx[c] for c in self.arm["ik_joints"]]
        a = rng_.uniform(0.3, 1.0) * self.args.action_noise
        alpha = 1.0 / (0.25 * self.args.fps)
        w = rng_.normal(size=(T + 60, len(cols)))
        z = np.zeros_like(w)
        for i in range(1, len(w)):
            z[i] = (1 - alpha) * z[i - 1] + alpha * w[i]
        z = z[60:] / max(z[60:].std(), 1e-9) * a
        w_ = {**self.NOISE_W, **{n: self.args.noise_near for n in ("approach", "lift", "place")}}
        seg_w = np.array([w_.get(next((n for n, v in {**self.services.SEG, **self.services.PLACE_SEG}.items() if v == s_), ""), 0.0)
                          for s_ in traj.seg], float)
        seg_w = np.convolve(seg_w, np.ones(15) / 15, mode="same")
        out = np.zeros_like(traj.q)
        for j, c in enumerate(cols):
            out[:, c] = z[:, j] * seg_w * (0.33 if self.arm["ik_joints"][j].startswith("torso") else 1.0)
        traj.info["exec_noise_rad"] = round(float(a), 4)
        return out

    def noise_of(self, traj):
        return getattr(traj, "noise", None)

    def succeeded(self, o):
        """pick: (lift, tilt, dist) lifted, upright, in hand; place: (.., place_meas) at the goal, released, retreated."""
        if len(o) > 3:
            return self.services.place_succeeded(o[3], self.PCFG)
        lift, tilt, dist = o
        return lift > 0.06 and tilt < 45.0 and dist < 0.2

    def reset(self, q0, traj=None):
        """World reset (with the plan's receptacle built first), then the trajectory's initial joint state held for half
        a second."""
        if traj is not None:
            self.show_goal(traj)
        self.world.reset()
        self.rob = self.services.Robot(self.world, self.spec)
        if self.ik_rob is not None:
            self.ik_rob = self.services.Robot(self.world, self.spec, path=self.IK_PATH)
        self.planner.rob = self.rob
        self.rob.teleport(q0)
        if self.CUP_START is not None:  # --resume-states: the cup where the policy left it
            self.obj.set_world_pose(self.CUP_START[:3], self.CUP_START[3:7])
            self.obj.set_linear_velocity(np.zeros(3))
            self.obj.set_angular_velocity(np.zeros(3))
        for _ in range(self.phys_steps(60)):
            self.rob.command(q0)
            self.world.step(render=False)

    def play(self, traj, record):
        if record:  # a search can run for minutes without a Kit frame: the cameras warmed up first (render only)
            dt = self.services.warm(list(self.annots.values()), self.world.render)
            if dt > 1:
                print(f"cameras ready after {dt:.0f} s of rendering", flush=True)
        reach, arrived, k_rel = None, None, self.release_frame(traj)
        k0, k1 = self.lift_frames(traj)
        rel0 = rel1 = None
        nz = self.noise_of(traj)
        for k, q_cmd in enumerate(traj.q):
            self.rob.command(q_cmd if nz is None else q_cmd + nz[k])
            for _ in range(self.phys_steps(4)):
                self.world.step(render=False)
            if k == traj.info.get("grasp_frame"):  # how far the hand got from the planned grasp (blocked arm, pushed cup)
                reach = float(np.linalg.norm(self.rob.tcp_pose(self.side)[0] - np.asarray(traj.info["grasp_tcp"])))
                traj.info["reach_err_m"] = reach
            if k == k_rel:
                arrived = np.asarray(self.obj.get_world_pose()[0], float)
            if k in (k0, k1):  # the object's turn in the hand from the start of the lift to the end of the hold
                rel = self.in_hand(self.rob.tcp_pose(self.side)[1], self.obj.get_world_pose()[1])
                rel0, rel1 = (rel, rel1) if k == k0 else (rel0, rel)
                traj.info["inhand_rot_deg"] = self.rot_deg(rel0, rel1)
            if self.sweep[0] is not None and k % 3 == 0:
                self.sweep[0].sample()
            if record:
                self.services.capture_frame(self, q_cmd)
        o = self.outcome()
        if "goal" in traj.info:
            p, q = self.obj.get_world_pose()
            o = (*o, self.place_meas(traj, p, q, self.rob.tcp_pose(self.side)[0], self.rob.q(), self.p_obj[2], arrived=arrived))
        return o

    def trial(self, traj, scene):
        self.reset(traj.q[0], traj)
        self.sweep[0] = self.body.restart(self.rob)
        o = self.play(traj, record=False)
        return {"scene": scene, "lift_m": o[0], "tilt_deg": o[1], "tcp_dist_m": o[2], "success": self.succeeded(o),
                "reach_err_m": traj.info.get("reach_err_m"), "inhand_rot_deg": traj.info.get("inhand_rot_deg"),
                **({"place": o[3]} if len(o) > 3 else {})}

    def batch_trial(self, items, video=None, sweeps=False, rec=None):
        """Roll out [(tgt, traj, t)] at once, item i in scene copy i; returns ([(lift, tilt, dist[, place])] per item,
        [RobotSweep per item or None]). With `video` (a directory), one tiled frame per command: each copy seen by the
        third-person camera, shifted with it. sweeps=True: each copy's robot is swept (sim.sweep; in that copy's world
        coordinates). `rec` ([episode directory or None] per item): recorded as record() does, every robot camera and the
        third-person one of the item's copy rendered per command; poses are stored in the main scene's coordinates."""
        for i in range(len(self.BATCH)):
            self.show_goal(items[i][1] if i < len(items) else None, f"/World/Batch/env{i}/Goal")
            b_ = items[i][1].info.get("base") if i < len(items) else None
            if b_ is not None:  # a plan made on another copy's base: this copy's robot onto it
                self.place_copy(i, b_)
        self.world.reset()
        self.rob = self.services.Robot(self.world, self.spec)
        self.planner.rob = self.rob
        robs = [self.services.Robot(self.world, self.spec, path=path) for _, path, _ in self.BATCH[:len(items)]]
        objs = [self.services.SingleRigidPrim(path) for _, _, path in self.BATCH[:len(items)]]
        sws = [self.body.fork(r_) if sweeps else None for r_ in robs]
        for r_, (_, traj, _) in zip(robs, items):
            r_.teleport(traj.q[0])
        for _ in range(self.phys_steps(60)):
            for r_, (_, traj, _) in zip(robs, items):
                r_.command(traj.q[0])
            self.world.step(render=False)
        z0 = [float(o_.get_world_pose()[0][2]) for o_ in objs]
        n = max(len(traj.q) for _, traj, _ in items)
        anns = []
        if video is not None:
            eye, at = (np.asarray(v, float) for v in self.s.config("views.json")["third_person"])
            for i, (off, _, _) in enumerate(self.BATCH[:len(items)]):
                cam = self.services.add_camera(self.stage, f"/World/Batch/cam{i}", self.services.pinhole(640, 400), 640, 400, eye + off, at + off)
                anns.append(self.services.rgb_annotator(cam.GetPath(), 640, 400))
            video.mkdir(parents=True, exist_ok=True)
            for _ in range(10):
                self.world.render()
        recs = [None] * len(items)
        vids = [None] * len(items)  # --video-only: per item {stream: VideoWriter}
        if rec is not None:
            for i, d in enumerate(rec[:len(items)]):
                if d is None:
                    continue
                if i not in self.BATCH_ANNOTS:  # once per process and copy (render products pile up otherwise)
                    self.BATCH_ANNOTS[i] = {k_: self.services.rgb_annotator(cam.GetPath(), w, h) for k_, (cam, w, h, _) in
                                       self.BATCH_CAMS[i].items()}
                if self.args.video_only:
                    (d / "raw" / "frames").mkdir(parents=True, exist_ok=True)
                    vids[i] = {k_: self.services.VideoWriter(d / "raw" / "frames" / f"{k_}.mp4", w, h, self.args.fps)
                               for k_, (_, w, h, _) in self.BATCH_CAMS[i].items()}
                    first_ = next((k_ for k_ in self.BATCH_CAMS[i] if k_ != "third"), None)
                    pw = self.BATCH_CAMS[i]["third"][1] + (self.BATCH_CAMS[i][first_][1] if first_ else 0)
                    vids[i]["_preview"] = self.services.VideoWriter(d / "episode.mp4", pw, self.BATCH_CAMS[i]["third"][2], self.args.fps, crf=20)
                else:
                    for k_ in self.BATCH_ANNOTS[i]:
                        (d / "raw" / "frames" / k_).mkdir(parents=True, exist_ok=True)
                    (d / "frames").mkdir(parents=True, exist_ok=True)
                recs[i] = {k_: [] for k_ in ("q", "qd", "action", "tcp_pose", "object_pose")}
                recs[i].update({f"cam_{k_}_T_world": [] for k_ in self.robot_cams})
            for _ in range(10):
                self.world.render()
        cols = int(np.ceil(np.sqrt(len(items))))
        rows = int(np.ceil(len(items) / cols))
        k_rel = [self.release_frame(traj) for _, traj, _ in items]
        arrived = [None] * len(items)
        # the object's (z, tilt) at the start of every place phase and at the end: where a cup tips or sticks to the hand
        marks = [{int(np.flatnonzero(traj.seg == v)[0]): n for n, v in self.services.PLACE_SEG.items() if np.any(traj.seg == v)}
                 for _, traj, _ in items]
        trace = [{} for _ in items]
        windows = [self.lift_frames(traj) for _, traj, _ in items]
        rels = [[None, None] for _ in items]
        writer = ThreadPoolExecutor(8) if rec is not None else None
        for k in range(n):
            q_cmds = [traj.q[min(k, len(traj.q) - 1)] for _, traj, _ in items]
            for r_, q_c, (_, traj_, _) in zip(robs, q_cmds, items):
                nz = self.noise_of(traj_)
                r_.command(q_c if nz is None else q_c + nz[min(k, len(nz) - 1)])
            for _ in range(self.phys_steps(4)):
                self.world.step(render=False)
            for i, (r_, o_, (k0, k1)) in enumerate(zip(robs, objs, windows)):
                if k_rel[i] == k:
                    arrived[i] = np.asarray(o_.get_world_pose()[0], float)
                if k in marks[i] or k == n - 1:
                    p_, q_ = o_.get_world_pose()
                    trace[i][marks[i].get(k, "end")] = [round(float(p_[2] - z0[i]), 3), round(self.services.tilt_deg(q_), 1)]
                if k in (k0, k1):
                    rels[i][k != k0] = self.in_hand(r_.tcp_pose(self.side)[1], o_.get_world_pose()[1])
                if sws[i] is not None and k % 3 == 0:
                    sws[i].sample()
            if any(r is not None for r in recs):
                self.world.render()
                for i, lg in enumerate(recs):
                    if lg is None or k >= len(items[i][1].q):  # a shorter rollout is done
                        continue
                    off = self.BATCH[i][0]
                    imgs = {k_: self.services.grab(a_, self.world.render) for k_, a_ in self.BATCH_ANNOTS[i].items()}
                    first = next((k_ for k_ in imgs if k_ != "third"), None)
                    d = rec[i]
                    preview = np.concatenate([imgs["third"]] + ([imgs[first]] if first else []), 1)
                    if vids[i] is not None:
                        vids[i]["_preview"].write(preview)
                        for k_, im in imgs.items():
                            vids[i][k_].write(im)
                    else:
                        writer.submit(Image.fromarray(preview).save, d / "frames" / f"{k:05d}.jpg", quality=92)
                        for k_, im in imgs.items():
                            writer.submit(Image.fromarray(im).save, d / "raw" / "frames" / k_ / f"{k:05d}.png",
                                          compress_level=1)
                    p, R = robs[i].tcp_pose(self.side)
                    po, qo = objs[i].get_world_pose()
                    lg["q"].append(robs[i].q())
                    lg["qd"].append(np.asarray(robs[i].robot.get_joint_velocities(), float))
                    lg["action"].append(np.asarray(q_cmds[i], float))
                    lg["tcp_pose"].append(np.concatenate([p - off, Rotation.from_matrix(R).as_quat(scalar_first=True)]))
                    lg["object_pose"].append(np.concatenate([np.asarray(po, float) - off, np.asarray(qo, float)]))
                    for name, (cam, *_) in self.BATCH_CAMS[i].items():
                        if name == "third":
                            continue
                        T = np.asarray(self.services.omni.usd.get_world_transform_matrix(cam.GetPrim())).T @ np.diag([1, -1, -1, 1])
                        T[:3, 3] -= off
                        lg[f"cam_{name}_T_world"].append(T)
            if anns:
                self.world.render()
                tile = np.zeros((rows * 400, cols * 640, 3), np.uint8)
                for i, a_ in enumerate(anns):
                    tile[(i // cols) * 400:(i // cols + 1) * 400, (i % cols) * 640:(i % cols + 1) * 640] = self.services.grab(a_, self.world.render)
                Image.fromarray(tile).save(video / f"{k:05d}.jpg", quality=88)
        if writer is not None:
            writer.shutdown(wait=True)
        for v_ in vids:
            for w_ in (v_ or {}).values():
                w_.close()
        for (_, traj, _), (a, b) in zip(items, rels):
            traj.info["inhand_rot_deg"] = self.rot_deg(a, b)
        out_ = []
        for r_, o_, z, (_, traj, _), (off, _, _), arr in zip(robs, objs, z0, items, self.BATCH, arrived):
            p, q = o_.get_world_pose()
            Rq = Rotation.from_quat(np.asarray(q, float), scalar_first=True)
            tilt = float(np.degrees(np.arccos(np.clip(Rq.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0))))
            o = (float(np.asarray(p)[2] - z), tilt,
                 float(np.linalg.norm(r_.tcp_pose(self.side)[0] - (np.asarray(p) + Rq.apply(self.c_local)))))
            if "goal" in traj.info:
                o = (*o, self.place_meas(traj, p, q, r_.tcp_pose(self.side)[0], r_.q(), z, np.asarray(off, float), arr))
                o[3]["trace"] = trace[len(out_)]
            out_.append(o)
        if rec is not None:
            self.BATCH_LOGS[:] = recs
        return out_, sws

    def move_base(self, x, y, yaw_deg):
        """The robot (and the batch copies' robots) onto another base, then a world reset."""
        self.services.place_robot(self.stage, self.spec, (x, y), np.radians(yaw_deg), support_top=self.support_top)
        for off, rpath, _ in self.BATCH:
            self.services.place_robot(self.stage, self.spec, (x + off[0], y + off[1]), np.radians(yaw_deg), path=rpath, support_top=self.support_top)
        if self.IK_OFF is not None:
            self.services.place_robot(self.stage, self.spec, (x + self.IK_OFF[0], y + self.IK_OFF[1]), np.radians(yaw_deg), path=self.IK_PATH,
                        support_top=self.support_top)
        self.reset(self.q_home)

    def run_batch(self):
        """Validate the pending plans at once in the batch envs (--batch); successes are recorded in the full scene."""
        t0 = time.time()
        video = self.s / "episodes" / "_batch" / f"{self.etag0}_frames" if self.args.batch_video and self.n_batches[0] == 0 else None
        res, sws = self.batch_trial(self.pending, video, sweeps=True)
        dt = time.time() - t0
        for i, ((tgt, traj, t), o) in enumerate(zip(self.pending, res)):
            self.n_phys += 1
            t.update({"scene": "batch", "lift_m": o[0], "tilt_deg": o[1], "tcp_dist_m": o[2], "success": self.succeeded(o),
                      "trial_s": round(dt / len(self.pending), 1), "batch": self.n_batches[0],
                      "inhand_rot_deg": traj.info.get("inhand_rot_deg")})
            pl = ""
            if len(o) > 3:
                t["place"] = o[3]
                rd = traj.info["goal"].get("receptacle") or {}
                ae = o[3]["arrive_err_m"]
                pl = (f" | {rd.get('kind', 'point')}: arrived {'-' if ae is None else f'{ae:.3f}'} goal err "
                      f"{o[3]['goal_err_m']:.3f} inside {o[3]['inside']} rest {o[3]['rest_dz_m']:+.3f} "
                      f"open {o[3]['hand_closed_frac']:.2f} | z/tilt {o[3].get('trace')}")
            self.trials.append(t)
            print(f"[batch {self.n_batches[0]}] grasp {t['grasp']} var {t['variation']}: lift {o[0]:.3f} tilt {o[1]:.0f} "
                  f"in hand {o[2]:.2f} turned {t['inhand_rot_deg'] or 0:.0f} deg{pl} -> {t['success']}", flush=True)
            gi_ = t.get("grasp")
            self.GRASP_PHYS.setdefault(gi_, [0, 0])[0 if t["success"] else 1] += 1
            if not t["success"] and self.GRASP_PHYS[gi_][0] == 0 and self.GRASP_PHYS[gi_][1] >= 3 and gi_ not in self.UNUSABLE:
                self.UNUSABLE.add(gi_)  # three rollouts, never a success: this grasp does not hold the object here
                print(f"grasp {gi_}: {self.GRASP_PHYS[gi_][1]} physics failures, no success -- dropped", flush=True)
            if t["success"]:
                self.chosen.append((tgt, traj, t))
                self.SWEEPS[id(traj)] = (sws[i], self.BATCH[i][0])  # where its furniture must give way, in copy i's coordinates
            elif self.best_failed is None or t["lift_m"] > self.best_failed[2]["lift_m"]:
                self.best_failed = (tgt, traj, t)
        print(f"[batch {self.n_batches[0]}] {len(self.pending)} rollouts in {dt:.1f} s ({dt / len(self.pending):.1f} s each)", flush=True)
        if video is not None:
            self.services.encode_video(video, video.parent / f"{self.etag0}.mp4", self.args.fps)
            shutil.rmtree(video, ignore_errors=True)
        self.n_batches[0] += 1
        self.pending.clear()

    def note_unusable(self, gi):
        """A pool grasp whose pregrasp hand no opening or back-off clears of the target: listed beside its pool
        (grasps/<hand>/<mesh>__pool/unusable.txt), where calibrated_scene_reconstruction.grasping.synth's selection leaves it out once listed
        twice."""
        pf = self.records[gi].get("_pool_file") if self.hand and gi < len(self.records) else None
        if pf:
            pool = Path(pf).parents[1]
            with open(self.s.dir / pool / "unusable.txt", "a") as f:
                f.write(f"{Path(pf).relative_to(pool).as_posix()}\n")

    def set_resume_state(self, si):
        """--resume-states: the scene as failure state si left it -- the robot on its base, the cup where it lay (the task
        frame, the planner's view of the target and the grasp targets follow it); False for a cup not standing upright."""
        c = np.asarray(self.RESUME["cup"][si], float)
        if self.services.tilt_deg(c[3:7]) > 15.0:
            return False
        b = [float(v) for v in self.RESUME["base"][si]]
        self.move_base(*b)
        self.bx, self.by, self.byaw = b
        self.p_obj, self.q_obj = c[:3].copy(), c[3:7].copy()
        self.T_obj = np.eye(4)
        self.T_obj[:3, :3], self.T_obj[:3, 3] = Rotation.from_quat(self.q_obj, scalar_first=True).as_matrix(), self.p_obj
        self.tf = TaskFrame(self.s.scene_graph, self.p_obj)
        self.planner.tf = self.tf
        self.planner.obj = [self.services.hull_planes(P @ self.T_obj[:3, :3].T + self.p_obj) for P in self.parts] if self.parts else \
            self.services.hull_planes(self.pts @ self.T_obj[:3, :3].T + self.p_obj)
        self.TARGETS = [self.services.bodex_target(r, self.T_obj, self.rob, self.side, self.asset) for r in self.records[:self.args.max_grasps]]
        self.CUP_START = c
        return True

    def plan_grasp(self, gi, tgt, pass_=0, round_=0):
        """One grasp's planning as a batch unit (sim.batch_kinematics): its feasibility, then --variations goals and
        plans on a slot robot (the copy's articulation serves the IK); returns [(kind, payload, line)] with kind "trial"
        (a rejection, as the sequential search records it) or "plan" ((tgt, traj, t), validated later)."""
        def unit(slot):
            pl = copy.copy(self.planner)
            pl.rob = slot
            pl.pts_fn = lambda h: slot.call(self.hand_pts_for, h)  # posing the main robot: the main thread's
            r_ = np.random.default_rng([self.args.seed, gi, pass_, sum(map(ord, self.etag0))] + ([round_] if round_ else []))
            out = []
            s_ = getattr(slot, "_slot", None)
            if s_ is not None and self.COPY_BASE and self.COPY_PLANS[s_] >= self.args.plans_per_copy:
                return out  # this copy's base has its plan for the round
            t0 = time.time()
            if s_ is not None and self.COPY_BASE and (s_, gi) in self.COPY_INFEASIBLE:
                return out
            ok, qs, errs = pl.feasible(tgt, r_, restarts=1 if s_ is not None and self.COPY_BASE else 3)
            if not ok:
                if s_ is not None and self.COPY_BASE:  # out of reach from this copy's base, perhaps not from the others'
                    self.COPY_INFEASIBLE.add((s_, gi))
                else:
                    self.INFEASIBLE.add(gi)
                out.append(("trial", {"grasp": gi, "source": tgt.source, "ik_ok": False, "ik_err": [list(e) for e in errs]},
                            f"grasp {gi} {tgt.params}: no IK (pre {errs[0][0]*1000:.0f} mm, grasp {errs[1][0]*1000:.0f}"
                            f" mm) [{time.time() - t0:.1f} s]"))
                return out
            for v in range(self.args.variations):
                t0 = time.time()
                stats = {}
                ci = getattr(slot, "_slot", None)  # --object-window: every copy stands on its own base this round
                cb = self.COPY_BASE[ci] if ci is not None and self.COPY_BASE else None
                if cb is not None and self.COPY_PLANS[ci] >= self.args.plans_per_copy:  # this base has its plan: leave the
                    return out  # rest of the round to the others (an easy base would otherwise take every plan)
                goal = self.new_goal(tgt, qs[1], pl, r_, stats, base=cb and cb["base"], quadrant=cb and cb["quadrant"]) \
                    if self.TASK == "place" else None
                if self.TASK == "place" and goal is None:
                    out.append(("trial", {"grasp": gi, "variation": v, "source": tgt.source, "params": tgt.params,
                                          "ik_ok": True, "plan": {"fail": "no place goal", "rejected": dict(stats)}}, None))
                    continue
                q0_ = None
                if cb is not None and "q_start" in cb:  # this copy stands where the policy did: from its state
                    q0_ = self.START_STATES[cb["q_start"]].copy()
                    q0_[self.FIXED_DOFS] = self.q_home[self.FIXED_DOFS]
                elif self.START_STATES is not None and self.START_BASES is None and r_.uniform() < self.args.start_states_p:
                    q0_ = self.START_STATES[int(r_.integers(len(self.START_STATES)))].copy()  # the arm and hand as the policy
                    q0_[self.FIXED_DOFS] = self.q_home[self.FIXED_DOFS]  # left them; the joints this robot holds (torso, head, left arm)
                traj, info = pl.plan(tgt, r_, qs[0], place=None if goal is None else (goal, self.p_obj, self.PCFG), q_start=q0_)
                info["plan_s"] = round(time.time() - t0, 1)
                if q0_ is not None:
                    info["policy_start"] = True  # a DAgger plan: on from a state the policy reached
                if cb is not None:
                    info["base"] = list(cb["base"])
                    if traj is not None:
                        traj.info["base"] = list(cb["base"])
                t = {"grasp": gi, "variation": v, "source": tgt.source, "params": tgt.params, "ik_ok": True,
                     "plan": {k: info[k] for k in info if k not in ("source", "params")}}
                if traj is None:
                    out.append(("trial", t, f"grasp {gi} var {v}: plan rejected ({info.get('fail')}) [{info['plan_s']} s]"))
                    if "into the target and neither opening" in str(info.get("fail")):  # the grasp's own geometry
                        self.UNUSABLE.add(gi)
                        self.note_unusable(gi)
                        return out
                    continue
                if self.args.action_noise > 0 and r_.uniform() < self.args.noise_p:
                    traj.noise = self.exec_noise(traj, r_)
                out.append(("plan", (tgt, traj, t), f"grasp {gi} var {v}: planned [{info['plan_s']} s]"))
                if cb is not None:
                    self.COPY_PLANS[ci] += 1
                    if self.COPY_PLANS[ci] >= self.args.plans_per_copy:
                        return out
            return out
        return unit

    def draw_base(self):
        """--base-jitter: a new base for the next planning round -- an accepted placement (shifted with the target)
        moved by up to the jitter, clear of the room at home -- the main robot and every copy onto it; False when 30
        draws all stood in the room."""
        jx, jyaw = (float(v) for v in self.args.base_jitter.split(","))
        room_boxes = self.services.room_obstacles(self.ts)
        win = [float(v) for v in self.args.object_window.split(",")] if self.args.object_window else None
        for _ in range(60 if win else 30):
            c = self.acc[int(self.rng.integers(len(self.acc)))]
            yaw_ = c["base"][2] + self.rng.uniform(-jyaw, jyaw)
            if win:  # the object at a uniform point of the window in the new base's frame
                fw, lf = self.rng.uniform(win[0], win[1]), self.rng.uniform(win[2], win[3])
                a_ = np.radians(yaw_)
                x = self.p_obj[0] - (np.cos(a_) * fw - np.sin(a_) * lf)
                y = self.p_obj[1] - (np.sin(a_) * fw + np.cos(a_) * lf)
            else:
                x, y = c["base"][0] + self.shift[0] + self.rng.uniform(-jx, jx), c["base"][1] + self.shift[1] + self.rng.uniform(-jx, jx)
            if self.PHOTO_BASE and np.hypot(x - self.PHOTO_BASE[0], y - self.PHOTO_BASE[1]) > self.PHOTO_BASE[2]:
                continue
            self.move_base(x, y, yaw_)
            self.body.restart(self.rob)
            self.body.sample()
            if not any(d > self.services.DEPTH_OK for _, _, d in self.body.in_boxes(room_boxes)):
                self.bx, self.by, self.byaw = x, y, yaw_
                self.base_check.append({"candidate": c["id"], "base": [x, y, yaw_], "jittered": True,
                                   **({"object_in_base": [round(fw, 3), round(lf, 3)]} if win else {})})
                print(f"base for the next round: ({x:.2f}, {y:.2f}, {yaw_:.0f}) from cand {c['id']}", flush=True)
                return True
        print("no base draw clears the room" + (" with the object in the window (too far from the counter's edge?)"
                                                 if win else ""), flush=True)
        return False

    def feasible_quadrants(self, fw, lf):
        """The goal quadrants an object at (forward fw, left lf) in the base frame leaves within reach, the goal 0.15-0.40 m
        from it: in front only while the object is near (fw <= 0.45), behind only while it is not too near (fw >= 0.38)."""
        q = ["left", "right"]
        if fw <= 0.45:
            q.append("front")
        if fw >= 0.38:
            q.append("back")
        return [q_ for q_ in q if q_ in self.args.goal_quadrants.split(",")]

    def place_copy(self, i, base):
        off, rpath, _ = self.BATCH[i]
        self.services.place_robot(self.stage, self.spec, (base[0] + off[0], base[1] + off[1]), np.radians(base[2]), path=rpath,
                    support_top=self.support_top)

    def base_clear(self, x, y, yaw_, room_boxes):
        """The robot at home on base (x, y, yaw deg) clears the room boxes (HOME_PTS moved there; DEPTH_OK)."""
        if not self.HOME_PTS:
            self.reset(self.q_home)
            self.body.restart(self.rob)
            self.body.sample()
            Rb = Rotation.from_euler("z", np.radians(self.byaw)).as_matrix()
            P = np.concatenate([h.P for _, h in self.body.hulls])
            self.HOME_PTS.append((P - np.array([self.bx, self.by, 0.0])) @ Rb)  # world -> base frame (rows: Rb.T @ (p - b))
        R = Rotation.from_euler("z", np.radians(yaw_)).as_matrix()
        P = self.HOME_PTS[0] @ R.T + np.array([x, y, 0.0])
        for lo, hi in room_boxes.values():
            inside = np.all((P > lo + self.services.DEPTH_OK) & (P < hi - self.services.DEPTH_OK), axis=1)
            if inside.any():
                return False
        return True

    def goal_room(self, base, quadrant, tries=120):
        """Whether the support has room for the goal marker in this quadrant of the base (sample_goal's geometry: the
        support's edges, the obstacles, the object's start; no reach check -- the plan does that)."""
        import copy as copy_

        c = copy_.copy(self.PCFG)
        c.tries = tries
        obst = self.services.support_obstacles(self.tf.support_top, [b for b in self.planner.env if float(b[0][2] + (np.abs(b[2]) @ b[1])[2])
                                                  >= self.support_top + self.FLAT], self.OBJ_BOXES) + self.services.keep_out(self.s.scene_graph)
        return self.services.sample_goal(self.tf, self.p_obj, self.ext, self.sbox, self.rng, c, obstacles=obst, support=self.s.scene_graph["task"]["support"],
                           weights=self.WEIGHTS, base=base, surfaces=self.SURFACES, quadrant=quadrant) is not None

    def draw_copy_bases(self, rnd):
        """--object-window: a base per batch copy for this round -- the object uniform in the window of the base frame,
        the yaw an accepted placement's +- the --base-jitter degrees, clear of the room -- and the copy's goal quadrant:
        of those the object's place leaves within reach (feasible_quadrants), the one with the fewest episodes so far
        (QUAD_DONE plus this round's), ties at random. Each copy's robot placed there, then a reset. False when no copy
        found a base."""
        win = [float(v) for v in self.args.object_window.split(",")]
        jyaw = float(self.args.base_jitter.split(",")[1])
        room_boxes = self.services.room_obstacles(self.ts)
        load = dict(self.QUAD_DONE)
        self.COPY_BASE.clear()
        for i in range(len(self.BATCH)):
            if self.START_BASES is not None and self.rng.uniform() < self.args.start_states_p:  # from a policy state, on its base
                j = int(self.rng.integers(len(self.START_BASES)))
                x, y, yaw_ = (float(v) for v in self.START_BASES[j])
                a_ = np.radians(yaw_)
                d_ = np.asarray(self.p_obj[:2], float) - [x, y]
                fw, lf = np.cos(a_) * d_[0] + np.sin(a_) * d_[1], -np.sin(a_) * d_[0] + np.cos(a_) * d_[1]
                qs_ = [q for q in self.feasible_quadrants(fw, lf) if self.goal_room((x, y, yaw_), q)]
                if qs_:
                    m = min(load[q] for q in qs_)
                    q_ = str(self.rng.choice([q for q in qs_ if load[q] == m]))
                    load[q_] += 1
                    self.COPY_BASE.append({"base": (x, y, yaw_), "quadrant": q_, "object_in_base": (round(fw, 3), round(lf, 3)),
                                      "q_start": j})
                    continue
            for _ in range(80):
                c = self.acc[int(self.rng.integers(len(self.acc)))]
                yaw_ = c["base"][2] + self.rng.uniform(-jyaw, jyaw)
                fw, lf = self.rng.uniform(win[0], win[1]), self.rng.uniform(win[2], win[3])
                a_ = np.radians(yaw_)
                x = self.p_obj[0] - (np.cos(a_) * fw - np.sin(a_) * lf)
                y = self.p_obj[1] - (np.sin(a_) * fw + np.cos(a_) * lf)
                if self.PHOTO_BASE and np.hypot(x - self.PHOTO_BASE[0], y - self.PHOTO_BASE[1]) > self.PHOTO_BASE[2]:
                    continue  # the object out of reach from near the photo's base
                qs_ = [q for q in self.feasible_quadrants(fw, lf) if self.goal_room((x, y, yaw_), q)] \
                    if self.base_clear(x, y, yaw_, room_boxes) else []
                if qs_:
                    m = min(load[q] for q in qs_)
                    q_ = str(self.rng.choice([q for q in qs_ if load[q] == m]))
                    load[q_] += 1
                    self.COPY_BASE.append({"base": (float(x), float(y), float(yaw_)), "quadrant": q_,
                                      "object_in_base": (round(fw, 3), round(lf, 3))})
                    break
            else:
                self.COPY_BASE.append(None)
        if not any(self.COPY_BASE):
            print("no copy found a base clear of the room with the object in the window", flush=True)
            self.COPY_BASE.clear()
            return False
        for i, cb in enumerate(self.COPY_BASE):  # a copy without a base takes another copy's
            if cb is None:
                self.COPY_BASE[i] = dict(next(c for c in self.COPY_BASE if c))
            self.place_copy(i, self.COPY_BASE[i]["base"])
        self.reset(self.q_home)
        self.COPY_PLANS[:] = [0] * len(self.COPY_BASE)
        self.COPY_INFEASIBLE.clear()
        self.base_check.append({"round": rnd, "copies": [dict(c) for c in self.COPY_BASE]})
        print(f"round {rnd}: copy bases " + " ".join(f"{c['quadrant'][0]}{c['object_in_base']}" for c in self.COPY_BASE),
              flush=True)
        return True

    def draw_policy_base(self):
        """--policy-new-bases: a new base for the next rollout as draw_copy_bases draws a copy's (the object uniform in the
        window, the yaw an accepted placement's +- the jitter, clear of the room) and the goal quadrant, the least drawn of
        those feasible; the robot moved there. None when no draw clears the room."""
        win = [float(v) for v in self.args.object_window.split(",")]
        jyaw = float(self.args.base_jitter.split(",")[1])
        room_boxes = self.services.room_obstacles(self.ts)
        for _ in range(200):
            c = self.acc[int(self.rng.integers(len(self.acc)))]
            yaw_ = c["base"][2] + self.rng.uniform(-jyaw, jyaw)
            fw, lf = self.rng.uniform(win[0], win[1]), self.rng.uniform(win[2], win[3])
            a_ = np.radians(yaw_)
            x = self.p_obj[0] - (np.cos(a_) * fw - np.sin(a_) * lf)
            y = self.p_obj[1] - (np.sin(a_) * fw + np.cos(a_) * lf)
            if self.PHOTO_BASE and np.hypot(x - self.PHOTO_BASE[0], y - self.PHOTO_BASE[1]) > self.PHOTO_BASE[2]:  # as the collection's draws
                continue
            qs_ = [q for q in self.feasible_quadrants(fw, lf) if self.goal_room((x, y, yaw_), q)] \
                if self.base_clear(x, y, yaw_, room_boxes) else []
            if qs_:
                m = min(self.POLICY_QUAD[q] for q in qs_)
                q_ = str(self.rng.choice([q for q in qs_ if self.POLICY_QUAD[q] == m]))
                self.POLICY_QUAD[q_] += 1
                self.move_base(float(x), float(y), float(yaw_))
                self.bx, self.by, self.byaw = float(x), float(y), float(yaw_)
                print(f"rollout base ({x:.2f}, {y:.2f}, {yaw_:.0f}), object in base ({fw:.2f}, {lf:.2f}), goal {q_}",
                      flush=True)
                return q_
        print("no rollout base clears the room with the object in the window", flush=True)
        return None

    def batch_search(self, on_round=None):
        """Batch planning: up to --batch grasps planned at once on the scene copies' robots, each round until a
        validation batch is full, then validated (run_batch) and, with `on_round`, recorded right away; with
        --base-jitter every round after the first stands on a new base (draw_base)."""

        per_round = bool(self.args.object_window)  # every round a new base: what one base cannot reach, the next may

        def fresh_units(r):
            return (self.plan_grasp(gi, tgt, k, r) for k in range(self.args.plan_passes) for gi, tgt in enumerate(self.TARGETS)
                    if gi not in self.INFEASIBLE and gi not in self.UNUSABLE)

        units = fresh_units(0)
        ran_out, rnd = False, 0
        while (per_round or not ran_out) and len(self.RECORDED) + len(self.chosen) < self.args.episodes and self.n_phys < self.args.max_trials \
                and rnd < self.args.max_rounds:
            if self.args.fail_fast and self.n_phys >= self.args.fail_fast and not (self.RECORDED or self.chosen):
                print(f"[fail-fast] {self.n_phys} physics trials and no success: giving the configuration up", flush=True)
                break
            if per_round:
                if not self.draw_copy_bases(rnd):
                    break
            elif rnd and self.args.base_jitter != "0,0" and self.base_order and not self.draw_base():
                break
            if per_round:
                self.INFEASIBLE.clear()
                units = fresh_units(rnd + 1)
            rnd += 1
            self.reset(self.q_home)
            kin = self.services.BatchKinematics(self.world, self.rob, [p_ for _, p_, _ in self.BATCH], [o_ for o_, _, _ in self.BATCH])
            got = [0]

            def take(res):
                for kind, payload, line in res:
                    if line:
                        print(line, flush=True)
                    if kind == "trial":
                        self.trials.append(payload)
                    else:
                        self.pending.append(payload)
                        got[0] += 1

            t0 = time.time()
            _, ran_out = kin.run(units, stop=lambda: len(self.pending) >= self.args.batch, on_result=take)
            print(f"[plan] {got[0]} plans in {time.time() - t0:.1f} s on {len(self.BATCH)} slots", flush=True)
            while len(self.pending) >= self.args.batch or ((ran_out or on_round) and self.pending):
                batch_items, rest = self.pending[:self.args.batch], self.pending[self.args.batch:]
                self.pending[:] = batch_items
                self.run_batch()
                self.pending.extend(rest)
                if len(self.RECORDED) + len(self.chosen) >= self.args.episodes or self.n_phys >= self.args.max_trials:
                    break
            if on_round and self.chosen:  # recorded on this round's base before the next one moves it
                self.RECORDED.extend(on_round(list(self.chosen)))
                self.chosen.clear()
                self.pending.clear()  # plans of this base are not valid on the next

    def ee_command(self, a, q_prev):
        """An end-effector action (PolicyRunner: the hand pose in the robot's base frame, the hand's joint targets) as
        joint targets: the pose to world, the arm's IK (the spec's ik_joints and ik_limits) warm-started from the previous
        command on the IK robot beside the room (the live one is never teleported); the hand's joints as given, every
        other joint as before. Returns (q, IK error (m, rad))."""
        Rb = Rotation.from_euler("z", np.radians(self.byaw)).as_matrix()
        p_w, R_w = Rb @ np.asarray(a["p"], float) + np.array([self.bx, self.by, 0.0]), Rb @ np.asarray(a["R"], float)
        q_ik, err = self.ik_rob.solve_ik(self.side, p_w + self.IK_OFF, R_w, q_prev, iters=60, tol=(0.002, 0.02))
        q = np.asarray(q_prev, float).copy()
        cols = [self.rob.idx[c] for c in self.arm["ik_joints"]]
        q[cols] = q_ik[cols]
        for i, v in a["hand"].items():
            q[int(i)] = v
        return q, err

    def furniture_state(self, i, root=None):
        """{prim path: (translate op values, active)} of copy i's movable furniture (`root`: that scope's instead)."""
        out = {}
        for prim in self.services.children(self.stage, root or f"/World/Batch/env{i}/Support"):
            ops = {o.GetOpName(): o.Get() for o in self.services.UsdGeom.Xformable(prim).GetOrderedXformOps()
                   if o.GetOpType() == self.services.UsdGeom.XformOp.TypeTranslate}
            out[str(prim.GetPath())] = (ops, prim.IsActive())
        return out

    def restore_furniture(self, i, state=None):
        """Copy i's movable furniture back where the scene put it (a recorded batch slid some of it aside); `state`: a
        furniture_state to put back instead."""
        for path, (ops, active) in (state if state is not None else self.BATCH_FURNITURE[i]).items():
            prim = self.stage.GetPrimAtPath(path)
            prim.SetActive(active)
            for o in self.services.UsdGeom.Xformable(prim).GetOrderedXformOps():
                if o.GetOpType() == self.services.UsdGeom.XformOp.TypeTranslate:  # one sim.clearance.shift added: back to zero
                    o.Set(ops.get(o.GetOpName(), type(o.Get())(0.0, 0.0, 0.0)))

    def evaluate_policy(self, plans):
        return self.services.evaluate_policy(self, plans)

    def plans_from_episodes(self):
        """--from-episodes: [(None, Traj, {})] of this tag's successful recordings (episodes/<etag0>[__e<k>]): the
        commanded joints and segments (the place check reads the hand's open / closed postures from them) and the goal;
        each rollout stands on its recording's base."""
        import re

        plans, base = [], None
        for d in sorted((self.s / "episodes").glob(f"{self.etag0}*")):
            if not re.fullmatch(re.escape(self.etag0) + r"(__e\d+)?", d.name) or not (d / "raw" / "steps.npz").exists():
                continue
            r = json.loads((d / "result.json").read_text())
            if not r.get("success"):
                continue
            z = np.load(d / "raw" / "steps.npz")
            plans.append((None, self.services.Traj(q=np.asarray(z["action"], float), seg=np.asarray(z["segment"]),
                                     info={"goal": r["goal"], "source": d.name, "base": r["base"],
                                           "cup0": np.asarray(z["object_pose"][0], float).tolist()}), {}))
            base = r["base"]
        plans = plans[:self.args.episodes]
        if plans:  # each rollout moves to its own recording's base (--base-jitter: one per planning round)
            base = plans[0][1].info["base"]
            self.move_base(*base)
            self.bx, self.by, self.byaw = base
        return plans

    def render_recorded(self, names):
        """--render-cams: the named robot cameras over this tag's recorded episodes, replayed kinematically -- each frame's
        recorded joint state and cup pose set, one physics step (the cup falls 0.03 mm in it), rendered. The scene is the
        tag's variant, as recorded; furniture the recording slid aside stands where the variant has it."""
        import re

        for n in names:
            cam, w, h, _ = self.robot_cams[n]
            self.annots[n] = self.services.rgb_annotator(cam.GetPath(), w, h)
        n_done = 0
        for d in sorted((self.s / "episodes").glob(f"{self.etag0}*")):
            if not re.fullmatch(re.escape(self.etag0) + r"(__e\d+)?", d.name) or not (d / "raw" / "steps.npz").exists():
                continue
            meta_f = d / "raw" / "meta.json"
            meta = json.loads(meta_f.read_text())
            dst = Path(self.args.render_out) / d.name if self.args.render_out else d / "raw" / "frames"
            if not self.args.render_overwrite and not self.args.render_out and all(
                    (dst / f"{n}.mp4").exists() and n in meta["cameras"] for n in names):
                continue
            r = json.loads((d / "result.json").read_text())
            if "base" not in r:  # a failure record over the recording (an older evaluation's): no base to replay from
                print(f"[{d.name}] skipped: no base in result.json", flush=True)
                continue
            z = np.load(d / "raw" / "steps.npz")
            q, po = np.asarray(z["q"], float), np.asarray(z["object_pose"], float)
            if not np.allclose(r["base"], [self.bx, self.by, self.byaw]):
                self.move_base(*r["base"])
                self.bx, self.by, self.byaw = r["base"]
            if self.args.render_cup_offset:  # base frame -> world
                yw = np.radians(r["base"][2]); f_, l_ = self.args.render_cup_offset
                po = po.copy(); po[:, 0] += np.cos(yw) * f_ - np.sin(yw) * l_; po[:, 1] += np.sin(yw) * f_ + np.cos(yw) * l_
            if self.args.render_joint_offset:
                q = q.copy(); q[:, self.rob.idx[self.args.render_joint_offset[0]]] += float(self.args.render_joint_offset[1])
            self.reset(q[0], self.services.Traj(q=q, seg=np.asarray(z["segment"]), info={"goal": r.get("goal")}))
            dst.mkdir(parents=True, exist_ok=True)
            sinks, poses = {}, {"tcp": [], "cup": []}
            for k in range(len(q)):
                self.rob.teleport(q[k])
                self.obj.set_world_pose(po[k, :3], po[k, 3:7])
                self.obj.set_linear_velocity(np.zeros(3))
                self.obj.set_angular_velocity(np.zeros(3))
                self.world.step(render=True)
                if self.args.render_out:  # what was drawn: the hand and the cup, world frame
                    poses["tcp"].append(np.asarray(self.rob.tcp_pose(self.side)[0], float))
                    poses["cup"].append(np.asarray(self.obj.get_world_pose()[0], float))
                for n in names:
                    im = self.services.grab(self.annots[n])
                    if n not in sinks:
                        sinks[n] = self.services.VideoWriter(dst / f"{n}.mp4", im.shape[1], im.shape[0], self.args.fps)
                    sinks[n].write(im)
            for w_ in sinks.values():
                w_.close()
            if self.args.render_out:
                np.savez(dst / "drawn_poses.npz", tcp=np.asarray(poses["tcp"]), cup=np.asarray(poses["cup"]), base=np.asarray(r["base"]))
            if not self.args.render_out:
                for n in names:
                    _, w, h, K = self.robot_cams[n]
                    meta["cameras"][n] = {"K": K.tolist(), "size": [w, h]}
                meta_f.write_text(json.dumps(meta, indent=1))
            n_done += 1
            print(f"[{d.name}] rendered {', '.join(names)}: {len(q)} frames", flush=True)
            if self.args.render_max_episodes and n_done >= self.args.render_max_episodes:
                break
        print(f"[{self.etag0}] {n_done} recorded episodes rendered", flush=True)

    def finalize(self, traj, t):
        """A batch-validated rollout in the full scene: swept with movable furniture off, the furniture in the robot's way
        slides aside, then validated again with it colliding (as a single trial is) -- the recorded run must not push
        through a stool the batch copies never had in the way."""
        if t.get("scene") != "batch" or "final" in t:
            return t.get("success", True)
        self.services.support_collisions(self.stage, False)
        self.trial(traj, "furniture off")
        resolved = self.services.clear_robot(self.ts, self.sweep[0], self.target)
        self.services.support_collisions(self.stage, True)
        t2 = self.trial(traj, "final")
        self.services.support_collisions(self.stage, False)
        t.update({"final": t2, "success": t2["success"], "furniture": self.services.summarize(resolved)})
        pl = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t2.get("place", {}).items()}
        print(f"[final] {self.services.summarize(resolved)} | lift {t2['lift_m']:.3f} tilt {t2['tilt_deg']:.0f} {pl} -> {t2['success']}",
              flush=True)
        return t2["success"]

    def record(self, k, tgt, traj, t):
        return self.services.record(self, k, tgt, traj, t)

    def write_episode(self, out, etag, tgt, traj, t, o, logs, n_frames, final):
        return self.services.write_episode(self, out, etag, tgt, traj, t, o, logs, n_frames, final)

    def record_batches(self, cands):
        return self.services.record_batches(self, cands)

    def run(self):
        if self.args.render_cams:  # the recorded episodes' starts, no planning or trials
            self.args.from_episodes = True
        self.services.configure_render()
        self.spec = load_spec(self.args.robot_spec or self.args.robot)
        self.side = next(iter(self.spec["arms"]))
        self.arm = self.spec["arms"][self.side]
        self.s = self.services.Session(self.args.session)
        self.TASK = self.services.task_type(self.s.scene_graph, self.args.task_type)
        # place episodes live beside the pick ones: episodes/<tag>[__<robot>][__place][__e<k>]
        self.etag0 = (self.args.tag if self.args.robot == DEFAULT_ROBOT else f"{self.args.tag}__{self.args.robot}") + ("__place" if self.TASK == "place" else "")
        self.rng = np.random.default_rng([self.args.seed, sum(map(ord, self.etag0))])
        # where a run that records nothing says why: the first episode's directory, or -- adding to a tag's episodes
        # (--first-k) -- beside them, leaving the recorded ones alone
        FAIL_DIR = self.etag0 if self.args.first_k == 0 else f"{self.etag0}__add{self.args.first_k}"
        if self.args.policy or self.args.render_cams:  # an evaluation's or a render's failure record never lands on a recording
            FAIL_DIR = f"_runs_failed/{self.etag0}"
        variant = self.services.load_variant(self.args.variant)
        self.target = self.s.scene_graph["task"]["target"]
        self.PCFG = self.services.PlaceCfg()
        if self.args.style == "fixed":
            self.PCFG = self.services.fixed_style(self.PCFG)
        self.PCFG.pre_release_s = tuple(float(v) for v in self.args.pre_release_s.split(","))
        self.PCFG.done_hold_s = tuple(float(v) for v in self.args.done_hold_s.split(","))
        self.PCFG.direction = self.args.goal_direction
        self.PCFG.rel_dist = tuple(float(v) for v in self.args.goal_dist.split(","))
        self.PCFG.max_reach = self.args.goal_max_reach
        if self.args.goal_tries:
            self.PCFG.tries = self.args.goal_tries
        if self.args.transport_speed:
            self.PCFG.transport_speed = tuple(float(v) for v in self.args.transport_speed.split(","))
        self.hand = self.services.hand_name(self.spec, self.side)
        self.substitute = None
        if self.hand is None and self.arm["ee"]["type"] == "parallel":  # wider than the stroke: the largest YCB cup that fits

            variant, self.substitute = self.services.fit_target(self.s, variant, 0.8 * float(self.arm["ee"].get("max_width", 0.08)))  # ~1 cm margin
            if self.substitute:
                print(f"target {self.target} is wider than the {self.spec['name']} gripper: replaced by {self.substitute}", flush=True)
        self.prepared = self.services.prepare_variant(self.s.dir, variant)




        self.world = self.services.World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
        self.stage = self.services.omni.usd.get_context().get_stage()
        self.ts = self.services.build_scene(self.stage, self.services.prepare_scene(self.s.dir), self.prepared)
        self.support_top = float(self.s.scene_graph["furniture"][self.s.scene_graph["task"]["support"]]["box"][1][2])
        self.PHOTO_BASE = None  # (x, y, radius): --photo-base
        if self.args.base in ("random", "best"):
            pl = json.loads(self.placements_file().read_text())
            self.acc = [c for c in pl["candidates"] if c["accepted"]]
            if not self.acc:
                self.s.write_json(f"episodes/{FAIL_DIR}/result.json", {"robot": self.spec["name"], "success": False, "n_trials": 0,
                                                              "reason": f"no accepted placement in {self.placements_file().name}"})
                print(f"[{self.etag0}] object lift 0.000 m -> FAILED (no accepted placement)", flush=True)
                self.app.close()
                raise SystemExit(0)
            if self.args.photo_base > 0:  # where the photo shows the real robot: the one base (its yaw +- the jitter), drawn
                pb = (self.s.scene_graph.get("photo_robot") or {}).get("base")  # bases kept within --photo-base m of it
                if not pb:
                    raise SystemExit("--photo-base: no photo_robot.base in the scene graph")
                self.acc = [{"id": "photo", "base": [float(pb["xy"][0]), float(pb["xy"][1]), float(pb["yaw_deg"])],
                        "accepted": True, "coverage": 1.0}]
                self.PHOTO_BASE = (float(pb["xy"][0]), float(pb["xy"][1]), float(self.args.photo_base))
            self.best = self.acc[int(self.rng.integers(len(self.acc)))] if self.args.base == "random" else max(self.acc, key=lambda c: c["coverage"])
            # the placements' bases follow the target from where apps/placements.py had it: the built prim's position here
            prim_t = self.stage.GetPrimAtPath(f"/World/Objects/{self.target}")
            self.shift = np.asarray(self.services.UsdGeom.XformCache().GetLocalToWorldTransform(prim_t))[3, :2] - np.array(pl["object"][:2])
            if self.PHOTO_BASE:  # the photo's base stays where it is (not with the target)
                self.shift = np.zeros(2)
            self.bx, self.by, self.byaw = self.best["base"][0] + self.shift[0], self.best["base"][1] + self.shift[1], self.best["base"][2]
            # bases for the reach check below (apps/placements.py checked a point above the target, not these grasps): the
            # drawn one, the other accepted ones, then the collision-free reaching ones its head-camera coverage bar rejected
            self.base_order = [self.best] + [c for c in self.acc if c is not self.best] + sorted(
                (c for c in pl["candidates"] if not c["accepted"] and c.get("collision_free") and c.get("reachable")),
                key=lambda c: -c["coverage"])
        else:
            self.bx, self.by, self.byaw = (float(v) for v in self.args.base.split(","))
            self.base_order = None

        self.services.add_robot(self.stage, self.spec, (self.bx, self.by), np.radians(self.byaw), support_top=self.support_top)
        self.services.support_collisions(self.stage, False)
        self.BATCH = []  # [(offset, robot path, target path)]: copies of the whole scene beside the room (--batch)
        BATCH_FAR, BATCH_GAP = 20.0, 30.0  # m: a batch copy's camera clipping distance; the free space between copies
        if self.args.batch and not self.args.policy:
            # every collider, rigid body, physics material and resolved placement of the built scene, copied spec for spec
            # (Sdf.CopySpec) under an offset Xform: room planes and furniture boxes, static objects, the target and the other
            # dynamic objects, seats (collisions off, as in the validation below), the invented furniture (visual only); only the
            # lights stay behind


            # The copies are also where batch episodes are recorded: each gets the room's local lights (window, ceiling and
            # fill emitters; the sky dome and the sun are global and the same for every copy of one configuration), and the
            # copies stand BATCH_GAP apart, more than every camera's far clipping distance (BATCH_FAR): no camera of one copy
            # sees another copy, and another copy's lamps are that far behind walls
            layer = self.stage.GetRootLayer()
            rx, ry = self.ts.manifest["surfaces"]["room_x"], self.ts.manifest["surfaces"]["room_y"]
            cols = int(np.ceil(np.sqrt(self.args.batch)))
            sx, sy = rx[1] - rx[0] + BATCH_GAP, ry[1] - ry[0] + BATCH_GAP
            local_lights = [p for p in self.stage.GetPrimAtPath("/World/Lights").GetChildren()
                            if not p.IsA(self.services.UsdLux.DomeLight) and not p.IsA(self.services.UsdLux.DistantLight)]
            for i in range(self.args.batch):
                off = np.array([(1 + i % cols) * sx, (i // cols) * sy, 0.0])
                root = f"/World/Batch/env{i}"
                self.services.set_pose(self.services.UsdGeom.Xform.Define(self.stage, root).GetPrim(), off)
                for part in ("Room", "Static", "Objects", "Support", "Synthetic"):  # Synthetic: visual only, in view
                    if layer.GetPrimAtPath(f"/World/{part}"):
                        self.services.Sdf.CopySpec(layer, f"/World/{part}", layer, f"{root}/{part}")
                self.services.UsdGeom.Scope.Define(self.stage, f"{root}/Lights")
                for li in local_lights:
                    self.services.Sdf.CopySpec(layer, li.GetPath(), layer, f"{root}/Lights/{li.GetName()}")
                if self.args.batch_video:  # the tiled overview looks from above: ceiling and walls hidden (they still collide)
                    for prim in self.stage.GetPrimAtPath(f"{root}/Room").GetChildren():
                        if prim.GetName().startswith(("ceiling", "wall", "window")):
                            self.services.UsdGeom.Imageable(prim).MakeInvisible()
                rpath = f"/World/Batch/robot{i}"  # outside the offset Xform: the root joint anchors it in world coordinates
                self.services.add_robot(self.stage, self.spec, (self.bx + off[0], self.by + off[1]), np.radians(self.byaw), path=rpath, support_top=self.support_top)
                self.BATCH.append((off, rpath, f"{root}/Objects/{self.target}"))
            for li in local_lights:  # the main scene only plans (and records without --batch): its lamps off
                li.SetActive(False)
        self.IK_PATH, self.IK_OFF = "/World/IKRobot", None
        if self.args.policy:
            # --policy's end-effector actions are solved on a copy of the robot far beside the room (unseen, alone): IK on the
            # live articulation teleports its joints while iterating, and putting them back every frame breaks the hand's
            # contacts (the recorded end-effector actions so executed lifted 0 of 9 cups, the recorded joint commands 9 of 9)
            rx_ = self.ts.manifest["surfaces"]["room_x"]
            self.IK_OFF = np.array([rx_[1] - rx_[0] + BATCH_GAP, 0.0, 0.0])
            self.services.add_robot(self.stage, self.spec, (self.bx + self.IK_OFF[0], self.by + self.IK_OFF[1]), np.radians(self.byaw), path=self.IK_PATH, support_top=self.support_top)
            self.services.UsdGeom.Imageable(self.stage.GetPrimAtPath(self.IK_PATH)).MakeInvisible()
        self.cam_dr = (self.prepared.get("dr") or {}).get("camera")
        if self.cam_dr:  # a domain-randomized configuration (calibrated_scene_reconstruction.dr): the head camera's mount off by a calibration's error
            def _jitter(c):
                q = Rotation.from_euler("xyz", self.cam_dr["rot_deg"], degrees=True) * Rotation.from_quat(
                    c.get("quat_wxyz", (1, 0, 0, 0)), scalar_first=True)
                return {**c, "pos": (np.asarray(c.get("pos", [0, 0, 0]), float) + self.cam_dr["pos"]).tolist(),
                        "quat_wxyz": q.as_quat(scalar_first=True).tolist()}
            self.spec = {**self.spec, "cameras": [self._jitter(c) if c.get("role") == "head" else c for c in self.spec["cameras"]]}
        self.robot_cams = self.services.add_cameras(self.stage, self.spec, size=(self.W, self.H))
        self.third = self.services.add_camera(self.stage, "/World/Cams/third", self.services.pinhole(self.W, self.H), self.W, self.H, *self.s.config("views.json")["third_person"])
        self.BATCH_CAMS = []  # per batch copy {camera name: (camera, W, H, K)}, the third-person one included
        for i, (off, rpath, _) in enumerate(self.BATCH):
            eye, at = (np.asarray(v, float) for v in self.s.config("views.json")["third_person"])
            cams_i = self.services.add_cameras(self.stage, self.spec, path=rpath, size=(self.W, self.H), far=BATCH_FAR)
            cams_i["third"] = (self.services.add_camera(self.stage, f"/World/Batch/third{i}", self.services.pinhole(self.W, self.H), self.W, self.H, eye + off, at + off,
                                          far=BATCH_FAR), self.W, self.H, self.services.pinhole(self.W, self.H))
            self.BATCH_CAMS.append(cams_i)
        self.world.reset()
        self.rob = self.services.Robot(self.world, self.spec)
        self.ik_rob = self.services.Robot(self.world, self.spec, path=self.IK_PATH) if self.IK_OFF is not None else None
        self.q_home = self.rob.home()
        self.rob.teleport(self.q_home)
        for _ in range(self.phys_steps(90)):
            self.world.step(render=False)
        self.body = self.services.RobotSweep(self.stage, self.rob)
        self.obj = self.services.SingleRigidPrim(f"/World/Objects/{self.target}")
        self.p_obj, self.q_obj = (np.asarray(v, float) for v in self.obj.get_world_pose())
        self.CUP_START = None  # --resume-states: the cup pose (xyz + wxyz) reset() puts the cup at
        self.T_obj = np.eye(4)
        self.T_obj[:3, :3], self.T_obj[:3, 3] = Rotation.from_quat(self.q_obj, scalar_first=True).as_matrix(), self.p_obj
        self.pts = np.concatenate([np.asarray(self.services.UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in
                              self.services.Usd.PrimRange(self.stage.GetPrimAtPath(f"/World/Objects/{self.target}")) if p.IsA(self.services.UsdGeom.Mesh)])
        self.ext = self.pts.max(0) - self.pts.min(0)
        self.c_local = (self.pts.max(0) + self.pts.min(0)) / 2  # the object's centre in its prim frame (the prim origin is its bottom)
        self.tf = TaskFrame(self.s.scene_graph, self.p_obj)
        if self.PHOTO_BASE:  # the frame faces the robot from the edge on its side: the nearest edge of a target across the desk is
            self.to_b = np.asarray(self.PHOTO_BASE[:2]) - self.p_obj[:2]  # the far one, and the start pose (behind the target) went out of reach
            self.tf = TaskFrame(self.s.scene_graph, self.p_obj, edge=max(approach_edges(self.s.scene_graph, self.p_obj),
                                                          key=lambda e: float(np.dot(e["normal"], self.to_b))))
        wrist_pts = self.services.hand_points(self.rob, self.side, self.body)  # hand + wrist: analytic candidates are raised clear of the support
        self.hl = self.services.hand_links(self.stage, self.spec, self.side)
        self._pts_cache = {}




        self.sbox = self.s.scene_graph["furniture"][self.s.scene_graph["task"]["support"]]["box"]
        # the target as the planner's hand check sees it: its convex parts when the simulator collides with them (a BODex
        # target's solid, see grasping.targets), so a hand hooked through a handle is not refused by the whole mug's hull
        _tspec = self.services.target_spec(self.s, self.prepared)
        _mesh = self.services.local_mesh(self.s, _tspec)
        self.parts = self.services.collision_parts(self.s, self.services.mesh_key(_tspec, _mesh[0])) if _mesh is not None else None
        obj_planes = [self.services.hull_planes(P @ self.T_obj[:3, :3].T + self.p_obj) for P in self.parts] if self.parts else \
            self.services.hull_planes(self.pts @ self.T_obj[:3, :3].T + self.p_obj)
        self.planner = self.services.TrajectoryPlanner(self.rob, self.side, self.tf, self.q_home, self.hand_pts_for, obj_planes,
                                    cfg=self.services.traj_fixed_style(self.services.TrajCfg()) if self.args.style == "fixed" else None,
                                    body_links={n: L[::max(1, len(L) // 48)] for n, L in self.body.local.items() if n not in self.hl},
                                    support_xy=(self.sbox[0][:2], self.sbox[1][:2]),
                                    env=[b for b in self.services.scene_env_world(self.s, self.args.tag)  # not the support (its top is the floor check)
                                         if b[3] != f"room/{self.s.scene_graph['task']['support']}"])




        self.FLAT = 0.012  # m: a set piece less proud of the top than this (a cooktop) is a surface, not an obstacle
        self.FLAT_PIECES = []




        self.OBJ_BOXES = self._object_boxes()
        self.SURFACES = self._surfaces()
        self.WEIGHTS = self.services.receptacle_weights(self.s.scene_graph) if self.args.receptacles == "scene" else \
            {} if self.args.receptacles == "none" else {k: 1.0 for k in self.args.receptacles.split(",")}
        rcp.set_match(((self.s.scene_graph.get("task") or {}).get("place") or {}).get("match"))  # the photo's place target




        self.GOAL_STATS = {}  # the draws each check of the last goal sampling rejected




        # --- grasp targets ---------------------------------------------------------------------------------------------------
        if self.hand:
            tspec = self.services.target_spec(self.s, self.prepared)
            key = self.services.grasp_key(self.services.mesh_key(tspec, self.services.local_mesh(self.s, tspec)[0]), self.services.arm_direction(self.s, self.spec, self.side, tspec, self.args.tag,
                                                                                  photo_base=self.args.photo_base > 0),
                            self.services.scene_env(self.s, self.args.tag)[1])
            self.records = self.services.load_records(self.s, self.hand, key)
            if not self.records:  # none synthesized, or none passed the quality gate: a failed episode, not a crash
                why = "no grasp passed the quality gate" if (self.s / "grasps" / self.hand / key / "summary.json").exists() else \
                    "not synthesized (run calibrated_scene_reconstruction.grasping.synth)"
                shutil.rmtree(self.s / "episodes" / FAIL_DIR, ignore_errors=True)
                self.s.write_json(f"episodes/{FAIL_DIR}/result.json", {"robot": self.spec["name"], "success": False, "n_trials": 0,
                                                              "grasp_source": f"bodex:{self.hand}/{key}", "reason": why})
                print(f"[{self.etag0}] object lift 0.000 m -> FAILED (BODex grasps for {self.hand} on {key}: {why})", flush=True)
                self.app.close()
                raise SystemExit(0)
            self.asset = yaml.safe_load((ROOT / self.arm["ee"]["bodex"]["asset"]).read_text())
            self.TARGETS = [self.services.bodex_target(r, self.T_obj, self.rob, self.side, self.asset) for r in self.records[:self.args.max_grasps]]
            self.grasp_source = f"bodex:{self.hand}/{key}"

        else:
            cands = self.services.clear_support(candidates(self.tf, self.p_obj, self.ext, self.arm, self.args.max_grasps), wrist_pts, self.tf.support_top)
            if self.TASK == "place":  # carried around: held near its centre of mass height first (a cup pinched at its rim swings)
                self.z_com = self.p_obj[2] + mass_props(self.target, self.pts)["com"][2]  # pts: the prim's frame, origin at its bottom
                cands = sorted(cands, key=lambda c: abs(float(c["grasp"][2]) - self.z_com))
            self.TARGETS = [self.services.analytic_target(c, self.rob, self.side) for c in cands]
            self.grasp_source = f"analytic:{self.arm['ee']['grasp']}"
        print(f"target at {np.round(self.p_obj, 3)}, extent {np.round(self.ext, 3)}, base ({self.bx:.2f}, {self.by:.2f}, {self.byaw:.0f}), "
              f"robot {self.spec['name']}, {len(self.TARGETS)} grasps ({self.grasp_source})", flush=True)

        # --- physics ---------------------------------------------------------------------------------------------------------
        self.annots, self.frame, self.sweep, self.log = {}, [0], [None], [None]
        self.out = None
















        # executed-command noise (--action-noise): weight per segment -- none while the fingers close, hold or open
        # (near the target the arm alone, at the edge of its reach, cannot take much: 0.5 on the approach halved an arm-only
        # robot's physics success, 50 % -> 26 % at 0.08 rad)
        self.NOISE_W = {"transit": 1.0, "approach": 0.2, "close": 0.0, "squeeze": 0.0, "lift": 0.2, "transport": 1.0, "place": 0.2,
                   "release": 0.0, "retreat": 1.0}
















        self.BATCH_ANNOTS, self.BATCH_LOGS = {}, []  # per copy: its cameras' annotators; the last recorded batch's per-item logs
        self.SINKS = {}  # --video-only, single-scene recording: {stream: VideoWriter}


        # --- base: one that reaches the grasps ----------------------------------------------------------------------------


        self.base_check = []
        if self.base_order and self.args.reach_check > 0 and self.TARGETS:
            # a placement is accepted on reach to a point above the target; the synthesized grasps (a hand beside a cup, its
            # wrist turned) can lie out of the arm's reach from there. IK-check the first grasps from the drawn base and, when
            # it reaches fewer than `need`, from the other placements in base_order; keep the first that does (else the best)
            rng_reach = np.random.default_rng([self.args.seed, 1])
            checked = self.TARGETS[:self.args.reach_check]
            need = min(3, len(checked))
            room_boxes = self.services.room_obstacles(self.ts)
            for i, c in enumerate(self.base_order):
                x, y, yaw_ = c["base"][0] + self.shift[0], c["base"][1] + self.shift[1], c["base"][2]
                if i:
                    self.move_base(x, y, yaw_)
                # shifted with the target, a placement can stand in the room (a cousin's cup at another spot of a
                # peninsula put the base 18 cm into its body): the robot at home must clear the furniture boxes and walls
                self.reset(self.q_home)
                self.body.restart(self.rob)
                self.body.sample()
                pen = [(link, box, d) for link, box, d in self.body.in_boxes(room_boxes) if d > self.services.DEPTH_OK]
                if pen:
                    link, box, d = max(pen, key=lambda p_: p_[2])
                    self.base_check.append({"candidate": c["id"], "base": [x, y, yaw_], "accepted": bool(c["accepted"]),
                                       "coverage": c["coverage"], "grasps_reached": -1, "grasps_checked": 0,
                                       "blocked": f"{link} {100 * d:.1f} cm into {box}"})
                    print(f"base cand {c['id']} ({x:.2f}, {y:.2f}, {yaw_:.0f}): blocked ({self.base_check[-1]['blocked']})",
                          flush=True)
                    continue
                n = 0
                for tgt in checked:
                    self.reset(self.q_home)
                    n += bool(self.planner.feasible(tgt, rng_reach, restarts=1)[0])
                self.base_check.append({"candidate": c["id"], "base": [x, y, yaw_], "accepted": bool(c["accepted"]),
                                   "coverage": c["coverage"], "grasps_reached": n, "grasps_checked": len(checked)})
                print(f"base cand {c['id']} ({x:.2f}, {y:.2f}, {yaw_:.0f}){'' if c['accepted'] else ' (coverage-rejected)'}: "
                      f"{n} of {len(checked)} grasps in IK reach", flush=True)
                if n >= need:
                    break
            pick = max(self.base_check, key=lambda b: b["grasps_reached"])  # the first of the best
            if pick["grasps_reached"] < 0 and (self.args.object_window and (self.BATCH and not (self.args.policy or self.args.plan_only)
                                                                       or self.args.policy and self.args.policy_new_bases)
                                               or self.args.resume_states or self.args.from_episodes):  # the failure states and
                # the recordings bring their own bases
                # every shifted placement stands in the room (the object moved far from where the placements were made):
                # the rounds (or --policy-new-bases' rollouts) draw their bases around the object (draw_base) anyway
                print("no shifted placement clears the room: the rounds draw their bases around the object", flush=True)
                pick = dict(self.base_check[0], grasps_reached=0)
                self.base_check.append(pick)
            if pick["grasps_reached"] < 0:  # every placement stands in the room
                self.s.write_json(f"episodes/{FAIL_DIR}/result.json", {"robot": self.spec["name"], "success": False, "n_trials": 0,
                                                              "base_check": self.base_check,
                                                              "reason": "no placement clears the room"})
                print(f"[{self.etag0}] object lift 0.000 m -> FAILED (no placement clears the room)", flush=True)
                self.app.close()
                raise SystemExit(0)
            if pick is not self.base_check[-1]:
                self.move_base(*pick["base"])
            elif len(self.base_check) == 1:
                self.reset(self.q_home)
            self.bx, self.by, self.byaw = pick["base"]
            pick["chosen"] = True


        # --- search ----------------------------------------------------------------------------------------------------------
        self.trials, self.chosen, self.best_failed = [], [], None
        self.SWEEPS = {}  # id(traj) -> (its batch rollout's RobotSweep, that copy's offset)
        self.n_phys = 0
        self.pending, self.n_batches = [], [0]



        self.INFEASIBLE = set()  # grasps whose IK failed: not tried again on a later planning pass
        self.UNUSABLE = set()  # grasps whose plan fails wherever the base stands (the pregrasp hand inside the target): never again


        self.GRASP_PHYS = {}  # grasp -> [physics successes, failures] this run
        _ss = [Path(f.replace("{tag}", self.etag0)) for f in (self.args.start_states or [])]
        _ss = [np.load(f) for f in _ss if f.exists()]
        self.START_STATES = np.concatenate([z["q"] for z in _ss]) if _ss else None
        # a state's base (saved by --policy since the per-copy bases): a copy starting from it stands there
        self.START_BASES = np.concatenate([z["base"] for z in _ss]) if _ss and all("base" in z for z in _ss) else None
        # the joints neither the IK nor the hand moves: a start state from another robot's policy takes this one's home there
        self.FIXED_DOFS = np.array([i for i, n in enumerate(self.rob.names) if n not in self.arm["ik_joints"]
                               and not re.search(self.arm["ee"]["joints"], n)], int)
        if self.START_STATES is not None:
            print(f"{len(self.START_STATES)} policy states as plan starts (p {self.args.start_states_p})", flush=True)
        self._rs = [Path(f.replace("{tag}", self.etag0)) for f in (self.args.resume_states or [])]
        self._rs = [np.load(f) for f in self._rs if f.exists()]
        if self.args.resume_states and not self._rs:  # never a fresh collection in place of the resumed one
            sys.exit(f"--resume-states: no file for {self.etag0}")
        self.RESUME = {k_: np.concatenate([z[k_] for z in self._rs]) for k_ in ("q", "base", "cup", "kind")} if self._rs else None
        if self.RESUME is not None:
            print(f"{len(self.RESUME['q'])} failure states: {dict(zip(*np.unique(self.RESUME['kind'], return_counts=True)))}", flush=True)
            # the cup resting on the support only (a missed grasp, a drop, a push; a knock leaves it lying there): a freeze
            # leaves it in the hand, in mid-air
            self._keep = np.isin(self.RESUME["kind"], ["miss", "drop", "push", "knock"])
            self.RESUME = {k_: v[self._keep] for k_, v in self.RESUME.items()}
            print(f"{len(self.RESUME['q'])} failure states to resume from", flush=True)






        BATCH_SEARCH = bool(self.args.batch) and not (self.args.plan_only or self.args.policy or self.args.from_episodes)
        self.RECORDED = []  # --batch: the episodes recorded so far (record_batches, round by round)




        self.COPY_BASE = []  # --object-window: per batch copy {"base": (x, y, yaw deg), "quadrant"} of the current round
        self.COPY_PLANS = []  # plans made on each copy's base this round (at most --plans-per-copy)
        self.COPY_INFEASIBLE = set()  # (copy, grasp) without IK from that copy's base this round
        self.HOME_PTS = []  # the robot's hull points at home in its base frame (fast room checks of a candidate base)
        self.QUAD_DONE = {"front": 0, "left": 0, "back": 0, "right": 0}  # recorded episodes per goal quadrant (this run)












        self.POLICY_QUAD = {q: 0 for q in ("front", "left", "back", "right")}  # --policy-new-bases: goal quadrants drawn so far


        NO_BASE = [0]  # rounds in which draw_policy_base found no base






        if BATCH_SEARCH and self.args.no_record:
            self.batch_search()

        if self.RESUME is not None and not self.args.batch and not self.args.policy:  # one episode per failure state, its first plan that holds
            for si in self.rng.permutation(len(self.RESUME["q"])):
                if len(self.chosen) >= self.args.episodes or self.n_phys >= self.args.max_trials:
                    break
                if not self.set_resume_state(si):
                    continue
                kind = str(self.RESUME["kind"][si])
                for gi in self.rng.permutation(len(self.TARGETS))[:self.args.resume_grasps]:
                    tgt = self.TARGETS[gi]
                    self.reset(self.q_home)
                    ok, qs, errs = self.planner.feasible(tgt, self.rng)
                    if not ok:
                        continue
                    goal = self.new_goal(tgt, qs[1]) if self.TASK == "place" else None
                    if self.TASK == "place" and goal is None:
                        continue
                    t0 = time.time()
                    traj, self.info = self.planner.plan(tgt, self.rng, qs[0], place=None if goal is None else (goal, self.p_obj, self.PCFG),
                                              q_start=np.asarray(self.RESUME["q"][si], float),
                                              backoff=(self.rng.uniform(0.04, 0.08), self.rng.uniform(0.03, 0.08)))
                    if traj is None:
                        print(f"state {si} ({kind}) grasp {gi}: plan rejected ({self.info.get('fail')})", flush=True)
                        continue
                    self.info.update(plan_s=round(time.time() - t0, 1), policy_start=True, resume_state=int(si), resume_kind=kind)
                    traj.info.update(base=[self.bx, self.by, self.byaw], resume=int(si))
                    t = {"grasp": int(gi), "variation": 0, "source": tgt.source, "params": tgt.params, "ik_ok": True,
                         "plan": {k_: self.info[k_] for k_ in self.info if k_ not in ("source", "params")}}
                    self.n_phys += 1
                    t.update(self.trial(traj, "furniture off"))
                    self.trials.append(t)
                    print(f"state {si} ({kind}) grasp {gi}: lift {t['lift_m']:.3f} -> {t['success']}", flush=True)
                    if t["success"]:
                        self.chosen.append((tgt, traj, t))
                        break

        # --policy-layouts: the starts a previous run's search found (the plans' joints, segments and info; the generator's
        # state after the search, as this run's would be)
        LAYOUTS = (Path(self.args.policy_layouts.replace("{tag}", self.etag0)) if self.args.policy_layouts and self.args.policy
                   and self.args.policy_new_bases and not self.args.from_episodes else None)
        LAYOUTS_READ = LAYOUTS is not None and LAYOUTS.exists()
        if LAYOUTS_READ:
            _lay = pickle.loads(LAYOUTS.read_bytes())
            self.chosen = [(None, self.services.Traj(q=q_, seg=seg_, info=info_), {}) for q_, seg_, info_ in _lay["plans"]][:self.args.episodes]
            self.rng.bit_generator.state = _lay["rng"]
            self.reset(self.q_home)  # the scene stepped once, as the search's first try steps it
            print(f"[{self.etag0}] {len(self.chosen)} rollout starts from {LAYOUTS}", flush=True)

        # --policy-new-bases: the grasps again and again, each try on a new base (one base reaches few of them)
        for gi, tgt in enumerate([] if self.args.from_episodes or self.RESUME is not None or (self.args.batch and not (self.args.plan_only or self.args.policy))
                                 or LAYOUTS_READ else self.TARGETS * 30 if self.args.policy and self.args.policy_new_bases else self.TARGETS):
            if len(self.chosen) >= self.args.episodes or self.n_phys >= self.args.max_trials:
                break
            t0 = time.time()
            quad = None
            if self.args.policy and self.args.policy_new_bases and (quad := self.draw_policy_base()) is None:
                NO_BASE[0] += 1
                if NO_BASE[0] >= 5:  # 1000 draws without one: this configuration has no such base (the loop over the
                    print("no rollout base in 5 rounds of draws: giving up", flush=True)  # grasps would draw for hours)
                    break
                continue
            self.reset(self.q_home)
            ok, qs, errs = self.planner.feasible(tgt, self.rng)
            t_ik = time.time() - t0
            if not ok:
                self.trials.append({"grasp": gi, "source": tgt.source, "ik_ok": False, "ik_err": [list(e) for e in errs]})
                print(f"grasp {gi} {tgt.params}: no IK (pre {errs[0][0]*1000:.0f} mm, grasp {errs[1][0]*1000:.0f} mm) "
                      f"[{t_ik:.1f} s]", flush=True)
                continue
            for v in range(self.args.variations):
                if len(self.chosen) >= self.args.episodes or self.n_phys >= self.args.max_trials:
                    break
                t0 = time.time()
                self.reset(self.q_home)
                goal = self.new_goal(tgt, qs[1], quadrant=quad) if self.TASK == "place" else None
                if self.TASK == "place" and goal is None:
                    self.trials.append({"grasp": gi, "variation": v, "source": tgt.source, "params": tgt.params, "ik_ok": True,
                                   "plan": {"fail": "no place goal", "rejected": dict(self.GOAL_STATS)}})
                    continue
                traj, self.info = self.planner.plan(tgt, self.rng, qs[0], place=None if goal is None else (goal, self.p_obj, self.PCFG))
                self.info["plan_s"] = round(time.time() - t0, 1)
                t = {"grasp": gi, "variation": v, "source": tgt.source, "params": tgt.params, "ik_ok": True,
                     "plan": {k: self.info[k] for k in self.info if k not in ("source", "params")}}
                if traj is None:
                    self.trials.append(t)
                    print(f"grasp {gi} var {v}: plan rejected ({self.info.get('fail')}) [{self.info['plan_s']} s]", flush=True)
                    continue
                if self.args.plan_only or self.args.policy:
                    q_ = (goal.info.get("quadrant"), goal.info.get("on")) if goal is not None else None
                    print(f"grasp {gi} var {v}: planned, standoff {self.info['standoff_m']:.2f} m, goal {q_} [{self.info['plan_s']} s]",
                          flush=True)
                    if self.args.policy:  # a start pose and a goal for an evaluation rollout
                        traj.info["base"] = [self.bx, self.by, self.byaw]  # the rollout stands where it was planned
                        self.chosen.append((tgt, traj, t))
                        if len(self.chosen) >= self.args.episodes:
                            break
                    continue
                if self.args.batch:  # collect; validated together below
                    if self.args.action_noise > 0 and self.rng.uniform() < self.args.noise_p:
                        traj.noise = self.exec_noise(traj, self.rng)
                    self.pending.append((tgt, traj, t))
                    if len(self.pending) == self.args.batch:
                        self.run_batch()
                    continue
                self.n_phys += 1
                t0 = time.time()
                t.update(self.trial(traj, "furniture off"))
                t["trial_s"] = round(time.time() - t0, 1)
                if t["success"]:
                    resolved = self.services.clear_robot(self.ts, self.sweep[0], self.target)  # furniture in the way of this rollout slides aside
                    self.services.support_collisions(self.stage, True)  # moved furniture re-creates colliders: validate in the final scene
                    t2 = self.trial(traj, "final")
                    self.services.support_collisions(self.stage, False)
                    t.update({"final": t2, "success": t2["success"], "furniture": self.services.summarize(resolved)})
                self.trials.append(t)
                print(f"grasp {gi} {tgt.params} var {v}: lift {t['lift_m']:.3f} tilt {t['tilt_deg']:.0f} in hand "
                      f"{t['tcp_dist_m']:.2f} turned {t['inhand_rot_deg'] or 0:.0f} deg reach err "
                      f"{1000 * (t['reach_err_m'] or 0):.0f} mm -> {t['success']} | standoff {self.info['standoff_m']:.2f} m, rot "
                      f"{np.round(self.info['standoff_rot_deg'], 0).tolist()}, {self.info['frames']} frames [plan {self.info['plan_s']} s, "
                      f"physics {t['trial_s']} s]", flush=True)
                if t["success"]:
                    self.chosen.append((tgt, traj, t))
                elif self.best_failed is None or t["lift_m"] > self.best_failed[2]["lift_m"]:
                    self.best_failed = (tgt, traj, t)

        if LAYOUTS is not None and not LAYOUTS_READ and self.chosen:  # every start the search could find (a short list too)
            LAYOUTS.parent.mkdir(parents=True, exist_ok=True)
            _tmp = LAYOUTS.with_suffix(".tmp")
            _tmp.write_bytes(pickle.dumps({"plans": [(traj.q, traj.seg, traj.info) for _, traj, _ in self.chosen],
                                           "rng": self.rng.bit_generator.state}))
            _tmp.rename(LAYOUTS)
            print(f"[{self.etag0}] {len(self.chosen)} rollout starts -> {LAYOUTS}", flush=True)


        self.VISITED = []  # --policy: joint states the policy reached (DAgger-style plan starts for a next collection)
        self.VISITED_BASE = []  # the base each was reached from








        self.BATCH_FURNITURE = []
        self.SLIDE_HINTS = {}  # {furniture item: [shift]}: what earlier rollouts of this scene slid it by (sim.sweep.slide)








        if self.args.render_cams:
            self.render_recorded(self.args.render_cams)
            self.app.close()
            raise SystemExit(0)

        if self.args.policy:
            if self.args.from_episodes:
                self.chosen = self.plans_from_episodes()
                print(f"[{self.etag0}] {len(self.chosen)} recorded episodes as starts", flush=True)
            self.evaluate_policy(self.chosen)
            self.app.close()
            raise SystemExit(0)
        if self.args.batch and self.pending:
            self.run_batch()
        if self.args.plan_only:
            self.app.close()
            raise SystemExit(0)
        if self.args.no_record:
            n_ok = sum(bool(t.get("success")) for t in self.trials if "lift_m" in t)
            self.s.write_json(f"episodes/_stats/{self.etag0}__s{self.args.seed}.json", {
                "robot": self.spec["name"], "tag": self.args.tag, "seed": self.args.seed, "grasp_source": self.grasp_source,
                "base": [self.bx, self.by, self.byaw], "base_check": self.base_check, "n_trials": self.n_phys, "n_success": n_ok, "trials": self.trials})
            print(f"[{self.etag0}] seed {self.args.seed}: {n_ok} of {self.n_phys} physics trials succeeded", flush=True)
            self.app.close()
            raise SystemExit(0)


        # --- record ----------------------------------------------------------------------------------------------------------






        self.ENCODES = []  # episode videos encoding in the background (ffmpeg), waited for before the app closes
        self.ENCODER = ThreadPoolExecutor(4)




        if BATCH_SEARCH:
            self.batch_search(on_round=self.record_batches)
            self.chosen = self.RECORDED
        elif self.args.batch:
            self.chosen = self.record_batches(self.chosen)
        else:
            final = []
            for tgt, traj, t in self.chosen:
                if len(final) >= self.args.episodes:
                    break
                if "resume" in traj.info:
                    self.set_resume_state(traj.info["resume"])
                if self.finalize(traj, t):
                    final.append((tgt, traj, t))
                elif self.best_failed is None:
                    self.best_failed = (tgt, traj, t)
            self.chosen = final
            for k, (tgt, traj, t) in enumerate(self.chosen):
                if "resume" in traj.info:
                    self.set_resume_state(traj.info["resume"])
                self.record(k, tgt, traj, t)
        if not self.chosen and self.best_failed is not None and not self.args.batch and not self.args.first_k:  # the failure is visible
            self.record(0, *self.best_failed)
        elif not self.chosen:
            shutil.rmtree(self.s / "episodes" / FAIL_DIR, ignore_errors=True)  # no stale recording of an earlier run
            self.s.write_json(f"episodes/{FAIL_DIR}/result.json", {"robot": self.spec["name"], "success": False,
                                                             "grasp_source": self.grasp_source, "n_trials": self.n_phys,
                                                             "trials": self.trials, "base_check": self.base_check})
            print(f"[{self.etag0}] object lift 0.000 m -> FAILED (no successful trajectory in {self.n_phys} trials)", flush=True)
        for f_ in self.ENCODES:  # every episode video written before the process ends
            f_.result()
        self.app.close()
