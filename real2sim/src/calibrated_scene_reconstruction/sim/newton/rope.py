"""A rope straightened by a Franka holding it by friction alone, in Newton: the base scene of the rope letters
(apps/newton_rope_letter.py). The coupled scene is
Newton's franka_cable_ik_pick_place example (Isaac Lab's Lift-Cable: the FR3 in MuJoCo on PD targets from GPU IK,
the rope a VBD rod of capsules, the gripper bodies exposed to VBD as proxies, so the fingers hold the rope by contact
and friction, no attachment):

  a rope 0.5 m long and 10 mm thick lies on the table in a random curve (smooth, not crossing itself, its rest shape
  straight); the goal is a straight line of the rope's length at a random pose in front of the robot. The policy is
  DeformableRavens' oracle for its cable tasks: each pick-and-place takes the segment farthest from its goal (the
  rope matched to the line end to end, whichever way round is closer) and puts it there, the fingers across the
  rope, up to 20 actions. Success as theirs: every segment within 1.75 cm of its goal.

    CUDA_VISIBLE_DEVICES=<index> tools/run_newton.sh apps/newton_rope.py [--seed 0] [--viewer usd --output-path outputs/deformable/newton/rope/x.usd]

Writes the USD animation (--viewer usd) and a .json beside it: per
action the segment taken, whether it stayed in the fingers while carried and came off when they opened, the errors
before and after; the fraction within 1.75 cm over time, success, the physics rate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD

# the example's (Isaac Lab's) Franka start, gains, gripper and contact; m, kg, s
FRANKA_Q = [-3.6802115e-03, 2.3901723e-02, 3.6804110e-03, -2.3683236e00, -1.2918962e-04, 2.3922248e00, 7.8549200e-01, 0.04, 0.04]
CONTACT_KE, CONTACT_KD = 1.0e3, 1.0e-1
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)  # (x, y, z, w): the hand's z down, its fingers closing along world y
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0
SURFACE_Z = 0.251  # the table top (the robot stands on it)
TOL = 0.0175  # DeformableRavens: a segment is in place within 1.75 cm of its goal
IK_YAW_RANGE = {0: 1.2, 2: 1.0}  # joint index: the IK's +- limit (rad) on joints 1 and 3 (see build_ik)


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    w = wp.tid()
    joint_q[w, idx0] = finger_pos[w]
    joint_q[w, idx1] = finger_pos[w]


@wp.kernel
def set_task_targets(target_positions: wp.array[wp.vec3], target_rotations: wp.array[wp.vec4], finger_pos: wp.array[float],
                     pos: wp.vec3, rot: wp.vec4, grip_width: float):
    w = wp.tid()
    target_positions[w] = pos
    target_rotations[w] = rot
    finger_pos[w] = grip_width


def quat_mul(a, b):
    """a * b for quaternions (x, y, z, w): b's rotation first."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def yaw_quat(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def gripper_across(tangent_angle):
    """The hand pointing down, its fingers closing across a rope running at tangent_angle (rad) in the plane."""
    a = (tangent_angle + np.pi / 2) % np.pi - np.pi / 2  # the fingers are symmetric: within +-90 deg
    return quat_mul(yaw_quat(a), GRIPPER_DOWN), a


def random_curve(rng, n_seg, seg_len, centre, tries=200):
    """A smooth planar curve of n_seg equal segments (its nodes, (n_seg + 1, 2)) that does not come within 3 cm of
    itself: the heading a random sum of a few sines along the length (SoftGym-like random rope, settled flat)."""
    s = np.linspace(0.0, 1.0, n_seg)
    for _ in range(tries):
        h = rng.uniform(-np.pi, np.pi) + sum(rng.uniform(-1.2, 1.2) * np.sin(k * np.pi * s + rng.uniform(0, 2 * np.pi))
                                             for k in (1, 2, 3))
        nodes = np.concatenate([[[0.0, 0.0]], np.cumsum(seg_len * np.stack([np.cos(h), np.sin(h)], 1), 0)])
        nodes += np.asarray(centre) - nodes.mean(0)
        mid = (nodes[1:] + nodes[:-1]) / 2
        d = np.linalg.norm(mid[:, None] - mid[None], axis=2)
        far = np.abs(np.arange(n_seg)[:, None] - np.arange(n_seg)[None]) > 2
        if d[far].min() > 0.03:
            return nodes
    raise RuntimeError("no curve without a near crossing")


class RopeStraighten:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.rng = np.random.default_rng(args.seed)
        self.radius, self.n_seg = args.radius, args.segments
        self.seg_len = args.length / args.segments

        self.build_scene()
        self.control = self.model.control()
        self.build_solvers()
        self.build_ik()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit",
                                                           shape_pairs_filtered=self.ground_shape_pairs())
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)
        newton.examples.configure_coupled_view(self, args)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.lay_out_rope()

        # the goal: a straight line of the rope's length, its centre and direction random in front of the robot
        c = np.array([self.rng.uniform(0.45, 0.55), self.rng.uniform(-0.1, 0.1)])
        a = self.rng.uniform(-np.pi / 2, np.pi / 2)
        u = np.array([np.cos(a), np.sin(a)])
        self.goal = c + np.outer((np.arange(self.n_seg) + 0.5) * self.seg_len - args.length / 2, u)  # per segment
        self.goal_angle = a

        self.keys = [[1.5, *self.home_tcp, *GRIPPER_DOWN, GRIP_OPEN]]  # settle with the hand up
        self.key_times = [1.5]
        self.actions = []
        self.done = False
        self.log = {"t": [], "fraction_in_place": [], "max_error_m": []}
        self.wall = 0.0
        self.capture()

    # --- the scene
    def add_franka(self, builder):
        builder.add_urdf(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
                         xform=wp.transform(wp.vec3(0.0, 0.0, SURFACE_Z), wp.quat_identity()), floating=False,
                         enable_self_collisions=False, parse_visuals_as_colliders=False, force_show_colliders=False)
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

    def build_scene(self):
        a = self.args
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        b.rigid_gap = 0.01
        SolverMuJoCo.register_custom_attributes(b)
        SolverVBD.register_custom_attributes(b)
        fb, fj, fs = b.body_count, b.joint_count, b.shape_count
        self.add_franka(b)
        b.joint_target_ke[:7] = [400.0] * 7  # Isaac Lab FRANKA_PANDA_HIGH_PD_CFG, a stiff gripper
        b.joint_target_kd[:7] = [80.0] * 7
        b.joint_target_ke[7:9] = [1000.0, 1000.0]
        b.joint_target_kd[7:9] = [100.0, 100.0]
        b.joint_effort_limit[:4] = [87.0] * 4
        b.joint_effort_limit[4:7] = [12.0] * 3
        b.joint_effort_limit[7:9] = [a.grip_force, a.grip_force]
        b.joint_armature[:7] = [1.0e-3] * 7
        b.joint_armature[7:9] = [0.0, 0.0]
        self.franka_bodies = list(range(fb, b.body_count))
        self.franka_joints = list(range(fj, b.joint_count))
        self.franka_shapes = list(range(fs, b.shape_count))
        gravcomp = b.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka_bodies:
            gravcomp.values[body] = 1.0

        # the rope, built straight (its rest shape) and laid out curved in the state afterwards
        rb, rj, rs = b.body_count, b.joint_count, b.shape_count
        cfg = newton.ModelBuilder.ShapeConfig(density=a.density, ke=CONTACT_KE, kd=CONTACT_KD, mu=a.mu, margin=0.0, gap=0.01)
        rod = newton.Rod.create_straight(start=wp.vec3(0.5 - a.length / 2, 0.0, SURFACE_Z + self.radius), direction=wp.vec3(1.0, 0.0, 0.0),
                                         length=a.length, segment_count=self.n_seg, twist_total=0.0, radius=self.radius)
        b.add_rod(rod=rod, body_frame_origin="com", cfg=cfg, stretch_stiffness=a.stretch, stretch_damping=1.0e-1,
                  bend_stiffness=a.bend, bend_damping=2.0e-3 * a.bend, label="rope")
        self.rope_bodies = list(range(rb, b.body_count))
        self.rope_joints = list(range(rj, b.joint_count))
        self.rope_shapes = list(range(rs, b.shape_count))
        self.gripper_bodies = [body for body in self.franka_bodies
                               if "hand" in b.body_label[body] or "finger" in b.body_label[body]]
        for shape, body in enumerate(b.shape_body):
            if body in self.gripper_bodies:
                b.shape_material_ke[shape], b.shape_material_kd[shape], b.shape_material_mu[shape] = CONTACT_KE, CONTACT_KD, a.mu
        plane = newton.ModelBuilder.ShapeConfig(ke=CONTACT_KE, kd=CONTACT_KD, mu=a.mu, margin=0.0, gap=0.01)
        self.ground_shapes = [b.add_ground_plane(height=SURFACE_Z, cfg=plane, label="table")]
        b.color()
        self.model = b.finalize()
        self.device = self.model.device

    def build_solvers(self):
        a = self.args
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="mjc", solver=lambda v: SolverMuJoCo(
                    model=v, solver="newton", integrator="implicitfast", cone="elliptic", iterations=a.mujoco_iterations,
                    ls_iterations=a.mujoco_ls_iterations, use_mujoco_contacts=False, njmax=256, nconmax=64),
                    bodies=self.franka_bodies, joints=self.franka_joints),
                SolverCoupled.Entry(name="vbd", solver=lambda v: SolverVBD(
                    model=v, iterations=a.vbd_iterations, rigid_compliant_alm=True, rigid_contact_history=False),
                    bodies=self.rope_bodies, joints=self.rope_joints),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(
                    source="mjc", destination="vbd", bodies=self.gripper_bodies, mass_scale=1.0, mode="lagged",
                    collision_pipeline=lambda model: newton.examples.create_collision_pipeline(model, broad_phase="explicit"),
                    collide_interval=1)],
                iterations=1))

    def ground_shape_pairs(self):
        dynamic = set(self.franka_shapes) | set(self.rope_shapes)
        ground = set(self.ground_shapes)
        pairs = [(int(p), int(q)) for p, q in self.model.shape_contact_pairs.numpy() if ({int(p), int(q)} & dynamic) and ({int(p), int(q)} & ground)]
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.device)

    def build_ik(self):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self.add_franka(b)  # a Franka-only model for the IK (its coordinates first in the coupled model too)
        self.ik_model = b.finalize(device=self.device)
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((1, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, -1))
        self.finger_pos = wp.full(1, GRIP_OPEN, dtype=float, device=self.device)
        hand = next(i for i, lbl in enumerate(self.ik_model.body_label) if lbl.endswith("fr3_hand"))
        self.home_tcp = (0.45, 0.0, SURFACE_Z + 0.25)
        self.rest_tcp = (0.25, 0.0, SURFACE_Z + 0.3)  # at the end: back towards the base, off the rope
        self.ik_pos = wp.array([wp.vec3(*self.home_tcp)], dtype=wp.vec3, device=self.device)
        self.ik_rot = wp.array([wp.vec4(*GRIPPER_DOWN)], dtype=wp.vec4, device=self.device)
        # the IK's limits: the arm's, the shoulder and upper arm yaw (joints 1 and 3) kept near their middle so the
        # solution, warm-started from the last, does not drift along the arm's redundancy (one hand pose, joint 1 and
        # 3 turned opposite ways) into far, swinging configurations
        lo = self.model.joint_limit_lower.numpy()[: self.n_coords].copy()
        hi = self.model.joint_limit_upper.numpy()[: self.n_coords].copy()
        for j, r in IK_YAW_RANGE.items():
            lo[j], hi[j] = max(lo[j], -r), min(hi[j], r)
        lo = wp.array(lo[None], dtype=float, device=self.device)
        hi = wp.array(hi[None], dtype=float, device=self.device)
        self.ik_solver = ik.IKSolver(
            model=self.ik_model, n_problems=1,
            objectives=[ik.IKObjectivePosition(link_index=hand, link_offset=wp.vec3(0.0, 0.0, 0.107), target_positions=self.ik_pos),
                        ik.IKObjectiveRotation(link_index=hand, link_offset_rotation=wp.quat_identity(), target_rotations=self.ik_rot),
                        ik.IKObjectiveJointLimit(joint_limit_lower=lo.flatten(), joint_limit_upper=hi.flatten(), weight=10.0)],
            lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def lay_out_rope(self):
        """The rope's segments moved onto a random curve (positions and orientations; VBD integrates the rod's bodies
        directly, so the state is the start and the straight rest shape stays)."""
        nodes = random_curve(self.rng, self.n_seg, self.seg_len, (self.rng.uniform(0.45, 0.55), self.rng.uniform(-0.08, 0.08)))
        mid = (nodes[1:] + nodes[:-1]) / 2
        head = np.arctan2(*(nodes[1:] - nodes[:-1]).T[::-1])
        for st in (self.state_0, self.state_1):
            q = st.body_q.numpy()
            for i, body in enumerate(self.rope_bodies):
                q0 = q[body, 3:7].copy()  # the straight rope's segment, along +x
                q[body, :3] = (mid[i, 0], mid[i, 1], SURFACE_Z + self.radius + 0.001)
                q[body, 3:7] = quat_mul(yaw_quat(head[i]), q0)
            st.body_q.assign(q)
            st.body_qd.zero_()

    # --- the policy
    def segments(self):
        return self.state_0.body_q.numpy()[self.rope_bodies, :3]

    def matched_goal(self, X):
        """The goal matched to the rope end to end, the closer way round."""
        e1 = np.linalg.norm(X[:, :2] - self.goal, axis=1)
        e2 = np.linalg.norm(X[:, :2] - self.goal[::-1], axis=1)
        return (self.goal, e1) if e1.mean() <= e2.mean() else (self.goal[::-1], e2)

    def choose(self, X, G, err):
        """DeformableRavens' oracle: the segment farthest from its place, set down there. -> (segment, (x, y), info)"""
        k = int(np.argmax(err))
        return k, G[k], {}

    def plan(self):
        """The next pick-and-place (self.choose's), appended to the keyframes; None when done."""
        a = self.args
        X = self.segments()
        G, err = self.matched_goal(X)
        if (err < TOL).all() or len(self.actions) >= a.max_actions:
            return None
        k, (xq, yq), info = self.choose(X, G, err)
        # where the segment goes: carried to the path's first point, then (a trace) dragged along the rest low over
        # the table, the rope behind it following; set down at the last
        path = np.asarray(info.get("path", [[xq, yq]]), float)
        nb = X[min(k + 1, self.n_seg - 1), :2] - X[max(k - 1, 0), :2]
        q_pick, yaw = gripper_across(np.arctan2(nb[1], nb[0]))
        if len(path) == 1:
            tangents = [G[min(k + 1, self.n_seg - 1)] - G[max(k - 1, 0)]]
        else:
            d = np.diff(path, axis=0)
            tangents = [*d, d[-1]]
        quats, y = [], yaw
        for t in tangents:  # the fingers across the rope there, turned the short way from the last
            q, yn = gripper_across(np.arctan2(t[1], t[0]))
            if abs(yn - y) > np.pi / 2:
                yn += np.pi if yn < y else -np.pi
                q = quat_mul(yaw_quat(yn), GRIPPER_DOWN)
            quats.append(q)
            y = yn
        xp, yp = X[k, :2]
        zg = SURFACE_Z + self.radius + a.grip_offset  # the TCP at the rope's centreline
        zu = zg + a.lift
        zd = zg + a.drag_lift
        zh = SURFACE_Z + 0.15
        dist = float(np.hypot(*(path[0] - X[k, :2])))
        keys = [[1.2, xp, yp, zh, *q_pick, GRIP_OPEN],  # over the segment
                [0.6, xp, yp, zg, *q_pick, GRIP_OPEN],  # down
                [0.6, xp, yp, zg, *q_pick, GRIP_CLOSED],  # pinch
                [0.5, xp, yp, zu, *q_pick, GRIP_CLOSED],  # lift
                [max(0.6, dist / 0.15), *path[0], zu, *quats[0], GRIP_CLOSED]]  # carry
        names = ["approach", "descend", "close", "lift", "carry"]
        if len(path) > 1:
            keys.append([0.4, *path[0], zd, *quats[0], GRIP_CLOSED])  # down to the table
            names.append("touch")
            for i in range(1, len(path)):  # along the letter
                keys.append([max(0.3, float(np.hypot(*(path[i] - path[i - 1]))) / a.drag_speed), *path[i], zd, *quats[i],
                             GRIP_CLOSED])
                names.append(f"drag{i}")
        xq, yq = path[-1]
        keys += [[0.4, xq, yq, zg, *quats[-1], GRIP_CLOSED],  # set down
                 [0.5, xq, yq, zg, *quats[-1], GRIP_OPEN],  # let go
                 [0.5, xq, yq, zh, *quats[-1], GRIP_OPEN]]  # away
        names += ["lower", "open", "retreat"]
        t0 = self.key_times[-1]
        times = t0 + np.cumsum([kk[0] for kk in keys])
        self.actions.append({"segment": k, "pick": X[k].tolist(), "place": [float(xq), float(yq)], "error_before_m": float(err[k]),
                             "max_error_before_m": float(err.max()), "fraction_before": float((err < TOL).mean()),
                             "phases": dict(zip(names, times.tolist())), "carried": [], "start": t0,
                             "rope_before_m": X[:, :2].round(4).tolist(), **info})
        self.keys += keys
        self.key_times += times.tolist()
        return k

    def update_ik_targets(self):
        t = self.sim_time
        i = int(np.searchsorted(self.key_times, t))
        i = min(i, len(self.keys) - 1)
        t_start = self.key_times[i - 1] if i > 0 else 0.0
        alpha = float(np.clip((t - t_start) / max(self.key_times[i] - t_start, 1e-6), 0.0, 1.0))
        cur = np.asarray(self.keys[i][1:], float)
        prev = np.asarray(self.keys[i - 1][1:], float) if i > 0 else cur
        x = (1.0 - alpha) * prev + alpha * cur
        q = x[3:7] / np.linalg.norm(x[3:7])
        wp.launch(set_task_targets, dim=1, inputs=[self.ik_pos, self.ik_rot, self.finger_pos, wp.vec3(*x[:3].tolist()),
                                                     wp.vec4(*q.tolist()), float(x[-1])], device=self.device)

    # --- stepping
    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=24)
        wp.launch(set_gripper_q, dim=1, inputs=[self.ik_joint_q, self.finger_pos, self.n_coords - 2, self.n_coords - 1], device=self.device)
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def tcp(self):
        bq = self.state_0.body_q.numpy()
        hand = next(b for b in self.franka_bodies if self.model.body_label[b].endswith("fr3_hand"))
        p, q = bq[hand, :3], bq[hand, 3:7]
        u = np.array(q[:3])
        v = np.array([0.0, 0.0, 0.107])
        return p + v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)

    def step(self, frame):
        if not self.done and self.sim_time >= self.key_times[-1] - 1e-9:
            if self.actions:
                self.finish_action()
            if self.plan() is None:  # done: the hand back up out of the way
                self.done = True
                self.t_done = self.sim_time
                self.keys.append([1.5, *self.rest_tcp, *GRIPPER_DOWN, GRIP_OPEN])
                self.key_times.append(self.sim_time + 1.5)
        self.update_ik_targets()
        t0 = time.perf_counter()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize()
        self.wall += time.perf_counter() - t0
        self.sim_time += self.frame_dt
        if frame % 6 == 0:
            X = self.segments()
            _, err = self.matched_goal(X)
            self.log["t"].append(self.sim_time)
            self.log["fraction_in_place"].append(float((err < TOL).mean()))
            self.log["max_error_m"].append(float(err.max()))
            if self.actions and not self.done:
                act = self.actions[-1]
                ph = act["phases"]
                if ph["lift"] <= self.sim_time <= ph["lower"]:  # carried: the taken segment's distance from the TCP
                    act["carried"].append(float(np.linalg.norm(X[act["segment"]] - self.tcp())))
                if ph["lower"] <= self.sim_time <= ph["open"]:  # where the hand let go (vs where it was sent)
                    act["tcp_release_m"] = self.tcp()[:2].round(4).tolist()
        if frame % 60 == 0:
            n = len(self.actions)
            print(f"t {self.sim_time:5.1f} s wall {self.wall:6.1f} s action {n} in place {self.log['fraction_in_place'][-1]:.2f} "
                  f"max error {100 * self.log['max_error_m'][-1]:.1f} cm", flush=True)

    def finish_action(self):
        act = self.actions[-1]
        X = self.segments()
        _, err = self.matched_goal(X)
        c = np.asarray(act["carried"]) if act["carried"] else np.array([np.inf])
        act.update({"held_through_carry": bool(c.max() < 0.03), "carried_distance_from_tcp_max_m": float(c.max()),
                    "released": bool(X[act["segment"], 2] - SURFACE_Z < 3 * self.radius),
                    "error_after_m": float(err[act["segment"]]), "fraction_after": float((err < TOL).mean()),
                    "max_error_after_m": float(err.max()), "errors_after_m": err.round(4).tolist(),
                    "rope_after_m": X[:, :2].round(4).tolist()})
        del act["carried"]
        print(f"[action {len(self.actions)}] segment {act['segment']}: held {act['held_through_carry']} "
              f"({100 * act['carried_distance_from_tcp_max_m']:.1f} cm from the TCP), released {act['released']}, in place "
              f"{act['fraction_before']:.2f} -> {act['fraction_after']:.2f}, max error {100 * act['max_error_before_m']:.1f} -> "
              f"{100 * act['max_error_after_m']:.1f} cm", flush=True)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if hasattr(self.viewer, "log_points"):  # the goal line, as points
            g = np.concatenate([self.goal, np.full((self.n_seg, 1), SURFACE_Z + 0.0005)], 1).astype(np.float32)
            self.viewer.log_points("/goal", wp.array(g, dtype=wp.vec3), 0.003, wp.array([wp.vec3(0.9, 0.3, 0.2)] * self.n_seg, dtype=wp.vec3))
        self.viewer.end_frame()

    def report(self):
        X = self.segments()
        _, err = self.matched_goal(X)
        a = self.args
        return {"seed": a.seed, "length_m": a.length, "radius_m": self.radius, "segments": self.n_seg, "density": a.density,
                "stretch": a.stretch, "bend": a.bend, "mu": a.mu, "grip_force": a.grip_force, "lift_m": a.lift,
                "goal": self.goal.tolist(), "actions": self.actions, "n_actions": len(self.actions),
                "fraction_in_place": float((err < TOL).mean()), "max_error_m": float(err.max()), "mean_error_m": float(err.mean()),
                "success": bool((err < TOL).all()), "all_held_and_released": all(x.get("held_through_carry") and x.get("released") for x in self.actions),
                "sim_seconds": self.sim_time, "wall_seconds": self.wall, "realtime_factor": self.sim_time / max(self.wall, 1e-9),
                "log": self.log}
