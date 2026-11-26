"""Generic robot in Isaac Sim, driven by a spec (calibrated_scene_reconstruction.robots.spec): stage setup, home pose, TCP pose, DLS IK, end
effector closing, cameras. `Robot(world, load_spec("vega_1p_sharpa"))` reproduces the former Vega wrapper exactly."""
import re
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.core.paths import robot_asset_path
from .usd import add_camera, set_pose
from calibrated_scene_reconstruction.robots.urdf import mimics

ROBOT_PATH = "/World/Robot"
_JOINTS = {UsdPhysics.RevoluteJoint: "angular", UsdPhysics.PrismaticJoint: "linear"}


def _joint_kind(prim):
    for cls, kind in _JOINTS.items():
        if prim.IsA(cls):
            return kind
    return None


def _drive_rule(spec, name):
    for rule in spec["drives"]:
        if re.search(rule["match"], name):
            return rule
    return None


def _ee_collisions(stage, spec, path):
    """End-effector links whose spec asks for a finer collision approximation (`ee.collision`, e.g.
    "convexDecomposition"): the importer's one convex hull per link fills the notch of an L-shaped finger (Galaxea R1:
    near the palm the hull's inner face reaches 3.7 cm from the closing axis and lands on a 7.6 cm cup's rim)."""
    for arm in spec["arms"].values():
        approx = arm["ee"].get("collision")
        if not approx:
            continue
        links = [UsdPhysics.Joint(p).GetBody1Rel().GetTargets()[0] for p in Usd.PrimRange(stage.GetPrimAtPath(path))
                 if _joint_kind(p) and re.search(arm["ee"]["joints"], p.GetName())]
        for link in links:
            for c in stage.GetPrimAtPath(link).GetChildren():
                if c.GetName() != "collisions":
                    continue
                c.SetInstanceable(False)  # the importer instances link geometry: edits need a real prim
                for q in Usd.PrimRange(c):
                    if q.HasAPI(UsdPhysics.MeshCollisionAPI):
                        UsdPhysics.MeshCollisionAPI(q).GetApproximationAttr().Set(approx)
                        if approx == "convexDecomposition":
                            cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(q)
                            cd.CreateMaxConvexHullsAttr(16)
                            cd.CreateHullVertexLimitAttr(32)
                            cd.CreateVoxelResolutionAttr(300000)


def _ee_links(stage, spec_ee, path):
    """Every link an end effector's joints connect (the hand's base link and its finger links), plus the links fixed
    below them: the Sharpa fingertip elastomers, the pads that touch the object, hang off the distal links by fixed
    joints. Fixed joints are followed from parent (body0) to child (body1) only, never up into the arm."""
    links, fixed = set(), []
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = [str(t) for t in j.GetBody0Rel().GetTargets()], [str(t) for t in j.GetBody1Rel().GetTargets()]
        if _joint_kind(p) and re.search(spec_ee["joints"], p.GetName()):
            links.update(b0 + b1)
        elif p.IsA(UsdPhysics.FixedJoint) and b0 and b1:
            fixed.append((b0[0], b1[0]))
    grew = True
    while grew:
        grew = False
        for parent, child in fixed:
            if parent in links and child not in links:
                links.add(child)
                grew = True
    return sorted(links)


def ee_contact_materials(stage, spec, path=ROBOT_PATH):
    """Bind an end effector's declared contact material (`ee.contact_material`) to every collider of its links.

    Without one, the hand's colliders carry no physics material and PhysX takes its default (0.5 / 0.5, combined by
    averaging with the object's), whatever the hand's skin. Returns {side: number of colliders bound}."""
    from .physics import physics_material

    bound = {}
    for side, arm in spec["arms"].items():
        m = arm.get("ee", {}).get("contact_material")
        if not m:
            continue
        mat = physics_material(stage, f"{path}/ContactMaterials/{side}_ee", m["static_friction"], m["dynamic_friction"],
                               m.get("restitution", 0.0), friction_combine=m.get("combine"))
        n = 0
        for link in _ee_links(stage, arm["ee"], path):
            for c in stage.GetPrimAtPath(link).GetChildren():
                if c.GetName() != "collisions":
                    continue
                c.SetInstanceable(False)  # the importer instances link geometry: a binding needs a real prim
                for q in Usd.PrimRange(c):
                    if q.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                        n += 1
        if n == 0:
            raise ValueError(f"arms.{side}.ee.contact_material: no collider found under the links its joints connect")
        bound[side] = n
    return bound


def add_robot(stage, spec, base_xy, yaw, path=ROBOT_PATH, support_top=None):
    """Reference the robot USD, author drives / physics / reset state, and place the base."""
    usd = Path(spec["usd"])
    if spec.get("urdf"):
        if not usd.is_file():
            raise FileNotFoundError(f"Build {spec['name']} with apps/robot_build.py before using {usd}.")
    else:
        usd = robot_asset_path(usd, required=True)
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(usd))
    phys = spec["physics"]
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and phys.get("disable_gravity", True):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
        kind = _joint_kind(p)
        if kind:
            rule = _drive_rule(spec, p.GetName())
            if rule:
                drive = UsdPhysics.DriveAPI.Apply(p, kind)
                drive.CreateStiffnessAttr(float(rule["kp"]))
                drive.CreateDampingAttr(float(rule.get("kd", 0.1 * rule["kp"])))
                drive.CreateMaxForceAttr(float(rule["max_force"]))
            if p.GetName() in spec["home"]:  # reset state: the USD zero pose may cut through nearby furniture
                v = spec["home"][p.GetName()]
                v = float(np.degrees(v)) if kind == "angular" else float(v)
                PhysxSchema.JointStateAPI.Apply(p, kind).CreatePositionAttr(v)
                UsdPhysics.DriveAPI.Apply(p, kind).CreateTargetPositionAttr(v)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(p)
            art.CreateEnabledSelfCollisionsAttr(bool(phys.get("self_collision", False)))
            pos, vel = phys.get("solver_iters", (32, 4))
            art.CreateSolverPositionIterationCountAttr(int(pos))
            art.CreateSolverVelocityIterationCountAttr(int(vel))
    _ee_collisions(stage, spec, path)
    ee_contact_materials(stage, spec, path)
    place_robot(stage, spec, base_xy, yaw, path, support_top)
    return prim


def base_height(spec, support_top=None):
    """Pedestal top, or a fixed body's root height (legs on the floor). "support" mounts the base at the support's top
    surface (the task frame passes it; 0.75 m, a table, when unknown)."""
    h = spec["base"].get("height", 0.0)
    return float(support_top if support_top is not None else 0.75) if h == "support" else float(h)


def place_robot(stage, spec, base_xy, yaw, path=ROBOT_PATH, support_top=None):
    z = base_height(spec, support_top)
    c, s = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
    set_pose(stage.GetPrimAtPath(path), [base_xy[0], base_xy[1], z], (c, 0, 0, s))
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):  # the fixed root joint anchors the base: world side = base pose
        if p.GetName() == "root_joint" and p.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(p)
            if not joint.GetBody0Rel().GetTargets():
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(float(base_xy[0]), float(base_xy[1]), z))
                joint.GetLocalRot0Attr().Set(Gf.Quatf(c, 0.0, 0.0, s))
    if spec["base"].get("mount") == "pedestal" and z > 0:
        ped = UsdGeom.Cube.Define(stage, "/World/RobotPedestal" if path == ROBOT_PATH else f"{path}_pedestal")
        r = float(spec["base"].get("pedestal_radius", 0.12))
        ped.CreateSizeAttr(1.0)
        set_pose(ped.GetPrim(), [base_xy[0], base_xy[1], z / 2], (1, 0, 0, 0), scale=(2 * r, 2 * r, z))
        ped.CreateDisplayColorAttr([(0.25, 0.25, 0.27)])
        UsdPhysics.CollisionAPI.Apply(ped.GetPrim())


def add_cameras(stage, spec, path=ROBOT_PATH, size=None, far=100.0):
    """One camera prim per spec camera under its mount link; returns {name: (camera, W, H, K)}. `far`: the clipping
    distance (a batch copy's cameras stop short of the next copy)."""
    cams = {}
    for c in spec["cameras"]:
        w0, h0 = c["size"]
        w, h = (w0, h0) if c.get("native_size") or not size else size  # native_size: a camera kept at its own size
        K = np.asarray(c["K"], float) * np.array([[w / w0], [h / h0], [1]])
        cam = add_camera(stage, f"{path}/{c['link']}/{c['name']}_cam", K, w, h, far=far)
        set_pose(cam.GetPrim(), c.get("pos", [0, 0, 0]), tuple(c.get("quat_wxyz", (1, 0, 0, 0))))  # outside the shell
        cams[c["name"]] = (cam, w, h, K)
    return cams


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Robot:
    """Live articulation handle (create after world.reset())."""

    def __init__(self, world, spec, path=ROBOT_PATH):
        from isaacsim.core.prims import Articulation, SingleArticulation

        self.world, self.spec, self.path = world, spec, path
        self.robot = SingleArticulation(path, name=spec["name"])
        self.robot.initialize()
        self.view = Articulation(path)
        self.view.initialize()
        self.names = list(self.robot.dof_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.body_names = list(self.view.body_names)
        lim = self.view.get_dof_limits()
        lim = lim.cpu().numpy() if hasattr(lim, "cpu") else np.asarray(lim)
        self.lower, self.upper = lim[0, :, 0], lim[0, :, 1]
        J = self._jac()
        self.fixed_base = J.shape[1] == len(self.body_names) - 1  # fixed base: no root row and no base columns
        built = Path(spec["usd"]).with_suffix(".urdf")
        self.mimic = {k: v for k, v in (mimics(built) if spec.get("urdf") and built.exists() else {}).items()
                      if k in self.idx and v[0] in self.idx}  # driven joints follow their master in with_ee()
        self._col0 = 0 if self.fixed_base else 6

    # --- joint space
    def home(self):
        q = np.zeros(len(self.names))
        for k, v in self.spec["home"].items():
            if k in self.idx:
                q[self.idx[k]] = v
        return q

    def q(self):
        return np.asarray(self.robot.get_joint_positions(), float)

    def ee_joints(self, side):
        rx = re.compile(self.spec["arms"][side]["ee"]["joints"])
        return [n for n in self.names if rx.search(n)]

    def with_ee(self, q, close, side=None):
        """Joint vector with the end effector of `side` closed by `close` (0 open, 1 the spec's closed pose; >1 squeezes
        further for frac_upper hands)."""
        side = side or next(iter(self.spec["arms"]))
        ee = self.spec["arms"][side]["ee"]
        cl = ee["close"]
        q = np.asarray(q, float).copy()
        for n in self.ee_joints(side):
            i = self.idx[n]
            if cl["mode"] == "frac_upper":
                f = next((v for k, v in cl["fraction"].items() if n.endswith(k)), 0.0)
                v = min(f * close, cl.get("cap", 0.95)) * self.upper[i]
            else:
                o, c = cl["open"].get(n, 0.0), cl["closed"].get(n, cl["open"].get(n, 0.0))
                v = o + min(close, cl.get("cap", 1.0)) * (c - o)
            q[i] = float(np.clip(v, self.lower[i], self.upper[i]))
        for n, (master, mult, off) in self.mimic.items():
            i = self.idx[n]
            q[i] = float(np.clip(q[self.idx[master]] * mult + off, self.lower[i], self.upper[i]))
        return q

    def teleport(self, q):
        from isaacsim.core.utils.types import ArticulationAction

        q = np.asarray(q, float)
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros(len(q)))
        self.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))

    def sync(self):
        """Link transforms and Jacobians of the current joint state without a physics step (IK / FK after a teleport);
        a physics step where the tensor API cannot update kinematics on its own."""
        try:
            self.world.physics_sim_view.update_articulations_kinematic()
        except Exception:  # noqa: BLE001
            self.world.step(render=False)

    def command(self, q, qd=None):
        """Position targets, optionally with velocity targets (feed-forward: a PD drive tracking a moving target
        without them lags by damping / stiffness seconds)."""
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.get_articulation_controller().apply_action(ArticulationAction(
            joint_positions=np.asarray(q, float), joint_velocities=None if qd is None else np.asarray(qd, float)))

    # --- Cartesian
    def link_transforms(self):
        """(links, 7) world poses of every link at the current state: position and quaternion xyzw (PhysX)."""
        tf = self.view._physics_view.get_link_transforms()
        return (tf.cpu().numpy() if hasattr(tf, "cpu") else np.asarray(tf))[0]

    def link_pose(self, name):
        t = self.link_transforms()[self.body_names.index(name)]
        return t[:3], Rotation.from_quat(t[3:7]).as_matrix()  # PhysX quaternions are xyzw

    def tcp_pose(self, side=None):
        side = side or next(iter(self.spec["arms"]))
        arm = self.spec["arms"][side]
        p, R = self.link_pose(arm["tcp_link"])
        return p + R @ np.asarray(arm.get("tcp_offset", (0, 0, 0)), float), R

    def _jac(self):
        J = self.view.get_jacobians()
        return J.cpu().numpy() if hasattr(J, "cpu") else np.asarray(J)

    def jacobian(self, name, cols, offset_world=None):
        row = self.body_names.index(name) - (1 if self.fixed_base else 0)
        J = self._jac()[0, row][:, [self._col0 + self.idx[c] for c in cols]].copy()
        if offset_world is not None:  # velocity of a point rigidly attached to the link: v + w x r
            J[:3] -= _skew(offset_world) @ J[3:]
        return J

    def solve_ik(self, side, p_target, R_target, q0, cols=None, iters=250, tol=(0.004, 0.03), damping=0.05,
                 rot_weight=0.6, low_weight_scale=1.0):
        """Damped least-squares IK of the side's TCP on the live articulation (teleports joints while iterating).
        `low_weight_scale` scales the joints the spec already weights below 1 (a torso): < 1 moves them even less."""
        arm = self.spec["arms"][side]
        cols = cols or arm["ik_joints"]
        off = np.asarray(arm.get("tcp_offset", (0, 0, 0)), float)
        q = np.asarray(q0, float).copy()
        ci = [self.idx[c] for c in cols]
        # joint weights (spec arms.<side>.ik_weights {regex: w}): weighted DLS moves low-weight joints (a humanoid's
        # torso) less, so the arm does the reaching
        w = np.ones(len(cols))
        for rx, v in arm.get("ik_weights", {}).items():
            w[[i for i, c in enumerate(cols) if re.search(rx, c)]] = float(v) * (low_weight_scale if v < 1 else 1.0)
        lo, hi = self.lower[ci].copy(), self.upper[ci].copy()
        for j, c in enumerate(cols):  # spec arms.<side>.ik_limits {joint: [lo, hi]}: a working range within the limits
            if c in arm.get("ik_limits", {}):
                lo[j], hi[j] = max(lo[j], arm["ik_limits"][c][0]), min(hi[j], arm["ik_limits"][c][1])
        q[ci] = np.clip(q[ci], lo, hi)
        err = None
        for _ in range(iters):
            self.teleport(q)
            self.sync()
            p_l, R = self.link_pose(arm["tcp_link"])
            r = R @ off
            ep, er = p_target - (p_l + r), Rotation.from_matrix(R_target @ R.T).as_rotvec()
            err = (float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
            if err[0] < tol[0] and err[1] < tol[1]:
                break
            J = self.jacobian(arm["tcp_link"], cols, r if off.any() else None)
            J[3:] *= rot_weight
            JW = J * w
            dq = JW.T @ np.linalg.solve(JW @ J.T + damping ** 2 * np.eye(6), np.concatenate([ep, rot_weight * er]))
            q[ci] = np.clip(q[ci] + np.clip(dq, -0.15, 0.15), lo, hi)
        return q, err


def hand_links(stage, spec, side, path=ROBOT_PATH):
    """The TCP link, every link moved by the end effector's joints and everything those carry (fingertip pads, sensors
    on fixed joints): the hand proper, without the wrist."""
    arm = spec["arms"][side]
    rx = re.compile(arm["ee"]["joints"])
    links, kids = {arm["tcp_link"]}, {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = [t.name for t in j.GetBody0Rel().GetTargets()], [t.name for t in j.GetBody1Rel().GetTargets()]
        if _joint_kind(p) and rx.search(p.GetName()):
            links.update(b1)
        for a in b0:
            kids.setdefault(a, set()).update(b1)
    todo = [n for n in links if n != arm["tcp_link"]]
    while todo:
        for k in kids.get(todo.pop(), ()):
            if k not in links:
                links.add(k)
                todo.append(k)
    return links


def hand_points(rob, side, sweep, radius=0.25, links=None):
    """Convex-hull points of every link within `radius` of the TCP (hand and wrist; or only `links`), in the TCP frame,
    at the robot's current pose (sim.sweep's link hulls)."""
    p, R = rob.tcp_pose(side)
    out = []
    for name, L in sweep.local.items():
        if links is not None and name not in links:
            continue
        lp, lR = rob.link_pose(name)
        W = L @ lR.T + lp
        W = W[np.linalg.norm(W - p, axis=1) < radius]
        if len(W):
            out.append((W - p) @ R)
    return np.concatenate(out) if out else np.zeros((0, 3))


def clear_support(cands, pts_tcp, support_top, margin=0.005):
    """Raise grasp candidates whose hand would reach below the support's top (a large palm held thumb-up next to a
    short object): pre-grasp, grasp and lift move up together."""
    for c in cands:
        if not len(pts_tcp):
            break
        dz = support_top + margin - float((pts_tcp @ c["R"].T + c["grasp"])[:, 2].min())
        if dz > 0:
            for k in ("pre", "grasp", "lift"):
                c[k] = c[k] + np.array([0.0, 0.0, dz])
            c["raised_m"] = round(dz, 4)
    return cands
