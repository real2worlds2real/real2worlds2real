"""Station environment factory for a scene delivered by the task-workspace route: lift one rigid participant.

A worked example of the factory a twin station needs (README.md in this directory), for the route whose delivery record
names a composed scene (`real2sim-harness deliver --run <run>`, `route: task_workspace`, `level: scene`).

    build(source, task, num_envs, run_dir) -> [LiftEnv]        (the twin station's protocol)

What the route delivers and what it does not.  The record pins `scene.usda`: the accepted participants under
/World/Objects/<id> with their own rigid bodies, colliders and materials, static support boxes under
/World/Environment/<id>, visual backgrounds.  It contains no physics scene and no robot, and the record states neither
metric scale nor a robot placement (`statements.metric_scale` and `statements.robot_placed` are null).  The robot
interaction setup is therefore this station's own declaration, written by its author and carried in the task spec as
`twin.setup` (`python -m astra_station new-twin-task --delivery <record> --setup <file.json>`):

    {"text": "<the task in words>", "kind": "lift", "target": "<object id>", "support": "<support id>",
     "robot": "<robot spec name or file>", "base": {"xy": [x, y], "yaw_deg": yaw},
     "lift_m": 0.08, "max_tilt_deg": 20, "hold_within_m": 0.12}

The delivered file is referenced, never opened for writing; diagnostics go to the station run directory.

Interface: state_vector, privileged simulator state (simulation verification only, not a deployable observation).
  obs (8, float32): TCP - target centre (3, m, world axes); target lowest point - support top (1, m); target centre
                    displacement from its start (xy, m); hand closure (0 open .. 1 closed); target tilt / 90 deg
  action (4): TCP translation (3) scaled by STEP_M per control step, hand closure change scaled by STEP_CLOSE; each
              component clipped to [-1, 1]; zeros hold still.  The TCP orientation stays the home orientation.
Hidden judge (measure()["success"]): the target's lowest point is at least `lift_m` above the support's top, it is
tilted at most `max_tilt_deg` from its start, and the hand carries it: closure >= 0.3 and the TCP within
`hold_within_m` of the target's centre.  How long it must hold is the station's --twin-hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONTROL_HZ, PHYSICS_HZ = 20, 120
STEP_M, STEP_CLOSE = 0.01, 0.05
MIN_CLOSURE_HOLDING = 0.3
IK_OFFSET = np.array([0.0, 12.0, 0.0])      # a second, invisible copy of the robot solves IK away from the scene
START_JITTER_RAD = 0.03
TWIN = "/World/Twin"
DEFAULTS = {"kind": "lift", "lift_m": 0.08, "max_tilt_deg": 20.0, "hold_within_m": 0.12}


def read_setup(task: dict) -> tuple[dict, dict]:
    """The delivery record and the station author's setup for it (no simulator needed)."""
    twin = task.get("twin") or {}
    if twin.get("route") != "task_workspace" or not twin.get("delivery"):
        raise SystemExit("this factory serves a task-workspace delivery (twin.route, twin.delivery in the task spec)")
    record = json.loads(Path(twin["delivery"]).read_text())
    if record.get("level") != "scene" or not record.get("scene"):
        raise SystemExit("the delivery record names no scene")
    setup = {**DEFAULTS, **(twin.get("setup") or {})}
    missing = [key for key in ("target", "support", "robot", "base") if not setup.get(key)]
    if missing:
        raise SystemExit(f"twin.setup lacks {missing}: the route delivers no robot interaction setup, the station declares it")
    if setup["kind"] != "lift":
        raise SystemExit("this factory judges a lift")
    if setup["target"] not in (record.get("task", {}).get("objects") or {}):
        raise SystemExit(f"{setup['target']!r} is not a participant of the delivered scene")
    return record, setup


def rotation(quat_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def judge(state: dict, setup: dict) -> dict:
    """The lift criterion on a measured state (plain numbers; shared by the environment and its tests)."""
    checks = {"lifted": state["height_m"] >= float(setup["lift_m"]),
              "upright": state["tilt_deg"] <= float(setup["max_tilt_deg"]),
              "held": state["hand_closure"] >= MIN_CLOSURE_HOLDING and state["tcp_distance_m"] <= float(setup["hold_within_m"])}
    failed = "target_off_support" if state["height_m"] < -0.05 else None
    return {**state, "checks": checks, "success": bool(all(checks.values())), "failed": failed}


class LiftEnv:
    def __init__(self, source: Path, task: dict, run_dir: Path):
        self.record, self.setup = read_setup(task)
        scene = Path(self.record["scene"]["path"])

        from calibrated_scene_reconstruction.sim.application import SimulationApp

        self.app = SimulationApp({"headless": True})
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Usd, UsdGeom, UsdPhysics

        from calibrated_scene_reconstruction.robots.spec import load_spec
        from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot

        self.run_dir = Path(run_dir)
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / 30, device="cpu",
                           backend="numpy")                      # the world brings the physics scene and gravity
        self.stage = omni.usd.get_context().get_stage()
        self.stage.DefinePrim(TWIN, "Xform").GetReferences().AddReference(str(scene))

        target_root = self.stage.GetPrimAtPath(f"{TWIN}/Objects/{self.setup['target']}")
        support = self.stage.GetPrimAtPath(f"{TWIN}/Environment/{self.setup['support']}")
        if not target_root.IsValid() or not support.IsValid():
            raise SystemExit("the delivered scene has no such target or support")
        bodies = [p for p in Usd.PrimRange(target_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise SystemExit(f"the target must be one rigid body; found {len(bodies)}")
        # composed supports are invisible collision boxes: the bound must not skip them
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                           UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                                  useExtentsHint=False, ignoreVisibility=True)
        bound = cache.ComputeWorldBound(support).ComputeAlignedRange()
        if bound.IsEmpty():
            raise SystemExit("the support has no geometry to stand on")
        self.support_top = float(bound.GetMax()[2])
        # the target's surface points in its body frame, once: the judge follows them through the body's pose
        xforms = UsdGeom.XformCache()
        points = []
        for prim in Usd.PrimRange(bodies[0]):
            if prim.IsA(UsdGeom.Mesh) and (values := UsdGeom.Mesh(prim).GetPointsAttr().Get()):
                local, _ = xforms.ComputeRelativeTransform(prim, bodies[0])
                points += [local.Transform(point) for point in values]
        if not points:
            raise SystemExit("the target body carries no mesh points")
        self.local_points = np.asarray(points, float)

        self.spec = load_spec(self.setup["robot"])
        self.side = next(iter(self.spec["arms"]))
        xy, yaw = np.asarray(self.setup["base"]["xy"], float), np.radians(float(self.setup["base"]["yaw_deg"]))
        add_robot(self.stage, self.spec, xy, yaw, support_top=self.support_top)
        add_robot(self.stage, self.spec, xy + IK_OFFSET[:2], yaw, path="/World/PolicyIKRobot", support_top=self.support_top)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/PolicyIKRobot")).MakeInvisible()
        self.world.reset()
        self.robot = Robot(self.world, self.spec)
        self.ik = Robot(self.world, self.spec, path="/World/PolicyIKRobot")
        self.body = SingleRigidPrim(str(bodies[0].GetPath()), reset_xform_properties=False)
        self.body.initialize()

        self.ticks = PHYSICS_HZ // CONTROL_HZ
        self.home = np.clip(self.robot.home(), self.robot.lower, self.robot.upper)
        self.robot.teleport(self.home)
        self.ik.teleport(self.home)
        self._settle(0.5)
        self.p0, self.quat0 = (np.asarray(v, float) for v in self.body.get_world_pose())     # the delivered pose, at rest
        self.R0 = rotation(self.quat0)
        _, self.R_tcp = self.robot.tcp_pose(self.side)
        self.q, self.close = self.home.copy(), 0.0
        self.arm_cols = [self.robot.idx[c] for c in self.spec["arms"][self.side]["ik_joints"]]
        self.centre0 = self._state()["centre"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        selfcheck = self._judge_selfcheck()
        (self.run_dir / "twin_factory.json").write_text(json.dumps({
            "delivery": str((task.get("twin") or {}).get("delivery")), "scene": str(scene),
            "scene_sha256": self.record["scene"]["sha256"], "setup": self.setup,
            "setup_source": "station author: the task-workspace route states no robot placement and no metric scale",
            "physics_hz": PHYSICS_HZ, "physics_steps_per_control": self.ticks, "support_top_m": self.support_top,
            "target_body": str(bodies[0].GetPath()), "target_points": len(self.local_points),
            "target_start": {"p": self.p0.tolist(), "quat_wxyz": self.quat0.tolist()},
            "judge_selfcheck": selfcheck}, indent=1, default=str) + "\n")
        if not selfcheck["ok"]:
            raise SystemExit(f"the judge failed its self-check on this scene; see {self.run_dir / 'twin_factory.json'}")

    def _settle(self, seconds: float):
        for _ in range(int(round(seconds * PHYSICS_HZ))):
            self.world.step(render=False)

    # ------------------------------------------------------------------ protocol
    def info(self):
        return {"text": self.setup.get("text") or self.record["task"]["text"], "action_dim": 4, "interface": "state_vector", "hold": "zero",
                "kinematics": False, "proprio": ["obs"],
                "observation_layout": {"tcp_minus_target_centre_m": [0, 3], "target_bottom_above_support_m": [3, 4],
                                       "target_displacement_xy_m": [4, 6], "hand_closure": [6, 7],
                                       "target_tilt_over_90deg": [7, 8]},
                "privileged_state": True}

    def _place_target(self, p, quat):
        self.body.set_world_pose(np.asarray(p, float), np.asarray(quat, float))
        self.body.set_linear_velocity(np.zeros(3))
        self.body.set_angular_velocity(np.zeros(3))

    def reset(self, case):
        seed = None if case is None else int(case.get("seed", 0)) if isinstance(case, dict) else int(case)
        rng = np.random.default_rng(seed)
        q = self.home.copy()
        q[self.arm_cols] += rng.uniform(-START_JITTER_RAD, START_JITTER_RAD, len(self.arm_cols))
        self.close = 0.0
        self.q = self.robot.with_ee(np.clip(q, self.robot.lower, self.robot.upper), 0.0)
        self.robot.teleport(self.q)          # episode initialization only
        self.robot.sync()
        self.robot.command(self.q)
        self._place_target(self.p0, self.quat0)
        for _ in range(self.ticks * 4):
            self.world.step(render=False)
        _, self.R_tcp = self.robot.tcp_pose(self.side)

    def _state(self):
        p, quat = self.body.get_world_pose()
        R = rotation(quat)
        points = self.local_points @ R.T + np.asarray(p, float)
        centre = (points.min(0) + points.max(0)) / 2
        tcp, _ = self.robot.tcp_pose(self.side)
        up = R @ (self.R0.T @ np.array([0.0, 0.0, 1.0]))
        return {"centre": centre, "tcp": np.asarray(tcp, float), "height_m": float(points[:, 2].min()) - self.support_top,
                "tilt_deg": float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))), "hand_closure": float(self.close),
                "tcp_distance_m": float(np.linalg.norm(np.asarray(tcp, float) - centre))}

    def observe(self):
        s = self._state()
        obs = np.r_[s["tcp"] - s["centre"], s["height_m"], (s["centre"] - self.centre0)[:2], self.close, s["tilt_deg"] / 90.0]
        return {"obs": obs.astype(np.float32)}

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(-1)[:4], -1.0, 1.0)
        tcp, _ = self.robot.tcp_pose(self.side)
        self.close = float(np.clip(self.close + STEP_CLOSE * a[3], 0.0, 1.0))
        if np.any(a[:3] != 0.0):
            self.q, _ = self.ik.solve_ik(self.side, np.asarray(tcp, float) + STEP_M * a[:3] + IK_OFFSET, self.R_tcp,
                                         self.robot.q(), iters=40, tol=(0.0005, 0.005))
        self.q = self.robot.with_ee(self.q, self.close)
        self.robot.command(self.q)
        for _ in range(self.ticks):
            self.world.step(render=False)

    def measure(self):
        s = self._state()
        return judge({k: s[k] for k in ("height_m", "tilt_deg", "hand_closure", "tcp_distance_m")}, self.setup)

    # ------------------------------------------------------------------ qualification of the judge on this scene
    def _judge_selfcheck(self):
        """Once at start-up, outside any episode.  The untouched start must not count; nor may the target merely being
        high: raised by hand above the lift height with the robot's hand open and away, 'lifted' turns true and
        'held' stays false.  A positive case needs a grasp, which is the policy's work, so none is fabricated here."""
        self.reset(0)
        start = self.measure()
        self._place_target(self.p0 + np.array([0.0, 0.0, float(self.setup["lift_m"]) + 0.02]), self.quat0)
        self.world.step(render=False)
        raised = self.measure()
        self.reset(0)
        keep = ("success", "checks", "height_m", "tilt_deg", "tcp_distance_m")
        ok = (not start["success"] and not start["checks"]["lifted"] and abs(start["height_m"]) < 0.01
              and raised["checks"]["lifted"] and not raised["checks"]["held"] and not raised["success"])
        return {"ok": bool(ok), "start": {k: start[k] for k in keep}, "raised_without_the_hand": {k: raised[k] for k in keep}}


def build(source, task, num_envs, run_dir):
    if int(num_envs) != 1:
        raise SystemExit("this twin factory builds one environment (one PhysX scene); start the station with num_envs 1")
    return [LiftEnv(Path(source), task, Path(run_dir))]
