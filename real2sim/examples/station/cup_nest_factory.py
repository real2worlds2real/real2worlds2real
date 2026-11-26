"""Station environment factory for a scene-pipeline twin: a frozen two-cup session.

A worked example of the factory a twin station needs (README.md in this directory).  It is written for this task; copy it and change the judge for another.

    build(session, task, num_envs, run_dir) -> [CupNestEnv]        (the twin station's protocol)

Task: "pick up the blue cup and put it into the white cup".  The scene is built read-only from the frozen session exactly
as apps/scene/task_check.py builds it (prepare_scene -> build_scene, the session's simulation.json profile, nominal
physical priors); the robot selected in the station setup stands at the scene graph's calibrated
photo_robot.base.  Nothing is written into the session; diagnostics go to the station run directory.

Interface: state_vector, privileged simulator state (simulation verification only, not a deployable observation).
  obs (8, float32): TCP - blue cup centre (3, m, world axes); blue cup lowest-point centre - white opening centre
                    (xy, m) and blue lowest z - white rim z (m); hand closure (0 open .. 1 closed); blue cup tilt / 90 deg
  action (4): TCP translation (3) scaled by STEP_M per control step, hand closure change scaled by STEP_CLOSE; each
              component clipped to [-1, 1]; zeros hold still.  The TCP orientation stays the home orientation.
Hidden judge (measure()["success"]): the task definition of the session's own agent/task_check.py (inserted >= 0.5 of the
white opening depth, lateral <= 0.5 of the opening radius, both cups tilted <= 5 deg, the white cup slid <= 0.01 m), and
the hand let go: closure <= 0.15 and the TCP >= 0.05 m from the blue cup's centre.  Resting is the station's --twin-hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TARGET, GOAL = "blue_cup", "white_cup"
TARGET_BODY, GOAL_BODY = TARGET + "/model", GOAL + "/model"
CONTROL_HZ = 20
STEP_M, STEP_CLOSE = 0.01, 0.05
MIN_INSERTED_FRACTION, MAX_LATERAL_FRACTION, MAX_TILT_DEG, MAX_GOAL_SLIDE_M = 0.5, 0.5, 5.0, 0.01   # agent/task_check.py
MAX_CLOSURE_RELEASED, MIN_TCP_DISTANCE_M = 0.15, 0.05
IK_OFFSET = np.array([0.0, 12.0, 0.0])
START_JITTER_RAD = 0.03


class CupNestEnv:
    def __init__(self, session: Path, task: dict, run_dir: Path):
        from calibrated_scene_reconstruction.sim.application import SimulationApp

        self.app = SimulationApp({"headless": True})
        import omni.usd
        from isaacsim.core.api import World
        from pxr import UsdGeom

        from calibrated_scene_reconstruction.robots.spec import load_spec
        from calibrated_scene_reconstruction.scene.session import Session
        from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene
        from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant
        from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot
        from calibrated_scene_reconstruction.sim.scene_stage import build_scene
        from calibrated_scene_reconstruction.sim.simulation import audit as audit_simulation
        from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx

        self.run_dir = Path(run_dir)
        self.s = Session(session)
        graph = self.s.scene_graph
        if graph["task"]["target"] != TARGET or graph["task"]["goal"] != GOAL:
            raise SystemExit("this factory is written for target blue_cup / goal white_cup")
        self.world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30, device="cpu", backend="numpy")
        self.stage = omni.usd.get_context().get_stage()
        runtime = prepare_scene(self.s)
        runtime.physics_case = "nominal"
        self.ts = build_scene(self.stage, runtime, photo_goal(self.s, prepare_variant(self.s, {})))
        self.profile = audit_simulation(self.stage, runtime.simulation)
        robot = (task.get("twin", {}).get("setup") or {}).get("robot", "vega_1p_sharpa_arm")
        self.spec = load_spec(robot)
        self.side = next(iter(self.spec["arms"]))
        base = graph["photo_robot"]["base"]
        top = float(graph["furniture"][graph["task"]["support"]]["box"][1][2])
        xy, yaw = np.asarray(base["xy"], float), np.radians(float(base["yaw_deg"]))
        add_robot(self.stage, self.spec, xy, yaw, support_top=top)
        add_robot(self.stage, self.spec, xy + IK_OFFSET[:2], yaw, path="/World/PolicyIKRobot", support_top=top)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/PolicyIKRobot")).MakeInvisible()
        self.world.reset()
        self.robot = Robot(self.world, self.spec)
        self.ik = Robot(self.world, self.spec, path="/World/PolicyIKRobot")
        self.ctx = Ctx(self.s, self.ts, self.world, self.stage, None)
        dt = float(self.world.get_physics_dt())
        ratio = 1.0 / (dt * CONTROL_HZ)
        if abs(ratio - round(ratio)) > 1e-6 or ratio < 1:
            raise SystemExit(f"control rate {CONTROL_HZ} Hz does not divide the physics rate {1 / dt:.1f} Hz")
        self.ticks = int(round(ratio))
        self.home = np.clip(self.robot.home(), self.robot.lower, self.robot.upper)
        self.robot.teleport(self.home)
        self.ik.teleport(self.home)
        for _ in range(int(round(0.5 / dt))):
            self.world.step(render=False)
        self.source = {n: self.ctx.pose(n) for n in (TARGET_BODY, GOAL_BODY)}       # the captured poses, at rest
        self.opening0 = self.ctx.opening(GOAL, link="model")
        self.goal_p0, self.goal_R0 = self.source[GOAL_BODY]
        self.target_R0 = self.source[TARGET_BODY][1]
        _, self.R_tcp = self.robot.tcp_pose(self.side)
        self.q = self.home.copy()
        self.close = 0.0
        self.arm_cols = [self.robot.idx[c] for c in self.spec["arms"][self.side]["ik_joints"]]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "twin_factory.json").write_text(json.dumps({
            "session": str(self.s.dir), "robot": robot, "base": base, "physics_dt": dt, "physics_steps_per_control": self.ticks,
            "simulation_profile": self.profile, "opening": self.opening0,
            "source_poses": {n: {"p": np.asarray(p).tolist(), "R": np.asarray(R).tolist()} for n, (p, R) in self.source.items()},
            "judge_selfcheck": self._judge_selfcheck()}, indent=1, default=str) + "\n")

    # ------------------------------------------------------------------ protocol
    def info(self):
        return {"text": "pick up the blue cup and put it into the white cup", "action_dim": 4, "interface": "state_vector",
                "hold": "zero", "kinematics": False, "proprio": ["obs"],
                "observation_layout": {"tcp_minus_blue_centre_m": [0, 3], "blue_bottom_minus_white_opening_m": [3, 6],
                                       "hand_closure": [6, 7], "blue_tilt_over_90deg": [7, 8]},
                "privileged_state": True}

    def _restore_objects(self):
        for name, (p, R) in self.source.items():
            self.ctx.place(name, p, R)

    def reset(self, case):
        rng = np.random.default_rng(None if case is None else int(case) if not isinstance(case, dict) else int(case.get("seed", 0)))
        q = self.home.copy()
        q[self.arm_cols] += rng.uniform(-START_JITTER_RAD, START_JITTER_RAD, len(self.arm_cols))
        q = np.clip(q, self.robot.lower, self.robot.upper)
        self.close = 0.0
        self.q = self.robot.with_ee(q, 0.0)
        self.robot.teleport(self.q)          # episode initialization only
        self.robot.sync()
        self.robot.command(self.q)
        self._restore_objects()
        for _ in range(self.ticks * 4):
            self.world.step(render=False)
        _, self.R_tcp = self.robot.tcp_pose(self.side)

    def _state(self):
        points = self.ctx.points(TARGET_BODY)
        opening = self.ctx.opening(GOAL, link="model")
        low = float(points[:, 2].min())
        centre_xy = (points.min(0)[:2] + points.max(0)[:2]) / 2
        centre = (points.min(0) + points.max(0)) / 2
        tcp, _ = self.robot.tcp_pose(self.side)
        return points, opening, low, centre_xy, centre, np.asarray(tcp, float)

    def observe(self):
        _, opening, low, centre_xy, centre, tcp = self._state()
        tilt = self.ctx.tilt(TARGET_BODY, self.target_R0)
        obs = np.r_[tcp - centre, centre_xy - np.asarray(opening["xy"]), low - float(opening["rim_z"]), self.close, tilt / 90.0]
        return {"obs": obs.astype(np.float32)}

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(-1)[:4], -1.0, 1.0)
        tcp, _ = self.robot.tcp_pose(self.side)
        self.close = float(np.clip(self.close + STEP_CLOSE * a[3], 0.0, 1.0))
        if np.any(a[:3] != 0.0):
            q, _ = self.ik.solve_ik(self.side, np.asarray(tcp, float) + STEP_M * a[:3] + IK_OFFSET, self.R_tcp, self.robot.q(),
                                    iters=40, tol=(0.0005, 0.005))
            self.q = q
        self.q = self.robot.with_ee(self.q, self.close)
        self.robot.command(self.q)
        for _ in range(self.ticks):
            self.world.step(render=False)

    def measure(self):
        _, opening, low, _, centre, tcp = self._state()
        p_target, _ = self.ctx.pose(TARGET_BODY)
        p_goal, R_goal = self.ctx.pose(GOAL_BODY)
        axis = R_goal @ (self.goal_R0.T @ np.array([0.0, 0.0, 1.0]))
        offset = p_target - p_goal
        m = {"inserted_m": float(opening["rim_z"]) - low,
             "lateral_m": float(np.linalg.norm(offset - axis * float(offset @ axis))),
             "tilt_target_deg": self.ctx.tilt(TARGET_BODY, self.target_R0), "tilt_goal_deg": self.ctx.tilt(GOAL_BODY, self.goal_R0),
             "goal_slide_m": float(np.linalg.norm((p_goal - self.goal_p0)[:2])),
             "hand_closure": float(self.close), "tcp_distance_m": float(np.linalg.norm(tcp - centre))}
        checks = {"inserted": m["inserted_m"] >= MIN_INSERTED_FRACTION * float(self.opening0["depth"]),
                  "centred": m["lateral_m"] <= MAX_LATERAL_FRACTION * float(self.opening0["radius"]),
                  "upright": m["tilt_target_deg"] <= MAX_TILT_DEG and m["tilt_goal_deg"] <= MAX_TILT_DEG,
                  "goal_stays": m["goal_slide_m"] <= MAX_GOAL_SLIDE_M,
                  "released": m["hand_closure"] <= MAX_CLOSURE_RELEASED and m["tcp_distance_m"] >= MIN_TCP_DISTANCE_M}
        failed = "target_off_support" if low < float(self.goal_p0[2]) - 0.05 else None
        return {**m, "checks": checks, "success": bool(all(checks.values())), "failed": failed}

    # ------------------------------------------------------------------ qualification of the judge on this twin
    def _judge_selfcheck(self):
        """Once at start-up, outside any episode: the judge on the untouched start (must be False) and on the blue cup
        released 4 mm over the white rim under gravity for 4 s, as the session's task check does (must be True)."""
        self.reset(0)
        before = self.measure()
        self.ctx.hold_over(TARGET_BODY, np.asarray(self.opening0["xy"], float), float(self.opening0["rim_z"]) + 0.004, self.target_R0)
        self.ctx.step(4.0)
        after = self.measure()
        self.reset(0)
        return {"start": {k: before[k] for k in ("success", "checks", "inserted_m", "tcp_distance_m")},
                "nested_by_gravity": {k: after[k] for k in ("success", "checks", "inserted_m", "lateral_m", "tilt_target_deg",
                                                              "goal_slide_m", "tcp_distance_m")}}


def build(session, task, num_envs, run_dir):
    if int(num_envs) != 1:
        raise SystemExit("this twin factory builds one environment (one PhysX scene); start the station with num_envs 1")
    return [CupNestEnv(Path(session), task, Path(run_dir))]
