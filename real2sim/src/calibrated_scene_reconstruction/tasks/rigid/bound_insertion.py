"""Source-geometry pregrasp insertion adapter for bound policy verification.

No object geometry, robot base, mass, friction or collider is replaced. The
explicit scope is terminal insertion from an initialized pregrasp fixture; the
fixture is removed before scoring. Scene selection, robot and object names,
initial-state distribution, controller and success thresholds are configuration
inputs pinned by the outer harness, not Peg constants in a policy worker.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def choose_action(observation, method, parameters, *, low, high, clearance, target_depth):
    """Cartesian servo or finite-horizon shooting under an explicit local model.

    MPC uses a nominal diagonal action-response model, not an asserted perfect
    simulator oracle. It penalizes approaching the rim while laterally outside
    clearance; true contact dynamics and outcomes are measured by PhysX.
    """
    position = np.asarray(observation, float)
    low, high = np.asarray(low, float), np.asarray(high, float)
    goal = np.array([0., 0., -target_depth])
    gain = float(parameters.get("gain", 0.7))
    direct = np.clip((goal - position) * gain, low, high)
    gate = clearance * float(parameters.get("alignment_fraction", 0.6))
    if np.linalg.norm(position[:2]) > gate:
        direct[2] = max(0., direct[2])
    if method == "servo":
        return direct
    if method != "mpc":
        raise ValueError("This adapter offers only servo and local-model MPC collection")
    response = float(parameters.get("response", 0.55))
    horizon = int(parameters.get("horizon", 3))
    if not 0 < gain <= 1 or not 0 < response <= 1 or not 1 <= horizon <= 8:
        raise ValueError("Invalid bounded local controller parameters")
    # Receding horizon: candidates are generated anew from the measured state.
    candidates = [np.zeros(3), direct]
    for fraction in (0.25, 0.5, 1.0):
        horizontal = np.clip(np.r_[-position[:2] * fraction, 0.], low, high)
        candidates.extend([horizontal, np.clip(horizontal + [0., 0., low[2] * fraction], low, high)])
    best, best_cost = candidates[0], float("inf")
    scale = max(clearance, 1e-6)
    for first in candidates:
        predicted, cost = position.copy(), 0.
        for tick in range(horizon):
            command = first if tick == 0 else np.clip((goal - predicted) * gain, low, high)
            if tick and np.linalg.norm(predicted[:2]) > gate:
                command[2] = max(0., command[2])
            predicted += response * command
            radial = np.linalg.norm(predicted[:2])
            cost += float(8 * (radial / scale)**2 + ((predicted[2] - goal[2]) / max(target_depth, scale))**2
                          + 0.001 * np.sum((command / np.maximum(np.abs(low), np.abs(high)))**2))
            if predicted[2] < 0 and radial > gate:
                cost += float(100 * (radial / scale)**2)
        if cost < best_cost:
            best, best_cost = first, cost
    return best


def _rotation(value):
    matrix = np.asarray(value, float)
    if matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1):
        raise ValueError("Explicit pregrasp orientations must be proper rotations")
    return matrix


def _quat_wxyz(matrix):
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(_rotation(matrix)).as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def _score(measurement, rules):
    # This runtime scorer is deliberately independent from artifact review's
    # implementation. Review recomputes each predicate from raw measurements.
    outcomes = []
    for rule in rules:
        value, target = measurement.get(rule["field"]), rule["target"]
        operation = rule["operator"]
        if operation == "eq":
            outcomes.append(type(value) is type(target) and value == target)
        elif type(value) not in (int, float) or not np.isfinite(value):
            outcomes.append(False)
        elif operation == "le":
            outcomes.append(value <= target)
        elif operation == "ge":
            outcomes.append(value >= target)
        elif operation == "abs_le":
            outcomes.append(abs(value) <= target)
        else:
            raise ValueError("Unsupported frozen physical success predicate")
    return bool(outcomes and all(outcomes))


class InsertionEnvironment:
    def __init__(self, binding, request, config, app, *, session, prepared, variant, diagnostic_directory):
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from calibrated_scene_reconstruction.core.paths import resolve
        from calibrated_scene_reconstruction.robots.spec import SPEC_DIR, load_spec
        from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot
        from calibrated_scene_reconstruction.sim.scene_stage import build_scene
        from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors
        from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx

        self.app, self.binding, self.config = app, binding, config
        self.native_errors = PhysicsErrors(Path(diagnostic_directory) / "native_physics_errors.json")
        self.settings = config["environment"]
        if self.settings.get("scope") != "pregrasp_insertion" or self.settings.get("fixture_release_before_success") is not True:
            raise ValueError("This adapter requires explicitly authorized pregrasp insertion with released scoring")
        self.session = session
        graph = self.session.scene_graph
        if graph["task"]["kind"] != "insert":
            raise ValueError("Insertion adapter cannot silently reinterpret another task")
        self.moving = graph["task"]["target"]
        self.receptacle = graph["task"]["goal"]
        spec_name = graph["photo_robot"]["spec"]
        spec_path = Path(spec_name)
        self.robot_spec_file = (spec_path if spec_path.suffix else SPEC_DIR / (spec_name + ".json")).resolve(strict=True)
        self.spec = load_spec(spec_name)
        initializer = self.settings.get("robot_initializer", "spec_home")
        self.source_joint_initializer = None
        if initializer == "source_visual_estimate":
            source_state = self.session.dir / "splat/robot_visual_fit.json"
            fitted = json.loads(source_state.read_text())
            joints = fitted.get("estimated_joints_rad", {})
            if not fitted.get("ok") or not joints or any(not np.isfinite(v) for v in joints.values()):
                raise ValueError("Requested source visual robot initializer is absent or invalid")
            self.spec["home"].update(joints)
            self.source_joint_initializer = {"source": str(source_state.resolve()), "sha256": _hash(source_state),
                                             "estimated_joints_rad": joints, "measurement_type": "visual estimate, not encoder readout"}
        elif initializer != "spec_home":
            raise ValueError("Unsupported disclosed robot initializer")
        self.side = self.settings.get("arm", next(iter(self.spec["arms"])))
        self.arm = self.spec["arms"][self.side]
        self.upright = _rotation(self.settings["object_rotation"])
        self.rotation = _rotation(self.settings["tcp_rotation"])
        self.long_axis = int(self.settings["object_long_axis"])
        self.tcp_from_origin = np.asarray(self.settings["tcp_from_object_origin_m"], float)
        if self.tcp_from_origin.shape != (3,) or not np.isfinite(self.tcp_from_origin).all():
            raise ValueError("Explicit pregrasp TCP offset must be a finite 3-vector in world axes")
        control_hz = float(binding["policy_io"]["control_hz"])
        physics_hz = float(self.settings["physics_hz"])
        ratio = physics_hz / control_hz
        if not np.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError("Control rate must divide native physics rate")
        self.ticks = int(round(ratio))
        self.settle_ticks = int(round(float(self.settings["release_settle_seconds"]) * physics_hz))
        if self.settle_ticks < 1:
            raise ValueError("Released outcome requires positive physical settling duration")
        if ({"field": "attachment_active", "operator": "eq", "target": False} not in config["success_rule"]
                or not any(rule["field"] == "released_ticks" and rule["operator"] == "ge"
                           and rule["target"] >= self.settle_ticks for rule in config["success_rule"])):
            raise ValueError("Frozen success must require detached object and the full physical settling interval")
        self.world = World(stage_units_in_meters=1., physics_dt=1 / physics_hz, rendering_dt=1 / control_hz)
        self.native_errors.guard_steps(self.world)
        self.stage = omni.usd.get_context().get_stage()
        self.scene = build_scene(self.stage, prepared, variant)
        base = graph["photo_robot"]["base"]
        self.base = {"xy": list(base["xy"]), "yaw_deg": base["yaw_deg"]}
        top = float(graph["dynamic_objects"][self.receptacle]["fix"]["support_z"])
        self.ik_offset = np.array([0., 12., 0.])
        add_robot(self.stage, self.spec, np.asarray(base["xy"]), np.radians(base["yaw_deg"]), support_top=top)
        add_robot(self.stage, self.spec, np.asarray(base["xy"]) + self.ik_offset[:2], np.radians(base["yaw_deg"]),
                  path="/World/PolicyIKRobot", support_top=top)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/PolicyIKRobot")).MakeInvisible()
        if self.settings.get("hold_receptacle") is not True:
            raise ValueError("This verified stage currently requires the disclosed source-pose receptacle fixture")
        target_prim = self.scene.objects[self.receptacle]
        matrix = np.asarray(UsdGeom.Xformable(target_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        target_q = _quat_wxyz(matrix[:3, :3])
        fixture = UsdPhysics.FixedJoint.Define(self.stage, "/World/PolicyReceptacleFixture")
        fixture.CreateBody1Rel().SetTargets([target_prim.GetPath()])
        fixture.CreateLocalPos0Attr(Gf.Vec3f(*map(float, matrix[:3, 3])))
        fixture.CreateLocalRot0Attr(Gf.Quatf(float(target_q[0]), Gf.Vec3f(*map(float, target_q[1:]))))
        fixture.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        fixture.CreateLocalRot1Attr(Gf.Quatf(1.))
        fixture.CreateExcludeFromArticulationAttr(True)
        self.world.reset()
        self.robot = Robot(self.world, self.spec)
        self.ik = Robot(self.world, self.spec, path="/World/PolicyIKRobot")
        self.context = Ctx(self.session, self.scene, self.world, self.stage, None)
        self.body = self.context.body(self.moving)
        self.target_body = self.context.body(self.receptacle)
        self.source_target_pose = self.context.pose(self.receptacle)
        self.local_points = np.asarray(self.context._local[self.moving], float)
        self.initial_opening = self.context.opening(self.receptacle)
        if not self.initial_opening:
            raise ValueError("Frozen source model has no reviewed opening semantics")
        rotated = self.local_points @ self.upright.T
        self.center_xy = (rotated.min(0)[:2] + rotated.max(0)[:2]) / 2
        self.bottom_offset = float(rotated[:, 2].min())
        self.radius = float(max(np.ptp(rotated[:, :2], axis=0)) / 2)
        self.clearance = float(self.initial_opening["radius"] - self.radius)
        if self.clearance <= 0:
            raise ValueError("Current source geometry does not have positive radial insertion clearance")
        self.ik_tolerance = (min(.00005, self.clearance / 16), .001)
        requested_home = self.ik.home()
        self.home = np.clip(requested_home, self.ik.lower, self.ik.upper)
        self.initializer_limit_projection = {
            name: {"requested_rad": float(requested_home[index]), "initialized_rad": float(self.home[index]),
                   "native_limits_rad": [float(self.ik.lower[index]), float(self.ik.upper[index])]}
            for index, name in enumerate(self.ik.names) if abs(requested_home[index]-self.home[index]) > 1e-8}
        self.joint_path = "/World/PolicyPregraspFixture"
        self.attached, self.released_ticks = False, 0
        self.handle_refreshes = []
        self.last_reset = None
        self.loaded = {}
        source_paths = [self.session.config_dir / "scene_graph.json", self.session.dir / "splat/robot_visual_fit.json"]
        for name in (self.moving, self.receptacle):
            fix = graph["dynamic_objects"][name]["fix"]
            if fix.get("mesh"):
                source_paths.append(resolve(fix["mesh"]))
            if fix.get("model"):
                source_paths.append(self.session.config_dir / fix["model"])
        support = graph["furniture"][graph["task"]["support"]].get("fix", {})
        if support.get("mesh"):
            source_paths.append(resolve(support["mesh"]))
        if support.get("model"):
            source_paths.append(self.session.config_dir / support["model"])
        for path in source_paths:
            path = Path(path).resolve(strict=True)
            self.loaded[str(path)] = _hash(path)
        # Robot CAD is an explicit permitted source asset, separate from
        # reconstructed task geometry. Hash its entry and all dependencies
        # through the upstream asset qualification rather than inventing CAD.
        self.robot_usd = str(Path(self.spec["usd"]).resolve(strict=True))
        from pxr import Sdf, UsdUtils
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(self.robot_usd))
        runtime_files = {self.robot_spec_file, Path(self.robot_usd)}
        runtime_files.update(Path(layer.realPath).resolve(strict=True) for layer in layers if layer.realPath)
        runtime_files.update(Path(asset).resolve(strict=True) for asset in assets)
        self.external_runtime_inputs = {str(path): _hash(path) for path in sorted(runtime_files)}
        self.unresolved_robot_assets = list(map(str, unresolved))

    def metadata(self):
        return {"world_sha256": self.binding["world_sha256"], "task_id": self.binding["task_id"],
                "backend_id": self.binding["backend_id"], "native_session": self.binding["native_session"],
                "policy_io_sha256": _digest(self.binding["policy_io"]), "engine": "isaacsim-physx",
                "loaded_scene_inputs": self.loaded, "robot_usd": self.robot_usd,
                "robot_usd_sha256": _hash(self.robot_usd), "source_robot_base": self.base,
                "external_runtime_inputs": self.external_runtime_inputs,
                "robot_unresolved_assets": self.unresolved_robot_assets,
                "source_joint_initializer": self.source_joint_initializer,
                "initializer_limit_projection": self.initializer_limit_projection,
                "ik_numerical_tolerance_m_rad": list(self.ik_tolerance),
                "robot_runtime": {"controlled_side": self.side,
                    "controlled_arm_joints": self.arm["ik_joints"], "controlled_hand_joints": self.robot.ee_joints(self.side),
                    "passive_joints": [name for name in self.robot.names if name not in self.arm["ik_joints"]
                                       and name not in self.robot.ee_joints(self.side)],
                    "physics": {"disable_gravity": self.spec["physics"].get("disable_gravity", True),
                                "self_collision": self.spec["physics"].get("self_collision", False),
                                "solver_iters": self.spec["physics"].get("solver_iters", [32, 4])},
                    "drives": self.spec["drives"],
                    "limitation": "Existing simulation robot settings retained; real gravity compensation, self-collision safety and real deployment are unverified."},
                "verification_scope": "pregrasp_insertion_simulation_only", "privileged_state": True,
                "robot_deployable": False, "success_rule": self.config["success_rule"],
                "geometry_replacement": False, "source_opening": self.initial_opening,
                "observations": "Measured simulated moving-body bottom/centre relative to current source opening, metres.",
                "fixture": "Source-pose receptacle fixed; pregrasp object attached to actual articulated TCP until release depth, then free settlement.",
                "limitations": ["Pregrasp is initialized, not a validated autonomous grasp.",
                                "Simulator body observations require a real perception adapter before deployment.",
                                "Reconstructed blind cavity depth and contact properties retain source uncertainty."]}

    def _remove_attachment(self):
        if self.stage.GetPrimAtPath(self.joint_path).IsValid():
            self.stage.RemovePrim(self.joint_path)
        self.attached = False

    def _refresh_articulations(self):
        """Recreate invalidated handles without resetting or advancing physics.

        Isaac 5.1 Articulation._on_prim_deletion deletes its tensor handle even
        when the removed prim is our separate temporary grasp joint. Its public
        initialize() then reads that missing attribute. New public wrappers are
        safe, but must not alter the native state. Compare native tensor state
        directly rather than trusting the invalidated Python wrappers.
        """
        from calibrated_scene_reconstruction.sim.robot_articulation import Robot

        def state(view):
            return {name: np.asarray(getattr(view, "get_" + name)()).copy()
                    for name in ("dof_positions", "dof_velocities", "link_transforms")}

        for role in ("robot", "ik"):
            old = getattr(self, role)
            if getattr(old.view, "_physics_view", None) is not None:
                continue
            # Robot.path is the reference container; the native articulation
            # root can be a descendant resolved by Isaac's public wrapper.
            roots = list(old.view.prim_paths)
            native = self.world.physics_sim_view.create_articulation_view(roots)
            before, started = state(native), float(self.world.current_time)
            replacement = Robot(self.world, self.spec, path=old.path)
            after = state(native)
            same = (started == float(self.world.current_time) and
                    all(np.array_equal(before[key], after[key]) for key in before))
            event = {"role": role, "path": old.path, "native_roots": roots, "sim_time_s": started,
                     "native_state_before_sha256": _digest({k: v.tolist() for k, v in before.items()}),
                     "native_state_after_sha256": _digest({k: v.tolist() for k, v in after.items()}),
                     "positions_velocities_link_poses_time_unchanged": same,
                     "reason": "Temporary joint removal invalidated the Isaac articulation wrapper."}
            self.handle_refreshes.append(event)
            if not same:
                raise RuntimeError("Articulation handle refresh changed native physical state")
            setattr(self, role, replacement)

    def _attach(self):
        from pxr import Gf, Usd, UsdPhysics
        from omni.physx import get_physx_interface
        position, rotation = self.robot.link_pose(self.arm["tcp_link"])
        body_position, body_rotation = self.context.pose(self.moving)
        before_q = self.robot.q().copy()
        before_time = float(self.world.current_time)
        # Publish the actual native initialization to the parser's USD output
        # before adding a new joint. No physics step, setter or velocity output.
        get_physx_interface().update_transformations(False, True, False, False)
        after_p, after_r = self.context.pose(self.moving)
        if (float(self.world.current_time) != before_time or not np.array_equal(before_q, self.robot.q())
                or not np.array_equal(body_position, after_p) or not np.array_equal(body_rotation, after_r)):
            raise RuntimeError("Native transform publication changed physical setup state")
        local = rotation.T @ (body_position - position)
        relative_rotation = rotation.T @ body_rotation
        local_rotation = _quat_wxyz(relative_rotation)
        self.attachment_local_position = local
        self.attachment_local_rotation = relative_rotation
        link = next(prim for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Robot"))
                    if prim.GetName() == self.arm["tcp_link"])
        joint = UsdPhysics.FixedJoint.Define(self.stage, self.joint_path)
        joint.CreateBody0Rel().SetTargets([link.GetPath()])
        joint.CreateBody1Rel().SetTargets([self.body.prim_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, local)))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(local_rotation[0]), Gf.Vec3f(*map(float, local_rotation[1:]))))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateExcludeFromArticulationAttr(True)
        self.attached = True
        self.attachment_initial_audit = self._attachment_measurement()
        if self.attachment_initial_audit["position_residual_m"] > 1e-6 or self.attachment_initial_audit["rotation_residual_rad"] > 1e-6:
            raise RuntimeError("Actual native attachment frames do not coincide at creation")

    def _attachment_measurement(self):
        from scipy.spatial.transform import Rotation
        position, rotation = self.robot.link_pose(self.arm["tcp_link"])
        body_position, body_rotation = self.context.pose(self.moving)
        frame_position = position + rotation @ self.attachment_local_position
        frame_rotation = rotation @ self.attachment_local_rotation
        return {"position_residual_m": float(np.linalg.norm(frame_position - body_position)),
                "rotation_residual_rad": float(np.linalg.norm(Rotation.from_matrix(frame_rotation.T @ body_rotation).as_rotvec())),
                "tcp_link_position_m": position.tolist(), "body_position_m": body_position.tolist(),
                "frame0_position_m": frame_position.tolist()}

    def reset(self, case):
        self._remove_attachment()
        self._refresh_articulations()
        self.context.place(self.receptacle, *self.source_target_pose)
        opening = self.context.opening(self.receptacle)
        rng = np.random.default_rng(int(_digest(case)[:16], 16))
        bounds = np.asarray(self.settings["initial_offset_bounds_m"], float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or not (bounds[:, 0] <= bounds[:, 1]).all():
            raise ValueError("Frozen reset bounds must be a finite 3x2 matrix")
        offset = rng.uniform(bounds[:, 0], bounds[:, 1])
        if offset[2] <= 0:
            raise ValueError("Pregrasp reset must begin above the opening")
        object_origin = np.r_[np.asarray(opening["xy"]) + offset[:2] - self.center_xy,
                              opening["rim_z"] + offset[2] - self.bottom_offset]
        tcp = object_origin + self.tcp_from_origin
        ik_start_time = float(self.world.current_time)
        self.q, error = self.ik.solve_ik(self.side, tcp + self.ik_offset, self.rotation, self.home,
                                        iters=int(self.settings.get("reset_ik_iterations", 350)), tol=self.ik_tolerance)
        if float(self.world.current_time) != ik_start_time:
            raise RuntimeError("IK-only synchronization unexpectedly advanced the physical scene")
        if error[0] > float(self.settings["reset_position_tolerance_m"]) or error[1] > float(self.settings["reset_rotation_tolerance_rad"]):
            raise RuntimeError("Source-base pregrasp IK unreachable: " + str(error))
        self.q = self.ik.with_ee(self.q, float(self.settings["hand_close_fraction"]))
        self.robot.teleport(self.q)  # Episode initialization only; never in step.
        self.robot.sync()
        self.robot.command(self.q)
        self.context.place(self.moving, object_origin, self.upright)
        self._attach()
        attachment_history = [self.attachment_initial_audit]
        self.released_ticks = 0
        for _ in range(int(self.settings["reset_settle_ticks"])):
            self.world.step(render=False)
            attachment_history.append(self._attachment_measurement())
        reset_commands = []
        # Position drives and grasp contact can settle away from a teleported
        # initializer. Physically reach the sampled start before recording the
        # student episode; never move the object pose to hide that drive error.
        for _ in range(int(self.settings.get("max_initialization_corrections", 16))):
            measured = self.observe()
            correction = offset - measured
            if np.linalg.norm(correction) <= min(.001, float(self.settings["reset_position_tolerance_m"])):
                break
            command = np.clip(correction, self.config["action_low"], self.config["action_high"])
            tcp, _ = self.robot.tcp_pose(self.side)
            native_time = float(self.world.current_time)
            self.q, correction_error = self.ik.solve_ik(self.side, tcp + command + self.ik_offset,
                self.rotation, self.robot.q(), iters=int(self.settings.get("step_ik_iterations", 65)), tol=self.ik_tolerance)
            if float(self.world.current_time) != native_time:
                raise RuntimeError("Initialization IK unexpectedly advanced the physical scene")
            self.q = self.robot.with_ee(self.q, float(self.settings["hand_close_fraction"]))
            self.robot.command(self.q)
            for _ in range(self.ticks):
                self.world.step(render=False)
            reset_commands.append({"before_offset_m": measured.tolist(), "command_m": command.tolist(),
                                   "after_offset_m": self.observe().tolist(), "physics_steps": self.ticks,
                                   "ik_error_m_rad": list(correction_error),
                                   "tcp_before_m": tcp.tolist(), "tcp_after_m": self.robot.tcp_pose(self.side)[0].tolist(),
                                   "max_joint_tracking_error_rad": float(np.max(np.abs(self.robot.q()-self.q))),
                                   "arm_tracking_error_rad": {name:float(self.robot.q()[self.robot.idx[name]]-self.q[self.robot.idx[name]]) for name in self.arm['ik_joints']},
                                   "ik_command_fk_m": (self.ik.tcp_pose(self.side)[0]-self.ik_offset).tolist(),
                                   "attachment": self._attachment_measurement()})
        actual = self.observe()
        self.last_reset = {"case": case, "desired_offset_m": offset.tolist(), "actual_offset_m": actual.tolist(),
                           "ik_error_m_rad": list(error), "teleport_scope": "episode initialization only",
                           "physical_initialization_commands": reset_commands,
                           "initial_attachment_history": attachment_history,
                           "native_handle_refreshes": list(self.handle_refreshes),
                           "initialization_scope": "Bounded physical setup before the first policy observation; no reset controller during recorded policy execution."}
        if np.linalg.norm(actual - offset) > float(self.settings["reset_position_tolerance_m"]):
            raise RuntimeError("Initialized physical pregrasp drift exceeds the frozen reset tolerance")
        self.policy_steps = 0
        return actual

    def observe(self):
        points = self.context.points(self.moving)
        opening = self.context.opening(self.receptacle)
        centre_xy = (points.min(0)[:2] + points.max(0)[:2]) / 2
        return np.r_[centre_xy - opening["xy"], points[:, 2].min() - opening["rim_z"]]

    def measure(self):
        points = self.context.points(self.moving)
        _, rotation = self.context.pose(self.moving)
        opening = self.context.opening(self.receptacle)
        bottom = float(points[:, 2].min())
        centre_xy = (points.min(0)[:2] + points.max(0)[:2]) / 2
        result = {"radial_m": float(np.linalg.norm(centre_xy - opening["xy"])),
                  "depth_m": float(opening["rim_z"] - bottom),
                  "floor_error_m": float(bottom - (opening["rim_z"] - opening["depth"])),
                  "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(rotation[2, self.long_axis]), 0., 1.)))),
                  "clearance_m": self.clearance, "attachment_active": bool(self.attached),
                  "released_ticks": int(self.released_ticks), "body_bottom_m": bottom,
                  "sim_time_s": float(self.world.current_time)}
        result["success"] = _score(result, self.config["success_rule"])
        return result

    def step(self, action):
        self._refresh_articulations()
        action = np.asarray(action, float)
        started_time = float(self.world.current_time)
        tcp, _ = self.robot.tcp_pose(self.side)
        self.q, error = self.ik.solve_ik(self.side, tcp + action + self.ik_offset, self.rotation,
                                        self.robot.q(), iters=int(self.settings.get("step_ik_iterations", 65)), tol=self.ik_tolerance)
        if float(self.world.current_time) != started_time:
            raise RuntimeError("IK-only synchronization unexpectedly advanced the physical scene")
        self.q = self.ik.with_ee(self.q, float(self.settings["hand_close_fraction"]) if self.attached else 0.)
        self.robot.command(self.q)
        for _ in range(self.ticks):
            self.world.step(render=False)
        if not self.attached:
            self.released_ticks += self.ticks
        before_release = self.measure()
        released_now = False
        if (self.attached and before_release["depth_m"] >= float(self.settings["release_depth_m"])
                and before_release["radial_m"] <= float(self.settings["release_radial_m"])
                and before_release["tilt_deg"] <= float(self.settings["release_tilt_deg"])):
            self._remove_attachment()
            self.q = self.robot.with_ee(self.q, 0.)
            self.robot.command(self.q)
            for _ in range(self.settle_ticks):
                self.world.step(render=False)
            self.released_ticks += self.settle_ticks
            released_now = True
        physical_steps = int(round((float(self.world.current_time) - started_time) / self.world.get_physics_dt()))
        reset_evidence = self.last_reset if self.policy_steps == 0 else {"case": self.last_reset["case"]}
        self.policy_steps += 1
        return {"command": action.tolist(), "physics_steps": physical_steps, "teleport_used": False,
                "articulated_position_control": True, "fixture_released_now": released_now,
                "attachment_frame_residual": self._attachment_measurement() if self.attached else None,
                "reset": reset_evidence}

    def controller(self, method, parameters):
        return lambda observation: choose_action(observation, method, parameters,
                    low=self.config["action_low"], high=self.config["action_high"], clearance=self.clearance,
                    target_depth=float(self.settings["controller_target_depth_m"]))

    def close(self):
        self.native_errors.close()
        self.app.close()

    def diagnostics(self):
        return {"native_physics_errors": self.native_errors.report(),
                "collider_precision": self.native_errors.precision_audit(),
                "initialization": self.last_reset,
                "native_handle_refreshes": self.handle_refreshes,
                "final_measurement": self.measure() if self.last_reset is not None else None}

    def snapshot(self, output):
        """Qualification-only visualization; does not move geometry or policy state."""
        import cv2
        import omni.replicator.core as rep
        from calibrated_scene_reconstruction.sim.scene_stage import configure_render
        from calibrated_scene_reconstruction.sim.usd import add_camera
        configure_render(mode="rt", auto_exposure=True)
        opening = self.context.opening(self.receptacle)
        look = np.r_[opening["xy"], opening["rim_z"] + .04]
        eye = look + np.array([.24, -.32, .19])
        width, height = 960, 720
        intrinsic = np.array([[width / (2 * np.tan(np.radians(25))), 0., width / 2],
                              [0., width / (2 * np.tan(np.radians(25))), height / 2], [0., 0., 1.]])
        camera = add_camera(self.stage, "/World/Cams/native_policy_probe", intrinsic, width, height, eye=eye, target=look)
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(product)
        for _ in range(20):
            self.world.render()  # Rendering only; no extra physical integration.
        frame = np.asarray(annotator.get_data())
        if frame.ndim != 3 or frame.shape[:2] != (height, width):
            raise RuntimeError("Native probe camera returned no valid image")
        output = Path(output).resolve()
        if not cv2.imwrite(str(output), np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])):
            raise RuntimeError("Could not save native probe visualization")
        annotator.detach(product)
        return str(output)
