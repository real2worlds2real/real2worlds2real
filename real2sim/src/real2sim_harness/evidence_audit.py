"""Reusable numerical diagnostics with explicit identity, coverage and scope."""
from __future__ import annotations

import json

import numpy as np

from .artifacts import atomic_json, sha256_file


def trajectory(time_s, body_names, poses_xyzw):
    times = np.asarray(time_s, dtype=float)
    names = list(body_names)
    poses = np.asarray(poses_xyzw, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or poses.shape != (len(times), len(names), 7)
            or not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("Use time_s[T], unique body_names[B] and poses_xyzw[T,B,7], T>=2")
    if not np.isfinite(times).all() or not np.isfinite(poses).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory states must be finite and timestamps strictly increasing")
    norms = np.linalg.norm(poses[:, :, 3:], axis=-1)
    if np.any(np.abs(norms - 1) > 1e-3):
        raise ValueError("Trajectory quaternions must be unit XYZW")
    poses = poses.copy()
    poses[:, :, 3:] /= norms[:, :, None]
    return times, names, poses


def body_index(names, body):
    if not isinstance(body, str) or body not in names:
        raise ValueError("Select the recorded rigid body by its exact identity")
    return names.index(body)


def window_statistics(times, names, poses, *, body, last_seconds, require_full_window=True):
    times, names, poses = trajectory(times, names, poses)
    if not np.isfinite(last_seconds) or last_seconds <= 0 or type(require_full_window) is not bool:
        raise ValueError("Choose a positive final window and an explicit full-window policy")
    complete = bool(times[-1] - times[0] >= last_seconds - 1e-9)
    if require_full_window and not complete:
        raise ValueError("Recorded trajectory does not cover the requested final window")
    index = body_index(names, body)
    selected = times >= times[-1] - last_seconds - 1e-9
    x, t = poses[selected, index], times[selected]
    if len(t) < 2:
        raise ValueError("Final window needs at least two recorded samples")
    angles = 2 * np.arccos(np.clip(np.abs(x[:, 3:] @ x[0, 3:]), 0, 1))
    return {"body": body, "body_index": index, "samples": len(t), "window_complete": complete,
            "window_start_s": float(t[0]), "window_end_s": float(t[-1]),
            "position_mean_m": x[:, :3].mean(0).tolist(),
            "max_position_deviation_from_mean_m": float(np.linalg.norm(x[:, :3] - x[:, :3].mean(0), axis=1).max()),
            "max_sampled_speed_m_s": float(np.linalg.norm(np.diff(x[:, :3], axis=0) / np.diff(t)[:, None], axis=1).max()),
            "max_orientation_change_from_window_start_deg": float(np.degrees(angles.max())),
            "scope": "Recorded window statistics; no support contact or perturbation-stability certification."}


def command_tracking(times, names, poses, *, body, commands, tolerance_m, require_full_command=True):
    times, names, poses = trajectory(times, names, poses)
    cmd = np.asarray(commands, dtype=float)
    if (cmd.ndim != 2 or cmd.shape[1] != 4 or len(cmd) < 2 or not np.isfinite(cmd).all()
            or np.any(np.diff(cmd[:, 0]) <= 0)):
        raise ValueError("Commands need increasing finite [seconds,x_m,y_m,z_m] rows")
    if not np.isfinite(tolerance_m) or tolerance_m < 0 or type(require_full_command) is not bool:
        raise ValueError("Use a finite nonnegative tolerance and explicit full-command policy")
    if times[0] < cmd[0, 0] - 1e-9 or times[-1] > cmd[-1, 0] + 1e-9:
        raise ValueError("Command extrapolation is not allowed")
    complete = bool(times[0] <= cmd[0, 0] + 1e-9 and times[-1] >= cmd[-1, 0] - 1e-9)
    if require_full_command and not complete:
        raise ValueError("Observed timestamps do not cover the complete command interval")
    observed = poses[:, body_index(names, body), :3]
    expected = np.stack([np.interp(times, cmd[:, 0], cmd[:, i]) for i in (1, 2, 3)], axis=1)
    error = np.linalg.norm(observed - expected, axis=1)
    worst = int(np.argmax(error))
    return {"within_tolerance": bool(np.all(error <= tolerance_m)), "body": body,
            "complete_command_coverage": complete, "samples": len(times),
            "maximum_error_m": float(error[worst]), "rms_error_m": float(np.sqrt(np.mean(error**2))),
            "failed_samples": int(np.count_nonzero(error > tolerance_m)),
            "worst": {"time_s": float(times[worst]), "observed_m": observed[worst].tolist(),
                      "commanded_m": expected[worst].tolist()},
            "scope": "Translation at recorded timestamps only; does not establish readback provenance, rotation, contact or forces."}


def axial_motion(times, names, poses, *, moving, reference, axis, interval, max_excursion_m=None):
    from scipy.spatial.transform import Rotation

    times, names, poses = trajectory(times, names, poses)
    bounds, direction = np.asarray(interval, float), np.array(axis, dtype=float, copy=True)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Choose two finite increasing interval bounds")
    if bounds[0] < times[0] - 1e-9 or bounds[1] > times[-1] + 1e-9:
        raise ValueError("Trajectory must cover the declared fit interval")
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError("Axis must be a finite nonzero 3-vector in reference-body coordinates")
    if max_excursion_m is not None and (not np.isfinite(max_excursion_m) or max_excursion_m < 0):
        raise ValueError("Maximum excursion must be finite and nonnegative")
    if moving == reference:
        raise ValueError("Moving and reference bodies must differ")
    m, r = body_index(names, moving), body_index(names, reference)
    direction /= np.linalg.norm(direction)
    rr, rm = Rotation.from_quat(poses[:, r, 3:]), Rotation.from_quat(poses[:, m, 3:])
    relative_rotation = rr.inv() * rm
    q = relative_rotation.as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1
    projection = q[:, :3] @ direction
    if np.any(np.hypot(projection, q[:, 3]) < 1e-8):
        raise ValueError("Twist is undefined near a 180-degree swing")
    angle = np.unwrap(2 * np.arctan2(projection, q[:, 3]))
    relative = rr.inv().apply(poses[:, m, :3] - poses[:, r, :3])
    delta = relative - relative[0]
    travel = delta @ direction
    selected = (times >= bounds[0]) & (times <= bounds[1])
    if selected.sum() < 5 or np.ptp(angle[selected]) < .25:
        raise ValueError("Fit needs at least five samples and .25 radians of observed rotation")
    slope, offset = np.polyfit(angle[selected], travel[selected], 1)
    residual = travel[selected] - (slope * angle[selected] + offset)
    excursion = float(np.linalg.norm(delta, axis=1).max())
    swing = (relative_rotation * Rotation.from_rotvec(angle[:, None] * direction).inv()).magnitude()
    return {"moving": moving, "reference": reference, "fit_interval_s": bounds.tolist(),
            "lead_m_per_revolution": float(slope * 2 * np.pi),
            "fit_rms_residual_m": float(np.sqrt(np.mean(residual**2))),
            "fit_max_abs_residual_m": float(np.abs(residual).max()),
            "full_max_excursion_m": excursion, "full_axial_range_m": float(np.ptp(travel)),
            "full_max_transverse_change_m": float(np.linalg.norm(delta - travel[:, None] * direction, axis=1).max()),
            "full_max_swing_rad": float(swing.max()),
            "largest_sampled_twist_increment_rad": float(np.abs(np.diff(angle)).max()),
            "flags": ["full_trajectory_excursion_exceeds_declared_bound"]
            if max_excursion_m is not None and excursion > max_excursion_m else [],
            "scope": "Sampled relative poses; angle unwrapping assumes less than pi per sample. "
                     "Lead/residuals do not prove thread contact, clamping, release or continuous-time nonpenetration."}


def sphere_clearance(colliders, spheres, margin_m=0):
    from scipy.spatial import ConvexHull, QhullError

    if not colliders or not spheres or not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Supply convex colliders, spheres and a nonnegative finite margin")
    hulls = []
    for i, item in enumerate(colliders):
        vertices = np.asarray(item["vertices"], float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("Each convex material piece needs at least four finite XYZ vertices")
        try:
            planes = ConvexHull(vertices).equations
        except QhullError as error:
            raise ValueError("Collider vertices must span a 3D convex volume") from error
        planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        hulls.append((item.get("name", str(i)), planes))
    result = []
    for i, item in enumerate(spheres):
        center, radius = np.asarray(item["center_m"], float), float(item["radius_m"])
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Sphere needs finite XYZ center and positive radius")
        distances = [(name, float(np.max(eq[:, :3] @ center + eq[:, 3]))) for name, eq in hulls]
        name, lower = min(distances, key=lambda r: r[1])
        epsilon = 64 * np.finfo(float).eps * max(1, np.linalg.norm(center), radius)
        inside = [name for name, distance in distances if distance < -epsilon]
        result.append({"name": item.get("name", str(i)), "limiting_collider": name,
                       "plane_distance_lower_bound_m": lower,
                       "status": "center_inside" if inside else
                       "clear_proven" if lower > radius + margin_m + epsilon else "unresolved"})
    return {"all_clear_proven": all(r["status"] == "clear_proven" for r in result), "spheres": result,
            "scope": "Conservative clearance against separately supplied convex material pieces in one metre frame. "
                     "Unresolved does not prove collision; no cooking, swept-motion or simulator-contact certification."}


def run_audit(request, base, output):
    from .posed_observations import file_path

    if request.get("schema_version") != 1 or not isinstance(request.get("assumptions"), str) or not request["assumptions"].strip():
        raise ValueError("Use schema_version 1 and describe units/frame/readback assumptions")
    mode = request["audit"]
    hashes = {}
    if mode in {"rigid_window", "command_tracking", "axial_motion"}:
        path = file_path(base, request["trajectory"])
        hashes[str(path)] = sha256_file(path)
        layout = request.get("layout", {})
        if set(layout) - {"time", "bodies", "poses", "quaternion_order"}:
            raise ValueError("Unknown trajectory layout fields")
        with np.load(path, allow_pickle=False) as data:
            t = data[layout.get("time", "time_s")]
            names = data[layout.get("bodies", "body_names")].tolist()
            p = data[layout.get("poses", "poses_xyzw")].copy()
        order = layout.get("quaternion_order", "XYZW")
        if order == "WXYZ":
            p = p[..., [0, 1, 2, 4, 5, 6, 3]]
        elif order != "XYZW":
            raise ValueError("Declare XYZW or WXYZ; never infer quaternion order")
        if mode == "rigid_window":
            result = window_statistics(t, names, p, body=request["body"], last_seconds=request["last_seconds"],
                                       require_full_window=request.get("require_full_window", True))
            passed = None
        elif mode == "command_tracking":
            result = command_tracking(t, names, p, body=request["body"], commands=request["commands"],
                                      tolerance_m=request["tolerance_m"],
                                      require_full_command=request.get("require_full_command", True))
            passed = result["within_tolerance"] and result["complete_command_coverage"]
        else:
            result = axial_motion(t, names, p, moving=request["moving"], reference=request["reference"],
                                  axis=request["axis"], interval=request["interval_s"],
                                  max_excursion_m=request.get("max_excursion_m"))
            passed = False if result["flags"] else None
    elif mode == "convex_sphere_starts":
        result = sphere_clearance(request["colliders"], request["spheres"], request.get("margin_m", 0))
        passed = result["all_clear_proven"]
    elif mode == "support_coverage":
        from .scene_support import audit_support_requirements, support_requirements
        path = file_path(base, request["support_report"])
        hashes[str(path)] = sha256_file(path)
        rows = request["requirements"]
        requirements = support_requirements(rows, {r["asset_id"] for r in rows}, {r["support_id"] for r in rows})
        result = audit_support_requirements(json.loads(path.read_text()), requirements)
        passed = result["ok"]
    else:
        raise ValueError("Choose rigid_window, command_tracking, axial_motion, convex_sphere_starts or support_coverage")
    report = {"ok": True, "operation": "evidence_audit", "audit": mode, "audit_passed": passed,
              "result": result, "input_hashes": hashes, "assumptions": request["assumptions"],
              "asset_accepted": False, "preview_images": [],
              "scope": "ok means audit computation completed; audit_passed describes only its stated numerical scope."}
    atomic_json(output / "report.json", report)
    return report
