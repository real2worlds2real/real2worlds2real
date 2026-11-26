"""Propose a disclosed pregrasp insertion configuration from source mesh/opening.

This is a proposal, not qualification evidence. Native IK, body tracking,
collision behaviour, release and negative-control outcomes must be inspected
before this configuration is frozen into a policy experiment. No grasp pose is
claimed to have been observed in the source video.
"""
import argparse
from pathlib import Path

import numpy as np

from bound_training import write_new


def policy_io():
    """Actual adapter I/O, explicitly privileged and simulation-only."""
    return {"observations": [{"name": "source_body_relative_opening", "shape": [3], "dtype": "float32",
                             "units": "m", "frame": "world", "available_on_robot": False}],
            "actions": [{"name": "tcp_delta", "shape": [3], "dtype": "float32", "units": "m",
                         "frame": "world", "available_on_robot": True}], "control_hz": 30.,
            "max_latency_ms": 33.333, "action_alignment": "observation_t -> executed_command_t",
            "normalization": "train-only dense BC statistics",
            "deployment_controller": "simulation-only bounded Cartesian IK/position drives"}


def propose(session_path, backend_id, *, initial_xy_m=.006, initial_z_m=(.008, .018), tcp_above_top_m=.06):
    from calibrated_scene_reconstruction.core.paths import resolve
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.robots.spec import SPEC_DIR, load_spec
    from calibrated_scene_reconstruction.tasks.rigid import bound_insertion
    import bound_insertion_factory
    session = Session(session_path)
    graph = session.scene_graph
    robot_spec = graph["photo_robot"]["spec"]
    robot_spec_path = Path(robot_spec)
    robot_spec_path = (robot_spec_path if robot_spec_path.suffix else SPEC_DIR / (robot_spec + ".json")).resolve(strict=True)
    robot_usd = Path(load_spec(robot_spec)["usd"]).resolve(strict=True)
    # A conservative declaration; the native factory independently enumerates
    # the actual USD dependency closure and rejects any undeclared file later.
    runtime_assets = [str(robot_spec_path), *[str(path.resolve()) for path in sorted(robot_usd.parent.rglob('*')) if path.is_file()]]
    task = graph["task"]
    if task["kind"] != "insert":
        raise ValueError("Source task is not insertion")
    moving = graph["dynamic_objects"][task["target"]]["fix"]
    target = graph["dynamic_objects"][task["goal"]]["fix"]
    with np.load(resolve(moving["mesh"]), allow_pickle=False) as archive:
        points = archive["V"].astype(float)
    # Select the source mesh's longest dimension and rotate it vertically.
    axis = int(np.argmax(np.ptp(points, axis=0)))
    rotations = (np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
                 np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), np.eye(3))
    rotation = rotations[axis]
    world_points = points @ rotation.T
    radius = float(max(np.ptp(world_points[:, :2], axis=0)) / 2)
    opening = target["opening"]
    clearance = float(opening["radius"] - radius)
    if clearance <= 0:
        raise ValueError("Source mesh has no positive aperture clearance")
    depth = float(opening["depth"])
    rules = [
        {"field": "clearance_m", "operator": "ge", "target": 1e-9},
        {"field": "depth_m", "operator": "ge", "target": min(.005, .5 * depth)},
        {"field": "floor_error_m", "operator": "abs_le", "target": .0015},
        {"field": "radial_m", "operator": "le", "target": max(.001, clearance)},
        {"field": "tilt_deg", "operator": "le", "target": 3.},
        {"field": "attachment_active", "operator": "eq", "target": False},
        {"field": "released_ticks", "operator": "ge", "target": 84},
    ]
    return {"schema": "astra_bound_native_v1", "backend_id": backend_id, "simulation_only": True,
            "policy_io": policy_io(),
            "runtime_asset_files": sorted(set(runtime_assets)),
            "factory": "bound_insertion_factory:build", "factory_file": str(Path(bound_insertion_factory.__file__).resolve()),
            "implementation_files": [str(Path(bound_insertion.__file__).resolve())],
            "engine": "isaacsim-physx", "action_low": [-.003, -.003, -.003], "action_high": [.003, .003, .003],
            "max_steps": 180, "success_ticks": 1,
            "collectors": {"servo": {"gain": .7, "alignment_fraction": .6},
                           "mpc": {"gain": .7, "alignment_fraction": .6, "response": .55, "horizon": 3}},
            "success_rule": rules,
            "proposal_provenance": {"grasp": "Engineered pregrasp fixture initialization; not an observed or learned grasp.",
                    "task_thresholds": "Proposed terminal insertion criteria; review against the source task definition before freeze. Explicit fixture removal and settling are mandatory.",
                    "mesh": str(resolve(moving["mesh"])), "source_opening": opening,
                    "requires_native_qualification": True},
            "environment": {"scope": "pregrasp_insertion", "fixture_release_before_success": True,
                "hold_receptacle": True, "object_rotation": rotation.tolist(), "object_long_axis": axis,
                "tcp_rotation": [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
                "tcp_from_object_origin_m": [0., 0., float(world_points[:, 2].max()) + tcp_above_top_m],
                "initial_offset_bounds_m": [[-initial_xy_m, initial_xy_m], [-initial_xy_m, initial_xy_m], list(initial_z_m)],
                "physics_hz": 120, "reset_settle_ticks": 30, "reset_position_tolerance_m": .004,
                "reset_rotation_tolerance_rad": .05, "hand_close_fraction": .85,
                "release_depth_m": depth - .001, "release_radial_m": max(.001, clearance),
                "release_tilt_deg": 3., "release_settle_seconds": .7, "controller_target_depth_m": depth - .0005,
                "reset_ik_iterations": 350, "step_ik_iterations": 65}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-xy-m", type=float, default=.006)
    parser.add_argument("--tcp-above-top-m", type=float, default=.06)
    args = parser.parse_args()
    write_new(args.output, propose(args.session, args.backend_id, initial_xy_m=args.initial_xy_m,
                                  tcp_above_top_m=args.tcp_above_top_m))


if __name__ == "__main__":
    main()
