"""Rigid-body handoff without invented fragment impulses (quaternions are XYZW)."""

import numpy as np


def rotation(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def release_states(pose, velocity, offsets):
    pose, velocity, offsets = map(lambda x: np.asarray(x, dtype=float), (pose, velocity, offsets))
    if pose.shape != (7,) or velocity.shape != (6,) or offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("Expected pose[7], velocity[6], center offsets[N,3]")
    if (
        not all(np.isfinite(x).all() for x in (pose, velocity, offsets))
        or np.linalg.norm(pose[3:]) < 0.9
    ):
        raise ValueError("Finite rigid handoff states and a unit quaternion required")
    world = offsets @ rotation(pose[3:]).T
    transforms = np.repeat(pose[None, :], len(world), axis=0)
    transforms[:, :3] += world
    velocities = np.repeat(velocity[None, :], len(world), axis=0)
    velocities[:, :3] += np.cross(velocity[3:], world)
    return transforms.astype(np.float32), velocities.astype(np.float32)


def audit_motion(configuration, poses, velocities, inertias, broken):
    """Independent retained-state geometry and mechanical-energy checks for a free drop."""
    poses, velocities, inertias = map(np.asarray, (poses, velocities, inertias))
    rows = configuration["pieces"]
    masses = np.asarray([r["mass_kg"] for r in rows])
    if (
        poses.shape != (*velocities.shape[:2], 7)
        or velocities.shape[-1] != 6
        or poses.shape[1] != len(rows)
    ):
        raise ValueError("Native poses and velocities must map to every shard")
    inertias = inertias.reshape(len(rows), 3, 3)
    minimum_z = 0.0
    energies = []
    for frame, speed, released in zip(poses, velocities, broken, strict=True):
        matrices = np.asarray([rotation(p[3:]) for p in frame])
        omega_local = np.einsum("nji,nj->ni", matrices, speed[:, 3:])
        rotational = 0.5 * np.einsum("ni,nij,nj->n", omega_local, inertias, omega_local)
        energies.append(
            float(
                (
                    0.5 * masses * (speed[:, :3] ** 2).sum(axis=1)
                    + rotational
                    + masses * 9.81 * frame[:, 2]
                ).sum()
            )
        )
        if released:
            for row, pose, r in zip(rows, frame, matrices, strict=True):
                z = np.asarray(row["vertices"]) @ r[2] + pose[2]
                minimum_z = min(minimum_z, float(z.min()))
    initial = max(energies[0], 1e-12)
    rest = np.asarray([r["center"] for r in rows])
    initial_dist = np.linalg.norm(rest[:, None] - rest[None, :], axis=2)
    final_dist = np.linalg.norm(poses[-1, :, None, :3] - poses[-1, None, :, :3], axis=2)
    separation = float(np.percentile(abs(final_dist - initial_dist), 95))
    return {
        "max_visible_vertex_floor_penetration_m": max(0.0, -minimum_z),
        "initial_mechanical_energy_j": energies[0],
        "max_retained_mechanical_energy_j": max(energies),
        "max_energy_ratio_to_initial": max(energies) / initial,
        "relative_pair_distance_change_p95_m": separation,
        "audited_frames": len(poses),
        "scope": "Every retained shard mesh vertex vs analytic plane; native inertia/velocity energy bound for unforced drop. Not a full inter-shard mesh collision certificate.",
    }
