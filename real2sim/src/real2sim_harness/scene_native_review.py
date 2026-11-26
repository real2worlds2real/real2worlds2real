"""Reapply retained source behavior checks to actual shared-world trajectories."""

from __future__ import annotations

import numpy as np


def review_native(records, bodies, particles, times):
    from scipy.spatial.transform import Rotation

    from .deformable_behavior import measure_response
    from .deformable_geometry import (
        audit_box_contacts,
        audit_sphere_contacts,
        shell_metrics,
        solid_metrics,
    )
    from .rigid_contact import thread_motion

    results = []
    for name, record in records.items():
        action = record.get("native_action")
        if not action or action["type"] != "native_test":
            continue
        c = record["native_configuration"]
        local_times = np.asarray(times) - action["start_seconds"]
        duration = (c["frames"] - 1) / c["fps"]
        selected = np.flatnonzero((local_times >= -1e-8) & (local_times <= duration + 1e-8))
        row = {"asset": name, "ok": True, "failures": [], "action_start_seconds": action["start_seconds"]}
        if len(selected) < 2 or local_times[-1] < duration - 1e-6:
            row.update(ok=False, failures=["incomplete_native_action_window"])
            results.append(row)
            continue
        placement = np.asarray(record["transform"])
        rotation = Rotation.from_matrix(placement[:3, :3])

        def local_body(body_name, record=record, placement=placement, rotation=rotation):
            index = record["body_ids"][body_name]
            if index < 0:
                return np.tile(record["native_static_poses"][body_name], (len(times), 1))
            pose = bodies[:, index].copy()
            pose[:, :3] = (pose[:, :3] - placement[:3, 3]) @ placement[:3, :3]
            pose[:, 3:] = (rotation.inv() * Rotation.from_quat(pose[:, 3:])).as_quat()
            return pose

        if record["native_representation"] == "rigid_contact":
            row["thread_checks"] = []
            for check in c.get("thread_checks", []):
                reference, moving = local_body(check["reference_body"]), local_body(check["moving_body"])
                inverse = Rotation.from_quat(reference[:, 3:]).inv()
                relative = np.column_stack((inverse.apply(moving[:, :3]-reference[:, :3]),
                                           (inverse * Rotation.from_quat(moving[:, 3:])).as_quat()))
                start, end = check.get("start_frame", 0)/c["fps"], check.get("end_frame", c["frames"]-1)/c["fps"]
                ids = np.flatnonzero((local_times >= start-1e-8) & (local_times <= end+1e-8))
                measured = thread_motion(relative[ids], lead_m_per_turn=check["lead_m_per_turn"],
                                         axis=check.get("axis_reference", [0,0,1]),
                                         tolerance_m=check["tolerance_m"], minimum_turns=check.get("minimum_turns", .25))
                row["thread_checks"].append(measured)
                if not measured["ok"]:
                    row["failures"].append("thread_motion")
            bounds = {}
            for body in c["bodies"]:
                pose = local_body(body["id"])[selected[-1]]
                vertices = Rotation.from_quat(pose[3:]).apply(np.asarray(body["vertices"])) + pose[:3]
                bounds[body["id"]] = vertices.min(0), vertices.max(0)
            row["separation_checks"] = []
            for check in c.get("separation_checks", []):
                lo0, hi0 = bounds[check["body0"]]
                lo1, hi1 = bounds[check["body1"]]
                gap = float(np.linalg.norm(np.maximum(np.maximum(lo0-hi1, lo1-hi0), 0)))
                row["separation_checks"].append({**check, "final_aabb_gap_m": gap, "ok": gap >= check["minimum_gap_m"]})
                if gap < check["minimum_gap_m"]:
                    row["failures"].append("body_separation")
        else:
            points = (particles[:, record["particle_start"]:record["particle_end"]] - placement[:3,3]) @ placement[:3,:3]
            g = c["geometry"]
            volume = record["native_representation"] == "volume"
            metrics = (solid_metrics(g["vertices"], g["tetrahedra"], points) if volume
                       else shell_metrics(g["vertices"], g["triangles"], points))
            row["metrics"] = metrics
            if metrics.get("inverted_element_samples", 0) or metrics.get("minimum_area_ratio", 1) < .05:
                row["failures"].append("invalid_native_elements")
            stretch = metrics.get("maximum_principal_stretch", metrics.get("maximum_edge_stretch"))
            if stretch > c.get("review", {}).get("maximum_stretch", 1.5):
                row["failures"].append("excessive_native_stretch")
            measurements = measure_response(points[selected], local_times[selected], c.get("measurements", []))
            row["measurements"] = measurements
            row["failures"].extend("measurement:"+m["id"] for m in measurements if not m["ok"])
            probes = [p for p in c.get("probes", []) if p["id"] in record.get("native_probe_ids", {})]
            if probes:
                poses = bodies[:, [record["native_probe_ids"][p["id"]] for p in probes]].copy()
                poses[..., :3] = (poses[..., :3]-placement[:3,3]) @ placement[:3,:3]
                poses[..., 3:] = (rotation.inv()*Rotation.from_quat(poses[...,3:].reshape(-1,4))).as_quat().reshape(poses[...,3:].shape)
                kwargs = {"closed":volume, "tolerance_m":c.get("review",{}).get("contact_tolerance_m",.001),
                          "surface_thickness_m":c["material"].get("thickness_m",0) if not volume else 0}
                row["sphere_contacts"] = audit_sphere_contacts(points,g["triangles"],poses,probes,**kwargs)
                row["box_contacts"] = audit_box_contacts(points,g["triangles"],poses,probes,**kwargs)
                if not row["sphere_contacts"]["ok"] or not row["box_contacts"]["ok"]:
                    row["failures"].append("source_fixture_penetration")
        row["ok"] = not row["failures"]
        results.append(row)
    return results
