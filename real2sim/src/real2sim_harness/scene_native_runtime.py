"""Append retained native construction to a shared builder without installing a test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_part(folder, *, device, sdf_cache, action=None):
    import newton
    from newton.solvers import SolverVBD

    folder = Path(folder)
    bundle = json.loads((folder / "asset_bundle.json").read_text())
    mode = bundle["native_representation"]
    c = json.loads((folder / ("native_test_configuration.json" if action else "native_configuration.json")).read_text())
    if action and "controls" in action:
        if mode != "rigid_contact":
            raise ValueError("Wrench overrides require a native rigid representation")
        c["controls"] = action["controls"]
    local = newton.ModelBuilder(gravity=(0, 0, -9.81))
    SolverVBD.register_custom_attributes(local)
    record = {"body_ids": {}, "joint_ids": {}, "drives": [], "attachments": [],
              "loop_anchors": [], "rest_matrices": {}, "native_representation": mode,
              "native_configuration": c, "native_action": action}
    if mode == "rigid_contact":
        from .rigid_contact_runtime import load_rigid_contact
        local, body_ids, c = load_rigid_contact(c, cache_dir=sdf_cache, device=device, builder=local)
        record["native_configuration"] = c
        record["body_ids"] = body_ids
        record["native_source_bindings"] = c["source_bindings"]
        record["native_static_poses"] = {b["id"]: b.get("pose", [0, 0, 0, 0, 0, 0, 1])
                                         for b in c["bodies"] if b.get("static", False)}
    elif mode in {"surface", "volume"}:
        from .deformable_runtime import load_deformable
        # Source fixtures are part of a surface assembly. Diagnostic load probes
        # are not installed as passive volume geometry.
        if mode == "volume" and not action:
            c = dict(c, probes=[])
        part = load_deformable(c, device=device, builder=local, build_only=True,
                               include_ground=False, include_probes=(mode == "surface" or bool(action)),
                               prescribed_motion=bool(action))
        record["control_mesh"] = {"faces": c["geometry"]["triangles"]}
        record["native_probe_ids"] = {p["specification"]["id"]: p["body"] for p in part["probes"]}
        record["native_triangles"] = part["triangles"]
        record["native_tetrahedra"] = part["tetrahedra"]
        record["native_test_probes"] = [p["specification"] for p in part["probes"]
                                         if not p["specification"].get("source_visual_objects")]
    else:
        raise ValueError("Unsupported native scene representation")
    return local, record


def excluded_self_pairs(builder, records, *, maximum_pairs=20_000_000):
    """Disable only an opted-out asset's own pairs, preserving cross-asset contacts."""
    triangles = np.asarray(builder.tri_indices, dtype=np.int32).reshape(-1, 3)
    edges = np.asarray(builder.edge_indices, dtype=np.int32).reshape(-1, 4)
    vertex_map, edge_map, count = {}, {}, 0
    for r in records.values():
        if r["parameters"]["solver"]["self_contact"]:
            continue
        lo, hi = r["particle_start"], r["particle_end"]
        owned_triangles = np.flatnonzero(((triangles >= lo) & (triangles < hi)).all(axis=1)).tolist()
        owned_edges = np.flatnonzero(((edges[:, 2:] >= lo) & (edges[:, 2:] < hi)).all(axis=1)).tolist()
        count += (hi - lo) * len(owned_triangles) + len(owned_edges)**2
        if count > maximum_pairs:
            raise ValueError(
                f"Self-contact exclusion budget exceeded at asset {r.get('id', '<unnamed>')}: "
                f"required {count} pairs > budget {maximum_pairs}; "
                f"particles={hi-lo}, own_boundary_triangles={len(owned_triangles)}, "
                f"own_edges={len(owned_edges)}. Preserve detailed visual followers while "
                "coarsening the physical control, or use a compatible contact adapter; "
                "do not disable another participant's required self-contact."
            )
        vertex_map.update({i: owned_triangles for i in range(lo, hi)})
        edge_map.update({i: owned_edges for i in owned_edges})
    return vertex_map, edge_map
