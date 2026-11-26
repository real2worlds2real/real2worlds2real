"""Passive concave mesh bodies for native Newton SDF contact experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .rigid_contact import validate_request


def load_rigid_contact(request, *, cache_dir, device="cuda:0", builder=None):
    """Append passive bodies to an optional builder; never install test drives."""
    import newton
    import warp as wp

    c = validate_request(request)
    own_builder = builder is None
    builder = newton.ModelBuilder() if own_builder else builder
    if own_builder:
        builder.gravity = tuple(c.get("gravity", [0, 0, -9.81]))
    contact = c["contact"]
    cfg = newton.ModelBuilder.ShapeConfig(
        margin=0.0,
        gap=contact["gap_m"],
        mu=contact["friction"],
        density=contact["density_kg_m3"],
        ke=1e6,
        kd=100.0,
        mu_torsional=0.0,
        mu_rolling=0.0,
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    body_ids = {}
    with wp.ScopedDevice(device):
        for body in c["bodies"]:
            mesh = newton.Mesh(
                np.asarray(body["vertices"], np.float32),
                np.asarray(body["triangles"], np.int32).ravel(),
            )
            mesh.build_sdf(
                max_resolution=contact["sdf_resolution"],
                target_voxel_size=contact.get("sdf_voxel_size_m"),
                narrow_band_range=(-contact["sdf_band_m"], contact["sdf_band_m"]),
                margin=contact["sdf_band_m"],
                cache_dir=str(cache_dir),
            )
            pose = body.get("pose", [0, 0, 0, 0, 0, 0, 1])
            transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            if body.get("static", False):
                idx = -1
                builder.add_shape_mesh(idx, mesh=mesh, xform=transform, cfg=cfg, label=body["id"])
            else:
                idx = builder.add_body(xform=transform, label=body["id"])
                builder.add_shape_mesh(idx, mesh=mesh, cfg=cfg, label=body["id"])
            body_ids[body["id"]] = idx
    return builder, body_ids, c
