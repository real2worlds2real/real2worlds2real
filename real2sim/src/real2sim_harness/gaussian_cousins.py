"""Optional CPU Gaussian background cousins with explicit appearance and spatial edits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .gaussian_background import _input, similarity
from .measurement import rigid_transform


def _color(spec):
    matrix = np.asarray(spec.get("matrix", np.eye(3)), dtype=float)
    offset = np.asarray(spec.get("offset", [0, 0, 0]), dtype=float)
    if (
        matrix.shape != (3, 3)
        or offset.shape != (3,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
        or np.abs(matrix).max() > 8
        or np.abs(offset).max() > 8
    ):
        raise ValueError("Color grading uses a finite bounded RGB 3x3 matrix and 3-vector offset")
    return matrix, offset


def edit_background(request, base, output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, UsdVol, Vt

    from .scene import _check_stage

    source = _input(base, request.get("source"))
    digest = sha256_file(source)
    original, root, _, _ = _check_stage(source)
    if any(
        p.HasAPI(UsdPhysics.RigidBodyAPI)
        or p.HasAPI(UsdPhysics.CollisionAPI)
        or p.IsA(UsdPhysics.Joint)
        for p in original.Traverse()
    ):
        raise ValueError(
            "Gaussian backgrounds are visual-only; keep interactive/collision objects separate"
        )
    if not hasattr(UsdVol, "ParticleField3DGaussianSplat"):
        raise ValueError("CPU Gaussian cousin edits require standard ParticleField support")
    asset_to_scene = rigid_transform(request.get("asset_to_scene", np.eye(4).tolist()))
    edits = request.get("edits", [])
    if not isinstance(edits, list) or len(edits) > 64:
        raise ValueError("Use at most 64 optional region edits")
    global_color = _color(request.get("color", {}))
    parsed = []
    for row in edits:
        bounds = np.asarray(row["bounds_m"], dtype=float)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or not (bounds[1] > bounds[0]).all()
        ):
            raise ValueError("Region bounds must have ordered scene-frame corners")
        transform, _ = similarity(row.get("transform", np.eye(4).tolist()))
        parsed.append((bounds, transform, _color(row.get("color", {}))))
    maximum = request.get("max_particles", 1_000_000)
    if type(maximum) is not int or not 1 <= maximum <= 5_000_000:
        raise ValueError("Choose a CPU particle budget from 1 to 5000000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layer = original.Flatten()
    path = output / "edited.usda"
    layer.Export(str(path))
    stage = Usd.Stage.Open(str(path))
    fields = [p.GetPath() for p in stage.Traverse() if p.IsA(UsdVol.ParticleField3DGaussianSplat)]
    if not fields:
        raise ValueError(
            "No standard Gaussian ParticleField found; use a configured converter for opaque NuRec, or a mesh background"
        )
    statistics = []
    total = 0
    for prim_path in fields:
        prim = stage.GetPrimAtPath(prim_path)
        field = UsdVol.ParticleField3DGaussianSplat(prim)
        points = np.asarray(field.GetPositionsAttr().Get(), dtype=float)
        scales = np.asarray(field.GetScalesAttr().Get(), dtype=float)
        rotations = field.GetOrientationsAttr().Get()
        opacity = np.asarray(field.GetOpacitiesAttr().Get(), dtype=float)
        degree = field.GetRadianceSphericalHarmonicsDegreeAttr().Get()
        if degree not in (0, 1, 2, 3):
            raise ValueError("Expected Gaussian SH degree 0..3")
        count = (degree + 1) ** 2
        coefficients = np.asarray(
            field.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(), dtype=float
        )
        total += len(points)
        if total > maximum:
            raise ValueError("Background exceeds requested CPU particle budget")
        if (
            points.shape != (len(points), 3)
            or not len(points)
            or scales.shape != points.shape
            or len(rotations) != len(points)
            or opacity.shape != (len(points),)
            or coefficients.shape != (len(points) * count, 3)
            or not all(np.isfinite(v).all() for v in (points, scales, opacity, coefficients))
        ):
            raise ValueError("Incomplete or nonfinite Gaussian arrays")
        coefficients = coefficients.reshape(len(points), count, 3)
        source_matrix = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ).T
        world_matrix = asset_to_scene @ source_matrix
        world = points @ world_matrix[:3, :3].T + world_matrix[:3, 3]
        assigned = np.full(len(points), -1, dtype=int)
        overlaps = 0
        for i, (bounds, _, _) in enumerate(parsed):
            inside = ((world >= bounds[0]) & (world <= bounds[1])).all(axis=1)
            overlaps += int((inside & (assigned >= 0)).sum())
            assigned[inside & (assigned < 0)] = i
        container_path = prim_path.GetParentPath().AppendChild(prim_path.name + "__cousin_regions")
        while stage.GetPrimAtPath(container_path):
            container_path = container_path.GetParentPath().AppendChild(
                container_path.name + "_next"
            )
        UsdGeom.Xform.Define(stage, container_path)
        parent = stage.GetPrimAtPath(prim_path.GetParentPath())
        parent_world = (
            asset_to_scene
            @ np.asarray(
                UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ).T
        )
        groups = []
        for group in [-1, *range(len(parsed))]:
            ids = np.flatnonzero(assigned == group)
            if not len(ids):
                continue
            group_path = container_path.AppendChild("Region_" + str(group + 1))
            wrapper = UsdGeom.Xform.Define(stage, group_path)
            edit = np.eye(4) if group < 0 else parsed[group][1]
            # Keep SH directions and anisotropic ellipsoids in the same frame.
            # Spatial changes are reference transforms, not positions-only PLY edits.
            local = (
                np.eye(4)
                if np.array_equal(edit, np.eye(4))
                else np.linalg.inv(parent_world) @ edit @ parent_world
            )
            wrapper.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
            target_path = group_path.AppendChild("Gaussians")
            if not Sdf.CopySpec(stage.GetRootLayer(), prim_path, stage.GetRootLayer(), target_path):
                raise RuntimeError("Could not preserve the source Gaussian field")
            target = UsdVol.ParticleField3DGaussianSplat(stage.GetPrimAtPath(target_path))
            target.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(points[ids].astype(np.float32)))
            target.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales[ids].astype(np.float32)))
            target.GetOrientationsAttr().Set([rotations[int(i)] for i in ids])
            target.GetOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(opacity[ids].astype(np.float32)))
            colors = coefficients[ids].copy()
            # SH RGB(c) = 0.5 + C0*dc + higher-order terms.
            for matrix, offset in [global_color, *(([parsed[group][2]]) if group >= 0 else [])]:
                colors = colors @ matrix.T
                colors[:, 0] += (matrix @ np.full(3, 0.5) + offset - 0.5) / 0.28209479177387814
            target.GetRadianceSphericalHarmonicsCoefficientsAttr().Set(
                Vt.Vec3fArray.FromNumpy(colors.reshape(-1, 3).astype(np.float32))
            )
            radius = scales[ids].max(axis=1)[:, None] * 3
            target.GetExtentAttr().Set(
                [
                    Gf.Vec3f(*map(float, (points[ids] - radius).min(0))),
                    Gf.Vec3f(*map(float, (points[ids] + radius).max(0))),
                ]
            )
            groups.append({"edit_index": group, "particles": len(ids)})
        stage.RemovePrim(prim_path)
        statistics.append(
            {
                "source_field": str(prim_path),
                "groups": groups,
                "overlapping_selection_votes": overlaps,
            }
        )
    stage.GetRootLayer().Save()
    packaged = output / "background.usdz"
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(path)), str(packaged)):
        raise RuntimeError("Edited Gaussian dependencies could not be packaged")
    _check_stage(packaged)
    if sha256_file(source) != digest:
        raise ValueError("Parent Gaussian background changed during editing")
    # Numeric map only; final rendered appearance belongs to scene-level review.
    return {
        "schema_version": 1,
        "ok": True,
        "operation": "cousin",
        "source_sha256": digest,
        "backend": {"name": "OpenUSD Gaussian region/appearance editor", "version": "1"},
        "background": {
            "path": str(packaged),
            "format": "particle_field",
            "observed": False,
            "transform": asset_to_scene.tolist(),
        },
        "metrics": {"particles": total, "fields": statistics},
        "simulation_verified": False,
        "limits": [
            "Scene-frame region boxes select particle centers; overlapping edits use first-match priority.",
            "This CPU branch grades existing appearance and transforms existing regions; it does not generate unseen geometry or new semantic textures.",
            "Moved regions may reveal holes. Astra must inspect/reconstruct exposed areas and update separate collision/support geometry.",
            "Native SH directions, anisotropic shape, opacity and materials are retained; final target-renderer/scene review is still required.",
        ],
    }
