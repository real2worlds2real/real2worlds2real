"""Reusable original-material rigid replay for motion and arbitrary local views."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def load_player(folder):
    folder = Path(folder).resolve()
    native = json.loads((folder / "validation.json").read_text())
    for name, digest in native.get("evidence_sha256", {}).items():
        path = (folder / name).resolve()
        if (
            not path.is_relative_to(folder)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Native contact request or trajectory changed")
    request = json.loads((folder / "request.json").read_text())
    digest = request.get("source_scene_sha256")
    if not digest or hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest() != digest:
        raise ValueError("Loaded contact source differs from the collision export")
    cache = np.load(folder / "motion.npz", allow_pickle=False)
    poses = cache["body_poses"]
    if len(poses) != native["frames"] or not np.isfinite(poses).all():
        raise ValueError("Native contact trajectory is incomplete or nonfinite")
    visual_map = {row["body"]: row["visuals"] for row in request["source_bindings"]}
    bindings = []
    objects = []
    deps = bpy.context.evaluated_depsgraph_get()
    for i, body in enumerate(request["bodies"]):
        for spec in visual_map[body["id"]]:
            obj = bpy.data.objects.get(spec["object"])
            if obj is None or obj in objects:
                raise ValueError("Native source visuals are missing or multiply bound")
            local = Matrix(spec["body_local_matrix"])
            evaluated = obj.evaluated_get(deps)
            baked = bpy.data.meshes.new_from_object(
                evaluated, preserve_all_data_layers=True, depsgraph=deps
            )
            points = np.asarray([list(local @ v.co) for v in baked.vertices])
            lo, hi = points.min(axis=0), points.max(axis=0)
            corners = np.asarray(list(itertools.product(*zip(lo, hi, strict=True))))
            world = evaluated.matrix_world.copy()
            obj.modifiers.clear()
            obj.constraints.clear()
            obj.parent = None
            obj.data = baked
            obj.matrix_world = world
            objects.append(obj)
            bindings.append((i, obj, local, corners))

    def apply(frame):
        if not 0 <= frame < len(poses):
            raise ValueError("Native contact frame outside retained states")
        for i, obj, local, _ in bindings:
            q = poses[frame, i]
            obj.matrix_world = (
                Matrix.Translation(Vector(q[:3]))
                @ Quaternion((q[6], *q[3:6])).to_matrix().to_4x4()
                @ local
            )
        bpy.context.view_layer.update()

    apply(0)
    return apply, objects, bindings
