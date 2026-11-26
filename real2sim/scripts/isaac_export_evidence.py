"""Retain the actual simulator state with portable local USD references."""

import hashlib
import os
import shutil
from pathlib import Path

from pxr import Gf, UsdGeom, UsdUtils


def bake_body_poses(stage, paths, poses):
    cache = UsdGeom.XformCache()
    desired = {}
    for path, pose in zip(paths, poses, strict=True):
        original = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
        transform = Gf.Transform()
        transform.SetScale(original.GetScale())
        transform.SetRotation(Gf.Rotation(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))))
        transform.SetTranslation(Gf.Vec3d(*map(float, pose[:3])))
        desired[path] = transform.GetMatrix()
    for path in sorted(desired, key=lambda item: item.count("/")):
        prim = stage.GetPrimAtPath(path)
        parent_world = cache.GetLocalToWorldTransform(prim.GetParent())
        local = desired[path] * parent_world.GetInverse()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(local)
        cache.Clear()


def save_stage(stage, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = stage.Flatten()

    def portable(asset):
        path = Path(asset)
        if path.is_absolute() and path.is_file():
            if not path.is_relative_to(output.parent):
                dependencies = output.parent / "evidence_dependencies"
                dependencies.mkdir(exist_ok=True)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()[:12]
                target = dependencies / f"{digest}_{path.name}"
                if not target.is_file():
                    shutil.copy2(path, target)
                path = target
            return os.path.relpath(path, output.parent)
        return asset

    UsdUtils.ModifyAssetPaths(layer, portable)
    if not layer.Export(str(output)):
        raise RuntimeError(f"Could not export actual simulator state to {output}")
    return output
