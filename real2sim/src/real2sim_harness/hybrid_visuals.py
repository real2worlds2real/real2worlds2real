"""Apply native Newton rigid/cloth trajectories to the retained USD appearance."""

import numpy as np
from pxr import Gf, UsdGeom

from .conversion_audit import named_objects
from .usd_surface_binding import UsdSurfaceBinding


def pose_matrix(pose):
    result = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
    result.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
    return result


class HybridVisuals:
    def __init__(self, folder, stage, report, root="/Asset"):
        self.stage, self.root, self.report = stage, root, report
        self.objects = named_objects(stage)
        self.binding = UsdSurfaceBinding(folder, stage, root=root)
        self.order = sorted((name for name in report["body_ids"] if name != "__base__"),
                            key=lambda name: self.objects[name].GetPath().pathElementCount)

    def update(self, poses, particles):
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.root))
        root.ClearXformOpOrder()
        root.AddTransformOp().Set(pose_matrix(poses[self.report["body_ids"]["__base__"]]))
        for name in self.order:
            prim = self.objects[name]
            scale = np.linalg.norm(np.asarray(self.report["rest_matrices"][name])[:3, :3], axis=0)
            desired = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, scale))) * pose_matrix(poses[self.report["body_ids"][name]])
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(desired * parent_world.GetInverse())
        self.binding.update(particles, refresh_transforms=True)
