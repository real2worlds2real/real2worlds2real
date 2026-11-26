"""Replay actual scene states onto the retained visual hierarchy.

The renderer imports this module without Newton or Warp. No physics is computed
here; transformations and control-surface mappings consume the saved solver data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Tf, Usd, UsdGeom

from .hybrid_visuals import pose_matrix
from .usd_surface_binding import UsdSurfaceBinding


class SceneNewtonVisuals:
    def __init__(self, folder, stage, bindings, root="/World/Scene"):
        self.stage, self.assets, self.fixtures = stage, [], []
        for name, record in bindings["assets"].items():
            path = f"{root}/Objects/{name}/Asset"
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Missing scene participant visual: " + name)
            objects = {}
            for child in Usd.PrimRange(prim):
                attr = child.GetAttribute("userProperties:blender:object_name")
                if attr and attr.Get():
                    objects[str(attr.Get())] = child
            native = record.get("native_representation")
            order = sorted(
                (key for key in record["body_ids"] if key != "__base__"),
                key=lambda key: objects[key].GetPath().pathElementCount,
            ) if not native else []
            surface = None
            if record["particle_end"] > record["particle_start"]:
                surface = UsdSurfaceBinding(Path(folder) / "assets" / name, stage, root=path)
            self.assets.append((record, prim, objects, order, surface))
            for index, probe in enumerate(record.get("native_test_probes", [])):
                path = f"{root}/TestFixtures/{Tf.MakeValidIdentifier(name)}_{index}"
                if probe["shape"] == "sphere":
                    geom = UsdGeom.Sphere.Define(stage, path)
                    geom.CreateRadiusAttr(probe["radius_m"])
                else:
                    geom = UsdGeom.Cube.Define(stage, path)
                    geom.CreateSizeAttr(1.)
                geom.CreateDisplayColorAttr([Gf.Vec3f(.19,.28,.32)])
                self.fixtures.append((geom.GetPrim(), probe, record["native_probe_ids"][probe["id"]],record.get("native_action")))

    def update(self, bodies, particles, *, time=None):
        for record, prim, objects, order, surface in self.assets:
            if record.get("native_representation") == "rigid_contact":
                placement = Gf.Matrix4d(*np.asarray(record["transform"]).T.ravel().tolist())
                for binding in record["native_source_bindings"]:
                    body = record["body_ids"][binding["body"]]
                    pose = (record["native_static_poses"][binding["body"]] if body < 0 else bodies[body])
                    desired_body = pose_matrix(pose) * placement if body < 0 else pose_matrix(pose)
                    for visual in binding["visuals"]:
                        child = objects[visual["object"]]
                        local = Gf.Matrix4d(*np.asarray(visual["body_local_matrix"]).T.ravel().tolist())
                        desired = local * desired_body
                        parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(child.GetParent())
                        xf = UsdGeom.Xformable(child)
                        xf.ClearXformOpOrder()
                        xf.AddTransformOp().Set(desired * parent_world.GetInverse())
            elif record["body_ids"]:
                desired = pose_matrix(bodies[record["body_ids"]["__base__"]])
                parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTransformOp().Set(desired * parent_world.GetInverse())
                for name in order:
                    child = objects[name]
                    scale = np.linalg.norm(
                        np.asarray(record["rest_matrices"][name])[:3, :3], axis=0
                    )
                    desired = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, scale))) * pose_matrix(
                        bodies[record["body_ids"][name]]
                    )
                    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(child.GetParent())
                    xf = UsdGeom.Xformable(child)
                    xf.ClearXformOpOrder()
                    xf.AddTransformOp().Set(desired * parent_world.GetInverse())
            if surface:
                # Binding offsets were constructed in asset coordinates. Rotate
                # the actual world-space control nodes back before deformation,
                # then transform the detailed result into the placed scene.
                placement = np.asarray(record["transform"])
                world = particles[record["particle_start"] : record["particle_end"]]
                local = (world - placement[:3, 3]) @ placement[:3, :3]
                surface.update(local, refresh_transforms=True, world_from_binding=placement)
        for prim, probe, body, action in self.fixtures:
            desired = pose_matrix(bodies[body])
            if probe["shape"] == "box":
                desired = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float,probe["size_m"]))) * desired
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(desired*parent_world.GetInverse())
            if time is not None and action:
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible" if time<action["start_seconds"] else "inherited")
