"""Summarize a USD asset's geometry, metadata, and physics schemas."""

import sys

from pxr import Usd, UsdGeom, UsdPhysics

stage = Usd.Stage.Open(sys.argv[1])
if stage is None:
    raise SystemExit(f"cannot open USD: {sys.argv[1]}")
prims = list(stage.Traverse())
print(f"USD_UP_AXIS={UsdGeom.GetStageUpAxis(stage)}")
print(f"USD_METERS_PER_UNIT={UsdGeom.GetStageMetersPerUnit(stage)}")
print(f"USD_PRIMS={len(prims)} USD_MESHES={sum(p.IsA(UsdGeom.Mesh) for p in prims)}")
for prim in prims:
    name = prim.GetAttribute("userProperties:blender:object_name")
    joint_type = prim.GetAttribute("userProperties:joint_type")
    if joint_type and joint_type.IsValid() or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
        print(
            "USD_PHYSICS",
            prim.GetPath(),
            "blender_name=", name.Get() if name and name.IsValid() else None,
            "joint_type=", joint_type.Get() if joint_type and joint_type.IsValid() else None,
            "rigid=", prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "joint=", prim.IsA(UsdPhysics.Joint),
        )
