"""Metric solid volume from authored physical geometry, including USD scale."""
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics


def rigid_volume(prim):
    stage=prim.GetStage();cache=UsdGeom.XformCache()
    source=[p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)
            and p.GetCustomDataByKey('calibrated_scene_reconstruction:mass_source_surface') is True]
    meshes=source or [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)
                     and p.HasAPI(UsdPhysics.CollisionAPI)]
    volume=0.;units=UsdGeom.GetStageMetersPerUnit(stage)
    for p in meshes:
        mesh=UsdGeom.Mesh(p);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if np.any(counts!=3):raise ValueError('Density inference requires triangulated physical solids')
        solid=trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=True)
        if not solid.is_volume:raise ValueError('Density inference requires a closed outward physical solid')
        transform=np.asarray(cache.GetLocalToWorldTransform(p),float)
        volume+=float(solid.volume)*abs(float(np.linalg.det(transform[:3,:3])))*units**3
    if not np.isfinite(volume) or volume<=0:raise ValueError('Positive metric physical volume required')
    return volume
