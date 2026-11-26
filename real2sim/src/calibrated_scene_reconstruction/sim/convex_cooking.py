"""Explicit convex cooking controls; fidelity must be measured after cooking."""
import math
import numpy as np

_FIELDS={'hull_vertex_limit':('HullVertexLimit',8,64),'max_convex_hulls':('MaxConvexHulls',1,1024),
         'voxel_resolution':('VoxelResolution',10000,10000000)}




def apply_hull(prim, settings, coordinate_scale=1.):
    """Keep metric thickness fixed when source coordinates are normalized."""
    from pxr import PhysxSchema, UsdGeom
    values=validate_hull(settings);unit=UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    scales=np.asarray(coordinate_scale,float)
    if scales.shape not in [(),(3,)] or not np.isfinite(scales).all() or np.any(scales<=0) or not math.isfinite(unit) or unit<=0:
        raise ValueError('Positive metric unit and collision coordinate scale required')
    api=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
    api.CreateHullVertexLimitAttr(values['hull_vertex_limit'])
    # For anisotropic conditioning this bounds any introduced thickness in
    # world metres; it does not claim that cooking preserves the source hull.
    api.CreateMinThicknessAttr(values['minimum_thickness_m']/(unit*float(scales.max())))
    return values


def sdf_from_declared_hull(prim,resolution):
    """Represent the same intended source convex solid with a closed SDF mesh.

    Imported CAD may declare a convex hull on an open triangle soup. Preserve
    that declared point-cloud hull; never apply this conversion to a concave
    task mesh. Native cooking, fidelity and contact qualification still apply.
    """
    from pxr import PhysxSchema,UsdGeom,UsdPhysics,Vt
    from calibrated_scene_reconstruction.assets.convex_source_solids import _hull
    if (type(resolution)is not int or not 64<=resolution<=2048 or not prim.IsA(UsdGeom.Mesh)
            or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull'):
        raise ValueError('Declared source convex collider and explicit supported SDF resolution required')
    mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),float);hull=_hull(v)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(hull.vertices,dtype='float32')))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]*len(hull.faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(np.asarray(hull.faces,int).ravel().tolist()))
    UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr('sdf')
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(resolution)
    prim.SetCustomDataByKey('calibrated_scene_reconstruction:declared_convex_source_sdf',True)
    return {'approximation':'sdf','resolution':resolution,'source_interpretation':'declared_convex_hull',
        'source_vertices':len(v),'closed_vertices':len(hull.vertices),'physical_geometry_changed':False}




def apply(prim,settings):
    from pxr import PhysxSchema,UsdGeom
    values=validate(settings);unit=UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    if not math.isfinite(unit) or unit<=0:raise ValueError('Positive metric stage units required')
    api=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    for key,(name,_,_) in _FIELDS.items():getattr(api,'Create'+name+'Attr')(values[key])
    api.CreateMinThicknessAttr(values['minimum_thickness_m']/unit)
    api.CreateErrorPercentageAttr(values['error_percentage']);api.CreateShrinkWrapAttr(values['shrink_wrap'])
    return values


from calibrated_scene_reconstruction.assets.convex_cooking import validate_hull, validate  # compatibility exports
