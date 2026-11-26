"""Author a Blender table's visual parts and its identical static collision surface."""
import numpy as np
from pxr import UsdGeom, UsdPhysics
from .usd import add_mesh, color_material, textured_material


def add_support(stage, path, model, *, visual_only=False):
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:model_sha256', model['model_sha256'])
    a = np.radians(model['yaw_deg'])
    R = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    offset = np.r_[model['xy'], 0.]
    for i, part in enumerate(model['parts']):
        look = f'{path}/looks/part_{i}'
        if part.get('texture'):
            material = textured_material(stage, look, part['texture'], roughness=part['roughness'],
                                         metallic=part['metallic'], tint=part['color'],wrap=part['wrap'])
        else:
            material = color_material(stage, look, part['color'],
                                      roughness=part['roughness'], metallic=part['metallic'])
        add_mesh(stage, f'{path}/visual_{i}', part['vertices']@R.T+offset, part['faces'],
                 UV=part.get('uv'), material=material)
    if visual_only:
        root.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:static_appearance_only', True)
        return root
    vertices,faces=model['vertices'],model['faces']
    if model.get('collision_solids') is not None:
        from pxr import Gf
        from scipy.spatial.transform import Rotation
        from calibrated_scene_reconstruction.assets.collision_groups import normalize
        UsdGeom.Xform.Define(stage,f'{path}/collision')
        for i,solid in enumerate(model['collision_solids']):
            world=solid['V']@R.T+offset
            v,centre,basis,scale=normalize(world,principal_axes=True,scale_mode='axis_extents')
            collision=add_mesh(stage,f'{path}/collision/solid_{i}',v,solid['F'])
            xf=UsdGeom.Xformable(collision);xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
            q=Rotation.from_matrix(basis).as_quat()
            xf.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
            xf.AddScaleOp().Set(Gf.Vec3f(*map(float,np.broadcast_to(scale,(3,)))))
            collision.CreatePurposeAttr('guide')
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr('convexHull')
            from .convex_cooking import apply_hull
            apply_hull(collision.GetPrim(),{'hull_vertex_limit':64,'minimum_thickness_m':1e-7},scale)
            collision.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:source_solid',solid['source_name'])
            if 'source_boundary_error_bound_m' in solid:
                collision.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:source_boundary_error_bound_m',solid['source_boundary_error_bound_m'])
        return root
    if model.get('collision_max_edge_m') is not None:
        from calibrated_scene_reconstruction.assets.collision_tessellation import subdivide
        vertices,faces=subdivide(vertices,faces,model['collision_max_edge_m'])
    collision = add_mesh(stage, f'{path}/collision', vertices@R.T+offset, faces)
    if model.get('collision_max_edge_m') is not None:
        collision.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:collision_max_edge_m',model['collision_max_edge_m'])
    collision.CreatePurposeAttr('guide')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr('none')
    return root
