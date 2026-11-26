"""Author explicit auxiliary material cells on their existing rigid owner.

No new body, mass, interior filler, joint or visual geometry is introduced.
The cell producer and caller must bound source-surface error and validate the
native contact pair. This does not replace a precision mechanism collider.
"""
import numpy as np


def add(stage, body_path, cells, report, *, name='ParticleBoundary', material=None):
    from pxr import Gf, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    from calibrated_scene_reconstruction.assets.collision_groups import normalize
    from calibrated_scene_reconstruction.sim.convex_cooking import apply_hull
    from calibrated_scene_reconstruction.sim.physics import bind_colliders
    from calibrated_scene_reconstruction.sim.usd import add_mesh

    body = stage.GetPrimAtPath(body_path)
    root_path = str(body.GetPath())+'/'+name if body else ''
    if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI) or stage.GetPrimAtPath(root_path):
        raise ValueError('Auxiliary cells need a distinct path on the actual rigid owner')
    bound = report.get('source_boundary_error_bound_m')
    if not cells or bound is None or not np.isfinite(bound) or bound < 0:
        raise ValueError('Source material cells and an explicit geometry error bound required')
    root = UsdGeom.Xform.Define(stage, root_path)
    root.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:source_boundary_error_bound_m', float(bound))
    root.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:auxiliary_source_material_cells', True)
    for i, cell in enumerate(cells):
        v, centre, basis, scale = normalize(cell['V'], principal_axes=True, scale_mode='axis_extents')
        collision = add_mesh(stage, root_path+'/cell_'+str(i), v, cell['F'])
        xf = UsdGeom.Xformable(collision)
        xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
        q = Rotation.from_matrix(basis).as_quat()
        xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
        collision.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr('convexHull')
        apply_hull(collision.GetPrim(), {'hull_vertex_limit': 64, 'minimum_thickness_m': 1e-7}, scale)
    if material:
        bind_colliders(stage, [root_path], material)
    return root
