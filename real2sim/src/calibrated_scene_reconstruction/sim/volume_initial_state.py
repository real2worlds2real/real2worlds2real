"""Explicit volume material references and separate posed initial geometry.

Author only a stopped initial stage. Never overwrite an active simulation or
replace the original material reference with the current compressed geometry.
The conforming volume_fem route preserves both through native USD reload;
the experimental legacy explicit-state recipe still requires qualification.
"""
import numpy as np


def author(stage,mesh,state,settings,*,backend='legacy'):
    import omni.timeline
    from pxr import PhysxSchema,UsdShade,Vt
    from omni.physx.scripts import deformableUtils
    from calibrated_scene_reconstruction.assets.volume_initial_state import validate
    from calibrated_scene_reconstruction.assets.volume_contact import settings as contact_settings
    if not omni.timeline.get_timeline_interface().is_stopped():
        raise ValueError('Initial volume state authoring requires a stopped native timeline')
    value,report=validate(state)
    required={'youngs_modulus','poissons_ratio','density','dynamic_friction'}
    required.add('linear_damping' if backend=='volume_fem' else 'damping_scale')
    if (not required<=settings.keys() or not np.isfinite([settings[k] for k in required]).all()
            or min(settings[k] for k in ['youngs_modulus','density'])<=0
            or min(settings[k] for k in ['dynamic_friction',('linear_damping' if backend=='volume_fem' else 'damping_scale')])<0
            or not -1<settings['poissons_ratio']<.5):
        raise ValueError('Explicit physical FEM initial-state settings required')
    contact=contact_settings(settings);path=mesh.GetPath();prim=mesh.GetPrim()
    if backend=='volume_fem':
        from pxr import UsdGeom
        from calibrated_scene_reconstruction.assets.volume_initial_state import skin_mapping
        skin=np.asarray(mesh.GetPointsAttr().Get(),float)
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        _,surface_faces=skin_mapping(value,skin,faces)
        if (not np.array_equal(value['simulation_indices'],value['collision_indices'])
                or not np.array_equal(value['simulation_rest_points'],value['collision_rest_points'])
                or not np.array_equal(value['simulation_points'],value['collision_points'])):
            raise ValueError('Explicit native volume FEM requires one conforming material mesh')
        body=UsdGeom.TetMesh.Define(stage,path)
        body.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['simulation_points'].astype('float32')))
        body.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(value['simulation_indices'].astype('int32')))
        ok=deformableUtils.set_physics_volume_deformable_body(stage,path)
        prim=body.GetPrim();ok=ok and prim.ApplyAPI('PhysxBaseDeformableBodyAPI')
        prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(value['simulation_rest_points'].astype('float32')))
        from calibrated_scene_reconstruction.sim.deformable_filter_pose import author as author_filter_pose
        author_filter_pose(prim,value['simulation_rest_points'],value['simulation_indices'],
                           separation_m=4*contact.get('contact_offset',.0003))
        body.GetSurfaceFaceVertexIndicesAttr().Set(Vt.Vec3iArray.FromNumpy(surface_faces.astype('int32')))
        for key,setting in [('selfCollision',True),('solverPositionIterationCount',contact['solver_position_iterations']),
                            ('maxLinearVelocity',10.),('maxDepenetrationVelocity',.1),
                            ('sleepThreshold',0.),('settlingThreshold',0.),('linearDamping',settings['linear_damping'])]:
            prim.GetAttribute('physxDeformableBody:'+key).Set(setting)
        material_path=str(path)+'_material'
        ok=ok and deformableUtils.add_deformable_material(stage,material_path,density=settings['density'],
            youngs_modulus=settings['youngs_modulus'],poissons_ratio=settings['poissons_ratio'],
            dynamic_friction=settings['dynamic_friction'])
        if not ok:raise RuntimeError('Native volume FEM reference authoring failed')
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(contact.get('contact_offset',.0003));collision.CreateRestOffsetAttr(contact.get('rest_offset',0.))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,material_path),
                                                 UsdShade.Tokens.weakerThanDescendants,'physics')
        from calibrated_scene_reconstruction.assets.tetra_boundary import determinants
        reference_volume=float(np.abs(determinants(value['simulation_rest_points'],value['simulation_indices'])).sum()/6)
        mass=reference_volume*settings['density']
        prim.GetAttribute('omniphysics:mass').Set(mass)
        import json
        prim.SetCustomDataByKey('calibrated_scene_reconstruction:volume_contact_profile',json.dumps(contact))
        prim.SetCustomDataByKey('calibrated_scene_reconstruction:volume_reference_mass',json.dumps({'reference_volume_m3':reference_volume,'mass_kg':mass,'density_kg_m3':settings['density'],'direct_mass_readback':False}))
        return report
    if backend!='legacy':raise ValueError('Explicit native volume backend required')
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['skin_rest_points'].astype('float32')))
    # Retain the legacy cook configuration alongside explicit material data.
    # The installed parser rejects the utility's custom-mesh-only recipe.
    # Native reload tests must verify that these settings do not recook away
    # the supplied reference or initial state.
    ok=deformableUtils.add_physx_deformable_body(stage,path,kinematic_enabled=False,
        simulation_hexahedral_resolution=settings.get('resolution',8),
        collision_simplification=False,self_collision=True,
        solver_position_iteration_count=contact['solver_position_iterations'])
    material_path=str(path)+'_material'
    ok=ok and deformableUtils.add_deformable_body_material(stage,material_path,**{k:settings[k] for k in required})
    if not ok:raise RuntimeError('Native explicit volume reference authoring failed')
    api=PhysxSchema.PhysxDeformableBodyAPI(prim)
    base=PhysxSchema.PhysxDeformableAPI(prim)
    base.CreateRestPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['skin_rest_points'].astype('float32')))
    api.CreateSimulationRestPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['simulation_rest_points'].astype('float32')))
    base.CreateSimulationIndicesAttr().Set(value['simulation_indices'].ravel().tolist())
    api.CreateCollisionRestPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['collision_rest_points'].astype('float32')))
    api.CreateCollisionIndicesAttr().Set(value['collision_indices'].ravel().tolist())
    api.CreateSimulationPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['simulation_points'].astype('float32')))
    api.CreateCollisionPointsAttr().Set(Vt.Vec3fArray.FromNumpy(value['collision_points'].astype('float32')))
    if 'contact_offset' in contact:
        collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(contact['contact_offset']);collision.CreateRestOffsetAttr(contact['rest_offset'])
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,material_path),
                                             UsdShade.Tokens.weakerThanDescendants,'physics')
    return report
