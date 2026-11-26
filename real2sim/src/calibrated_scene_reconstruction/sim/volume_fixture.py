"""Native volume backend selection for independent contact fixtures only.

This does not register a production asset backend or approve source contents.
Keep beta and legacy identities explicit; compare their actual cooked surfaces.
"""
import numpy as np


def author(stage,path,points,tetrahedra,*,backend,youngs_modulus,friction):
    from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Vt
    from omni.physx.scripts import deformableUtils
    from calibrated_scene_reconstruction.assets.tetra_boundary import surface
    v=np.asarray(points,float);t=np.asarray(tetrahedra,int)
    material_path=path+'_material'
    if backend=='beta':
        body=UsdGeom.TetMesh.Define(stage,path)
        body.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype('float32')))
        body.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(t.astype('int32')))
        ok=deformableUtils.set_physics_volume_deformable_body(stage,body.GetPath())
        prim=body.GetPrim();ok=ok and prim.ApplyAPI('PhysxBaseDeformableBodyAPI')
        for key,value in [('selfCollision',False),('solverPositionIterationCount',128),
                          ('maxLinearVelocity',10.),('maxDepenetrationVelocity',.1),
                          ('sleepThreshold',0.),('settlingThreshold',0.)]:
            prim.GetAttribute('physxDeformableBody:'+key).Set(value)
        ok=ok and deformableUtils.add_deformable_material(stage,material_path,density=500.,
            youngs_modulus=youngs_modulus,poissons_ratio=.2,dynamic_friction=friction)
    elif backend=='legacy':
        f=surface(v,t)
        body=UsdGeom.Mesh.Define(stage,path)
        body.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype('float32')))
        body.CreateFaceVertexCountsAttr([3]*len(f));body.CreateFaceVertexIndicesAttr(f.ravel().tolist())
        body.CreateSubdivisionSchemeAttr('none')
        prim=body.GetPrim()
        ok=deformableUtils.add_physx_deformable_body(stage,path,kinematic_enabled=False,
            simulation_hexahedral_resolution=8,collision_simplification=False,
            solver_position_iteration_count=128,self_collision=False)
        ok=ok and deformableUtils.add_deformable_body_material(stage,material_path,
            youngs_modulus=youngs_modulus,poissons_ratio=.2,density=500.,dynamic_friction=friction,
            damping_scale=.05)
    else:raise ValueError('Explicit beta or legacy volume fixture required')
    if not ok:raise RuntimeError('Native '+backend+' volume fixture authoring unavailable')
    collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(.0003);collision.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,material_path),
                                             UsdShade.Tokens.weakerThanDescendants,'physics')
    return prim,material_path


class View:
    def __init__(self,owner,prim,backend):
        from pxr import PhysxSchema
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        from calibrated_scene_reconstruction.assets.tetra_boundary import surface
        self.backend=backend
        path=str(prim.GetPath())
        if backend=='beta':
            self.native=owner.create_volume_deformable_body_view(path)
            if self.native is None or self.native.count!=1:raise ValueError('Native beta volume missing')
            self.rest=as_numpy(self.native.get_rest_nodal_positions())[0].copy()
            self.tetrahedra=as_numpy(self.native.get_simulation_element_indices())[0].astype(int)
            collision_points=self.simulation_points()
            collision_tetrahedra=self.tetrahedra
        else:
            self.native=owner.create_soft_body_view(path)
            if self.native is None or self.native.count!=1:raise ValueError('Native legacy volume missing')
            self.rest=np.asarray(PhysxSchema.PhysxDeformableBodyAPI(prim).GetSimulationRestPointsAttr().Get(),float)
            self.tetrahedra=as_numpy(self.native.get_sim_element_indices())[0].astype(int)
            collision_points=as_numpy(self.native.get_nodal_positions())[0].copy()
            collision_tetrahedra=as_numpy(self.native.get_element_indices())[0].astype(int)
        self.faces=surface(collision_points,collision_tetrahedra)
        if self.simulation_points().shape!=self.rest.shape:
            raise ValueError('Actual native volume rest/simulation dimensions differ')

    def simulation_points(self):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        method=(self.native.get_simulation_nodal_positions if self.backend=='beta'
                else self.native.get_sim_nodal_positions)
        return as_numpy(method())[0].copy()

    def collision_points(self):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        return (self.simulation_points() if self.backend=='beta'
                else as_numpy(self.native.get_nodal_positions())[0].copy())
