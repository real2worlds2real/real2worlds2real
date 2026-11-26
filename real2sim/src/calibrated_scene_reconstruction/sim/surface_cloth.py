"""Experimental native surface FEM in Isaac Sim5.1 / OmniPhysics107.3.

No UI preference or persistent setting is changed. This backend uses continuum
material parameters, not particle-spring constants. Native integration uses
uniform node mass with PGS, verified by independent rigid-contact and impulse controls;
area-lumped reference weights must not be used as native node masses.
"""
import numpy as np

from calibrated_scene_reconstruction.assets.cloth_rest import area


def validated_force_solver(solver):
    """The installed 5.1 surface adapter has force/contact controls for PGS.

    TGS passive contact alone is insufficient: the independent impulse and
    rigid collision fixtures fail. Do not infer a mass correction from them.
    """
    if solver!='PGS':
        raise ValueError('Surface FEM nodal-force adapter requires validated PGS; TGS impulse/contact controls are unverified or failed')
    return solver


def lumped_masses(rest, faces, total_mass):
    rest, faces = np.asarray(rest, float), np.asarray(faces)
    surface_area = area(rest, faces)
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError('Positive finite surface mass required')
    tri = rest[faces]
    weights = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)/6
    masses = np.zeros(len(rest))
    for column in range(3):
        np.add.at(masses, faces[:,column], weights)
    if (masses <= 0).any():
        raise ValueError('Every FEM node must own positive rest material area')
    return masses*(total_mass/surface_area)


def material_values(settings, rest, faces):
    required = {'youngs_modulus','poissons_ratio','thickness','flexural_rigidity',
                'elasticity_damping','linear_damping','friction'}
    if not required <= settings.keys() or not np.isfinite([settings[k] for k in required]).all():
        raise ValueError('Surface FEM requires explicit continuum material and damping estimates')
    if (min(settings[k] for k in ['youngs_modulus','thickness','flexural_rigidity']) <= 0
            or not -1 < settings['poissons_ratio'] < .5
            or min(settings[k] for k in ['elasticity_damping','linear_damping','friction']) < 0):
        raise ValueError('Invalid surface FEM constitutive parameters')
    bend_damping=settings.get('bend_damping',settings['elasticity_damping'])
    if not np.isfinite(bend_damping) or bend_damping<0:
        raise ValueError('Bending damping must be a finite nonnegative rate')
    if len({'mass','areal_density'} & settings.keys()) != 1:
        raise ValueError('Surface FEM needs exactly one total mass or areal density')
    surface_area = area(rest,faces)
    kg = settings.get('mass',settings.get('areal_density',0.)*surface_area)
    lumped_masses(rest,faces,kg)
    from calibrated_scene_reconstruction.assets.surface_mass import distribution
    discretization=distribution(rest,faces,kg)
    if not discretization['pass']:
        raise ValueError('Uniform native node masses distort independent sheet masses; split actual bodies or remesh proportional to material area')
    static = settings.get('static_friction',settings['friction'])
    if not np.isfinite(static) or static < settings['friction']:
        raise ValueError('FEM static friction must be finite and at least dynamic friction')
    return {'mass_kg':float(kg),'rest_area_m2':surface_area,
            'native_mass_distribution':discretization,
            'density_kg_m3':float(kg/(surface_area*settings['thickness'])),
            'native_bend_modulus_Pa':float(settings['flexural_rigidity']/settings['thickness']**3),
            'friction_scope':'Isaac5.1 beta supports dynamicFriction only; staticFriction is not independently implemented',
            'static_friction':float(static),
            'bend_damping':float(bend_damping),
            'bend_damping_basis':'explicit' if 'bend_damping' in settings else 'legacy_shared_elasticity_damping'}


def apply_material(stage, surface, settings, rest, faces):
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade
    values = material_values(settings,rest,faces)
    path = str(surface.GetPath())+'_surface_fem_material'
    mat = UsdShade.Material.Define(stage,path)
    if not deformableUtils.add_surface_deformable_material(stage,path,
            density=values['density_kg_m3'],static_friction=values['static_friction'],
            dynamic_friction=settings['friction'],youngs_modulus=settings['youngs_modulus'],
            poissons_ratio=settings['poissons_ratio'],surface_thickness=settings['thickness'],
            surface_bend_stiffness=values['native_bend_modulus_Pa']):
        raise RuntimeError('Isaac5.1 surface FEM material unavailable')
    if not mat.GetPrim().ApplyAPI('PhysxSurfaceDeformableMaterialAPI'):
        raise RuntimeError('Isaac5.1 surface FEM material damping unavailable')
    mat.GetPrim().GetAttribute('physxDeformableMaterial:elasticityDamping').Set(float(settings['elasticity_damping']))
    mat.GetPrim().GetAttribute('physxDeformableMaterial:bendDamping').Set(values['bend_damping'])
    surface.GetAttribute('omniphysics:mass').Set(values['mass_kg'])
    surface.GetAttribute('physxDeformableBody:linearDamping').Set(float(settings['linear_damping']))
    UsdShade.MaterialBindingAPI.Apply(surface).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    return values


def author(stage, mesh, part, settings, *, diagnostic_bend_angles=False):
    from calibrated_scene_reconstruction.assets.cloth_curvature import require_supported
    require_supported(part.get('cloth_rest'),diagnostic=diagnostic_bend_angles)
    from omni.physx.scripts import deformableUtils
    from pxr import PhysxSchema, Vt
    from calibrated_scene_reconstruction.assets.cloth_rest import validate
    rest = part.get('cloth_rest')
    if rest is None:
        raise ValueError('Surface FEM requires explicit intrinsic material coordinates')
    validate(rest,part['V'],part['F'])
    material_values(settings,rest['points'],part['F'])
    contact = settings.get('contact_offset')
    rigid_rest = settings.get('rest_offset')
    filtering = settings.get('self_collision_filter_distance')
    iterations = settings.get('solver_position_iterations',128)
    updates = settings.get('collision_pair_updates',min(16,iterations))
    subiterations = settings.get('collision_iteration_multiplier',2)
    depenetration=settings.get('max_depenetration_velocity',.05)
    if (any(value is None for value in [contact,rigid_rest,filtering])
            or not np.isfinite([contact,rigid_rest,filtering]).all()
            or not 0 < rigid_rest < contact or not 2*rigid_rest <= filtering
            or type(iterations) is not int or not 4 <= iterations <= 255
            or type(updates) is not int or not 1 <= updates <= iterations
            or type(subiterations) is not int or not 1 <= subiterations <= iterations//2):
        raise ValueError('Explicit valid FEM contact offsets/filter and solver budgets required')
    if not np.isfinite(depenetration) or not 0<depenetration<=10:
        raise ValueError('Surface FEM requires a finite positive contact-correction velocity budget')
    prim = mesh.GetPrim()
    if not deformableUtils.set_physics_surface_deformable_body(stage,prim.GetPath()):
        raise RuntimeError('Isaac5.1 surface FEM body unavailable')
    if not prim.ApplyAPI('PhysxSurfaceDeformableBodyAPI'):
        raise RuntimeError('Isaac5.1 surface FEM solver unavailable')
    natural=rest.get('bend_reference',{}).get('points',rest['points'])
    prim.GetAttribute('omniphysics:restShapePoints').Set(Vt.Vec3fArray.FromNumpy(np.asarray(natural,np.float32)))
    from calibrated_scene_reconstruction.sim.deformable_filter_pose import author as author_filter_pose
    author_filter_pose(prim,rest['points'],part['F'],separation_m=4*float(filtering))
    prim.GetAttribute('omniphysics:restBendAnglesDefault').Set('restShapeDefault' if ('bend_reference' in rest or 'bend_angles' in rest) else 'flatDefault')
    if 'bend_angles' in rest:
        from calibrated_scene_reconstruction.assets.cloth_curvature import validate as validate_curvature
        pairs,angles=validate_curvature(rest['bend_angles'],part['F'])
        prim.GetAttribute('omniphysics:restAdjTriPairs').Set(Vt.Vec2iArray.FromNumpy(np.ascontiguousarray(pairs,np.int32)))
        prim.GetAttribute('omniphysics:restBendAngles').Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(angles,np.float32)))
    for name,value in [('selfCollision',True),('solverPositionIterationCount',iterations),
                       ('selfCollisionFilterDistance',float(filtering)),('enableSpeculativeCCD',True),
                       ('collisionPairUpdateFrequency',updates),('collisionIterationMultiplier',subiterations),
                       ('maxLinearVelocity',float(settings.get('max_linear_velocity',10.))),('settlingDamping',0.),
                       ('maxDepenetrationVelocity',float(depenetration)),
                       ('settlingThreshold',0.),('sleepThreshold',0.)]:
        prim.GetAttribute('physxDeformableBody:'+name).Set(value)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr(float(contact));collision.CreateRestOffsetAttr(float(rigid_rest))
    apply_material(stage,prim,settings,rest['points'],part['F'])
    prim.SetCustomDataByKey('calibrated_scene_reconstruction:cloth_solver','surface_fem')
    # Dense folds generate many self-contact pairs. Reserve by actual scene
    # topology, not video identity; runtime error monitoring remains mandatory.
    from pxr import UsdPhysics
    faces=sum(len(p.GetAttribute('faceVertexIndices').Get())//3 for p in stage.Traverse()
              if p.HasAPI('OmniPhysicsSurfaceDeformableSimAPI'))
    capacity=1<<max(20,(32*faces-1).bit_length())
    if capacity>1<<24:
        raise ValueError('Surface contact capacity exceeds validated budget; revise the numerical mesh')
    scenes=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)!=1:raise ValueError('Surface FEM requires one explicit physics scene')
    scene=PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    scene.CreateGpuMaxDeformableSurfaceContactsAttr(max(capacity,int(scene.GetGpuMaxDeformableSurfaceContactsAttr().Get())))
    prim.SetCustomDataByKey('calibrated_scene_reconstruction:surface_contact_capacity',capacity)


class SurfaceView:
    """Source-indexed nodal state and bounded-impulse compatibility adapter.

    Native rest coordinates are in world space. Cooking may reorder nodes; a
    one-to-one rest mapping and identical triangles are required, never assumed.
    Masses are explicitly derived, rather than falsely reported as read back.
    """
    mass_source = 'Native uniform node-mass model validated with PGS rigid-contact and impulse controls; total mass authored, not a direct nodal mass readback; other solvers unverified'
    backend = 'surface_fem'

    def __init__(self, stage, path, part):
        import omni.physics.tensors as tensors
        import torch
        from pxr import UsdGeom
        from scipy.spatial import cKDTree
        self.owner = tensors.create_simulation_view('torch')
        self.owner.set_subspace_roots('/')
        self.view = self.owner.create_surface_deformable_body_view(path)
        if self.view is None:
            raise ValueError('Native surface body unavailable after cooking; inspect actual PhysX initialization errors before using this reference')
        if self.view.count != 1:
            raise ValueError('Expected one actual native surface FEM body')
        self.prim = stage.GetPrimAtPath(path)
        self.stage = stage
        self.rest = np.asarray(part['cloth_rest']['points'],float)
        self.faces = np.asarray(part['F'])
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.prim))
        if not np.allclose(transform[:3,:3]@transform[:3,:3].T,np.eye(3),atol=1e-6):
            raise ValueError('Surface FEM material coordinates require unscaled metric transforms')
        natural=np.asarray(part['cloth_rest'].get('bend_reference',{}).get('points',self.rest),float)
        expected = natural@transform[:3,:3]+transform[3,:3]
        native = self.view.get_rest_nodal_positions()[0].cpu().numpy()
        tolerance=max(2e-7,8*np.finfo(np.float32).eps*max(1.,abs(expected).max()))
        native_faces = self.view.get_simulation_element_indices()[0].cpu().numpy()
        # Ideal zero-thickness natural folds can have coincident rest points.
        # Identical *indexed* topology and ordered rest coordinates establish
        # identity without guessing a nearest neighbour among those points.
        direct=(native.shape==expected.shape and np.array_equal(native_faces,self.faces)
                and np.linalg.norm(native-expected,axis=1).max()<=tolerance)
        if direct:
            distance=np.linalg.norm(native-expected,axis=1);indices=np.arange(len(expected))
            self.correspondence='Identical indexed simulation topology and ordered natural coordinates'
        else:
            distance,indices = cKDTree(expected).query(native)
            self.correspondence='Unique natural-coordinate matching and verified indexed topology'
        if (len(indices) != len(expected) or len(set(indices)) != len(expected)
                or distance.max() > tolerance):
            raise ValueError('FEM cooking has no exact source-material node correspondence')
        if set(map(tuple,np.sort(indices[native_faces],axis=1))) != set(map(tuple,np.sort(self.faces,axis=1))):
            raise ValueError('FEM cooking changed source material topology')
        device = self.view.get_simulation_nodal_positions().device
        self.order = torch.as_tensor(np.argsort(indices),dtype=torch.long,device=device)
        self.inverse = torch.as_tensor(indices,dtype=torch.long,device=device)
        self.body_index = torch.zeros(1,dtype=torch.int32,device=device)
        self.mass_weights = np.full(len(self.rest),1./len(self.rest))

    def get_world_positions(self):
        return self.view.get_simulation_nodal_positions()[:,self.order]

    def get_velocities(self):
        return self.view.get_simulation_nodal_velocities()[:,self.order]

    def restore_setup_state(self,positions,velocities):
        """Independent trial initialization, not a force or motion adapter.

        The task context prohibits this operation during an active assay.
        Callers must restore a real captured state or its common rigid setup
        transform; no material rest shape is modified.
        """
        import torch
        device=self.view.get_simulation_nodal_positions().device
        q=np.asarray(positions);v=np.asarray(velocities)
        if q.shape!=(1,len(self.rest),3) or v.shape!=q.shape or not np.isfinite([q,v]).all():
            raise ValueError('Finite source-indexed native positions and velocities required')
        points=torch.as_tensor(q,dtype=torch.float32,device=device)[:,self.inverse].contiguous()
        speed=torch.as_tensor(v,dtype=torch.float32,device=device)[:,self.inverse].contiguous()
        self.view.set_simulation_nodal_positions(points,self.body_index)
        self.view.set_simulation_nodal_velocities(speed,self.body_index)

    def require_validated_force_profile(self, *, diagnostic_attachments=False):
        from pxr import PhysxSchema,UsdPhysics
        scenes=[p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes)!=1:raise ValueError('Surface nodal-force validation requires one actual physics scene')
        attached = any('Attachment' in p.GetTypeName()
                       and str(self.prim.GetPath()) in {str(x) for x in p.GetRelationship('omniphysics:src0').GetTargets()}
                       for p in self.stage.Traverse())
        if attached and not diagnostic_attachments:
            raise ValueError('Dynamic attachments need independent coupled nodal-force qualification; the free-surface PGS control does not establish it. Drive the qualified rigid connection or run an explicit diagnostic.')
        self._diagnostic_attachment_forces = bool(diagnostic_attachments)
        solver = validated_force_solver(str(PhysxSchema.PhysxSceneAPI(scenes[0]).GetSolverTypeAttr().Get()))
        self._force_profile_validated = True
        return solver

    def set_velocities(self, value):
        if not getattr(self, '_force_profile_validated', False):
            self.require_validated_force_profile(diagnostic_attachments=getattr(self, '_diagnostic_attachment_forces', False))
        import omni.physx
        import omni.usd
        from pxr import PhysicsSchemaTools
        self.view.set_simulation_nodal_velocities(value[:,self.inverse].contiguous(),self.body_index)
        omni.physx.get_physx_simulation_interface().wake_up(
            omni.usd.get_context().get_stage_id(),PhysicsSchemaTools.sdfPathToInt(self.prim.GetPath()))

    def get_particle_masses(self):
        """Compatibility name; uniform native model, with authored total mass."""
        import torch
        kg = self.prim.GetAttribute('omniphysics:mass').Get()
        if kg is None or not np.isfinite(kg) or kg <= 0:
            raise ValueError('Native authored FEM total mass unavailable')
        return torch.as_tensor((self.mass_weights*kg)[None],dtype=torch.float32,device=self.order.device)
