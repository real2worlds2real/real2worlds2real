"""Independent native PhysX fluid instances with explicit material hypotheses.

Seeds are metric world positions supplied by reconstruction, never a preset
puddle shape. Each instance owns its system, material and mass ledger. PhysX
PBD coefficients are effective solver parameters, not SI viscosity/tension.
The caller must validate source geometry, material sensitivity and contacts.
"""
import numpy as np


def validate_seed(points, specification):
    from scipy.spatial import cKDTree

    p = np.asarray(points, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 8 or not np.isfinite(p).all():
        raise ValueError('Fluid requires at least eight finite metric seed positions')
    allowed = {'basis', 'spacing_m', 'density_kg_m3', 'pbd', 'iterations',
               'max_velocity_m_s', 'max_depenetration_velocity_m_s', 'isosurface',
               'solver_volume_calibration', 'isosurface_options'}
    if not isinstance(specification, dict) or set(specification) - allowed:
        raise ValueError('Explicit supported fluid specification required')
    if not isinstance(specification.get('basis'), str) or not specification['basis'].strip():
        raise ValueError('Fluid requires a disclosed source/material basis')
    spacing = specification.get('spacing_m')
    density = specification.get('density_kg_m3')
    if (not isinstance(spacing, (float, int)) or not np.isfinite(spacing)
            or not 1e-5 <= spacing <= .05 or not isinstance(density, (float, int))
            or not np.isfinite(density) or not 1 <= density <= 20000):
        raise ValueError('Finite metric spacing and SI density required')
    material = specification.get('pbd')
    coefficients = {'friction', 'cohesion', 'surface_tension', 'viscosity', 'adhesion', 'damping'}
    if not isinstance(material, dict) or set(material) != coefficients:
        raise ValueError('Declare every effective PBD coefficient; do not silently map SI values')
    if any(not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0 for v in material.values()):
        raise ValueError('Finite nonnegative effective PBD coefficients required')
    iterations = specification.get('iterations', 32)
    if type(iterations) is not int or not 1 <= iterations <= 256:
        raise ValueError('Fluid iterations must be an integer in 1..256')
    for key in ('max_velocity_m_s', 'max_depenetration_velocity_m_s'):
        value = specification.get(key, 2.)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Positive finite numerical velocity limits required')
    if 'isosurface' in specification and type(specification['isosurface']) is not bool:
        raise ValueError('Isosurface setting must be boolean')
    surface=specification.get('isosurface_options',{})
    if not isinstance(surface,dict) or set(surface)-{'grid_spacing_ratio','surface_distance_ratio',
                                                   'mesh_smoothing_passes','normal_smoothing_passes'}:
        raise ValueError('Only explicit bounded isosurface resolution/distance options are supported')
    for key,limits in [('grid_spacing_ratio',(.1,1.)),('surface_distance_ratio',(.5,1.5))]:
        if key in surface and (not isinstance(surface[key],(int,float)) or not np.isfinite(surface[key])
                               or not limits[0]<=surface[key]<=limits[1]):
            raise ValueError('Isosurface support must remain bounded by native particle resolution')
    # PhysX 107.3's authored schema requires distance < 2.5 * grid spacing.
    # Checking each option independently admits unsupported combinations.
    if surface.get('surface_distance_ratio',.8) >= 2.5*surface.get('grid_spacing_ratio',.7):
        raise ValueError('Isosurface distance must be less than 2.5 times grid spacing')
    for key, maximum in [('mesh_smoothing_passes',8),('normal_smoothing_passes',16)]:
        if key in surface and (type(surface[key]) is not int or not 0 <= surface[key] <= maximum):
            raise ValueError('Isosurface smoothing needs bounded integer pass counts')
    separation = cKDTree(p).query(p, k=2)[0][:, 1]
    if separation.min() < spacing * .25:
        raise ValueError('Duplicate or excessively overlapping initial fluid particles')
    factor=1.
    calibration=specification.get('solver_volume_calibration')
    if calibration is not None:
        import hashlib
        import json
        from calibrated_scene_reconstruction.core.paths import resolve
        if not isinstance(calibration,dict) or set(calibration)!={'evidence','sha256'}:
            raise ValueError('Solver volume calibration needs an exact independent evidence artifact')
        path=resolve(calibration['evidence'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=calibration['sha256']:
            raise ValueError('Solver volume calibration evidence changed or is missing')
        report=json.loads(path.read_text())
        factor=report.get('particle_volume_scale')
        interval=report.get('validated_spacing_interval_m',[])
        if (report.get('calibration_pass') is not True or not report.get('basis')
                or not isinstance(factor,(int,float)) or not np.isfinite(factor) or not 0<factor<=2
                or len(interval)!=2 or not np.isfinite(interval).all() or not interval[0]<=spacing<=interval[1]):
            raise ValueError('Successful independent volume calibration at the requested resolution required')
    volume=float(factor*spacing**3)
    mass = float(density * volume)
    return p, {'particles': len(p), 'particle_mass_kg': mass,
               'initial_mass_kg': len(p) * mass, 'initial_volume_m3': len(p) * volume,
               'nominal_seed_cell_volume_m3':len(p)*spacing**3,'particle_volume_scale':float(factor),
               'solver_volume_calibration':calibration,
               'minimum_seed_separation_m': float(separation.min()),
               'volume_basis': ('Independent native volume calibration times spacing cubed'
                                if calibration else 'Uncalibrated spacing-cubed material cell per seed particle')}


class NativeFluid:
    """Native motion plus an optional finite-capacity, contact-only sink.

    A sink implements points(), velocities(), masses(), faces,
    apply_contact_velocities(), and accept_water(mass, momentum). Its native
    mass must increase by the removed liquid mass. There is no region eraser,
    prescribed liquid trajectory, rigid liquid surrogate or hidden mass loss.
    """

    def __init__(self, stage, path, scene_path, points, specification, *, visual_material=None):
        from pxr import PhysxSchema, Sdf, UsdGeom, UsdShade, Vt
        from omni.physx.scripts import particleUtils, physicsUtils

        p, self.initial = validate_seed(points, specification)
        root = Sdf.Path(path)
        if (not root.IsAbsolutePath() or not root.IsPrimPath() or str(root) == '/'
                or stage.GetPrimAtPath(root)):
            raise ValueError('Fluid needs a distinct unused absolute prim path')
        if not stage.GetPrimAtPath(scene_path):
            raise ValueError('Fluid needs an existing native physics scene')
        self.stage, self.root = stage, root
        self.specification = dict(specification)
        self.spacing = float(specification['spacing_m'])
        self.rest = self.spacing / 2
        self.contact = self.rest / (.99 * .6)
        self.particle_mass = self.initial['particle_mass_kg']
        self.initial_count = len(p)
        self.absorbed_count = 0
        self.empty = False
        self.events = []
        self.clock = 0.
        self.sink = None
        UsdGeom.Xform.Define(stage, root)
        self.system_path = root.AppendChild('System')
        self.material_path = root.AppendChild('PhysicsMaterial')
        mat = UsdShade.Material.Define(stage, self.material_path)
        particleUtils.AddPBDMaterialWater(mat.GetPrim())
        api = PhysxSchema.PhysxPBDMaterialAPI.Get(stage, self.material_path)
        attributes = {'friction': 'Friction', 'cohesion': 'Cohesion', 'viscosity': 'Viscosity',
                      'surface_tension': 'SurfaceTension', 'adhesion': 'Adhesion', 'damping': 'Damping'}
        for name, attribute in attributes.items():
            getattr(api, 'Create' + attribute + 'Attr')(float(specification['pbd'][name]))
        system = particleUtils.add_physx_particle_system(
            stage, self.system_path, simulation_owner=scene_path,
            contact_offset=self.contact, rest_offset=self.rest,
            particle_contact_offset=self.contact, solid_rest_offset=self.rest,
            fluid_rest_offset=self.rest, solver_position_iterations=specification.get('iterations', 32),
            max_velocity=specification.get('max_velocity_m_s', 2.),
            max_depenetration_velocity=specification.get('max_depenetration_velocity_m_s', 2.),
            enable_ccd=True)
        physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), self.material_path)
        self.particles = particleUtils.add_physx_particleset_points(
            stage, root.AppendChild('Particles'), Vt.Vec3fArray.FromNumpy(p),
            Vt.Vec3fArray.FromNumpy(np.zeros_like(p)), [2 * self.rest] * len(p), self.system_path,
            self_collision=True, fluid=True, particle_group=0,
            particle_mass=self.particle_mass, density=0.)
        self.particles.GetPrim().CreateAttribute('physxParticle:maxParticles', Sdf.ValueTypeNames.Int).Set(len(p))
        self.pset = PhysxSchema.PhysxParticleSetAPI(self.particles.GetPrim())
        if visual_material is not None:
            UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(visual_material)
            UsdShade.MaterialBindingAPI.Apply(self.particles.GetPrim()).Bind(visual_material)
        if specification.get('isosurface', True):
            surface=specification.get('isosurface_options',{})
            particleUtils.add_physx_particle_isosurface(
                stage, self.system_path, enabled=True, grid_spacing=self.spacing * surface.get('grid_spacing_ratio',.7),
                surface_distance=self.spacing * surface.get('surface_distance_ratio',.8), max_vertices=400000, max_triangles=800000,
                num_mesh_smoothing_passes=surface.get('mesh_smoothing_passes',2),
                num_mesh_normal_smoothing_passes=surface.get('normal_smoothing_passes',2))
            self.particles.MakeInvisible()

    def state(self):
        if self.empty:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        import omni.physx

        omni.physx.get_physx_interface().update_transformations(False, True, True, False)
        sim = self.pset.GetSimulationPointsAttr()
        attr = sim if sim.HasAuthoredValue() else self.particles.GetPointsAttr()
        points = np.asarray(attr.Get(), np.float32).reshape(-1, 3).copy()
        velocity = np.asarray(self.particles.GetVelocitiesAttr().Get(), np.float32).reshape(-1, 3).copy()
        if (points.shape != velocity.shape or not np.isfinite(points).all()
                or not np.isfinite(velocity).all() or len(points) + self.absorbed_count != self.initial_count):
            raise RuntimeError('Native fluid state or particle conservation failed')
        return points, velocity

    def ledger(self):
        points, _ = self.state()
        free = len(points) * self.particle_mass
        absorbed = self.absorbed_count * self.particle_mass
        return {'initial_mass_kg': self.initial['initial_mass_kg'], 'free_mass_kg': free,
                'absorbed_mass_kg': absorbed, 'mass_error_kg': free + absorbed - self.initial['initial_mass_kg'],
                'remaining_particles': len(points), 'absorbed_particles': self.absorbed_count}

    def bind_absorber(self, absorber, *, capacity_kg, mass_rate_kg_s,
                      contact_distance_m, minimum_dwell_s, mean_extra_dwell_s=0., seed=17):
        from calibrated_scene_reconstruction.assets.wetting import ContactAbsorption

        if self.sink is not None:
            raise ValueError('One immutable absorber/capacity ledger per fluid instance')
        values = [capacity_kg, mass_rate_kg_s, contact_distance_m, minimum_dwell_s]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Positive finite capacity, flux, contact distance and dwell required')
        if not np.isfinite(mean_extra_dwell_s) or mean_extra_dwell_s < 0:
            raise ValueError('Finite nonnegative additional dwell required')
        if contact_distance_m > 2 * self.contact:
            raise ValueError('Absorption range cannot exceed the resolved contact shell')
        masses = np.asarray(absorber.masses(), float)
        if masses.ndim != 1 or not np.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError('Absorber requires finite native positive masses')
        self.sink = absorber
        self.sink_initial_mass = float(masses.sum())
        self.contact_distance = float(contact_distance_m)
        self.uptake = ContactAbsorption(self.initial_count, self.particle_mass, seed,
                                       capacity_kg, mass_rate_kg_s, minimum_dwell_s, mean_extra_dwell_s)

    def absorb(self, dt, *, enabled=True):
        from pxr import PhysxSchema, Sdf, UsdGeom, Vt
        import omni.physx
        from calibrated_scene_reconstruction.assets.wet_contact import sheet_contact_geometry, contact_impulses

        if not np.isfinite(dt) or dt <= 0 or self.sink is None:
            raise ValueError('Absorption requires a bound native surface and a positive step')
        if self.empty:
            return 0
        self.clock += dt
        p, v = self.state()
        cloth = self.sink
        q = cloth.points()
        geometry = sheet_contact_geometry(p, q, cloth.faces)
        fluid_v, cloth_v, contact = contact_impulses(
            p, v, self.particle_mass, q, cloth.velocities(), cloth.masses(), dt,
            distance=self.contact_distance, faces=cloth.faces, geometry=geometry)
        if contact['impulse_Ns'] > 0:
            self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(fluid_v.astype(np.float32)))
            cloth.apply_contact_velocities(cloth_v)
        distance = geometry[3]
        # Fractional coverage near the physical shell; exactly zero out of contact.
        weight = np.clip((self.contact_distance - distance) / (self.contact_distance * .5), 0, 1)
        remove = self.uptake.advance(weight if enabled else np.zeros_like(weight), dt)
        count = int(remove.sum())
        if count:
            mass = count * self.particle_mass
            momentum = self.particle_mass * fluid_v[remove].sum(0)
            before_mass = float(np.asarray(cloth.masses()).sum())
            cloth.accept_water(mass, momentum)
            after_mass = float(np.asarray(cloth.masses()).sum())
            if abs(after_mass - before_mass - mass) > max(1e-8, mass * 1e-4):
                raise RuntimeError('Absorbed fluid must enter the native absorber mass')
            self.absorbed_count += count
            if count == len(p):
                PhysxSchema.PhysxParticleAPI(self.particles.GetPrim()).CreateParticleEnabledAttr(False)
                self.particles.MakeInvisible()
                UsdGeom.Imageable(self.stage.GetPrimAtPath(self.root)).MakeInvisible()
                self.empty = True
            else:
                keep = ~remove
                values = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p[keep]))
                # simulationPoints must precede render points when native readback is enabled.
                with Sdf.ChangeBlock():
                    sim = self.pset.GetSimulationPointsAttr()
                    if sim.HasAuthoredValue():
                        sim.Set(values)
                    self.particles.GetPointsAttr().Set(values)
                    self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(fluid_v[keep], np.float32)))
                    self.particles.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(keep.sum(), 2 * self.rest, np.float32)))
            self.events.append({'time_s': self.clock, 'particles': count, 'mass_kg': mass,
                                'momentum_kg_m_s': momentum.tolist(),
                                'maximum_contact_distance_m': float(distance[remove].max())})
        omni.physx.get_physx_simulation_interface().flush_changes()
        return count
