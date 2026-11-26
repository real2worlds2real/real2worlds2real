"""Small native PhysX puddle with contact absorption for the RGB wiping demo.

This environment owns the true particles. The perception/planning module never
imports this file or receives these positions. Absorption is an engineered sink.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdShade, Vt
from omni.physx.scripts import particleUtils, physicsUtils


class WipePuddle:
    def __init__(self, stage, scene_path, center, table_z, bounds, seed=7, spacing=.0018, appearance=None,
                 layers=3,footprint_scale=1.):
        self.stage, self.z, self.bounds = stage, float(table_z), np.asarray(bounds)
        self.spacing, self.rest = spacing, spacing / 2
        self.contact = self.rest / (.99 * .6)
        self.empty, self.absorbed = False, 0
        self.system_path = Sdf.Path('/World/WipingWaterSystem')
        self.appearance = dict(glass_color=[.995, .999, 1.], depth=0., frosting_roughness=.006,
                               glass_ior=1.333, thin_walled=True,
                               approximation='Thin-film optical surface; native fluid physics unchanged')
        if appearance:
            self.appearance.update(appearance)
        path = Sdf.Path('/World/Looks/WipingWater')
        mat = UsdShade.Material.Define(stage, path)
        mdl = UsdShade.Shader.Define(stage, str(path) + '/Glass')
        mdl.SetSourceAsset(Sdf.AssetPath('OmniGlass.mdl'), 'mdl')
        mdl.SetSourceAssetSubIdentifier('OmniGlass', 'mdl')
        for name in ['glass_ior', 'frosting_roughness', 'depth']:
            value = self.appearance[name]
            mdl.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
        mdl.CreateInput('glass_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.appearance['glass_color']))
        mdl.CreateInput('thin_walled', Sdf.ValueTypeNames.Bool).Set(self.appearance['thin_walled'])
        for channel in ('surface', 'volume', 'displacement'):
            mat.CreateOutput('mdl:' + channel, Sdf.ValueTypeNames.Token).ConnectToSource(mdl.ConnectableAPI(), 'out')
        particleUtils.AddPBDMaterialWater(mat.GetPrim())
        system = particleUtils.add_physx_particle_system(
            stage, self.system_path, simulation_owner=scene_path, contact_offset=self.contact,
            rest_offset=self.rest, particle_contact_offset=self.contact, solid_rest_offset=self.rest,
            fluid_rest_offset=self.rest, solver_position_iterations=12, max_velocity=.6,
            max_depenetration_velocity=.2, enable_ccd=True)
        physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), path)
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(mat)
        particleUtils.add_physx_particle_isosurface(
            stage, self.system_path, enabled=True, grid_spacing=spacing * .7,
            surface_distance=self.rest * 1.6, max_vertices=180000, max_triangles=360000,
            num_mesh_smoothing_passes=3, num_mesh_normal_smoothing_passes=3)
        x, y, z = np.meshgrid(np.arange(-.060*footprint_scale, .060*footprint_scale, spacing), np.arange(-.043*footprint_scale, .043*footprint_scale, spacing),
                              np.arange(layers) * spacing + self.z + self.rest + .0004, indexing='ij')
        p = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
        theta = np.arctan2(p[:, 1] / (.039*footprint_scale), p[:, 0] / (.056*footprint_scale))
        boundary = 1 + .065 * np.sin(3 * theta) + .035 * np.cos(5 * theta)
        p = p[(p[:, 0] / (.056*footprint_scale))**2 + (p[:, 1] / (.039*footprint_scale))**2 < boundary**2]
        p += np.random.default_rng(seed).uniform(-spacing * .025, spacing * .025, p.shape).astype(np.float32)
        p[:, :2] += np.asarray(center, np.float32)
        self.initial_count = len(p)
        self.particles = particleUtils.add_physx_particleset_points(
            stage, Sdf.Path('/World/WipingWater'), Vt.Vec3fArray.FromNumpy(p),
            Vt.Vec3fArray.FromNumpy(np.zeros_like(p)), [2 * self.rest] * len(p), self.system_path,
            self_collision=True, fluid=True, particle_group=0, particle_mass=0., density=1000.)
        self.particles.GetPrim().CreateAttribute('physxParticle:maxParticles', Sdf.ValueTypeNames.Int).Set(len(p))
        self.particles.MakeInvisible()
        self.pset = PhysxSchema.PhysxParticleSetAPI(self.particles.GetPrim())
        self.surface = UsdGeom.Mesh.Define(stage, '/World/WipingSurface')
        self.surface.CreateSubdivisionSchemeAttr('none')
        self.surface.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self.surface.MakeInvisible()
        UsdShade.MaterialBindingAPI.Apply(self.surface.GetPrim()).Bind(mat)

    def state(self):
        if self.empty:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        simulation = self.pset.GetSimulationPointsAttr()
        points = simulation if simulation.HasAuthoredValue() else self.particles.GetPointsAttr()
        p = np.asarray(points.Get(), np.float32).reshape(-1, 3).copy()
        v = np.asarray(self.particles.GetVelocitiesAttr().Get(), np.float32).reshape(-1, 3).copy()
        if p.shape != v.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise RuntimeError('Invalid native fluid buffers')
        return p, v

    def absorb(self, center, rotation, size, enabled=True):
        return self.absorb_boxes([(center, rotation, size)], enabled=enabled)

    def absorb_boxes(self, boxes, enabled=True):
        """Apply the union of simultaneous physical contacts in one buffer edit."""
        if self.empty or not enabled:
            return 0
        p, v = self.state()
        if not len(p):
            return 0
        hit = np.zeros(len(p), dtype=bool)
        for center, rotation, size in boxes:
            excess = np.abs((p - center) @ rotation) - (np.asarray(size) / 2 + self.contact + .0002)
            hit |= (excess <= 0).all(1)
        count = int(hit.sum())
        if not count:
            return 0
        self.absorbed += count
        if hit.all():
            self.stage.RemovePrim(self.particles.GetPath())
            self.stage.RemovePrim(self.system_path)
            self.surface.MakeInvisible()
            self.empty = True
            return count
        keep = ~hit
        values = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p[keep]))
        with Sdf.ChangeBlock():
            sim = self.pset.GetSimulationPointsAttr()
            if sim.HasAuthoredValue():
                sim.Set(values)
            self.particles.GetPointsAttr().Set(values)
            self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v[keep])))
            self.particles.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(keep.sum(), 2 * self.rest, np.float32)))
        return count

    def update_surface(self):
        if self.empty:
            return
        native = UsdGeom.Mesh.Get(self.stage, str(self.system_path) + '/Isosurface')
        if not native:
            return
        native.MakeInvisible()
        vertices = np.asarray(native.GetPointsAttr().Get(), np.float32).reshape(-1, 3).copy()
        faces = np.asarray(native.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
        if not len(faces) or not len(vertices):
            self.surface.MakeInvisible()
            return
        tri = vertices[faces]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        faces = faces[areas > 1e-20]
        if not len(faces):
            self.surface.MakeInvisible()
            return
        normals = np.asarray(native.GetNormalsAttr().Get(), np.float32)
        if normals.shape != vertices.shape:
            normals = self.normals(vertices, faces)
        near = ((vertices[:, :2] >= self.bounds[0]).all(1)
                & (vertices[:, :2] <= self.bounds[1]).all(1)
                & (np.abs(vertices[:, 2] - self.z) < 2 * self.contact))
        # The already-validated render contact fitting from demo_fluid_wipe:
        # put the optical underside 0.1 mm into the opaque table, physics intact.
        vertices[near & (normals[:, 2] < 0), 2] = self.z - .0001
        vertices[near, 2] = np.maximum(vertices[near, 2], self.z - .0001)
        self.surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        self.surface.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        self.surface.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()))
        self.surface.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(self.normals(vertices, faces)))
        self.surface.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0), vertices.max(0)])))
        self.surface.MakeVisible()

    @staticmethod
    def normals(vertices, faces):
        tri = vertices[faces]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = np.zeros_like(vertices)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
