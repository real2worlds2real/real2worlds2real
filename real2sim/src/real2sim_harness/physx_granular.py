"""Passive solid particles in a caller-owned PhysX world; no Kit startup.

These are coarse PBD grains, not a calibrated continuum sand law. Geometry,
sampling and contact/material assumptions remain explicit and editable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GranularSettings:
    radius_m: float
    mass_kg: float  # per grain
    friction: float = 0.75
    cohesion: float = 0.0
    damping: float = 0.1
    adhesion: float = 0.0
    particle_friction_scale: float = 1.0
    particle_adhesion_scale: float = 1.0
    particle_contact_offset_m: float | None = None
    solid_rest_offset_m: float | None = None
    contact_offset_m: float | None = None
    max_velocity_m_s: float = 10.0

    def resolved(self):
        c = asdict(self)
        for name in ("radius_m", "mass_kg", "max_velocity_m_s"):
            if isinstance(c[name], bool) or not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"Positive finite {name} required")
        for name in ("friction", "cohesion", "damping", "adhesion",
                     "particle_friction_scale", "particle_adhesion_scale"):
            if isinstance(c[name], bool) or not np.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"Nonnegative finite {name} required")
        for name, factor in (("solid_rest_offset_m", 1.0),
                             ("particle_contact_offset_m", 1.25),
                             ("contact_offset_m", 1.875)):
            if c[name] is None:
                c[name] = factor * c["radius_m"]
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"Positive finite {name} required")
        if not (c["particle_contact_offset_m"] > c["solid_rest_offset_m"]
                and c["contact_offset_m"] > c["solid_rest_offset_m"]):
            raise ValueError("Contact offsets must exceed the solid rest offset")
        return c


def sample_lattice(lower_m, upper_m, spacing_m):
    """Grain centers in a half-open box. Boundaries are centers, not container walls."""
    lo, hi = np.asarray(lower_m, float), np.asarray(upper_m, float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
            or np.any(hi <= lo) or not np.isfinite(spacing_m) or spacing_m <= 0):
        raise ValueError("Finite nonempty bounds and positive spacing required")
    sizes = np.ceil((hi - lo) / spacing_m)
    if not np.isfinite(sizes).all() or np.prod(sizes, dtype=float) > 1_000_000:
        raise ValueError("More than one million seed grains; choose sampling explicitly")
    axes = [np.arange(a, b, spacing_m) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def author_granular(stage, path, positions_m, settings, *, simulation_owner,
                    color=(0.86, 0.74, 0.52)):
    """Author a passive solid particle set in an existing GPU-dynamics scene.

    The caller supplies containers and controls. This function neither changes
    the world's solver/clock nor instantiates a robot or demonstration policy.
    """
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdShade, Vt

    c = settings.resolved()
    p = np.asarray(positions_m, float)
    rgb = np.asarray(color, float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("Finite nonempty grain centers Nx3 required")
    if len(np.unique(p, axis=0)) != len(p):
        raise ValueError("Coincident seed grain centers are not a valid initial state")
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("Color must be finite RGB in [0,1]")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError("Author granular particles in a metre stage")
    if not Sdf.Path(path).IsAbsolutePath() or stage.GetPrimAtPath(path):
        raise ValueError("Choose a new absolute asset prim path")
    owner = stage.GetPrimAtPath(simulation_owner)
    if not owner or not owner.GetAttribute("physxScene:enableGPUDynamics").Get():
        raise ValueError("PhysX solid particles need an explicit GPU-dynamics simulation owner")
    system_path, particles_path = path + "/System", path + "/Particles"
    try:
        UsdGeom.Xform.Define(stage, path).SetResetXformStack(True)
        system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
        system.CreateSimulationOwnerRel().SetTargets([Sdf.Path(simulation_owner)])
        system.CreateParticleContactOffsetAttr(c["particle_contact_offset_m"])
        system.CreateContactOffsetAttr(c["contact_offset_m"])
        system.CreateRestOffsetAttr(c["solid_rest_offset_m"])
        system.CreateSolidRestOffsetAttr(c["solid_rest_offset_m"])
        system.CreateFluidRestOffsetAttr(c["solid_rest_offset_m"])
        system.CreateMaxVelocityAttr(c["max_velocity_m_s"])
        material_path = path + "/ContactMaterial"
        particleUtils.add_pbd_particle_material(
            stage, Sdf.Path(material_path),
            **{k: c[k] for k in ("friction", "cohesion", "damping", "adhesion",
                                 "particle_friction_scale", "particle_adhesion_scale")},
            gravity_scale=1.0,
        )
        physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), material_path)
        particleUtils.add_physx_particleset_pointinstancer(
            stage=stage, path=Sdf.Path(particles_path),
            positions=Vt.Vec3fArray.FromNumpy(p.astype(np.float32)),
            velocities=Vt.Vec3fArray.FromNumpy(np.zeros_like(p, dtype=np.float32)),
            particle_system_path=Sdf.Path(system_path), self_collision=True, fluid=False,
            particle_group=0, particle_mass=c["mass_kg"], density=0.0,
        )
        instancer = UsdGeom.PointInstancer.Get(stage, particles_path)
        proto = UsdGeom.Sphere.Get(stage, instancer.GetPrototypesRel().GetTargets()[0])
        if not proto:
            raise ValueError("PhysX particle utility did not create its sphere prototype")
        proto.CreateRadiusAttr(c["radius_m"])
        proto.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*rgb)])
        visual = UsdShade.Material.Define(stage, path + "/Surface")
        shader = UsdShade.Shader.Define(stage, path + "/Surface/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(visual)
    except Exception:
        stage.RemovePrim(path)
        raise
    return {"path": path, "particles_path": particles_path, "system_path": system_path,
            "settings": c, "particle_count": len(p), "total_mass_kg": len(p) * c["mass_kg"]}


def read_granular_positions(stage, asset):
    """Read CPU-pipeline USD state; a Fabric/tensor caller must supply its own reader."""
    from pxr import Gf, UsdGeom

    inst = UsdGeom.PointInstancer.Get(stage, asset["particles_path"])
    points = np.asarray(inst.GetPositionsAttr().Get(), float)
    if points.shape != (asset["particle_count"], 3) or not np.isfinite(points).all():
        raise ValueError("Particle count changed or returned nonfinite state")
    matrix = inst.ComputeLocalToWorldTransform(0)
    return np.array([matrix.Transform(Gf.Vec3d(*p)) for p in points])


def audit_grains(positions_m, *, expected_count, radius_m, floor_z_m,
                 bounds_xy_m=None, tolerance_m=0.0):
    """Sampled center/radius support and leakage checks, not continuous CCD proof."""
    p = np.asarray(positions_m, float)
    if (p.ndim != 2 or p.shape[1] != 3 or radius_m <= 0 or tolerance_m < 0
            or not np.isfinite([radius_m, floor_z_m, tolerance_m]).all()):
        raise ValueError("Valid particle positions and audit distances required")
    finite = np.isfinite(p).all(axis=1)
    below = int(np.sum(p[finite, 2] - radius_m < floor_z_m - tolerance_m))
    outside = 0
    if bounds_xy_m is not None:
        limits = np.asarray(bounds_xy_m, float)
        if limits.shape != (2, 2) or not np.isfinite(limits).all() or np.any(limits[1] <= limits[0]):
            raise ValueError("XY bounds are [lower_xy, upper_xy]")
        outside = int(np.sum(np.any((p[finite, :2] < limits[0] - tolerance_m)
                                    | (p[finite, :2] > limits[1] + tolerance_m), axis=1)))
    return {"ok": bool(len(p) == expected_count and finite.all() and not below and not outside),
            "count": len(p), "nonfinite": int((~finite).sum()),
            "below_support": below, "outside_xy": outside}
