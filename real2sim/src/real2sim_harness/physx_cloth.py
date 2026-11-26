"""Passive PhysX particle-cloth construction for an already running Isaac Sim.

The defaults are simulation priors, not fabric identification. No Kit session,
grasp controller, animation cache or acceptance decision is created here.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .deformable_geometry import surface_mesh


def resolve_cloth_settings(configuration):
    """Resolve metric mesh-dependent contacts and mass, preserving overrides."""
    c = configuration
    if c.get("representation", "surface") != "surface":
        raise ValueError("PhysX particle cloth requires a surface representation")
    points, faces, geometry_info = surface_mesh(
        c["geometry"]["vertices"], c["geometry"]["triangles"]
    )
    if len(np.unique(faces)) != len(points):
        raise ValueError("Cloth must not contain unused vertices")
    edges = geometry_info["edges"]
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    spacing = float(np.median(lengths))
    area = float(
        np.linalg.norm(
            np.cross(
                points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
            ),
            axis=1,
        ).sum()
        / 2
    )
    material = c["material"]
    density = _positive(material["density_kg_m3"], "density_kg_m3")
    thickness = _positive(material["thickness_m"], "thickness_m")
    mass = _positive(material.get("total_mass_kg", area * density * thickness), "total_mass_kg")
    fps = c.get("fps", 30)
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError("fps must be an integer in 1..120")
    defaults = {
        "substeps": math.ceil(480 / fps),
        "position_iterations": 128,
        "solid_rest_offset": spacing / 2,
        "particle_contact_offset": spacing * 2 / 3,
        "rest_offset": spacing / 6,
        "contact_offset": spacing / 2,
        "stretch_stiffness": 500.0,
        "shear_stiffness": 250.0,
        "bend_stiffness": 0.02,
        "spring_damping": 0.01,
        "material_damping": 2.0,
        "friction": 0.6,
        "max_velocity": 5.0,
        "max_depenetration_velocity": 0.5,
        "self_collision": True,
        "self_collision_filter": True,
        "enable_particle_ccd": True,
    }
    overrides = c.get("physx", {})
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown PhysX cloth settings: {sorted(unknown)}")
    settings = defaults | overrides
    for key in ("substeps", "position_iterations"):
        if type(settings[key]) is not int or not 1 <= settings[key] <= 255:
            raise ValueError(f"{key} must be an integer in 1..255")
    for key in ("self_collision", "self_collision_filter", "enable_particle_ccd"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in set(settings) - {
        "substeps",
        "position_iterations",
        "self_collision",
        "self_collision_filter",
        "enable_particle_ccd",
    }:
        settings[key] = _positive(
            settings[key],
            key,
            allow_zero=key
            in {
                "rest_offset",
                "bend_stiffness",
                "shear_stiffness",
                "spring_damping",
                "material_damping",
                "friction",
            },
        )
    if settings["contact_offset"] <= settings["rest_offset"]:
        raise ValueError("contact_offset must exceed rest_offset")
    if settings["particle_contact_offset"] <= settings["solid_rest_offset"]:
        raise ValueError("particle_contact_offset must exceed solid_rest_offset")
    return {
        "settings": settings,
        "dt_s": 1 / (fps * settings["substeps"]),
        "area_m2": area,
        "total_mass_kg": mass,
        "particle_mass_kg": mass / len(points),
        "vertices": len(points),
        "edge_length_percentiles_m": np.percentile(lengths, [0, 5, 50, 95, 100]).tolist(),
        "overridden_fields": sorted(overrides),
        "scope": "Initial numerical/material assumptions; validate contact, manipulation and settling on the target asset.",
    }


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"Invalid {name}")
    return value


def create_cloth(stage, configuration, *, prim_path, simulation_owner):
    """Author a passive garment and return native objects plus resolved settings.

    Call after SimulationApp startup, before world.reset(). The caller owns the
    world, timestep, environment and controls. Prim paths must be unused. The
    returned cloth can be added to world.scene; this function never pins nodes.
    """
    from isaacsim.core.api.materials.particle_material import ParticleMaterial
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from pxr import Sdf, UsdGeom, Vt

    resolved = resolve_cloth_settings(configuration)
    cfg = resolved["settings"]
    path = Sdf.Path(prim_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("Use an absolute non-root prim path")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Cloth geometry uses meters and Z-up; configure the stage first")
    paths = [prim_path, prim_path + "ParticleSystem", prim_path + "Material"]
    if any(stage.GetPrimAtPath(p).IsValid() for p in paths):
        raise ValueError("Cloth, particle system and material paths must be unused")
    if not stage.GetPrimAtPath(simulation_owner).IsValid():
        raise ValueError("simulation_owner must refer to the caller's physics scene")
    frequency = (
        stage.GetPrimAtPath(simulation_owner).GetAttribute("physxScene:timeStepsPerSecond").Get()
    )
    if frequency and not math.isclose(1 / frequency, resolved["dt_s"], rel_tol=1e-6):
        raise ValueError("Configure the caller's physics timestep to match resolved dt_s")
    geometry = configuration["geometry"]
    points = np.asarray(geometry["vertices"], dtype=np.float32)
    faces = np.asarray(geometry["triangles"], dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    # Explicit offsets prevent runtime defaults from changing the tested asset.
    system = SingleParticleSystem(
        prim_path=paths[1],
        simulation_owner=simulation_owner,
        enable_ccd=cfg["enable_particle_ccd"],
        solver_position_iteration_count=cfg["position_iterations"],
        global_self_collision_enabled=cfg["self_collision"],
        non_particle_collision_enabled=True,
        contact_offset=cfg["contact_offset"],
        rest_offset=cfg["rest_offset"],
        particle_contact_offset=cfg["particle_contact_offset"],
        solid_rest_offset=cfg["solid_rest_offset"],
        fluid_rest_offset=cfg["solid_rest_offset"],
        max_velocity=cfg["max_velocity"],
        max_depenetration_velocity=cfg["max_depenetration_velocity"],
    )
    material = ParticleMaterial(
        prim_path=paths[2],
        friction=cfg["friction"],
        damping=cfg["material_damping"],
        drag=0.0,
        lift=0.0,
        gravity_scale=1.0,
    )
    cloth = SingleClothPrim(
        prim_path,
        system,
        particle_material=material,
        particle_mass=resolved["particle_mass_kg"],
        self_collision=cfg["self_collision"],
        self_collision_filter=cfg["self_collision_filter"],
        stretch_stiffness=cfg["stretch_stiffness"],
        shear_stiffness=cfg["shear_stiffness"],
        bend_stiffness=cfg["bend_stiffness"],
        spring_damping=cfg["spring_damping"],
    )
    mesh.GetPrim().SetCustomDataByKey("real2sim:clothSettings", json.dumps(resolved))
    return cloth, system, material, resolved
