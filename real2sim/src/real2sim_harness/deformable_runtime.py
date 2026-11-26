"""Native shell/solid behavior runtime, independent of object names and Blender."""

from __future__ import annotations

import numpy as np

from .deformable_behavior import validate_behavior
from .deformable_geometry import shell_bending_coefficients, surface_mesh, tetrahedra


def apply_deformable_environment(runtime, state):
    """Optional mass-proportional still-air force; never edits cached trajectories."""
    import warp as wp

    from .newton_forces import particle_air_drag_rates

    rate = runtime["configuration"].get("environment", {}).get("air_drag_rate_per_second", 0.0)
    if rate:
        wp.launch(
            particle_air_drag_rates,
            dim=runtime["model"].particle_count,
            inputs=[state.particle_qd, runtime["model"].particle_inv_mass, runtime["drag_rates"]],
            outputs=[state.particle_f],
            device=runtime["model"].device,
        )


def elastic_constants(young, poisson, *, thickness=None):
    if (
        not np.isfinite(young)
        or young <= 0
        or not np.isfinite(poisson)
        or not -0.9 < poisson < 0.49
    ):
        raise ValueError("Use positive Young's modulus and Poisson ratio in (-.9,.49)")
    if thickness is not None:
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("A shell needs a positive physical thickness")
        return young * thickness / (2 * (1 + poisson)), young * thickness * poisson / (
            1 - poisson**2
        )
    return young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))


def load_deformable(
    configuration,
    *,
    device="cuda:0",
    builder=None,
    prescribed_motion=False,
    include_probes=False,
    include_ground=None,
    build_only=False,
):
    """Load passive geometry by default, optionally into a prepopulated shared world.

    Demonstration probes and particle grasps are explicitly opt-in. The caller
    owns subsequent controls, forces and shared-world solver stepping.
    """
    import newton
    import warp as wp
    from newton.solvers import SolverVBD

    c = validate_behavior(configuration)
    if c.get("schema_version") != 1 or c.get("representation") not in {"surface", "volume"}:
        raise ValueError("Use schema 1 and a surface or volume representation")
    geometry, material = c["geometry"], c["material"]
    points, faces, _ = surface_mesh(
        geometry["vertices"], geometry["triangles"], closed=c["representation"] == "volume"
    )
    if len(points) > 100000 or len(faces) > 200000:
        raise ValueError("Behavior control mesh exceeds its particle/triangle budget")
    density = float(material["density_kg_m3"])
    radius = float(material["contact_radius_m"])
    if not np.isfinite([density, radius]).all() or min(density, radius) <= 0:
        raise ValueError("Use positive density and contact radius")
    env = c.get("environment", {})
    drag = float(env.get("air_drag_rate_per_second", 0.0))
    if not np.isfinite(drag) or drag < 0:
        raise ValueError("Use a finite nonnegative still-air drag rate")
    for key, default in (("friction", 0.5), ("contact_stiffness", 2e5), ("contact_damping", 0.1)):
        value = float(env.get(key, default))
        if not np.isfinite(value) or value < 0 or (key == "contact_stiffness" and value == 0):
            raise ValueError("Use finite nonnegative contact coefficients and positive stiffness")
    if not np.isfinite(env.get("floor_height", 0.0)):
        raise ValueError("Use a finite ground height")
    gravity = np.asarray(env.get("gravity", [0, 0, -9.81]), dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Use finite scene-frame gravity")
    wp.set_device(device)
    own_world = builder is None
    if own_world:
        builder = newton.ModelBuilder(gravity=tuple(gravity))
    particle_start = len(builder.particle_q)
    young, poisson = float(material["youngs_modulus_pa"]), float(material["poisson_ratio"])
    damping = float(material.get("damping", 1e-5))
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("Use nonnegative material damping")
    if c["representation"] == "surface":
        if "bending_rigidity_nm" in material and "bending_stiffness" in material:
            raise ValueError(
                "Choose physical plate rigidity or an explicit native bending override"
            )
        h = material["thickness_m"]
        mu, lmbd = elastic_constants(young, poisson, thickness=h)
        rigidity = material.get("bending_rigidity_nm", young * h**3 / (12 * (1 - poisson**2)))
        bending = material.get("bending_stiffness", 0.0)
        if (
            not np.isfinite([bending, material.get("bending_damping", 1e-6)]).all()
            or min(bending, material.get("bending_damping", 1e-6)) < 0
        ):
            raise ValueError("Use finite nonnegative bending coefficients")
        edge_start = len(builder.edge_indices)
        builder.add_cloth_mesh(
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0),
            vertices=points.tolist(),
            indices=faces.reshape(-1).tolist(),
            density=density * h,
            tri_ke=mu,
            tri_ka=lmbd,
            tri_kd=damping,
            edge_ke=bending,
            edge_kd=material.get("bending_damping", 1e-6),
            particle_radius=radius,
        )
        if "bending_stiffness" not in material:
            indices = np.asarray(builder.edge_indices[edge_start:], dtype=np.int32)
            indices = np.where(indices >= 0, indices - particle_start, -1)
            coefficients = shell_bending_coefficients(points, indices, rigidity)
            for i, stiffness in enumerate(coefficients, start=edge_start):
                builder.edge_bending_properties[i] = (
                    float(stiffness),
                    material.get("bending_damping", 1e-6),
                )
        elements = None
    else:
        mu, lmbd = elastic_constants(young, poisson)
        if lmbd < 0:
            raise ValueError("This solid adapter currently requires nonnegative lambda")
        _, elements, _, _ = tetrahedra(points, geometry["tetrahedra"])
        builder.add_soft_mesh(
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0),
            vertices=points.tolist(),
            indices=elements.reshape(-1).tolist(),
            density=density,
            k_mu=mu,
            k_lambda=lmbd,
            k_damp=damping,
            particle_radius=radius,
            validate_mesh=True,
        )
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=env.get("friction", 0.5), ke=2e5, kd=1.0)
    if include_ground is True or (include_ground is None and own_world):
        builder.add_ground_plane(height=env.get("floor_height", 0.0), cfg=shape_cfg)
    probes = []
    for spec in c.get("probes", []) if include_probes else []:
        position = np.asarray(spec["position_m"], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Use a finite probe position")
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            is_kinematic=True,
            label=spec["id"],
        )
        if spec.get("collision", True):
            if spec["shape"] == "sphere":
                builder.add_shape_sphere(body, radius=spec["radius_m"], cfg=shape_cfg)
            elif spec["shape"] == "box":
                size = np.asarray(spec["size_m"]) / 2
                builder.add_shape_box(body, hx=size[0], hy=size[1], hz=size[2], cfg=shape_cfg)
            else:
                raise ValueError("Supported physical probes are spheres and boxes")
        builder.body_mass[body] = 0.0
        builder.body_inv_mass[body] = 0.0
        builder.body_inertia[body] = wp.mat33(0.0)
        builder.body_inv_inertia[body] = wp.mat33(0.0)
        probes.append({"body": body, "specification": spec})
    for grasp in c.get("grasps", []) if prescribed_motion else []:
        ids = np.asarray(grasp["indices"])
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(points):
            raise ValueError("Grasp indices must name control vertices")
        for i in ids:
            builder.particle_flags[particle_start + int(i)] &= ~int(newton.ParticleFlags.ACTIVE)
    initialization = {
        "mu": mu, "lambda": lmbd,
        "scope": "Assumed elastic model; shell coefficients use plane stress and areal density.",
        "bending_mapping": (
            "physical plate rigidity divided by barycentric dual width; native edge-length integration"
            if c["representation"] == "surface" and "bending_stiffness" not in material
            else "explicit native override" if c["representation"] == "surface" else None
        ),
    }
    if build_only:
        return {
            "builder": builder, "points": points, "triangles": faces, "tetrahedra": elements,
            "particle_ids": np.arange(particle_start, particle_start + len(points), dtype=np.int32),
            "probes": probes, "configuration": c, "material_initialization": initialization,
            "prescribed_motion": prescribed_motion,
        }
    builder.color(include_bending=True)
    model = builder.finalize(device=device)
    model.soft_contact_ke = float(env.get("contact_stiffness", 2e5))
    model.soft_contact_kd = float(env.get("contact_damping", 0.1))
    model.soft_contact_mu = float(env.get("friction", 0.5))
    settings = c.get("solver", {})
    full_surface = settings.get("full_surface_contact", False)
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=2 * radius, enable_rigid_soft_full_surface_contact=full_surface
    )
    solver = SolverVBD(
        model,
        iterations=settings.get("iterations", 16),
        rigid_compliant_alm=True,
        rigid_body_particle_contact_buffer_size=max(
            256, (8 * model.particle_count if full_surface else model.particle_count + 16)
        ),
        particle_enable_self_contact=settings.get("self_contact", c["representation"] == "surface"),
        particle_self_contact_margin=settings.get("self_contact_margin_m", radius * 2),
        particle_self_contact_gap=settings.get("self_contact_gap_m", radius),
        particle_topological_contact_filter_threshold=2,
        particle_enable_tile_solve=False,
        **({"collision_pipeline": pipeline} if full_surface else {}),
    )
    ids = np.arange(particle_start, particle_start + len(points), dtype=np.int32)
    rates = np.zeros(model.particle_count, dtype=np.float32)
    rates[ids] = drag
    return {
        "model": model,
        "solver": solver,
        "pipeline": pipeline,
        "contacts": solver.contacts if full_surface else pipeline.contacts(),
        "solver_owns_pipeline": full_surface,
        "state0": model.state(),
        "state1": model.state(),
        "control": model.control(),
        "points": points,
        "particle_ids": ids,
        "drag_rates": wp.array(rates, dtype=wp.float32, device=device),
        "prescribed_motion": prescribed_motion,
        "triangles": faces,
        "tetrahedra": elements,
        "probes": probes,
        "configuration": c,
        "material_initialization": initialization,
    }
