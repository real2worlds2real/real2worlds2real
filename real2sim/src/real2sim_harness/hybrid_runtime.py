"""Native Newton loader for articulated frames, cloth and explicit attachments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .hybrid_contract import validate_hybrid_contract
from .rigid_frames import rotation_only
from .sim_parameters import load_parameters


def load_hybrid_asset(folder, *, device="cuda:0", build_only=False):
    """Build an editable Newton model; no object category or robot task is assumed."""
    import newton
    import warp as wp
    from newton.solvers import SolverVBD

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    configuration = json.loads((folder / "hybrid_configuration.json").read_text())
    control_mesh = json.loads((folder / "sim_mesh.json").read_text())
    validate_hybrid_contract(configuration, manifest["joints"], control_mesh)
    parameters = load_parameters(folder, create=not build_only)
    full_surface_contact = parameters["solver"]["hybrid_full_surface_contact"]
    wp.set_device(device)
    builder = newton.ModelBuilder(gravity=tuple(parameters["environment"]["gravity"]))
    if not hasattr(builder, "add_attachment_particle_body"):
        raise RuntimeError("Hybrid assets require the isolated Newton 1.7 native attachment runtime")
    SolverVBD.register_custom_attributes(builder)

    def transform(matrix):
        matrix = np.asarray(matrix)
        rotation = wp.quat_from_matrix(wp.mat33(rotation_only(matrix)))
        return wp.transform(wp.vec3(*matrix[:3, 3]), rotation)

    body_ids, rest = {}, {"__base__": np.eye(4)}
    rest.update({spec["name"]: np.asarray(spec["matrix_world"]) for spec in manifest["joints"]})
    components = json.loads((folder / "collision_meshes.json").read_text())
    for name, matrix in rest.items():
        source_spec = next((row for row in manifest["joints"] if row["name"] == name), {})
        owned = [part for part in components if (part["owner_joint"] or "__base__") == name]
        vertices = np.concatenate([np.asarray(part["vertices"]) for part in owned]) if owned else np.zeros((1, 3))
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        size = np.maximum(np.ptp(vertices * scale, axis=0), 0.003)
        mass = source_spec.get("mass_kg") or parameters["rigid"]["mass_kg"] or float(np.clip(np.prod(size) * parameters["rigid"]["density_kg_m3"], 0.01, 20.0))
        inertia = mass / 12 * np.array([size[1]**2+size[2]**2, size[0]**2+size[2]**2, size[0]**2+size[1]**2])
        center = ((vertices * scale).min(axis=0) + (vertices * scale).max(axis=0)) / 2
        body = builder.add_link(xform=transform(matrix), mass=mass, inertia=wp.mat33(np.diag(inertia)),
                                com=wp.vec3(*center), lock_inertia=True, label=name)
        body_ids[name] = body
        for part in owned:
            mesh_vertices = np.asarray(part["vertices"], dtype=np.float32) * scale
            mesh = newton.Mesh(vertices=mesh_vertices,
                               indices=np.asarray(part["faces"], dtype=np.int32).reshape(-1))
            shape_config = newton.ModelBuilder.ShapeConfig(density=0.0, mu=parameters["material"]["friction"])
            if full_surface_contact:
                # Resolve thin features, rather than spending a fixed voxel budget
                # along a long rod while losing its transverse thickness.
                extents = np.ptp(mesh_vertices, axis=0)
                positive = extents[extents > 1e-7]
                voxel = min(parameters["surface"]["particle_radius"] / 2,
                            float(positive.min()) / 4 if len(positive) else 0.001)
                mesh.build_sdf(target_voxel_size=max(voxel, 0.0001), narrow_band_range=(-0.01, 0.01))
            builder.add_shape_mesh(body, mesh=mesh, label=part["object_name"],
                                   cfg=shape_config)

    joints, all_joints, dofs = {}, [], {}
    if manifest["fixed_base"]:
        all_joints.append(builder.add_joint_fixed(-1, body_ids["__base__"], label="held_base"))
    else:
        all_joints.append(builder.add_joint_free(body_ids["__base__"], label="free_base"))
    controlled = {record["joint"] for record in configuration["motion"]}
    for spec in manifest["joints"]:
        name, parent = spec["name"], spec["parent_joint"] or "__base__"
        parent_frame = wp.transform_multiply(wp.transform_inverse(transform(rest[parent])), transform(rest[name]))
        axis = wp.vec3(*np.eye(3)["XYZ".index(spec["axis"])])
        kwargs = dict(parent=body_ids[parent], child=body_ids[name], parent_xform=parent_frame,
                      axis=axis, label=name, target_ke=spec["drive_stiffness"] if name in controlled else 0.0,
                      target_kd=spec["drive_damping"] if name in controlled else 0.0)
        dofs[name] = builder.joint_dof_count
        if spec["joint_type"] in {"revolute", "continuous"}:
            kwargs.update(limit_lower=math.radians(spec["lower_degrees"]-spec["current_degrees"]),
                          limit_upper=math.radians(spec["upper_degrees"]-spec["current_degrees"]))
            joint = builder.add_joint_revolute(**kwargs)
        elif spec["joint_type"] == "prismatic":
            joint = builder.add_joint_prismatic(**kwargs, limit_lower=spec["lower_m"], limit_upper=spec["upper_m"])
        else:
            raise ValueError(f"Unsupported hybrid joint: {spec['joint_type']}")
        joints[name] = joint
        all_joints.append(joint)
    builder.add_articulation(all_joints.copy(), label="generated_frame")
    loop_anchors = []
    for index, constraint in enumerate(configuration["loop_constraints"]):
        a, b = constraint["body_a"], constraint["body_b"]
        anchor = wp.transform(wp.vec3(*constraint["anchor_world"]), wp.quat_identity())
        kwargs = dict(parent=body_ids[a], child=body_ids[b],
                      parent_xform=wp.transform_multiply(wp.transform_inverse(transform(rest[a])), anchor),
                      child_xform=wp.transform_multiply(wp.transform_inverse(transform(rest[b])), anchor),
                      label=f"loop_{index:03d}")
        function = builder.add_joint_ball if constraint["type"] == "ball" else builder.add_joint_fixed
        all_joints.append(function(**kwargs))
        loop_anchors.append((body_ids[a], body_ids[b],
                             np.asarray(wp.transform_get_translation(kwargs["parent_xform"])),
                             np.asarray(wp.transform_get_translation(kwargs["child_xform"]))))
    # The articulated frame's own intersecting hinge proxies are not a self-
    # collision model. External geometry and cloth remain separate interfaces.
    for first in range(builder.shape_count):
        for second in range(first + 1, builder.shape_count):
            if builder.shape_body[first] != builder.shape_body[second]:
                builder.add_shape_collision_filter_pair(first, second)
    surface = parameters["surface"]
    builder.add_cloth_mesh(pos=wp.vec3(0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0),
        vertices=[wp.vec3(*point) for point in control_mesh["vertices"]],
        indices=np.asarray(control_mesh["faces"], dtype=np.int32).reshape(-1).tolist(),
        density=surface["density"], tri_ke=surface["tri_ke"], tri_ka=surface["tri_ka"], tri_kd=surface["tri_kd"],
        edge_ke=surface["edge_ke"], edge_kd=surface["edge_kd"], particle_radius=surface["particle_radius"])
    attachments = []
    for group in configuration["attachments"]:
        body = body_ids[group["body"]]
        inverse = wp.transform_inverse(transform(rest[group["body"]]))
        for particle in group["particle_indices"]:
            point = wp.transform_point(inverse, wp.vec3(*control_mesh["vertices"][particle]))
            builder.add_attachment_particle_body(particle, body, body_point=point,
                stiffness=group["stiffness"], damping=group["damping"])
            attachments.append((particle, body, np.asarray(point)))
    construction = {"builder": builder, "body_ids": body_ids, "joint_ids": joints,
                    "joint_dofs": dofs, "rest_matrices": rest, "attachments": attachments,
                    "loop_anchors": loop_anchors, "configuration": configuration,
                    "parameters": parameters, "control_mesh": control_mesh,
                    "rigid_self_collision": False}
    if build_only:
        return construction
    builder.color()
    model = builder.finalize()
    iterations = parameters["solver"]["iterations"]
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=full_surface_contact)
    pipeline.contacts()
    solver = SolverVBD(model, iterations=iterations, rigid_compliant_alm=True,
            rigid_body_particle_contact_buffer_size=max(1024, 8*model.particle_count),
            particle_enable_self_contact=parameters["solver"]["self_contact"],
            particle_self_contact_radius=surface["particle_radius"], particle_self_contact_margin=2*surface["particle_radius"],
            rigid_soft_enable_dat=parameters["solver"]["hybrid_rigid_soft_dat"],
            collision_pipeline=pipeline)
    return {"model": model, "solver": solver, "control": model.control(), "body_ids": body_ids,
            "joint_ids": joints, "joint_dofs": dofs, "rest_matrices": rest, "attachments": attachments,
            "loop_anchors": loop_anchors,
            "configuration": configuration, "parameters": parameters, "control_mesh": control_mesh,
            "collision_pipeline": pipeline, "rigid_self_collision": False}
