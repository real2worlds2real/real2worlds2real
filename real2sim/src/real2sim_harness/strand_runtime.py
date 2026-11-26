"""Native Newton rod/cable loader, independent of Blender and object category."""

from __future__ import annotations

import numpy as np

from .strand_contract import validate_configuration


def apply_strand_environment(runtime, state):
    """Apply the retained world's optional linear air force after clearing forces."""
    import warp as wp

    from .newton_forces import rigid_still_air_drag

    rate = runtime["configuration"]["environment"]["air_drag_rate_per_second"]
    if rate:
        wp.launch(
            rigid_still_air_drag,
            dim=runtime["model"].body_count,
            inputs=[state.body_qd, runtime["model"].body_mass, float(rate)],
            outputs=[state.body_f],
        )


def load_strand_asset(configuration, device="cuda:0", *, builder=None, prescribed_motion=False):
    import newton
    import warp as wp
    from newton.solvers import SolverVBD

    c = validate_configuration(configuration)
    wp.set_device(device)
    if builder is None:
        builder = newton.ModelBuilder(gravity=tuple(c["environment"]["gravity"]))
    SolverVBD.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=1100.0,
        mu=c["environment"]["friction"],
        ke=c["environment"].get("contact_stiffness", 1e5),
        kd=c["environment"].get("contact_damping", 1.0),
    )
    # Delivery is passive by default; validation may explicitly hold/control grasps.
    driven = {m["body"] for m in c["motion"]} if prescribed_motion else set()
    bodies = {}
    for b in c["bodies"]:
        mat = np.asarray(b["matrix_world"])
        pose = wp.transform(wp.vec3(*mat[:3, 3]), wp.quat_from_matrix(wp.mat33(mat[:3, :3])))
        mass = float(b["mass_kg"])
        inertia = (
            mass
            / 12
            * np.diag(
                [
                    3 * b["radius"] ** 2 + (2 * b["half_height"]) ** 2,
                    3 * b["radius"] ** 2 + (2 * b["half_height"]) ** 2,
                    6 * b["radius"] ** 2,
                ]
            )
        )
        body = builder.add_body(
            xform=pose,
            mass=mass,
            inertia=wp.mat33(inertia),
            lock_inertia=True,
            is_kinematic=b["name"] in driven,
            label=b["name"],
        )
        shape = builder.add_shape_capsule(
            body, radius=b["radius"], half_height=b["half_height"], cfg=cfg, label=b["name"]
        )
        if b["name"] in driven:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)
        bodies[b["name"]] = {"body": body, "shape": shape}
    rods = []
    attachments = []
    for s in c["strands"]:
        cfg.density = s["density"]
        rod = newton.Rod(
            s["points"],
            radius=s["radius"],
            **{
                k: s[k]
                for k in ("stretch_rigidity", "shear_rigidity", "bend_rigidity", "twist_rigidity")
            },
        )
        ids, joints = builder.add_rod(
            rod=rod,
            cfg=cfg,
            body_frame_origin="com",
            label=s["name"],
            stretch_damping=s["stretch_damping"],
            bend_damping=s["bend_damping"],
            twist_damping=s["twist_damping"],
        )
        lengths = np.linalg.norm(np.diff(s["points"], axis=0), axis=1)
        rods.append(
            {
                "body_ids": ids,
                "joint_ids": joints,
                "rest_quaternions": np.asarray(rod.quaternions),
                "lengths": lengths,
                "specification": s,
            }
        )
        for a in s["attachments"]:
            body = bodies[a["body"]]["body"]
            i = 0 if a["node"] == 0 else len(ids) - 1
            local_z = -lengths[i] / 2 if i == 0 and a["node"] == 0 else lengths[i] / 2
            point = wp.vec3(*s["points"][a["node"]])
            anchor = wp.transform_point(wp.transform_inverse(builder.body_q[body]), point)
            builder.add_joint_ball(
                body,
                ids[i],
                parent_xform=wp.transform(anchor, wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0, 0, local_z), wp.quat_identity()),
                label=s["name"] + "_endpoint",
            )
            # Explicit attachment neighborhood: outside this small pair set, collisions stay active.
            neighbor_ids = ids[:2] if a["node"] == 0 else ids[-2:]
            for shape_index, owner in enumerate(builder.shape_body):
                if owner in neighbor_ids:
                    builder.add_shape_collision_filter_pair(bodies[a["body"]]["shape"], shape_index)
            attachments.append(
                {
                    "body": body,
                    "rod_body": ids[i],
                    "body_anchor": np.asarray(anchor),
                    "rod_anchor": np.array([0, 0, local_z]),
                }
            )
    builder.add_ground_plane(height=c["environment"]["floor_height"], cfg=cfg)
    for fixture in c["environment"].get("capsule_obstacles", []):
        builder.add_shape_capsule(
            -1,
            xform=wp.transform(
                wp.vec3(*fixture["center_m"]),
                wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*fixture["axis"])),
            ),
            radius=fixture["radius_m"],
            half_height=fixture["half_height_m"],
            cfg=cfg,
            label=fixture["id"],
        )
    builder.color()
    model = builder.finalize()
    history = c["solver"].get("contact_history", False)
    pipeline = newton.CollisionPipeline(
        model, **({"contact_matching": "latest"} if history else {})
    )
    contacts = pipeline.contacts()
    solver = SolverVBD(
        model,
        iterations=c["solver"]["iterations"],
        rigid_compliant_alm=True,
        rigid_contact_history=history,
        collision_pipeline=pipeline,
        rigid_body_contact_buffer_size=128,
    )
    return {
        "model": model,
        "solver": solver,
        "pipeline": pipeline,
        "contacts": contacts,
        "state0": model.state(),
        "state1": model.state(),
        "control": model.control(),
        "rods": rods,
        "bodies": bodies,
        "attachments": attachments,
        "configuration": c,
    }
