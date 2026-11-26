"""Build retained scene assets in one native Newton VBD model.

No Blender object category or demonstration trajectory is embedded here.
Rigid construction consumes the same exported frames/collision meshes as the
standalone adapters. Cloth keeps its control topology and detailed binding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .rigid_frames import rotation_only
from .scene import verify_scene
from .sim_parameters import load_parameters


def _transform(matrix):
    import warp as wp

    matrix = np.asarray(matrix, dtype=float)
    return wp.transform(
        wp.vec3(*matrix[:3, 3]), wp.quat_from_matrix(wp.mat33(rotation_only(matrix)))
    )


def _array(value, shape):
    return value.numpy().copy() if value is not None else np.empty(shape, dtype=np.float32)


def _build_frame(folder, builder, parameters, *, drive_joints, sdf_resolution):
    import newton
    import warp as wp

    manifest = json.loads((folder / "manifest.json").read_text())
    parts = json.loads((folder / "collision_meshes.json").read_text())
    rest = {"__base__": np.eye(4)}
    rest.update({item["name"]: np.asarray(item["matrix_world"]) for item in manifest["joints"]})
    bodies, joints, drives = {}, {}, []
    for name, matrix in rest.items():
        owned = [part for part in parts if (part["owner_joint"] or "__base__") == name]
        scale = np.linalg.norm(matrix[:3, :3], axis=0)
        vertices = (
            np.concatenate([np.asarray(part["vertices"]) * scale for part in owned])
            if owned
            else np.zeros((1, 3))
        )
        size = np.maximum(np.ptp(vertices, axis=0), 0.003)
        spec = next((item for item in manifest["joints"] if item["name"] == name), {})
        mass = (
            spec.get("mass_kg")
            or parameters["rigid"]["mass_kg"]
            or float(np.clip(np.prod(size) * parameters["rigid"]["density_kg_m3"], 0.01, 20))
        )
        inertia = mass / 12 * (np.sum(size * size) - size * size)
        body = builder.add_link(
            xform=_transform(matrix),
            mass=mass,
            inertia=wp.mat33(np.diag(inertia)),
            com=wp.vec3(*((vertices.min(0) + vertices.max(0)) / 2)),
            lock_inertia=True,
            label=name,
        )
        bodies[name] = body
        for part in owned:
            mesh = newton.Mesh(
                vertices=np.asarray(part["vertices"], dtype=np.float32) * scale,
                indices=np.asarray(part["faces"], dtype=np.int32).reshape(-1),
            )
            mesh.build_sdf(max_resolution=sdf_resolution, narrow_band_range=(-0.02, 0.02))
            cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=parameters["material"]["friction"],
                restitution=parameters["material"]["restitution"],
            )
            builder.add_shape_mesh(body, mesh=mesh, cfg=cfg, label=part["object_name"])
    chain = [
        builder.add_joint_fixed(
            -1, bodies["__base__"], child_xform=wp.transform_identity(), label="fixed_base"
        )
        if manifest["fixed_base"]
        else builder.add_joint_free(bodies["__base__"], label="free_base")
    ]
    for spec in manifest["joints"]:
        name, parent = spec["name"], spec["parent_joint"] or "__base__"
        kwargs = dict(
            parent=bodies[parent],
            child=bodies[name],
            parent_xform=wp.transform_multiply(
                wp.transform_inverse(_transform(rest[parent])), _transform(rest[name])
            ),
            axis=wp.vec3(*np.eye(3)["XYZ".index(spec["axis"])]),
            label=name,
            target_ke=max(
                100.0, spec.get("drive_stiffness", parameters["joint_drive"]["stiffness"])
            )
            if drive_joints
            else spec.get("drive_stiffness", parameters["joint_drive"]["stiffness"]),
            target_kd=max(5.0, spec.get("drive_damping", parameters["joint_drive"]["damping"]))
            if drive_joints
            else spec.get("drive_damping", parameters["joint_drive"]["damping"]),
        )
        if spec["joint_type"] in {"revolute", "continuous"}:
            lower = math.radians(spec["lower_degrees"] - spec["current_degrees"])
            upper = math.radians(spec["upper_degrees"] - spec["current_degrees"])
            joint = builder.add_joint_revolute(**kwargs, limit_lower=lower, limit_upper=upper)
            excursion = 0.25
        elif spec["joint_type"] == "prismatic":
            lower, upper = spec["lower_m"], spec["upper_m"]
            joint = builder.add_joint_prismatic(**kwargs, limit_lower=lower, limit_upper=upper)
            excursion = 0.03
        else:
            raise ValueError("Unsupported Newton scene joint family: " + spec["joint_type"])
        joints[name] = joint
        delta = min(upper * 0.8, excursion) if upper >= -lower else max(lower * 0.8, -excursion)
        if drive_joints and abs(delta) > 1e-5:
            drives.append({"name": name, "joint": joint, "delta": delta})
        chain.append(joint)
    builder.add_articulation(chain, label="frame")
    return {
        "body_ids": bodies,
        "joint_ids": joints,
        "rest_matrices": rest,
        "drives": drives,
        "attachments": [],
        "loop_anchors": [],
    }


def build_scene(
    folder,
    *,
    device="cuda:0",
    drive_joints=True,
    iterations=None,
    substeps=None,
    disable_environment_collision=False,
    disable_interobject_collision=False,
    sdf_resolution=64,
    body_contact_capacity=None,
):
    import newton
    import warp as wp
    from newton.solvers import SolverVBD

    from .hybrid_runtime import load_hybrid_asset

    folder = Path(folder).resolve()
    for name, value, maximum in (
        ("iterations", iterations, 1000),
        ("substeps", substeps, 128),
        ("body_contact_capacity", body_contact_capacity, 32768),
    ):
        if value is not None and (type(value) is not int or not 1 <= value <= maximum):
            raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    verification = verify_scene(folder)
    if not verification["ok"]:
        raise ValueError("Invalid composed scene: " + str(verification["failures"]))
    spec = json.loads((folder / "request.json").read_text())
    if spec["backend"] != "newton":
        raise ValueError("This adapter requires a Newton scene")
    wp.init()
    wp.set_device(device)
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    SolverVBD.register_custom_attributes(builder)
    records, shape_owner, particle_owner = {}, [], []
    all_parameters, contact_radii, soft_parameters = [], [], []
    for entry in spec["objects"]:
        asset = folder / "assets" / entry["id"]
        bundle = json.loads((asset / "asset_bundle.json").read_text())
        parameters = load_parameters(asset)
        all_parameters.append(parameters)
        if not bundle.get("native_representation") and not np.allclose(parameters["environment"]["gravity"], (0, 0, -9.81)):
            raise ValueError(
                "This scene adapter uses shared gravity (0,0,-9.81); adapt other environments explicitly"
            )
        kind = bundle["asset_kind"]
        native_mode = bundle.get("native_representation")
        local = newton.ModelBuilder(gravity=(0, 0, -9.81))
        SolverVBD.register_custom_attributes(local)
        record = {
            "body_ids": {},
            "joint_ids": {},
            "rest_matrices": {},
            "drives": [],
            "attachments": [],
            "loop_anchors": [],
        }
        if native_mode:
            from .scene_native_runtime import build_part
            local, record = build_part(asset, device=device, sdf_cache=folder / ".cache/native-sdf",
                                       action=spec.get("actions", {}).get(entry["id"]))
        elif kind == "hybrid":
            record = load_hybrid_asset(asset, device=device, build_only=True)
            local = record.pop("builder")
            record["drives"] = []  # Retained hybrid motion is handled separately below.
        elif kind in {"rigid", "articulated"}:
            record = _build_frame(
                asset, local, parameters, drive_joints=drive_joints, sdf_resolution=sdf_resolution
            )
        if kind == "deformable" and not native_mode:
            mesh = json.loads((asset / "sim_mesh.json").read_text())
            surface = parameters["surface"]
            local.add_cloth_mesh(
                pos=wp.vec3(0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0),
                vertices=mesh["vertices"],
                indices=np.asarray(mesh["faces"], dtype=np.int32).reshape(-1),
                **{
                    key: surface[key]
                    for key in (
                        "density",
                        "tri_ke",
                        "tri_ka",
                        "tri_kd",
                        "edge_ke",
                        "edge_kd",
                        "particle_radius",
                    )
                },
            )
            if parameters["constraints"]["use_source_pins"]:
                for index in mesh["pin_indices"]:
                    local.particle_mass[index] = 0.0
            record["control_mesh"] = mesh
        if local.particle_count:
            contact_radii.append(parameters["surface"]["particle_radius"])
            soft_parameters.append(parameters)
        body_offset, particle_offset, joint_offset = (
            builder.body_count,
            builder.particle_count,
            builder.joint_count,
        )
        shape_offset = len(shape_owner)
        shape_owner.extend([entry["id"]] * local.shape_count)
        if native_mode:
            record["native_shape_ids"] = {label: shape_offset+i for i, label in enumerate(local.shape_label)}
            for probe in record.get("native_test_probes", []):
                body = record["native_probe_ids"][probe["id"]]
                for i, owner in enumerate(local.shape_body):
                    if owner == body:
                        shape_owner[shape_offset+i] = "fixture:"+entry["id"]+":"+probe["id"]
        particle_owner.extend([entry["id"]] * local.particle_count)
        builder.add_builder(local, xform=_transform(entry["transform"]), label_prefix=entry["id"])
        record.update(
            id=entry["id"],
            kind=kind,
            transform=entry["transform"],
            folder=str(asset),
            body_offset=body_offset,
            particle_start=particle_offset,
            particle_end=builder.particle_count,
            parameters=parameters,
            scene_action=spec.get("actions", {}).get(entry["id"]),
        )
        record["body_ids"] = {
            name: value + body_offset if value >= 0 else -1 for name, value in record["body_ids"].items()
        }
        if native_mode:
            record["native_probe_ids"] = {name: i+body_offset for name, i in record.get("native_probe_ids", {}).items()}
        record["joint_ids"] = {
            name: value + joint_offset for name, value in record["joint_ids"].items()
        }
        record["drives"] = [
            {**drive, "joint": drive["joint"] + joint_offset} for drive in record["drives"]
        ]
        record["attachments"] = [
            (particle + particle_offset, body + body_offset, anchor)
            for particle, body, anchor in record["attachments"]
        ]
        record["loop_anchors"] = [
            (a + body_offset, b + body_offset, p, q) for a, b, p, q in record["loop_anchors"]
        ]
        records[entry["id"]] = record
    if disable_interobject_collision:
        # Negative control: disable particle contact on participant shapes,
        # retaining support contacts. Hybrid internal contact is disabled too.
        for shape in range(len(shape_owner)):
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for first, owner in enumerate(shape_owner):
            for second in range(first):
                if owner != shape_owner[second]:
                    builder.add_shape_collision_filter_pair(first, second)
    for support in spec["supports"]:
        if disable_environment_collision:
            continue
        half = np.array(support["size_m"]) / 2
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=support["dynamic_friction"], restitution=support["restitution"]
        )
        builder.add_shape_box(
            -1,
            xform=_transform(support["transform"]),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=cfg,
            label=support["id"],
        )
        shape_owner.append("environment:" + support["id"])
    builder.color(include_bending=True)
    model = builder.finalize()
    native_materials = [r for r in records.values() if r.get("native_representation") in {"surface", "volume"}]
    active_materials = [r for r in native_materials if r["parameters"]["solver"]["self_contact"]]
    shared_contact = None
    if native_materials:
        # Native VBD exposes one particle-side contact material per model.
        # Preserve the active self-contact material first, and report the shared
        # assumption explicitly rather than silently inheriting model defaults.
        chosen = (active_materials or native_materials)[0]
        env = chosen["native_configuration"].get("environment", {})
        shared_contact = {"stiffness":env.get("contact_stiffness",2e5),
                          "damping":env.get("contact_damping",.1),
                          "friction":env.get("friction",.5),"source_asset":chosen["id"]}
        for other in active_materials:
            e = other["native_configuration"].get("environment", {})
            if not np.allclose([e.get("contact_stiffness",2e5),e.get("contact_damping",.1),e.get("friction",.5)],
                               [shared_contact["stiffness"],shared_contact["damping"],shared_contact["friction"]]):
                raise ValueError("Active native self-contact materials disagree; supply a compatible shared contact adapter")
        model.soft_contact_ke = float(shared_contact["stiffness"])
        model.soft_contact_kd = float(shared_contact["damping"])
        model.soft_contact_mu = float(shared_contact["friction"])
    full_surface = any(
        (record["kind"] == "hybrid" and record["parameters"]["solver"]["hybrid_full_surface_contact"])
        or record.get("native_configuration", {}).get("solver", {}).get("full_surface_contact", False)
        for record in records.values()
    )
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=full_surface)
    pipeline.contacts()
    radius = min(contact_radii, default=0.008)
    actual_iterations = iterations or max(p["solver"]["iterations"] for p in all_parameters)
    actual_substeps = substeps or max(p["solver"]["substeps"] for p in all_parameters)
    native_soft = [r for r in records.values() if r.get("native_representation") in {"surface", "volume"}]
    exclusions, edge_exclusions = {}, {}
    native_contact = {}
    if native_soft:
        from .scene_native_runtime import excluded_self_pairs
        exclusions, edge_exclusions = excluded_self_pairs(builder, records)
        active = [r for r in native_soft if r["parameters"]["solver"]["self_contact"]]
        distances = {(r["native_configuration"].get("solver", {}).get("self_contact_margin_m", 2*r["native_configuration"]["material"]["contact_radius_m"]),
                      r["native_configuration"].get("solver", {}).get("self_contact_gap_m", r["native_configuration"]["material"]["contact_radius_m"])) for r in active}
        if len(distances) > 1:
            raise ValueError("Native surfaces request different self-contact distances; configure an explicitly compatible shared model")
        margin, gap = next(iter(distances), (2*radius, radius))
        native_contact = {"particle_self_contact_margin": margin, "particle_self_contact_gap": gap}
    else:
        native_contact = {"particle_self_contact_radius": radius, "particle_self_contact_margin": 2*radius}
    solver = SolverVBD(
        model,
        iterations=actual_iterations,
        rigid_compliant_alm=True,
        collision_pipeline=pipeline,
        particle_enable_self_contact=any(p["solver"]["self_contact"] for p in soft_parameters),
        **native_contact,
        particle_external_vertex_contact_filtering_map=exclusions,
        particle_external_edge_contact_filtering_map=edge_exclusions,
        particle_vertex_contact_buffer_size=max(
            (p["solver"]["self_contact_vertex_capacity"] for p in soft_parameters), default=64
        ),
        particle_edge_contact_buffer_size=max(
            (p["solver"]["self_contact_edge_capacity"] for p in soft_parameters), default=128
        ),
        rigid_body_contact_buffer_size=body_contact_capacity
        or max(128, min(8192, model.shape_count * 128)),
        rigid_body_particle_contact_buffer_size=max(1024, model.particle_count * 8),
    )
    observable = solver.observables([solver.ObservableKind.CONTACT_F], contacts=solver.contacts)
    particle_rates, body_rates = np.zeros(model.particle_count), np.zeros(model.body_count)
    for record in records.values():
        rate = record["parameters"]["environment"]["air_drag_rate_per_second"]
        particle_rates[record["particle_start"] : record["particle_end"]] = rate
        for body in record["body_ids"].values():
            if body >= 0:
                body_rates[body] = rate
    return {
        "model": model,
        "solver": solver,
        "control": model.control(),
        "records": records,
        "shape_owner": shape_owner,
        "particle_owner": particle_owner,
        "observables": observable,
        "substeps": actual_substeps,
        "iterations": actual_iterations,
        "spec": spec,
        "particle_drag_rates": wp.array(particle_rates, dtype=float),
        "body_drag_rates": wp.array(body_rates, dtype=float),
        "has_air_drag": bool(particle_rates.any() or body_rates.any()),
        "full_surface_contact": full_surface,
        "shared_contact_material":shared_contact,
        "negative_controls": {
            "environment_collision_disabled": disable_environment_collision,
            "interobject_collision_disabled": disable_interobject_collision,
        },
    }


def _contact_evidence(runtime, pairs):
    contacts, owners = runtime["solver"].contacts, runtime["shape_owner"]
    forces = runtime["observables"].contact_f.numpy()
    count = int(contacts.soft_contact_count.numpy()[0])
    if count > contacts.soft_contact_max:
        raise RuntimeError("Soft contact buffer overflow")
    if count:
        shapes = contacts.soft_contact_shape.numpy()[:count]
        particles = contacts.soft_contact_indices.numpy()[:count, 0]
        # A full-surface edge/face record stays within one asset's topology.
        for i, (shape, particle) in enumerate(zip(shapes, particles, strict=True)):
            if shape < 0 or particle < 0:
                continue
            force = float(np.linalg.norm(forces[contacts.rigid_contact_max + i, :3]))
            a, b = owners[shape], runtime["particle_owner"][particle]
            if a != b and force > 1e-7:
                key = tuple(sorted((a, b)))
                pairs[key] = max(force, pairs.get(key, 0.0))
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count > contacts.rigid_contact_max:
        raise RuntimeError("Rigid contact buffer overflow")
    if count:
        left, right = (
            contacts.rigid_contact_shape0.numpy()[:count],
            contacts.rigid_contact_shape1.numpy()[:count],
        )
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if a < 0 or b < 0:
                continue
            force = float(np.linalg.norm(forces[i, :3]))
            if owners[a] != owners[b] and force > 1e-7:
                key = tuple(sorted((owners[a], owners[b])))
                pairs[key] = max(force, pairs.get(key, 0.0))


def simulate_scene(folder, output, *, frames=120, fps=30, capture_graph=False, **options):
    import newton
    import warp as wp

    folder, output = Path(folder).resolve(), Path(output).resolve()
    if not 30 <= frames <= 1800 or not 1 <= fps <= 120:
        raise ValueError("Use 30–1800 continuous frames at 1–120 FPS")
    output.mkdir(parents=True, exist_ok=True)
    runtime = build_scene(folder, **options)
    model, solver, control = (runtime[k] for k in ("model", "solver", "control"))
    state, next_state = model.state(), model.state()
    initial = _array(state.particle_q, (0, 3))
    poses, particles, joints, pairs = [], [], [], {}
    targets = _array(control.joint_target_q, (0,))
    q_starts = _array(model.joint_q_start, (0,)).astype(np.int64)
    dt = 1 / (fps * runtime["substeps"])
    from .scene_test_controls import SceneTestControls
    test_controls = SceneTestControls(runtime)
    if capture_graph and not model.device.is_cuda:
        raise ValueError("CUDA graph replay requires a CUDA device")
    # Capture only solver work. Time-dependent controls remain outside the graph
    # and are applied at every physical substep, never frozen at capture time.
    step_graphs = {}
    capacity_maxima = wp.zeros(2, dtype=wp.int32, device=model.device)
    from .native_controls import retain_contact_capacity_maxima
    edge_records = {}
    for name, record in runtime["records"].items():
        if record.get("control_mesh"):
            faces = np.asarray(record["control_mesh"]["faces"])
            edges = (
                np.unique(
                    np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0
                )
                + record["particle_start"]
            )
            lengths = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
            if (lengths < 1e-8).any():
                raise ValueError("Degenerate control edges")
            edge_records[name] = (edges, lengths)
    stretch, max_attachment_gap, max_loop_gap = {name: 1.0 for name in edge_records}, 0.0, 0.0
    overflow_max, rigid_overflow_max = 0, 0
    for frame in range(frames):
        if frame:
            time = frame / fps
            values = targets.copy()
            for record in runtime["records"].values():
                for drive in record["drives"]:
                    if record.get("scene_action"):
                        continue
                    values[q_starts[drive["joint"]]] = (
                        targets[q_starts[drive["joint"]]]
                        + drive["delta"] * math.sin(math.pi * time / ((frames - 1) / fps)) ** 2
                    )
                if record["kind"] == "hybrid" and options.get("drive_joints", True):
                    for motion in record["configuration"]["motion"]:
                        keys = np.asarray(motion["keyframes"])
                        values[q_starts[record["joint_ids"][motion["joint"]]]] = np.interp(
                            time, keys[:, 0], keys[:, 1]
                        )
            if control.joint_target_q is not None:
                test_controls.joint_targets(values, q_starts, time)
                control.joint_target_q.assign(values)
            for substep in range(runtime["substeps"]):
                state.clear_forces()
                if runtime["has_air_drag"]:
                    from .newton_forces import body_air_drag_rates, particle_air_drag_rates

                    if model.particle_count:
                        wp.launch(
                            particle_air_drag_rates,
                            dim=model.particle_count,
                            inputs=[
                                state.particle_qd,
                                model.particle_inv_mass,
                                runtime["particle_drag_rates"],
                            ],
                            outputs=[state.particle_f],
                        )
                    if model.body_count:
                        wp.launch(
                            body_air_drag_rates,
                            dim=model.body_count,
                            inputs=[state.body_qd, model.body_mass, runtime["body_drag_rates"]],
                            outputs=[state.body_f],
                        )
                test_controls.apply(state, next_state, (frame-1)/fps+(substep+1)*dt, dt)
                parity = id(state)
                if capture_graph and frame + substep > 1:
                    if parity not in step_graphs:
                        with wp.ScopedCapture(device=model.device) as capture:
                            solver.step(state, next_state, control, None, dt,
                                        observables=runtime["observables"])
                        step_graphs[parity] = capture.graph
                    wp.capture_launch(step_graphs[parity])
                else:
                    solver.step(state, next_state, control, None, dt,
                                observables=runtime["observables"])
                wp.launch(retain_contact_capacity_maxima, dim=1,
                          inputs=[solver.body_body_contact_overflow_max,
                                  solver.body_particle_contact_overflow_max, capacity_maxima])
                state, next_state = next_state, state
            _contact_evidence(runtime, pairs)
        if model.joint_count:
            newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        points, bodies = _array(state.particle_q, (0, 3)), _array(state.body_q, (0, 7))
        if not np.isfinite(points).all() or not np.isfinite(bodies).all():
            raise RuntimeError("Nonfinite Newton scene state")
        for name, (edges, lengths) in edge_records.items():
            stretch[name] = max(
                stretch[name],
                float(
                    np.quantile(
                        np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1) / lengths,
                        0.95,
                    )
                ),
            )
        for record in runtime["records"].values():
            for particle, body, anchor in record["attachments"]:
                max_attachment_gap = max(
                    max_attachment_gap,
                    float(
                        np.linalg.norm(
                            points[particle]
                            - np.asarray(
                                wp.transform_point(
                                    wp.transform(
                                        *[wp.vec3(*bodies[body, :3]), wp.quat(*bodies[body, 3:])]
                                    ),
                                    wp.vec3(*anchor),
                                )
                            )
                        )
                    ),
                )
            for a, b, p, q in record["loop_anchors"]:
                left = wp.transform_point(
                    wp.transform(wp.vec3(*bodies[a, :3]), wp.quat(*bodies[a, 3:])), wp.vec3(*p)
                )
                right = wp.transform_point(
                    wp.transform(wp.vec3(*bodies[b, :3]), wp.quat(*bodies[b, 3:])), wp.vec3(*q)
                )
                max_loop_gap = max(
                    max_loop_gap, float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
                )
        poses.append(bodies.copy())
        particles.append(points.copy())
        joints.append(_array(state.joint_q, (0,)))
        if frame % 30 == 0:
            print(f"NEWTON_SCENE_FRAME {frame}/{frames} contact_pairs={len(pairs)}", flush=True)
    rigid_overflow_max, overflow_max = map(int, capacity_maxima.numpy())
    required = json.loads((folder / "scene_bundle.json").read_text())["required_capabilities"]
    environment = [
        (a, b) for a, b in pairs if a.startswith("environment:") or b.startswith("environment:")
    ]
    interobject = [
        (a, b)
        for a, b in pairs
        if not a.startswith("environment:") and not b.startswith("environment:")
    ]
    joint_checks = []
    for record in runtime["records"].values():
        specs = {
            spec["name"]: spec
            for spec in json.loads((Path(record["folder"]) / "manifest.json").read_text())["joints"]
        }
        for name, joint in record["joint_ids"].items():
            values = np.asarray(joints)[:, q_starts[joint]]
            span = float(np.ptp(values))
            spec = specs[name]
            if spec["joint_type"] in {"revolute", "continuous"}:
                lower = math.radians(spec["lower_degrees"] - spec["current_degrees"])
                upper = math.radians(spec["upper_degrees"] - spec["current_degrees"])
                tolerance = 0.04
            else:
                lower, upper, tolerance = spec["lower_m"], spec["upper_m"], 0.005
            limits_ok = values.min() >= lower - tolerance and values.max() <= upper + tolerance
            action = record.get("scene_action", {}) or {}
            expected = next((j for j in action.get("joints", []) if j["joint"] == name), None)
            action_ok, expected_span, hold_error = True, None, None
            if expected is not None:
                keys = np.asarray(expected["keyframes"], dtype=float)
                expected_span = float(np.ptp(keys[:,1]))
                local_time = np.arange(frames)/fps-action["start_seconds"]
                errors = []
                for before, after in zip(keys[:-1], keys[1:], strict=True):
                    if abs(before[1]-after[1]) < 1e-8:
                        ids = (local_time >= (before[0]+after[0])/2) & (local_time <= after[0])
                        if ids.any():
                            errors.append(float(np.abs(values[ids]-after[1]).max()))
                hold_error = max(errors, default=0.)
                action_ok = (local_time[-1] >= keys[-1,0] and span >= .9*expected_span
                             and hold_error <= tolerance)
            joint_checks.append(
                {
                    "asset": record["id"],
                    "joint": name,
                    "span": span,
                    "ok": bool(span > 1e-4 and limits_ok and action_ok),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "lower_limit": lower,
                    "upper_limit": upper,
                    "limit_tolerance": tolerance,
                    "expected_action_span":expected_span,
                    "maximum_hold_target_error":hold_error,
                }
            )
    checks = [
        {"name": "scene_loaded", "ok": True},
        {"name": "finite_state", "ok": True},
        {
            "name": "environment_contacts",
            "ok": bool(environment),
            "pairs": environment,
            "method": "nonzero Newton contact force at sampled states",
        },
        {
            "name": "deformation_integrity",
            "ok": all(value < 2 for value in stretch.values()),
            "edge_ratio_p95_max": stretch,
        },
        {
            "name": "contact_capacity",
            "ok": overflow_max <= getattr(solver, "body_particle_contact_buffer_pre_alloc", 0)
            and rigid_overflow_max <= getattr(solver, "body_body_contact_buffer_pre_alloc", 0),
            "overflow_max": overflow_max,
            "rigid_count_max": rigid_overflow_max,
            "rigid_per_body_capacity": getattr(solver, "body_body_contact_buffer_pre_alloc", 0),
            "soft_per_body_capacity": getattr(solver, "body_particle_contact_buffer_pre_alloc", 0),
        },
    ]
    if "articulated" in required or "hybrid" in required:
        checks.append(
            {
                "name": "joint_motion",
                "ok": bool(joint_checks) and all(item["ok"] for item in joint_checks),
                "joints": joint_checks,
            }
        )
    if "rigid_deformable_contact" in required:
        soft_kinds = {"deformable", "hybrid", "solid"}
        meaningful = [
            pair
            for pair in interobject
            if any(runtime["records"].get(v, {}).get("kind") in soft_kinds for v in pair)
        ]
        checks.append(
            {
                "name": "rigid_deformable_contact",
                "ok": bool(meaningful),
                "pairs": meaningful,
                "scope": "Actual nonzero force between distinct asset IDs; not every possible task contact is certified",
            }
        )
    if "hybrid" in required:
        checks.append(
            {
                "name": "attachment_integrity",
                "ok": max_attachment_gap < 0.02 and max_loop_gap < 0.01,
                "max_attachment_gap_m": max_attachment_gap,
                "max_loop_anchor_gap_m": max_loop_gap,
            }
        )
    from .scene_native_review import review_native
    native_reviews = review_native(runtime["records"], np.asarray(poses), np.asarray(particles), np.arange(frames)/fps)
    if native_reviews:
        checks.append({"name":"native_source_behavior", "ok":all(r["ok"] for r in native_reviews),
                       "assets":native_reviews,
                       "scope":"Original source behavior windows, topology and retained-frame fixture checks reapplied in the shared world; no physical calibration."})
    # Plain JSON bindings let a renderer consume trajectories without importing Newton.
    records = {
        name: {
            key: record[key]
            for key in ("id", "kind", "body_ids", "particle_start", "particle_end", "transform")
        }
        | {"rest_matrices": {k: np.asarray(v).tolist() for k, v in record["rest_matrices"].items()}}
        for name, record in runtime["records"].items()
    }
    for name, r in runtime["records"].items():
        if r.get("native_representation"):
            records[name]["native_representation"] = r["native_representation"]
            records[name]["native_action"] = r.get("native_action")
            if r["native_representation"] == "rigid_contact":
                records[name]["native_source_bindings"] = r["native_source_bindings"]
                records[name]["native_static_poses"] = r["native_static_poses"]
            else:
                records[name]["native_probe_ids"] = r.get("native_probe_ids", {})
                records[name]["native_test_probes"] = r.get("native_test_probes", [])
    atomic_json(output / "trajectory_bindings.json", {"schema_version": 1, "assets": records})
    np.savez_compressed(
        output / "scene_motion.npz",
        bodies=np.asarray(poses),
        particles=np.asarray(particles),
        joints=np.asarray(joints),
        times=np.arange(frames) / fps,
    )
    report = {
        "schema_version": 1,
        "ok": all(check["ok"] for check in checks),
        "runtime": {"name": "native Newton unified VBD", "version": newton.__version__},
        "scene_sha256": sha256_file(folder / "scene.usda"),
        "checked_asset_ids": list(runtime["records"]),
        "checks": checks,
        "images": [],
        "videos": [],
        "frames": frames,
        "fps": fps,
        "substeps": runtime["substeps"],
        "iterations": runtime["iterations"],
        "solver_graph_replay": bool(capture_graph),
        "shared_environment": {"gravity":[0,0,-9.81], "source_gravity":{
            name:r["parameters"]["environment"]["gravity"] for name,r in runtime["records"].items()},
            "particle_contact_material":runtime["shared_contact_material"],
            "scope":"Shared gravity and a shared particle-side contact material are scene assumptions; per-shape friction remains retained. Other standalone diagnostic environments are not silently claimed identical."},
        "body_properties": {name:{label:{"body_id":body, "static":body<0,
            "mass_kg":float(model.body_mass.numpy()[body]) if body>=0 else None}
            for label,body in r["body_ids"].items()} for name,r in runtime["records"].items()},
        "contact_pairs": [
            {"assets": list(pair), "maximum_sampled_force_n": force}
            for pair, force in pairs.items()
        ],
        "full_surface_contact": runtime["full_surface_contact"],
        "negative_controls": runtime["negative_controls"],
        "real_physics_calibrated": False,
        "scope": "One physical solver for all participants; target rendering replays actual states without advancing physics.",
    }
    atomic_json(output / "physics_report.json", report)
    return report
