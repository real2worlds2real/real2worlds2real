"""Measure declared rigid intrinsic tests in an explicitly unaccepted source scene.

The producer records candidate evidence for an independent semantic reviewer.
It never changes source geometry/materials, writes a passing semantic review,
starts policy training, or issues a policy-development receipt.
"""
import argparse
import json
from pathlib import Path
import traceback

from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors


class UnknownFaceDiagnosticErrors(PhysicsErrors):
    """Sample unresolved reporting faults, preserving a failing native verdict.

    Never use this diagnostic monitor for accepted policy or scene qualification.
    All other solver/reporting errors retain the ordinary fail-fast behavior.
    """
    def check(self):
        self.poll()
        if self.count != self.unknown_face_material_count:
            super().check()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--settled-window", type=float, default=.5)
    parser.add_argument("--force-limit", type=float, default=.02)
    parser.add_argument("--translation", type=float, default=.001)
    parser.add_argument("--stiffness", type=float, default=20.)
    parser.add_argument("--maximum-penetration", type=float, default=.001)
    parser.add_argument("--diagnose-contact-reporting", action="store_true")
    parser.add_argument("--audit-contact-reporting", action="store_true")
    args = parser.parse_args(argv)
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.twin.qualification_inputs import prepare_qualification_scene
    from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant
    session = Session(Path(args.session).resolve())
    output = Path(args.output).resolve()
    if not output.is_relative_to((session.dir / "qa").resolve()) or output.exists():
        raise ValueError("Use a new output directory inside this session QA")
    output.mkdir(parents=True)
    result = {"schema": "astra_intrinsic_assay_batch_v1", "qualified": False,
              "task_approved": False, "policy_receipts_allowed": False, "assays": [],
              "scope": "Candidate intrinsic measurements only; independent source/semantic review remains required."}
    app, errors = None, None
    try:
        runtime = prepare_qualification_scene(session, purpose="intrinsic_assay", output=output / "qualification_inputs.json")
        from calibrated_scene_reconstruction.sim.application import SimulationApp
        app = SimulationApp({"headless": True})
        import omni.usd
        import numpy as np
        from isaacsim.core.api import World
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
        from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors
        from calibrated_scene_reconstruction.sim.scene_stage import build_scene
        from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx
        from calibrated_scene_reconstruction.tasks.rigid.native_assay import run as native_assay_run
        from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration
        from calibrated_scene_reconstruction.scene.objects.rigid_intrinsic import run
        from calibrated_scene_reconstruction.tasks.mechanisms.rigid_probe import validate
        monitor = UnknownFaceDiagnosticErrors if args.diagnose_contact_reporting else PhysicsErrors
        errors = monitor(output / "native_physics_errors.json")
        result["diagnose_contact_reporting"] = bool(args.diagnose_contact_reporting)
        world = World(stage_units_in_meters=1., physics_dt=1 / 120, rendering_dt=1 / 30)
        errors.guard_steps(world)
        stage = omni.usd.get_context().get_stage()
        scene = build_scene(stage, runtime, prepare_variant(session))
        if args.diagnose_contact_reporting and args.audit_contact_reporting:
            raise ValueError('Choose unresolved diagnostic sampling or audited metadata classification, not both')
        if args.audit_contact_reporting:
            result['contact_report_metadata_audit']=errors.audit_contact_report_metadata(stage)
        materials = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            row = {"collider": str(prim.GetPath()), "type": prim.GetTypeName(),
                   "material": str(material.GetPath()) if material else None,
                   "binding": str(relation.GetPath()) if relation else None,
                   "uses_native_default": not bool(material),
                   "geometry_subsets": [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Subset)]}
            if material:
                api = UsdPhysics.MaterialAPI(material.GetPrim())
                row["authored_coefficients"] = {"static_friction": api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": api.GetDynamicFrictionAttr().Get(), "restitution": api.GetRestitutionAttr().Get()}
            materials.append(row)
        result["authored_material_bindings"] = materials
        report_bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                report_bodies.append(str(prim.GetPath()))
        result["contact_instrumentation"] = {"authored_before_first_reset": True,
                                             "report_threshold": 0., "body_paths": report_bodies,
                                             "changes_material_or_collision": False}
        authored = {str(prim.GetPath()): np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                    for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        world.reset()
        initial_context = Ctx(session, scene, world, stage, None)
        initial_state = initial_context.capture_assay_state()
        for name in initial_state["rigid"]:
            body_path = initial_context.body(name).prim_path
            if body_path not in authored:
                raise ValueError("Native assay body has no original authored pose: " + body_path)
            matrix = authored[body_path]
            initial_state["rigid"][name] = (matrix[:3, 3].copy(), matrix[:3, :3].copy())
        assumptions = {"initial_state": "Before each assay, explicitly restore recorded authored pre-physics USD poses and zero velocities; verify native readback within 1e-5m/matrix norm. No pose commands after recording starts.",
                       "attachments": "No temporary attachment, grasp, guide or fixed joint added.",
                       "fixture": "Only the source modeled static support; target objects remain dynamic under native gravity.",
                       "material_uncertainty": "Current source material priors retained; no real mass, stiffness or friction identification.",
                       "contact_uncertainty": "Current source mesh and authored SDF/contact settings unchanged; native errors and penetration recorded.",
                       "source_robot_placement": "Robot is absent from these object intrinsic assays; source robot placement is recorded unchanged in scene graph and source calibration."}
        for obj, spec in session.scene_graph.get("dynamic_objects", {}).items():
            if not (spec.get("fix") or {}).get("model"):
                continue
            item = declaration(session, obj)
            for part in item["parts"]:
                for test in part["physics_tests"]:
                    if test["purpose"] not in {"passive_state", "contact_response", "material_response"}:
                        continue
                    runtime.assert_unchanged()
                    # Keep the native scene/material/contact handles alive.
                    # Reinitializing PhysX after a contact-instrumented trial can
                    # leave stale material references; explicit setup restore
                    # below gives independent starts without resetting runtime.
                    ctx = Ctx(session, scene, world, stage, None)
                    ctx.native_errors = errors
                    ctx.restore_assay_state(initial_state)
                    setup = {}
                    for name, (expected_p, expected_r) in initial_state["rigid"].items():
                        actual_p, actual_r = ctx.pose(name)
                        position_error = float(np.linalg.norm(actual_p - expected_p))
                        rotation_error = float(np.linalg.norm(actual_r - expected_r))
                        if position_error > 1e-5 or rotation_error > 1e-5:
                            raise ValueError("Native independent assay reset differs from authored source pose: " + name)
                        setup[name] = {"position_m": actual_p.tolist(), "rotation": actual_r.tolist(),
                                       "authored_position_m": expected_p.tolist(), "authored_rotation": expected_r.tolist(),
                                       "position_error_m": position_error, "rotation_matrix_error": rotation_error}
                    (output / (test["id"] + ".initial_state.json")).write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
                    plan = None
                    if test["purpose"] != "passive_state":
                        plan = {"link": obj + "/" + part["links"][0], "start_s": .5, "duration_s": .5,
                                "translation_m": [args.translation, 0., 0.], "force_limit_N": args.force_limit,
                                "stiffness_N_m": args.stiffness, "gravity_compensation": False}
                        validate(plan, [obj + "/" + link for link in part["links"]], args.seconds)
                    print("INTRINSIC_BEGIN " + test["id"], flush=True)
                    try:
                        proof = run(ctx, obj, part["id"], test["id"], output / test["id"], seconds=args.seconds,
                                    native_run=native_assay_run,
                                    settled_window_s=args.settled_window, assumptions=assumptions, probe_plan=plan,
                                    maximum_penetration_m=args.maximum_penetration)
                        result["assays"].append({"object": obj, "part_id": part["id"], "test_id": test["id"], **proof})
                    except Exception as error:
                        result["assays"].append({"object": obj, "part_id": part["id"], "test_id": test["id"],
                                                  "pass": False, "error": type(error).__name__ + ": " + str(error),
                                                  "traceback": traceback.format_exc()})
                        print(traceback.format_exc(), flush=True)
                    (output / "batch_progress.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["source_inputs_unchanged"] = runtime.assert_unchanged()
        result["native_physics"] = errors.report()
        result["native_precision"] = errors.precision_audit()
    except Exception as error:
        result["fatal_error"] = type(error).__name__ + ": " + str(error)
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], flush=True)
    finally:
        result["measured_tests_pass"] = bool(result["assays"] and not result.get("fatal_error")
                                             and all(row.get("pass") for row in result["assays"]))
        if errors is not None:
            result["native_physics"] = errors.report()
            errors.close()
        (output / "intrinsic_batch.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print({"intrinsic_batch": str(output / "intrinsic_batch.json"), "measured_tests_pass": result["measured_tests_pass"],
               "tests": len(result["assays"]), "qualified": False}, flush=True)
        if app is not None:
            app.close()  # Native shutdown can terminate Python; result is already durable.


if __name__ == "__main__":
    main()
