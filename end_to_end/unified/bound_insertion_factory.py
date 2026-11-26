"""Composition root: prepare scene-domain inputs, then launch native task runtime."""
from pathlib import Path
import json
import traceback


def build(binding, request, config):
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene
    from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant
    from calibrated_scene_reconstruction.sim.application import SimulationApp
    from calibrated_scene_reconstruction.tasks.rigid.bound_insertion import InsertionEnvironment

    session = Session(binding["native_session"])
    prepared, variant = prepare_scene(session), prepare_variant(session)
    if request.get("schema") == "astra_bound_job_v1":
        diagnostics = Path(binding["oversight_run"]) / "policy_jobs" / request["job_id"]
    elif request.get("operation") == "qualification_probe":
        diagnostics = Path(request["diagnostic_directory"]).resolve(strict=True)
    else:
        raise ValueError("Native insertion requires an owned bound job or explicitly unbound qualification probe")
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": 960, "height": 720})
    try:
        return InsertionEnvironment(binding, request, config, app, session=session, prepared=prepared,
                                    variant=variant, diagnostic_directory=diagnostics)
    except BaseException as error:
        # Native framework shutdown may terminate the interpreter. Persist the
        # exception before closing, otherwise a startup failure looks like exit0.
        report = {"schema": "astra_native_factory_failure_v1", "error": type(error).__name__ + ": " + str(error),
                  "traceback": traceback.format_exc(), "is_policy_receipt": False}
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / "factory_failure.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report["traceback"], flush=True)
        app.close()
        raise
