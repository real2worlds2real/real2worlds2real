"""The twin's options and task information (docs/twin_pipeline.md), for pipeline/run_session.sh and the stages.

    python -m calibrated_scene_reconstruction.twin sessions/<name> get background|scale|scale_check|cousins|task_look
    python -m calibrated_scene_reconstruction.twin sessions/<name> gate <stage>        exit 0 when qa/review/<stage>_r<k>.json says pass
    python -m calibrated_scene_reconstruction.twin sessions/<name> record <review.json>   the agent's review round -> qa/review/<stage>_r<k>.json
    python -m calibrated_scene_reconstruction.scene.twin.gates sessions/<name> status  read-only gate and object-loop results
    python -m calibrated_scene_reconstruction.twin sessions/<name> show

configs/<name>/twin.json: {"background": "splat | mesh | completion | zone", "scale": "board | robot_joints | robot_parts |
prior", "scale_check": one of the scales (default: the best other available)}. Without it the session
keeps the pipeline's earlier behaviour (no gates; --splat and completion.json decide the background). Defaults when it
exists: background `mesh` for a video, `completion` for a photo; scale `board` with a board config, else
`robot_joints` with a joint readout, else `robot_parts`. No measured size is ever an input or a check.
configs/<name>/task.json: the manipulated objects, the task and the support (all optional; "inferred":
true when the agent wrote it).
"""
import json
import sys
from pathlib import Path

from calibrated_scene_reconstruction.scene.session import Session

BACKGROUNDS = ("splat", "mesh", "completion", "zone", "blender")
SCALES = ("board", "robot_joints", "robot_visual", "robot_parts", "prior")  # no measured task sizes
LOOKS = ("kit", "frames")  # the task objects' look: the kit's flat colour, or textures from the walk's frames


def options(s):
    """The twin's options with their defaults, or None when the session has no twin.json (the earlier pipeline)."""
    from calibrated_scene_reconstruction.scene.capture.inputs import validate_session

    validate_session(s)
    if not (s.config_dir / "twin.json").exists():
        return None
    cfg = s.config("twin.json")
    video = (s / "capture" / "keyframes.json").exists()
    have = {"board": (s.config_dir / "charuco_board.json").exists(),
            "robot_joints": (s.config_dir / "robot_state.json").exists(),
            }
    scale = cfg.get("scale") or next((k for k in ("board", "robot_joints") if have[k]), "robot_parts")
    check_order = ("robot_parts", "prior") if scale == 'robot_visual' else ("robot_joints", "board", "robot_parts")
    out = {"background": cfg.get("background", "mesh" if video else "completion"), "scale": scale,
           "scale_check": cfg.get("scale_check") or next(
               (k for k in check_order if k != scale and
                (have.get(k, True))), "prior"),
           "cousins": int(cfg.get("cousins", 0)), "task_look": cfg.get("task_look", "kit")}
    out['support_route'] = cfg.get('support_route', 'agent_blender' if scale == 'robot_visual' else 'carve')
    out['task_validation'] = 'object_model'
    if 'environment_quality_version' in cfg:
        version = cfg['environment_quality_version']
        if type(version) is not int or version != 1:
            raise SystemExit('environment_quality_version must be1')
        out['environment_quality_version'] = version
    if out['support_route'] not in ('agent_blender', 'carve'):
        raise SystemExit('twin.json support_route must be agent_blender or carve')
    if out["task_look"] not in LOOKS:
        raise SystemExit(f"twin.json task_look {out['task_look']!r}: one of {LOOKS}")
    if out["background"] not in BACKGROUNDS:
        raise SystemExit(f"twin.json background {out['background']!r}: one of {BACKGROUNDS}")
    if out['background'] == 'blender' and out.get('environment_quality_version') != 1:
        raise SystemExit('Agent Blender environments require environment_quality_version:1')
    for k in ("scale", "scale_check"):
        if out[k] not in SCALES:
            raise SystemExit(f"twin.json {k} {out[k]!r}: one of {SCALES}")
    if out["scale"] in have and not have[out["scale"]]:
        raise SystemExit(f"twin.json scale {out['scale']!r} has no input: "
                         + {"board": "configs/<name>/charuco_board.json", "robot_joints": "configs/<name>/robot_state.json"}[out["scale"]])
    return out


def meshed_structures(s):
    """Whether the work zone's other structures (a frame, a fixture, cables: scene graph `rigs`, `structures`) are
    built as meshes. Only what the task touches -- the support, the robot, the manipulated objects -- has to be a mesh
    at its measured size and place: with background `splat` the rest is the splat's, with `zone` it is left out (the
    user, 2026-10-27). With `mesh` and `completion` (no splat) they are meshes; without twin.json as before."""
    o = options(s)
    return o is None or o["background"] in ("mesh", "completion")


def task(s):
    return s.config("task.json", {}) or {}


def review(s, stage):
    """The latest review record of a stage (qa/review/<stage>_r<k>.json), or None."""
    recs = sorted((s / "qa" / "review").glob(f"{stage}_r*.json"),
                  key=lambda p: int(p.stem.rsplit("_r", 1)[1]) if p.stem.rsplit("_r", 1)[1].isdigit() else -1)
    return json.loads(recs[-1].read_text()) if recs else None


def status(s):
    """Read all applicable gates without changing artifacts or review timestamps."""
    from .reviews import status as evaluate
    from calibrated_scene_reconstruction.scene.objects.blender.export import verify_round

    opts = options(s)
    graph = s.config("scene_graph.json", {}) if opts is not None else {}
    return evaluate(s, opts, graph, task(s), verify_round)


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        return
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m calibrated_scene_reconstruction.twin SESSION {get,gate,record,show,status}; use --help")
    s = Session(sys.argv[1])
    cmd = sys.argv[2]
    opts = options(s)
    if cmd == "get":
        print("" if opts is None else opts[sys.argv[3]])
    elif cmd == "status":
        print(json.dumps(status(s), indent=2, ensure_ascii=False))
    elif cmd == "gate":
        if len(sys.argv) < 4:
            raise SystemExit("gate requires a stage name")
        stage = sys.argv[3]
        if opts is None:
            if stage in {'S5','S6'}:
                from .process_result import state as process_state
                process = process_state(s,task(s))
                if not process['passed']:
                    raise SystemExit(f"[gate] {stage}: blocked: Physical process: {process['reason']}")
            print(f"[gate] {stage}: legacy workflow has no twin gates")
            return
        result = status(s)
        state = result["stages"].get(stage)
        if state is None and stage.startswith("OBJ."):
            state = result["objects"].get(stage[4:])
        if state is None and stage.startswith('SUP.'):
            state = result['supports'].get(stage[4:])
        if state is None:
            raise SystemExit(f"Unknown or inactive gate {stage!r}")
        print(f"[gate] {stage}: {state['state']}: {state['reason']}", flush=True)
        sys.exit(0 if state["passed"] else 1)
    elif cmd == "record":  # the agent's review round: its json in, numbered and stored as qa/review/<stage>_r<k>.json
        rec = json.loads(Path(sys.argv[3]).read_text())
        stage = rec["stage"]
        for k in ("checks", "review", "findings", "pass"):
            if k not in rec:
                raise SystemExit(f"a review record needs {k!r}")
        if not isinstance(rec["pass"], bool) or not isinstance(rec["findings"], list):
            raise SystemExit("pass must be boolean and findings a list")
        if any(not isinstance(f, dict) or not f.get("action") for f in rec["findings"]):
            raise SystemExit("every finding requires an action")
        if rec["pass"] and any(f.get("status") != "resolved" for f in rec["findings"]):
            raise SystemExit("unresolved findings cannot pass")
        from .reviews import latest_record
        previous = latest_record(s, stage)
        latest_round = int(previous.stem.rsplit('_r', 1)[1]) if previous else 0
        number = rec.get('round', latest_round + 1)
        if type(number) is not int or number <= latest_round:
            raise SystemExit('Review round must append after the latest filename number')
        rec['round'] = number
        if stage == 'S2' and rec['pass']:
            from .inventory import state as inventory_state
            inventory = inventory_state(s, s.config('scene_graph.json', {}), task(s))
            if not inventory['passed']:
                raise SystemExit('Object inventory: '+inventory['reason'])
            if inventory['applicable']:
                rec['object_inventory_sha256'] = inventory['sha256']
        if stage == 'S0' and rec['pass']:
            from calibrated_scene_reconstruction.scene.capture.exposure import current as current_exposure
            rec['exposure_review_sha256'] = current_exposure(s)
            frame_path = s/'parsing/world_frame.json'
            if frame_path.exists() and 'observed_auxiliary_floor' in json.loads(frame_path.read_text()):
                from calibrated_scene_reconstruction.scene.capture.draft_floor import verify_active
                verify_active(s)
        if (stage == 'S3' and rec['pass'] and opts is not None
                and opts.get('environment_quality_version') == 1 and opts['background'] != 'zone'):
            from calibrated_scene_reconstruction.scene.review.environment import current as environment_quality
            _, rec['environment_quality_sha256'] = environment_quality(s)
        if stage == 'S6' and rec['pass']:
            from .process_result import state as process_state
            process = process_state(s, task(s))
            if not process['passed']:
                raise SystemExit('Physical process: '+process['reason'])
        if stage == 'S6' and rec['pass'] and opts is not None:
            current = status(s)
            unmet = [k for k, v in current['stages'].items() if k != 'S6' and v['applicable'] and not v['passed']]
            unmet += ['OBJ.' + k for k, v in current['objects'].items() if not v['passed']]
            unmet += ['SUP.' + k for k, v in current.get('supports', {}).items() if not v['passed']]
            if unmet:
                raise SystemExit('Delivery prerequisites failed: ' + ', '.join(unmet))
            if task(s).get('mechanism_contract'):
                from .reviews import delivery_package
                rec['package_report_sha256'] = delivery_package(s, rec)
            if opts['background']=='blender':
                from calibrated_scene_reconstruction.scene.background.environment_physics import current as physical_current
                _,rec['environment_delivery_physics_sha256']=physical_current(s,delivery=True)
        object_event = None
        if stage.startswith('SUP.'):
            from calibrated_scene_reconstruction.scene.support.blender import verify, model_path, sha
            name = stage[4:]
            if name != s.scene_graph.get('task', {}).get('support'):
                raise SystemExit('Support review must name the current task support')
            rec['model_sha256'] = sha(model_path(s))
            if rec['pass']:
                result = verify(s, name)
                rec.update(model_round=result['round'], mesh_sha256=result['mesh_sha256'],
                           visual_sha256=result['visual_sha256'])
                from calibrated_scene_reconstruction.scene.objects.object_understanding import current as semantic_current
                from calibrated_scene_reconstruction.scene.objects.structural_review import current as structural_current
                directory=s/'support'/name/f"round_{result['round']}"
                rec['semantic_review_sha256']=semantic_current(s,name,directory,result['build'])
                rec['structural_review_sha256']=structural_current(s,name,directory)
        if stage.startswith("OBJ."):
            obj = stage[4:]
            spec = s.scene_graph.get("dynamic_objects", {}).get(obj, {})
            if (spec.get("fit") or {}).get("model"):
                from calibrated_scene_reconstruction.scene.objects.blender.fit import model_path
                from calibrated_scene_reconstruction.scene.objects.blender.contract import contract
                from calibrated_scene_reconstruction.scene.objects.blender.store import event
                fingerprint = contract(model_path(s, obj))["sha256"]
                rec["model_sha256"] = fingerprint
                if rec["pass"]:
                    fx = spec.get("fix") or {}
                    if fx.get("model_sha256") != fingerprint:
                        raise SystemExit("the reviewed model has not been exported at its current content hash")
                    from calibrated_scene_reconstruction.scene.objects.blender.export import verify_round
                    result = verify_round(s, obj, s / "objects" / obj / f"round_{fx['round']}", exported=True)
                    if not result.get('effective_ok',result['ok']):
                        raise SystemExit("the Blender fit has not met its numeric thresholds")
                    rec['model_round'] = fx['round']
                    rec['sdf'] = fx.get('sdf')
                    rec['mesh_sha256'] = result['mesh_sha256']
                object_event = (s, obj, "accepted" if rec["pass"] else "rejected", fingerprint)
        out = s / "qa" / "review" / f"{stage}_r{rec['round']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x') as stream:
            stream.write(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        if object_event:
            event(*object_event[:3], round=number, model_sha256=object_event[3])
        print(f"{out}: {'pass' if rec['pass'] else 'open'} ({len(rec['findings'])} finding(s))")
    elif cmd == "show":
        print(json.dumps({"options": opts, "task": task(s)}, indent=1, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
