"""Read-only evaluation of review records, object loops and delivery prerequisites."""
import json
import re


def latest_record(session, stage):
    """Select the largest round number; file modification order is not round order."""
    pattern = re.compile(re.escape(stage) + r"_r(\d+)\.json$")
    records = [(int(m.group(1)), p) for p in (session / "qa/review").glob(f"{stage}_r*.json")
               if (m := pattern.fullmatch(p.name))]
    return max(records, default=(None, None))[1]


def record_state(session, stage):
    path = latest_record(session, stage)
    state = dict(stage=stage, applicable=True, passed=False, state="missing", reason="No review record",
                 record=None, fresh=False)
    if path is None:
        return state
    state["record"] = str(path)
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or type(record.get("pass")) is not bool:
            raise ValueError("Review must contain a boolean pass")
        if record.get("stage") != stage:
            raise ValueError("Review stage does not match its file")
        findings = record.get("findings")
        if not isinstance(findings, list) or any(not isinstance(f, dict) for f in findings):
            raise ValueError("Review findings must be a list of objects")
        if record["pass"] and any(f.get("status") != "resolved" for f in findings):
            raise ValueError("Passing review contains unresolved findings")
        stamp = session / f"qa/review/{stage}.ran"
        state["fresh"] = not stamp.exists() or path.stat().st_mtime_ns > stamp.stat().st_mtime_ns
        if not state["fresh"]:
            state.update(state="stale", reason="Review predates the latest stage run")
        elif record["pass"]:
            state.update(passed=True, state="passed", reason="Latest review passed")
        else:
            state.update(state="rejected", reason="Latest review did not pass")
    except (OSError, ValueError, TypeError) as exc:
        state.update(state="invalid", reason=str(exc))
    return state


def active_objects(graph):
    """Keep exported Blender loops active even if their fit metadata was removed."""
    return {name: spec for name, spec in graph.get("dynamic_objects", {}).items()
            if spec.get("fit") or (spec.get("fix") or {}).get("model")}


def object_state(session, name, spec, verify):
    state = record_state(session, "OBJ." + name)
    if not state["passed"]:
        return state
    fit, fix = spec.get("fit") or {}, spec.get("fix") or {}
    if fit.get("model") or fix.get("model"):
        try:
            if not fit.get("model") or not fix.get("model"):
                raise ValueError("Active Blender object is missing its fit or exported model reference")
            result = verify(session, name, session / "objects" / name / f"round_{fix['round']}", exported=True)
            record = json.loads(latest_record(session, "OBJ." + name).read_text())
            source_pass = result.get('ok') is True or result.get('source_acceptance',{}).get('pass') is True
            if not source_pass or record.get("model_sha256") != result["model_sha256"]:
                raise ValueError("Review does not match the passing exported Blender round")
            if 'model_round' in record and (record['model_round'] != fix['round']
                    or record.get('mesh_sha256') != result['mesh_sha256']):
                raise ValueError('Review belongs to a different exported model round or mesh')
            if 'sdf' in record and record['sdf'] != fix.get('sdf'):
                raise ValueError('Collider precision changed since the object review')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            state.update(passed=False, state="invalid", reason=str(exc))
    if state['passed'] and hasattr(session, 'config_dir') and not (fit.get('model') or fix.get('model')):
        # Template/external geometry cannot bypass intrinsic-object checks.
        try:
            from calibrated_scene_reconstruction.scene.objects.object_understanding import current
            directory = session / 'objects' / name / f"round_{fix['round']}"
            build = json.loads((directory/'fit.json').read_text())['build']
            current(session, name, directory, build)
            from calibrated_scene_reconstruction.scene.objects.structural_review import current as structure
            structure(session, name, directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            state.update(passed=False, state='invalid', reason=str(exc))
    return state


def acceptance_evidence_extension(recorded, current, evidence):
    """Allow only appended, already hash-bound evidence in a criterion proof.

    Observation assembly may append auxiliary native files to a task's shared
    evidence list after its acceptance snapshot. Required files, their order,
    checks, defaults and every other verdict must remain unchanged. The caller
    still verifies the complete current mechanism file hashes below.
    """
    from copy import deepcopy
    normalized = deepcopy(current)
    try:
        if set(recorded['criteria']) != set(current['criteria']):
            return False
        for name, criterion in normalized['criteria'].items():
            old = recorded['criteria'][name]['result'].get('evidence_files')
            new = criterion['result'].get('evidence_files')
            if old == new:
                continue
            if (not isinstance(old, list) or not old or not isinstance(new, list)
                    or len(new) <= len(old) or new[:len(old)] != old
                    or any(not isinstance(p, str) or not evidence.get(p) for p in new)):
                return False
            criterion['result']['evidence_files'] = old
        return normalized == recorded
    except (KeyError, TypeError, AttributeError):
        return False


def task_result(session, declared):
    """Read the result contract only; never execute or import a task implementation."""
    path = session / "qa/task_check.json"
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Task result must be an object")
        if result.get("pass") is True:
            from calibrated_scene_reconstruction.scene.session import Session
            from calibrated_scene_reconstruction.scene.objects.mechanisms import fingerprint
            proof_session = session if hasattr(session, 'config_dir') else Session(session)
            expected = fingerprint(proof_session)
            if expected is None or 'task_input_provenance' in result:
                from .task_provenance import current as current_task_inputs
                current_task_inputs(proof_session, result)
            task=proof_session.config('task.json',{})
            if task.get('acceptance'):
                from calibrated_scene_reconstruction.assets.acceptance import evaluate
                recorded=result.get('input_acceptance',{})
                defaults={name:row['raw'] for name,row in recorded.get('defaults',{}).items()}
                accepted=evaluate(task,defaults,result.get('input_criteria',{}))
                same = accepted == recorded or (expected is not None and acceptance_evidence_extension(
                    recorded, accepted, result.get('mechanism_evidence_sha256', {})))
                if not defaults or accepted.get('pass') is not True or not same:
                    return False,'Explicit input acceptance has not passed with the recorded default verdicts'
            if expected is not None and (result.get('input_fingerprint') != expected
                                         or result.get('mechanism_validation', {}).get('pass') is not True):
                return False, 'Task evidence does not match the current geometry, mechanism or physics'
            if expected is not None:
                from calibrated_scene_reconstruction.scene.objects.mechanisms import evidence_files
                actual = evidence_files(proof_session, result.get('mechanisms', {}))
                if actual != result.get('mechanism_evidence_sha256', {}):
                    return False, 'Saved mechanism evidence changed since validation'
                from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration as physical_priors
                if physical_priors(proof_session) is not None:
                    sensitivity = result.get('physical_sensitivity', {})
                    if sensitivity.get('pass') is not True or any(sensitivity.get('cases', {}).get(c) is not True
                                                                for c in ('nominal', 'low', 'high')):
                        return False, 'Physical-prior sensitivity trials have not all passed'
                    reports = sensitivity.get('trial_reports', {})
                    if set(reports) != {'nominal', 'low', 'high'} or any(
                            p not in result.get('sensitivity_evidence_sha256', {}) for p in reports.values()):
                        return False, 'Physical sensitivity trial reports missing'
                    import hashlib
                    for name, checksum in result.get('sensitivity_evidence_sha256', {}).items():
                        file = (proof_session.dir/name).resolve()
                        if not file.is_relative_to(proof_session.dir.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
                            return False, 'Physical sensitivity evidence changed since validation'
            return True, "Task check passed"
        if "pass" in result and result["pass"] is None and not declared:
            return True, "No task declared; task check skipped"
        return False, "Declared task requires pass: true" if declared else "Task check failed"
    except (OSError, ValueError) as exc:
        return False, "Task result unavailable: " + str(exc)


def support_state(session, name):
    state = record_state(session, 'SUP.'+name)
    if not state['passed']:
        return state
    try:
        from calibrated_scene_reconstruction.scene.support.blender import verify
        result = verify(session, name)
        record = json.loads(latest_record(session, 'SUP.'+name).read_text())
        if any(record.get(key) != result[key] for key in ('model_sha256', 'mesh_sha256', 'visual_sha256')):
            raise ValueError('Support review belongs to different source, mesh or visual')
        if record.get('model_round') != result['round']:
            raise ValueError('Support review belongs to a different fitted round')
        from calibrated_scene_reconstruction.scene.objects.object_understanding import current as semantics
        from calibrated_scene_reconstruction.scene.objects.structural_review import current as structure
        directory=session/'support'/name/f"round_{result['round']}"
        if (record.get('semantic_review_sha256')!=semantics(session,name,directory,result['build'])
                or record.get('structural_review_sha256')!=structure(session,name,directory)):
            raise ValueError('Support gate does not bind its current semantic and structural reviews')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        state.update(passed=False, state='invalid', reason=str(exc))
    return state


def package_startup(session, asset, profile, input_fingerprint):
    """A Python reset override must not be required to play the delivered file."""
    if not (profile or {}).get('initialization_hz'):return None
    import hashlib
    from pathlib import Path
    from calibrated_scene_reconstruction.core.paths import ROOT
    root=session.dir.resolve();asset=Path(asset).resolve()
    path=root/'qa'/('package_physics_'+asset.stem+'.json')
    if not path.is_file():raise ValueError('Standalone native package startup report is required')
    proof=json.loads(path.read_text());digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    producer=proof.get('producer',{})
    if (proof.get('pass') is not True or proof.get('asset')!=str(asset)
            or proof.get('asset_sha256')!=digest(asset)
            or proof.get('source_input_fingerprint')!=input_fingerprint
            or proof.get('simulation_profile')!=profile
            or not str(proof.get('simulator','')).startswith('Isaac Sim 5.1.')
            or proof.get('seconds',0)<2 or proof.get('inputs_unchanged') is not True
            or proof.get('native_handles_verified') is not True
            or abs(proof.get('runtime_advanced_s',0)-proof.get('seconds',0))>1e-5
            or proof.get('initialization')!='Ordinary native reset at the authored clock; no Python small-step wrapper'
            or producer.get('path')!='apps/scene/check_package_physics.py'
            or producer.get('sha256')!=digest(ROOT/'apps/scene/check_package_physics.py')
            or proof.get('native_physics',{}).get('pass') is not True
            or proof.get('precision',{}).get('pass') is not True):
        raise ValueError('Standalone native package startup evidence is stale or failed')
    rows=proof.get('samples',[]);times=[r.get('time_s',-1) for r in rows]
    if (len(rows)<2 or times[0]!=0 or abs(times[-1]-proof['seconds'])>1e-8
            or any(r.get('finite') is not True for r in rows)
            or any(not 0<b-a<=.05+1e-8 for a,b in zip(times,times[1:]))
            or abs(proof.get('dt_s',0)-1/profile['physics_hz'])>1e-10):
        raise ValueError('Incomplete standalone native startup sampling')
    paths=proof.get('body_paths',[]);states=proof.get('raw_states',{})
    if not paths or len(paths)!=len(set(paths)) or set(states)!=set(paths):
        raise ValueError('Standalone native startup has incomplete body coverage')
    for entry in states.values():
        file=(root/entry['file']).resolve()
        if not file.is_relative_to(root/'qa') or not file.is_file() or digest(file)!=entry.get('sha256'):
            raise ValueError('Standalone native startup raw states changed')
    return {'report':str(path.relative_to(root)),'sha256':digest(path)}


def delivery_package(session, record):
    """Read the reopened package proof; bind it to native QA and actual bytes."""
    import hashlib
    root = session.dir.resolve()
    name = record.get('package_report')
    if not isinstance(name, str):
        raise ValueError('Mechanism delivery needs a package_report from view_usd.py')
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Package proof must remain inside this session')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get('package_report_sha256', checksum) != checksum:
        raise ValueError('Reopened package proof changed since delivery review')
    proof = json.loads(path.read_text())
    task = json.loads((root/'qa/task_check.json').read_text())
    if (proof.get('pass') is not True or proof.get('passive') is not True
            or proof.get('source_input_fingerprint') != task.get('input_fingerprint')
            or proof.get('simulation_profile') != task.get('simulation_profile')
            or not proof.get('simulation_profile') or not proof.get('collider_precision')):
        raise ValueError('Reopened package does not match the checked passive mechanism/profile')
    from pathlib import Path
    asset = Path(proof['asset']).resolve()
    if (not asset.is_relative_to(root) or not asset.is_file()
            or hashlib.sha256(asset.read_bytes()).hexdigest() != proof.get('asset_sha256')):
        raise ValueError('Delivered asset bytes changed or are outside this session')
    startup=package_startup(session,asset,proof['simulation_profile'],task.get('input_fingerprint'))
    if startup is not None and proof.get('native_startup')!=startup:
        raise ValueError('Reopened package must bind its actual native startup proof')
    return checksum


def status(session, opts, graph, task_config, verify):
    """Evaluate each record once and compose delivery requirements without mutating evidence."""
    stages = {f"S{i}": record_state(session, f"S{i}") for i in range(7)}
    objects = {name: object_state(session, name, spec, verify)
               for name, spec in active_objects(graph).items()}
    supports = {}
    from .process_result import state as process_state
    process = process_state(session, task_config)
    from .inventory import state as inventory_state
    inventory = inventory_state(session, graph, task_config)
    if opts is None:
        for state in stages.values():
            state.update(applicable=False, passed=True, state="not_applicable", reason="Legacy workflow has no twin gates")
        if not process['passed']:
            for stage in ('S5','S6'):
                stages[stage].update(passed=False, state='blocked', reason='Physical process: '+process['reason'])
        if not inventory['passed']:
            for stage in ('S2', 'S6'):
                stages[stage].update(passed=False, state='blocked', reason=inventory['reason'])
        return dict(stages=stages, objects=objects, supports=supports, physics_process=process,
                    object_inventory=inventory, passed=process['passed'] and inventory['passed'])
    if stages['S0']['passed']:
        try:
            from calibrated_scene_reconstruction.scene.capture.exposure import current as current_exposure
            checksum = current_exposure(session)
            frame_path = session/'parsing/world_frame.json'
            if frame_path.exists() and 'observed_auxiliary_floor' in json.loads(frame_path.read_text()):
                from calibrated_scene_reconstruction.scene.capture.draft_floor import verify_active
                verify_active(session)
            record = json.loads(latest_record(session, 'S0').read_text())
            if record.get('exposure_review_sha256') != checksum:
                raise ValueError('S0 does not bind the current exposure review')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            stages['S0'].update(passed=False, state='blocked', reason='Capture exposure: '+str(exc))
    if opts.get('support_route') == 'agent_blender':
        name = graph.get('task', {}).get('support')
        supports[name or 'task_support'] = support_state(session, name or 'task_support')
    if opts["background"] == "zone":
        stages["S3"].update(applicable=False, passed=True, state="not_applicable", reason="Zone background")
    elif opts.get('environment_quality_version') == 1 and stages['S3']['passed']:
        try:
            from calibrated_scene_reconstruction.scene.review.environment import current as environment_quality
            _, digest = environment_quality(session)
            record = json.loads(latest_record(session, 'S3').read_text())
            if record.get('environment_quality_sha256') != digest:
                raise ValueError('S3 does not bind the current whole-scene visual review')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            stages['S3'].update(passed=False, state='blocked', reason='Environment photorealism: '+str(exc))
    if opts.get('scale') == 'robot_visual':
        try:
            from calibrated_scene_reconstruction.scene.capture.robot_visual import verify_anchor
            verify_anchor(session, graph)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            stages['S1'].update(passed=False, state='blocked', reason=str(exc))
    if opts['background']=='blender' and stages['S6']['passed']:
        try:
            from calibrated_scene_reconstruction.scene.background.environment_physics import current as physical_current
            _,physical_digest=physical_current(session,delivery=True)
            record=json.loads(latest_record(session,'S6').read_text())
            if record.get('environment_delivery_physics_sha256')!=physical_digest:
                raise ValueError('S6 does not bind the actual delivery physical audit')
        except (OSError,ValueError,TypeError,KeyError) as exc:
            stages['S6'].update(passed=False,state='blocked',reason='Environment delivery physics: '+str(exc))
    if opts.get('scale') == 'robot_joints' and not (session/'qa/independent_freeze.json').exists():
        try:
            from calibrated_scene_reconstruction.scene.capture.robot_visual import verify_anchor
            verify_anchor(session, graph, joint_prior=True)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            stages['S1'].update(passed=False, state='blocked', reason='Uncertain joint calibration: '+str(exc))
    declared = bool(task_config.get("task") or any((graph.get("task") or {}).get(k) for k in ("kind", "target", "goal")))
    if task_config.get('mechanism_contract') and stages['S6']['passed']:
        try:
            delivery_package(session, json.loads(latest_record(session, 'S6').read_text()))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            stages['S6'].update(passed=False, state='blocked', reason='Mechanism delivery: '+str(exc))
    ok, reason = task_result(session, declared)
    if not ok:
        stages["S5"].update(passed=False, state="blocked", reason=reason)
    if not process['passed']:
        stages['S5'].update(passed=False, state='blocked', reason='Physical process: '+process['reason'])
    failed_objects = ["OBJ." + name for name, state in objects.items() if not state["passed"]]
    failed_supports = ['SUP.'+name for name, state in supports.items() if not state['passed']]
    if inventory['applicable']:
        try:
            if not inventory['passed']:
                raise ValueError(inventory['reason'])
            if stages['S2']['passed']:
                record = json.loads(latest_record(session, 'S2').read_text())
                if record.get('object_inventory_sha256') != inventory['sha256']:
                    raise ValueError('S2 does not bind the current input object inventory')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            stages['S2'].update(passed=False, state='blocked', reason='Object inventory: '+str(exc))
    if (failed_objects or failed_supports) and stages["S2"]["passed"]:
        stages["S2"].update(passed=False, state="blocked", reason="Geometry loops failed: " + ", ".join(failed_objects+failed_supports))
    failed = [name for name, state in stages.items() if name != "S6" and state["applicable"] and not state["passed"]]
    if failed or failed_objects or failed_supports:
        stages["S6"].update(passed=False, state="blocked", reason="Prerequisites failed: " + ", ".join(failed + failed_objects+failed_supports))
    return dict(stages=stages, objects=objects, supports=supports, physics_process=process,
                object_inventory=inventory, passed=stages["S6"]["passed"])
