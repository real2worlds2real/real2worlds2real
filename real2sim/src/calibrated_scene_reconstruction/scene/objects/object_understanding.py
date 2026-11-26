"""Required source-grounded part understanding, independent of task success.

An agent supplies hypotheses and judgments. This module enforces coverage,
provenance and measured bounds; it does not recognize object semantics itself.
"""
import hashlib
import json
from pathlib import Path

import numpy as np














def current(session, obj, round_dir, build):
    item = validate_build(session, obj, build)
    path = session.dir / 'qa/object_understanding' / (obj + '.json')
    report = json.loads(path.read_text())
    candidate = {n: digest(Path(round_dir) / n) for n in ('model.py', 'mesh.npz')}
    if (report.get('version') != 1 or report.get('object') != obj
            or report.get('understanding') != item or report.get('candidate_sha256') != candidate
            or report.get('pass') is not True):
        raise ValueError('Current passing semantic and physical review required')
    checks = report.get('parts', {})
    if set(checks) != {p['id'] for p in item['parts']}:
        raise ValueError('Every semantic part requires review, including secondary parts')
    for part in item['parts']:
        row = checks[part['id']]
        for kind in ('semantics', 'geometry', 'representation'):
            check = row.get(kind, {})
            if check.get('pass') is not True or not _text(check.get('reason')):
                raise ValueError('Part semantics, geometry and representation must each pass')
            _files(session, check.get('evidence'), qa=True)
        from calibrated_scene_reconstruction.assets.response_policy import approximate_part
        mechanisms_path=Path(session.config_dir)/'agent/mechanisms.json'
        mechanisms=json.loads(mechanisms_path.read_text()) if mechanisms_path.is_file() else {}
        declared_links=mechanisms.get('objects',{}).get(obj,{}).get('links',{})
        for link in part['links']:
            if declared_links.get(link,{}).get('initial_support',{}).get('connected_folds'):
                from .support_retention import current as support_retention
                support_retention(session,obj,link,round_dir,
                    row.get('support_retention',{}).get(link,{}))
        if approximate_part(part):
            plausibility=row.get('physical_plausibility',{})
            if plausibility.get('pass') is not True or not _text(plausibility.get('reason')):
                raise ValueError('Approximate flexible response needs a physical-plausibility review')
            _files(session,plausibility.get('evidence'),qa=True)
            # Exact response tests remain diagnostics. The native task and
            # semantic/structure gates remain required for final delivery.
            continue
        native = row.get('physics', {})
        if set(native) != {t['id'] for t in part.get('physics_tests',[])}:
            raise ValueError('Task success cannot replace intrinsic physical checks')
        for test in part.get('physics_tests',[]):
            proof = native[test['id']]
            _files(session, proof.get('evidence'), qa=True)
            relative = proof.get('report')
            if relative not in proof['evidence']:
                raise ValueError('Native report must be hash-bound evidence')
            measured = json.loads((session.dir / relative).read_text())
            if (measured.get('simulator') != 'Isaac Sim 5.1' or measured.get('test_id') != test['id']
                    or measured.get('candidate_sha256') != candidate
                    or measured.get('understanding') != item or measured.get('task_actuation') is not False
                    or measured.get('pass') is not True):
                raise ValueError('Independent current Isaac physical response evidence required')
            # Bind the producer and raw recording too; a render or self-declared
            # passing task report is insufficient physical evidence.
            _files(session, measured.get('raw_evidence'), qa=True)
            producer = measured.get('producer', {})
            code = (Path(__file__).resolve().parents[4] / producer.get('path', '')).resolve()
            root = Path(__file__).resolve().parents[4]
            if (not code.is_relative_to(root) or not code.is_file()
                    or digest(code) != producer.get('sha256')):
                raise ValueError('Native physical test producer changed or missing')
            for metric in test['metrics']:
                value = measured.get('measurements', {}).get(metric['name'], {})
                number = value.get('value')
                if (type(number) not in (int, float) or not np.isfinite(number)
                        or value.get('unit') != metric['unit']
                        or not metric['bounds'][0] <= number <= metric['bounds'][1]):
                    raise ValueError('Native physical measurement outside declared prior bounds')
    structural=report.get('structural_checks',{})
    if set(structural)!={p['id'] for p in item.get('structural_priors',[])}:
        raise ValueError('Every declared structural prior needs review')
    for check in structural.values():
        if check.get('pass') is not True or not _text(check.get('reason')):
            raise ValueError('Structural layout, connections and support must pass')
        _files(session,check.get('evidence'),qa=True)
    for prior in item.get('structural_priors',[]):
        measurement=prior.get('measurement')
        if measurement is None:continue
        if measurement.get('kind')!='reflection':raise ValueError('Unsupported structural measurement')
        from calibrated_scene_reconstruction.assets.structural_layout import reflection, equivalent
        with np.load(Path(round_dir)/'source_solids.npz') as solid:
            measured=reflection(solid,measurement)
        if not measured['pass']:raise ValueError('Actual geometry violates structural prior: '+prior['id'])
        if not equivalent(measured, structural[prior['id']].get('measurement')):
            raise ValueError('Structural review does not match actual measured geometry')
    return digest(path)


from calibrated_scene_reconstruction.assets.object_understanding import digest, _text, static_support, _files, declaration, validate_build  # compatibility exports
