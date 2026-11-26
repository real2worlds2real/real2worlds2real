"""Read checked contact traces for rendering, including immutable deliveries.

Frozen evidence is replayed under its recorded engine fingerprint, never
relabelled as a new physical validation under current code.
"""
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.review.environment import digest


def checked(session, usd, trace):
    usd, trace = Path(usd).resolve(), Path(trace).resolve()
    qa_path = session/'qa/task_check.json'
    package_path = session/f'qa/package_{usd.stem}.json'
    qa, package = [json.loads(p.read_text()) for p in (qa_path, package_path)]
    if qa.get('pass') is not True:
        raise ValueError('Passing source task evidence required')
    freeze = session/'qa/independent_freeze.json'
    frozen = freeze.exists()
    required = [qa_path, package_path, usd, trace, trace.with_name('trace.json'), trace.with_name('phases.json')]
    if frozen:
        from calibrated_scene_reconstruction.scene.twin.freeze import verify
        record = json.loads(freeze.read_text())
        if record.get('status', {}).get('passed') is not True or not verify(session)['unchanged']:
            raise ValueError('Frozen accepted evidence has changed')
        files = {r['path'] for r in record['files'] if r['base'] == 'session'}
        if any(str(p.resolve().relative_to(session.dir.resolve())) not in files for p in required):
            raise ValueError('Replay sources must belong to the accepted frozen delivery')
    else:
        from calibrated_scene_reconstruction.scene.objects.mechanisms import fingerprint
        if qa.get('input_fingerprint') != fingerprint(session):
            raise ValueError('Current passing task evidence required')
    if (package.get('pass') is not True or package.get('passive') is not True
            or package.get('source_input_fingerprint') != qa['input_fingerprint']
            or package.get('asset_sha256') != digest(usd)):
        raise ValueError('Checked passive delivery and source trace must match')
    proofs = {**qa.get('mechanism_evidence_sha256', {}), **qa.get('sensitivity_evidence_sha256', {})}
    for p in required[3:]:
        relative = str(p.relative_to(session.dir.resolve()))
        if proofs.get(relative) != digest(p):
            raise ValueError('Trace evidence changed or is unbound: '+relative)
    return {'qa': qa, 'package': package, 'metadata': json.loads(trace.with_name('trace.json').read_text()),
            'phases': json.loads(trace.with_name('phases.json').read_text()),
            'frozen': frozen, 'source_input_fingerprint': qa['input_fingerprint'],
            'input_sha256': {str(p): digest(p) for p in required}}
