"""Remeasure a retained GS field after rebuilding native background geometry.

    python -m calibrated_scene_reconstruction.scene.background.measure_context SESSION CONTEXT CAMERAS REGIONS --output DIR

Uses the current physically audited export and fresh black/white/empty/field
renders. Produces coverage for context_rebind, never an appearance approval.
"""
import argparse
import json
from pathlib import Path
import subprocess

from calibrated_scene_reconstruction.scene.background.blender_env import verify
from calibrated_scene_reconstruction.scene.background.capture_gaps import prepare as coverage
from calibrated_scene_reconstruction.scene.background.environment_physics import current
from calibrated_scene_reconstruction.scene.background.gs_opacity import prepare as opacity
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def candidate_field(session, context, field, override):
    """Explicit field experiments keep audited geometry and calibrated placement."""
    override = Path(override).resolve()
    if (override != context.parent/'context.usdz' or not override.is_file()
            or not override.is_relative_to(session.dir.resolve()) or digest(override) != field['asset_sha256']):
        raise ValueError('Candidate override must be this session-local context package')
    sources = {str(p.resolve()): digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')}
    if not sources or field.get('source_sha256') != sources:
        raise ValueError('Field candidate must retain current agent geometry sources')
    inputs = {**field['input_sha256'], **sources, str(override): digest(override)}
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Candidate field evidence changed')
    return override, inputs


def run(session, context, cameras, regions, output, background_field=None):
    context, cameras, regions, out = [Path(p).resolve() for p in (context, cameras, regions, output)]
    if out.exists() or not out.is_relative_to(session.dir.resolve()):
        raise ValueError('Use a new session-local measurement directory')
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: create an environment derivative')
    native = verify(session)
    audit, _ = current(session)
    field = json.loads(context.read_text())
    if (field['protected_task_state'] != protected_state(session)
            or field['camera_sha256'] != digest(session/'parsing/posed_frames/cameras.npz')
            or field['parameters_sha256'] != parameter_digest(session.config('environment.json')['parameters'])):
        raise ValueError('Use the current geometry, capture and protected state')
    # The ordinary path requires the bound field. An explicit field experiment
    # can reuse unchanged audited geometry; actual override bytes and calibrated
    # placement are checked again by gs_opacity/capture_gaps, never guessed.
    override, extra_inputs = None, {}
    if background_field is not None:
        override, extra_inputs = candidate_field(session, context, field, background_field)
    elif native['splat_context']['evidence_sha256'] != digest(context):
        raise ValueError('Native export must bind the requested context manifest')
    pointer = session/'scene/environment/twin/current.json'
    inputs = {**extra_inputs, **{str(p): digest(p) for p in [context, cameras, regions, Path(audit['usd']), Path(__file__)]}}
    if pointer.is_file():
        inputs[str(pointer.resolve())] = digest(pointer)
    out.mkdir(parents=True)
    modes = {'black': ['--context-only', '--context-background', '0', '0', '0'],
             'white': ['--context-only', '--context-background', '1', '1', '1'],
             'empty': ['--context-only', '--context-empty'], 'field': ['--context-only']}
    for name, flags in modes.items():
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(['tools/run_isaac.sh', 'apps/scene/review_usd_cameras.py', audit['usd'],
                            str(cameras), '--renderer', 'rt', *flags,
                            *(['--background-field', str(override)] if override is not None else []),
                            '--output', str(out/name)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        # Kit can exit zero after a Python exception. Require its output proof.
        proof = out/name/'review.json'
        if not proof.is_file():
            raise ValueError('Native render incomplete: '+name)
        print('Fresh retained-field measurement:', name, flush=True)
    opacity(session, context, out/'black/review.json', out/'white/review.json', out/'opacity')
    result = coverage(session, context, out/'field/review.json', regions, out/'coverage',
                      out/'empty/review.json', out/'opacity/opacity.json')
    current(session)  # Reject concurrent geometry/config edits, even after the last render.
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Measurement inputs changed during rendering')
    result = {'coverage': str(out/'coverage/coverage.json'), 'input_sha256': inputs,
              'rows': len(result['rows']), 'background_field_override': str(override) if override else None,
              'photorealism_approved': False,
              'scope': 'Fresh actual-field coverage only; rebind, then inspect the composed source/free views.'}
    (out/'measurement.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'cameras', 'regions'):
        parser.add_argument(name)
    parser.add_argument('--output', required=True)
    parser.add_argument('--background-field', help='Explicit session-local field candidate; keep current audited geometry')
    args = parser.parse_args()
    print(json.dumps(run(Session(args.session), args.context, args.cameras, args.regions, args.output, args.background_field), indent=2))


if __name__ == '__main__':
    main()
