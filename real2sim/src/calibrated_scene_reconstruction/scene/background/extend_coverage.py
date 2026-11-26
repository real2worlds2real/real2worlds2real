"""Add measured GS openings to named native owners, preserving existing cutouts.

New source observations must not reset previously reviewed local mesh repairs.
Only named native replacements change ownership; field rows and geometry stay
fixed. The result is a candidate, never a whole-scene appearance approval.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
from calibrated_scene_reconstruction.scene.background.appearance_ownership import specification
from calibrated_scene_reconstruction.scene.background.appearance_plan import validate as validate_plan
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def extend(context, owners):
    if not isinstance(owners, dict) or not 1 <= len(owners) <= 8:
        raise ValueError('Name a bounded set of native background owners')
    result = copy.deepcopy(context)
    for owner, row in owners.items():
        if (context['appearance_ownership'].get(owner, {}).get('role') != 'native_replacement'
                or not isinstance(row, dict) or not str(row.get('basis', '')).strip()
                or set(row)-{'basis', 'pixel_m'}):
            raise ValueError('Only explicitly reviewed current native replacements may gain GS openings')
        pixel = row.get('pixel_m', .02)
        if isinstance(pixel, bool) or not isinstance(pixel, (float, int)) or not .005 <= pixel <= .1:
            raise ValueError('Bounded native atlas resolution required')
        result['appearance_ownership'][owner] = {'role':'gap_fill', 'preserve_gs':True,
            'basis':row['basis'], 'mesh_coverage':{'compositing':'captured_projection',
                'coverage_entity':owner, 'pixel_m':pixel, 'radius_m':.15}}
        result['replaced_entities'].remove(owner)
        result['gap_fill_entities'].append(owner)
    return result


def prepare(session, context_path, coverage_path, request_path, output):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current, extract
    from calibrated_scene_reconstruction.scene.background import capture_gaps, crop_gaps, mesh_textures
    paths = [Path(p).resolve() for p in (context_path, coverage_path, request_path, output)]
    context_path, coverage_path, request_path, out = paths
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local coverage extension required')
    native = verify(session); physical, _ = current(session)
    context, coverage, request = [json.loads(p.read_text()) for p in paths[:3]]
    cfg = session.config('environment.json')
    if (native['splat_context']['evidence_sha256'] != digest(context_path)
            or context['protected_task_state'] != protected_state(session)
            or context['parameters_sha256'] != parameter_digest(cfg['parameters'])
            or coverage.get('version') != 2 or not coverage.get('opacity_evidence')):
        raise ValueError('Current field, protected task, geometry parameters and measured opacity required')
    result = extend(context, request['owners'])
    job_path = Path(json.loads((session/'qa/environment_physics/twin/current.json').read_text())['path']).parent/'job.json'
    job = json.loads(job_path.read_text()); signature = geometry_signature(job)
    if signature != context['selection_frozen']['geometry_signature']:
        raise ValueError('Actual audited geometry changed since field binding')
    cloud = context_path.parent/'context.ply'; field = PlyData.read(str(cloud))['vertex'].data
    if coverage['field_rows_sha256'] != hashlib.sha256(field.tobytes()).hexdigest():
        raise ValueError('Coverage must measure these exact retained field rows')
    names = {r['entity'] for r in job['parts'] if not r['external']}
    specification(result['appearance_ownership'], names, result['replaced_entities'], result['gap_fill_entities'], cfg['completion'])
    plan, plan_inputs = validate_plan(session, request['appearance_plan'], result['appearance_ownership'])
    inputs = {**context['input_sha256'], **plan_inputs}
    for p, sha in coverage['input_sha256'].items():
        if p in inputs and inputs[p] != sha:
            raise ValueError('Conflicting source measurement evidence')
        inputs[p] = sha
    for p in (*paths[:3], job_path, Path(job['geometry']), Path(physical['usd']), Path(coverage['render_usd']),
              Path(__file__).resolve(), Path(geometry_signature.__code__.co_filename).resolve(),
              Path(extract.__code__.co_filename).resolve(), *(Path(m.__file__).resolve() for m in (capture_gaps, crop_gaps, mesh_textures))):
        inputs[str(p)] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Coverage extension inputs changed')
    # Material changes can reuse measurements only for the identical actual
    # oriented world geometry and the same field transform/photometry.
    measured, active = [Usd.Stage.Open(p) for p in (coverage['render_usd'], physical['usd'])]
    a, b = [dict(stage.GetRootLayer().customLayerData) for stage in (measured, active)]
    keys = ('asset_sha256', 'matrix_json', 'native_light_gain', 'background_color_linear_json', 'source_background_scope_json')
    if (a.get('r2s2r_environment_render') != b.get('r2s2r_environment_render')
            or any((a.get('r2s2r_environment_context') or {}).get(k) != (b.get('r2s2r_environment_context') or {}).get(k) for k in keys)):
        raise ValueError('Actual measured field or photometry changed')
    parts, geometry = extract(measured, session.scene_graph['task']['support'], session.scene_graph.get('dynamic_objects', {}))
    out.mkdir(parents=True); measured_geometry = out/'measured_geometry.npz'
    np.savez_compressed(measured_geometry, **geometry)
    if geometry_signature({'geometry':str(measured_geometry), 'parts':parts}) != signature:
        raise ValueError('Source coverage and current native world geometry differ')
    inputs[str(measured_geometry)] = digest(measured_geometry)
    points = (np.c_[field['x'], field['y'], field['z'], np.ones(len(field))]@np.asarray(context['matrix']).T)[:, :3]
    for owner in request['owners']:
        result['gap_masks'][owner], evidence = capture_gaps.bake(session, job, owner,
            result['appearance_ownership'][owner]['mesh_coverage'], field, points, coverage_path, out)
        inputs.update(evidence)
    shutil.copyfile(cloud, out/'context.ply'); inputs[str(out/'context.ply')] = digest(out/'context.ply')
    result.update(input_sha256=inputs, appearance_plan=plan, photorealism_approved=False,
        coverage_extension={'owners':list(request['owners']), 'previous_cutouts_preserved':True,
            'field_rows_unchanged':True, 'geometry_signature':signature, 'appearance_pass':False})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Coverage extension evidence changed during baking')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'coverage', 'request'): parser.add_argument(name)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    result = prepare(Session(args.session), args.context, args.coverage, args.request, args.output)
    print('Unapproved coverage extension:', result['coverage_extension']['owners'])


if __name__ == '__main__': main()
