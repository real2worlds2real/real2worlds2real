"""Scoped source-pose controls for explicitly requested functional acceptance.

Only independently labelled flexible pixels can be excluded from the rigid
control. Rendered geometry never selects unknown source pixels. Raw full-object
and part measurements remain unchanged, and no control grants a waiver itself.
"""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np


def policy(session, obj):
    from calibrated_scene_reconstruction.assets.acceptance import declaration
    from .part_evidence import requirements
    if not hasattr(session, 'config_dir'):
        return []
    if hasattr(session, 'config'):
        task = session.config('task.json', {})
    else:
        path = session.config_dir / 'task.json'
        task = json.loads(path.read_text()) if path.exists() else {}
    gates = {r['gate'] for r in declaration(task)['relaxations']}
    from .source_pose_groups import declaration as pose_groups
    explicit = [{'gate': 'observation:' + row['id'], 'id': row['id'],
             'link': row['link'].split('/')[1]}
            for row in requirements(session, obj)
            if row.get('kind') == 'passive_shape' and 'observation:' + row['id'] in gates]+pose_groups(session,obj)
    # Preserve the established input-specific contract when one explicitly
    # scopes this object's source pose. Default qualitative flexibility is a
    # separate policy; it never fabricates an input acceptance condition.
    if explicit:return explicit
    understood=Path(session.config_dir)/'agent/object_understanding.json'
    if not understood.is_file():return []
    from calibrated_scene_reconstruction.assets.object_understanding import declaration as understanding
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration as mechanisms
    from .qualitative_source import scopes
    plan=mechanisms(session)
    if not plan or obj not in plan.get('objects',{}):return []
    kinds={name:link['kind'] for name,link in plan['objects'][obj]['links'].items()}
    return scopes(understanding(session,obj),requirements(session,obj),task,obj,kinds)


def remap_mask(mask, window, target_window, shape):
    """Independent source mask into another crop of the same camera."""
    x, y, _, _, scale = window
    tx, ty, _, _, target_scale = target_window
    affine = np.array([[target_scale / scale, 0., (x - tx) * target_scale],
                       [0., target_scale / scale, (y - ty) * target_scale]])
    return cv2.warpAffine(np.asarray(mask, np.uint8), affine, shape[::-1],
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0


def scoped_comparison(F, observations, parts, requested):
    """Return source-only domains and retained faces; never grant approval."""
    if not requested:
        return None
    z, frames, masks, windows = observations
    excluded = np.zeros(len(F), bool)
    sources = {}
    for request in requested:
        names = request.get('links') or [request['link']]
        kind = 'rigid' if request.get('kind') == 'rigid_group' else 'cloth'
        for name in names:
            part = next((p for p in parts or [] if p['name'] == name), None)
            if request.get('kind')=='qualitative_flexible':
                from .qualitative_source import FLEXIBLE_KINDS
                valid=bool(part is not None and part['kind'] in FLEXIBLE_KINDS
                           and part['kind']==request.get('physical_kinds',{}).get(name))
            else:valid=part is not None and part['kind']==kind
            if not valid:
                raise ValueError('Source pose relaxation has a missing or incompatible physical link')
            first = part['face_start']; excluded[first:first + part['triangle_count']] = True
        source = next((p for p in z.get('_source_parts', []) if p['id'] == request['id']), None)
        if source is None or (source.get('links') or [source.get('link')]) != names:
            raise ValueError('Independent source-part labels required for a pose relaxation')
        sz, sf, sm, sw = source['observations']
        sources[request['id']] = (sz, {int(i): (mask, window) for i, mask, window in zip(sf, sm, sw)})
    all_qualitative=all(row.get('kind')=='qualitative_flexible' for row in requested)
    if excluded.all() and not all_qualitative:
        raise ValueError('No unrelaxed source geometry remains')
    domains, positives = [], []
    for k, (i, mask, window) in enumerate(zip(frames, masks, windows)):
        ignored = np.zeros(mask.shape, bool)
        for sz, labels in sources.values():
            if int(i) not in labels:
                raise ValueError('Every object view needs an independent pose-group label')
            if any(not np.array_equal(z[key][i], sz[key][i]) for key in ['K', 'R_world_cam', 'centre']):
                raise ValueError('Part and whole source cameras differ')
            part_mask, part_window = labels[int(i)]
            ignored |= remap_mask(part_mask, part_window, window, mask.shape)
        domain = np.asarray(z['_object_domains'][k], bool).copy() if '_object_domains' in z else np.ones(mask.shape, bool)
        domain &= ~ignored
        domains.append(domain); positives.append(mask & domain)
    if not excluded.all() and sum(int(m.sum()) >= 50 and int((d & ~m).sum()) >= 50 for m, d in zip(positives, domains)) < 8:
        raise ValueError('Pose relaxation leaves fewer than eight informative geometry views')
    cameras = {key: value for key, value in z.items()
               if key not in ['_source_parts', '_source_material', '_proposal_excluded_links']}
    cameras['_object_domains'] = domains
    return ~excluded, (cameras, frames, positives, windows), [int((m & ~d).sum()) for m, d in zip(masks, domains)]


def controls(V, F, pose, observations, parts, requested):
    from .evidence import score
    if not requested:return None
    try:retained, scoped, ignored = scoped_comparison(F, observations, parts, requested)
    except ValueError as error:
        if not all(row.get('kind')=='qualitative_flexible' for row in requested):raise
        return {'policy':requested,'geometry':{'ok':False},'reason':str(error),
                'scope':'Default qualitative source controls are unavailable. Raw full-object/source-part fitting and every original gate remain active.'}
    if retained.any():metrics, _ = score(V, np.asarray(F)[retained], pose, scoped)
    else:
        metrics={'ok':True,'applicable':False,
                 'scope':'This independently labelled asset contains only actual flexible links; no rigid source geometry is omitted. Complete material, shape semantics and native task evidence remain required.'}
    return {'policy': requested, 'geometry': metrics, 'ignored_source_pixels': ignored,
            'scope': 'Unrelaxed geometry in all source views, excluding only independently observed authorized-part pixels; dimensions and native function remain mandatory.'}


def decision(session, obj, mesh, result):
    """Revalidate current immutable functional proofs at export and gate reads."""
    raw = result.get('ok') is True and result.get('metrics', {}).get('ok') is True
    if raw:
        return {'pass': True, 'default_pass': True, 'relaxed': False}
    requested = policy(session, obj)
    saved = result.get('metrics', {}).get('source_pose_controls', {})
    if not requested or saved.get('policy') != requested:
        return {'pass': False, 'default_pass': False, 'relaxed': False,
                'reason': 'No matching scoped source-pose controls'}
    if all(row.get('kind')=='qualitative_flexible' for row in requested):
        from .qualitative_source import current_evidence,verdict,FLEXIBLE_KINDS
        from .material_reference import declaration as material_references
        from calibrated_scene_reconstruction.scene.objects.object_understanding import validate_build
        try:
            validate_build(session,obj,result['build'])
            actual={part['name']:part['kind'] for part in result['build']['parts']}
            for row in requested:
                for name,kind in row['physical_kinds'].items():
                    if actual.get(name)!=kind or kind not in FLEXIBLE_KINDS:
                        raise ValueError('Qualitative source policy differs from actual flexible geometry')
        except (ValueError,KeyError,TypeError,OSError) as error:
            return {'pass':False,'default_pass':False,'relaxed':False,'reason':str(error)}
        expected={(row['link'],row['region']) for row in material_references(session,obj)}
        observed={(row.get('link'),row.get('region')) for row in result['metrics'].get('source_material',[])}
        proof=current_evidence(session,obj,mesh,result)
        accepted=verdict(result['metrics'],saved,requested,
            material_validity=result['build'].get('initial_state_admissibility',{}).get('pass') is True,
            source_material_complete=expected<=observed,
            semantic_review_pass=proof['semantic_review_pass'],native_task_pass=proof['native_task_pass'])
        accepted['verified_evidence']=proof
        return accepted
    from calibrated_scene_reconstruction.assets.acceptance import declaration, evaluate
    from calibrated_scene_reconstruction.scene.acceptance_evidence import read, subject, completed_task_criteria
    task = session.config('task.json', {})
    criteria = {row['id']: read(session, row['id'], subject(session, obj, mesh, 'nominal'), 'nominal')
                for row in declaration(task)['criteria']}
    criteria.update(completed_task_criteria(session,obj,mesh))
    metrics = result['metrics']; waived_ids = {row['id'] for row in requested}
    from .material_reference import declaration as material_references
    expected_material={(row['link'],row['region']) for row in material_references(session,obj)}
    actual_material={(row.get('link'),row.get('region')) for row in metrics.get('source_material',[])}
    defaults = {'source_geometry': {'pass': saved.get('geometry', {}).get('ok') is True,
                                     'measurements': deepcopy(saved.get('geometry', {}))},
                'material_validity': {'pass': result.get('build', {}).get('initial_state_admissibility', {}).get('pass') is True},
                'source_material': {'pass': expected_material <= actual_material and
                                    all(p.get('ok') is True for p in metrics.get('source_material', []))}}
    if any(r.get('kind') == 'rigid_group' for r in requested):
        # Independently screen the actual candidate, not a remembered verdict
        # from a different shape. Native functional proofs are also required.
        from calibrated_scene_reconstruction.assets.rigid_initial_state import measure
        from .material_reference import mesh_parts
        with np.load(mesh) as data:
            physical=[]
            for part in mesh_parts(data):
                if part['kind'] != 'rigid':
                    raise ValueError('Mixed flexible/rigid pose groups require a joint material-state contract')
                start=part['vertex_start'];count=part['vertex_count'];face=part['face_start']
                physical.append({'name':part['name'],'V':data['V'][start:start+count],
                                 'F':data['F'][face:face+part['triangle_count']]-start})
            screening=measure(physical,spacing_m=.0005,device='cpu')
        defaults['material_validity']={'pass':screening['pass'],'measurements':screening}
    requested_gates={row['id']:row['gate'] for row in requested}
    for part in metrics.get('source_parts', []):
        gate=requested_gates.get(part['id'],'observation:'+part['id'])
        defaults[gate] = {'pass': part.get('ok') is True, 'measurements': deepcopy(part)}
    if not waived_ids <= {p['id'] for p in metrics.get('source_parts', [])}:
        return {'pass': False, 'default_pass': False, 'relaxed': False, 'reason': 'Missing raw source part measurements'}
    accepted = evaluate(task, defaults, criteria)
    return {'pass': accepted['pass'], 'default_pass': False, 'relaxed': accepted['pass'],
            'raw_full_object': deepcopy(metrics), 'input_acceptance': accepted}
