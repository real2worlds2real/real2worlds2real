"""Default flexible source accuracy, separate from explicit input waivers.

This module chooses a narrow source scope and combines *verified* evidence.
It never reads a native diagnostic as a task receipt or changes source labels.
Callers remain responsible for current geometry/source/review fingerprints.
"""
from copy import deepcopy
import json
from pathlib import Path

from calibrated_scene_reconstruction.assets.response_policy import mode

FLEXIBLE_KINDS = {'cloth', 'deformable', 'rope'}


def current_evidence(session, obj, mesh, result):
    """Read current semantic/native receipts, never a candidate-assay result.

    The task reader validates the entire current mechanism fingerprint, all
    saved measurements and physical-prior cases. This adds actual geometry
    identity and rejects diagnostic reports even if their experiment passed.
    """
    from .store import digest
    from calibrated_scene_reconstruction.scene.objects.object_understanding import current
    from calibrated_scene_reconstruction.scene.acceptance_evidence import mesh_digest
    from calibrated_scene_reconstruction.scene.twin.reviews import task_result
    from calibrated_scene_reconstruction.core.paths import resolve
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    import numpy as np
    mesh=Path(mesh)
    record={'semantic_review_pass':False,'native_task_pass':False,'evidence':{},'reasons':{}}
    try:
        if not mesh.is_file() or digest(mesh)!=result.get('mesh_sha256'):
            raise ValueError('Qualitative source candidate geometry changed')
        checksum=current(session,obj,mesh.parent,result['build'])
        if not isinstance(checksum,str) or not checksum:
            raise ValueError('Current semantic review did not return its evidence fingerprint')
        record['semantic_review_pass']=True
        record['evidence']['qa/object_understanding/'+obj+'.json']=checksum
    except (OSError,ValueError,KeyError,TypeError) as error:
        record['reasons']['semantic_review']=str(error)
    try:
        path=session.dir/'qa/task_check.json';proof=json.loads(path.read_text())
        if (proof.get('pass') is not True or proof.get('diagnostic_only') is True
                or proof.get('task_approved') is False
                or proof.get('mechanism_validation',{}).get('pass') is not True
                or not isinstance(proof.get('input_fingerprint'),str) or not proof['input_fingerprint']):
            raise ValueError('A complete current native task receipt is required; diagnostics cannot approve source')
        fix=session.scene_graph['dynamic_objects'][obj].get('fix',{})
        if not fix.get('mesh') or mesh_digest(mesh)!=mesh_digest(resolve(fix['mesh'])):
            raise ValueError('Native task used different actual physical geometry')
        pose=result['pose']
        if (not np.allclose([*fix['xy'],fix['support_z']],[*pose['xy'],pose['support_z']],rtol=0,atol=1e-9)
                or not np.allclose(rotation(fix),rotation(pose),rtol=0,atol=1e-9)):
            raise ValueError('Native task used a different source assembly pose')
        valid,reason=task_result(session,True)
        if valid is not True:raise ValueError(reason)
        record['native_task_pass']=True;record['evidence']['qa/task_check.json']=digest(path)
    except (OSError,ValueError,KeyError,TypeError) as error:
        record['reasons']['native_task']=str(error)
    return record


def scopes(understanding, requirements, task, obj, link_kinds):
    """Use independently labelled whole flexible links; rigid labels stay strict.

    Global quantitative mode and an explicitly quantitative semantic part both
    win over the default approximate policy. A material patch cannot stand in
    for a complete physical-link label. Missing source coverage does not grant
    an exemption, and a label mixing rigid and flexible structure is rejected.
    """
    if mode(task.get('flexible_accuracy')) == 'quantitative':
        return []
    eligible = set()
    for part in understanding.get('parts', []):
        if (part.get('representation') != 'deformable'
                or mode(part.get('response_accuracy')) != 'task_and_semantics'):
            continue
        names = set(part['links'])
        if not names or not names <= link_kinds.keys():
            raise ValueError('Flexible source policy requires complete actual link ownership')
        if any(link_kinds[name] not in FLEXIBLE_KINDS for name in names):
            raise ValueError('A rigid substitute cannot receive flexible source accuracy')
        eligible |= names
    result = []
    for item in requirements:
        if not item.get('part_prompt') or 'materials' in item:
            continue
        refs = item.get('link_group') or [item['link']]
        if any(not ref.startswith(obj+'/') or ref.count('/') != 1 for ref in refs):
            raise ValueError('Independent flexible labels must belong to the same object')
        names = [ref.split('/')[1] for ref in refs]
        overlap = set(names) & eligible
        if not overlap:
            continue
        if overlap != set(names):
            raise ValueError('Mixed strict and qualitative parts require separate source labels')
        if len(set(names)) != len(names):
            raise ValueError('A source group cannot repeat physical links')
        result.append({'id':item['id'], 'gate':'qualitative_source:'+item['id'],
                       'link':names[0], **({'links':names} if len(names)>1 else {}),
                       'kind':'qualitative_flexible',
                       'physical_kinds':{name:link_kinds[name] for name in names},
                       'accuracy_mode':'task_and_semantics'})
    if len({row['id'] for row in result}) != len(result):
        raise ValueError('Independent qualitative source identities must be unique')
    return result


def verdict(metrics, controls, requested, *, material_validity,
            source_material_complete, semantic_review_pass, native_task_pass):
    """Combine raw source evidence with separately verified native/semantic proof.

    ``native_task_pass`` must come from the current complete task result reader,
    including physical-prior cases. A standalone candidate experiment is never
    sufficient. This pure function deliberately cannot authorize that proof.
    """
    if (not requested or any(row.get('kind') != 'qualitative_flexible' for row in requested)
            or controls.get('policy') != requested):
        raise ValueError('Matching independent qualitative source controls required')
    ids = {row['id'] for row in requested}
    parts = metrics.get('source_parts', [])
    actual = {row['id'] for row in parts}
    source_valid = (len(actual) == len(parts) and ids <= actual
                    and controls.get('geometry', {}).get('ok') is True
                    and all(row.get('ok') is True for row in parts if row['id'] not in ids))
    materials_valid = (material_validity is True and source_material_complete is True
                       and all(row.get('ok') is True for row in metrics.get('source_material', [])))
    pending = bool(source_valid and materials_valid)
    checks = {'unrelaxed_source_geometry':bool(source_valid),
              'material_and_initial_geometry':bool(materials_valid),
              'current_semantic_and_plausibility_review':semantic_review_pass is True,
              'complete_current_native_task':native_task_pass is True}
    passed = all(checks.values())
    return {'pass':passed, 'default_pass':False, 'relaxed':passed,
            'policy':'task_and_semantics', 'native_candidate_only':bool(pending and not passed),
            'pending_native_eligible':pending, 'checks':checks,
            'raw_full_object':deepcopy(metrics), 'source_scopes':deepcopy(requested),
            'scope':'Only independently labelled flexible shape error is diagnostic. Rigid source accuracy, complete material, actual flexible representation, current semantics/plausibility and complete native tasks remain mandatory.'}
