"""Agent-facing process reconstruction operations; no motion taxonomy required."""
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.physics.process import workflow


DESCRIPTIONS = {
    'propose': 'Plan or revise physical bodies, interactions, parameter priors and whole-process observations. For region centroids/extents, review source identity/completeness and declare matching source/native observation_definitions; a photometric subset is not a complete material region. Review inherited initial motion when reusing a scene. Keep held-out samples out of proposal selection.',
    'run': 'Fit the current source-bound process model with native Isaac Sim 5.1 and preserve trial evidence.',
    'status': 'Read process-loop state; report missing, failed, stale or review-pending evidence honestly.',
    'request_geometry': 'Return specific missing geometry or mechanism requirements to the real2sim modeling loop.',
    'request_observations': 'Record missing video, motion or force evidence and the uncertainty it leaves.',
    'review': 'Review an executed process round; numerical fitting never approves itself.',
    'accepted': 'Return a current accepted model, parameter estimates, uncertainties and native replay artifacts.',
    'attach': 'Attach an accepted process to real2sim; its delivery then requires current process evidence.',
    'prepare_legacy': 'Adapt existing real video tracks and named robot trajectories into generic observation channels.',
    'legacy_recipe': 'Return the existing motion-specific tool commands; no implicit template fallback.',
}


def schemas():
    result = {}
    for name in DESCRIPTIONS:
        properties = {'session': {'type': 'string', 'minLength': 1}}
        required = ['session']
        if name in {'propose', 'request_geometry', 'request_observations', 'review'}:
            key = 'plan' if name=='propose' else 'review' if name=='review' else 'request'
            properties[key] = {'type': 'object'}
            if name == 'propose':
                from calibrated_scene_reconstruction.physics.process.plan_schema import schema
                properties[key] = schema()
            required.append(key)
        if name == 'attach':
            properties['recording'] = {'type': 'string', 'minLength': 1}
            required.append('recording')
        if name == 'legacy_recipe':
            properties['route'] = {'enum': ['replay', 'video', 'soft']}
            required.append('route')
        result[name] = {'name': name, 'input': {'type': 'object', 'properties': properties,
                         'required': required, 'additionalProperties': False},
                        'output': {'type': 'object', 'required': ['ok', 'evidence_images'],
                                   'properties': {'ok': {'type': 'boolean'},
                                                  'evidence_images': {'type': 'array', 'items': {'type': 'string'}}}}}
    return result


def execute(name, request, output):
    root = Path(resolve(request['session']))
    if name in {'prepare_legacy', 'legacy_recipe'}:
        from calibrated_scene_reconstruction.physics.process import legacy
        value = legacy.prepare(root, output) if name == 'prepare_legacy' else legacy.commands(root, request['route'])
        return {'ok': True, 'legacy': value, 'evidence_images': []}
    if name == 'propose':
        value = workflow.propose(root, request['plan'])
    elif name == 'request_geometry':
        value = workflow.request_geometry(root, request['request'])
    elif name == 'request_observations':
        value = workflow.request_observations(root, request['request'])
    elif name == 'review':
        value = workflow.review(root, request['review'])
    elif name == 'accepted':
        return {'ok': True, 'process': workflow.accepted(root), 'evidence_images': []}
    elif name == 'attach':
        value = workflow.attach(resolve(request['recording']), root)
    else:
        value = getattr(workflow, name)(root)
    return {'ok': value.get('state') not in {'failed', 'stale', 'rejected', 'interrupted'},
            'process': value, 'evidence_images': []}
