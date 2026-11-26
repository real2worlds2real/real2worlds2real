"""Agent-neutral Blender iteration tools for the static task support."""
from pathlib import Path
from calibrated_scene_reconstruction.core.paths import rel


DESCRIPTIONS = {
    'blender_run': 'Build the agent-authored table/support and return appearance previews.',
    'fit': 'Fit support parameters and XY/yaw to current-view masks and the observed support plane.',
    'export': 'Export a passing support round with identical visual and static collision geometry.',
    'refine_run': 'Fit and export one support round; SUP review remains a separate agent decision.'}


def schemas():
    output = {'type': 'object', 'required': ['ok', 'evidence_images'],
              'properties': {'ok': {'type': 'boolean'}, 'evidence_images': {'type': 'array', 'items': {'type': 'string'}}}}
    result = {}
    for name in DESCRIPTIONS:
        properties = {'session': {'type': 'string', 'minLength': 1}}
        required = ['session']
        if name in ('fit', 'refine_run'):
            properties['max_evaluations'] = {'type': 'integer', 'minimum': 10, 'maximum': 2000}
        elif name == 'export':
            properties['round'] = {'type': 'integer', 'minimum': 1}
            required.append('round')
        else:
            properties['params'] = {'type': 'object', 'additionalProperties': {'type': 'number'}}
        result[name] = {'name': name, 'input': {'type': 'object', 'properties': properties,
                         'required': required, 'additionalProperties': False}, 'output': output}
    return result


def execute(name, request, output):
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.support import blender
    from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender

    session = Session(request['session'])
    blender.require_mutable(session)
    if name == 'blender_run':
        with Blender(blender.model_path(session), Path(output)/'cache', max_extent_m=10.,
                     source_files=blender.source_assets(session)) as builder:
            path, result = builder.evaluate(request.get('params'), artifacts=True)
        return {**result, 'evidence_images': [rel(path/p) for p in result['previews']]}
    if name == 'export':
        result = blender.export(session, request['round'])
        result['evidence_images'] = []
        return result
    result = blender.run(session, request.get('max_evaluations', 300))
    if name == 'refine_run' and result['ok']:
        result['export'] = blender.export(session, result['round'])
    return {key: value for key, value in {**result, 'evidence_images': [result['sheet']]}.items() if key != 'trace'}
