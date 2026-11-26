"""Read the bundled workflow contracts without importing scene execution backends."""
import copy
import json
from pathlib import Path

from .process_tool import schemas as physics_schemas
from .support_tool import schemas as support_schemas


STAGES = {'S0': ('poses', 'poses'), 'S1': ('anchor', 'anchor'), 'S2': ('task_zone', 'task_zone'),
          'S3': ('room_views', 'room_views'), 'S4': ('surfaces', 'surfaces'),
          'S5': ('clearance', 'object_review'), 'S6': ('deliver', 'deliver')}


def object_contracts():
    directory = Path(__file__).with_name('schemas') / 'objects'
    return {path.stem: json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(directory.glob('*.json'))}


def schemas():
    """List the contracts shipped with this installation."""
    result = {}
    for group, items in [('object', object_contracts()), ('support', support_schemas()), ('physics', physics_schemas())]:
        for operation, item in items.items():
            name = group + '_' + operation
            result[name] = {**copy.deepcopy(item), 'name': name}

    base = {'session': {'type': 'string', 'minLength': 1}}
    output = {'type': 'object', 'required': ['ok', 'evidence_images'],
              'properties': {'ok': {'type': 'boolean'}, 'evidence_images': {'type': 'array', 'items': {'type': 'string'}}}}
    stage_names = ['stage_' + stage.lower() for stage in STAGES]
    for name in stage_names + ['status', 'record', 'report']:
        properties, required = dict(base), ['session']
        if name.startswith('stage_'):
            properties.update(robot={'type': 'string', 'pattern': '^[A-Za-z0-9_-]+$'},
                              object_route={'enum': ['agent_blender', 'template']},
                              reuse_camera_model={'type': 'boolean'},
                              pose_workers={'type': 'integer', 'anyOf': [{'const': -1}, {'minimum': 1}]},
                              timeout={'type': 'integer', 'minimum': 1, 'maximum': 86400})
        if name == 'record':
            properties['review'] = {'type': 'object', 'required': ['stage', 'checks', 'review', 'findings', 'pass'],
                'properties': {'stage': {'type': 'string', 'pattern': r'^(S[0-6]|(?:OBJ|SUP)\.[A-Za-z0-9_-]+)$'},
                               'checks': {}, 'review': {'type': 'string', 'minLength': 1},
                               'findings': {'type': 'array', 'items': {'type': 'object'}}, 'pass': {'type': 'boolean'},
                               'round': {'type': 'integer', 'minimum': 1}}}
            required.append('review')
        result[name] = {'name': name, 'input': {'$schema': 'https://json-schema.org/draft/2020-12/schema',
                        'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False},
                        'output': copy.deepcopy(output)}
    return result
