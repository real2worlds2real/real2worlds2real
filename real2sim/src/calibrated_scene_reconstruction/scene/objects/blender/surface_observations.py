"""Independent source regions within one body must not hide in its union mask."""


def requirements(session, obj):
    graph = getattr(session, 'scene_graph', {})
    values = graph.get('dynamic_objects', {}).get(obj, {}).get('fit', {}).get('source_regions', [])
    required = {'id', 'prompt', 'part_review', 'evidence_files'}
    if not isinstance(values, list):
        raise ValueError('Source surface regions must be a list')
    result = []; ids = set()
    for value in values:
        selector = 'links' if isinstance(value, dict) and 'links' in value else 'materials'
        if (not isinstance(value, dict) or set(value) != required | {selector}
                or not all(isinstance(value[k], str) and value[k].strip() for k in ('id', 'prompt', 'part_review'))
                or value['id'] in ids or not isinstance(value[selector], list) or not value[selector]
                or any(not isinstance(name, str) or not name or '/' in name for name in value[selector])
                or len(set(value[selector])) != len(value[selector])
                or not isinstance(value['evidence_files'], dict) or not value['evidence_files']):
            raise ValueError('Source region needs unique identity, either actual materials or links, and hash-bound label review')
        ids.add(value['id'])
        row = {'id': 'surface_'+value['id'], 'part_prompt': value['prompt'],
               'part_review': value['part_review'], 'evidence_files': value['evidence_files']}
        if selector == 'materials':
            row.update(link=obj+'/surface:'+value['id'], materials=value['materials'])
        else:
            row.update(link=obj+'/'+value['links'][0],
                       link_group=[obj+'/'+name for name in value['links']])
        result.append(row)
    return result
