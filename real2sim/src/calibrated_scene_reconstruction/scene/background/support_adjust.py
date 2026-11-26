"""Propose a background support edit with its supported objects and photo planes.

Changing a table height invalidates the photographs baked on objects resting
on it. Move those objects together and require fresh texture projection and
geometry/physics/appearance checks; never transfer an earlier approval.
"""
from copy import deepcopy

import numpy as np


def propose(config, entity, height, protected=(), surfaces=()):
    result = deepcopy(config)
    definitions = {row['id']: row for row in result['parameters']['entities']}
    if entity not in definitions or entity in protected:
        raise ValueError('An unfrozen background support is required')
    support = definitions[entity]
    if support['kind'] not in ('desk', 'table', 'cabinet', 'cabinet_open', 'platform'):
        raise ValueError('An explicit horizontal background support is required')
    if not np.isfinite(height) or height <= support['position'][2] + .02:
        raise ValueError('A finite support top above its base is required')
    relations = result['physics']['entities']
    moved = {entity}
    while True:
        children = {name for name, row in relations.items()
                    if set(row.get('on', ())) & moved}
        added = children - moved
        if not added:
            break
        moved |= added
    if set(protected) & moved or not moved <= definitions.keys():
        raise ValueError('Support edit reaches a frozen or undefined object')
    # A cycle, or an object shared by two supports, needs an explicit assembly
    # decision. Silently translating it can create a collision or a float.
    for name in moved:
        parents = relations.get(name, {}).get('on', [])
        if name == entity:
            if set(parents) & moved:
                raise ValueError('Cyclic support assembly')
        elif len(parents) != 1 or parents[0] not in moved:
            raise ValueError('Supported object has ambiguous parents')
        visited = {name}
        cursor = name
        while cursor != entity and cursor in moved:
            cursor = relations[cursor]['on'][0]
            if cursor in visited:
                raise ValueError('Cyclic support assembly')
            visited.add(cursor)
    old_top = support['position'][2] + support['size'][2]
    delta = float(height - old_top)
    support['size'][2] += delta
    for name in moved - {entity}:
        definitions[name]['position'][2] += delta
    # A support's top moves, but its side photographs deform rather than
    # translate. Require a new face declaration for that case.
    photos = deepcopy(list(surfaces))
    for row in photos:
        if row.get('kind') != 'planar' or row.get('entity') not in moved:
            continue
        quad = np.asarray(row['quad_world'], float)
        if quad.shape != (4, 3) or not np.isfinite(quad).all():
            raise ValueError('A finite photographic world quad is required')
        if row['entity'] == entity and not np.allclose(quad[:, 2], old_top, atol=.02):
            raise ValueError('Re-declare photographs on deforming support sides')
        quad[:, 2] += delta
        row['quad_world'] = quad.tolist()
    return result, photos, {
        'entity': entity, 'old_height_m': old_top, 'height_m': float(height),
        'translation_z_m': delta, 'moved_objects': sorted(moved - {entity}),
        'rebake_surfaces': [r['id'] for r in photos if r.get('entity') in moved],
        'approved': False,
        'required': ['rebake_photographs', 'rebuild_geometry', 'physics',
                     'remeasure_hybrid_coverage', 'whole_scene_appearance'],
    }
