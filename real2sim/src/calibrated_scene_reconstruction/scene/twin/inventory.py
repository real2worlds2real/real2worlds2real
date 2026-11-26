"""Check input object coverage before judging the objects already in the graph.

Coverage is not geometry or task acceptance. A declared input may use a mesh,
a registered non-mesh component, or an explicitly requested task-reset tool. Missing
objects cannot disappear from the gate simply because they were never curated.
"""
import hashlib
import json


def assess(graph, task, mechanisms):
    if 'object_inventory_contract' not in task:
        return dict(applicable=False, passed=True, reason='Legacy input has no inventory contract')
    if type(task['object_inventory_contract']) is not int or task['object_inventory_contract'] != 1:
        raise ValueError('object_inventory_contract must be 1')
    objects = task.get('objects')
    routes = task.get('object_routes', {})
    if not isinstance(objects, dict) or not objects or not isinstance(routes, dict):
        raise ValueError('Inventory requires named input objects and explicit alternative routes')
    if set(routes) - set(objects):
        raise ValueError('Inventory route names an undeclared input object')
    from calibrated_scene_reconstruction.assets.mechanisms import DECLARATION_FIELDS
    from calibrated_scene_reconstruction.core.capabilities import require
    for key in mechanisms.keys() - DECLARATION_FIELDS:
        require('mechanism_fields', key)
    meshes = graph.get('dynamic_objects', {})
    rows, missing = {}, []
    for name, specification in objects.items():
        route = routes.get(name, {'kind': 'scene_object', 'object': name})
        kind = route.get('kind')
        if kind == 'scene_object':
            owner = route.get('object')
            present = owner in meshes
        elif kind == 'task_reset_only':
            owner = route.get('object')
            present = (specification.get('initial_presence') == 'task_reset_only'
                       and bool(specification.get('provenance'))
                       and bool(mechanisms.get('objects', {}).get(owner, {}).get('links')))
        else:
            from calibrated_scene_reconstruction.core.capabilities import invoke
            owner, present = invoke('inventory_route', kind, name, route, mechanisms)
        rows[name] = {'kind': kind, 'owner': owner, 'covered': bool(present)}
        if not present:
            missing.append(name)
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    components, declarations = {}, {}
    for key in entries('inventory_components'):
        current, declared, absent = invoke('inventory_components', key, graph, mechanisms, rows)
        components.update(current); declarations.update(declared); missing.extend(absent)
    proof = {'objects': rows, 'components': components, 'task_objects': objects,
             'routes': routes, 'mechanism_components': declarations}
    checksum = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(applicable=True, passed=not missing, objects=rows, components=components,
                sha256=checksum, reason='Missing input objects: '+', '.join(missing) if missing else
                'Every input object has a declared reconstruction route; geometry/physics checks still required')


def state(session, graph, task):
    if 'object_inventory_contract' not in task:
        return assess(graph, task, {})
    try:
        mechanisms = session.config('agent/mechanisms.json', {})
        return assess(graph, task, mechanisms)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return dict(applicable=True, passed=False, reason=str(exc))
