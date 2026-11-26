"""Explicit input routing and included-liquid ownership."""


def route(name, route, mechanisms):
    continuum = mechanisms.get('continuum_components', {})
    owner = route.get('component')
    component = continuum.get(owner, {})
    present = (component.get('task_object') == name and bool(component.get('backend'))
               and bool(component.get('checks')))
    return owner, present


def contents(graph, mechanisms, rows):
    meshes = graph.get('dynamic_objects', {})
    continuum = mechanisms.get('continuum_components', {})
    missing = []
    # Included fluids (e.g. water inside a bottle) need an actual container
    # object even when they are not separately named in the input inventory.
    components = {}
    for name, value in continuum.items():
        container = value.get('container_link')
        if container:
            owner, sep, link = container.partition('/')
            valid = (bool(sep and link) and owner in meshes
                     and link in mechanisms.get('objects', {}).get(owner, {}).get('links', {}))
        else:
            valid = any(r['kind'] == 'continuum' and r['owner'] == name for r in rows.values())
        components[name] = {'container_link': container, 'covered': bool(valid)}
        if not valid:
            missing.append('component:'+name)
    return components, continuum, missing
