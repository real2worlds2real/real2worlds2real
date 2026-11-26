"""Expose cross-sheet root intersections separately from free-sheet contacts.

An attachment is not permission to hide overlapping material. Conversely,
embedded composite construction cannot be declared universally invalid from a
triangle count alone. This diagnostic requires a structural/native review and
never installs a collision filter or grants acceptance.
"""
import itertools

import numpy as np


def _components(count, faces):
    parent = np.arange(count)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return int(i)
    for a, b, c in faces:
        for j in (b, c):
            x, y = root(int(a)), root(int(j))
            if x != y:parent[y] = x
    labels = np.array([root(i) for i in range(count)])
    return np.unique(labels, return_inverse=True)[1]


def audit(parts, bindings, *, device='cpu'):
    from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
    by_name = {p['name']: p for p in parts}
    if len(by_name) != len(parts):raise ValueError('Distinct actual physical part names required')
    bonds = {b['surface']: b for b in bindings}
    if len(bonds) != len(bindings):raise ValueError('One explicit root ownership declaration per surface required')
    vertices, faces, groups, categories, owners, records = [], [], [], [], [], []
    offset = group_offset = 0
    for part in parts:
        if part['kind'] != 'cloth':continue
        v, f = np.asarray(part['V'], float), np.asarray(part['F'], int)
        if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
                or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
                or f.min() < 0 or f.max() >= len(v)):
            raise ValueError('Finite indexed actual sheet triangles required')
        bond = bonds.get(part['name']); root = np.zeros(len(v), bool); owner = None
        if bond:
            owner = bond['rigid']
            if owner not in by_name or by_name[owner]['kind'] != 'rigid':
                raise ValueError('Root must belong to a real rigid part')
            ids = np.asarray(bond['vertices'])
            if (ids.dtype.kind not in 'iu' or not len(ids) or ids.min() < 0
                    or ids.max() >= len(v) or len(np.unique(ids)) != len(ids)):
                raise ValueError('Actual distinct bound-node indices required')
            root[ids] = True
        components = _components(len(v), f)
        for index in range(int(components.max())+1):
            records.append({'surface': part['name'], 'component': index, 'rigid_owner': owner})
        count = root[f].sum(1)
        categories.extend(np.where(count == 3, 0, np.where(count > 0, 1, 2)))
        faces.extend(f+offset); groups.extend(components[f[:, 0]]+group_offset)
        owners.extend([owner]*len(f)); vertices.extend(v)
        offset += len(v); group_offset += int(components.max())+1
    if not faces:raise ValueError('At least one actual cloth surface required')
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    groups, categories = np.asarray(groups), np.asarray(categories)
    names = ['fully_bound', 'transition', 'free']; raw = {}; counts = {}; examples = []
    def measure(selected):
        ids = np.flatnonzero(selected)
        if not len(ids):return {'intersecting_pairs': 0, 'example_face_pairs': []}, ids
        result = self_intersections(vertices, faces[ids], face_groups=groups[ids], device=device)
        return result, ids
    for category in range(3):
        result, ids = measure(categories == category); raw[(category,)] = result['intersecting_pairs']
        counts[names[category]+'__'+names[category]] = result['intersecting_pairs']
        examples.extend((int(ids[a]), int(ids[b])) for a, b in result['example_face_pairs'])
    for a, b in itertools.combinations(range(3), 2):
        result, ids = measure(np.isin(categories, [a, b]))
        count = result['intersecting_pairs']-raw[(a,)]-raw[(b,)]
        if count < 0:raise ValueError('Inconsistent contact partition; preserve diagnostics instead of accepting')
        counts[names[a]+'__'+names[b]] = count
        examples.extend((int(ids[x]), int(ids[y])) for x, y in result['example_face_pairs']
                        if categories[ids[x]] != categories[ids[y]])
    example_rows = []
    for a, b in sorted(set(examples)):
        left, right = records[groups[a]], records[groups[b]]
        example_rows.append({'face_ids': [a, b], 'surface0': left, 'surface1': right,
            'categories': [names[categories[a]], names[categories[b]]],
            'same_rigid_owner': left['rigid_owner'] is not None and left['rigid_owner'] == right['rigid_owner']})
    root_pairs = sum(value for key, value in counts.items() if key != 'free__free')
    return {'cross_component_intersections': counts, 'root_intersections': root_pairs,
        'structural_review_required': bool(root_pairs), 'example_pairs': example_rows,
        'complete_pair_enumeration': False, 'triangles': len(faces), 'components': records,
        'collision_filters_changed': False, 'object_approved': False, 'task_approved': False,
        'scope': 'Actual separate material components, including components within one physical actor. '
                 'Fully-bound and transition faces remain visible as diagnostics. Review actual '
                 'embedding, material ownership and native contacts; root labels alone establish '
                 'neither a legal composite exception nor a universal rejection.'}
