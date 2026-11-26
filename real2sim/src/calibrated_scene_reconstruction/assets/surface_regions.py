"""Compact named material regions on the actual mesh, independent of rigid links."""
import numpy as np


def annotate(data, parts):
    if not parts or not {'face_material', 'material_names'} <= set(data):
        return parts
    names = list(map(str, data['material_names'])); labels = np.asarray(data['face_material'])
    if (len(set(names)) != len(names) or labels.shape != (len(data['F']),)
            or labels.dtype.kind not in 'iu' or labels.min() < 0 or labels.max() >= len(names)):
        raise ValueError('Surface regions require actual triangle/material correspondence')
    for part in parts:
        first, count = part['face_start'], part['triangle_count']
        selected = labels[first:first+count]; regions = {}
        for label in np.unique(selected):
            mask = selected == label; complement = bool(mask.sum() > count/2)
            regions[names[label]] = {'complement': complement,
                                    'faces': np.flatnonzero(~mask if complement else mask).tolist()}
        part['surface_regions'] = regions
    return parts


def select(parts, names, face_count):
    target = np.zeros(face_count, bool); found = set()
    for part in parts or []:
        first, count = part['face_start'], part['triangle_count']
        if first < 0 or count <= 0 or first+count > face_count:
            raise ValueError('Surface region link lies outside the actual mesh')
        for name in names:
            item = part.get('surface_regions', {}).get(name)
            if item is None:
                continue
            ids = np.asarray(item['faces'], int)
            if (ids < 0).any() or (ids >= count).any() or type(item.get('complement')) is not bool:
                raise ValueError('Surface region face indices or complement flag are invalid')
            region = np.full(count, item['complement'], bool); region[ids] = not item['complement']
            target[first:first+count] |= region; found.add(name)
    if found != set(names) or not target.any():
        raise ValueError('Missing actual named surface material: '+', '.join(sorted(set(names)-found)))
    return target
