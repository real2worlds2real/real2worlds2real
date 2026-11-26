"""Transfer this capture's legal CAD geometry to the GPU environment without USD imports."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .articulated_model import ArticulatedModel, Joint


def save(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(model.surfaces)
    arrays = {}
    groups = {'surfaces': model.surfaces, 'normals': model.normals,
              'triangles': model.triangles, 'floor_vertices': model.floor_vertices}
    for kind, group in groups.items():
        for index, name in enumerate(names):
            if name in group:
                arrays[f'{kind}_{index}'] = group[name]
    np.savez_compressed(directory/'cad.npz', **arrays)
    joints = [{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in asdict(joint).items()} for joint in model.joints]
    metadata = {'schema': 1, 'root': model.root, 'names': names, 'joints': joints,
                'source': model.source, 'dependencies': list(model.dependencies)}
    (directory/'cad.json').write_text(json.dumps(metadata, indent=2)+'\n')


def load(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'cad.json').read_text())
    if metadata['schema'] != 1:
        raise ValueError('Unsupported visual CAD transfer format')
    arrays = np.load(directory/'cad.npz')
    groups = {kind: {name: arrays[f'{kind}_{index}'] for index, name in enumerate(metadata['names'])
                     if f'{kind}_{index}' in arrays.files}
              for kind in ('surfaces', 'normals', 'triangles', 'floor_vertices')}
    joints = []
    for values in metadata['joints']:
        values = dict(values)
        for key in ('frame0', 'frame1_inverse', 'axis'):
            values[key] = np.asarray(values[key], float)
        joints.append(Joint(**values))
    return ArticulatedModel(metadata['root'], joints, groups['surfaces'], metadata['source'],
                            groups['floor_vertices'], tuple(metadata['dependencies']),
                            groups['normals'], groups['triangles'])
