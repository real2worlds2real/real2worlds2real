"""Read generated ragged mask caches across reconstruction/Isaac NumPy versions.

NumPy 2 writes private ``numpy._core`` pickle names; Isaac 5.1 uses NumPy 1.
Map only the array constructors needed by these local caches. Do not install
global module aliases, change dependencies, or rewrite reviewed source bytes.
"""
import pickle

import numpy as np


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        core = np._core if hasattr(np, '_core') else np.core
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(core.multiarray, name)
        if module in {'numpy.core.numeric', 'numpy._core.numeric'} and name == '_frombuffer':
            return core.numeric._frombuffer
        raise pickle.UnpicklingError('Unexpected mask-cache constructor: '+module+'.'+name)


def read_masks(archive, key='masks'):
    """Return unchanged 2D boolean masks; never run arbitrary pickle globals."""
    with archive.zip.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if reader is None:
            raise ValueError('Unsupported mask-cache NPY version')
        shape, _, dtype = reader(stream)
        if dtype.hasobject:
            values = _ArrayUnpickler(stream).load()
            if not isinstance(values, np.ndarray) or values.shape != shape:
                raise ValueError('Mask cache differs from its declared shape')
        else:
            values = archive[key]
    masks = []
    for value in values:
        value = np.asarray(value)
        if value.dtype.hasobject and all(isinstance(item, (bool, np.bool_)) for item in value.flat):
            value = value.astype(bool)
        masks.append(value)
    if any(value.ndim != 2 or value.dtype != np.bool_ for value in masks):
        raise ValueError('Mask cache requires two-dimensional boolean arrays')
    return masks
