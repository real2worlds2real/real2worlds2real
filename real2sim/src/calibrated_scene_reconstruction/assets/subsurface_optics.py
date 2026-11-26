"""Explicit optical scattering priors, separate from mechanical material data.

Blender and OmniSurface use different scattering models. Carrying weight and
metric mean-free-path proposals preserves intent, not radiometric equivalence;
actual Isaac renders still need source review.
"""
import numpy as np


def _number(value, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError('Finite numeric ' + name + ' required')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Finite numeric ' + name + ' required')
    return value


def validate(value):
    if not isinstance(value, dict) or set(value) != {'weight', 'radius_m', 'anisotropy'}:
        raise ValueError('Subsurface requires weight, RGB radius_m and anisotropy')
    weight = _number(value['weight'], 'subsurface weight')
    anisotropy = _number(value['anisotropy'], 'subsurface anisotropy')
    radius = np.asarray(value['radius_m'])
    if (radius.shape != (3,) or radius.dtype.kind not in 'fiu'
            or not np.isfinite(radius).all() or np.any(radius <= 0)
            or not 0 < weight <= 1 or not -1 < anisotropy < 1):
        raise ValueError('Positive RGB scattering distances and bounded weight/anisotropy required')
    return {'weight': weight, 'radius_m': radius.astype(float).tolist(), 'anisotropy': anisotropy}


def from_blender(inputs):
    """Read constant Principled inputs. Inactive legacy materials stay unchanged."""
    weight_socket = inputs.get('Subsurface Weight') or inputs.get('Subsurface')
    if weight_socket is None:
        return None
    if getattr(weight_socket, 'is_linked', False):
        raise ValueError('Linked subsurface weight requires an explicit optical texture export')
    weight = _number(weight_socket.default_value, 'subsurface weight')
    if weight == 0:
        return None
    def read(name, default=None):
        socket = inputs.get(name)
        if socket is None:
            if default is None:
                raise ValueError('Missing Blender ' + name)
            return default
        if getattr(socket, 'is_linked', False):
            raise ValueError('Linked subsurface input requires an explicit optical texture export')
        return socket.default_value
    scale = _number(read('Subsurface Scale', 1.), 'subsurface scale')
    if scale <= 0:
        raise ValueError('Positive subsurface scale required')
    return validate({'weight': weight,
                     'radius_m': (np.asarray(read('Subsurface Radius'), float) * scale).tolist(),
                     'anisotropy': read('Subsurface Anisotropy', 0.)})


def omni_parameters(value, meters_per_unit):
    """Keep scattering length constant across USD length units."""
    value = validate(value)
    unit = _number(meters_per_unit, 'stage meters per unit')
    if unit <= 0:
        raise ValueError('Positive stage length unit required')
    radius = np.asarray(value['radius_m'])
    largest = float(radius.max())
    return {'enable_diffuse_transmission': True,
            'subsurface_weight': value['weight'],
            'subsurface_scattering_color': (radius / largest).tolist(),
            'subsurface_scale': largest / unit,
            'subsurface_anisotropy': value['anisotropy']}
