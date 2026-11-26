"""Adapt a Gaussian asset's declared photometric settings to the renderer."""
from numbers import Real
import math

PROFILE_KEY = 'r2s2r_background_render'
PREFIXES = ('/rtx/post/tonemap/', '/rtx/post/registeredCompositing/',
            '/rtx/rtpt/gaussian/skipTonemapping/')


def photometric_profile(layer_data):
    settings = layer_data.get('renderSettings', {})
    if not isinstance(settings, dict):
        raise ValueError('Gaussian renderSettings must be a dictionary')
    result = {}
    for key, value in settings.items():
        if not isinstance(key, str):
            raise ValueError('Renderer setting names must be strings')
        path = '/'+key.replace(':', '/') if key.startswith('rtx:') else key
        if not (path.startswith(PREFIXES) or path == '/rtx/post/histogram/enabled'):
            continue
        if not isinstance(value, (bool, Real)) or (isinstance(value, Real) and not math.isfinite(value)):
            raise ValueError('Photometric render settings must be finite scalars')
        result[path] = value
    return result
