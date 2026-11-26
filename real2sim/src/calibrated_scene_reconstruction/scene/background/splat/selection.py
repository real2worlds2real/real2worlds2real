"""Auditable background export filters, separate from task geometry acceptance."""
from numbers import Real
from pathlib import Path
import json

import numpy as np


def prepare_asset_directory(directory):
    """Replace exporter-owned files; preserve cousin assets and review evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('background.usdz', 'background_particlefield.usdz', 'cameras.npz', 'transform.json'):
        path = directory/name
        if path.is_dir():
            raise ValueError('An export file path is a directory: '+str(path))
    owned = [directory/name for name in ('background.usdz', 'background_particlefield.usdz', 'cameras.npz', 'transform.json')
             if (directory/name).exists()]
    if owned:
        history = directory/'asset_history'
        history.mkdir(exist_ok=True)
        numbers = [int(p.name[6:]) for p in history.glob('round_*') if p.name[6:].isdigit()]
        previous = history/f'round_{max(numbers, default=0)+1}'
        previous.mkdir()
        for path in owned:
            path.rename(previous/path.name)
    return directory


def prepare_work_directory(directory):
    """Retain previous selection evidence rather than deleting a diagnostic input."""
    directory = Path(directory)
    if directory.exists():
        history = directory.parent/'export_history'
        history.mkdir(exist_ok=True)
        numbers = [int(p.name[6:]) for p in history.glob('round_*') if p.name[6:].isdigit()]
        previous = history/f'round_{max(numbers, default=0)+1}'
        directory.rename(previous)
        (previous/'preserved_from.json').write_text(json.dumps({'previous_path': str(directory),
            'scope': 'Original export/selection files retained with original file mtimes.'}, indent=2)+'\n')
    directory.mkdir(parents=True)
    return directory


def appearance_filters(opacity, world_sigma, crop):
    opacity, world_sigma = np.asarray(opacity, float), np.asarray(world_sigma, float)
    minimum = crop.get('min_opacity', .005)
    maximum = crop.get('max_sigma_m', .5)
    if (isinstance(minimum, bool) or isinstance(maximum, bool)
            or not isinstance(minimum, Real) or not isinstance(maximum, Real)
            or not np.isfinite([minimum, maximum]).all()
            or not 0 <= minimum <= 1 or maximum <= 0):
        raise ValueError('Export min_opacity must be0..1 and max_sigma_m finite/positive')
    if (opacity.shape != world_sigma.shape or opacity.ndim != 1
            or not np.isfinite(opacity).all() or not np.isfinite(world_sigma).all()
            or ((opacity < 0) | (opacity > 1)).any() or (world_sigma <= 0).any()):
        raise ValueError('Finite opacity0..1 and positive metric sigma per Gaussian required')
    return opacity < minimum, world_sigma > maximum
