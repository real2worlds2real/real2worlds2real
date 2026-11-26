"""Bind source approval to the actual binary pixel-coverage implementation."""
import hashlib
from pathlib import Path


def sources():
    root = Path(__file__).resolve().parents[3]
    return [root/'core/geometry/raster.py', root/'core/geometry/pixel_coverage.py',
            root/'scene/objects/object_refine.py', root/'core/geometry/visibility.py',
            Path(__file__).resolve()]


def current():
    return {'version': 1, 'sampling': 'opencv_integer_pixel_centres',
            'acceptance_backend': 'cpu_exact_triangle_union_and_visible_part_rays',
            'source_sha256': {str(path.relative_to(Path(__file__).resolve().parents[3])):
                              hashlib.sha256(path.read_bytes()).hexdigest() for path in sources()}}


def verify(record):
    if record != current():
        raise ValueError('Source raster contract missing or changed; remeasure and review a new round '
                         '(legacy conservative coverage is diagnostic only)')
