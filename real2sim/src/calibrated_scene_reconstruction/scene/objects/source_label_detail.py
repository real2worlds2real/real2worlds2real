"""Source-only native-pixel detail crops for small-part identity review."""
import cv2
import numpy as np


def render(rgb, mask, *, margin=32, max_side=1024):
    """Keep context and every source label; use nearest pixels for enlargement.

    A whole-scene thumbnail can hide a narrow part's false continuation onto
    another object. This additional source view exposes that mistake without
    reading modeled geometry or declaring the label correct.
    """
    rgb, mask = np.asarray(rgb), np.asarray(mask)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
            or mask.shape != rgb.shape[:2] or mask.dtype != bool
            or type(margin) is not int or margin < 0
            or type(max_side) is not int or max_side < 32):
        raise ValueError('Native uint8 RGB and matching boolean source mask required')
    if not mask.any():
        return None, {'observed': False, 'basis': 'No source positive; original whole frame remains available.'}
    y, x = np.nonzero(mask)
    x0, y0 = max(0, int(x.min())-margin), max(0, int(y.min())-margin)
    x1, y1 = min(mask.shape[1], int(x.max())+margin+1), min(mask.shape[0], int(y.max())+margin+1)
    source = rgb[y0:y1, x0:x1].copy();positive = mask[y0:y1, x0:x1]
    overlay = source.copy()
    contours, _ = cv2.findContours(positive.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 1)
    height, width = source.shape[:2]
    factor = min(4., max_side/max(height, width))
    shape = (max(1, round(width*factor)), max(1, round(height*factor)))
    tiles = [cv2.resize(tile, shape, interpolation=cv2.INTER_NEAREST) for tile in [source, overlay]]
    return np.hstack(tiles), {'observed': True, 'source_window_xywh': [x0, y0, x1-x0, y1-y0],
                             'display_shape_wh': list(shape), 'interpolation': 'nearest',
                             'source_positive_pixels': int(mask.sum()),
                             'scope': 'Original RGB pixels and source labels only; no generated detail or model acceptance.'}
