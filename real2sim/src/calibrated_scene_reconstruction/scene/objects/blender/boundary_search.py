"""Smooth source-search distance; never an accuracy or acceptance threshold.

Connected-component thresholds in local discrepancy are deliberately strict,
but make a discontinuous optimization objective. Symmetric boundary distance
can propose shape/pose refinements while every original gate remains mandatory.
Visibility-cut edges are excluded using the independent observation domain.
"""
import cv2
import numpy as np


def prepare(mask, domain=None):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('A nonempty observed silhouette is required')
    valid = np.ones(mask.shape, bool) if domain is None else np.asarray(domain, bool)
    if valid.shape != mask.shape:
        raise ValueError('Visibility domain must match source raster')
    safe = cv2.erode(valid.astype('uint8'), np.ones((3, 3), 'uint8'),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    boundary = mask & ~cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    boundary &= safe
    if not boundary.any():
        raise ValueError('No independently visible source boundary')
    distance = cv2.distanceTransform((~boundary).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dict(boundary=boundary, distance=distance, safe=safe,
                characteristic_length_px=float(np.sqrt(mask.sum())))


def distance(predicted, source):
    predicted = np.asarray(predicted, bool)
    if predicted.shape != source['boundary'].shape:
        raise ValueError('Prediction must match source raster')
    boundary = predicted & ~cv2.erode(predicted.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    boundary &= source['safe']
    if not boundary.any():
        return 1.
    reverse = cv2.distanceTransform((~boundary).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Each direction has equal weight: shrinking a silhouette to a few source
    # edge pixels cannot win by omitting the reverse/missing-boundary term.
    value = .5 * (float(np.mean(source['distance'][boundary]**2))
                  + float(np.mean(reverse[source['boundary']]**2)))
    return value / source['characteristic_length_px']**2
