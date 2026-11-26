"""Known-strip sampling controls, independent of any reconstructed candidate.

These diagnostics expose what native-label nearest resampling and bounded
source-pixel perturbations do to the existing metrics. They grant no tolerance
and cannot establish camera uncertainty or approve real source geometry.
"""
import numpy as np
from .object_refine import pieces


def strip(width_crop_px, source_to_crop, *, shift_source_px=0., width_ratio=1.,
          angle_delta_deg=0., angle_deg=13., phase_source_px=(.2, .4), size=192):
    if (not np.isfinite([width_crop_px, source_to_crop, shift_source_px, width_ratio,
                         angle_delta_deg, angle_deg, *phase_source_px]).all()
            or not 0 < width_crop_px < size/4 or not 0 < source_to_crop <= 1
            or width_ratio <= 0 or len(phase_source_px) != 2):
        raise ValueError('Resolved positive strip dimensions and native-to-crop sampling required')
    y, x = np.mgrid[:size, :size]
    exact = np.stack([x, y], axis=-1)/source_to_crop
    # INTER_NEAREST picks floor(output_pixel/source_to_crop) in the raw crop.
    sampled = np.floor(exact)
    centre = np.array([size/2, size/2])/source_to_crop+phase_source_px
    angle = np.deg2rad(angle_deg)
    normal = np.array([np.cos(angle), np.sin(angle)])
    direction = np.array([-normal[1], normal[0]])
    width = width_crop_px/source_to_crop
    half_length = size*.33/source_to_crop
    reference = ((np.abs((sampled-centre)@normal) <= width/2)
                 & (np.abs((sampled-centre)@direction) <= half_length))
    other_angle = np.deg2rad(angle_deg+angle_delta_deg)
    other_normal = np.array([np.cos(other_angle), np.sin(other_angle)])
    other_direction = np.array([-other_normal[1], other_normal[0]])
    predicted = ((np.abs((exact-centre-normal*shift_source_px)@other_normal) <= width*width_ratio/2)
                 & (np.abs((exact-centre-normal*shift_source_px)@other_direction) <= half_length))
    return {'iou': float((reference & predicted).sum()/max(1, (reference | predicted).sum())),
            'local_difference': float(sum(p[0] for p in pieces(reference, predicted, .003))),
            'reference_pixels': int(reference.sum()), 'predicted_pixels': int(predicted.sum()),
            'width_crop_px': float(width_crop_px), 'source_to_crop': float(source_to_crop),
            'shift_source_px': float(shift_source_px), 'width_ratio': float(width_ratio),
            'angle_delta_deg': float(angle_delta_deg), 'accepted': False}
