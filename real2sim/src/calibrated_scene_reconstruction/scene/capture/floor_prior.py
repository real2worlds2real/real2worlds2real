"""Observed floor uncertainty for CAD calibration, without task dimensions."""
import numpy as np


def estimate(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        raise ValueError('Floor uncertainty needs at least30 finite independent observed tracks')
    z = points[:, 2]
    median = float(np.median(z))
    sigma = max(1e-6, float(1.4826*np.median(abs(z-median))))
    return {'sigma_map': sigma, 'max_offset_map': 3*sigma, 'weight': .005,
            'tracks': len(points), 'observed_median_z_map': median,
            'source': 'Current raw floor-labelled COLMAP tracks; robust MAD, not task/support dimensions',
            'world_rebase': 'After acceptance, ground CAD wheel surfaces and translate the entire task/camera frame consistently'}
