"""Coordinate-independent error of a positive physical inertia tensor.

USD stores principal moments and axes as float32. Elementwise relative errors
on nearly zero products of inertia are not physical errors: a rotation changes
which entries are near zero. Bound the relative rotational energy for every
angular velocity instead, including the smallest principal moment.
"""
import numpy as np


def relative_energy_error(reference, actual):
    """Return max |w.T (actual-reference) w| / (w.T reference w).

Both tensors must be finite, symmetric and strictly positive definite. The
result is invariant under shared rotations and changes of mass/length units.
It also detects corruption of a small principal moment that a Frobenius norm
relative to the largest moment can hide.
    """
    reference, actual = np.asarray(reference, float), np.asarray(actual, float)
    if (reference.shape != (3, 3) or actual.shape != (3, 3)
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()):
        raise ValueError('Finite 3x3 physical inertia tensors required')
    scale = float(np.max(np.abs(reference)))
    if scale <= 0:
        raise ValueError('Positive physical inertia required')
    reference, actual = reference / scale, actual / scale
    for tensor in (reference, actual):
        if (np.linalg.norm(tensor - tensor.T) > 1e-12 * np.linalg.norm(tensor)
                or np.linalg.eigvalsh(tensor).min() <= 0):
            raise ValueError('Symmetric positive physical inertia required')
    values, axes = np.linalg.eigh(reference)
    inverse_root = (axes / np.sqrt(values)) @ axes.T
    difference = inverse_root @ (actual - reference) @ inverse_root
    return float(np.max(np.abs(np.linalg.eigvalsh((difference + difference.T) / 2))))
