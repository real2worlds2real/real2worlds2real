"""Finite-radius accordion cross-sections for offline sheet construction.

The coordinate is material arclength. These are shape proposals, not a new
rest metric, native constraints, or evidence that a drape is physically valid.
"""
import numpy as np


def accordion(width, panels, half_angle, radius, samples):
    """Return a centred 2D curve sampled uniformly in material arclength.

    Straight panels alternate their tangent angle. Circular folds join them
    continuously, so a near-closed pleat retains a finite bend radius rather
    than collapsing neighbouring layers onto a zero-radius crease.
    """
    if (not np.isfinite([width, half_angle, radius]).all()
            or width <= 0 or radius <= 0 or not 0 <= half_angle <= np.pi/2
            or type(panels) is not int or panels < 2
            or type(samples) is not int or samples < 2):
        raise ValueError('Finite positive dimensions and bounded accordion angle required')
    arc = 2 * half_angle * radius
    straight = (width - (panels-1)*arc) / panels
    if straight <= 0:
        raise ValueError('Fold arcs consume the available material width')
    positions = np.linspace(0., width, samples)
    result = np.zeros((samples, 2))
    start = np.zeros(2); travelled = 0.
    for panel in range(panels):
        angle = half_angle * (-1 if panel % 2 == 0 else 1)
        direction = np.array([np.cos(angle), np.sin(angle)])
        selected = (positions >= travelled) & (positions <= travelled+straight+1e-12)
        result[selected] = start + (positions[selected]-travelled)[:, None]*direction
        start = start + straight*direction
        travelled += straight
        if panel == panels-1 or arc == 0:
            continue
        sign = 1 if panel % 2 == 0 else -1
        selected = (positions >= travelled) & (positions <= travelled+arc+1e-12)
        theta = angle + sign*(positions[selected]-travelled)/radius
        result[selected] = start + radius/sign*np.column_stack([
            np.sin(theta)-np.sin(angle), -np.cos(theta)+np.cos(angle)])
        end = angle + sign*2*half_angle
        start = start + radius/sign*np.array([
            np.sin(end)-np.sin(angle), -np.cos(end)+np.cos(angle)])
        travelled += arc
    result -= (result.min(0)+result.max(0))/2
    return result
