"""Physical solid-contact screening, separate from numerical contact padding."""
import math


def solid_tolerance(thickness, discretization, *, legacy_rest_offset=None):
    """Keep the existing two-thickness budget independent of solver padding.

    Older particle assets without a declared physical thickness retain their
    legacy four-rest-offset budget. This fallback is not a thickness estimate.
    """
    if not math.isfinite(discretization) or discretization<0:
        raise ValueError('Finite nonnegative rigid discretization bound required')
    if thickness is None:
        if legacy_rest_offset is None or not math.isfinite(legacy_rest_offset) or legacy_rest_offset<=0:
            raise ValueError('Physical thickness or an explicit legacy contact offset required')
        budget=4*legacy_rest_offset
    else:
        if not math.isfinite(thickness) or thickness<=0:
            raise ValueError('Positive finite physical sheet thickness required')
        budget=2*thickness
    return float(max(budget,discretization))
