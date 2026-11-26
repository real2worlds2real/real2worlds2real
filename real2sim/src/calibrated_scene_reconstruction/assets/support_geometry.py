"""Collision-box support heights in metric world coordinates."""
TOP_TOL = 0.05  # a box whose top is at most this far above an item's bottom can be what it stands on


def support_top(boxes, lo_xy, hi_xy, z_bottom):
    """Height of the highest collision box top under a footprint, at most TOP_TOL above its bottom; 0 (the floor)
    when none."""
    best = 0.0
    for lo, hi in boxes.values():
        overlap = lo[0] < hi_xy[0] and lo_xy[0] < hi[0] and lo[1] < hi_xy[1] and lo_xy[1] < hi[1]
        if overlap and hi[2] <= z_bottom + TOP_TOL:
            best = max(best, float(hi[2]))
    return best
