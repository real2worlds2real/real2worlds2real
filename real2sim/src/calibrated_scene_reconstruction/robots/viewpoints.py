"""Position the review cameras relative to a valid robot placement from the current capture."""
import numpy as np


def is_photo_base(c, photo_base, tol=0.02):
    """The candidate apps/placements.py made of the real robot's base (its first: id 0 at photo_robot.base)."""
    if not photo_base or "xy" not in photo_base:
        return False
    return bool(np.hypot(c["base"][0] - photo_base["xy"][0], c["base"][1] - photo_base["xy"][1]) < tol
                and abs((c["base"][2] - photo_base["yaw_deg"] + 180) % 360 - 180) < 1.0)


def view_bases(candidates, photo_base=None):
    """(bases, warning or None): the accepted bases, the real robot's first when it is one of them."""
    presentation = [c for c in candidates if c.get('presentation_only')]
    if presentation:
        return sorted(presentation,key=lambda c:not is_photo_base(c,photo_base)),None
    accepted = [c for c in candidates if c["accepted"]]
    # the twin's robot stands where it was measured, accepted or not; its reach is reported, never held
    real = [c for c in candidates if is_photo_base(c, photo_base) and c.get("collision_free", True)]
    if real:
        return real + [c for c in accepted if c not in real], None
    if accepted:
        return sorted(accepted, key=lambda c: not is_photo_base(c, photo_base)), None
    reach = [c for c in candidates if c.get("reachable")]
    note = (f"no accepted base: the robot stands at the {len(reach) or len(candidates)} rejected candidates that "
            f"{'reach the target' if reach else 'were tried'}, for the views only (the rejections: placements)")
    return sorted(reach or list(candidates), key=lambda c: not is_photo_base(c, photo_base)), note


def coverage_score(candidate):
    """Unscored presentation standpoints are usable without implying a reach check."""
    return float(candidate.get('coverage') or 0.)
