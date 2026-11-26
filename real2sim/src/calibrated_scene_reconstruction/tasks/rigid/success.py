"""The evaluation's success criterion, standard library only: tools/eval_place reports it from any python (the
Isaac launchers' system python3 has no numpy), tools/eval_breakdown and the rest import it from here."""


# the evaluation's distance from the goal (the plate's centre): an upright cup resting on the plate 5-8 cm off its
# centre, outside the collection's accept region (the flat surface shrunk by half the cup's radius, 5 cm), was counted
# a failure though it stands on the plate (about 8 % of the rollouts). The collection keeps its accept region.
PLATE_TOL_M = 0.08


def on_mat(r):
    """The evaluation's success: the cup ends within PLATE_TOL_M of the goal, upright (<= 15 deg), resting on the
    support (base within 2 cm) and let go (the hand >= 5 cm from it); the fingers need not be open (the collection's
    check, rollout["success"], also asks that)."""
    p = r.get("place") or {}
    near = p.get("goal_err_m") is not None and p["goal_err_m"] <= PLATE_TOL_M
    return bool(r["success"]) or (near and p["tilt_deg"] <= 15.0 and abs(p["rest_dz_m"]) <= 0.02
                                  and p["tcp_dist_m"] >= 0.05)
