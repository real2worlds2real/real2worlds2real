"""Joint control schedules for real simulator demonstrations."""

import math


def sample_keyframes(keys, time):
    """Smooth a position command; the simulator still computes the actual motion."""
    if time <= keys[0][0]:
        return keys[0][1]
    for (start, first), (end, second) in zip(keys, keys[1:], strict=False):
        if time <= end:
            fraction = (time-start)/(end-start)
            fraction = fraction*fraction*(3-2*fraction)
            return first+(second-first)*fraction
    return keys[-1][1]


def resolve_controls(plan, names, limits):
    result = []
    for record in plan.get("controls", []):
        index = names.index(record["joint"])
        lower, upper = map(float, limits[index])
        keys, previous = [], -1.0
        for time, value in record["keyframes"]:
            value = lower if value == "lower" else upper if value == "upper" else float(value)
            if not math.isfinite(time) or time <= previous or not lower <= value <= upper:
                raise ValueError("Control times must increase and finite targets must lie inside joint limits")
            keys.append((float(time), value))
            previous = time
        if len(keys) < 2 or keys[0][0] != 0:
            raise ValueError("Provide at least two control keys beginning at time zero")
        result.append({"index": index, "joint": record["joint"], "keyframes": keys})
    if len({record["index"] for record in result}) != len(result):
        raise ValueError("Each joint has one control schedule")
    return result
