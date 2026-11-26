"""Editable phenomenological material transitions, without a thermal solver."""

from __future__ import annotations

import math

import numpy as np

MATERIAL_FIELDS = {
    "viscosity",
    "yield_stress",
    "yield_pressure",
    "young_modulus",
    "friction",
    "tensile_yield_ratio",
}
APPEARANCE_FIELDS = {
    "base_color",
    "roughness",
    "transmission",
    "subsurface_weight",
    "subsurface_scale",
    "ior",
    "coat_weight",
    "coat_roughness",
}


def validate_transition(value):
    if not isinstance(value, dict):
        raise ValueError("Material transition must be an object")
    allowed = {
        "start_seconds",
        "duration_seconds",
        "height_delay_seconds_per_m",
        "height_origin_m",
        "material",
        "appearance",
    }
    if set(value) - allowed:
        raise ValueError("Unknown material transition fields")
    for key, default in (
        ("start_seconds", 0),
        ("duration_seconds", 0),
        ("height_delay_seconds_per_m", 0),
        ("height_origin_m", 0),
    ):
        x = value.get(key, default)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError("Transition timing and spatial coefficients must be finite")
    if (
        value.get("duration_seconds", 0) <= 0
        or value.get("start_seconds", 0) < 0
        or value.get("height_delay_seconds_per_m", 0) < 0
    ):
        raise ValueError("Transition requires positive duration and nonnegative start/delay")
    for section, fields in (("material", MATERIAL_FIELDS), ("appearance", APPEARANCE_FIELDS)):
        parameters = value.get(section, {})
        if not isinstance(parameters, dict) or set(parameters) - fields:
            raise ValueError(f"Unsupported transition {section} field")
        for key, x in parameters.items():
            if key == "base_color":
                if (
                    not isinstance(x, list)
                    or len(x) != 4
                    or any(
                        not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                        for v in x
                    )
                ):
                    raise ValueError("Transition base_color must be four values in [0,1]")
            elif (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x < 0
            ):
                raise ValueError(
                    "Transition material and appearance values must be finite and nonnegative"
                )
    if not value.get("material") and not value.get("appearance"):
        raise ValueError("Transition needs a material or appearance target")
    return value


def transition_progress(points, seconds, specification, previous=None):
    points = np.asarray(points)
    delay = np.maximum(
        points[:, 2] - specification.get("height_origin_m", 0), 0
    ) * specification.get("height_delay_seconds_per_m", 0)
    fraction = np.clip(
        (seconds - specification.get("start_seconds", 0) - delay)
        / specification["duration_seconds"],
        0,
        1,
    )
    result = fraction * fraction * (3 - 2 * fraction)
    return np.maximum(result, previous) if previous is not None else result
