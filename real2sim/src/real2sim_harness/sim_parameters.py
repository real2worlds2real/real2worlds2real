"""Editable physical defaults, kept independent of asset geometry and appearance."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from .artifacts import atomic_json

DEFAULTS = {
    "schema_version": 1,
    "provenance": "assumed defaults; editable independently of the visual asset",
    "environment": {"gravity": [0.0, 0.0, -9.81], "air_drag_rate_per_second": 0.0},
    "rigid": {"mass_kg": None, "density_kg_m3": 100.0, "enable_ccd": False},
    "material": {"friction": 0.6, "restitution": 0.0},
    "physx": {"mass_properties": {}, "friction_combine_mode": None, "enabled_collision_pairs": []},
    # Revolute gains use SI: N*m/rad and N*m*s/rad; max_force is N*m.
    "joint_drive": {"stiffness": 0.0, "damping": 0.0, "max_force": 1000.0},
    "surface": {"density": 1.0, "particle_radius": 0.008,
                "tri_ke": 10000.0, "tri_ka": 10000.0, "tri_kd": 0.0000015,
                "edge_ke": 5.0, "edge_kd": 0.01},
    "solver": {"substeps": 8, "iterations": 20, "self_contact": True,
               "self_contact_radius": 0.0, "self_contact_margin": 0.0,
               "self_contact_topology_distance": 2, "self_contact_vertex_capacity": 64,
               "self_contact_edge_capacity": 128,
               "hybrid_full_surface_contact": True, "hybrid_rigid_soft_dat": False},
    "constraints": {"use_source_pins": False, "lift_selection": "all",
                    "lift_distance_m": .08, "lift_duration_seconds": None,
                    "lift_delay_seconds": 0.0, "quiet_start_seconds": None,
                    "quiet_speed_tolerance_m_s": None},
}


def resolve_surface_contact(parameters, *, source_physics, median_edge_length):
    """Choose thin-surface contact distances without assuming centimetre-scale cloth."""
    solver = parameters["solver"]
    distance = solver["self_contact_radius"]
    if distance:
        origin = "explicit physics_parameters override"
    else:
        distances = [float(record["collision_distance_m"]) for record in source_physics
                     if isinstance(record.get("collision_distance_m"), (int, float))
                     and math.isfinite(record["collision_distance_m"])
                     and record["collision_distance_m"] > 0]
        if distances:
            distance = min(distances)
            origin = "source cloth collision distance; assumed starting point, not calibrated material equivalence"
        else:
            if not math.isfinite(median_edge_length) or median_edge_length <= 0:
                raise ValueError("A positive finite control-mesh edge length is required")
            distance = min(2*parameters["surface"]["particle_radius"], median_edge_length*.25)
            origin = "thin-surface assumption bounded by control-mesh edge scale"
    query = solver["self_contact_margin"] or 2*distance
    if query < distance:
        raise ValueError("Self-contact detection margin must cover its interaction distance")
    return {"interaction_distance_m": distance, "query_distance_m": query,
            "median_control_edge_m": median_edge_length, "provenance": origin,
            "scope": "Thin-surface contact initialization. Explicit overrides remain available for other models."}


def load_parameters(folder: Path, *, create=False):
    folder = Path(folder)
    path = folder / "physics_parameters.json"
    values = deepcopy(DEFAULTS)
    if path.is_file():
        provided = json.loads(path.read_text())
        for key, value in provided.items():
            if key not in values:
                raise ValueError(f"Unknown physics parameter section: {key}")
            if isinstance(values[key], dict):
                if not isinstance(value, dict) or set(value) - set(values[key]):
                    raise ValueError(f"Unknown or invalid physics parameters in {key}")
                values[key].update(value)
            else:
                values[key] = value
    else:
        manifest_path = folder / "manifest.json"
        if manifest_path.is_file():
            source = json.loads(manifest_path.read_text()).get("physics_assumptions", {})
            for key in values["rigid"]:
                if key in source:
                    values["rigid"][key] = source[key]
            for key in values["material"]:
                if key in source:
                    values["material"][key] = source[key]
            if "gravity" in source:
                values["environment"]["gravity"] = source["gravity"]
    if values["schema_version"] != 1:
        raise ValueError("Unsupported physical parameter schema")
    from .physx_mass_properties import validate_mass_properties
    mass_properties = values["physx"]["mass_properties"]
    if not isinstance(mass_properties, dict):
        raise ValueError("PhysX mass_properties must map base or joint:<name> to coupled mass records")
    for key, record in mass_properties.items():
        if not isinstance(key, str) or not (key == "base" or key.startswith("joint:") and len(key) > 6):
            raise ValueError("PhysX mass properties select base or joint:<name>")
        validate_mass_properties(record)
    combine = values["physx"]["friction_combine_mode"]
    if combine is not None and combine not in ("average", "min", "max", "multiply"):
        raise ValueError("Invalid PhysX friction combine mode")
    pairs = values["physx"]["enabled_collision_pairs"]
    if not isinstance(pairs, list):
        raise ValueError("PhysX enabled_collision_pairs must be a list of body pairs")
    seen_pairs = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(name, str) or not (name == "base" or name.startswith("joint:") and len(name) > 6)
                       for name in pair) or pair[0] == pair[1]):
            raise ValueError("Each collision pair needs two distinct base or joint:<name> bodies")
        key = frozenset(pair)
        if key in seen_pairs:
            raise ValueError("Duplicate PhysX collision pair")
        seen_pairs.add(key)
    gravity = values["environment"]["gravity"]
    if not isinstance(gravity, (list, tuple)) or len(gravity) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in gravity):
        raise ValueError("Environment gravity must contain three finite components")
    drag = values["environment"]["air_drag_rate_per_second"]
    if isinstance(drag, bool) or not isinstance(drag, (int, float)) or not math.isfinite(drag) or drag < 0:
        raise ValueError("Air drag rate must be finite and nonnegative")
    for section in ("rigid", "material", "joint_drive", "surface", "solver"):
        for key, value in values[section].items():
            if value is None and key == "mass_kg":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative physical parameter: {section}.{key}")
    for section, key in (("rigid", "density_kg_m3"), ("surface", "density"),
                         ("surface", "particle_radius"), ("solver", "substeps"),
                         ("solver", "iterations")):
        if values[section][key] <= 0:
            raise ValueError(f"Physical parameter must be positive: {section}.{key}")
    if values["rigid"]["mass_kg"] is not None and values["rigid"]["mass_kg"] <= 0:
        raise ValueError("Mass must be positive")
    if type(values["rigid"]["enable_ccd"]) is not bool:
        raise ValueError("Rigid enable_ccd must be a boolean")
    if values["material"]["restitution"] > 1:
        raise ValueError("Restitution must lie between zero and one")
    for key in ("substeps", "iterations", "self_contact_topology_distance", "self_contact_vertex_capacity", "self_contact_edge_capacity"):
        if type(values["solver"][key]) is not int:
            raise ValueError(f"Solver {key} must be an integer")
    if type(values["solver"]["self_contact"]) is not bool or type(values["constraints"]["use_source_pins"]) is not bool:
        raise ValueError("Self-contact and source-pin switches must be booleans")
    if values["constraints"]["lift_selection"] not in {"all", "positive_x", "negative_x", "positive_y", "negative_y", "positive_z", "negative_z"}:
        raise ValueError("Unknown source-pin lift selection")
    for key in ("lift_distance_m", "lift_duration_seconds", "lift_delay_seconds", "quiet_start_seconds", "quiet_speed_tolerance_m_s"):
        value = values["constraints"][key]
        if value is None and key in {"lift_duration_seconds", "quiet_start_seconds", "quiet_speed_tolerance_m_s"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (key != "lift_distance_m" and value < 0):
            raise ValueError(f"Invalid surface operation setting: {key}")
        if key in {"lift_duration_seconds", "quiet_speed_tolerance_m_s"} and value == 0:
            raise ValueError(f"Surface operation setting must be positive: {key}")
    for key in ("hybrid_full_surface_contact", "hybrid_rigid_soft_dat"):
        if type(values["solver"][key]) is not bool:
            raise ValueError(f"Solver {key} must be a boolean")
    if create and not path.is_file():
        atomic_json(path, values)
    return values
