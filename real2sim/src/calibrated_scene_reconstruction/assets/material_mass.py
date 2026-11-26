"""Material estimates shared by Blender export and Isaac mass authoring."""
import math

import numpy as np
import trimesh

DENSITY = {  # kg / m^3 of a task object's material (the agent names it: dynamic_objects.<obj>.material)
    "pp": 900, "polypropylene": 900, "pe": 950, "abs": 1050, "pla": 1240, "petg": 1270, "acrylic": 1180,
    "plastic": 1000, "nylon": 1140, "rubber": 1100, "silicone": 1100, "glass": 2500, "ceramic": 2300,
    "porcelain": 2400, "steel": 7850, "stainless": 8000, "aluminium": 2700, "aluminum": 2700, "brass": 8500,
    "wood": 600, "paper": 700, "cardboard": 300, "foam": 50}


def mass(volume, spec):
    if spec.get("mass") is not None:
        if not spec.get("_mass_why"):
            raise ValueError("Explicit mass requires _mass_why")
        value = float(spec["mass"])
        why = spec["_mass_why"]
    else:
        rho = spec.get("density", DENSITY.get(str(spec.get("material", "")).lower()))
        if rho is None or not math.isfinite(float(rho)) or float(rho) <= 0:
            raise ValueError("Provide a known material or a finite positive density")
        value = float(volume) * float(rho)
        why = f"solid volume {volume:.8g} m^3 times density {rho} kg/m^3"
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Mass must be finite and positive")
    return value, why


def link_mass(volume, assembly, link, total_volume):
    """Explicit per-link mass/density takes precedence over assembly defaults.

    The legacy whole-assembly mass still distributes by solid volume when a
    link has no independent physical declaration.
    """
    if link.get('mass') is not None or link.get('density') is not None:
        selected = {key: link[key] for key in ('mass', 'density', '_mass_why') if key in link}
        kg, why = mass(volume, selected)
        return kg, 'independent physical link: '+why
    kg, why = mass(volume, assembly)
    if assembly.get('mass') is not None:
        if not math.isfinite(total_volume) or total_volume <= 0:
            raise ValueError('Positive total rigid volume required for assembly mass distribution')
        kg *= volume/total_volume
        why += '; assembly mass distributed by solid link volume'
    return kg, why


def kit_mass(kit, graph_obj, default):
    """(mass kg, how) of a task object built from a kit: the agent's estimate (dynamic_objects.<obj>.mass with its
    `_mass_why`), else the kit's solid volume times its material's density (`density` kg/m^3, or `material`
    in DENSITY), else `default`."""
    if graph_obj.get("_mass_why") and graph_obj.get("mass"):
        return float(graph_obj["mass"]), f"the agent's estimate: {graph_obj['_mass_why']}"
    rho = graph_obj.get("density") or DENSITY.get(str(graph_obj.get("material", "")).lower())
    if rho:
        vol = sum(abs(trimesh.Trimesh(np.asarray(p["V"], float), np.asarray(p["F"], int), process=True).volume)
                  for p in kit["parts"])
        return float(rho * vol), f"{1e6 * vol:.1f} cm^3 of {graph_obj.get('material') or 'density'} at {rho} kg/m^3"
    return float(default), "default"
