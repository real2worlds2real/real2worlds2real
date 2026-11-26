"""The physical parameter vector calibrated against a real recording, and configs/<name>/physics.json.

Names are dotted paths into physics.json. Pair friction is what the optimizer sees: sim.physics gives the object a
material with friction 1 and combine mode `multiply`, so the gripper pads' and the support's own coefficients are the
pair coefficients. Dynamic friction is parametrised as a ratio of static (<= 1). Drive gains are multipliers on the
robot spec's stiffness / damping. gravity and time_scale are sanity checks, off by default: a fit far from 9.81 or 1
means a clock (video fps, controller timestamps) or a scale (reconstruction) is wrong, not new physics.
twin.depth_scale is not physics: it scales a monocular tabletop twin (calibrated_scene_reconstruction.physics.replay.tabletop.scaled) about the camera,
the one degree of freedom its first frame cannot fix; fit it with the physical parameters, or first on its own.
twin.support_dz lifts support and object together, for a depth error that is not a scale (the first frames' IoU
prices it).
"""
from calibrated_scene_reconstruction.physics.common.schema import write_document
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    log: bool = False


PARAMS = [
    Param("object.mass", 0.02, 3.0, 0.3, log=True),                     # kg
    Param("object.com_offset.x", -0.03, 0.03, 0.0),                     # m, from the mesh centre, object frame
    Param("object.com_offset.y", -0.03, 0.03, 0.0),
    Param("object.com_offset.z", -0.03, 0.03, 0.0),
    Param("object.restitution", 0.0, 0.6, 0.05),
    Param("gripper.static_friction", 0.1, 2.0, 1.0, log=True),          # object-pad pair
    Param("gripper.dynamic_ratio", 0.5, 1.0, 0.85),
    Param("support.static_friction", 0.05, 1.5, 0.5, log=True),         # object-support pair
    Param("support.dynamic_ratio", 0.5, 1.0, 0.85),
    Param("drive.stiffness_scale", 0.25, 4.0, 1.0, log=True),
    Param("drive.damping_scale", 0.25, 4.0, 1.0, log=True),
    Param("gravity", 9.0, 10.6, 9.81),
    Param("time_scale", 0.9, 1.1, 1.0),
    Param("twin.depth_scale", 0.7, 1.3, 1.0),                           # tabletop twin about the camera (nuisance)
    Param("twin.support_dz", -0.05, 0.05, 0.0),                         # m, support and object lifted (nuisance)
]
DEFAULT_ACTIVE = ("object.mass", "object.com_offset.z", "gripper.static_friction", "gripper.dynamic_ratio",
                  "support.static_friction", "support.dynamic_ratio", "object.restitution")


def require_fixed_geometry(active=(), calibrations=()):
    """A reviewed Blender asset cannot acquire new geometry during physics fitting."""
    geometry = {'twin.depth_scale': 1.0, 'twin.support_dz': 0.0}
    if set(active) & geometry.keys():
        raise ValueError('Reviewed Blender geometry is fixed; revise and review a new model round '
                         'before changing twin.depth_scale or twin.support_dz')
    for cal in calibrations:
        for name, expected in geometry.items():
            value = float(cal.get('twin', {}).get(name.split('.')[1], expected))
            if not np.isfinite(value) or value != expected:
                raise ValueError('Reviewed Blender geometry is fixed; nonidentity geometry calibration '
                                 'requires a new reviewed model round')


class Space:
    """The active subset of PARAMS as a box [0, 1]^d (log-uniform for `log` params); the rest stay at their defaults
    (or at `fixed` values)."""

    def __init__(self, active=DEFAULT_ACTIVE, fixed=None, params=PARAMS):
        by = {p.name: p for p in params}
        unknown = [n for n in list(active) + list(fixed or {}) if n not in by]
        if unknown:
            raise KeyError(f"unknown parameters {unknown}")
        self.all, self.active = params, [by[n] for n in active]
        self.fixed = {p.name: p.default for p in params}
        self.fixed.update(fixed or {})

    @property
    def dim(self):
        return len(self.active)

    def names(self):
        return [p.name for p in self.active]

    def to_values(self, u):
        """(n, d) or (d,) unit-box points -> physical values of the active params."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        lo = np.array([np.log(p.lo) if p.log else p.lo for p in self.active])
        hi = np.array([np.log(p.hi) if p.log else p.hi for p in self.active])
        v = lo + u * (hi - lo)
        lg = np.array([p.log for p in self.active], dtype=bool)
        v[..., lg] = np.exp(v[..., lg])
        return v

    def to_unit(self, values):
        v = np.asarray(values, float)
        lo = np.array([p.lo for p in self.active])
        hi = np.array([p.hi for p in self.active])
        lg = np.array([p.log for p in self.active], dtype=bool)
        v = np.clip(v, lo, hi)
        out = (v - lo) / (hi - lo)
        out[..., lg] = np.log(v[..., lg] / lo[lg]) / np.log(hi[lg] / lo[lg])
        return out

    def default_unit(self):
        return self.to_unit([self.fixed[p.name] for p in self.active])

    def flat(self, u):
        """All parameters {name: value} for one unit-box point."""
        d = dict(self.fixed)
        d.update(zip(self.names(), (float(v) for v in self.to_values(u))))
        return d


def nested(flat):
    """{"a.b": v} -> {"a": {"b": v}}; resolves dynamic ratios into dynamic_friction and com offsets into a list."""
    out = {}
    for k, v in flat.items():
        *path, leaf = k.split(".")
        node = out
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = float(v)
    for part in ("gripper", "support"):
        g = out.get(part, {})
        if "dynamic_ratio" in g:
            g["dynamic_friction"] = g["static_friction"] * g.pop("dynamic_ratio")
    obj = out.get("object", {})
    if isinstance(obj.get("com_offset"), dict):
        c = obj["com_offset"]
        obj["com_offset"] = [c.get("x", 0.0), c.get("y", 0.0), c.get("z", 0.0)]
    return out


def flatten(phys, params=PARAMS):
    """physics.json (nested) -> {name: value} of the PARAMS it sets (the inverse of nested; provenance dropped)."""
    flat = {}
    for p in params:
        *path, leaf = p.name.split(".")
        node = phys
        for k in path:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        if leaf == "dynamic_ratio" and "dynamic_friction" in node and node.get("static_friction"):
            flat[p.name] = node["dynamic_friction"] / node["static_friction"]
        elif path[-1:] == ["com_offset"]:  # a list in physics.json, below
            pass
        elif isinstance(node, dict) and leaf in node:
            flat[p.name] = float(node[leaf])
    com = phys.get("object", {}).get("com_offset")
    if isinstance(com, (list, tuple)):
        flat.update({f"object.com_offset.{a}": float(v) for a, v in zip("xyz", com)})
    return flat


def write_physics(path, flat, **info):
    """configs/<name>/physics.json: the nested parameters plus provenance (`info`: source recording, loss, ...)."""
    d = nested(flat)
    d["calibration"] = info
    path.parent.mkdir(parents=True, exist_ok=True)
    write_document(path, d, indent=1)
    return d
