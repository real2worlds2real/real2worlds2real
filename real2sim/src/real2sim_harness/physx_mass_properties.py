"""Validate and author coupled rigid mass properties for the PhysX asset adapter."""
from __future__ import annotations

import numpy as np


def validate_mass_properties(record):
    """Inertia is about the specified COM, expressed in unscaled body axes."""
    fields = {"mass_kg", "center_of_mass_m", "inertia_kg_m2", "source"}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Mass properties require mass_kg, center_of_mass_m, inertia_kg_m2 and source together")
    mass = record["mass_kg"]
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Explicit mass must be finite and positive")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError("Mass properties need measurement or assumption provenance in source")
    try:
        com_raw = np.asarray(record["center_of_mass_m"])
        inertia_raw = np.asarray(record["inertia_kg_m2"])
        if com_raw.dtype.kind not in "fiu" or inertia_raw.dtype.kind not in "fiu":
            raise ValueError("Mass properties must be numerical arrays")
        com, inertia = com_raw.astype(float), inertia_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid mass-property arrays") from exc
    if (com.shape != (3,) or inertia.shape != (3, 3)
            or not np.isfinite(com).all() or not np.isfinite(inertia).all()):
        raise ValueError("COM must be finite XYZ; inertia must be a finite 3x3 tensor")
    scale = max(float(np.abs(inertia).max()), np.finfo(float).tiny)
    if not np.allclose(inertia, inertia.T, rtol=0, atol=scale * 1e-7):
        raise ValueError("Inertia tensor must be symmetric")
    inertia = (inertia + inertia.T) / 2
    moments = np.linalg.eigvalsh(inertia)
    if moments[0] <= 0 or moments[-1] > moments[:-1].sum() + scale * 1e-7:
        raise ValueError("Inertia needs positive principal moments satisfying the triangle inequality")
    return {"mass_kg": float(mass), "center_of_mass_m": com.tolist(),
            "inertia_kg_m2": inertia.tolist(), "source": record["source"]}


def author_mass_properties(prim, record):
    """Author all coupled values before simulation; never patch one inertia scalar live."""
    from pxr import Gf, Sdf, UsdPhysics
    from scipy.spatial.transform import Rotation

    values = validate_mass_properties(record)
    moments, axes = np.linalg.eigh(np.asarray(values["inertia_kg_m2"]))
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    q = Rotation.from_matrix(axes).as_quat()
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(values["mass_kg"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*values["center_of_mass_m"]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    prim.CreateAttribute("remo:massPropertiesSource", Sdf.ValueTypeNames.String).Set(values["source"])
