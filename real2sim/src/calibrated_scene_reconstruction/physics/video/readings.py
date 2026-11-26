"""Read a recording's physics.json as parameter values with standard errors,
bounds, or an explanation for a missing measurement. The metadata summary
uses the same representation across motion types.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

# sources of a parameter in physics.json that are no reading of the video
NOT_READ = ("prior", "default", "not taken", "assumed")


def readings(path):
    """Every reading of a recording in one form: (motion, [{"parameter", "kind": "value" | "lower_bound" |
    "upper_bound", "value", "stderr" (None where none is stated)}], why nothing was read). None when the recording
    was not run."""
    path = Path(path)
    if not (path / "physics.json").exists():
        return None
    p = read_document(path / "physics.json")
    out = []
    for name, m in p.get("parameters", {}).items():
        src = str(m.get("source", ""))
        if src.startswith(NOT_READ) or name == "object.mass":
            continue
        se = m.get("stderr", m.get("std"))
        read = src.startswith("measured")  # ("measured (motion not seen to its end)" is a reading too)
        if m.get("lower_bound") is not None and (m.get("value") is None or not read):
            out.append({"parameter": name, "kind": "lower_bound", "value": m["lower_bound"], "stderr": None})
        elif m.get("upper_bound") is not None and not read:
            out.append({"parameter": name, "kind": "upper_bound", "value": m["upper_bound"], "stderr": None})
        elif m.get("value") is not None and read:
            out.append({"parameter": name, "kind": "value", "value": m["value"], "stderr": se})
    fp = p.get("free_phase") or {}
    if fp.get("tilt_deg") is not None:  # geometry and scale a free phase gives beside its parameters
        out.append({"parameter": "support.tilt_deg", "kind": "value", "value": fp["tilt_deg"],
                    "stderr": fp.get("tilt_std_deg")})
    if fp.get("radius") is not None and "start" in fp:
        out.append({"parameter": "object.radius", "kind": "value", "value": fp["radius"], "stderr": None})
        b = (p.get("bounces") or [{}])[0]
        # the fall to the first bounce: how far a body let go from rest fell
        if fp.get("fall") and b.get("speed_in") and b.get("flight_before") is None:
            out.append({"parameter": "drop.height", "kind": "value", "stderr": None,
                        "value": b["speed_in"] ** 2 / (2 * fp["gravity_apparent"])})
    if out and (path / "twin.json").exists():  # the scene's scale, as the camera's height over the support
        tw = read_document(path / "twin.json")
        if tw.get("camera_height"):
            rel = float(tw.get("scale", {}).get("rel_std", 0.0)) or None
            out.append({"parameter": "scene.camera_height", "kind": "value", "value": tw["camera_height"],
                        "stderr": rel and rel * tw["camera_height"]})
    if p.get("measured"):  # the bodies that deform keep what they read under their own name
        s = p.get("strain", {})
        axial = "across_peak" if str(s.get("load_axis", "")).startswith("across") else "along_peak"
        for key, name, std in (("poisson", "strain.poisson", "poisson_std"), (axial, "strain.axial", None),
                               ("twist_end_deg", "strain.twist_deg", "twist_std_deg"),
                               ("bow_over_width", "strain.bow_over_width", None)):
            if s.get(key) is not None and (key != axial or s.get("kind") in ("stretch", "compress")):
                out.append({"parameter": name, "kind": "value", "stderr": s.get(std) if std else None,
                            "value": abs(s[key]) if key != axial and key != "poisson" else s[key]})
        s = p.get("sheet", {})
        if s.get("areal_density"):
            lo, hi = s["areal_density_range"]
            out.append({"parameter": "sheet.areal_density", "kind": "value", "value": s["areal_density"],
                        "stderr": 0.5 * (hi - lo), "range": [lo, hi]})
        elif s.get("areal_density_at_least"):
            out.append({"parameter": "sheet.areal_density", "kind": "lower_bound", "stderr": None,
                        "value": s["areal_density_at_least"]})
        b = p.get("beam", {})
        if b.get("period"):
            out.append({"parameter": "beam.period", "kind": "value", "value": b["period"], "stderr": None})
        if b.get("specific_modulus"):
            out.append({"parameter": "beam.specific_modulus", "kind": "value", "value": b["specific_modulus"],
                        "stderr": None})
    why = None if out else str(p.get("reason") or p.get("note") or "nothing was measured")
    return p.get("motion", read_document(path / "meta.json").get("motion", "slide")), out, why
