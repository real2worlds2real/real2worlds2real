"""Asset labels, geometry frames and collision cache data shared by planning and simulation."""
from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.contracts.scene import SceneInputs as Session


def object_label(s: Session, spec: dict) -> str:
    """Semantic label of an object spec (a twin instance, its augmented version, a scan or a library model standing
    in for it): what assets.physprops classifies."""
    name = spec.get("as", spec.get("name", spec.get("ph_model", "")))
    return (spec.get("label") or (spec.get("augment") or {}).get("label")
            or s.scene_graph.get("dynamic_objects", {}).get(name, {}).get("_label")
            or spec.get("name") or spec.get("ph_model") or name)  # a distractor "prop_2": its source, "014_lemon"


def level(V):
    """A reconstructed object's world vertices set down on its own base (assets.physprops.level_rotation, about its
    bounding-box centre): the frame both the simulator and the grasp synthesis use."""
    from ..assets.physprops import level_rotation

    c = (V.min(0) + V.max(0)) / 2
    return (V - c) @ level_rotation(V).T + c


def mesh_key(spec: dict, V) -> str:
    """Synthesis cache key: the source name and a hash of the geometry (two cousins with the same cup share it)."""
    stem = Path(spec["mesh_path"]).parent.parent.name if spec.get("mesh_path") else f"twin_{spec['name']}"
    digest = hashlib.sha1(np.round(np.asarray(V, np.float64), 5).tobytes()).hexdigest()[:8]
    return f"{stem}_{digest}"


def scene_key(key: str) -> str:
    """The synthesis scene of target mesh `key` (its solid, see solid_mesh)."""
    return f"{key}__solid"


def collision_parts(s: Session, key: str, directory="grasps"):
    """The convex parts (CoACD, in the target's prim frame) of the solid the grasp synthesis used for target mesh
    `key`, or None. The simulator collides the target with exactly these, so a synthesized grasp meets the geometry
    it was planned on."""
    k = scene_key(key)
    f = s / directory / "_scenes" / k / f"{k}_coacd_parts.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return [np.asarray(d[f"part_{i}"], float) for i in range(int(d["num_parts"]))]
