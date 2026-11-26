"""Cousin object augmentation: the reconstructed object itself, varied, instead of a library swap.

    augment(s, name, rng, out) -> an object spec for cousins (mesh_path + texture, like a YCB scan)

shape     footprint and height scaled separately, a slight footprint aspect change, a taper (narrower or wider
          towards the top) and a belly, all about the vertical axis through the base: the bottom stays flat (it still
          stands) and the object stays recognisably itself;
colour    by semantic class (scenegraph label): a glazed or painted thing (cup, mug, bowl, vase, bottle, box ...) may
          take any hue and a glaze tint; produce, bread, wood and wicker only shift a little;
finish    roughness from glossy glaze to matte, per class;
mass      the twin's mass times the volume change (density kept).
Writes <out>/{mesh.obj, albedo.png, augment.json}.
"""
import json
import re

import cv2
import numpy as np
import trimesh
from PIL import Image

from calibrated_scene_reconstruction.core.paths import rel

RECOLOR = r"cup|mug|bowl|plate|vase|jar|bottle|can|box|pot|pitcher|kettle|tumbler|glass"
NATURAL = r"apple|orange|lemon|banana|pear|bread|fruit|wood|basket|wicker|plant|board"


def _world_mesh(s, name):
    m = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"])
    return m, np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]


def deform(V, rng, strength=1.0):
    """Shape variation about the vertical axis through the base (the lowest 30 % of the points: a mug's body, not
    its handle). Returns (vertices, parameters, volume factor)."""
    lo, hi = V.min(0), V.max(0)
    h = max(hi[2] - lo[2], 1e-6)
    t = (V[:, 2] - lo[2]) / h  # 0 at the bottom, 1 at the top
    c = np.median(V[t < 0.3, :2], axis=0) if (t < 0.3).sum() >= 10 else (lo[:2] + hi[:2]) / 2
    p = {"scale_xy": rng.uniform(0.85, 1.15), "aspect": rng.uniform(0.93, 1.07), "scale_z": rng.uniform(0.85, 1.2),
         "taper": rng.uniform(-0.15, 0.15), "belly": rng.uniform(-0.05, 0.1)}
    p = {k: float(1 + (v - 1) * strength) if k.startswith(("scale", "aspect")) else float(v * strength)
         for k, v in p.items()}
    r = 1 + p["taper"] * (t - 0.5) + p["belly"] * np.sin(np.pi * t)
    W = V.copy()
    W[:, 0] = c[0] + (V[:, 0] - c[0]) * p["scale_xy"] * p["aspect"] * r
    W[:, 1] = c[1] + (V[:, 1] - c[1]) * p["scale_xy"] / p["aspect"] * r
    W[:, 2] = lo[2] + (V[:, 2] - lo[2]) * p["scale_z"]
    vol = p["scale_xy"] ** 2 * p["scale_z"] * float(np.mean(r ** 2))
    return W, p, vol


def recolor(img, label, rng):
    """HSV variation of an albedo texture by class; returns (image, parameters, roughness)."""
    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV).astype(np.float32)
    if re.search(RECOLOR, label):
        p = {"hue": float(rng.uniform(0, 180)), "sat": float(rng.uniform(0.6, 1.5)), "val": float(rng.uniform(0.75, 1.15))}
        rough = float(rng.uniform(0.15, 0.6))
    elif re.search(NATURAL, label):
        p = {"hue": float(rng.uniform(-5, 5)), "sat": float(rng.uniform(0.85, 1.15)), "val": float(rng.uniform(0.85, 1.1))}
        rough = float(rng.uniform(0.45, 0.75))
    else:
        p = {"hue": float(rng.uniform(-15, 15)), "sat": float(rng.uniform(0.8, 1.2)), "val": float(rng.uniform(0.85, 1.1))}
        rough = float(rng.uniform(0.3, 0.7))
    hsv[..., 0] = (hsv[..., 0] + p["hue"]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * p["sat"], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * p["val"], 0, 255)
    out = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    tint = None
    if re.search(RECOLOR, label) and rng.random() < 0.5:  # a white or grey glaze takes no hue: a coloured glaze
        g = rng.uniform(0.55, 1.0, 3)
        tint = [float(v) for v in g / g.max()]
    return out, {**p, "tint": tint}, rough


def augment(s, name, rng, out, label=None, mass=0.3, strength=1.0):
    """The twin object `name` varied in shape, colour and finish; returns a cousin object spec."""
    out.mkdir(parents=True, exist_ok=True)
    label = label or s.scene_graph["dynamic_objects"].get(name, {}).get("_label", name)
    m, V = _world_mesh(s, name)
    W, shape, vol = deform(V, rng, strength)
    img, colour, rough = recolor(Image.open(s / "objects" / name / "albedo.png"), label, rng)
    img.save(out / "albedo.png")
    uv = getattr(m.visual, "uv", None)
    mesh = trimesh.Trimesh(W, m.faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, image=img) if uv is not None else None)
    mesh.export(out / "mesh.obj")
    meta = {"from": name, "label": label, "shape": shape, "colour": colour, "roughness": rough, "volume": vol}
    (out / "augment.json").write_text(json.dumps(meta, indent=1))
    spec = {"name": f"{name}_aug", "mesh_path": rel(out / "mesh.obj"), "texture": rel(out / "albedo.png"),
            "roughness": rough, "mass": float(max(0.05, mass * vol)), "augment": meta}
    if colour["tint"]:
        spec["tint"] = colour["tint"]
    return spec
