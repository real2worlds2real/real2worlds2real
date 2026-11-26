"""Resolve observed set pieces and furniture looks before USD authoring."""
import json

import numpy as np

from calibrated_scene_reconstruction.assets.support_geometry import support_top
from calibrated_scene_reconstruction.scene.graph import face_group


def face_camera_yaw(s, xy):
    """Yaw that turns a kit's front (-y) toward the photo camera."""
    cam = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float)[:2, 3]
    dx, dy = cam - np.asarray(xy, float)
    return float(np.arctan2(dx, -dy))


def kit_for_object(s, boxes, name, fix, V):
    """(kind, params, origin, yaw) of the set piece that replaces a reconstructed static object (world points V):
    from its photo measurement (recon.setpieces: objects/<name>/kit.json) when there is one, else from the
    reconstruction's footprint facing the camera; the fix's own params win."""
    kind = fix["kit"]
    params = {k: v for k, v in fix.items() if k not in ("kit", "center", "yaw_deg", "seed", "prompt")}
    lo, hi = V.min(0), V.max(0)
    z = support_top(boxes, lo[:2], hi[:2], float(lo[2]))
    xy, yaw = ((lo + hi) / 2)[:2], None
    f = s / "objects" / name / "kit.json"
    m = json.loads(f.read_text()) if f.exists() else {}
    if m.get("kit") == kind:
        z = m["support_z"]
        if "front" in m:  # a row or an inlay: its front line, the piece behind it
            yaw = m["yaw"]
            depth = params.get("depth", m.get("depth", 0.16))
            xy = np.asarray(m["front"]) - (depth / 2 + (0.01 if kind == "book_row" else 0.0)) * np.array(
                [np.sin(yaw), -np.cos(yaw)])
            params = {**{k: m[k] for k in ("length", "height", "colours", "depth") if k in m}, **params}
        elif "center" in m:
            xy = m["center"]
            params = {"radius": m["radius"], **params}
    elif kind == "tray_set":
        params.setdefault("radius", float(0.5 * max(hi[:2] - lo[:2])))
    yaw = face_camera_yaw(s, xy) if yaw is None else yaw
    if "center" in fix:
        xy = fix["center"]
    if "yaw_deg" in fix:
        yaw = np.radians(fix["yaw_deg"])
    return kind, params, [float(xy[0]), float(xy[1]), float(z)], float(yaw)


def reference_albedo(s, name, observed_mean, ref="sides", seen=0.05):
    """The photo's albedo of a piece of furniture for a look's tint, from its best-seen face of group `ref`, else of
    any other group. A table's sides by default: its top faces the windows and the lamps, and what stood on it is
    smeared over its projection (glare reads as grey-white wood); a counter's from its top (ref "top": its sides are
    cabinet doors of another material)."""
    parts = {p: v for p, v in s.manifest["parts"].items() if p.startswith(f"{name}_")}
    for faces in ([p for p in parts if face_group(p, name) == ref], [p for p in parts if face_group(p, name) != ref]):
        best = max(faces, key=lambda p: parts[p].get("observed_fraction", 0), default=None)
        if best is not None and parts[best].get("observed_fraction", 0) > seen:
            return observed_mean(s, best)
    return None


def furniture_overrides(s):
    """room_overrides from the scene graph's furniture `look`s, {group: look} for the groups top / sides / legs (or
    "all"): a Poly Haven texture {"pbr", "tile_m" (default its real size), "tint": "match" (to the photo's albedo of
    the furniture, reference_albedo, from its faces of group "ref") | [r, g, b]} or a plain finish {"color",
    "roughness"}."""
    from calibrated_scene_reconstruction.assets.receptacles import TEX_SIZE
    from calibrated_scene_reconstruction.scene.background.materials import lin, observed_mean, tex_mean

    out = {}
    parts = s.manifest["parts"]
    for name, f in s.scene_graph.get("furniture", {}).items():
        looks = f.get("look") or {}
        for group, look in looks.items():
            mine = [p for p in parts if p.startswith(f"{name}_") and group in ("all", face_group(p, name))]
            if not mine:
                continue
            if "color" in look:
                for part in mine:
                    out[part] = {"color": look["color"], "roughness": look.get("roughness", 0.5)}
                continue
            tint = look.get("tint", "match")
            if tint == "match":
                ref = reference_albedo(s, name, observed_mean, look.get("ref", "sides"))
                tint = (np.clip(lin(ref) / np.maximum(lin(tex_mean(look["pbr"])), 1e-3), 0.3, 2.0).round(3).tolist()
                        if ref is not None else [1, 1, 1])
            for part in mine:
                out[part] = {"pbr": look["pbr"], "tile_m": look.get("tile_m", TEX_SIZE.get(look["pbr"], 1.0)),
                             "tint": tint}
    return out


def furniture_kits(s):
    """{furniture name: (kind, params, origin, yaw)} for furniture the agent rebuilt as a set piece
    (furniture.<name>.fix {"kit": ...}): on the floor at its box's footprint, `height` its box top."""
    out = {}
    for name, f in s.scene_graph.get("furniture", {}).items():
        fix = f.get("fix") or {}
        if not fix.get("kit"):
            continue
        lo, hi = (np.asarray(v, float) for v in f["box"])
        params = {k: v for k, v in fix.items() if k not in ("kit", "center", "yaw_deg", "seed")}
        params.setdefault("height", float(hi[2]))
        params.setdefault("diameter", float(np.mean(hi[:2] - lo[:2])))
        params.setdefault("length", float(hi[0] - lo[0]))
        params.setdefault("depth", float(hi[1] - lo[1]))
        xy = fix.get("center", ((lo + hi) / 2)[:2])
        out[name] = (fix["kit"], params, [float(xy[0]), float(xy[1]), 0.0], float(np.radians(fix.get("yaw_deg", 0.0))))
    return out
