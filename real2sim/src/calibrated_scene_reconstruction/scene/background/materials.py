"""S4c twin defaults: PBR materials for never-observed surfaces and the reviewed surfaces the robot works at
(recon.surfaces), movable furniture, retrieval swaps, lights.

    python -m calibrated_scene_reconstruction.recon.materials sessions/<name>

Every wall / ceiling / floor gets a clean PBR material: the agent's look from the surface review (recon.surfaces),
else one colour-matched to what the photos observed of it (a wall never observed: the paint of the best-seen wall).
A photo texture stays only where the agent kept it on a clean surface ({"photo": true}: the photo where observed, the
PBR fill elsewhere, feathered at the seam); the room completion is never a texture. Movable furniture comes from
recon.furniture (shared reconstruction) or recon.retrieve (library model); objects that fell back to retrieval
become static swaps; the room's objects reconstructed from the completion (recon.room_objects, scene graph
"synthetic_furniture") are passed on as they are.
Writes scene/twin.json (the base variant every scene builds on).
"""
import argparse

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR
from calibrated_scene_reconstruction.scene.graph import walls
from calibrated_scene_reconstruction.scene.session import Session

WALL_CANDS = ["painted_plaster_wall", "beige_wall_001", "beige_wall_002", "clay_plaster", "plastered_wall",
              "proc_painted_drywall"]  # clean paint: the stained scans (grey_plaster, concrete_wall_003, white_plaster_02) read
# as a derelict building once a whole wall is filled with them
FLOOR_CANDS = ["laminate_floor_02", "wood_floor", "brown_planks_05", "dark_wooden_planks", "diagonal_parquet",
               "concrete_floor_02", "brushed_concrete", "grey_cartago_01",
               "proc_carpet_tile_grey", "proc_carpet_tile_bluegrey", "proc_carpet_tile_charcoal", "proc_carpet_tile_beige",
               "proc_vinyl_grey", "proc_vinyl_blue", "proc_epoxy_grey"]  # proc_*: assets.proc_textures  # no patterned tiles: a fill must not invent a pattern
MIN_OBSERVED = 0.03  # a surface seen less than this gets the PBR material outright


def lin(c):
    return (np.asarray(c, float) / 255.0) ** 2.2


def observed_mean(s, part, tex="albedo"):
    t = np.asarray(Image.open(s / "scene" / "parts" / f"{part}_{tex}.png").convert("RGB"))
    o = np.asarray(Image.open(s / "scene" / "parts" / f"{part}_observed.png").convert("L")) > 127
    return t[o].reshape(-1, 3).mean(0) if o.sum() >= 50 else None


def tex_mean(name):
    return np.asarray(Image.open(POLYHAVEN_DIR / "textures" / name / "diff.jpg").convert("RGB").resize((256, 256))).reshape(-1, 3).mean(0)


def local_std(g):
    """Texture contrast without the lighting: the std of the image minus its blur (grain, planks, stains)."""
    import cv2

    g = np.asarray(g, np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 6)).std())


def tex_std(name):
    return local_std(np.asarray(Image.open(POLYHAVEN_DIR / "textures" / name / "diff.jpg").convert("L").resize((512, 512))))


def match(cands, target, target_std=None):
    """Closest chromaticity (plus brightness, plus texture contrast when the observed patch's is given: plain
    concrete is not matched to wood grain) among candidates, and the tint that maps it onto the target."""
    t = lin(target)
    cands = [n for n in cands if (POLYHAVEN_DIR / "textures" / n / "diff.jpg").exists()]

    def cost(n):
        m = lin(tex_mean(n))
        c = np.linalg.norm(m / m.sum() - t / t.sum()) + 0.3 * abs(m.mean() - t.mean())
        return c + (0.01 * abs(tex_std(n) - target_std) if target_std is not None else 0.0)
    best = min(cands, key=cost)
    return best, [float(v) for v in np.clip(t / np.maximum(lin(tex_mean(best)), 1e-4), 0.2, 3.0)]


def observed_std(s, part, tex="albedo"):
    import cv2

    g = np.asarray(Image.open(s / "scene" / "parts" / f"{part}_{tex}.png").convert("L"), np.float32)
    o = np.asarray(Image.open(s / "scene" / "parts" / f"{part}_observed.png").convert("L")) > 127
    if o.sum() < 500:
        return None
    hp = g - cv2.GaussianBlur(g, (0, 0), 6)
    o = cv2.erode(o.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)  # not the mask's own edges
    return float(hp[o].std()) if o.sum() >= 200 else None


def composite(s, part, pbr, tint, tile_m, tex="albedo", feather_m=0.08):
    """The part's photo texture where observed, the tinted PBR diffuse (tiled at tile_m) elsewhere, blended over
    `feather_m`. Writes parts/<part>_fill.png and returns its path relative to scene/."""
    import cv2

    d = np.load(s / "scene" / s.manifest["parts"][part]["npz"])
    V = d["V"]
    su, sv = float(np.linalg.norm(V[1] - V[0])), float(np.linalg.norm(V[3] - V[0]))
    photo = np.asarray(Image.open(s / "scene" / "parts" / f"{part}_{tex}.png").convert("RGB"), np.float32)
    mpath = s / "scene" / "parts" / f"{part}_observed.png"  # the photos' own texels: a completion is never a texture
    obs = np.asarray(Image.open(mpath).convert("L"), np.float32) / 255.0
    h, w = photo.shape[:2]
    diff = Image.open(POLYHAVEN_DIR / "textures" / pbr / "diff.jpg").convert("RGB")
    tw, th = max(int(round(w * tile_m / su)), 8), max(int(round(h * tile_m / sv)), 8)
    tile = np.asarray(diff.resize((tw, th), Image.LANCZOS), np.float32)
    pbr_img = np.tile(tile, (h // th + 1, w // tw + 1, 1))[:h, :w]
    pbr_img = (((pbr_img / 255.0) ** 2.2 * np.asarray(tint)) ** (1 / 2.2) * 255.0).clip(0, 255)
    k = max(int(feather_m / max(su / w, 1e-6)) | 1, 3)
    a = cv2.GaussianBlur(cv2.erode(obs, np.ones((k // 2 + 1, k // 2 + 1), np.uint8)), (k, k), 0)[..., None]
    out = (a * photo + (1 - a) * pbr_img).clip(0, 255).astype(np.uint8)
    rel = f"parts/{part}_fill.png"
    Image.fromarray(out).save(s / "scene" / rel)
    return rel


def furniture(s: Session):
    """stool_models of the twin: retrieval result > shared reconstruction > fixed library model at measured spots."""
    out = []
    for group, spec in s.scene_graph.get("movable_furniture", {}).items():
        d = s / "objects" / group
        if (d / "retrieved.json").exists():
            r = s.read_json(f"objects/{group}/retrieved.json")
            out += [{"model": r["model"], "center": m["center"], "height": round(r["height"], 3),
                     "yaw": r["yaw"] + m["yaw"], "group": group} for m in r["members"] if not m.get("drop")]
        elif (d / "furniture.json").exists() and s.read_json(f"objects/{group}/furniture.json").get("ok"):
            out += [{"recon": group, "center": m["center"], "yaw": m["yaw"], "group": group}
                    for m in s.read_json(f"objects/{group}/furniture.json")["members"] if not m.get("drop")]
        elif spec.get("retrieve"):
            out += [{"model": spec["retrieve"], "center": st["center"], "height": round(st["seat_top"], 3), "yaw": 0.0,
                     "group": group} for st in s.manifest["stools"] if st["id"] in spec["ids"]]
    return out


def static_swaps(s: Session):
    swaps = {}
    for name in s.scene_graph.get("static_objects", {}):
        if (s / "objects" / name / "retrieved.json").exists():
            r = s.read_json(f"objects/{name}/retrieved.json")
            swaps[name] = {"model": r["model"], "height": round(r["height"], 3), "yaw": r["yaw"], "xy": r["xy"],
                           "support_z": r["support_z"], "retrieved": True}
    return swaps


def synthetic(s: Session):
    """The room objects reconstructed from the completion (recon.room_objects), as the scene graph keeps them."""
    return [{"name": k, **{f: v[f] for f in ("source", "box", "center", "height", "yaw")}, "synthetic": True}
            for k, v in s.scene_graph.get("synthetic_furniture", {}).items()]


def surface_overrides(s: Session, auto=False):
    """room_overrides of the surfaces recon.surfaces reviewed (qa/surfaces.json): the agent's `look` on a structure
    plane ({"pbr", "tint": "match" | [r, g, b], "tile_m"}, {"color", "roughness"} or {"photo": true}) first; then,
    once the agent has reviewed the proposals (scene graph `_surface_review`, or `auto`), each remaining group's
    proposal. Furniture groups with a `look` of the agent's are left to sim.setpieces (a twin override would hide it).
    Returns (overrides, keep_photo: structure parts the agent kept on their photo texture)."""
    import json

    from calibrated_scene_reconstruction.assets.proc_textures import SIZE as PROC_SIZE
    from calibrated_scene_reconstruction.assets.receptacles import TEX_SIZE

    sg = s.scene_graph
    path = s / "qa" / "surfaces.json"
    report = json.loads(path.read_text())["groups"] if path.exists() else {}
    reviewed = auto or "_surface_review" in sg
    out, keep = {}, []

    def as_override(look, color_srgb):
        if look.get("photo"):
            return None
        if "color" in look:
            return {"color": look["color"], "roughness": look.get("roughness", 0.5)}
        tint = look.get("tint", "match")
        if "color_linear" in look:  # the agent's colour (linear RGB) on the PBR's pattern: the tint that makes its mean so
            tint = [float(v) for v in np.clip(np.asarray(look["color_linear"], float)
                                              / np.maximum(lin(tex_mean(look["pbr"])), 1e-4), 0.02, 3.0)]
        elif tint == "match":
            tint = ([float(v) for v in np.clip(lin(color_srgb) / np.maximum(lin(tex_mean(look["pbr"])), 1e-4), 0.02, 3.0)]
                    if color_srgb is not None else [1.0, 1.0, 1.0])
        tile = look.get("tile_m", TEX_SIZE.get(look["pbr"], PROC_SIZE.get(look["pbr"], 1.2)))
        return {"pbr": look["pbr"], "tile_m": tile, "tint": tint}

    planes = list(sg["structure"]) + [k for k in report if ":" not in k and k not in sg["structure"]]  # + closures
    for name in planes:
        st = sg["structure"].get(name, {})  # a closure piece (scenegraph.close_room) takes its proposal
        g = report.get(name, {})
        look = st.get("look") or (g.get("proposal") if reviewed else None)
        if not look:
            continue
        ov = as_override(look, g.get("color_srgb"))
        if ov is None:
            keep.append(name)
        else:
            out[name] = ov
    if reviewed:
        for gname, g in report.items():
            if ":" not in gname or not g.get("proposal"):
                continue
            fname, group = gname.split(":", 1)
            looks = (sg["furniture"].get(fname) or {}).get("look") or {}
            if group in looks or "all" in looks:
                continue
            ov = as_override(g["proposal"], g.get("color_srgb"))
            for p in g["parts"]:
                if ov is not None:
                    out[p] = ov
    return out, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--auto-surfaces", action="store_true",
                    help="apply recon.surfaces' proposals without the agent's review (run_session.sh --auto)")
    args = ap.parse_args()
    s = Session(args.session)
    m = s.manifest
    tex = "albedo" if "albedo" in m["textures"] else "baked"
    seen = walls(s.scene_graph, m, observed=True) or walls(s.scene_graph)
    best_wall = max(seen, key=lambda w: m["parts"].get(w, {}).get("observed_fraction", 0.0))
    wall = observed_mean(s, best_wall, tex)
    floor = observed_mean(s, "floor", tex)
    wall_tex, wall_tint = match(WALL_CANDS, wall, observed_std(s, best_wall, tex))
    floor_tex, floor_tint = match(FLOOR_CANDS, floor if floor is not None else wall * 0.5,
                                  observed_std(s, "floor", tex) if floor is not None else None)
    ceil_tex, ceil_tint = match(WALL_CANDS, np.minimum(wall * 1.08, 255),  # ceilings: wall paint, a touch lighter
                                observed_std(s, best_wall, tex))
    unseen = walls(s.scene_graph, m, observed=False)
    overrides = {k: {"pbr": wall_tex, "tile_m": 2.0, "tint": wall_tint} for k in unseen}
    overrides["ceiling"] = {"pbr": ceil_tex, "tile_m": 2.0, "tint": ceil_tint}
    overrides["floor"] = {"pbr": floor_tex, "tile_m": 1.6, "tint": floor_tint}
    # every structure plane gets a clean material: a wall the photos saw takes the PBR of its own observed colour
    # (what the photos show on or beyond it -- a desk, a door, a corridor -- is no wall: it is reconstructed in 3D,
    # recon.room_objects), one they did not see the wall paint of the best-seen wall; the room completion is never a
    # texture (a piece of furniture painted on a wall is flat from every other viewpoint)
    for k in seen:
        col = observed_mean(s, k, tex)
        if col is not None:
            pbr, tint = match(WALL_CANDS, col, observed_std(s, k, tex))
            overrides[k] = {"pbr": pbr, "tile_m": 2.0, "tint": tint}
    surf, keep = surface_overrides(s, args.auto_surfaces)
    overrides.update(surf)
    for k in keep:  # the agent kept the photo on these (a clean surface, surface review): photo where observed
        ov = overrides.get(k) or {"pbr": wall_tex, "tile_m": 2.0, "tint": wall_tint}
        if "pbr" in ov:
            overrides[k] = {"tex": composite(s, k, ov["pbr"], ov["tint"], ov["tile_m"], tex), "pbr_fill": ov["pbr"]}
        else:
            overrides.pop(k, None)
    twin = {
        "room_overrides": overrides,
        "lights": s.config("lights.json", {"lights": {}})["lights"],
        "stool_models": furniture(s),
        "static_swaps": static_swaps(s),
        "synthetic_furniture": synthetic(s),
    }
    s.write_json("scene/twin.json", twin)
    stools = sorted({st.get("recon") and f"recon:{st['recon']}" or st["model"] for st in twin["stool_models"]})
    print(f"walls -> {wall_tex}, floor -> {floor_tex}, ceiling -> {ceil_tex}, "
          f"furniture -> {stools} x{len(twin['stool_models'])}, retrieval swaps -> {sorted(twin['static_swaps'])}, "
          f"synthetic furniture x{len(twin['synthetic_furniture'])}, reviewed surfaces x{len(surf)} parts"
          + (f" (photo kept: {keep})" if keep else ""))


if __name__ == "__main__":
    main()
