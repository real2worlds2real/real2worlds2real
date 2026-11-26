"""Photoreal digital cousins: every swap uses scanned assets or PBR materials.

    python -m calibrated_scene_reconstruction.cousins sessions/<name> [--n 6] [--seed 1000]

axes: a style theme (configs/themes, assets.styles) sets the room: support tops (PBR), cabinet repaint (tint on the
photo-derived albedo keeps mouldings and handles), wall paint over observed walls and PBR on unseen ones, floor (PBR),
ceiling tint, furniture stain, and the sky: an outdoor HDRI for the dome light with a matching light temperature /
intensity range. Cousin i uses theme i mod #themes, so a batch covers every style. Then stools (the reconstructed
twin stool in the theme's stain, or any converted library stool: Poly Haven + ABO), target object (every third
cousin the twin as reconstructed, the others the twin itself varied in shape, colour and finish: assets.augment),
other dynamic objects (bowls swap to YCB / Poly Haven bowls), distractors (Poly Haven / YCB) off the hand's approach
corridor, background prop swaps (fixed lists or a library category).
Semantics (assets.shape): library stools keep their true size and must seat at counter height (knee room under the
slab); a stool with a backrest turns it away from the counter, a backless one stands square to it; an elongated swap
keeps the twin instance's long axis; a front-facing prop (a picture frame on its easel) faces the photo camera; a
file that lays several variants out in a row (assets.library.variant_sheet) is not one prop; a target mug's handle
points away from the hand.
Which parts each axis touches comes from the scene graph's roles (scenegraph.cousin_parts); configs/<name>/
cousins.json may override the part lists and add static_swaps. Directions (approach corridor, handle away from the
hand) come from the task frame (sim.taskframe). Writes cousins/cousin_XX.json + manifest.json.
"""
import argparse
import json
import re

import numpy as np
import trimesh

from calibrated_scene_reconstruction.assets.augment import augment
from calibrated_scene_reconstruction.assets.library import model_profile, usd_index, variant_sheet
from calibrated_scene_reconstruction.assets.library import models as library_models
from calibrated_scene_reconstruction.assets.shape import angle, footprint_axes, nearest_counter, profile, seat_range, toward_box, wrap
from calibrated_scene_reconstruction.assets.styles import hdri_path, sample, themes
from calibrated_scene_reconstruction.core.paths import YCB_ROOT, rel, resolve
from calibrated_scene_reconstruction.scene.graph import cousin_parts
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame

COUNTERS = ["marble_01", "grey_cartago_01", "brushed_concrete", "american_walnut_veneer", "ash_veneer",
            "cherry_veneer", "black_walnut_veneer_01", "terrazzo_tiles"]
FLOORS = ["diagonal_parquet", "brown_planks_05", "dark_wooden_planks", "brown_floor_tiles", "laminate_floor_02",
          "wood_floor"]
PAINTS = {"sage": (0.72, 0.86, 0.70), "navy": (0.38, 0.48, 0.78), "white": (1.25, 1.22, 1.15),
          "charcoal": (0.45, 0.45, 0.47), "terracotta": (1.15, 0.75, 0.58), "butter": (1.2, 1.12, 0.8)}
WALLS = {"warm_white": (1.05, 1.02, 0.95), "grey_blue": (0.85, 0.92, 1.05), "blush": (1.08, 0.94, 0.9),
         "greige": (0.95, 0.93, 0.88), "mint": (0.9, 1.04, 0.95)}
STAINS = {"natural": (1, 1, 1), "walnut": (0.55, 0.42, 0.32), "ebony": (0.3, 0.27, 0.25), "white_wash": (1.3, 1.28, 1.22),
          "honey": (1.1, 0.9, 0.62)}
PH_PROPS = ["food_apple_01", "food_pears_asian_01", "lemon", "bananas", "wooden_bowl_02", "metal_jug",
            "vintage_electric_kettle", "wooden_cutting_board", "pot_enamel_01", "ceramic_vase_03"]  # scans: true scale
YCB_PROPS = ["013_apple", "014_lemon", "017_orange", "011_banana", "016_pear"]
# what stands on a bench in a lab / workshop / office instead of fruit and kitchenware (YCB scans, true scale)
WORK_ROOMS = ("lab", "workshop", "office")
WORK_PROPS = ["035_power_drill", "037_scissors", "040_large_marker", "042_adjustable_wrench", "043_phillips_screwdriver",
              "044_flat_screwdriver", "048_hammer", "050_medium_clamp", "061_foam_brick", "036_wood_block",
              "077_rubiks_cube", "026_sponge", "003_cracker_box", "004_sugar_box"]


FABRICS = ["cotton_jersey", "rough_linen"]  # a waffle or hessian weave tinted far from its own colour read as a towel
FABRIC_COLOURS = {"beige": (196, 184, 160), "light_grey": (182, 182, 180), "off_white": (226, 222, 214),
                  "blue_grey": (122, 136, 152), "charcoal": (72, 72, 74), "sage": (150, 164, 140)}
FABRIC_KITS = ("folding_screen",)  # set pieces of cloth a cousin re-covers


def set_piece_looks(sg, rng):
    """A cousin's own cloth for the scene's fabric set pieces (a folding screen): another Poly Haven fabric tinted to
    a colour of the palette -- the twin's photo-matched cloth would otherwise stand unchanged in every cousin."""
    from calibrated_scene_reconstruction.scene.background.materials import lin, tex_mean

    out = {}
    for name, sp in (sg.get("set_pieces") or {}).items():
        if sp.get("kit") not in FABRIC_KITS:
            continue
        fab = FABRICS[int(rng.integers(len(FABRICS)))]
        col = list(FABRIC_COLOURS.values())[int(rng.integers(len(FABRIC_COLOURS)))]
        tint = np.clip(lin(np.asarray(col, float)) / np.maximum(lin(tex_mean(fab)), 1e-4), 0.02, 2.0)
        out[name] = {"tex": fab, "tint": [round(float(v), 3) for v in tint]}
    return out


def scene_room(sg):
    """The room type of a scene (scene graph task.place.room, else a top-level "room"), None if not named."""
    return (((sg.get("task") or {}).get("place") or {}).get("room")) or sg.get("room")


def ceiling_look(look):
    """A cousin's ceiling: clean paint in the theme's ceiling tint."""
    return surface("proc_painted_drywall", 1.0, look["ceiling_tint"])


def room_themes(sg):
    """The style themes whose `rooms` include the scene's room (all of them when the room is not named or no theme
    lists it): a lab does not get a farmhouse kitchen's counters, floors and fruit."""
    all_themes = themes()
    room = scene_room(sg)
    fit = {k: v for k, v in all_themes.items() if room and room in v.get("rooms", ())}
    return fit or all_themes


def surface(choice, tile_m, tint=None):
    """A theme's surface choice as a room override: a Poly Haven texture, or "color:r,g,b" a plain finish (laminate,
    painted metal, epoxy)."""
    if str(choice).startswith("color:"):
        c = [float(v) for v in choice[len("color:"):].split(",")]
        if tint is not None:
            c = [a * b for a, b in zip(c, tint)]
        return {"color": c, "roughness": 0.45}
    return {"pbr": choice, "tile_m": tile_m, **({"tint": tint} if tint is not None else {})}


def ycb(name, as_name, xy, yaw, mass):
    d = YCB_ROOT / name / "google_16k"
    return {"name": name, "as": as_name, "mesh_path": rel(d / "textured.obj"), "texture": rel(d / "texture_map.png"),
            "mass": mass, "xy": [float(v) for v in xy], "yaw": float(yaw)}


def fits(s, name, model, height=None, slack=1.25):
    """A swap must fit the space the twin instance occupies: footprint (largest side, at the given or natural height)
    and height within `slack` of the instance; scene build (sim.clearance) checks the rest."""
    m = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"])
    ext = np.ptp(np.asarray(m.vertices) @ T[:3, :3].T, axis=0)
    size = np.asarray(usd_index()["meta"][model]["size"], float)
    size = size * (height / size[2]) if height else size
    return size[:2].max() <= slack * ext[:2].max() and size[2] <= slack * ext[2]


def rests(model, tol=0.01):
    """Passed the library drop test (apps/library_drop_test.py): its visuals rest on what its collision rests on."""
    off = usd_index()["meta"][model].get("rest_offset_m")
    return off is not None and abs(off) <= tol


def natural_height(model):
    return float(usd_index()["meta"][model]["size"][2])


def stool_pool(s, xy):
    """Library stools whose seat suits the support next to xy (assets.shape.seat_range)."""
    boxes = s.manifest["collision_boxes"]
    name, lo, hi = nearest_counter(np.asarray(xy), boxes)
    s0, s1 = seat_range(boxes, name, lo, hi)
    return [m for m in library_models("stool") if s0 <= model_profile(m)["seat_h"] <= s1]


def stool_yaw(s, model, xy, rng):
    """A backrest turned away from the nearest counter, a backless stool square to it."""
    _, lo, hi = nearest_counter(np.asarray(xy), s.manifest["collision_boxes"])
    a = angle(toward_box(xy, lo, hi))
    back = model_profile(model)["back"]
    if back:
        return wrap(a + np.pi - angle(back) + rng.uniform(-0.15, 0.15))
    return wrap(a + rng.uniform(-0.2, 0.2))


def furniture(s, twin, rng, stain=None):
    """One stool design per cousin: the reconstructed twin stool in the theme's stain, or a library stool."""
    base = twin["stool_models"]
    if not base:
        return [], "none"
    recon = next((st["recon"] for st in base if st.get("recon")), None)
    pool = ([f"recon:{recon}"] if recon else []) + stool_pool(s, base[0]["center"])
    pick = pool[rng.integers(len(pool))]
    if stain is None:
        stain = list(STAINS)[rng.integers(len(STAINS))]
    stain_name, stain_tint = stain if isinstance(stain, tuple) else (stain, STAINS[stain])
    out = []
    for st in base:
        if rng.random() < 0.2:  # sometimes a stool is missing
            continue
        c = [st["center"][0] + float(rng.uniform(-0.12, 0.12)), st["center"][1] + float(rng.uniform(-0.06, 0.08))]
        if pick.startswith("recon:"):
            out.append({"recon": recon, "center": c, "yaw": st.get("yaw", 0.0) + float(rng.uniform(-0.4, 0.4)),
                        "tint": stain_tint})
        else:
            out.append({"model": pick, "center": c, "yaw": stool_yaw(s, pick, c, rng), "height": natural_height(pick)})
    return out, (f"{pick}/{stain_name}" if pick.startswith("recon:") else pick)


from calibrated_scene_reconstruction.assets.mesh_frames import twin_world as twin_world  # noqa: E402,F401 -- public helper


def neighbour_points(s, exclude, z0, z1=None):
    """xy of the twin's static and other dynamic objects' surface points between heights z0 and z1 (the band a
    target on the support occupies): where a moved target must not land."""
    z1 = z0 + 0.2 if z1 is None else z1
    out = []
    sg = s.scene_graph
    for name in list(sg.get("static_objects", {})) + list(sg.get("dynamic_objects", {})):
        if name in exclude or not (s / "objects" / name / "mesh_cam.glb").exists():
            continue
        V = twin_world(s, name)
        V = V[(V[:, 2] > z0 - 0.01) & (V[:, 2] < z1)]
        if len(V):
            out.append(V[:: max(1, len(V) // 2000), :2])
    return np.concatenate(out) if out else np.zeros((0, 2))


def footprint_radius(s, spec):
    """Horizontal radius of an object spec as the scene builds it (a YCB mesh, or the scaled twin)."""
    if spec.get("mesh_path"):
        V = np.asarray(trimesh.load(resolve(spec["mesh_path"]), force="mesh").vertices)
    else:
        V = twin_world(s, spec["name"])
    c = (V.min(0) + V.max(0))[:2] / 2
    return float(np.max(np.linalg.norm(V[:, :2] - c, axis=1))) * float(spec.get("scale", 1.0))


def free_target_xy(s, target, r, t0, tf, rng, gap=0.015, tries=200):
    """A jittered target position in the task frame whose footprint (radius r) keeps `gap` from the scene's other
    objects (a cousin's cup must not land on the tray beside it); the twin's own position when no draw is free."""
    top = s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2]
    pts = neighbour_points(s, {target}, top)
    (sx0, sy0, _), (sx1, sy1, _) = s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"]

    def free(xy):
        on = sx0 + r + 0.02 < xy[0] < sx1 - r - 0.02 and sy0 + r + 0.02 < xy[1] < sy1 - r - 0.02
        return on and (not len(pts) or np.linalg.norm(pts - xy, axis=1).min() > r + gap)
    for _ in range(tries):
        xy = np.asarray(t0) + (rng.uniform(-0.05, 0.08) * tf.right + rng.uniform(-0.07, 0.04) * tf.forward)[:2]
        if free(xy):
            return [float(v) for v in xy]
    # nothing free around the twin's spot (a reconstruction swollen into its neighbours): the nearest free spot along
    # the support's edge (within 0.35 m), at the twin's distance from it -- the robot's placement shifts with the
    # target, so the target must not move into the support
    cand = sorted(((a, b) for a in np.linspace(-0.35, 0.35, 36) for b in np.linspace(-0.07, 0.04, 6)),
                  key=lambda ab: (abs(ab[0]), abs(ab[1])))
    for a, b in cand:
        xy = np.asarray(t0) + (a * tf.right + b * tf.forward)[:2]
        if free(xy):
            return [float(v) for v in xy]
    return [float(v) for v in t0]


def twin_xy(s, name):
    V = twin_world(s, name)
    return (V.min(0) + V.max(0))[:2] / 2


def swap_yaw(s, name, model, rng, base=None):
    """Orientation that keeps the twin instance's semantics: a front-facing prop shows its front (away from its
    easel) to the photo camera; an elongated piece keeps the instance's long axis (the twin swap's model and yaw when
    the twin itself is a library model, else the reconstruction's footprint); anything else any yaw."""
    p = model_profile(model)
    if p["stand"]:
        cam = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"])[:2, 3]
        return wrap(angle(cam - twin_xy(s, name)) - angle(-np.asarray(p["stand"])) + rng.uniform(-0.3, 0.3))
    if p["long"] is not None:
        if base and model_profile(base["model"])["long"] is not None:
            axis = base["yaw"] + model_profile(base["model"])["long"]
        else:
            axis = footprint_axes(twin_world(s, name))[0]
        return wrap(axis - p["long"] + np.pi * rng.integers(2) + rng.uniform(-0.15, 0.15))
    return float(rng.uniform(-np.pi, np.pi))


def handle_away(s, spec, away=np.pi):
    """Yaw that turns a target's handle away from the hand: the lateral grasp's palm comes from the robot's right of
    the object (robots.grasps.power_lateral), so the handle points to the task frame's left; the sampled yaw only adds a little variation (no extra random draw)."""
    V = (np.asarray(trimesh.load(resolve(spec["mesh_path"]), force="mesh").vertices) if spec.get("mesh_path")
         else twin_world(s, spec["name"]))
    h = profile(V)["handle"]
    return spec["yaw"] if h is None else wrap(away - angle(h) + 0.35 * np.sin(spec["yaw"]))


FIT_CUPS = ["065-e_cups", "065-d_cups", "065-c_cups", "065-b_cups", "065-a_cups"]  # 7.6 .. 5.7 cm wide


def grip_width(V):
    """Narrowest horizontal width of a point set (rotating calipers on its 2D hull): what a parallel gripper closes
    across -- a mug's body, not its body plus handle."""
    from scipy.spatial import ConvexHull

    P = np.asarray(V, float)[:, :2]
    H = P[ConvexHull(P).vertices]
    E = np.roll(H, -1, axis=0) - H
    N = np.stack([-E[:, 1], E[:, 0]], 1) / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return float(min(np.ptp(H @ n) for n in N))


def object_width(s, spec):
    """Grip width (grip_width) of a variant object (twin mesh, YCB scan); None for library models."""
    if spec.get("mesh_path"):
        V = np.asarray(trimesh.load(resolve(spec["mesh_path"]), force="mesh").vertices)
    elif spec.get("name") and (s / "objects" / spec["name"] / "mesh_cam.glb").exists():
        V = twin_world(s, spec["name"])
    else:
        return None
    return grip_width(V) * spec.get("scale", 1.0)


def fit_target(s: Session, variant, max_width, min_scale=0.6):
    """Robot-aware target: a parallel gripper cannot close around a target wider than its stroke. The same object is
    scaled down to fit (every robot grasps the object the scene shows); only when that would shrink it below
    `min_scale` is it replaced by the largest YCB cup that fits, at the same place. Returns (variant, a label of the
    change or None)."""
    target = s.scene_graph["task"]["target"]
    objects = [dict(o) for o in variant.get("objects", [])] or \
        [{"name": n, "mass": spec.get("mass", 0.3)} for n, spec in s.manifest["dynamic_objects"].items()]
    k = next(i for i, o in enumerate(objects) if o.get("as", o.get("name")) == target)
    w = object_width(s, objects[k])
    if w is None or w <= max_width:
        return variant, None
    f = max_width / w
    if f >= min_scale:
        o = objects[k]
        objects[k] = {**o, "scale": float(o.get("scale", 1.0)) * f,
                      "mass": max(0.08, float(o.get("mass", 0.3)) * f ** 3)}
        if "name" not in objects[k] and not o.get("mesh_path"):
            objects[k]["name"] = target
        return {**variant, "objects": objects}, f"{o.get('name', target)} scaled x{f:.2f}"
    cup = next((c for c in FIT_CUPS if np.ptp(np.asarray(trimesh.load(YCB_ROOT / c / "google_16k" / "textured.obj",
                                                                          force="mesh").vertices)[:, :2], axis=0).max()
                <= max_width), FIT_CUPS[-1])
    xy = objects[k].get("xy") or list(twin_xy(s, target))
    objects[k] = ycb(cup, target, xy, objects[k].get("yaw", 0.0), 0.15)
    return {**variant, "objects": objects}, cup


def generate(s: Session, i, rng, cfg, twin):
    sg = s.scene_graph
    target = sg["task"]["target"]
    others = [n for n in sg["dynamic_objects"] if n != target]
    (sx0, sy0, _), (sx1, sy1, _) = sg["furniture"][sg["task"]["support"]]["box"]
    cp = {**cousin_parts(sg, s.manifest), **{k: v for k, v in cfg.items() if k.endswith("_parts")}}
    all_themes = room_themes(sg)
    look = sample(all_themes[sorted(all_themes)[i % len(all_themes)]], rng)
    floor_name = next(k for k, v in sg["structure"].items() if v.get("role", k) == "floor")
    ceil_name = next((k for k, v in sg["structure"].items() if v.get("role", k) == "ceiling"), "ceiling")
    ov = {k: surface(look["support"], 1.2) for k in cp["counter_parts"]}
    ov[floor_name] = surface(look["floor"], 1.5)
    ov.update({k: {"tint": look["paint_tint"]} for k in cp["paint_parts"]})
    # walls the photo saw get the theme's material too: a tint over the photo texture kept its baked light, shadows
    # and the inpainted holes of whatever stood in front (the same blots in every cousin)
    ov.update({k: surface(look["wall_pbr"], 2.0, look["wall_tint"])
               for k in cp["observed_wall_parts"] + cp["unobserved_wall_parts"]})
    ov[ceil_name] = ceiling_look(look)  # its own (the twin's may be the photo's dark open ceiling)

    t0 = twin_xy(s, target)
    tf = TaskFrame(sg, [t0[0], t0[1], sg["furniture"][sg["task"]["support"]]["box"][1][2]])
    away = float(np.arctan2(tf.left[1], tf.left[0]))
    mass = float(sg["dynamic_objects"].get(target, {}).get("mass", 0.35))
    if i % 3 == 0:  # the target as reconstructed
        objects = [{"name": target, "as": target, "mass": mass, "scale": float(rng.uniform(0.9, 1.1)),
                    "xy": [float(v) for v in t0], "yaw": float(rng.uniform(-np.pi, np.pi))}]
    else:  # the reconstructed target itself varied in shape, colour and finish (assets.augment), not a library swap
        aug = augment(s, target, rng, s / "cousins" / f"cousin_{i:02d}" / f"aug_{target}" / "mesh", mass=mass)
        objects = [{**aug, "as": target, "xy": [float(v) for v in t0], "yaw": float(rng.uniform(-np.pi, np.pi))}]
    t_xy = free_target_xy(s, target, footprint_radius(s, objects[0]), t0, tf, rng)
    objects[0]["xy"] = t_xy
    objects[0]["yaw"] = handle_away(s, objects[0], away)
    taken = [t_xy]
    for name in others:  # bowls: YCB bowl or a Poly Haven wooden bowl at the twin position; others: the twin
        xy = twin_xy(s, name)
        taken.append(list(xy))
        if re.search("bowl", name):
            objects.append(ycb("024_bowl", name, xy, 0.0, 0.4) if i % 2 == 0 else
                           {"ph_model": "wooden_bowl_01", "as": name, "xy": [float(v) for v in xy], "height": 0.09,
                            "mass": 0.5})
        else:
            objects.append({"name": name, "as": name, "mass": sg["dynamic_objects"][name].get("mass", 0.3),
                            "xy": [float(v) for v in xy], "yaw": 0.0})
    no_go = list((((sg.get("task") or {}).get("place") or {}).get("keep_out") or {}).values())  # a cooktop, a sink
    boxes = s.manifest["collision_boxes"] if (s / "scene" / "manifest.json").exists() else {}
    sup = boxes.get(sg["task"]["support"])
    if sup:  # what stands on the support itself: props tipped over them
        no_go += [[lo[:2], hi[:2]] for n, (lo, hi) in boxes.items()
                  if n != sg["task"]["support"] and abs(lo[2] - sup[1][2]) < 0.01
                  and lo[0] < sup[1][0] and sup[0][0] < hi[0] and lo[1] < sup[1][1] and sup[0][1] < hi[1]]
    for k in range(int(rng.integers(2, 5))):
        for _ in range(300):  # off the hand's approach corridor (the target's right side), other objects, keep-outs
            xy = [float(rng.uniform(sx0 + 0.1, sx1 - 0.1)), float(rng.uniform(sy0 + 0.08, sy1 - 0.08))]
            d = np.asarray(xy) - np.asarray(t_xy)
            corridor = -0.05 < d @ tf.right[:2] < 0.55 and d @ tf.forward[:2] < 0.12
            off = all(not (a[0][0] - 0.06 < xy[0] < a[1][0] + 0.06 and a[0][1] - 0.06 < xy[1] < a[1][1] + 0.06)
                      for a in no_go)
            if not corridor and off and all(np.hypot(xy[0] - t[0], xy[1] - t[1]) > 0.22 for t in taken):
                break
        taken.append(xy)
        work = scene_room(sg) in WORK_ROOMS
        props = [] if work else [m for m in PH_PROPS if rests(m) and not variant_sheet(m)]
        if work:
            objects.append(ycb(WORK_PROPS[rng.integers(len(WORK_PROPS))], f"prop_{k}", xy, rng.uniform(-3, 3), 0.3))
        elif props and rng.random() < 0.6:
            mname = props[rng.integers(len(props))]
            objects.append({"ph_model": mname, "as": f"prop_{k}", "xy": xy, "height": natural_height(mname),
                            "yaw": float(rng.uniform(-3, 3)), "mass": 0.3})
        else:
            objects.append(ycb(YCB_PROPS[rng.integers(len(YCB_PROPS))], f"prop_{k}", xy, rng.uniform(-3, 3), 0.15))
    stools, stool = furniture(s, twin, rng, (look["stain"], look["stain_tint"]))
    swaps = {}
    for name, opts in cfg.get("static_swaps", {}).items():
        if rng.random() < 0.5:
            if isinstance(opts, dict):  # a library category: natural size, the twin's orientation and footprint
                pool = [m for m in library_models(opts["category"]) if fits(s, name, m)]
                if not pool:
                    continue
                mname = pool[rng.integers(len(pool))]
                base = twin.get("static_swaps", {}).get(name)
                swaps[name] = {**{k: base[k] for k in ("xy", "support_z") if base and k in base}, "model": mname,
                               "height": natural_height(mname), "yaw": swap_yaw(s, name, mname, rng, base)}
            else:
                opts = [(m, h) for m, h in opts if fits(s, name, m, h)]
                if not opts:
                    continue
                mname, h = opts[rng.integers(len(opts))]
                swaps[name] = {"model": mname, "height": h, "yaw": swap_yaw(s, name, mname, rng)}
    night = "night" in look["hdri"]
    hdri = hdri_path(look["hdri"])
    light = {"scale": look["light"]["scale"] * (0.8 if night else 1.0), "temp": look["light"]["temp"],
             "sun": bool(not night and rng.random() < 0.6), "hdri": hdri and rel(hdri), "hdri_name": look["hdri"],
             "hdri_intensity": 150.0 if night else 900.0}
    summary = (f"theme={look['theme']}, worktop={look['support']}, cabinet finish={look['paint']}, floor={look['floor']}, "
               f"wall={look['wall']}/{look['wall_pbr']}, sky={look['hdri']}, furniture={stool}x{len(stools)}, "
               f"target={objects[0]['name']}, prop replacements={len(swaps)}, light={light['temp']:.0f}K x{light['scale']:.2f}")
    return {"id": i, "theme": look, "room_overrides": ov, "objects": objects, "stool_models": stools,
            "static_swaps": swaps, "lights": light, "set_piece_overrides": set_piece_looks(sg, rng), "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    s = Session(args.session)
    cfg, twin = s.config("cousins.json", {}), s.read_json("scene/twin.json")
    index = []
    for i in range(args.n):
        c = generate(s, i, np.random.default_rng(args.seed + i), cfg, twin)
        old = s / "cousins" / f"cousin_{i:02d}.json"
        prev = json.loads(old.read_text()) if old.exists() else {}
        if prev.get("theme", {}).get("theme") == c["theme"]["theme"]:  # the same cousin made again (a run resumed
            # before this stage): what later stages wrote into its file stays -- its room completion, its room's
            # reconstructed pieces with the agent's review fixes
            for k in ("room_completion", "synthetic_furniture"):
                if k in prev:
                    c[k] = prev[k]
            for k, v in prev.get("room_overrides", {}).items():
                c["room_overrides"].setdefault(k, v)
        s.write_json(f"cousins/cousin_{i:02d}.json", c)
        index.append({"id": i, "file": f"cousin_{i:02d}.json", "summary": c["summary"]})
        print(i, c["summary"])
    s.write_json("cousins/manifest.json", index)


if __name__ == "__main__":
    main()
