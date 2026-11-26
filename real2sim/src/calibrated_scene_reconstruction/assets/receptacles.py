"""Visible place targets ("receptacles") for the place task: the goal is something the cameras see and a policy can
be conditioned on -- a coaster, a placemat, a tray, a plate, a cutting board, a taped square (the lab's designated
spot), or a container the object goes into: a bowl, a basket.

    sample(obj_radius, obj_height, rng, weights=None) -> Receptacle | None
    kinds_for(room) -> the kinds that belong on a table in that room (KINDS[kind]["rooms"])

Each kind declares the rooms it is found in (a taped square marks a spot on a lab bench, not a kitchen counter; a
cutting board lives in the kitchen); the scene graph names the room (task.place.room) and its weights pick among
those kinds (grasping.place.receptacle_weights). Procedural kinds are built here: convex slabs (coasters, placemats,
trays), a hemmed fabric panel, bamboo slats, tape strips, a turned ceramic plate; their looks are Poly Haven PBR
textures (wood veneers, placemat fabrics) at the texture's real-world size, generated texture sets (woven seagrass /
rattan, cork, slate: albedo, normal and roughness maps, cached under outputs/cache) or glaze / metal / tape colours.
Scanned kinds are Poly Haven library models scaled within a plausible range. Every receptacle is static in the
simulator with a triangle-mesh collider (a basket's inside stays hollow).

Receptacle frame: origin at the bottom centre on the support, z up. `place_z`: height of the surface the object
stands on; `rim_z`: the highest point (the object is carried above it); `accept`: where the object's base centre
must end -- ("circle", r) or ("rect", hx, hy) about `accept_xy`, the place surface shrunk by half the object's radius;
`half`: half extents of the footprint (goal sampling keeps it on the support and clear of other things). Scanned
models are measured from a top-down height map of their surface: the place surface is the flat region around the
centre at the centre's height.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR, ROOT

ROOMS = ("kitchen", "dining", "cafe", "living", "bedroom", "bathroom", "office", "lab", "workshop")
# Poly Haven textures with their real-world size (m, the asset's `dimensions`): UVs are metres / size, so a veneer's
# grain or a fabric's weave shows at the scale of the real thing
TEX_SIZE = {"oak_veneer_01": 1.83, "ash_veneer": 1.0, "american_walnut_veneer": 1.0, "black_walnut_veneer_01": 1.0,
            "cherry_veneer": 1.0, "bamboo_veneer": 1.0, "hessian_230": 0.268, "hessian_380": 0.274,
            "rough_linen": 0.271, "cotton_jersey": 0.264, "poly_wool_herringbone": 0.273,
            "waffle_pique_cotton": 0.279, "gingham_check": 0.29,
            # furniture tops (sim.setpieces looks, assets.kits)
            "wood_table": 1.0, "wood_table_large": 1.55, "wood_planks_dirt": 2.0, "wood_table_worn": 0.55,
            "plywood": 0.5, "wood_table_001": 1.0, "kitchen_wood": 0.6}
WOODS = ["oak_veneer_01", "ash_veneer", "american_walnut_veneer", "black_walnut_veneer_01", "cherry_veneer"]
FABRICS = ["hessian_230", "hessian_380", "rough_linen", "cotton_jersey", "poly_wool_herringbone",
           "waffle_pique_cotton", "gingham_check"]
FIBRE = [(0.74, 0.62, 0.43), (0.62, 0.47, 0.30), (0.80, 0.71, 0.55), (0.45, 0.34, 0.23), (0.66, 0.64, 0.58)]
CORK = [(0.66, 0.48, 0.31), (0.58, 0.41, 0.26), (0.72, 0.56, 0.38)]
SLATE = [(0.21, 0.22, 0.24), (0.26, 0.26, 0.27), (0.19, 0.21, 0.21)]
GLAZES = [(0.93, 0.92, 0.89), (0.90, 0.87, 0.80), (0.60, 0.60, 0.58), (0.60, 0.66, 0.56), (0.45, 0.52, 0.60),
          (0.80, 0.70, 0.56), (0.20, 0.20, 0.21)]
METALS = [{"color": (0.78, 0.78, 0.76), "metallic": 1.0, "roughness": 0.28},   # stainless
          {"color": (0.09, 0.09, 0.09), "metallic": 0.6, "roughness": 0.45}]   # black powder-coated
TAPE = [(0.16, 0.38, 0.78), (0.10, 0.55, 0.32), (0.92, 0.78, 0.18), (0.93, 0.93, 0.90), (0.85, 0.30, 0.12)]
# the generated texture sets: metres one image spans
PROC_TILE = {"woven": (0.10, 0.13), "cork": (0.10, 0.10), "slate": (0.15, 0.15)}

# kind: type, label (task text), source, the rooms it belongs in
KINDS = {
    "coaster": {"type": "flat", "label": "coaster", "source": "procedural",
                "rooms": ("kitchen", "dining", "cafe", "living", "bedroom", "office")},
    "placemat": {"type": "flat", "label": "placemat", "source": "procedural", "rooms": ("kitchen", "dining", "cafe")},
    "tray": {"type": "flat", "label": "tray", "source": "procedural",
             "rooms": ("kitchen", "dining", "cafe", "living", "bedroom", "bathroom", "office", "lab", "workshop")},
    "tape_square": {"type": "flat", "label": "taped square", "source": "procedural", "rooms": ("lab", "workshop")},
    "plate": {"type": "flat", "label": "plate", "source": "procedural", "rooms": ("kitchen", "dining", "cafe")},
    "cutting_board": {"type": "flat", "label": "cutting board", "source": "polyhaven:wooden_cutting_board",
                      "rooms": ("kitchen",)},
    "bowl": {"type": "container", "label": "bowl", "source": "polyhaven:wooden_bowl_01",
             "rooms": ("kitchen", "dining", "cafe", "living")},
    "basket": {"type": "container", "label": "basket", "source": "polyhaven:wicker_basket_01",
               "rooms": ("kitchen", "dining", "living", "bedroom", "bathroom")},
    # the one goal marker of a batch collection (apps/episode.py --receptacles marker): the same thing in every scene,
    # so a vision policy learns one goal to look for; not drawn by a room's default kinds (opt-in only)
    "marker": {"type": "flat", "label": "orange mat", "source": "procedural", "rooms": ()},
}
MARKER_COLOR, MARKER_RING = (0.86, 0.30, 0.05), (0.92, 0.91, 0.88)  # linear: a saturated orange, an off-white ring
DEFAULT_WEIGHTS = {k: 1.0 for k in KINDS if k != "marker"}
FINGER_ROOM = 0.045  # m, between the object and a container's wall for the fingers around it
TEXTURE_CACHE = ROOT / "outputs" / "cache" / "receptacle_textures"


def kinds_for(room):
    """The kinds that belong in a room (None: every kind)."""
    if room is None:
        return [k for k, v in KINDS.items() if v["rooms"]]  # the marker is opt-in
    if room not in ROOMS:
        raise ValueError(f"room {room!r}: one of {ROOMS}")
    return [k for k, v in KINDS.items() if room in v["rooms"]]


@dataclass
class Receptacle:
    kind: str
    label: str
    type: str                       # flat | container
    source: str                     # procedural | polyhaven:<model>
    place_z: float
    rim_z: float
    accept: list                    # ["circle", r] | ["rect", hx, hy]
    accept_xy: list
    half: list                      # footprint half extents (x, y)
    scale: float = 1.0              # library models
    parts: list = field(default_factory=list)   # procedural: [{"name", "V", "F", "UV", "material"}]
    material: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def to_dict(self, geometry=False):
        d = asdict(self)
        if not geometry:
            d.pop("parts")
        else:
            d["parts"] = [{k: (np.asarray(v).tolist() if k in ("V", "F", "UV") else v) for k, v in p.items()}
                          for p in self.parts]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["parts"] = [{k: (np.asarray(v) if k in ("V", "F", "UV") else v) for k, v in p.items()}
                      for p in d.get("parts", [])]
        return Receptacle(**d)


# --- procedural geometry (convex outlines, fan-triangulated) -------------------------------------------------------
def rounded_rect(hx, hy, r, n=8):
    r = min(r, hx, hy)
    out = []
    for cx, cy, a0 in ((hx - r, hy - r, 0), (-hx + r, hy - r, 90), (-hx + r, -hy + r, 180), (hx - r, -hy + r, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, n)):
            out.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(out)


def circle(r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def slab(outline, z0, z1, tile=0.3):
    """Extrusion of a convex outline (counter-clockwise) between z0 and z1, planar UVs (metres / tile)."""
    n = len(outline)
    c = outline.mean(0)
    V = np.r_[np.c_[outline, np.full(n, z0)], np.c_[outline, np.full(n, z1)], [[*c, z0], [*c, z1]]]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i],           # side
              [2 * n + 1, n + i, n + j], [2 * n, j, i]]   # top, bottom
    return V, np.array(F), V[:, :2] / tile


def ring(outer, inner, z0, z_floor, z_top, tile=0.3):
    """A tray: floor slab under `outer` up to z_floor plus a wall between `outer` and `inner` (same vertex count, both
    convex) up to z_top."""
    n = len(outer)
    V = np.r_[np.c_[outer, np.full(n, z0)], np.c_[outer, np.full(n, z_top)], np.c_[inner, np.full(n, z_top)],
              np.c_[inner, np.full(n, z_floor)], [[*outer.mean(0), z0], [*inner.mean(0), z_floor]]]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i],                                  # outer side
              [n + i, n + j, 2 * n + j], [n + i, 2 * n + j, 2 * n + i],          # top of the wall
              [2 * n + i, 2 * n + j, 3 * n + j], [2 * n + i, 3 * n + j, 3 * n + i],  # inner side
              [4 * n + 1, 3 * n + i, 3 * n + j], [4 * n, j, i]]                  # floor, bottom
    return V, np.array(F), V[:, :2] / tile


def box(lo, hi, tile=0.3):
    """Axis-aligned box; UVs along its two longest sides (wood grain runs along a board)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    out = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    V, F, _ = slab(out, lo[2], hi[2], tile)
    ax = np.argsort(hi - lo)[::-1][:2]
    return V, F, V[:, ax] / tile


def lathe(profile, n=64, tile=0.3):
    """Surface of revolution about z of a profile [(r, z), ...] whose first and last points lie on the axis (a plate:
    over the top from the centre out to the rim, back under it to the centre); planar UVs. Faces point outward
    whichever way the profile turns (a tumbler's runs out along its foot and up: the other way round), as a
    refracting material needs."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = profile[1:-1]
    V = np.r_[[[0.0, 0.0, profile[0][1]]], [[r * np.cos(t), r * np.sin(t), z] for r, z in rings for t in a],
              [[0.0, 0.0, profile[-1][1]]]]
    m, last = len(rings), len(V) - 1

    def k(i, j):
        return 1 + i * n + j % n

    F = [[0, k(0, j), k(0, j + 1)] for j in range(n)]
    for i in range(m - 1):
        for j in range(n):
            F += [[k(i, j), k(i + 1, j), k(i + 1, j + 1)], [k(i, j), k(i + 1, j + 1), k(i, j + 1)]]
    F += [[last, k(m - 1, j + 1), k(m - 1, j)] for j in range(n)]
    F = np.array(F)
    r, z = np.asarray(profile, float).T
    if np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z) > 0:  # counter-clockwise in (r, z): inward as built
        F = F[:, ::-1]
    return V, F, V[:, :2] / tile


# --- looks -----------------------------------------------------------------------------------------------------------
def _pick(rng, xs):
    return xs[int(rng.integers(len(xs)))]


def _choose(rng, p):
    """A key of {option: probability}."""
    return list(p)[int(rng.choice(len(p), p=list(p.values())))]


def _textured(rng, V, mat, along_y=False):
    """UVs of a textured part at its texture's real-world size (`tile` m per image), from a random spot of the image
    (no two coasters show the same piece of veneer); `along_y` turns the image so its vertical (a veneer's grain)
    runs along x."""
    tile = mat["tile"]
    uv = V[:, [1, 0]] if along_y else V[:, :2]
    return uv / tile + rng.uniform(0, 1, 2)


def _poly(rng, names):
    name = _pick(rng, names)
    return {"tex": name, "tile": TEX_SIZE[name]}


def _proc(rng, kind, colours, variants=4):
    """A generated texture set (texture_set): a few seeds per colour, so the cache stays small."""
    lo, hi = PROC_TILE[kind]
    return {"proc": kind, "rgb": list(_pick(rng, colours)), "seed": int(rng.integers(variants)),
            "tile": float(rng.uniform(lo, hi)) if hi > lo else lo}


def _write_set(d, albedo, height, rough, px):
    """diff / nor_gl / rough maps (Poly Haven's layout, sim.usd.pbr_material) of a tileable height field (m) sampled
    every px metres."""
    from PIL import Image
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / (2 * px)
    dy = -(np.roll(height, -1, 0) - np.roll(height, 1, 0)) / (2 * px)   # image rows run down, v runs up
    n = np.dstack([-dx, -dy, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(albedo, 0, 1) * 255).astype(np.uint8)).save(d / "diff.jpg", quality=92)
    Image.fromarray(((n + 1) * 127.5).astype(np.uint8)).save(d / "nor_gl.jpg", quality=92)
    Image.fromarray((np.clip(rough, 0, 1) * 255).astype(np.uint8)).save(d / "rough.jpg", quality=92)


def _unit(x):
    return (x - x.mean()) / max(x.std(), 1e-9)


def _woven_maps(rgb, rng, S, tile):
    """Basket-woven seagrass / rattan strands (pairs over and under), each strand's shade jittered, fine fibres."""
    n_threads = 32                                                                     # strands per image (tileable)
    n = S // n_threads
    y, x = np.mgrid[0:S, 0:S]
    i, j = x // n, y // n
    over = (i // 2 + j // 2) % 2 == 0                                                 # warp on top here
    u, v = (x % n) / n, (y % n) / n
    prof = np.where(over, np.sin(np.pi * u), np.sin(np.pi * v)) ** 0.6               # a round strand across its width
    shade = np.where(over, rng.normal(1.0, 0.07, n_threads + 1)[i], rng.normal(0.92, 0.07, n_threads + 1)[j])
    fibre = 1 + 0.06 * rng.standard_normal((S, S))
    albedo = np.asarray(rgb)[None, None] * (shade * (0.6 + 0.4 * prof) * fibre)[..., None]
    height = 0.0012 * np.where(over, 0.5 + 0.5 * prof, 0.5 * prof)
    return albedo, height, 0.82 + 0.08 * (1 - prof)


def _cork_maps(rgb, rng, S, tile):
    """Agglomerated cork: granules of a few millimetres in mixed shades, dark pores between them."""
    from scipy.ndimage import gaussian_filter
    px = tile / S
    gran = _unit(gaussian_filter(rng.random((S, S)), 0.0008 / px, mode="wrap"))
    coarse = _unit(gaussian_filter(rng.random((S, S)), 0.003 / px, mode="wrap"))
    pores = gaussian_filter(rng.random((S, S)), 0.0004 / px, mode="wrap")
    pores = pores < np.quantile(pores, 0.05)
    shade = (1 + 0.15 * gran + 0.03 * coarse) * np.where(pores, 0.45, 1.0)
    albedo = np.asarray(rgb)[None, None] * shade[..., None]
    height = 0.0002 * gran - 0.0004 * pores
    return albedo, height, np.full((S, S), 0.88)


def _slate_maps(rgb, rng, S, tile):
    """Natural slate: cleft layers (long streaks along one direction), fine grain, a little sheen."""
    from scipy.ndimage import gaussian_filter
    px = tile / S
    layers = _unit(gaussian_filter(rng.random((S, S)), (0.0015 / px, 0.03 / px), mode="wrap"))
    grain = _unit(gaussian_filter(rng.random((S, S)), 0.0004 / px, mode="wrap"))
    albedo = np.asarray(rgb)[None, None] * (1 + 0.07 * layers + 0.04 * grain)[..., None]
    height = 0.00025 * layers + 0.00004 * grain
    return albedo, height, 0.6 + 0.06 * layers


def texture_set(mat, size=512):
    """Directory (relative to the repository) of a generated material's texture set (diff / nor_gl / rough, as a
    Poly Haven texture), written on first use under outputs/cache; deterministic in the material's arguments."""
    rgb = mat["rgb"]
    key = f"{mat['proc']}_{''.join(f'{int(255 * c):02x}' for c in rgb)}_{mat['seed']}_{int(1000 * mat['tile'])}"
    d = TEXTURE_CACHE / key
    if not (d / "rough.jpg").exists():
        maps = {"woven": _woven_maps, "cork": _cork_maps, "slate": _slate_maps}[mat["proc"]]
        _write_set(d, *maps(rgb, np.random.default_rng(mat["seed"]), size, mat["tile"]), mat["tile"] / size)
    return str(d.relative_to(ROOT))


def look(mat):
    """A short name of a material for reports: oak_veneer_01, cork, stainless, glaze, tape."""
    if "tex" in mat:
        return mat["tex"]
    if "proc" in mat:
        return mat["proc"]
    return mat.get("name", "colour")


# --- procedural kinds ------------------------------------------------------------------------------------------------
def _merge(parts):
    V, F, UV, k = [], [], [], 0
    for v, f, uv in parts:
        V.append(v), F.append(np.asarray(f) + k), UV.append(uv)
        k += len(v)
    return np.concatenate(V), np.concatenate(F), np.concatenate(UV)


def _accept(shape, dims, inset):
    if shape == "circle":
        return ["circle", max(dims[0] - inset, 0.005)]
    return ["rect", max(dims[0] - inset, 0.005), max(dims[1] - inset, 0.005)]


def _rec(parts, place_z, rim_z, accept, half, **kw):
    return dict(parts=parts, place_z=place_z, rim_z=rim_z, accept=accept, accept_xy=[0.0, 0.0], half=half, **kw)


def _part(name, V, F, UV, material):
    return {"name": name, "V": V, "F": F, "UV": UV, "material": material}


def _coaster(r_obj, rng, inset):
    """Wood, cork, slate or glazed ceramic; a disc or a rounded square of 9-11 cm."""
    U = rng.uniform
    if 0.9 * r_obj > 0.06:  # a coaster holds a cup or a glass, not a pot
        return None
    finish = _choose(rng, {"wood": 0.3, "cork": 0.3, "slate": 0.2, "ceramic": 0.2})
    t = {"wood": U(0.006, 0.01), "cork": U(0.004, 0.006), "slate": U(0.005, 0.007), "ceramic": U(0.006, 0.008)}[finish]
    if rng.random() < 0.6:
        r = max(U(0.045, 0.055), 0.9 * r_obj)
        out, acc, half = circle(r), ("circle", (r,)), [r, r]
    else:
        h = max(U(0.045, 0.052), 0.9 * r_obj)
        out, acc, half = rounded_rect(h, h, U(0.005, 0.012)), ("rect", (h, h)), [h, h]
    V, F, UV = slab(out, 0.0, t)
    if finish == "wood":
        mat = _poly(rng, WOODS)
    elif finish == "ceramic":
        mat = {"name": "glaze", "color": _pick(rng, GLAZES), "roughness": U(0.15, 0.4)}
    else:
        mat = _proc(rng, finish, CORK if finish == "cork" else SLATE)
    if "tile" in mat:
        UV = _textured(rng, V, mat)
    return _rec([_part("body", V, F, UV, mat)], t, t, _accept(acc[0], acc[1], inset), half)


def _placemat(r_obj, rng, inset):
    """A fabric mat with a raised hem (linen, cotton, jute, herringbone, gingham at the cloth's real weave), a woven
    seagrass / rattan mat (rectangular or round) or a mat of bamboo slats running across it; 40-46 x 28-33 cm."""
    U = rng.uniform
    hx, hy = max(U(0.20, 0.23), r_obj + 0.05), max(U(0.14, 0.165), r_obj + 0.05)
    finish = _choose(rng, {"fabric": 0.5, "woven": 0.3, "bamboo": 0.2})
    if finish == "fabric":
        t, hem, rr = U(0.0018, 0.003), U(0.012, 0.02), U(0.004, 0.012)
        mat = _poly(rng, FABRICS)
        outer, inner = rounded_rect(hx, hy, rr), rounded_rect(hx - hem, hy - hem, max(rr - hem, 0.002))
        V, F, _ = slab(inner, 0.0, t)
        V2, F2, _ = ring(outer, inner, 0.0, 0.0005, t + 0.0009)   # the hem: a folded double layer around the edge
        parts = [_part("panel", V, F, _textured(rng, V, mat), mat),
                 _part("hem", V2, F2, _textured(rng, V2, mat), {**mat, "tint": [0.9, 0.9, 0.9]})]
        return _rec(parts, t, t + 0.0009, _accept("rect", (hx - hem, hy - hem), inset), [hx, hy])
    if finish == "woven":
        t = U(0.004, 0.006)
        mat = _proc(rng, "woven", FIBRE)
        if rng.random() < 0.4:   # round, 34-38 cm
            r = max(U(0.17, 0.19), r_obj + 0.05)
            V, F, _ = slab(circle(r, 64), 0.0, t)
            return _rec([_part("body", V, F, _textured(rng, V, mat), mat)], t, t,
                        _accept("circle", (r - 0.01,), inset), [r, r])
        V, F, _ = slab(rounded_rect(hx, hy, U(0.01, 0.03)), 0.0, t)
        return _rec([_part("body", V, F, _textured(rng, V, mat), mat)], t, t,
                    _accept("rect", (hx - 0.01, hy - 0.01), inset), [hx, hy])
    # bamboo slats, 7-10 mm wide with a 1 mm gap, the grain along each slat (across the mat)
    t, w, gap = U(0.003, 0.004), U(0.007, 0.01), 0.001
    n = int(2 * hx // (w + gap))
    hx = (n * (w + gap) - gap) / 2
    mat = {**_poly(rng, ["bamboo_veneer"]), "tint": [float(U(0.85, 1.0))] * 3}
    slats = []
    for i in range(n):
        x0 = -hx + i * (w + gap)
        V, F, _ = box((x0, -hy, 0.0), (x0 + w, hy, t))
        slats.append((V, F, _textured(rng, V, mat, along_y=True)))
    V, F, UV = _merge(slats)
    return _rec([_part("slats", V, F, UV, mat)], t, t, _accept("rect", (hx - 0.01, hy - 0.01), inset), [hx, hy])


def _tray(r_obj, rng, inset):
    """A wooden serving tray (rounded rectangle, 2.5-4 cm rim) or a metal one (stainless / black, round or
    rectangular, a thin 1.5-2.5 cm rim)."""
    U = rng.uniform
    need = r_obj + FINGER_ROOM + 0.02
    if rng.random() < 0.6:
        hx, hy = max(U(0.17, 0.23), need), max(U(0.12, 0.16), need)
        wall, floor, top, rr = U(0.008, 0.012), U(0.006, 0.01), U(0.025, 0.04), U(0.01, 0.03)
        mat = _poly(rng, WOODS)
        V, F, _ = ring(rounded_rect(hx, hy, rr), rounded_rect(hx - wall, hy - wall, rr - wall / 2), 0.0, floor, top)
        return _rec([_part("body", V, F, _textured(rng, V, mat), mat)], floor, top,
                    _accept("rect", (hx - wall - 0.01, hy - wall - 0.01), inset), [hx, hy])
    wall, floor, top = 0.0025, 0.002, U(0.015, 0.025)
    mat = {"name": "metal", **_pick(rng, METALS)}
    if rng.random() < 0.5:
        r = max(U(0.15, 0.18), need)
        V, F, UV = ring(circle(r, 64), circle(r - wall, 64), 0.0, floor, top)
        return _rec([_part("body", V, F, UV, mat)], floor, top, _accept("circle", (r - wall - 0.01,), inset), [r, r])
    hx, hy, rr = max(U(0.18, 0.22), need), max(U(0.12, 0.15), need), U(0.02, 0.04)
    V, F, UV = ring(rounded_rect(hx, hy, rr), rounded_rect(hx - wall, hy - wall, rr - wall), 0.0, floor, top)
    return _rec([_part("body", V, F, UV, mat)], floor, top, _accept("rect", (hx - wall - 0.01, hy - wall - 0.01),
                                                                     inset), [hx, hy])


def _plate(r_obj, rng, inset, match=None):
    """A glazed ceramic plate turned from a profile: a flat well on a foot ring, the rim rising to 1.3-1.9 cm;
    a side plate (19-23 cm), larger for a wide object up to a dinner plate (28 cm). `match` (the scene graph's
    task.place.match.plate: the photo's own plate, measured) sets its diameter, height, colour (linear rgb) and
    roughness, each varied a few percent per seed: the place target the real robot will see."""
    U = rng.uniform
    if match:
        R, wf = 0.5 * float(match["diameter"]) * U(0.96, 1.04), U(0.62, 0.7)
    else:
        R, wf = U(0.095, 0.115), U(0.6, 0.66)
    R = max(R, 0.9 * r_obj / wf)
    if R > (0.17 if match else 0.14):
        return None
    rw, t, foot = wf * R, U(0.004, 0.005), U(0.005, 0.007)
    zw = foot + t                                  # the well's top
    H = zw + (max(float(match["height"]) - zw, 0.008) * U(0.9, 1.1) if match else U(0.012, 0.018))  # the rim's top
    rf = U(0.5, 0.6) * rw                          # foot ring
    s = np.linspace(0, 1, 8)
    top = [(rw + a * (R - rw), zw + (H - zw) * (1 - (1 - a) ** 2)) for a in s]
    under = [(r, z - t) for r, z in top[::-1]]
    profile = [(0.0, zw), *top, (R + 0.002, H - 0.0025), *under, (rf + 0.004, foot), (rf + 0.002, 0.0),
               (rf - 0.002, 0.0), (rf - 0.004, 0.002), (0.0, 0.002)]
    V, F, UV = lathe(profile)
    if match:
        col = tuple(float(np.clip(c * U(0.93, 1.07), 0, 1)) for c in match["color"])
        mat = {"name": match.get("name", "plate"), "color": col, "roughness": float(match.get("roughness", 0.3))}
    else:
        mat = {"name": "glaze", "color": _pick(rng, GLAZES), "roughness": U(0.12, 0.35)}
    return _rec([_part("body", V, F, UV, mat)], zw, H, _accept("circle", (rw,), inset), [R + 0.002, R + 0.002],
                info={"r_flat": rw, "diameter_m": 2 * R})


def _tape_square(r_obj, rng, inset):
    U = rng.uniform
    side = max(U(0.13, 0.18), 2 * r_obj + 0.05) / 2   # inner half side
    w, t = U(0.019, 0.026), 0.0006
    strips = [box((-side - w, side, 0), (side + w, side + w, t)), box((-side - w, -side - w, 0), (side + w, -side, t)),
              box((-side - w, -side, 0), (-side, side, t)), box((side, -side, 0), (side + w, side, t))]
    V, F, UV = _merge(strips)
    mat = {"name": "tape", "color": _pick(rng, TAPE), "roughness": U(0.55, 0.7)}
    return _rec([_part("tape", V, F, UV, mat)], 0.0, t, _accept("rect", (side, side), inset), [side + w, side + w])


def _marker(r_obj, rng, inset):
    """The goal marker: a 3 mm round rubber mat in one orange with an off-white ring, 1.0-1.12x the object's radius
    (so the object fits inside the ring); only its size and a few percent of its shade vary."""
    U = rng.uniform
    r, t = max(U(1.0, 1.12) * r_obj, 0.07), 0.003
    V, F, UV = slab(circle(r, 64), 0.0, t)
    shade = U(0.94, 1.06)
    mat = {"name": "rubber", "color": tuple(min(c * shade, 1.0) for c in MARKER_COLOR), "roughness": U(0.6, 0.75)}
    rw, n = max(0.006, 0.05 * r), 64
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ro, ri, zr = 0.8 * r, 0.8 * r - rw, t + 0.0004  # a printed band just above the top
    Vr = np.r_[np.c_[ro * np.cos(a), ro * np.sin(a), np.full(n, zr)], np.c_[ri * np.cos(a), ri * np.sin(a), np.full(n, zr)]]
    Fr = np.array([f for i in range(n) for f in ([i, (i + 1) % n, n + (i + 1) % n], [i, n + (i + 1) % n, n + i])])
    UVr = Vr[:, :2] / 0.3
    ring_mat = {"name": "print", "color": MARKER_RING, "roughness": 0.6}
    return _rec([_part("body", V, F, UV, mat), _part("ring", Vr, Fr, UVr, ring_mat)], t, t + 0.0004,
                _accept("circle", (r,), inset), [r, r])


def build_procedural(kind, r_obj, h_obj, rng, match=None):
    """Geometry, materials and place data of a procedural kind sized for an object of radius r_obj, height h_obj;
    `match` the photo's own object of this kind, measured (_plate)."""
    build = {"coaster": _coaster, "placemat": _placemat, "tray": _tray, "plate": _plate, "tape_square": _tape_square,
             "marker": _marker}
    if kind not in build:
        raise KeyError(kind)
    if match and kind == "plate":
        return _plate(r_obj, rng, 0.5 * r_obj, match)
    return build[kind](r_obj, rng, 0.5 * r_obj)


# --- scanned models: measured from a top-down height map ----------------------------------------------------------
def load_scan(source):
    """(V, F) of a library model (polyhaven:<name>) in the receptacle frame at scale 1 (z up, bbox centred, bottom
    at 0)."""
    import trimesh
    name = source.split(":", 1)[1]
    sc = trimesh.load(POLYHAVEN_DIR / "models" / name / f"{name}_1k.gltf")
    m = sc.to_geometry() if hasattr(sc, "to_geometry") else sc
    V = np.asarray(m.vertices)
    V = np.c_[V[:, 0], -V[:, 2], V[:, 1]]                # glTF y-up -> z-up (as sim.assets.add_library_model)
    V = V - [(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()]
    return V, np.asarray(m.faces)


def measure(V, F, cell=0.005, flat_tol=0.006, n=250000, seed=0):
    """Top-down height map of a mesh's surface -> place height (the centre's top: the high points of a woven or
    grained floor, which the object stands on), rim height, and the radius of the largest circle about the centre
    whose cells all lie within flat_tol of the centre's median height (the inside of a bowl / basket, the top of a
    board)."""
    import trimesh
    P, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), n, seed=seed)
    ij = np.floor(P[:, :2] / cell).astype(int)
    lo = ij.min(0) - 1                                  # one empty ring of cells around the footprint
    shape = ij.max(0) - lo + 2
    H = np.full(shape, -np.inf)
    np.maximum.at(H, tuple((ij - lo).T), P[:, 2])
    xs = (np.arange(shape[0]) + lo[0] + 0.5) * cell
    ys = (np.arange(shape[1]) + lo[1] + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = np.hypot(X, Y)
    centre = H[R < 0.02]
    centre = centre[np.isfinite(centre)]
    bad = np.isfinite(H) & (np.abs(H - np.median(centre)) > flat_tol) | ~np.isfinite(H)
    r_flat = float(R[bad].min()) if bad.any() else float(R.max())
    return {"place_z": float(np.percentile(centre, 90)), "rim_z": float(np.max(H[np.isfinite(H)])), "r_flat": r_flat,
            "half": (np.abs(V[:, :2]).max(0)).tolist()}


_SCANS = {}


def scan_data(source):
    if source not in _SCANS:
        V, F = load_scan(source)
        _SCANS[source] = measure(V, F)
    return _SCANS[source]


def build_scan(kind, r_obj, h_obj, rng):
    k = KINDS[kind]
    m = scan_data(k["source"])
    s = float(rng.uniform(0.9, 1.1))
    need = (r_obj + FINGER_ROOM) if k["type"] == "container" else 0.9 * r_obj
    s = max(s, need / max(m["r_flat"], 1e-3))
    if s > 1.35:  # would have to be blown up past a plausible size
        return None
    r_flat = m["r_flat"] * s
    return _rec([], m["place_z"] * s, m["rim_z"] * s, _accept("circle", (r_flat,), 0.5 * r_obj),
                [h * s for h in m["half"]], scale=s, info={"r_flat": r_flat})


# --- sampling --------------------------------------------------------------------------------------------------------
def make(kind, r_obj, h_obj, seed, match=None):
    """The receptacle of this kind and seed sized for the object (deterministic: an episode stores kind + seed and
    rebuilds it), or None when the kind cannot hold it: a flat one must hold most of its base, a container must leave
    room for the fingers and a rim the hand can reach over. `match`: the photo's own object of this kind (scene graph
    task.place.match.<kind>), stored with the summary."""
    k, rng = KINDS[kind], np.random.default_rng(seed)
    d = (build_procedural(kind, r_obj, h_obj, rng, match) if k["source"] == "procedural"
         else build_scan(kind, r_obj, h_obj, rng))
    if d is None or d["rim_z"] - d["place_z"] > max(0.12, 0.9 * h_obj):
        return None
    d["info"] = {**d.get("info", {}), "seed": int(seed), "r_obj": float(r_obj), "h_obj": float(h_obj),
                 **({"match": match} if match else {}),
                 "look": look(d["parts"][0]["material"]) if d["parts"] else k["source"].split(":", 1)[1]}
    return Receptacle(kind=kind, label=k["label"], type=k["type"], source=k["source"],
                      material=d["parts"][0]["material"] if d["parts"] else {}, **d)


def rebuild(d):
    """A Receptacle from its stored summary (to_dict without geometry)."""
    i = d["info"]
    return make(d["kind"], i["r_obj"], i["h_obj"], i["seed"], i.get("match"))


MATCH = {}  # {kind: params}: the scene's own place target as the photo shows it (set_match; task.place.match)


def set_match(match):
    """The scene's measured place targets (scene graph task.place.match, e.g. {"plate": {diameter, height, color,
    roughness}}): every receptacle of such a kind is built to them (sample)."""
    MATCH.clear()
    MATCH.update(match or {})


def sample(r_obj, h_obj, rng, weights=None, kinds=None, match=None):
    """A receptacle of a random kind (weights {kind: w}: the scene graph's task.place.receptacles, chosen per scene)
    that fits an object of footprint radius r_obj and height h_obj; None if no kind fits. `match` {kind: params}:
    the photo's own place target (task.place.match), built to its measurements."""
    w = dict(weights or DEFAULT_WEIGHTS)
    names = [k for k in (kinds or w) if k in KINDS and w.get(k, 0) > 0]
    while names:
        p = np.array([w[k] for k in names], float)
        kind = names[int(rng.choice(len(names), p=p / p.sum()))]
        rec = make(kind, r_obj, h_obj, int(rng.integers(2 ** 31)), (match or MATCH).get(kind))
        if rec is not None:
            return rec
        names.remove(kind)
    return None


def to_world(rec_d, xy_local):
    """Receptacle-frame xy -> world xy for a placed receptacle summary (`origin` world xyz, `yaw` rad)."""
    c, s_ = np.cos(rec_d["yaw"]), np.sin(rec_d["yaw"])
    x, y = np.asarray(xy_local, float)[:2]
    return np.asarray(rec_d["origin"], float)[:2] + [c * x - s_ * y, s_ * x + c * y]


def inside(rec_d, xy_world):
    """Is a world point (the object's base centre) in a placed receptacle's accept region?"""
    c, s_ = np.cos(rec_d["yaw"]), np.sin(rec_d["yaw"])
    d = np.asarray(xy_world, float)[:2] - np.asarray(rec_d["origin"], float)[:2]
    local = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]]) - np.asarray(rec_d["accept_xy"], float)
    a = rec_d["accept"]
    if a[0] == "circle":
        return bool(np.linalg.norm(local) <= a[1])
    return bool(abs(local[0]) <= a[1] and abs(local[1]) <= a[2])


def footprint(rec_d):
    """World corners (4, 2) of a placed receptacle's footprint rectangle."""
    hx, hy = rec_d["half"]
    return np.array([to_world(rec_d, (sx * hx, sy * hy)) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def noun(name):
    """A scene-graph name as the instruction says it: cup_3 -> cup, coffee_mug -> coffee mug."""
    return re.sub(r"[_ ]\d+$", "", name).replace("_", " ")


def task_text(obj, rec, support=None):
    """The episode's instruction for a receptacle (a Receptacle or its summary dict; None: the invisible goal)."""
    o = noun(obj)
    if rec is None:
        return f"place the {o} at the marked spot on the {noun(support or 'table')}"
    d = rec if isinstance(rec, dict) else rec.to_dict()
    if d["kind"] == "tape_square":
        return f"place the {o} inside the taped square"
    if d["type"] == "container":
        return f"put the {o} in the {d['label']}"
    return f"place the {o} on the {d['label']}"
