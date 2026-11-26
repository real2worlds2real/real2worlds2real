"""Procedural set pieces for what a single-view reconstruction cannot rebuild: a row of books in a wire rack (one
merged blob of the spines the photo shows), a tray of glasses and coffee pots (glass and polished metal come back as
opaque diffuse shells), a glass-ceramic cooktop (a black patch of the counter texture with stair-stepped edges), a
round pedestal cafe table (furniture is reconstructed as boxes: a round top comes back square), a folding
fabric room divider (a flat wall plane with the photo's folds and whatever stood in front baked in).

    build(kind, params, seed) -> {"parts": [...], "half": [hx, hy], "height": h}

Parts follow assets.receptacles (name, V, F, UV, material) in the kit's frame: origin at the bottom centre on its
support, x along its length, -y its front (the side the photo camera saw). Materials: a Poly Haven texture
{"tex"}, a colour {"color", "roughness", "metallic"} or glass {"glass": True, "color", "ior", "roughness" (frosting),
"thin_walled"}.
Measured values (length, height, colours sampled from the photo) come in `params` (recon.setpieces); the rest
is drawn from the seed within the ranges of the real thing.
"""
import numpy as np

from .receptacles import METALS, WOODS, box, circle, lathe, ring, slab

GLASS = {"name": "glass", "glass": True, "color": (0.95, 0.97, 0.96), "ior": 1.5, "roughness": 0.0}
STAINLESS = {"name": "stainless", **METALS[0], "roughness": 0.18}
WIRE = {"name": "wire", "color": (0.05, 0.05, 0.05), "metallic": 0.5, "roughness": 0.5}
PAGES = {"name": "pages", "color": (0.86, 0.83, 0.74), "roughness": 0.9}


def _part(name, V, F, UV, material):
    return {"name": name, "V": np.asarray(V, float), "F": np.asarray(F, int), "UV": np.asarray(UV, float),
            "material": material}


def _rod(a, b, r=0.002, segs=8):
    from calibrated_scene_reconstruction.assets.templates import tube_mesh

    V, F = tube_mesh([a, b], r, segs)
    return V, F, V[:, :2] / 0.3


def _plate_with_corner_holes(px, py, pt, ins, rh, n=32):
    """A plate px x py, pt thick, with a round through hole (radius rh) at each corner, its middle `ins` in from
    both edges: one closed mesh (V, F, UV) -- its top one surface (pieces side by side showed seams). Each corner a
    square annulus round its hole, the cross between them ear-clipped through the annuli's edge points; the bottom
    the top's mirror; walls along every edge the top has alone (the rim and the holes)."""
    import trimesh

    a = np.linspace(0, 2 * np.pi, n, endpoint=False)  # n a multiple of 8: the squares' corners among the angles
    d = np.c_[np.cos(a), np.sin(a)]
    pts, tris = [], []

    def idx(p):
        pts.append(p)
        return len(pts) - 1
    X, Y = px / 2, py / 2
    q8 = n // 8
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = np.array([sx * (X - ins), sy * (Y - ins)])
            ic = [idx(p) for p in d * rh + c]
            # the square's corners alone (at 45 + 90 k deg, where circle points lie too): each side fanned to its arc
            # from its two corners -- no point on a side, no T-junction with the cross beside it
            sqc = [idx(c + ins * np.array([np.cos(np.radians(45 + 90 * k)), np.sin(np.radians(45 + 90 * k))]) *
                       np.sqrt(2)) for k in range(4)]
            for k in range(4):
                c0, c1 = sqc[k], sqc[(k + 1) % 4]
                arc = [ic[(q8 + 2 * q8 * k + j) % n] for j in range(2 * q8 + 1)]  # from corner k's angle to k+1's
                mid = len(arc) // 2
                tris.append([c0, c1, arc[mid]])
                tris += [[c0, arc[j + 1], arc[j]] for j in range(mid)]
                tris += [[c1, arc[j + 1], arc[j]] for j in range(mid, len(arc) - 1)]
    ax, by = X - 2 * ins, Y - 2 * ins
    corners = [(-ax, -Y), (ax, -Y), (ax, -by), (X, -by), (X, by), (ax, by), (ax, Y), (-ax, Y), (-ax, by), (-X, by),
               (-X, -by), (-ax, -by)]
    P = np.array(corners, float)
    ip = [idx(p) for p in P]
    tris += [[ip[i] for i in f] for f in _ear_clip(P)]
    Q = np.array(pts)
    key = np.round(Q / 1e-7).astype(np.int64)  # the annuli's edge points and the cross's are the same points
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Q2, T = Q[first], inv.ravel()[np.asarray(tris)]
    m = len(Q2)
    V = np.r_[np.c_[Q2, np.full(m, pt)], np.c_[Q2, np.zeros(m)]]
    F = [T, T[:, ::-1] + m]
    edges = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1)
    u, cnt = np.unique(edges, axis=0, return_counts=True)
    for i, j in u[cnt == 1]:
        F.append(np.array([[i, j, j + m], [i, j + m, i + m]]))
    mesh = trimesh.Trimesh(V, np.concatenate(F), process=True)
    mesh.fix_normals()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    return V, F, V[:, :2] / 0.3


def _merge(parts):
    V, F, UV, k = [], [], [], 0
    for v, f, uv in parts:
        V.append(v), F.append(np.asarray(f) + k), UV.append(uv)
        k += len(v)
    return np.concatenate(V), np.concatenate(F), np.concatenate(UV)


def book_row(p, rng):
    """Books standing spine-out along x over `length`, `height` the tallest; `colours` (front to back along x) the
    spine colours the photo shows, else muted cloth and paper colours; `spine_w` the range of book thicknesses. `rack`:
    a base board with a wire frame at both ends and along the back (a cafe's book rack): wooden (`wood`) with black
    wire, or painted metal all over (`rack_color`); else the books stand on the support between two bookends."""
    L, H, D = p["length"], p["height"], p.get("depth", 0.16)
    cols = [tuple(c) for c in p.get("colours") or []]
    palette = [(0.55, 0.15, 0.12), (0.12, 0.22, 0.38), (0.82, 0.78, 0.68), (0.18, 0.30, 0.22), (0.70, 0.55, 0.30),
               (0.25, 0.25, 0.27), (0.85, 0.83, 0.80), (0.45, 0.35, 0.50)]
    rack = p.get("rack", True)
    base_t = 0.018 if rack else 0.0
    parts, x, i = [], -L / 2 + (0.012 if rack else 0.0), 0
    end = L / 2 - (0.012 if rack else 0.0)
    while x < end - 0.01:
        w = min(rng.uniform(*p.get("spine_w", (0.012, 0.032))), end - x)
        h = H * rng.uniform(0.78, 1.0) - base_t
        d = D * rng.uniform(0.82, 1.0)
        c = (cols[min(int((x + w / 2 + L / 2) / L * len(cols)), len(cols) - 1)] if cols
             else palette[rng.integers(len(palette))])
        c = tuple(np.clip(np.asarray(c) * rng.uniform(0.9, 1.1), 0, 1))
        y0 = -D / 2  # spines flush at the front
        V, F, UV = box([x, y0, base_t], [x + w, y0 + d, base_t + h], tile=0.2)
        parts.append(_part(f"book_{i}", V, F, UV, {"name": "cloth", "color": c, "roughness": rng.uniform(0.55, 0.85)}))
        V, F, UV = box([x + 0.0015, y0 + 0.004, base_t + h - 0.002],
                       [x + w - 0.0015, y0 + d - 0.002, base_t + h + 0.0008])
        parts.append(_part(f"pages_{i}", V, F, UV, PAGES))
        if rng.random() < 0.35:  # a paper label band on the spine
            z = base_t + h * rng.uniform(0.62, 0.8)
            V, F, UV = box([x + 0.001, y0 - 0.0006, z], [x + w - 0.001, y0 + 0.002, z + h * rng.uniform(0.06, 0.12)])
            parts.append(_part(f"label_{i}", V, F, UV, {"name": "paper", "color": (0.9, 0.88, 0.82), "roughness": 0.9}))
        x += w + rng.uniform(0.0, 0.002)
        i += 1
    if rack:
        paint = {"name": "painted_steel", "color": tuple(p["rack_color"]), "roughness": 0.45, "metallic": 0.3} \
            if p.get("rack_color") else None
        wood = paint or {"tex": p.get("wood") or WOODS[rng.integers(len(WOODS))]}
        V, F, UV = box([-L / 2, -D / 2 - 0.01, 0.0], [L / 2, D / 2 + 0.01, base_t], tile=1.0)
        parts.append(_part("rack_base", V, F, UV, wood))
        hr, rods = 0.55 * H, []
        for sx in (-1, 1):  # an end loop: two uprights joined at the top
            xe = sx * (L / 2 - 0.006)
            ya, yb = -D / 2 + 0.01, D / 2 - 0.01
            rods += [_rod([xe, ya, base_t], [xe, ya, base_t + hr]), _rod([xe, yb, base_t], [xe, yb, base_t + hr]),
                     _rod([xe, ya, base_t + hr], [xe, yb, base_t + hr])]
        for z in (base_t + 0.4 * hr, base_t + hr):  # back rails
            rods.append(_rod([-L / 2 + 0.006, D / 2 - 0.01, z], [L / 2 - 0.006, D / 2 - 0.01, z]))
        parts.append(_part("rack_wire", *_merge(rods), paint or WIRE))
    return {"parts": parts, "half": [L / 2, D / 2 + (0.01 if rack else 0.0)], "height": H}


def _tumbler(r, h, wall=0.0025, bottom=0.007):
    """A stacking tumbler: tapered from 0.85 r at the foot to r at the rim, a thick glass bottom."""
    rb = 0.85 * r
    prof = [(0.0, 0.0), (rb - 0.002, 0.0), (rb, 0.003), (r, h), (r - wall, h), (rb - wall, bottom), (0.0, bottom)]
    return lathe(prof, n=40, tile=0.2)


def _coffee_pot(r, h, rng):
    """A stainless coffee server: a slightly tapered body, a domed lid with a knob, a spout and a handle."""
    from calibrated_scene_reconstruction.assets.templates import tube_mesh

    rt = r * rng.uniform(0.72, 0.85)
    prof = [(0.0, 0.0), (r, 0.0), (r, 0.01),
            *[(r + (rt - r) * t, 0.01 + (0.82 * h - 0.01) * t) for t in (0.33, 0.66, 1.0)],
            (rt * 0.96, 0.84 * h), (rt * 0.7, 0.92 * h), (0.25 * rt, 0.95 * h), (0.12 * rt, 0.95 * h),
            (0.12 * rt, h), (0.0, h)]
    parts = [lathe(prof, n=48, tile=0.2)]
    zs = 0.6 * h
    V, F = tube_mesh([[-rt * 0.95, 0, zs], [-rt - 0.025, 0, zs + 0.06], [-rt - 0.04, 0, zs + 0.11]], 0.006, 10)
    parts.append((V, F, V[:, :2] / 0.2))
    a = np.linspace(-np.pi / 2, np.pi / 2, 12)
    hr = 0.32 * h
    path = [[rt + 0.003 + 0.04 * np.cos(t), 0, 0.5 * h + hr * np.sin(t)] for t in a]
    V, F = tube_mesh(path, 0.007, 10)
    parts.append((V, F, V[:, :2] / 0.2))
    return _merge(parts)


def tray_set(p, rng):
    """A round metal tray with `glasses` tumblers (some stacked two high, as a cafe stacks them) in front and
    `pots` coffee servers behind them."""
    R = p.get("radius", 0.16)
    wall, floor, top = 0.003, 0.002, 0.018
    V, F, UV = ring(circle(R, 64), circle(R - wall, 64), 0.0, floor, top)
    tray_mat = STAINLESS if p.get("tray", "stainless") == "stainless" else {"name": "black metal", **METALS[1]}
    parts = [_part("tray", V, F, UV, tray_mat)]
    n_pots, n_glass = int(p.get("pots", 2)), int(p.get("glasses", 4))
    rg, hg = rng.uniform(0.033, 0.038), rng.uniform(0.085, 0.1)
    for k in range(n_pots):
        rp, hp = rng.uniform(0.04, 0.05), rng.uniform(0.17, 0.22)
        x = (k - (n_pots - 1) / 2) * 2.3 * rp
        V, F, UV = _coffee_pot(rp, hp, rng)
        yaw = rng.uniform(-0.6, 0.6) + np.pi  # spout toward the front-left
        c, s = np.cos(yaw), np.sin(yaw)
        V = V @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [x, R * 0.42, floor]
        parts.append(_part(f"pot_{k}", V, F, UV, STAINLESS))
    for k in range(n_glass):
        x = (k - (n_glass - 1) / 2) * 2.25 * rg
        y = -R * 0.45 + (0.25 * rg if k % 2 else 0.0)
        stack = 2 if rng.random() < 0.5 else 1
        for j in range(stack):
            V, F, UV = _tumbler(rg, hg)
            V = V + [x, y, floor + j * 0.45 * hg]  # one nested in the other
            parts.append(_part(f"glass_{k}_{j}", V, F, UV, GLASS))
    return {"parts": parts, "half": [R, R], "height": max(top, 0.22)}


def cooktop(p, rng):
    """A glass-ceramic hob set into the counter: a 6 mm black glass panel with a bevelled steel edge and the
    printed rings of its zones."""
    hx, hy = p["length"] / 2, p["depth"] / 2
    parts = []
    V, F, UV = box([-hx, -hy, 0.0], [hx, hy, 0.005])
    parts.append(_part("glass", V, F, UV, {"name": "black glass", "color": (0.015, 0.015, 0.017), "roughness": 0.05}))
    rods = []
    for a, b in [((-hx, -hy), (hx, -hy)), ((hx, -hy), (hx, hy)), ((hx, hy), (-hx, hy)), ((-hx, hy), (-hx, -hy))]:
        rods.append(_rod([*a, 0.0045], [*b, 0.0045], r=0.0018))
    parts.append(_part("frame", *_merge(rods), STAINLESS))
    n = int(p.get("zones", 4))  # 0: a plain panel (the zones print only shows when lit)
    cols = 2 if n >= 4 else max(n, 1)
    rows = int(np.ceil(n / cols))
    from calibrated_scene_reconstruction.assets.templates import tube_mesh

    rings = []
    for k in range(n):  # the printed outline of each zone: a thin circle
        cx = -hx + (k % cols + 0.5) * 2 * hx / cols
        cy = -hy + (k // cols + 0.5) * 2 * hy / rows
        rz = min(hx / cols, hy / rows) * (0.62 if k % 3 else 0.75)
        a = np.linspace(0, 2 * np.pi, 41)
        pts = np.c_[cx + rz * np.cos(a), cy + rz * np.sin(a), np.full(41, 0.0052)]
        V, F = tube_mesh(pts, 0.0007, 6)
        rings.append((V, F, V[:, :2] / 0.3))
    if rings:
        parts.append(_part("zones", *_merge(rings), {"name": "print", "color": (0.09, 0.09, 0.1), "roughness": 0.3}))
    return {"parts": parts, "half": [hx, hy], "height": 0.006}


def pedestal_table(p, rng):
    """A cafe table: a round wooden top (`diameter`, 3 cm thick, a Poly Haven wood at its real size) on a black
    metal column and a four-footed cross base, `height` to the top; `tint` the wood's colour multiplier."""
    from .receptacles import TEX_SIZE

    R, Hh = p["diameter"] / 2, p["height"]
    tex = p.get("top", "wood_table_worn")
    V, F, _ = slab(circle(R, 72), Hh - 0.03, Hh)
    UV = V[:, :2] / TEX_SIZE.get(tex, 1.0)
    parts = [_part("top", V, F, UV, {"tex": tex, "tint": tuple(p.get("tint", (1, 1, 1)))})]
    metal = {"name": "black metal", "color": (0.03, 0.03, 0.03), "metallic": 0.7, "roughness": 0.4}
    col = [lathe([(0.0, 0.03), (0.035, 0.03), (0.035, Hh - 0.03), (0.06, Hh - 0.03), (0.06, Hh - 0.04),
                  (0.0, Hh - 0.04)], n=32, tile=0.3)]
    reach = min(0.85 * R, 0.3)
    for a in (0.25, 0.75, 1.25, 1.75):
        c, s = np.cos(np.pi * a), np.sin(np.pi * a)
        V, F, UV = box([0.0, -0.022, 0.0], [reach, 0.022, 0.03])
        V = V @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        col.append((V, F, UV))
    parts.append(_part("base", *_merge(col), metal))
    return {"parts": parts, "half": [R, R], "height": Hh}


def folding_screen(p, rng):
    """A folding room divider: `panels` fabric panels (default 3) over `length` (the screen's span along x),
    `height` to the fabric's top, standing `clearance` above the floor, zig-zagging `angle` degrees off the line
    (default 15), the fabric gently pleated (`pleat` amplitude, m) between black poles with small feet. Fabric: a
    Poly Haven cloth `tex` (default cotton_jersey) tinted `tint`, or a plain `color`. Each panel is its own part (a
    thin convex collider), double-sided."""
    n = int(p.get("panels", 3))
    L, H = float(p["length"]), float(p.get("height", 1.8))
    z0 = float(p.get("clearance", 0.06))
    ang = np.radians(float(p.get("angle", 15.0)))
    amp = float(p.get("pleat", 0.02))
    c = L / n
    d = 0.5 * c * np.tan(ang)
    J = np.array([[-L / 2 + j * c, (d if j % 2 else -d) * 0.5] for j in range(n + 1)])  # joints in plan (x, y)
    if "color" in p:
        cloth = {"name": "fabric", "color": tuple(p["color"]), "roughness": 0.92}
    else:
        cloth = {"tex": p.get("tex", "cotton_jersey"), "tint": tuple(p.get("tint", (1, 1, 1)))}
    pole = {"name": "black metal", "color": (0.025, 0.025, 0.025), "metallic": 0.6, "roughness": 0.4}
    parts = []
    nu, nz = 64, 12
    for i in range(n):
        a, b = J[i], J[i + 1]
        t_ = b - a
        w = np.linalg.norm(t_)
        nrm = np.array([-t_[1], t_[0]]) / max(w, 1e-9)
        u = np.linspace(0, 1, nu)
        z = np.linspace(z0, H, nz)
        s = u * w
        # soft irregular folds: a few waves of 15-40 cm with random phases (a pressed, regular pleat reads as
        # corrugated sheet), swinging a little wider toward the hem, pinned flat at the poles
        waves = [(rng.uniform(0.15, 0.4), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.0)) for _ in range(3)]
        prof = sum(wt * np.sin(2 * np.pi * s / lam + ph) for lam, ph, wt in waves) / sum(wt for _, _, wt in waves)
        pin = np.clip(np.minimum(s, w - s) / 0.06, 0, 1)
        hem = 1.0 + 0.6 * (1.0 - (z - z0) / max(H - z0, 1e-6))  # wider at the bottom
        V = np.array([[*(a + ui * t_ + amp * hm * pi * pr * nrm), zz]
                      for zz, hm in zip(z, hem) for ui, pr, pi in zip(u, prof, pin)])
        UV = np.array([[si, zz] for zz in z for si in s]) / 0.5
        F = []
        for r in range(nz - 1):
            for q in range(nu - 1):
                k = r * nu + q
                F += [[k, k + 1, k + nu + 1], [k, k + nu + 1, k + nu]]
        F = np.asarray(F)
        Vb = V - 0.002 * np.r_[nrm, 0][None]  # the back side: its own faces, wound the other way
        parts.append(_part(f"panel{i}", np.r_[V, Vb], np.r_[F, F[:, ::-1] + len(V)], np.r_[UV, UV], cloth))
    for j, (x, y) in enumerate(J):
        rods = [_rod([x, y, 0.02], [x, y, H + 0.04], r=0.011, segs=12)]
        for dx, dy in ((0.12, 0.0), (-0.12, 0.0), (0.0, 0.12), (0.0, -0.12)):
            rods.append(_rod([x, y, 0.015], [x + dx, y + dy, 0.015], r=0.009, segs=8))
        parts.append(_part(f"pole{j}", *_merge(rods), pole))
    return {"parts": parts, "half": [L / 2 + 0.12, max(abs(d), 0.12) + 0.02], "height": H + 0.04}


def _plastic(p):
    return {"name": "plastic", "color": tuple(p.get("color", (0.92, 0.92, 0.9))), "roughness": p.get("roughness", 0.5)}


def cylinder(p, rng):
    """A plain cylinder (a peg, a dowel, a can): `diameter`, `length`; `lying` along x (resting on its side, the
    default) or standing along z; its end edges chamfered by `chamfer` (1 mm). Rings every 5 mm along it (one quad
    the cylinder's length long shades in bands)."""
    r, L, lying = p["diameter"] / 2, p["length"], p.get("lying", True)
    ch = min(p.get("chamfer", 0.001), r / 2, L / 4)
    side = [(r, zz) for zz in np.linspace(ch, L - ch, max(2, int(np.ceil((L - 2 * ch) / 0.005)) + 1))]
    V, F, UV = lathe([(0.0, 0.0), (r - ch, 0.0)] + side + [(r - ch, L), (0.0, L)], n=48)
    # its collider: a 24-sided prism, its end caps' corners alone (48 points: PhysX cooks at most 64 per hull and
    # coarsened the 1000-point lathe -- a peg 0.7 mm narrower than its hole stuck on the rim, apps/task_check)
    a_ = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    C = np.array([[r * np.cos(q), r * np.sin(q), zz] for zz in (0.0, L) for q in a_])
    if lying:  # its axis along x, on its side on the support (x and z swapped: a mirror, the faces turned back out)
        V, F = np.c_[V[:, 2] - L / 2, V[:, 1], V[:, 0] + r], F[:, ::-1]
        C = np.c_[C[:, 2] - L / 2, C[:, 1], C[:, 0] + r]
    return {"parts": [_part("body", V, F, UV, _plastic(p))], "half": [L / 2, r] if lying else [r, r],
            "height": 2 * r if lying else L, "collision": [C]}


def cup(p, rng):
    """An open cup or tumbler, tapered: `top_diameter`, `bottom_diameter`, `height`; `wall` (2 mm) and `bottom`
    (3 mm) thick, `rim` a rolled lip's radius (0: none; a paper cup's 1.5 mm)."""
    rt, rb, h = p["top_diameter"] / 2, p["bottom_diameter"] / 2, p["height"]
    w, b, lip = p.get("wall", 0.002), p.get("bottom", 0.003), p.get("rim", 0.0)
    prof = [(0.0, 0.0), (rb, 0.0)]
    if lip > 0:
        prof += [(rt, h - 2 * lip), (rt + lip, h - lip), (rt, h)]
    else:
        prof += [(rt, h)]
    prof += [(rt - w, h), (rb - w, b), (0.0, b)]
    V, F, UV = lathe(prof, n=64)
    # its colliders: the bottom and the wall in 24 sectors -- one hull of the body was a solid cone, and a cup put
    # into it stood on its rim
    n, ro = 24, max(rt, rb) + lip
    coll = [np.array([[r_ * np.cos(a_), r_ * np.sin(a_), z_] for a_ in np.linspace(0, 2 * np.pi, 24, endpoint=False)
                      for r_, z_ in ((rb, 0.0), (rb - w, b))] + [[0.0, 0.0, 0.0], [0.0, 0.0, b]])]
    for k in range(n):
        a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
        pts = []
        for a_ in np.linspace(a0, a1, 4):
            c_, s_ = np.cos(a_), np.sin(a_)
            for r_, z_ in ((rb, 0.0), (rt + lip, h), (rb - w, b), (rt - w, h)):
                pts.append([r_ * c_, r_ * s_, z_])
        coll.append(np.array(pts))
    return {"parts": [_part("body", V, F, UV, _plastic(p))], "half": [ro, ro], "height": h, "collision": coll}


def _rounded_rect(hx, hy, rc, n=64):
    """A rectangle's outline (half sides hx, hy) with its corners rounded by rc, sampled at n equal angles about its
    middle -- a circle sampled the same way pairs with it vertex for vertex (assets.receptacles.ring)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    d = np.c_[np.cos(a), np.sin(a)]
    ax, ay = np.abs(d[:, 0]) + 1e-12, np.abs(d[:, 1]) + 1e-12
    t = np.minimum(hx / ax, hy / ay)  # to the square-cornered outline
    cx, cy = hx - rc, hy - rc
    corner = (t * ax > cx) & (t * ay > cy)
    b = ax * cx + ay * cy  # |s d - (cx, cy)| = rc, the far root
    t = np.where(corner, b + np.sqrt(np.maximum(b * b - (cx * cx + cy * cy - rc * rc), 0.0)), t)
    return d * t[:, None]


def plate_post_hole(p, rng):
    """A peg-in-hole fixture: a base plate (`plate_x`, `plate_y`, `plate_thickness`) with a post on it (`post_x`,
    `post_y`, its vertical edges rounded by `post_corner` -- half the smaller side makes it round --, `post_height`
    above the plate, off the plate's middle by `post_dx`, `post_dy`) and a round hole down the post from its top
    (`hole_diameter`, `hole_depth`). The frames, all from the side, do not see a hole: the task sizes it (a
    peg-in-hole set's hole fits its peg: recon.object_fit). Convex collision pieces: the plate, the post under the
    hole, the wall round the hole in 16 sectors (one hull of the post would fill its hole)."""
    px, py, pt = p["plate_x"], p["plate_y"], p["plate_thickness"]
    hx, hy, hp = p["post_x"] / 2, p["post_y"] / 2, p["post_height"]
    rc = float(np.clip(p.get("post_corner", 0.005), 0.0, min(hx, hy)))
    o = np.array([p.get("post_dx", 0.0), p.get("post_dy", 0.0)])
    rh = min(p["hole_diameter"] / 2, min(hx, hy) - 0.002)
    top = pt + hp
    floor = top - min(p["hole_depth"], hp - 0.002)
    cut = p.get("post_corner_cut")  # {"radius"}: each corner of the post cut back by a quarter disc about the plate's
    n = 128 if cut else 64  # corner hole below it (room for a screw head) -- a concave corner, not a rounded one
    outer = _rounded_rect(hx, hy, rc, n) + o
    if cut and p.get("plate_holes"):
        ins = float(p["plate_holes"]["inset"])
        cs = np.array([[sx * (px / 2 - ins), sy * (py / 2 - ins)] for sx in (-1, 1) for sy in (-1, 1)])
        R = float(cut["radius"])
        d_ = (outer - o) / np.linalg.norm(outer - o, axis=1, keepdims=True)
        t = np.linalg.norm(outer - o, axis=1)
        for c_ in cs:  # along each ray from the post's middle, the near edge of a corner disc when it comes first
            w = c_ - o
            b = d_ @ w
            disc = b * b - (w @ w - R * R)
            t0 = b - np.sqrt(np.maximum(disc, 0.0))
            t = np.where((disc > 0) & (t0 > 0) & (t0 < t), t0, t)
        outer = d_ * t[:, None] + o
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.c_[rh * np.cos(a), rh * np.sin(a)] + o
    mat = _plastic(p)
    holes = p.get("plate_holes")  # {"diameter", "inset"}: a through hole at each corner, its middle `inset` in
    if holes:  # from both edges (the agent's, from the close-ups: the fit does not measure them)
        plate = _plate_with_corner_holes(px, py, pt, float(holes["inset"]), float(holes["diameter"]) / 2)
    else:
        plate = box([-px / 2, -py / 2, 0.0], [px / 2, py / 2, pt])
    post = ring(outer, inner, pt, floor, top)
    hulls = [np.array([[x, y, zz] for x in (-px / 2, px / 2) for y in (-py / 2, py / 2) for zz in (0.0, pt)]),
             np.r_[np.c_[outer, np.full(n, pt)], np.c_[outer, np.full(n, floor)]]]
    step = n // 16
    for k in range(16):
        idx = [(k * step + j) % n for j in range(step + 1)]
        hulls.append(np.r_[np.c_[outer[idx], np.full(len(idx), floor)], np.c_[outer[idx], np.full(len(idx), top)],
                           np.c_[inner[idx], np.full(len(idx), floor)], np.c_[inner[idx], np.full(len(idx), top)]])
    return {"parts": [_part("plate", *plate, mat), _part("post", *post, mat)], "half": [px / 2, py / 2],
            "height": top, "collision": hulls}


def box_frame(p, rng):
    """A rig of square profile standing on its support (a camera frame over a desk): four posts at the corners of
    `length` x `depth` (outer), `height` tall, top rails round the top and bottom bands along the sides named in
    `bands` ("front" -y, "back" +y, "left" -x, "right" +x), all of `profile` (4 cm) section, black anodised. Each
    piece a box: as a set piece its colliders, as the rig's measure (recon.object_fit) its silhouette."""
    L, D, H, w = p["length"], p["depth"], p["height"], p.get("profile", 0.04)
    x0, x1, y0, y1 = -L / 2, L / 2, -D / 2, D / 2
    boxes = {}
    for nx, x in (("left", x0), ("right", x1 - w)):
        for ny, y in (("front", y0), ("back", y1 - w)):
            boxes[f"post_{ny}_{nx}"] = ([x, y, 0.0], [x + w, y + w, H])
    if p.get("rails", True):  # the top rails (off: a top no frame saw -- the posts end where the frames last show them)
        boxes["rail_front"] = ([x0, y0, H - w], [x1, y0 + w, H])
        boxes["rail_back"] = ([x0, y1 - w, H - w], [x1, y1, H])
        boxes["rail_left"] = ([x0, y0, H - w], [x0 + w, y1, H])
        boxes["rail_right"] = ([x1 - w, y0, H - w], [x1, y1, H])
    sides = {"front": ([x0, y0, 0.0], [x1, y0 + w, w]), "back": ([x0, y1 - w, 0.0], [x1, y1, w]),
             "left": ([x0, y0, 0.0], [x0 + w, y1, w]), "right": ([x1 - w, y0, 0.0], [x1, y1, w])}
    for sd in p.get("bands", ["front", "left", "right"]):
        boxes[f"band_{sd}"] = sides[sd]
    mat = {"name": "anodised", "color": tuple(p.get("color", (0.03, 0.03, 0.035))), "roughness": 0.45,
           "metallic": 0.6}
    parts = [_part(n, *box(lo, hi), mat) for n, (lo, hi) in boxes.items()]
    return {"parts": parts, "half": [L / 2, D / 2], "height": H, "boxes": boxes}


def _ear_clip(P):
    """Triangles of a simple polygon (counter-clockwise), by ear clipping."""
    idx, F = list(range(len(P))), []

    def inside(p, a, b, c):
        d = [np.cross(b - a, p - a), np.cross(c - b, p - b), np.cross(a - c, p - c)]
        return min(d) >= 0

    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if np.cross(b - a, c - b) <= 1e-12:
                continue  # a reflex corner
            if any(inside(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            F.append([i0, i1, i2])
            idx.pop(k)
            break
        else:  # degenerate: fan the rest
            F += [[idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1)]
            idx = idx[:3]
    F.append(idx)
    return np.array(F)


def decal(p, rng):
    """A flat mark on its support (tape, a label, a mat): `outlines` (polygons [[x, y], ...] about the piece's
    origin, as recon.support_marks carved them), `color`, `thickness` (0.4 mm). No collider: a thing on it stands
    on the support."""
    t, parts = p.get("thickness", 0.0004), []
    for k, out in enumerate(p["outlines"]):
        P = np.asarray(out, float)
        if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
            P = P[::-1]  # counter-clockwise
        n = len(P)
        T = _ear_clip(P)
        V = np.r_[np.c_[P, np.zeros(n)], np.c_[P, np.full(n, t)]]
        F = [T[:, ::-1], T + n]  # bottom, top
        for i in range(n):
            j = (i + 1) % n
            F.append([[i, j, n + j], [i, n + j, n + i]])
        parts.append(_part(f"mark{k}", V, np.concatenate(F), V[:, :2] / 0.1,
                           {"name": "mark", "color": tuple(p.get("color", (0.1, 0.4, 0.85))),
                            "roughness": p.get("roughness", 0.6)}))
    allP = np.concatenate([np.asarray(o, float) for o in p["outlines"]])
    return {"parts": parts, "half": np.abs(allP).max(0).tolist(), "height": t, "collide": False}


def constrain(kind, p):
    """A kit's parameters made consistent (recon.object_fit's search may wander off): a fixture's post stands on its
    plate, a cup's foot is no wider than its rim."""
    p = dict(p)
    if kind == "plate_post_hole":
        p["post_x"] = min(p["post_x"], p["plate_x"] - 2 * abs(p.get("post_dx", 0.0)) - 0.004)
        p["post_y"] = min(p["post_y"], p["plate_y"] - 2 * abs(p.get("post_dy", 0.0)) - 0.004)
        p["post_x"], p["post_y"] = max(p["post_x"], 0.01), max(p["post_y"], 0.01)
    if kind == "cup":
        p["bottom_diameter"] = min(p["bottom_diameter"], p["top_diameter"])
    if kind == "cylinder" and p.get("lying", True):  # lying: longer than across (else it would stand)
        p["length"] = max(p["length"], p["diameter"])
    return p


# what recon.object_fit fits per kit, (low, high) in metres; a kit's other params stay as given
FIT = {"cylinder": {"diameter": (0.005, 0.10), "length": (0.01, 0.30)},
       "cup": {"top_diameter": (0.03, 0.15), "bottom_diameter": (0.02, 0.15), "height": (0.03, 0.25)},
       "box_frame": {"length": (0.3, 3.0), "depth": (0.3, 2.0), "height": (0.3, 2.0), "profile": (0.02, 0.08)},
       "plate_post_hole": {"plate_x": (0.03, 0.30), "plate_y": (0.03, 0.30), "plate_thickness": (0.002, 0.03),
                           "post_x": (0.015, 0.20), "post_y": (0.015, 0.20), "post_corner": (0.0, 0.05),
                           "post_height": (0.01, 0.20), "post_dx": (-0.05, 0.05), "post_dy": (-0.05, 0.05)}}


def opening(kind, p):
    """Where a kit takes something in (a task check's tool): {xy (its frame), rim_z, depth, radius}, or None for a kit
    with no opening (a session's kit: its module's OPENING)."""
    if kind in _OPENINGS:
        return _OPENINGS[kind](p)
    if kind == "cup":
        return {"xy": [0.0, 0.0], "rim_z": p["height"], "depth": p["height"] - p.get("bottom", 0.003),
                "radius": p["top_diameter"] / 2 - p.get("wall", 0.002)}
    if kind == "plate_post_hole":
        hx, hy, hp = p["post_x"] / 2, p["post_y"] / 2, p["post_height"]
        return {"xy": [p.get("post_dx", 0.0), p.get("post_dy", 0.0)], "rim_z": p["plate_thickness"] + hp,
                "depth": min(p["hole_depth"], hp - 0.002), "radius": min(p["hole_diameter"] / 2, min(hx, hy) - 0.002)}
    return None


def mesh(p, rng):
    """A measured mesh as a set piece (recon.structure_hull: a structure carved from the frames' silhouettes): `path`
    (the repository's) in the piece's frame, one `color`; its collider the triangle mesh itself (`collider`: a
    frame's bars round open space -- a convex hull would fill it)."""
    from pathlib import Path

    import trimesh

    f = Path(p["path"])
    f = f if f.is_absolute() else Path(__file__).resolve().parents[3] / f
    m = trimesh.load(f, force="mesh", process=False)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces, int)
    mat = {"name": "measured", "color": tuple(p.get("color", (0.5, 0.5, 0.5))), "roughness": p.get("roughness", 0.5),
           "metallic": p.get("metallic", 0.0)}
    return {"parts": [_part("body", V, F, np.zeros((len(V), 2)), mat)],
            "half": np.abs(V[:, :2]).max(0).tolist(), "height": float(V[:, 2].max()), "collider": "mesh"}


KITS = {"book_row": book_row, "tray_set": tray_set, "cooktop": cooktop, "pedestal_table": pedestal_table,
        "folding_screen": folding_screen, "cylinder": cylinder, "plate_post_hole": plate_post_hole, "decal": decal,
        "cup": cup, "box_frame": box_frame, "mesh": mesh}
from .office_kits import KITS as _OFFICE  # noqa: E402  (the code-written scene's furniture and shell)

KITS.update(_OFFICE)


_OPENINGS = {}


def use_session(config_dir):
    """The session's own kits (configs/<name>/agent/kits/*.py): each module defines KITS {kind: build(p, rng)},
    FIT {kind: {param: (lo, hi)}} -- the parameters recon.object_fit measures -- and optionally OPENING
    {kind: opening(p)}. A task object no kit here fits is given one of its own (a shape the library lacks), for this
    session alone; they join the library's for the session's stages."""
    import importlib.util
    from calibrated_scene_reconstruction.core.session_code import source_path

    for f in sorted(source_path(config_dir, "kits").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"session_kits_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        KITS.update(getattr(mod, "KITS", {}))
        FIT.update(getattr(mod, "FIT", {}))
        _OPENINGS.update(getattr(mod, "OPENING", {}))


def build(kind, params, seed=0):
    kit = KITS[kind](params, np.random.default_rng(seed))
    V = np.concatenate([np.asarray(p["V"], float) for p in kit["parts"]])
    kit["half"] = (np.ceil(np.maximum(kit["half"], np.abs(V[:, :2]).max(0)) * 1e4) / 1e4).tolist()  # a spout sticks out
    return kit
