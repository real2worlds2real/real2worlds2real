"""Procedural office furniture and room shell for a scene written as code (twin.json `background: code`,
recon.code_scene): each piece from its measured box (length x, depth y, height z; origin at its bottom centre, -y
its front) and its colours sampled from the walk's frames, in assets.kits' part format -- so it is a set piece like
any other (sim.setpieces: its parts' convex colliders, its materials).

    desk, cabinet, shelving_unit, partition, cardboard_box, monitor, pc_tower, whiteboard, office_chair, sofa,
    trash_bin, printer, panel

Nothing is a bare block (a room of boxes reads as a mock-up): the shapes are built from rounded boxes (every edge a
quarter-round of its radius, the faces finely divided so they shade smooth), slabs bent along an arc (a chair's seat
and back shell), tubes swept along curves (a chair's star base, a monitor's neck) and turned profiles (castors, a
bin). Materials: a Poly Haven texture at its real size {"tex", "tint"} or a colour {"color", "roughness",
"metallic"}. What the frames do not tell (a drawer's line, a chair's base) is drawn from the kind of thing an office
has.
"""
import numpy as np

from .receptacles import lathe


def _part(name, V, F, UV, material):
    return {"name": name, "V": np.asarray(V, float), "F": np.asarray(F, int), "UV": np.asarray(UV, float),
            "material": material}


def _merge(parts):
    V, F, UV, k = [], [], [], 0
    for v, f, uv in parts:
        V.append(v), F.append(np.asarray(f) + k), UV.append(uv)
        k += len(v)
    return np.concatenate(V), np.concatenate(F), np.concatenate(UV)


def _mat(p, key, default):
    """A part's material from params: {"tex"...} / {"color"...} as given, or a bare colour list."""
    m = p.get(key, default)
    if isinstance(m, (list, tuple)):
        return {"name": key, "color": tuple(m), "roughness": 0.6}
    return {"name": key, **m}


METAL_GREY = {"color": (0.55, 0.56, 0.57), "metallic": 0.6, "roughness": 0.45}
BLACK = {"color": (0.03, 0.03, 0.035), "roughness": 0.5}
ALU = {"color": (0.72, 0.72, 0.72), "metallic": 0.9, "roughness": 0.35}
RUBBER = {"color": (0.02, 0.02, 0.02), "roughness": 0.8}


def _axis(h, r, inner=6, edge=4):
    """Coordinates along one axis of a rounded box of half-extent h: the flat part evenly, the rounded strips
    (width r) along a quarter circle."""
    r = min(r, 0.95 * h)
    strip = (h - r) + r * np.sin(np.linspace(0, np.pi / 2, edge + 1))[1:]
    return np.r_[-strip[::-1], np.linspace(-(h - r), h - r, inner), strip], r


def rbox(lo, hi, r=0.01, tile=0.3, inner=6, edge=4):
    """A box with every edge and corner rounded by r (the Minkowski sum of a smaller box and a ball), its six faces
    divided into a grid that shares the edges' vertices (smooth shading); planar UVs along its two longest sides."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    r = float(min(r, 0.95 * h.min()))
    ax = [_axis(h[k], r, inner, edge)[0] for k in range(3)]
    V, F = [], []
    key = {}

    def vid(q):
        k = tuple(np.round(q, 7))
        if k not in key:
            inn = np.clip(q, -(h - r), h - r)
            d = q - inn
            n = np.linalg.norm(d)
            key[k] = len(V)
            V.append(c + inn + (r * d / n if n > 1e-12 else 0.0))
        return key[k]

    for a in range(3):
        u, v = [k for k in range(3) if k != a]
        for sgn in (-1, 1):
            U, W = ax[u], ax[v]
            ids = np.empty((len(U), len(W)), int)
            for i, x in enumerate(U):
                for j, y in enumerate(W):
                    q = np.zeros(3)
                    q[a], q[u], q[v] = sgn * h[a], x, y
                    ids[i, j] = vid(q)
            for i in range(len(U) - 1):
                for j in range(len(W) - 1):
                    t1 = [ids[i, j], ids[i + 1, j], ids[i + 1, j + 1]]
                    t2 = [ids[i, j], ids[i + 1, j + 1], ids[i, j + 1]]
                    if (sgn > 0) == (a == 1):  # outward winding
                        t1, t2 = t1[::-1], t2[::-1]
                    F += [t1, t2]
    V = np.asarray(V)
    long2 = np.argsort(hi - lo)[::-1][:2]
    return V, np.asarray(F), V[:, long2] / tile


def bend(V, depth, axis=0, toward=1):
    """Points bent about a vertical axis: y += depth * (x / half-length)^2 (a seat's or a backrest's shell)."""
    V = np.array(V, float)
    x = V[:, axis]
    half = max(np.abs(x).max(), 1e-6)
    V[:, 1] += toward * depth * (x / half) ** 2
    return V


def tube(points, r, segs=12):
    from calibrated_scene_reconstruction.assets.templates import tube_mesh

    V, F = tube_mesh(points, r, segs)
    return V, F, V[:, :2] / 0.3


def disc(r, z0, z1, n=24):
    """A short upright cylinder (a castor, a foot) as a turned profile."""
    return lathe([(0.0, z0), (r, z0), (r, z1), (0.0, z1)], n=n, tile=0.3)


def _place(V, F, UV, dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    V = np.asarray(V, float) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [dx, dy, dz]
    return V, F, UV


def desk(p, rng):
    """A desk or workbench: a 25 mm top with rounded edges over two T-legs and a cross beam (or `ends: panel`: two
    end panels and a modesty panel, a workstation's)."""
    L, D, H = p["length"], p["depth"], p["height"]
    t = p.get("top_thickness", 0.025)
    parts = [_part("top", *rbox([-L / 2, -D / 2, H - t], [L / 2, D / 2, H], 0.008, 0.6, inner=4),
                   _mat(p, "top", {"tex": "proc_laminate_white"}))]
    frame = []
    if p.get("ends") == "panel":
        for x in (-L / 2 + 0.01, L / 2 - 0.035):
            frame.append(rbox([x, -D / 2 + 0.03, 0], [x + 0.025, D / 2 - 0.03, H - t], 0.006))
        frame.append(rbox([-L / 2 + 0.04, D / 2 - 0.08, 0.3], [L / 2 - 0.04, D / 2 - 0.06, H - t - 0.02], 0.004))
    else:
        for x in (-L / 2 + 0.08, L / 2 - 0.08):
            frame.append(rbox([x - 0.03, -0.035, 0.03], [x + 0.03, 0.035, H - t - 0.03], 0.01))   # the column
            frame.append(rbox([x - 0.03, -D / 2 + 0.05, 0.0], [x + 0.03, D / 2 - 0.05, 0.035], 0.012))  # the foot
            frame.append(rbox([x - 0.025, -D / 2 + 0.06, H - t - 0.04], [x + 0.025, D / 2 - 0.06, H - t], 0.008))
        frame.append(rbox([-L / 2 + 0.1, -0.025, H - t - 0.09], [L / 2 - 0.1, 0.025, H - t - 0.05], 0.008))
    parts.append(_part("frame", *_merge(frame), _mat(p, "frame", METAL_GREY)))
    return {"parts": parts, "half": [L / 2, D / 2], "height": H}


def cabinet(p, rng):
    """A cabinet or credenza: its carcass, a door (or drawer) front every ~0.45 m with a 3 mm reveal and a bar pull,
    on a recessed plinth; a top of its own material if given."""
    L, D, H = p["length"], p["depth"], p["height"]
    body = _mat(p, "body", {"color": (0.9, 0.9, 0.88), "roughness": 0.5})
    parts = [_part("carcass", *rbox([-L / 2, -D / 2 + 0.02, 0.06], [L / 2, D / 2, H], 0.004), body)]
    n = max(1, int(round(L / 0.45)))
    fronts, pulls = [], []
    for i in range(n):
        x0 = -L / 2 + i * L / n + 0.0015
        x1 = x0 + L / n - 0.003
        fronts.append(rbox([x0, -D / 2, 0.065], [x1, -D / 2 + 0.019, H - 0.003], 0.003, inner=3))
        hx = x1 - 0.045 if i % 2 == 0 else x0 + 0.045
        zt, zb = H - 0.09, H - 0.25
        pulls.append(tube([[hx, -D / 2 + 0.005, zb], [hx, -D / 2 - 0.025, zb], [hx, -D / 2 - 0.025, zt],
                           [hx, -D / 2 + 0.005, zt]], 0.006, 10))
    parts.append(_part("fronts", *_merge(fronts), body))
    parts.append(_part("pulls", *_merge(pulls), ALU))
    parts.append(_part("plinth", *rbox([-L / 2 + 0.01, -D / 2 + 0.05, 0.0], [L / 2 - 0.01, D / 2 - 0.01, 0.065], 0.003),
                       {"name": "plinth", "color": (0.08, 0.08, 0.08), "roughness": 0.6}))
    if p.get("top"):
        parts.append(_part("top", *rbox([-L / 2, -D / 2 - 0.01, H], [L / 2, D / 2, H + 0.025], 0.006, 0.6),
                           _mat(p, "top", {})))
    return {"parts": parts, "half": [L / 2, D / 2 + 0.03], "height": H + (0.025 if p.get("top") else 0)}


def shelving_unit(p, rng):
    """Open shelving: rounded uprights, a back, a bottom, `shelves` shelves evenly up to its top (a wood top if
    given), a divider every 0.9 m."""
    L, D, H = p["length"], p["depth"], p["height"]
    s = p.get("shelves", 2)
    frame = _mat(p, "frame", METAL_GREY)
    pieces = [rbox([-L / 2, -D / 2, 0], [-L / 2 + 0.022, D / 2, H], 0.006),
              rbox([L / 2 - 0.022, -D / 2, 0], [L / 2, D / 2, H], 0.006),
              rbox([-L / 2 + 0.02, D / 2 - 0.012, 0.05], [L / 2 - 0.02, D / 2, H], 0.003, inner=3)]
    for k in range(s + 1):
        z = 0.06 + k * (H - 0.08) / s
        pieces.append(rbox([-L / 2 + 0.02, -D / 2 + 0.005, z], [L / 2 - 0.02, D / 2 - 0.012, z + 0.022], 0.005))
    for x in np.arange(-L / 2 + 0.9, L / 2 - 0.1, 0.9):
        pieces.append(rbox([x - 0.011, -D / 2, 0], [x + 0.011, D / 2 - 0.012, H], 0.005))
    parts = [_part("frame", *_merge(pieces), frame)]
    if p.get("top"):
        parts.append(_part("top", *rbox([-L / 2 - 0.01, -D / 2 - 0.01, H], [L / 2 + 0.01, D / 2, H + 0.03], 0.008, 0.6),
                           _mat(p, "top", {})))
    return {"parts": parts, "half": [L / 2, D / 2], "height": H + (0.03 if p.get("top") else 0)}


def partition(p, rng):
    """A cubicle partition / screen: a fabric or laminate face (`face`), slightly cushioned, in a rounded aluminium
    frame, on two feet."""
    L, H = p["length"], p["height"]
    T = p.get("thickness", 0.05)
    parts = [_part("face", *rbox([-L / 2 + 0.025, -T / 2 + 0.004, 0.06], [L / 2 - 0.025, T / 2 - 0.004, H - 0.025],
                                 0.012, 0.6), _mat(p, "face", {"color": (0.85, 0.7, 0.3), "roughness": 0.9}))]
    r = 0.012
    fr = [tube([[-L / 2 + 0.013, 0, 0.05], [-L / 2 + 0.013, 0, H - 0.013], [L / 2 - 0.013, 0, H - 0.013],
                [L / 2 - 0.013, 0, 0.05]], 0.013 if T > 0.03 else 0.008, 12),
          rbox([-L / 2 + 0.03, -T / 2, 0.04], [L / 2 - 0.03, T / 2, 0.065], r)]
    for x in (-L / 2 + 0.013, L / 2 - 0.013):
        fr.append(rbox([x - 0.02, -0.2, 0.0], [x + 0.02, 0.2, 0.05], 0.015))
    parts.append(_part("frame", *_merge(fr), _mat(p, "frame", ALU)))
    return {"parts": parts, "half": [L / 2, 0.2], "height": H}


def cardboard_box(p, rng):
    """A corrugated shipping box: kraft brown (or its sampled colour) with softened edges and slightly bowed sides,
    its top flaps meeting in a seam under a strip of tape."""
    L, D, H = p["length"], p["depth"], p["height"]
    kraft = _mat(p, "color", {"color": (0.62, 0.45, 0.28), "roughness": 0.85})
    V, F, UV = rbox([-L / 2, -D / 2, 0], [L / 2, D / 2, H], 0.006, 0.4)
    bulge = 0.006 * (1 - (2 * V[:, 2] / max(H, 1e-6) - 1) ** 2)  # the sides bow out a little at mid-height
    side = np.c_[V[:, 0] / (L / 2), V[:, 1] / (D / 2)]
    V[:, :2] += side / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-6) * bulge[:, None]
    parts = [_part("box", V, F, UV, kraft)]
    tw = min(0.048, D / 4)
    seam = rbox([-L / 2 + 0.002, -0.001, H - 0.0005], [L / 2 - 0.002, 0.001, H + 0.0005], 0.0004, inner=2)
    tape = rbox([-L / 2 - 0.002, -tw / 2, H], [L / 2 + 0.002, tw / 2, H + 0.0012], 0.0005, inner=2)
    parts.append(_part("tape", *_merge([tape, seam]), {"name": "tape", "color": (0.70, 0.55, 0.36), "roughness": 0.3}))
    return {"parts": parts, "half": [L / 2 + 0.01, D / 2 + 0.01], "height": H + 0.0012}


def monitor(p, rng):
    """A flat monitor: a thin rounded panel `length` wide with its top at `height`, a glossy screen behind a slim
    bezel, a curved neck to an oval foot."""
    L, H = p["length"], p["height"]
    ph = p.get("panel_height", min(0.6 * L, H - 0.1))
    z0 = H - ph
    fw = max(0.11, 0.18 * L)  # the foot's half-width: a wide panel on a narrow foot toppled under PhysX
    parts = [_part("panel", *rbox([-L / 2, -0.012, z0], [L / 2, 0.02, H], 0.008), BLACK),
             _part("screen", *rbox([-L / 2 + 0.008, -0.0135, z0 + 0.012], [L / 2 - 0.008, -0.011, H - 0.008], 0.001, inner=2),
                   {"name": "screen", "color": (0.012, 0.014, 0.018), "roughness": 0.06}),
             _part("stand", *_merge([tube([[0, 0.03, z0 + 0.08], [0, 0.07, z0 * 0.6], [0, 0.06, 0.02]], 0.018, 14),
                                     rbox([-fw, -0.09, 0.0], [fw, 0.15, 0.014], 0.007)]), BLACK)]
    return {"parts": parts, "half": [L / 2, 0.14], "height": H}


def pc_tower(p, rng):
    """A desktop computer case: a black box with rounded vertical edges, a glossy front, four feet."""
    L, D, H = p["length"], p["depth"], p["height"]
    feet = [disc(0.015, 0.0, 0.015) for _ in range(4)]
    feet = [_place(*f, dx=x, dy=y) for f, (x, y) in zip(feet, [(-L / 2 + 0.03, -D / 2 + 0.04), (L / 2 - 0.03, -D / 2 + 0.04),
                                                              (-L / 2 + 0.03, D / 2 - 0.04), (L / 2 - 0.03, D / 2 - 0.04)])]
    parts = [_part("case", *rbox([-L / 2, -D / 2 + 0.008, 0.015], [L / 2, D / 2, H], 0.01), _mat(p, "color", BLACK)),
             _part("front", *rbox([-L / 2 + 0.004, -D / 2, 0.02], [L / 2 - 0.004, -D / 2 + 0.01, H - 0.005], 0.004),
                   {"name": "front", "color": (0.06, 0.06, 0.07), "roughness": 0.15}),
             _part("feet", *_merge(feet), RUBBER)]
    return {"parts": parts, "half": [L / 2, D / 2], "height": H}


def whiteboard(p, rng):
    """A wall whiteboard: a white glossy board in a rounded aluminium frame, 2 cm proud of its wall, its bottom at
    `bottom` (m over the floor, the part's origin on the floor), a pen tray along its foot."""
    L, H, z0 = p["length"], p["height"], p.get("bottom", 0.9)
    parts = [_part("board", *rbox([-L / 2 + 0.012, -0.018, z0 + 0.012], [L / 2 - 0.012, 0.0, z0 + H - 0.012], 0.002, inner=3),
                   {"name": "board", "color": (0.93, 0.93, 0.92), "roughness": 0.12}),
             _part("frame", *_merge([tube([[-L / 2 + 0.01, -0.012, z0 + 0.01], [-L / 2 + 0.01, -0.012, z0 + H - 0.01],
                                           [L / 2 - 0.01, -0.012, z0 + H - 0.01], [L / 2 - 0.01, -0.012, z0 + 0.01],
                                           [-L / 2 + 0.01, -0.012, z0 + 0.01]], 0.011, 10),
                                     rbox([-L / 2 + 0.05, -0.07, z0 - 0.01], [L / 2 - 0.05, -0.005, z0 + 0.012], 0.006)]),
                   ALU)]
    if p.get("stand"):  # a mobile board: two legs down to castored feet (else it hangs on its wall)
        legs = []
        for x in (-L / 2 + 0.03, L / 2 - 0.03):
            legs.append(tube([[x, 0.012, 0.06], [x, 0.012, z0 + H - 0.02]], 0.014, 10))
            legs.append(rbox([x - 0.025, -0.28, 0.04], [x + 0.025, 0.28, 0.07], 0.012))
            for y in (-0.25, 0.25):
                legs.append(_place(*disc(0.025, 0.0, 0.045, 16), dx=x, dy=y))
        parts.append(_part("stand", *_merge(legs), ALU))
        return {"parts": parts, "half": [L / 2, 0.28], "height": z0 + H}
    return {"parts": parts, "half": [L / 2, 0.07], "height": z0 + H}


def office_chair(p, rng):
    """A task chair: five curved legs on castors, a gas column, a padded seat and a backrest shell bent to the
    sitter, armrests on loops (`arms`, default on), the upholstery in `color`."""
    H = p.get("height", 0.95)
    sh = p.get("seat_height", 0.47)
    up = _mat(p, "color", {"color": (0.08, 0.09, 0.14), "roughness": 0.85})
    base = []
    for k in range(5):
        a = 2 * np.pi * k / 5 + 0.3
        d = np.array([np.cos(a), np.sin(a), 0.0])
        pts = [np.array([0, 0, 0.11]) + d * 0.04, np.array([0, 0, 0.1]) + d * 0.18, np.array([0, 0, 0.075]) + d * 0.3]
        base.append(tube(pts, 0.016, 10))
        base.append(_place(*disc(0.025, 0.0, 0.05, 16), dx=d[0] * 0.31, dy=d[1] * 0.31))
    base.append(_place(*disc(0.05, 0.08, 0.13, 24)))
    base.append(_place(*disc(0.022, 0.13, sh - 0.08, 20)))
    seat = rbox([-0.245, -0.24, sh - 0.075], [0.245, 0.22, sh], 0.035, inner=6, edge=5)
    seat = (bend(seat[0], 0.02, toward=-1), seat[1], seat[2])
    back = rbox([-0.22, 0.20, sh + 0.06], [0.22, 0.25, H], 0.03, inner=6, edge=5)
    back = (bend(back[0], -0.05), back[1], back[2])
    parts = [_part("base", *_merge(base), BLACK), _part("seat", *seat, up), _part("back", *back, up),
             _part("spine", *_merge([tube([[0, 0.12, sh - 0.07], [0, 0.23, sh - 0.04], [0, 0.245, sh + 0.12]], 0.02, 12),
                                     rbox([-0.12, -0.12, sh - 0.11], [0.12, 0.12, sh - 0.075], 0.01)]), BLACK)]
    if p.get("arms", True):
        arms = []
        for x in (-0.27, 0.27):
            arms.append(tube([[x * 0.85, 0.05, sh - 0.07], [x, 0.05, sh + 0.05], [x, 0.0, sh + 0.2]], 0.014, 10))
            arms.append(rbox([x - 0.035, -0.12, sh + 0.19], [x + 0.035, 0.12, sh + 0.225], 0.015))
        parts.append(_part("arms", *_merge(arms), BLACK))
    return {"parts": parts, "half": [0.34, 0.34], "height": H}


def sofa(p, rng):
    """A sofa: soft, round-edged cushions (one per ~0.6 m) on a base, a back of cushions, padded arms, tapered
    wooden legs, in its upholstery `color`."""
    L, D, H = p["length"], p["depth"], p["height"]
    up = _mat(p, "color", {"color": (0.6, 0.52, 0.42), "roughness": 0.9})
    arm, back, leg = 0.17, 0.22, 0.12
    n = max(1, int(round((L - 2 * arm) / 0.6)))
    w = (L - 2 * arm) / n
    cush, backs = [], []
    for i in range(n):
        x0 = -L / 2 + arm + i * w
        cush.append(rbox([x0 + 0.004, -D / 2 + 0.01, leg + 0.17], [x0 + w - 0.004, D / 2 - back + 0.02, leg + 0.33],
                         0.05, inner=5, edge=5))
        bc = rbox([x0 + 0.004, D / 2 - back - 0.02, leg + 0.3], [x0 + w - 0.004, D / 2 - 0.05, H - 0.02], 0.06,
                  inner=5, edge=5)
        backs.append(bc)
    parts = [_part("base", *rbox([-L / 2 + 0.02, -D / 2 + 0.02, leg], [L / 2 - 0.02, D / 2, leg + 0.18], 0.04), up),
             _part("cushions", *_merge(cush), up), _part("backs", *_merge(backs), up),
             _part("frame_back", *rbox([-L / 2 + 0.02, D / 2 - 0.08, leg + 0.15], [L / 2 - 0.02, D / 2, H - 0.06], 0.04), up),
             _part("arms", *_merge([rbox([-L / 2, -D / 2, leg + 0.05], [-L / 2 + arm, D / 2, leg + 0.48], 0.07, inner=4, edge=5),
                                    rbox([L / 2 - arm, -D / 2, leg + 0.05], [L / 2, D / 2, leg + 0.48], 0.07, inner=4, edge=5)]),
                   up),
             _part("legs", *_merge([_place(*lathe([(0, 0), (0.016, 0), (0.024, leg), (0, leg)], n=16), dx=x, dy=y)
                                    for x in (-L / 2 + 0.07, L / 2 - 0.07) for y in (-D / 2 + 0.07, D / 2 - 0.07)]),
                   {"name": "legs", "color": (0.16, 0.1, 0.06), "roughness": 0.45})]
    return {"parts": parts, "half": [L / 2, D / 2], "height": H}


def trash_bin(p, rng):
    """A waste bin: a tapered tub with a rolled rim (`color`), open at the top."""
    H = p["height"]
    R = min(p["length"], p["depth"]) / 2
    prof = [(0.0, 0.0), (0.8 * R, 0.0), (0.82 * R, 0.004), (R, H - 0.012), (1.02 * R, H - 0.006), (R, H),
            (0.97 * R, H - 0.005), (0.78 * R, 0.012), (0.0, 0.012)]
    parts = [_part("tub", *lathe(prof, n=40, tile=0.3), _mat(p, "color", {"color": (0.1, 0.12, 0.15), "roughness": 0.5}))]
    return {"parts": parts, "half": [R * 1.02, R * 1.02], "height": H}


def printer(p, rng):
    """An office printer / copier: a white body with rounded edges, a dark scanner lid, a paper tray and a control
    panel."""
    L, D, H = p["length"], p["depth"], p["height"]
    white = _mat(p, "color", {"color": (0.82, 0.82, 0.8), "roughness": 0.4})
    dark = {"name": "lid", "color": (0.12, 0.12, 0.13), "roughness": 0.4}
    parts = [_part("body", *rbox([-L / 2, -D / 2, 0], [L / 2, D / 2, H - 0.06], 0.02), white),
             _part("lid", *rbox([-L / 2 + 0.02, -D / 2 + 0.03, H - 0.065], [L / 2 - 0.02, D / 2 - 0.02, H], 0.012), dark),
             _part("tray", *rbox([-L / 2 + 0.06, -D / 2 - 0.12, H * 0.55], [L / 2 - 0.06, -D / 2 + 0.01, H * 0.55 + 0.012], 0.005),
                   dark),
             _part("panel", *rbox([L / 2 - 0.22, -D / 2 - 0.04, H - 0.11], [L / 2 - 0.02, -D / 2 + 0.02, H - 0.08], 0.008),
                   dark)]
    return {"parts": parts, "half": [L / 2, D / 2 + 0.12], "height": H}


def panel(p, rng):
    """A flat piece of the room's shell: a `length` x `height` slab `thickness` thick standing on its bottom edge
    (a wall, a column face) or, `lying`, a floor or a ceiling `length` x `depth`; its material a texture at its
    real size or a colour; a 10 cm skirting board along a wall's foot (`skirting`: its colour)."""
    from .receptacles import box

    L, T = p["length"], p.get("thickness", 0.05)
    mat = _mat(p, "face", {"tex": "proc_painted_drywall"})
    if p.get("lying"):
        D = p["depth"]
        parts = [_part("face", *box([-L / 2, -D / 2, -T], [L / 2, D / 2, 0.0], p.get("tile", 1.0)), mat)]
        return {"parts": parts, "half": [L / 2, D / 2], "height": 0.0}
    H = p["height"]
    parts = [_part("face", *box([-L / 2, -T / 2, 0.0], [L / 2, T / 2, H], p.get("tile", 1.0)), mat)]
    if p.get("skirting"):
        parts.append(_part("skirting", *rbox([-L / 2, -T / 2 - 0.012, 0.0], [L / 2, -T / 2, 0.10], 0.003, inner=2),
                           _mat(p, "skirting", [0.25, 0.22, 0.2])))
    return {"parts": parts, "half": [L / 2, T / 2 + 0.012], "height": H}


KITS = {"desk": desk, "cabinet": cabinet, "shelving_unit": shelving_unit, "partition": partition,
        "cardboard_box": cardboard_box, "monitor": monitor, "pc_tower": pc_tower, "whiteboard": whiteboard,
        "office_chair": office_chair, "panel": panel, "sofa": sofa, "trash_bin": trash_bin, "printer": printer}
