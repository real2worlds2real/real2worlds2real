"""Role queries on a scene graph (configs/<name>/scene_graph.json) so stages do not depend on part names.

Roles: structure  floor | ceiling | wall | window
       furniture  support_top (the slab objects stand on) | support_body | cabinet | appliance | furniture
An entry's "role" wins; without one it is inferred from the task block and the name.
Observed-ness comes from layout's per-part observed fraction (scene/manifest.json), not from authoring."""
import json
import re

OBSERVED_MIN = 0.3  # a wall with less of its area seen in the photo gets a PBR surface instead of a tint


def dumps(sg, width=100):
    """A scene graph as written by hand: one key per line; short lists of numbers / strings and short dicts of plain
    values on one line."""
    def fmt(o, level):
        flat = json.dumps(o, ensure_ascii=False)
        if not isinstance(o, (dict, list)) or (isinstance(o, list) and len(flat) <= width and "{" not in flat) or \
                (isinstance(o, dict) and len(flat) <= 60 and not any(isinstance(v, (dict, list)) for v in o.values())):
            return flat
        if not o:
            return flat
        pad, inner = " " * level, " " * (level + 1)
        if isinstance(o, dict):
            body = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {fmt(v, level + 1)}" for k, v in o.items()]
            return "{\n" + ",\n".join(body) + f"\n{pad}}}"
        return "[\n" + ",\n".join(inner + fmt(v, level + 1) for v in o) + f"\n{pad}]"
    return fmt(sg, 0) + "\n"


def structure_role(name, entry):
    if entry.get("role"):
        return entry["role"]
    for r in ("floor", "ceiling", "window"):
        if name.startswith(r):
            return r
    return "wall"


CLOSE_TOL = 0.01  # gaps / overlaps narrower than this are seams, not openings
SAME_DEPTH = 0.05  # wall pieces of one side closer than this are one plane


def close_room(structure):
    """The structure with its walls closed around the floor rectangle (a derived view; the file stays as authored).

    Walls bound the room per side (axis x / y, inward normal + / -). A wall keeps the range the photo saw, so a side
    whose pieces stop short of the corners or of each other is open: unseen walls moved by the panorama, pieces at
    different depths (a window bay behind a pillar). Each gap gets a piece on the deeper neighbour's plane and, where
    neighbours differ in depth, a return wall at the nearer one's end. Added pieces ("closure": true) carry the
    neighbours' ids, so the photo textures them where it sees them, the completed panorama elsewhere."""
    st = {k: v for k, v in structure.items() if not v.get("closure")}
    floor = next((v for k, v in st.items() if structure_role(k, v) == "floor"), None)
    if floor is None:
        return {**structure}
    rng = floor["range"]
    out = dict(st)
    sides = {(0, "+"): "left", (0, "-"): "right", (1, "+"): "front", (1, "-"): "back"}
    walls = {k: v for k, v in st.items() if structure_role(k, v) == "wall" and v.get("plane") in ("x", "y")}
    returns = {}  # authored returns: pieces whose whole span lies in the depth band of a stepped perpendicular side
    for k, v in sorted(walls.items(), key=lambda kv: kv[1]["range"][0][1] - kv[1]["range"][0][0]):  # short first
        a, b = v["range"][0]
        for ax2, n2 in sides:
            ds = [float(w["at"]) for k2, w in walls.items() if k2 not in returns and w["plane"] == "xy"[ax2] != v["plane"]
                  and w.get("normal", "+") == n2]
            if len(ds) > 1 and max(ds) - min(ds) > SAME_DEPTH and min(ds) - SAME_DEPTH <= a and b <= max(ds) + SAME_DEPTH:
                returns[k] = (float(v["at"]), a, b)

    def has_return(t, d0, d1):
        return any(abs(at - t) < SAME_DEPTH and min(b, d1) - max(a, d0) > 0.5 * (d1 - d0) for at, a, b in returns.values())

    def close_side(ax, nrm):
        plane, o = "xy"[ax], 1 - ax
        lo, hi = rng[o]
        far = (lambda a, b: a < b) if nrm == "+" else (lambda a, b: a > b)  # a lies further out than b
        pieces = sorted(([float(v["at"]), *map(float, v["range"][0]), v["range"][1], v.get("ids", [])]
                         for k, v in walls.items() if k not in returns and v["plane"] == plane and
                         v.get("normal", "+") == nrm), key=lambda p: p[1])
        bound = rng[ax][0] if nrm == "+" else rng[ax][1]
        z = next((p[3] for p in pieces), [0.0, float(next((v["at"] for k, v in st.items()
                                                            if structure_role(k, v) == "ceiling"), 2.5))])
        added = []

        def wall(pl, at, a, b, zr, ids, normal):
            if b - a > CLOSE_TOL:
                added.append({"role": "wall", "plane": pl, "at": round(at, 3), "range": [[round(a, 3), round(b, 3)],
                              list(zr)], "ids": sorted(set(ids), key=str), "normal": normal, "closure": True})

        def corner(at, end):
            """How far a corner piece at depth `at` runs towards the room's `end` (0: low, 1: high) along the
            side: to the room bound, or to a perpendicular wall in front of it there."""
            ds = [float(w["at"]) for k2, w in walls.items() if k2 not in returns and w["plane"] == "xy"[o] and
                  w.get("normal", "+") == "+-"[end] and w["range"][0][0] - SAME_DEPTH <= at <= w["range"][0][1] + SAME_DEPTH]
            return min(ds + [hi]) if end else max(ds + [lo])

        if not pieces:
            wall(plane, bound, lo, hi, z, [], nrm)
        cur, prev = lo, None  # covered up to cur along the side; prev: the piece ending there
        for p in pieces:
            at, a, b, zr, ids = p
            a, b = max(a, lo), min(b, hi)
            if b <= cur + CLOSE_TOL:
                continue
            if prev is None:  # the corner
                wall(plane, at, corner(at, 0), a, zr, ids, nrm)
            elif abs(at - prev[0]) < SAME_DEPTH:
                wall(plane, at, cur, a, zr, ids + prev[4], nrm)
            else:
                deep = p if far(at, prev[0]) else prev
                if a > cur:  # a gap: the deeper plane runs on to the nearer piece's end
                    wall(plane, deep[0], cur, a, zr, ids + prev[4], nrm)
                t = cur if deep is p else a  # the nearer piece's end
                d0, d1 = sorted([at, prev[0]])
                if not has_return(t, d0, d1):
                    wall("xy"[o], t, d0, d1, zr, ids + prev[4], "+" if deep is p else "-")
            cur, prev = max(cur, b), p
        if prev is not None:
            wall(plane, prev[0], cur, corner(prev[0], 1), prev[3], prev[4], nrm)
        return added

    for (ax, nrm), side in sides.items():
        for k, w in enumerate(close_side(ax, nrm)):
            out[f"wall_closure_{side}_{k + 1}"] = w
    return out


def furniture_role(sg, name, entry):
    if entry.get("role"):
        return entry["role"]
    task = sg.get("task", {})
    if name == task.get("support"):
        return "support_top"
    if name == task.get("support_body"):
        return "support_body"
    if re.search(r"slab|counter_top|_top$", name):
        return "support_top"
    if re.search(r"fridge|refrigerator|oven|dishwasher", name):
        return "appliance"
    if re.search(r"counter|cabinet|upper|soffit|drawer|vanity", name):
        return "cabinet"
    return "furniture"


def roles(sg):
    """{part name: role} over structure and furniture."""
    out = {k: structure_role(k, v) for k, v in sg.get("structure", {}).items()}
    out.update({k: furniture_role(sg, k, v) for k, v in sg.get("furniture", {}).items()})
    return out


def parts(sg, *want):
    r = roles(sg)
    return [k for k, v in r.items() if v in want]


def support_tops(sg):
    """Slabs objects stand on: the task's support plus every other support_top (counter tops of the same kitchen)."""
    tops = parts(sg, "support_top")
    sup = sg.get("task", {}).get("support")
    return ([sup] if sup else []) + [t for t in tops if t != sup]


def walls(sg, manifest=None, observed=None):
    """Wall names, optionally filtered by observed-ness (needs scene/manifest.json)."""
    ws = parts(sg, "wall")
    if observed is None or manifest is None:
        return ws
    frac = {k: v.get("observed_fraction", 0.0) for k, v in manifest["parts"].items()}
    return [w for w in ws if (frac.get(w, 0.0) >= OBSERVED_MIN) == observed]


def movable_groups(sg):
    return sg.get("movable_furniture", {})


def cousin_parts(sg, manifest):
    """The part lists a cousins.json used to hand-author, derived from roles: counter tops (support slabs and the top
    face of floor cabinets at counter height) get PBR, cabinet bodies a paint tint (part overrides match by prefix,
    longest first, so "<box>_top" keeps its PBR), observed walls a tint over the photo albedo, unobserved walls PBR."""
    base_tops = [f"{k}_top" for k in parts(sg, "cabinet")  # floor cabinets carrying a counter: their top face
                 if sg["furniture"][k]["box"][0][2] < 0.05 and 0.8 <= sg["furniture"][k]["box"][1][2] <= 1.05]
    return {"counter_parts": support_tops(sg) + base_tops,
            "paint_parts": parts(sg, "cabinet", "support_body"),
            "observed_wall_parts": walls(sg, manifest, True),
            "unobserved_wall_parts": walls(sg, manifest, False)}


def face_group(part, name):
    """top | legs | basin | sides of a furniture part (recon.layout names its faces <name>_top, <name>_top_a (a top
    cut round a sink), <name>_basin_floor, <name>_leg0_front, ...)."""
    rest = part[len(name) + 1:]
    for group, prefix in (("top", "top"), ("legs", "leg"), ("basin", "basin")):
        if rest.startswith(prefix):
            return group
    return "sides"
