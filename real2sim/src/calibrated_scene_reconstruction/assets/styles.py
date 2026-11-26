"""Style themes for background augmentation (configs/themes/*.json) and the Poly Haven assets they need.

    python -m calibrated_scene_reconstruction.assets.styles            # download the themes' extra PBR textures + 2k HDRIs into assets/polyhaven

A theme is a coordinated look applied by role, not by part name: support tops (PBR or "color:r,g,b", a plain finish),
floor, walls (PBR, the observed ones too), cabinet paint (tint), movable furniture stain, ceiling tint, and an HDRI for
the sky dome / window views with a light temperature and intensity range. `rooms` lists the room types it belongs in
(cousins.room_themes: a lab scene draws lab themes only). Cousins sample a theme
first, then one option per role inside it, so a cousin reads as one interior style rather than independent swaps.
Index: textures merge into assets/polyhaven/index.json["textures"], HDRIs into ["hdris"] (kept by assets.polyhaven).
"""
import json

from calibrated_scene_reconstruction.core.paths import CONFIGS
from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR as OUT
from calibrated_scene_reconstruction.core.paths import rel
from .polyhaven import MAPS, files, get

THEME_DIR = CONFIGS / "themes"


def themes():
    return {p.stem: json.loads(p.read_text()) for p in sorted(THEME_DIR.glob("*.json"))}


def sample(theme, rng):
    """One concrete look from a theme: {role: choice}."""
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    wall, paint, stain = pick(list(theme["wall_tint"])), pick(list(theme["paint"])), pick(list(theme["stain"]))
    return {"theme": theme["name"], "support": pick(theme["support"]), "floor": pick(theme["floor"]),
            "wall": wall, "wall_tint": theme["wall_tint"][wall], "wall_pbr": pick(theme["wall_pbr"]),
            "paint": paint, "paint_tint": theme["paint"][paint], "stain": stain, "stain_tint": theme["stain"][stain],
            "ceiling_tint": theme["ceiling_tint"], "hdri": pick(theme["hdri"]),
            "light": {"temp": float(rng.uniform(*theme["light"]["temp"])),
                      "scale": float(rng.uniform(*theme["light"]["scale"]))}}


def hdri_path(name):
    idx = json.loads((OUT / "index.json").read_text())
    return idx.get("hdris", {}).get(name)


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    idx = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
    tex, hdris = idx.setdefault("textures", {}), idx.setdefault("hdris", {})
    need_tex = {(n, role) for t in themes().values() for role, key in (("counter", "support"), ("floor", "floor"),
                                                                        ("wall", "wall_pbr")) for n in t[key]
                if not str(n).startswith("color:")}  # plain finishes need no texture
    for name, role in sorted(need_tex):
        if name in tex:
            continue
        if name.startswith("proc_"):  # generated here, not downloaded (assets.proc_textures)
            from .proc_textures import write

            d = write(name)
            tex[name] = {"role": role, **{k: rel(d / f"{k}.jpg") for k in ("diff", "nor_gl", "rough")}}
            continue
        try:
            f = files(name)
        except Exception as e:  # renamed / missing upstream
            print("skip texture", name, e)
            continue
        entry = {"role": role}
        for key, short in MAPS.items():
            if key in f and "2k" in f[key]:
                dst = OUT / "textures" / name / f"{short}.jpg"
                get(f[key]["2k"]["jpg"]["url"], dst)
                entry[short] = rel(dst)
        tex[name] = entry
        print("texture", name, sorted(k for k in entry if k != "role"))
    for name in sorted({h for t in themes().values() for h in t["hdri"]}):
        if name in hdris:
            continue
        try:
            f = files(name)["hdri"]["2k"]["hdr"]
        except Exception as e:
            print("skip hdri", name, e)
            continue
        dst = OUT / "hdris" / f"{name}_2k.hdr"
        get(f["url"], dst)
        hdris[name] = rel(dst)
        print("hdri", name)
    (OUT / "index.json").write_text(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
