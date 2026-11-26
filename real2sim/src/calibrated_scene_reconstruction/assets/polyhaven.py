"""Download the CC0 Poly Haven PBR textures and glTF models used for unobserved surfaces and cousins.

    python -m calibrated_scene_reconstruction.assets.polyhaven   (then: tools/run_isaac.sh apps/gltf_to_usd.py)
"""
import json
import urllib.request
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import POLYHAVEN_DIR as OUT
from calibrated_scene_reconstruction.core.paths import rel

TEXTURES = {
    # counter tops
    "marble_01": "counter", "grey_cartago_01": "counter", "brushed_concrete": "counter",
    "american_walnut_veneer": "counter", "ash_veneer": "counter", "cherry_veneer": "counter",
    "black_walnut_veneer_01": "counter", "terrazzo_tiles": "counter",
    # walls / painted panels
    "beige_wall_001": "wall", "beige_wall_002": "wall", "clay_plaster": "wall", "painted_plaster_wall": "wall",
    "white_plaster_02": "wall",
    # floors
    "diagonal_parquet": "floor", "brown_planks_05": "floor", "dark_wooden_planks": "floor",
    "brown_floor_tiles": "floor", "laminate_floor_02": "floor", "wood_floor": "floor",
    # place targets (assets.receptacles): placemat fabrics, bamboo slats
    "hessian_230": "receptacle", "hessian_380": "receptacle", "rough_linen": "receptacle",
    "cotton_jersey": "receptacle", "poly_wool_herringbone": "receptacle", "waffle_pique_cotton": "receptacle",
    "gingham_check": "receptacle", "bamboo_veneer": "receptacle",
    # furniture looks the object review gives a table or a counter (sim.setpieces), wooden set pieces (assets.kits)
    "wood_table": "furniture", "wood_table_large": "furniture", "wood_planks_dirt": "furniture",
    "wood_table_worn": "furniture", "plywood": "furniture", "kitchen_wood": "furniture",
}
MODELS = [
    "wooden_stool_01", "wooden_stool_02", "painted_wooden_stool", "bar_chair_round_01", "metal_stool_01",
    "ceramic_vase_01", "ceramic_vase_02", "ceramic_vase_03", "antique_ceramic_vase_01", "jug_01", "metal_jug",
    "wooden_bowl_01", "wooden_bowl_02", "food_apple_01", "food_pears_asian_01", "lemon", "bananas",
    "potted_plant_01", "potted_plant_02", "wicker_basket_01", "wicker_basket_02", "standing_picture_frame_01",
    "standing_picture_frame_02", "vintage_electric_kettle", "wooden_cutting_board", "brass_pot_01", "pot_enamel_01",
]
# library categories (retrieval fallback and cousin furniture), see calibrated_scene_reconstruction.assets.library
CATEGORIES = {"wooden_stool_01": "stool", "wooden_stool_02": "stool", "painted_wooden_stool": "stool",
              "bar_chair_round_01": "stool", "metal_stool_01": "stool"}
MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough", "AO": "ao"}


def get(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "calibrated_scene_reconstruction-datagen/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())


def files(asset):
    req = urllib.request.Request(f"https://api.polyhaven.com/files/{asset}", headers={"User-Agent": "calibrated_scene_reconstruction-datagen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    index = {"textures": {}, "models": {}, "catalog": {}}
    for name, role in TEXTURES.items():
        try:
            f = files(name)
        except Exception as e:  # asset renamed / missing
            print("skip texture", name, e)
            continue
        entry = {"role": role}
        for key, short in MAPS.items():
            if key in f and "2k" in f[key]:
                url = f[key]["2k"]["jpg"]["url"]
                dst = OUT / "textures" / name / f"{short}.jpg"
                get(url, dst)
                entry[short] = rel(dst)
        index["textures"][name] = entry
        print("texture", name, sorted(k for k in entry if k != "role"))
    for name in MODELS:
        try:
            f = files(name)
            g = f["gltf"]["1k"]["gltf"]
        except Exception as e:
            print("skip model", name, e)
            continue
        base = OUT / "models" / name
        model_path = base / Path(g["url"]).name
        get(g["url"], model_path)
        for include_path, inc in g.get("include", {}).items():
            get(inc["url"], base / include_path)
        index["models"][name] = rel(model_path)
        if name in CATEGORIES:
            thumb = OUT / "thumbs" / f"{name}.png"
            get(f"https://cdn.polyhaven.com/asset_img/thumbs/{name}.png?width=512&height=512", thumb)
            index["catalog"][name] = {"category": CATEGORIES[name], "title": name, "thumb": rel(thumb)}
        print("model", name)
    old = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
    index["usd"], index["meta"] = old.get("usd", {}), old.get("meta", {})  # keep conversions (apps/gltf_to_usd.py)
    index["hdris"] = old.get("hdris", {})  # and the style themes' assets (assets.styles)
    index["textures"] = {**{k: v for k, v in old.get("textures", {}).items() if k not in index["textures"]},
                         **index["textures"]}
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
