"""ABO furniture catalog (Amazon Berkeley Objects, CC BY 4.0, credit: Amazon.com) for retrieval and cousins.

    python -m calibrated_scene_reconstruction.assets.abo [--categories stool bench chair table] [--fetch stool=8 bench=4 table=8]

Reads the ABO listings (Hugging Face parquet mirror of the official metadata), keeps one entry per 3D model of the
requested product types and downloads its catalog image as the retrieval thumbnail. GLBs (2-4k textures, tens of MB)
are fetched lazily by calibrated_scene_reconstruction.assets.library.ensure: only retrieval candidates and the cousin set (--fetch: visually
diverse models per category). Writes assets/abo/{catalog.json, thumbs/, models/}.
"""
import argparse
import json
import re

from calibrated_scene_reconstruction.core.paths import ABO_DIR, rel, resolve
from .download import fetch, fetch_many

HF = "https://huggingface.co/datasets/hyper3labs/amazon-berkeley-objects/resolve/main/data"
CATEGORIES = {"stool": ["STOOL_SEATING"], "bench": ["BENCH", "OTTOMAN"], "chair": ["CHAIR"], "table": ["TABLE"]}
# product types are noisy (STOOL_SEATING also holds dining-chair sets): the title must name the category too
KEYWORDS = {"stool": ["stool", "barstool", "counterstool", "hocker", "taburete", "sgabello", "tabouret", "kruk"],
            "bench": ["bench", "ottoman", "pouf", "bank", "banc", "panca", "banco"],
            "chair": ["chair", "stuhl", "silla", "sedia", "chaise", "sessel"],
            "table": ["table", "tisch", "mesa", "tavolo", "tafel", "bistro"]}
EXCLUDE = {"stool": ["chair", "stuhl", "silla", "sedia", "chaise"], "bench": ["chair", "table"], "chair": ["stool"],
           "table": ["chair", "lamp", "cloth", "runner", "cover", "mat", "set", "desk", "nightstand", "console", "tv"]}
# plausible natural heights (m) per category: drops miniatures, mis-scaled models and bar tables
HEIGHTS = {"stool": (0.4, 0.95), "bench": (0.3, 0.75), "chair": (0.7, 1.2), "table": (0.6, 1.15)}


def load_listings(n_shards=16):
    import pyarrow.parquet as pq

    rows = []
    for i in range(n_shards):
        f = fetch(f"{HF}/listings/listings-{i:05d}-of-{n_shards:05d}.parquet", ABO_DIR / "meta" / f"listings_{i:02d}.parquet",
                  workers=24, chunk=1 << 18)
        rows += pq.read_table(f, columns=["title", "product_type", "main_image_url", "raw_listing_json", "country"]).to_pylist()
    return rows


def load_models():
    import pyarrow.parquet as pq

    f = fetch(f"{HF}/3dmodels/3dmodels.parquet", ABO_DIR / "meta" / "3dmodels.parquet")
    return {r["3dmodel_id"]: r for r in pq.read_table(f).to_pylist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    ap.add_argument("--fetch", nargs="*", default=[], help="category=n: download n more visually diverse models for cousins")
    ap.add_argument("--rebuild", action="store_true", help="rebuild catalog.json (downloads are kept)")
    args = ap.parse_args()
    if args.rebuild or not (ABO_DIR / "catalog.json").exists():
        build_catalog(args.categories)
    if args.fetch:
        from .library import fetch_diverse

        for item in args.fetch:
            cat, n = item.split("=")
            got = fetch_diverse(cat, int(n))
            print(f"{cat}: {len(got)} new models downloaded")


def build_catalog(categories):
    models = load_models()
    want = {pt: c for c in categories for pt in CATEGORIES[c]}
    catalog = {}
    for r in load_listings():
        cat = want.get(r["product_type"])
        mid = json.loads(r["raw_listing_json"]).get("3dmodel_id")
        if not cat or not mid or mid not in models or not r["main_image_url"]:
            continue
        words = set(re.findall(r"[a-zà-ÿ]+", r["title"].lower()))
        words |= {w[:-1] for w in words if w.endswith("s")} | {w[:-2] for w in words if w.endswith("es")}  # plurals
        if not words & set(KEYWORDS[cat]) or words & set(EXCLUDE[cat]):
            continue
        m = models[mid]
        size = [float(m["extent_x"]), float(m["extent_z"]), float(m["extent_y"])]  # glTF y-up -> z-up
        lo, hi = HEIGHTS[cat]
        if not lo <= size[2] <= hi:
            continue
        name = f"abo_{mid}"
        if name in catalog and r["country"] != "US":  # same model listed in several marketplaces: keep English title
            continue
        catalog[name] = {"source": "abo", "category": cat, "title": r["title"], "size": size,
                         "url": m["model_url"], "thumb_url": r["main_image_url"],
                         "thumb": rel(ABO_DIR / "thumbs" / f"{mid}.jpg"), "file": rel(ABO_DIR / "models" / f"{mid}.glb")}
    ok = fetch_many([(c["thumb_url"], resolve(c["thumb"])) for c in catalog.values()], workers=16)
    catalog = {k: c for (k, c), f in zip(catalog.items(), ok) if f}
    (ABO_DIR / "catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False))
    for cat in categories:
        print(cat, sum(c["category"] == cat for c in catalog.values()), "models")


if __name__ == "__main__":
    main()
