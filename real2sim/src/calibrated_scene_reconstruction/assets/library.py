"""One furniture / prop library over Poly Haven (CC0) and ABO (CC BY 4.0) models.

catalog():  {name: {source, category, title, size (z-up natural metres), thumb, file, url?}}
ensure():   local glTF/GLB of a model (ABO GLBs are downloaded on first use)
rank():     a category's models ordered by DINOv2 similarity between a photo crop (object on white) and each model's
            catalog image / thumbnail; recon.retrieve re-ranks the top-k by silhouette fit.
usd_index(): converted USD per model + {up, mpu, size} (apps/gltf_to_usd.py), used by sim.assets.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.paths import ABO_DIR, POLYHAVEN_DIR, ROOT, resolve
from .download import fetch

EMB = ROOT / "assets" / "library_embeddings.npz"


def _read(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def _files(c):
    """A catalog entry with its stored paths (relative to the repository) made usable."""
    return {**c, **{k: str(resolve(c[k])) for k in ("file", "thumb") if c.get(k)}}


def catalog():
    out = {}
    ph = _read(POLYHAVEN_DIR / "index.json", {})
    for name, c in ph.get("catalog", {}).items():
        out[name] = _files({"source": "polyhaven", **c, "file": ph["models"][name]})
    out.update({k: _files(c) for k, c in _read(ABO_DIR / "catalog.json", {}).items()})
    return out


def usd_index():
    """{"usd": {name: path}, "meta": {name: {up, mpu, size}}} over every converted model."""
    idx = {"usd": {}, "meta": {}}
    for p in (POLYHAVEN_DIR / "index.json", ABO_DIR / "usd_index.json"):
        d = _read(p, {})
        idx["usd"].update({k: str(resolve(v)) for k, v in d.get("usd", {}).items()})
        idx["meta"].update(d.get("meta", {}))
    return idx


def ensure(name, cat=None):
    c = (cat or catalog())[name]
    if c["source"] == "abo":
        fetch(c["url"], c["file"], workers=24, chunk=1 << 20)
    return c["file"]


def model_file(name):
    """Local glTF / GLB of any library model (catalogued furniture or a Poly Haven prop)."""
    cat = catalog()
    if name in cat:
        return ensure(name, cat)
    return str(resolve(_read(POLYHAVEN_DIR / "index.json", {})["models"][name]))


_PROFILES = {}


def model_profile(name):
    """assets.shape.profile of a model in its own z-up frame (backrest / stand directions, seat height, long axis)."""
    if name not in _PROFILES:
        import trimesh

        from .shape import profile

        V = np.asarray(trimesh.load(model_file(name), force="mesh").vertices, float)
        _PROFILES[name] = profile(np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1))  # y-up glTF -> z-up (load_zup)
    return _PROFILES[name]


def variant_sheet(name):
    """A file laying several variants out in a row (Poly Haven `bananas`: four single bananas and a bunch along
    0.6 m), not one object: three or more parts whose centres are collinear and spread over more than twice the
    largest part. A loose pile (the Asian pears) is spread in two directions and stays one object."""
    import trimesh

    sc = trimesh.load(model_file(name))
    if not isinstance(sc, trimesh.Scene) or len(sc.graph.nodes_geometry) < 3:
        return False
    C, size = [], 0.0
    for node in sc.graph.nodes_geometry:
        T, g = sc.graph[node]
        V = trimesh.transform_points(sc.geometry[g].vertices, T)[:, [0, 2]]  # glTF y-up: x / z horizontal
        C.append((V.min(0) + V.max(0)) / 2)
        size = max(size, float(np.ptp(V, 0).max()))
    w, U = np.linalg.eigh(np.cov(np.asarray(C).T))
    spread = float(np.ptp(np.asarray(C) @ U[:, 1]))
    return bool(np.sqrt(max(w[0], 0) / max(w[1], 1e-12)) < 0.25 and spread > 2 * size)


def models(category, converted_only=True):
    cat, usd = catalog(), usd_index()["usd"]
    return sorted(n for n, c in cat.items() if c["category"] == category and (not converted_only or n in usd))


# --- retrieval ----------------------------------------------------------------------------------------------------
class Embedder:
    """DINOv2-base [CLS | mean patch] features; catalog images and photo crops are both objects on white."""

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

    def __call__(self, images, batch=32):
        torch = self.torch
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        feats = []
        for i in range(0, len(images), batch):
            x = np.stack([(np.asarray(square(im, 224), np.float32) / 255 - mean) / std for im in images[i:i + batch]])
            x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(self.device)
            with torch.no_grad():
                h = self.model(pixel_values=x).last_hidden_state
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(feats) if feats else np.zeros((0, 1536), np.float32)


def square(im, size, fill=(255, 255, 255)):
    """Letterbox to a white square (catalog images are white-background product shots)."""
    im = im.convert("RGBA")
    w, h = im.size
    s = max(w, h)
    bg = Image.new("RGBA", (s, s), (*fill, 255))
    bg.alpha_composite(im, ((s - w) // 2, (s - h) // 2))
    return bg.convert("RGB").resize((size, size), Image.BICUBIC)


def crop_on_white(img, mask, pad=0.06):
    """Photo crop of one instance composited on white."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(pad * max(y1 - y0, x1 - x0))
    y0, x0, y1, x1 = max(y0 - p, 0), max(x0 - p, 0), min(y1 + p, mask.shape[0]), min(x1 + p, mask.shape[1])
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    m = mask[y0:y1, x0:x1]
    out[m] = img[y0:y1, x0:x1][m]
    return Image.fromarray(out)


def embeddings(embedder=None):
    """Cached thumbnail embeddings for the whole catalog (recomputed when the catalog grows)."""
    cat = catalog()
    names = sorted(n for n, c in cat.items() if c.get("thumb"))
    if EMB.exists():
        d = np.load(EMB, allow_pickle=False)
        if list(d["names"]) == names:
            return names, d["feats"]
    embedder = embedder or Embedder()
    feats = embedder([Image.open(cat[n]["thumb"]) for n in names])
    np.savez(EMB, names=np.array(names), feats=feats)
    return names, feats


def rank(query, category, k=8, embedder=None):
    """[(name, cosine)] of `category`, best first. `query`: PIL image of the object on white."""
    embedder = embedder or Embedder()
    names, feats = embeddings(embedder)
    cat = catalog()
    q = embedder([query])[0]
    sims = feats @ q
    order = [i for i in np.argsort(-sims) if cat[names[i]]["category"] == category]
    return [(names[i], float(sims[i])) for i in order[:k]]


def fetch_diverse(category, n, embedder=None):
    """Download n ABO models of a category that are visually far from everything already available (farthest-point
    sampling on catalog embeddings, seeded with the downloaded / Poly Haven models), for cousins."""
    names, feats = embeddings(embedder)
    cat = catalog()
    idx = [i for i, nm in enumerate(names) if cat[nm]["category"] == category]
    have = [i for i in idx if Path(cat[names[i]]["file"]).exists()]
    abo = [i for i in idx if cat[names[i]]["source"] == "abo"]
    pick = list(have) or [abo[0]]
    new = []
    while len(new) < n and len(pick) < len(idx):
        cand = [i for i in abo if i not in pick]
        if not cand:
            break
        d = 1 - feats[cand] @ feats[pick].T
        i = cand[int(np.argmax(d.min(1)))]
        pick.append(i)
        try:
            ensure(names[i], cat)
            new.append(names[i])
        except Exception as e:
            print("skip", names[i], e)
    return new
