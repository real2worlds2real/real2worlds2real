"""S3c retrieval fallback: library models for pieces the reconstruction could not handle.

    python -m calibrated_scene_reconstruction.recon.retrieve sessions/<name> [--k 6]

Targets
  * movable-furniture groups whose shared reconstruction failed QA (recon.furniture) or that ask for
    `"source": "retrieve"`;
  * objects with `"fallback": {"category": .., "min_iou": ..}` whose alignment IoU (recon.align) is below min_iou.
1. Rank the category by appearance: DINOv2 similarity of the photo crop to each model's catalog image.
2. Re-rank the top-k by geometry: scale to the observed height, try 24 yaws with planar ICP to the observed points,
   score by silhouette IoU (a model that looks right but has the wrong shape loses here).
3. The best model is copied onto every furniture member, or replaces the object instance.
Writes objects/<name>/{retrieved.json, qa_retrieve.png}; recon.materials turns them into twin stools / static swaps.
"""
import argparse
import json

import numpy as np
import trimesh

from calibrated_scene_reconstruction.assets.library import Embedder, catalog, crop_on_white, ensure, rank
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.support.furniture import PhotoFrame, fit_upright, load_zup, member_mesh, pick_reference, place_members, place_upright, qa_overlay


def targets(s: Session):
    """[(name, kind, spec)] that need a library model."""
    sg, out = s.scene_graph, []
    for group, spec in sg.get("movable_furniture", {}).items():
        rep = s / "objects" / group / "furniture.json"
        failed = not rep.exists() or not json.loads(rep.read_text()).get("ok")
        if spec.get("source") == "retrieve" or (spec.get("shared") and failed):
            out.append((group, "furniture", spec))
    for kind in ("static_objects", "dynamic_objects"):
        for name, spec in sg.get(kind, {}).items():
            fb = spec.get("fallback")
            a = s / "objects" / name / "align.json"
            if fb and (not a.exists() or json.loads(a.read_text())["iou_final"] < fb.get("min_iou", 0.6)):
                out.append((name, kind, {**spec, "category": fb["category"]}))
    return out


def retrieve(s: Session, pf: PhotoFrame, name, kind, spec, k, embedder):
    ids = spec["ids"]
    ref = pick_reference(pf.masks, ids) if kind == "furniture" else np.any(pf.masks[ids], 0)
    mask, obs = pf.mask(ref), pf.observed(ref)
    if kind == "furniture":
        support_z = 0.0
    else:  # resting surface of the reconstructed instance
        tw = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
        support_z = float(pf.to_world(np.asarray(tw.vertices))[:, 2].min())
    height = float(np.percentile(obs[:, 2], 98)) - support_z
    cands = rank(crop_on_white(pf.img, mask), spec["category"], k=k, embedder=embedder)
    cat = catalog()
    scored = []
    for model, sim in cands:
        try:
            Vz, F, _ = load_zup(ensure(model, cat))
        except Exception as e:  # download / parse failure: skip the candidate
            print(f"  {model}: unavailable ({e})")
            continue
        f = fit_upright(pf, Vz, F, ref, height, support_z)
        scored.append({"model": model, "title": cat[model]["title"][:80], "sim": sim, **f})
        print(f"  {model:24s} sim {sim:.3f}  IoU {f['iou']:.3f}  {cat[model]['title'][:60]}")
    best = max(scored, key=lambda c: c["iou"] + 0.5 * c["sim"])
    res = {"kind": kind, "category": spec["category"], "model": best["model"], "height": height,
           "support_z": support_z, "yaw": best["yaw"], "xy": best["xy"], "iou": best["iou"], "candidates": scored}
    Vz, F, _ = load_zup(ensure(best["model"], cat))
    Vw = place_upright(Vz, best["yaw"], best["scale"], best["xy"], support_z)
    if kind == "furniture":
        res["members"] = place_members(pf, Vw, F, ref, ids)
        c = (Vw.min(0) + Vw.max(0))[:2] / 2
        meshes = [(member_mesh(Vw, c, m), F) for m in res["members"] if not m.get("drop")]
    else:
        meshes = [(Vw, F)]
    s.write_json(f"objects/{name}/retrieved.json", res)
    qa_overlay(pf, meshes, ids, s / "objects" / name / "qa_retrieve.png")
    print(f"[{name}] -> {best['model']} ({best['title']}), IoU {best['iou']:.3f}, height {height:.2f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    s = Session(args.session)
    todo = targets(s)
    for d in (s / "objects").glob("*/retrieved.json"):  # stale results of pieces that no longer need retrieval
        if d.parent.name not in {t[0] for t in todo}:
            d.unlink()
    if not todo:
        print("nothing to retrieve")
        return
    pf, embedder = PhotoFrame(s), Embedder()
    for name, kind, spec in todo:
        print(f"[{name}] retrieving a {spec['category']} ({kind})")
        retrieve(s, pf, name, kind, spec, args.k, embedder)


if __name__ == "__main__":
    main()
