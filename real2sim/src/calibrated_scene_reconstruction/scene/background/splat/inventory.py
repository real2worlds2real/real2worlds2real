"""Observed background inventory for code-room reconstruction.

python -m calibrated_scene_reconstruction.scene.background.splat.inventory sessions/<name> [things]

The historical splat/cousins/things storage path is preserved for existing captures.
This module does not edit or train splat cousins.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from .config import folder, sim3
from .geometry import world_of_model  # noqa: F401 -- legacy inventory API


VOCAB = ["desk", "office chair", "computer monitor", "cabinet", "shelving unit", "cardboard box", "sofa", "printer",
         "whiteboard", "cubicle partition", "potted plant", "trash bin", "computer tower", "stool", "table", "bookcase"]
STUFF = ["wall", "floor", "ceiling"]
CELL = 4  # px of the point z-buffer


def inventory_settings(s, n_frames=None):
    cfg = s.config('environment_inventory.json', {})
    count = cfg.get('n_frames', 80) if n_frames is None else n_frames
    if type(count) is not int or not 2 <= count <= 512:
        raise ValueError('Inventory review count must be2..512 registered frames')
    for key, default in (('vocabulary', VOCAB), ('stuff', STUFF)):
        values = cfg.get(key, default)
        if (not isinstance(values, list) or not values or len(values) > 64
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Inventory prompts must be nonempty distinct strings: '+key)
    return {**cfg, 'n_frames': count, 'vocabulary': cfg.get('vocabulary', VOCAB), 'stuff': cfg.get('stuff', STUFF)}


def frame_indices(names, count, frames=None):
    names = list(map(str, names))
    if frames is not None:
        if (not isinstance(frames, list) or len(frames) < 2 or len(set(frames)) != len(frames)
                or any(n not in names for n in frames)):
            raise ValueError('Inventory frames must be distinct registered image names')
        return np.asarray([names.index(n) for n in frames])
    if len(names) < 2:
        raise ValueError('Inventory needs multiple registered capture frames')
    return np.unique(np.linspace(0, len(names)-1, min(count, len(names))).astype(int))


def distinct_view_votes(candidates, members):
    """Multiple masks in one image never become independent camera evidence."""
    per_view = {}
    for index in members:
        candidate = candidates[index]
        per_view.setdefault(candidate[1], set()).update(map(int, candidate[2]))
    votes = {}
    for indices in per_view.values():
        for index in indices:
            votes[index] = votes.get(index, 0)+1
    return votes, len(per_view)


def associate_instances(candidates, minimum_overlap=.3):
    """Associate views without merging simultaneously distinct objects.

    A union through a broad/occluded proposal must not merge two disjoint masks
    seen in the same image. Duplicate overlapping proposals remain permissible.
    Strongest shared-3D-support links are considered first; no real dimensions
    or class-specific object counts enter this decision.
    """
    sets=[set(map(int,c[2])) for c in candidates]
    groups={i:{i} for i in range(len(candidates))};owner=list(range(len(candidates)))
    forbidden=set();edges=[]
    for a in range(len(candidates)):
        for b in range(a+1,len(candidates)):
            if candidates[a][0]!=candidates[b][0]:continue
            shared=len(sets[a]&sets[b])/max(1,min(len(sets[a]),len(sets[b])))
            if candidates[a][1]==candidates[b][1] and shared<minimum_overlap:
                forbidden.add((a,b))
            elif shared>=minimum_overlap:
                edges.append((-shared,a,b))
    rejected=0
    for _,a,b in sorted(edges):
        ga,gb=owner[a],owner[b]
        if ga==gb:continue
        if any(tuple(sorted((i,j))) in forbidden for i in groups[ga] for j in groups[gb]):
            rejected+=1;continue
        members=groups.pop(gb);groups[ga].update(members)
        for i in members:owner[i]=ga
    return list(groups.values()),{'cannot_link_pairs':len(forbidden),'rejected_bridge_links':rejected,
        'policy':'Disjoint same-frame instances cannot merge through a cross-view bridge. Counts remain unapproved observations.'}


def cdir(s, name=None):
    d = folder(s) / "cousins"
    return d / name if name else d


def largest_cluster(P, cell=0.1):
    """The points of the largest 26-connected cluster of `cell` m voxels (a thing's Gaussians, without the stray ones a
    mask's edge picked up across the room)."""
    from scipy import ndimage

    ij = np.floor(P / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    g = np.zeros(ij.max(0) + 1, bool)
    g[tuple(ij.T)] = True
    lab, n = ndimage.label(g, np.ones((3, 3, 3)))
    if n <= 1:
        return np.ones(len(P), bool)
    per = lab[tuple(ij.T)]
    best = np.bincount(per, minlength=n + 1)[1:].argmax() + 1
    return per == best


def things(s, n_frames=None, output=None):
    """The background splat as things for the agent's plans: SAM 3 instances of VOCAB in frames spread over the walk,
    each lifted to the Gaussians front-most inside it (a point z-buffer of the opaque ones); one class's instances
    whose Gaussian sets overlap across frames merged; every Gaussian to the thing that claimed it most; each thing its
    largest connected cluster. Writes splat/cousins/things/{gauss.npz, instances.npz, inventory.json, inventory.jpg
    (each thing's best crop), plan.png (the floor plan with the things' footprints)}."""
    import cv2
    import torch
    from PIL import Image
    from plyfile import PlyData
    from transformers import Sam3Model, Sam3Processor

    from calibrated_scene_reconstruction.scene.objects.object_hull import project

    if (s/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: use an environment derivative')
    cfg = inventory_settings(s, n_frames)
    vocabulary, stuff_names = cfg['vocabulary'], cfg['stuff']
    out = Path(output) if output is not None else cdir(s, "things")
    out = out.resolve()
    if not out.is_relative_to(s.dir.resolve()):
        raise ValueError('Inventory output belongs inside this session')
    camera = s/'parsing/posed_frames/cameras.npz'
    z = np.load(camera)
    indices = frame_indices(z['names'], cfg['n_frames'], cfg.get('frames'))
    out.mkdir(parents=True, exist_ok=False)
    ply = folder(s)/'export/background.ply'
    v = PlyData.read(str(ply))["vertex"].data
    def sha(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    inputs = {str(p.resolve()): sha(p) for p in [ply, camera, folder(s)/'align.json', Path(__file__)]}
    config_path = s.config_dir/'environment_inventory.json'
    if config_path.exists():
        inputs[str(config_path.resolve())] = sha(config_path)
    c, R, t = sim3(s)
    X = c * np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64) @ R.T + t
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    np.savez_compressed(out / "gauss.npz", X=X.astype(np.float32), opacity=op.astype(np.float32))
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    print(f'[inventory] {len(X)} background Gaussians; {len(indices)} registered review frames; {len(vocabulary)} object prompts', flush=True)
    model = Sam3Model.from_pretrained("facebook/sam3", local_files_only=True).to("cuda").eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3", local_files_only=True)

    def instances(img, prompt, thr=0.4):
        inp = proc(images=Image.fromarray(img), text=prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model(**inp)
        r = proc.post_process_instance_segmentation(o, threshold=thr, mask_threshold=0.5,
                                                    target_sizes=inp.get("original_sizes").tolist())[0]
        return [(m.cpu().numpy().astype(bool), float(sc)) for m, sc in zip(r["masks"], r["scores"])]

    opaque = op > 0.2
    cands, seen = [], np.zeros(len(X), np.int32)
    stuff_votes = np.zeros((len(X), len(stuff_names)), np.int32)
    observations = []
    for number, i in enumerate(indices):
        frame = str(z['names'][i])
        source = (img_dir/frame).resolve()
        inputs[str(source)] = sha(source)
        img = np.asarray(Image.open(source).convert("RGB"))
        H, W = img.shape[:2]
        u, vv, d = project(X, z["K"][i], z["R_world_cam"][i], z["centre"][i])
        idx = np.nonzero((d > 0.1) & (d < 12) & (u >= 0) & (u < W) & (vv >= 0) & (vv < H))[0]
        cu, cv_ = (u[idx] // CELL).astype(int), (vv[idx] // CELL).astype(int)
        zb = np.full((H // CELL + 1, W // CELL + 1), np.inf)
        oo = opaque[idx]
        np.minimum.at(zb, (cv_[oo], cu[oo]), d[idx][oo])
        front = idx[d[idx] <= zb[cv_, cu] * 1.03 + 0.03]
        seen[front] += 1
        ui, vi = u[front].astype(int), vv[front].astype(int)
        for k, cls in enumerate(stuff_names):
            for m, _ in instances(img, cls, 0.3):
                stuff_votes[front, k] += m[vi, ui]
        start = len(cands)
        for cls in vocabulary:
            for m, sc in instances(img, cls):
                gi = front[m[vi, ui]] if m.sum() >= 1500 else []
                if len(gi) >= 50:
                    ys, xs = np.nonzero(m)
                    cands.append((cls, int(i), gi, sc, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                                  int(m.sum())))
        observations.append({'frame': frame, 'candidates': [{'class': c[0], 'score': float(c[3]),
            'crop': c[4], 'mask_pixels': int(c[5]), 'front_gaussians': len(c[2])} for c in cands[start:]]})
        (out/'observations.json').write_text(json.dumps(observations, indent=2)+'\n')
        print(f'[inventory] {number+1}/{len(indices)} {frame}: {len(cands)-start} candidates', flush=True)
    # Preserve proposal-to-Gaussian membership so association can be inspected
    # and repeated without rerunning the perception model or changing sampling.
    offsets=np.r_[0,np.cumsum([len(c[2]) for c in cands])]
    np.savez_compressed(out/'candidate_rows.npz',offsets=offsets,
        rows=np.concatenate([c[2] for c in cands]) if cands else np.array([],int))
    (out/'candidates.json').write_text(json.dumps([
        {'class':c[0],'frame_index':c[1],'score':c[3],'crop':c[4],'mask_pixels':c[5]} for c in cands],indent=2)+'\n')
    groups,association=associate_instances(cands)
    (out/'association.json').write_text(json.dumps(association,indent=2)+'\n')
    claims, best, inv = np.full(len(X), -1, np.int32), np.zeros(len(X), np.int32), []
    for members in sorted(groups, key=len, reverse=True):
        members=sorted(members)
        cnt, views = distinct_view_votes(cands, members)
        gi = np.array([q for q, k in cnt.items() if k >= max(2, 0.25 * views)], int)
        if views < 2 or len(gi) < 200:
            continue
        gi = gi[largest_cluster(X[gi])]
        w = np.array([cnt[q] for q in gi])
        take = w > best[gi]
        tid = len(inv)
        claims[gi[take]], best[gi[take]] = tid, w[take]
        view = max(members, key=lambda a: cands[a][5] * cands[a][3])
        inv.append({"id": tid, "class": cands[members[0]][0], "frames": views,
                    'proposals': len(members),
                    "frame": str(z["names"][cands[view][1]]), "crop": cands[view][4],
                    'observations': [{'frame': str(z['names'][cands[a][1]]), 'crop': cands[a][4],
                        'score': float(cands[a][3])} for a in members]})
    for th in inv:
        P = X[claims == th["id"]]
        th["gaussians"] = len(P)
        if len(P):
            th["box"] = [P.min(0).round(3).tolist(), P.max(0).round(3).tolist()]
    inv = [th for th in inv if th["gaussians"] >= 200]
    stuff = np.where((stuff_votes.max(1) >= np.maximum(2, 0.5 * seen)) & (claims < 0), stuff_votes.argmax(1), -1)
    np.savez_compressed(out / "instances.npz", thing=claims, stuff=stuff.astype(np.int8), seen=seen)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError('Inventory input changed: '+path)
    (out / "inventory.json").write_text(json.dumps({'version': 1, "stuff": stuff_names, "things": inv,
        'settings': cfg, 'input_sha256': inputs, 'reviewed_frames': [str(z['names'][i]) for i in indices],
        'scope': 'Unreviewed background-instance proposals from captured images and legally calibrated GS. Not task-size truth, a completed environment model or photorealism approval. Training/video selection unchanged.'}, indent=1))
    tiles = []
    for th in inv:
        img = np.asarray(Image.open(img_dir / th["frame"]).convert("RGB"))
        x0, y0, x1, y1 = th["crop"]
        tile = cv2.resize(img[y0:y1 + 1, x0:x1 + 1], (200, 200))
        cv2.putText(tile, f"{th['id']} {th['class']}"[:24], (3, 14), 0, 0.45, (255, 255, 0), 1)
        tiles.append(tile)
    if not tiles:
        tiles.append(np.zeros((200, 200, 3), np.uint8))
    while len(tiles) % 8:
        tiles.append(np.zeros((200, 200, 3), np.uint8))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])).save(
        out / "inventory.jpg", quality=88)
    plan_view(X[(op > 0.2) & (X[:, 2] < 2.2)], inv, out / "plan.png")
    print(f"done: {len(inv)} things, {(claims >= 0).sum()} of {len(X)} Gaussians in them, "
          f"{(stuff >= 0).sum()} surface-labelled -> {out}", flush=True)


def plan_view(X, inv, dst, S=100):
    """The background from above (height-coloured, a 1 m grid) with each thing's footprint boxed and labelled."""
    import cv2

    lo, hi = np.percentile(X[:, :2], 1, 0) - 0.5, np.percentile(X[:, :2], 99, 0) + 0.5
    W, H = int((hi[0] - lo[0]) * S), int((hi[1] - lo[1]) * S)
    img = np.full((H, W, 3), 255, np.uint8)
    to = lambda x, y: (int((x - lo[0]) * S), int(H - 1 - (y - lo[1]) * S))  # noqa: E731
    for k in range(int(np.ceil(lo[0])), int(hi[0]) + 1):
        cv2.line(img, (to(k, 0)[0], 0), (to(k, 0)[0], H - 1), (225, 225, 225), 1)
        cv2.putText(img, f"x={k}", (to(k, 0)[0] + 2, 12), 0, 0.35, (120, 120, 120), 1)
    for k in range(int(np.ceil(lo[1])), int(hi[1]) + 1):
        cv2.line(img, (0, to(0, k)[1]), (W - 1, to(0, k)[1]), (225, 225, 225), 1)
        cv2.putText(img, f"y={k}", (2, to(0, k)[1] - 2), 0, 0.35, (120, 120, 120), 1)
    o = np.argsort(X[:, 2])
    u, v = ((X[o, 0] - lo[0]) * S).astype(int), (H - 1 - (X[o, 1] - lo[1]) * S).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zz = np.clip(X[o, 2][ok] / 2.0, 0, 1)
    img[v[ok], u[ok]] = np.stack([200 * (1 - zz), 160 + 0 * zz, 255 * zz], 1).astype(np.uint8)
    for th in inv:
        (x0, y0, _), (x1, y1, _) = th["box"]
        cv2.rectangle(img, to(x0, y1), to(x1, y0), (0, 0, 0), 1)
        cv2.putText(img, f"{th['id']} {th['class'][:12]}", (to(x0, y1)[0], to(x0, y1)[1] - 2), 0, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(dst), img)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("session")
    parser.add_argument("step", nargs="?", choices=["things"], default="things")
    parser.add_argument('--n-frames', type=int)
    parser.add_argument('--output', help='New output directory inside this session; existing evidence is never overwritten')
    args = parser.parse_args()
    things(Session(args.session), args.n_frames, args.output)


if __name__ == "__main__":
    main()
