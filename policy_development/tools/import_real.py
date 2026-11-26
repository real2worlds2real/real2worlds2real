"""Tool `import_real`: turn the workspace's real-robot demonstrations into a dataset, optionally cut to a task segment.

Cutting rules are task-independent predicates over what the demonstrations contain (labelled cloud, joints):
  until  label_rise    - end once a labelled object's centroid rose `rise_m` above its start (+ `extra_steps`)
         palm_rise     - end once the palm (cloud label 'hand') rose `rise_m` above its start (+ `extra_steps`)
         step_fraction - end at a fraction of each episode
  from   label_moved   - start where a labelled object first moved `moved_m` (drops the approach)
`pad_hold` appends copies of the last frame whose target equals the measured joints (= "stay here"), so a policy learns to
hold the end state instead of drifting on.  Output episodes keep the real demonstrations' format."""
import json

import numpy as np

from _adapter import main


def label_id(labels: dict, name: str) -> int:
    for k, v in labels.items():
        if v == name:
            return int(k)
    raise ValueError(f"unknown label {name!r}; labels: {sorted(labels.values())}")


def centroid_z(cloud, labels, lab):
    out = np.full(len(cloud), np.nan)
    for t in range(len(cloud)):
        m = labels[t] == lab
        if m.sum() >= 15:
            out[t] = np.median(cloud[t][m].astype(np.float32), axis=0)[2]
    return out


def centroid_xy(cloud, labels, lab):
    out = np.full((len(cloud), 2), np.nan)
    for t in range(len(cloud)):
        m = labels[t] == lab
        if m.sum() >= 15:
            out[t] = np.median(cloud[t][m].astype(np.float32)[:, :2], axis=0)
    return out


def first_true(x):
    i = np.flatnonzero(x)
    return int(i[0]) if len(i) else None


def run(job):
    a = job.args
    labels = job.binding["task"].get("labels", {})
    src = sorted((job.ws / "data").glob("*.npz"))[: a.get("max_episodes", 10 ** 6)]
    if not src:
        raise RuntimeError("the workspace has no real demonstrations")
    out = job.ws / "datasets" / a["name"]
    out.mkdir(parents=True)
    until, frm, pad = a.get("until", {"kind": "none"}), a.get("from", {"kind": "none"}), int(a.get("pad_hold", 0))
    kept, skipped, frames, lens = 0, [], 0, []
    for f in src:
        z = np.load(f)
        cloud, lab, q, tgt = z["cloud"], z["labels"], z["q29"], z["target"]
        T = len(q)
        lo, hi = 0, T
        if until["kind"] in ("label_rise", "palm_rise"):
            lid = label_id(labels, until["label"] if until["kind"] == "label_rise" else "hand")
            zc = centroid_z(cloud, lab, lid)
            z0 = np.nanmedian(zc[:5])
            i = first_true(np.nan_to_num(zc - z0, nan=-1) >= until["rise_m"])
            if i is None:
                skipped.append({"file": f.name, "why": f"{until['kind']} never reached {until['rise_m']} m"}); continue
            hi = min(T, i + int(until.get("extra_steps", 0)) + 1)
        elif until["kind"] == "step_fraction":
            hi = max(2, int(round(T * until["fraction"])))
        if frm["kind"] == "label_moved":
            xy = centroid_xy(cloud, lab, label_id(labels, frm["label"]))
            d = np.linalg.norm(xy - np.nanmedian(xy[:5], axis=0), axis=1)
            i = first_true(np.nan_to_num(d, nan=0) >= frm["moved_m"])
            lo = max(0, (i or 0) - int(frm.get("before_steps", 0)))
        if hi - lo < 5:
            skipped.append({"file": f.name, "why": "segment shorter than 5 steps"}); continue
        sl = slice(lo, hi)
        C, L, Q, A = cloud[sl], lab[sl], q[sl].astype(np.float32), tgt[sl].astype(np.float32)
        if pad:
            C = np.concatenate([C, np.repeat(C[-1:], pad, 0)]); L = np.concatenate([L, np.repeat(L[-1:], pad, 0)])
            A = np.concatenate([A, np.repeat(Q[-1:], pad, 0)]); Q = np.concatenate([Q, np.repeat(Q[-1:], pad, 0)])
        np.savez_compressed(out / f"ep_{kept:05d}.npz", cloud=C.astype(np.float16), labels=L.astype(np.uint8), q29=Q, target=A,
                            success=True, term="real_demo", source=f.name, segment=np.array([lo, hi]))
        kept += 1; frames += len(Q); lens.append(len(Q))
    if kept == 0:
        raise RuntimeError(f"no episode survived the cut: {skipped[:5]}")
    manifest = {"source": "real_demonstrations", "until": until, "from": frm, "pad_hold": pad, "episodes": kept, "frames": frames,
                "median_length": int(np.median(lens)), "skipped": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    job.finish(facts={"dataset": a["name"], "episodes": kept, "frames": frames, "median_length": manifest["median_length"],
                      "skipped": len(skipped), "skipped_examples": skipped[:3]},
               created={"datasets": [{"name": a["name"], "path": str(out), "episodes": kept, "frames": frames, "domain": "real",
                                      "note": f"real demonstrations, until={until}, from={frm}, pad_hold={pad}"}]})


if __name__ == "__main__":
    main(run)
