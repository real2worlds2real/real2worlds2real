"""Sim-over-real overlays of a replay: the simulated object's silhouette drawn on the real frames, for one or more sim
tracks (e.g. default vs calibrated physics), plus the image-space track of the object centre over time.

    python -m calibrated_scene_reconstruction.real2sim.overlay real/<name> [--tracks sim_track_default.npz sim_track_calibrated.npz] [--every 6]
    python -m calibrated_scene_reconstruction.real2sim.overlay real/<name> --scan          # replay/scan.json -> replay/scan.png

Reads real/<name>/{video.mp4, track.npz, camera.json, replay/<tracks>} (sim tracks carry the object pose per video
frame and its local points, apps/real2sim_replay.py); writes replay/overlay.png and replay/track_plot.png.
Colours: SAM 3 mask outline green; sim silhouettes red, blue, magenta in --tracks order; the first track's robot
links (origins) as yellow dots: the replayed arm against the real one checks base pose, extrinsics and the hand.
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.real2sim.data import load
from calibrated_scene_reconstruction.real2sim.observe import convex_silhouette, project, read_video

COLOURS = [(255, 40, 40), (40, 120, 255), (255, 0, 255)]


def silhouettes(rec, sim, shape):
    pts = sim["obj_pts_local"]
    R = Rotation.from_quat(sim["quat_wxyz"], scalar_first=True).as_matrix()
    return [convex_silhouette(rec.K, rec.T_world_cam, p + pts @ r.T, shape) for p, r in zip(sim["pos"], R)]


def outline(m):
    e = m.copy()
    e[1:-1, 1:-1] &= ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return e


def render(rec, frames, obs, sims, every=6, scale=2):
    s = int(obs["mask_stride"])
    H, W = frames.shape[1:3]
    sil = [silhouettes(rec, sim, (H, W)) for sim in sims]
    links = project(rec.K, rec.T_world_cam, sims[0]["link_pos"])[0] if "link_pos" in sims[0] else None
    tiles = []
    for f in range(0, len(frames), every):
        im = frames[f].copy()
        m = np.kron(obs["masks"][f], np.ones((s, s), bool))[:H, :W]
        im[outline(m)] = (40, 255, 40)
        for k, ss in enumerate(sil):
            im[outline(ss[f])] = COLOURS[k % len(COLOURS)]
        if links is not None:
            for u, v in links[f][np.all(np.isfinite(links[f]), 1)].astype(int):
                if 0 <= u < W and 0 <= v < H:
                    im[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = (255, 230, 0)
        tiles.append(np.kron(im, np.ones((scale, scale, 1), np.uint8)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[r:r + 4], 1) for r in range(0, len(tiles), 4)], 0)


def plot(obs, sims, labels, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, name in enumerate(("u (px)", "v (px)")):
        c = np.where(obs["valid"], obs["centroid"][:, j], np.nan)
        ax[j].plot(obs["t"], c, color="#2a2", label="SAM 3 mask centroid")
        for k, (sim, lab) in enumerate(zip(sims, labels)):
            ax[j].plot(sim["t"], sim["uv"][:, j], color=np.array(COLOURS[k % len(COLOURS)]) / 255, label=lab)
        ax[j].set_xlabel("video time (s)")
        ax[j].set_ylabel(name)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def plot_scan(scan, out):
    """Loss over each scanned parameter (apps/real2sim_replay.py --scan): the identifiability picture."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(scan), figsize=(3.2 * len(scan), 2.8), squeeze=False)
    for a, (name, d) in zip(ax[0], scan.items()):
        a.plot(d["values"], d["loss"], "o-", color="#2a6fdb", ms=3)
        if name.endswith(("mass", "friction")):
            a.set_xscale("log")
        a.set_xlabel(name)
        a.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    import json

    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--tracks", nargs="+", default=["sim_track.npz"])
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--scan", action="store_true", help="plot replay/scan.json instead")
    args = ap.parse_args()
    rec = load(args.recording)
    if args.scan:
        plot_scan(json.loads((rec.path / "replay" / "scan.json").read_text()), rec.path / "replay" / "scan.png")
        print(f"-> {rec.path / 'replay' / 'scan.png'}")
        return
    frames = read_video(rec.video)[0]
    obs = dict(np.load(rec.path / "track.npz"))
    sims = [dict(np.load(rec.path / "replay" / t)) for t in args.tracks]
    Image.fromarray(render(rec, frames, obs, sims, args.every)).save(rec.path / "replay" / "overlay.png")
    plot(obs, sims, [t.removesuffix(".npz") for t in args.tracks], rec.path / "replay" / "track_plot.png")
    print(f"-> {rec.path / 'replay' / 'overlay.png'}, track_plot.png")


if __name__ == "__main__":
    main()
