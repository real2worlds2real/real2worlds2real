"""COLMAP's sparse points in the twin's world, read without pycolmap (the recon environment has none).

    world_points(session) -> (X world (n, 3), reprojection error (n,), track length (n,))

splat/data/sparse/full/points3D.bin (COLMAP's binary model: per point its id, xyz, colour, error and track) placed by
splat/align.json's similarity (the robot's anchor when there is one).
"""
import json
import struct

import numpy as np








def world_points(s, min_track=3, max_error=1.5):
    """The session's sparse points in the twin's world, the well-observed ones."""
    sim = json.loads((s / "splat" / "align.json").read_text())["sim3"]
    P, err, trk = read_points3d(s / "splat" / "data" / "sparse" / "full" / "points3D.bin")
    keep = (trk >= min_track) & (err < max_error)
    X = float(sim["scale"]) * P[keep] @ np.asarray(sim["R"]).T + np.asarray(sim["t"])
    return X, err[keep], trk[keep]


from calibrated_scene_reconstruction.core.adapters.colmap import image_tracks, point_records, read_points3d  # compatibility exports
