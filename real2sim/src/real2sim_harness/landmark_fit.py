"""Optional fixed image landmarks for an agent-authored geometry fit."""

from __future__ import annotations
import json
import numpy as np
from .artifacts import sha256_file
from .posed_observations import file_path, project


def load_landmarks(filename, base, views, hashes):
    path = file_path(base, filename)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or set(data) - {"schema_version", "source", "observations"}:
        raise ValueError("Unknown landmark document fields")
    if (
        data.get("schema_version") != 1
        or not isinstance(data.get("source"), str)
        or not data["source"].strip()
    ):
        raise ValueError("Landmarks need schema_version 1 and annotation provenance")
    rows = data.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Supply observed landmarks")
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or set(r) - {
            "view",
            "point_index",
            "feature",
            "pixel",
            "covariance_px2",
        }:
            raise ValueError("Unknown landmark observation fields")
        v, index, name = r.get("view"), r.get("point_index", 0), r.get("feature")
        if (
            type(v) is not int
            or not 0 <= v < len(views)
            or type(index) is not int
            or index < 0
            or not isinstance(name, str)
            or not name
        ):
            raise ValueError(
                "Each landmark needs an existing view, feature name and nonnegative point_index"
            )
        if (v, name, index) in seen:
            raise ValueError("Duplicate landmark observation")
        seen.add((v, name, index))
        xy, cov = np.asarray(r.get("pixel"), float), np.asarray(r.get("covariance_px2"), float)
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or np.any(xy < 0)
            or np.any(xy >= [views[v]["width"], views[v]["height"]])
        ):
            raise ValueError("Landmark must lie in the declared image domain")
        if (
            cov.shape != (2, 2)
            or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, rtol=0, atol=1e-10)
            or np.linalg.eigvalsh(cov).min() <= 0
        ):
            raise ValueError("Landmark uncertainty must be symmetric positive definite")
    hashes[str(path)] = sha256_file(path)
    return rows


def score_landmarks(features, rows, views):
    by_name = {}
    for feature in features:
        if feature["name"] in by_name:
            raise ValueError("Feature names must be unique for landmark fitting")
        points = np.asarray(feature["points"], float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Model feature points must be finite Nx3")
        by_name[feature["name"]] = points
    residuals = []
    for row in rows:
        points = by_name.get(row["feature"])
        i = row.get("point_index", 0)
        if points is None or i >= len(points):
            raise ValueError("Model omitted an observed feature point")
        uv, depth, _ = project(points[i : i + 1], views[row["view"]])
        if depth[0] <= 0 or not np.isfinite(uv).all():
            raise ValueError("Observed model landmark projects behind the camera")
        delta = uv[0] - row["pixel"]
        white = np.linalg.solve(np.linalg.cholesky(row["covariance_px2"]), delta)
        residuals.append(
            {
                "view": row["view"],
                "feature": row["feature"],
                "point_index": i,
                "predicted_pixel": uv[0].tolist(),
                "residual_px": delta.tolist(),
                "squared_sigma": float(white @ white),
            }
        )
    loss = float(np.mean([r["squared_sigma"] for r in residuals]) / 2)
    return {"loss": loss, "rms_sigma": float(np.sqrt(loss)), "observations": residuals}
