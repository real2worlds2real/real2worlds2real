"""Local image residuals and projected features shared by fitting and review."""
from __future__ import annotations

import cv2
import numpy as np

from .posed_observations import project


def silhouette_metrics(predicted, observed, valid=None, *, min_fraction=0.003, min_width_px=2):
    predicted, observed = np.asarray(predicted, bool), np.asarray(observed, bool)
    valid = np.ones_like(observed) if valid is None else np.asarray(valid, bool)
    if predicted.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("Silhouette arrays must share one pixel domain")
    if not 0 <= min_fraction <= 1 or type(min_width_px) is not int or not 1 <= min_width_px <= 32:
        raise ValueError("Invalid local residual fraction/width")
    area = max(int((observed & valid).sum()), 1)
    union = int(((observed | predicted) & valid).sum())
    regions = []
    for name, mask in (("missing", observed & ~predicted & valid),
                       ("excess", predicted & ~observed & valid)):
        # Thickness rejects narrow registration fringes without translating or
        # shrinking accepted components (even-sized morphology anchors can shift).
        core = cv2.erode(mask.astype(np.uint8), np.ones((min_width_px, min_width_px), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        thick_labels = set(np.unique(labels[core]).tolist())
        for i in range(1, count):
            x, y, w, h, pixels = stats[i].tolist()
            if pixels / area >= min_fraction and i in thick_labels:
                regions.append({"kind": name, "pixels": pixels, "fraction": pixels / area,
                                "bbox_xywh": [x, y, w, h]})
    def extent(mask):
        y, x = np.nonzero(mask & valid)
        if not len(x):
            return {"area_px": 0, "bbox_xywh": None, "centroid_px": None}
        return {"area_px": int(len(x)),
                "bbox_xywh": [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)],
                "centroid_px": [float(x.mean()), float(y.mean())]}
    return {"iou": float((predicted & observed & valid).sum() / max(union, 1)),
            "local_difference": float(sum(row["fraction"] for row in regions)),
            "regions": sorted(regions, key=lambda row: -row["pixels"]),
            "observed_projection": extent(observed), "predicted_projection": extent(predicted)}


def feature_evidence(features, view, view_index=0):
    """Project arbitrary model landmarks/curves; hidden depths are not established."""
    output = []
    for feature in features:
        points = np.asarray(feature["points"], float)
        if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 4096 or not np.isfinite(points).all():
            raise ValueError("Feature points must be finite Nx3 coordinates, N=1..4096")
        uv, depth, inside = project(points, view)
        row = {"name": str(feature["name"]), "pixels": uv.tolist(),
               "in_frame": inside.tolist(), "camera_depth": depth.tolist()}
        observed = feature.get("observed_pixels_by_view", {}).get(str(view_index))
        if observed is not None:
            observed = np.asarray(observed, float)
            if observed.shape != uv.shape or not np.isfinite(observed).all():
                raise ValueError("Observed feature pixels must match the projected landmark count")
            errors = np.linalg.norm(uv - observed, axis=1)
            row.update(residual_px=errors.tolist(), mean_residual_px=float(errors.mean()))
        output.append(row)
    return output
