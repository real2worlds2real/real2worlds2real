"""Optional rectangular edge hypotheses from paired observed image samples."""
from __future__ import annotations

import cv2
import numpy as np

from .posed_observations import project


def refine_edges(views, origin, u, v, bounds, settings, *, support_value, occluder_value):
    bounds = np.asarray(settings.get("bounds_uv_m", bounds), float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError("Edge refinement needs increasing rectangular plane bounds")
    reach, step, probe = (float(settings.get(k, d)) for k, d in
                          (("search_m", .06), ("step_m", .005), ("probe_m", .02)))
    minimum = settings.get("min_pairs", 30)
    evidence = settings.get("evidence", "mask")
    min_score, improvement = float(settings.get("min_score", .3)), float(settings.get("min_improvement", .05))
    if (not np.isfinite([reach, step, probe, min_score, improvement]).all()
            or not 0 < step <= reach <= 1 or reach / step > 200 or not 0 < probe <= 1
            or type(minimum) is not int or not 1 <= minimum <= 100000
            or evidence not in {"mask", "color"} or not 0 <= min_score <= 1 or not 0 <= improvement <= 1):
        raise ValueError("Invalid edge search settings")
    offsets = np.unique(np.r_[np.arange(-reach, reach + step * .1, step), 0.])
    classifiers = []
    for view in views:
        usable = view["mask"] != occluder_value
        usable &= view.get("ignore", np.zeros_like(usable)) == 0
        if evidence == "color":
            if "image" not in view:
                raise ValueError("Color edge evidence requires RGB images")
            lab = cv2.cvtColor(view["image"], cv2.COLOR_RGB2LAB).astype(float)
            core = cv2.erode(((view["mask"] == support_value) & usable).astype(np.uint8),
                             np.ones((5, 5), np.uint8)) > 0
            if core.sum() < 20:
                classifiers.append(None)
                continue
            center = np.median(lab[core], axis=0)
            spread = np.percentile(np.linalg.norm(lab[core] - center, axis=1), 90)
            on = np.linalg.norm(lab - center, axis=2) <= max(float(spread), 8.)
        else:
            on = view["mask"] == support_value
        classifiers.append((on, usable))
    result, refined = {}, bounds.copy()
    for axis in (0, 1):
        other = 1 - axis
        along = np.linspace(bounds[0, other], bounds[1, other], 64)[2:-2]
        for side, sign in ((0, -1), (1, 1)):
            edge = bounds[side, axis]
            scores, pairs = [], []
            for offset in offsets:
                uv = np.tile(bounds.mean(axis=0), (2, len(along), 1))
                uv[:, :, other] = along
                uv[0, :, axis] = edge + sign * (offset - probe)
                uv[1, :, axis] = edge + sign * (offset + probe)
                xyz = origin + uv.reshape(-1, 2)[:, :1] * u + uv.reshape(-1, 2)[:, 1:] * v
                total, votes = 0., 0
                for view, classifier in zip(views, classifiers, strict=True):
                    if classifier is None:
                        continue
                    on, usable = classifier
                    pixels, _, inside = project(xyz, view)
                    p = np.rint(np.clip(pixels, [0, 0], [view["width"] - 1, view["height"] - 1])).astype(int)
                    available = (inside & usable[p[:, 1], p[:, 0]]).reshape(2, -1).all(axis=0)
                    labels = on[p[:, 1], p[:, 0]].reshape(2, -1).astype(float)
                    total += float((labels[0] - labels[1])[available].sum())
                    votes += int(available.sum())
                scores.append(total / max(votes, 1))
                pairs.append(votes)
            eligible = np.asarray(pairs) >= minimum
            baseline = int(np.flatnonzero(offsets == 0)[0])
            best = int(np.argmax(np.where(eligible, scores, -np.inf)))
            accepted = bool(eligible[best] and scores[best] >= min_score
                            and scores[best] >= scores[baseline] + improvement)
            chosen = float(offsets[best]) if accepted else 0.
            refined[side, axis] = edge + sign * chosen
            result[f"{'u' if axis == 0 else 'v'}_{'min' if side == 0 else 'max'}"] = {
                "offset_m": chosen, "accepted_hypothesis": accepted,
                "candidate_offsets_m": offsets.tolist(), "scores": scores, "valid_pairs": pairs}
    if not (refined[1] > refined[0]).all():
        raise ValueError("Edge search produced crossed boundaries; reduce the search range")
    return {"evidence": evidence, "initial_bounds_uv_m": bounds.tolist(),
            "candidate_bounds_uv_m": refined.tolist(), "edges": result,
            "observed_boundary_certificate": False,
            "limits": ["A rectangular modeling hypothesis, separate from raw footprint votes.",
                       "Similar-colored neighbors and uncertain segmentation can bias the proposal."]}
