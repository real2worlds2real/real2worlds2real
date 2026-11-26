"""Decode every active rigid contact pair, not only the manipulated body."""
import numpy as np

from calibrated_scene_reconstruction.tasks.mechanisms.contact import contact_separations


def depths(distances, counts, starts, body_paths, filter_paths):
    counts, starts = np.asarray(counts), np.asarray(starts)
    shape = (len(body_paths), len(filter_paths))
    if counts.shape != shape or starts.shape != shape or len(set(body_paths)) != len(body_paths):
        raise ValueError('Contact matrix must match unique physical body and filter paths')
    result = {}
    for i, body in enumerate(body_paths):
        for j, other in enumerate(filter_paths):
            if body == other:
                if counts[i, j]:
                    raise ValueError('A rigid body cannot contact itself as two independent bodies')
                continue
            selected = contact_separations(distances, counts[i:i+1, j], starts[i:i+1, j])
            key = tuple(sorted([body, other]))
            depth = max(0., -float(selected.min())) if len(selected) else 0.
            result[key] = max(result.get(key, 0.), depth)
    return result
