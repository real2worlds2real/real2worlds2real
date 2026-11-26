"""Triangle unions sampled at calibrated OpenCV integer pixel centres.

This is binary geometric coverage, not a drawing operation or antialiasing.
The implementation uses bounded NumPy scanline batches and needs no GPU,
compiler or optional geometry dependency. Boundary points belong to the union;
zero-area triangles do not cover pixels. It does not resolve depth/occlusion.
"""
import numpy as np


def triangle_mask(uv, faces, shape):
    """Return whether each integer ``(x, y)`` lies in any projected triangle.

    Unlike conservative polygon drawing, the sample footprint has zero area.
    Face winding and overlapping faces do not change coverage. Triangles may
    extend outside the image; clipping scanlines bounds temporary allocations.
    """
    if len(shape) != 2 or any(int(n) != n or n < 0 for n in shape):
        raise ValueError('Image shape must contain two nonnegative integers')
    h, w = map(int, shape)
    image = np.zeros((h, w), bool)
    points = np.asarray(uv, dtype=np.float64)
    indices = np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Projected vertices must be finite 2D coordinates')
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in 'iu':
        raise ValueError('Triangle faces must be an integer (N, 3) array')
    if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
        raise ValueError('Triangle index is outside the vertex array')
    if not h or not w or not len(indices):
        return image
    tri = points[indices]
    edge0, edge1 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    area = edge0[:, 0]*edge1[:, 1]-edge0[:, 1]*edge1[:, 0]
    lower, upper = tri.min(axis=1), tri.max(axis=1)
    valid = ((area != 0) & (upper[:, 0] >= 0) & (lower[:, 0] <= w-1)
             & (upper[:, 1] >= 0) & (lower[:, 1] <= h-1))
    tri, lower, upper = tri[valid], lower[valid], upper[valid]
    first = np.ceil(np.clip(lower[:, 1], 0, h-1)).astype(np.int64)
    last = np.floor(np.clip(upper[:, 1], 0, h-1)).astype(np.int64)
    count = np.maximum(0, last-first+1)
    cumulative = np.r_[0, np.cumsum(count)]
    start = 0
    while start < len(tri):
        # The row budget controls the dominant temporary arrays. The face
        # budget also bounds work for many subpixel triangles with no rows.
        end = min(len(tri), start+16384,
                  max(start+1, int(np.searchsorted(cumulative, cumulative[start]+262144,
                                                  side='right'))-1))
        lengths = count[start:end]
        owners = np.repeat(np.arange(end-start), lengths)
        if not len(owners):
            start = end
            continue
        offsets = np.repeat(np.cumsum(lengths)-lengths, lengths)
        rows = np.repeat(first[start:end], lengths)+np.arange(len(owners))-offsets
        selected = tri[start:end][owners]
        lo, hi = np.full(len(rows), np.inf), np.full(len(rows), -np.inf)
        for i in range(3):
            a, b = selected[:, i], selected[:, (i+1) % 3]
            dy = b[:, 1]-a[:, 1]
            active = ((dy != 0) & (rows >= np.minimum(a[:, 1], b[:, 1]))
                      & (rows <= np.maximum(a[:, 1], b[:, 1])))
            x = np.zeros(len(rows))
            np.divide((rows-a[:, 1])*(b[:, 0]-a[:, 0]), dy, out=x, where=active)
            x += a[:, 0]
            lo = np.minimum(lo, np.where(active, x, np.inf))
            hi = np.maximum(hi, np.where(active, x, -np.inf))
        # Closed boundaries, with a few ULPs for arithmetic at an exact
        # integer edge. This is numerical roundoff, not a pixel-sized margin.
        tolerance = 16*np.finfo(np.float64).eps*np.maximum(1, np.maximum(abs(lo), abs(hi)))
        keep = np.isfinite(lo) & np.isfinite(hi) & (hi >= 0) & (lo <= w-1)
        left = np.ceil(np.clip(lo[keep]-tolerance[keep], 0, w-1)).astype(np.int64)
        right = np.floor(np.clip(hi[keep]+tolerance[keep], 0, w-1)).astype(np.int64)
        rows = rows[keep]
        keep = right >= left
        begins = rows[keep]*(w+1)+left[keep]
        ends = rows[keep]*(w+1)+right[keep]+1
        changes = (np.bincount(begins, minlength=h*(w+1))
                   - np.bincount(ends, minlength=h*(w+1))).reshape(h, w+1)
        image |= np.cumsum(changes[:, :w], axis=1) > 0
        start = end
    return image
