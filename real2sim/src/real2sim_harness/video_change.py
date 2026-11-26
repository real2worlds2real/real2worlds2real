"""Localized visual-change diagnostics for sparse and full-frame demonstrations."""

from __future__ import annotations

import cv2
import numpy as np


def frame_change(reference, frame):
    difference = cv2.absdiff(reference, frame)
    magnitude = np.max(difference, axis=2)
    # Suppress isolated compression/sampling pixels, retaining connected thin features.
    smooth = cv2.GaussianBlur(magnitude, (3, 3), 0)
    significant = (smooth >= 12).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(significant, connectivity=8)
    return {
        "mean_change": float(difference.mean()),
        "changed_pixels": int(significant.sum()),
        "largest_changed_region_px": int(stats[1:, cv2.CC_STAT_AREA].max())
        if len(stats) > 1
        else 0,
    }


def has_visible_change(largest_changed_region):
    """A sparse moving object need not change the mean of a large static background."""
    return largest_changed_region >= 32
