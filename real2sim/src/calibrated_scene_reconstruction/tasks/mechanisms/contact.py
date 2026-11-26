"""Generic native contact and bounded-wrench measurements."""
import numpy as np


def contact_separations(distances, counts, starts):
    """Select only active pair ranges from the native shared contact buffer."""
    values = np.asarray(distances).reshape(-1)
    counts, starts = np.asarray(counts).reshape(-1), np.asarray(starts).reshape(-1)
    if counts.shape != starts.shape or (counts < 0).any():
        raise ValueError('Invalid contact-pair buffer ranges')
    pieces = []
    for count, start in zip(counts, starts):
        if not count:
            continue
        if count != int(count) or start != int(start) or start < 0 or start+count > len(values):
            raise ValueError('Contact-pair range exceeds native buffer')
        pieces.append(values[int(start):int(start+count)])
    selected = np.concatenate(pieces) if pieces else values[:0]
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite active native contact separation')
    return selected
