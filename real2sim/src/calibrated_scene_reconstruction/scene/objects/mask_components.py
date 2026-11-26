"""Associate contact-connected semantic pixels before fitting any object model."""
import cv2
import numpy as np


def contact_instances(candidates, anchor, gap_pixels, exclude=None):
    """Collect separate source instances touching an observed primary mask.

    Image contact establishes association only, never a joint or attachment.
    Preserve whole masks and decisions; no candidate mesh or desired count is
    used to select pixels, and the association margin does not dilate output.
    """
    anchor = np.asarray(anchor)
    if (anchor.ndim != 2 or anchor.dtype.kind != 'b' or not anchor.any()
            or type(gap_pixels) is not int or not 0 <= gap_pixels <= 8):
        raise ValueError('Observed primary mask and integer gap0..8 required')
    if exclude is not None and (np.asarray(exclude).shape != anchor.shape
                                or np.asarray(exclude).dtype.kind != 'b'):
        raise ValueError('Source exclusion must match the primary mask')
    region = cv2.dilate(anchor.astype('uint8'), np.ones((2*gap_pixels+1,)*2, 'uint8')) > 0
    selected = np.zeros_like(anchor)
    rows = []
    for index, (mask, confidence) in enumerate(candidates):
        mask = np.asarray(mask)
        if mask.shape != anchor.shape or mask.dtype.kind != 'b' or not np.isfinite(confidence):
            raise ValueError('Matching source instance masks and finite confidence required')
        pixels = int(mask.sum())
        conflict = int((mask & exclude).sum()) if exclude is not None else 0
        touches = bool((mask & region).any())
        accept = pixels > 0 and touches and conflict < .5*pixels
        if accept:
            selected |= mask
        rows.append({'index': index, 'confidence': float(confidence), 'pixels': pixels,
                     'touches_primary': touches, 'excluded_pixels': conflict, 'selected': accept})
    return (selected if selected.any() else None), {
        'instances': rows, 'gap_pixels': gap_pixels, 'selected_pixels': int(selected.sum()),
        'source': 'Current semantic instances and observed primary mask; no model geometry',
        'physical_attachment_inferred': False}


def policy(session, obj):
    if not hasattr(session, 'scene_graph'):
        return None
    value = (session.scene_graph['dynamic_objects'][obj].get('fit') or {}).get('mask_component_gap_pixels')
    if value is not None and (type(value) is not int or not 0 <= value <= 8):
        raise ValueError('mask_component_gap_pixels requires an integer0..8')
    return value


def anchored(parts, gap_pixels):
    """Keep complete first-part semantics, including pieces split by occlusion."""
    if (not isinstance(parts, list) or not parts or type(gap_pixels) is not int
            or not 0 <= gap_pixels <= 8):
        raise ValueError('Semantic part masks and an association gap0..8 required')
    masks = [np.asarray(m) for m in parts]
    if any(m.ndim != 2 or m.dtype.kind != 'b' or m.shape != masks[0].shape or not m.any() for m in masks):
        raise ValueError('Matching nonempty semantic part masks required')
    selected = masks[0].copy()
    original = np.logical_or.reduce(masks)
    for part in masks[1:]:
        _, labels = cv2.connectedComponents(part.astype('uint8'))
        touched = cv2.dilate(selected.astype('uint8'), np.ones((2*gap_pixels+1,)*2, 'uint8')) > 0
        candidates = np.unique(labels[touched & part])
        if not len(candidates):
            return None, {'reason': 'No observed contact with the semantic anchor; retain for identity review'}
        selected |= part & np.isin(labels, candidates)
    return selected, {'algorithm': 'contact_with_first_semantic_part_v1',
                      'original_pixels': int(original.sum()), 'selected_pixels': int(selected.sum()),
                      'discarded_pixels': int((original & ~selected).sum()), 'gap_pixels': gap_pixels,
                      'source': 'Independent first-part segmentation and contact labels; no rendered geometry'}


def query_anchor(mask, side):
    """A part-search pixel from observed first-part semantics, in image axes."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind != 'b' or not mask.any() or side not in {'top','bottom','left','right','center'}:
        raise ValueError('Require an observed semantic mask and image-axis part anchor')
    y,x = np.nonzero(mask)
    if side in {'top','bottom'}:
        keep = y <= np.percentile(y, 5) if side == 'top' else y >= np.percentile(y, 95)
    elif side in {'left','right'}:
        keep = x <= np.percentile(x, 5) if side == 'left' else x >= np.percentile(x, 95)
    else:
        keep = np.ones(len(x), bool)
    return [float(np.median(x[keep])), float(np.median(y[keep]))]


def seeded(mask, pixel, gap_pixels):
    mask = np.asarray(mask)
    pixel = np.asarray(pixel, float)
    if (mask.ndim != 2 or mask.dtype.kind != 'b' or not mask.any()
            or pixel.shape != (2,) or not np.isfinite(pixel).all()
            or type(gap_pixels) is not int or not 0 <= gap_pixels <= 8):
        raise ValueError('Require a nonempty semantic mask, finite observed search pixel and gap0..8')
    # Dilation associates labels across a small segmentation gap. It never adds
    # pixels to the returned observation or uses a candidate rendered surface.
    expanded = cv2.dilate(mask.astype('uint8'), np.ones((2*gap_pixels+1,)*2, 'uint8'))
    _, labels = cv2.connectedComponents(expanded)
    yy, xx = np.nonzero(mask)
    nearest = int(np.argmin((xx-pixel[0])**2+(yy-pixel[1])**2))
    component = labels[yy[nearest], xx[nearest]]
    selected = mask & (labels == component)
    return selected, {'components': int(len(np.unique(labels[mask]))),
                      'original_pixels': int(mask.sum()), 'selected_pixels': int(selected.sum()),
                      'discarded_pixels': int((mask & ~selected).sum()),
                      'gap_pixels': gap_pixels, 'search_pixel': pixel.tolist(),
                      'source': 'Semantic mask and existing projected object search point; no model-render error'}
