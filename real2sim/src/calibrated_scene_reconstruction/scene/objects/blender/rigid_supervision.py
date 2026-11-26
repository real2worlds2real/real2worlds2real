"""Independent source regions for rigid initialization of compound objects.

This helper does not accept an asset or change final source observations. Whole
physical links are omitted only in a proposal. Retained local face annotations
keep their identity, while global triangle offsets follow the reduced mesh.
"""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np


def mesh_without_links(faces, parts, excluded):
    """Remove complete links and rebase retained global face starts.

    Vertex indices and link-local surface/optical face indices stay unchanged.
    A rigid proposal cannot omit a rigid link or accept ambiguous ownership.
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Actual triangular face indices are required')
    names = [p['name'] for p in parts]
    excluded = set(excluded)
    if len(set(names)) != len(names) or not excluded or not excluded <= set(names):
        raise ValueError('Distinct named parts and known excluded links are required')
    owner = np.full(len(faces), -1, int)
    for index, part in enumerate(parts):
        first, count = part['face_start'], part['triangle_count']
        if (type(first) is not int or type(count) is not int or first < 0 or count <= 0
                or first + count > len(faces) or np.any(owner[first:first+count] >= 0)):
            raise ValueError('Physical links must own disjoint valid triangle ranges')
        owner[first:first+count] = index
        if part['name'] in excluded and part.get('kind', 'rigid') == 'rigid':
            raise ValueError('Rigid initialization cannot omit a rigid link')
    if (owner < 0).any():
        raise ValueError('Every actual face needs a physical link owner')
    keep = np.array([parts[i]['name'] not in excluded for i in owner])
    if (not keep.any() or not any(p['name'] not in excluded and p.get('kind', 'rigid') == 'rigid'
                                 for p in parts)):
        raise ValueError('Rigid initialization must retain rigid geometry')
    prefix = np.r_[0, np.cumsum(keep)]
    retained = []
    for part in parts:
        if part['name'] not in excluded:
            copied = deepcopy(part)
            copied['face_start'] = int(prefix[part['face_start']])
            retained.append(copied)
    return faces[keep], retained


def retained_sources(sources, excluded, excluded_masks):
    """Keep separate observed rigid regions with explicit soft occlusion domains.

    ``excluded_masks`` maps capture basenames to full-image, independently
    reviewed masks of omitted flexible links. ``None`` means that identity was
    unobserved; such a view stays in the original final observations and is
    recorded as unavailable for this proposal. Positive rigid labels are never
    removed by a conflicting flexible label.
    """
    excluded = set(excluded)
    result = []
    for source in sources:
        names = set(source.get('links') or [source['link']])
        overlap = names & excluded
        if overlap:
            if overlap != names:
                raise ValueError('A jointly observed group cannot be partly excluded')
            continue
        transmitters = {str(ref).split('/')[-1] for ref in source.get('transmitting_links', [])}
        if transmitters & excluded:
            raise ValueError('An excluded flexible transmitter needs separate source visibility evidence')
        z, frames, masks, windows = source['observations']
        selected, domains, skipped = [], [], []
        for k, (frame, positive, window) in enumerate(zip(frames, masks, windows)):
            name = Path(str(z['names'][frame])).name
            if name not in excluded_masks or excluded_masks[name] is None:
                skipped.append(name)
                continue
            soft = np.asarray(excluded_masks[name])
            if soft.ndim != 2 or soft.dtype != bool:
                raise ValueError('Independently reviewed full-image boolean occlusions required')
            x, y, w, h, _ = np.asarray(window, float)
            if min(w, h) <= 0 or positive.ndim != 2:
                raise ValueError('Valid source crop and observed part mask required')
            H, W = positive.shape
            transform = np.array([[W/w, 0., -x*W/w], [0., H/h, -y*H/h]])
            occluded = cv2.warpAffine(soft.astype('uint8'), transform, (W, H),
                                     flags=cv2.INTER_NEAREST, borderValue=0) > 0
            domain = np.array(z['_object_domains'][k], bool, copy=True) if '_object_domains' in z else np.ones_like(positive, bool)
            domain &= ~occluded | positive
            selected.append(k)
            domains.append(domain)
        if len(selected) < 8:
            raise ValueError('Independent rigid source region needs eight observed proposal views')
        copied = dict(source)
        cameras = dict(z)
        cameras['_object_domains'] = domains
        cameras['_proposal_unobserved_flexible_views'] = skipped
        copied['observations'] = (cameras, np.asarray(frames)[selected],
                                  [masks[k] for k in selected], np.asarray(windows)[selected])
        copied['proposal_visibility_scope'] = 'Retained rigid source positives; omitted flexible source pixels unknown only during initialization.'
        result.append(copied)
    return result
