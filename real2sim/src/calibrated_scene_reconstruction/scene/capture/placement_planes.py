"""Observed floor/support correspondences for provisional camera placement."""
import json
import numpy as np
import cv2


def masked_tracks(rec, image, masks, ids, minimum=30):
    """Raw COLMAP points actually tracked in the masked photo, not points behind it.

    The capture is undistorted with the same K as the COLMAP image. A resolution
    change is handled by scaling observed pixels into the mask image.
    """
    if not ids:
        if minimum==0:
            return np.empty((0,3),dtype=float)
        raise ValueError('Photo-plane placement needs floor and support instance IDs')
    mask = np.any(masks[ids], axis=0)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    h, w = mask.shape
    cam = rec.cameras[image.camera_id]
    points = []
    for observation in image.points2D:
        if not observation.has_point3D():
            continue
        u, v = observation.xy * [w / cam.width, h / cam.height]
        if not np.isfinite([u, v]).all() or not (0 <= u < w and 0 <= v < h) or not mask[int(v), int(u)]:
            continue
        point = rec.points3D[observation.point3D_id]
        if point.track.length() >= 3 and point.error < 1.5 and np.isfinite(point.xyz).all():
            points.append(point.xyz)
    if len(points) < minimum:
        raise ValueError(f'Photo-plane placement has only {len(points)} reliable masked tracks; at least{minimum} required')
    return np.array(points, dtype=float).reshape(-1, 3)


def from_photo(session, rec):
    """Read semantically identified surfaces without measured task/support sizes."""
    kf = json.loads((session / 'capture/keyframes.json').read_text())
    name = f"0_{kf['photo']:05d}.jpg"
    photo = next((image for image in rec.images.values() if image.name == name), None)
    if photo is None:
        raise ValueError('Photo-plane placement requires the reference photo to be registered')
    instances = json.loads((session / 'parsing/instances.json').read_text())
    floor_ids = [i['id'] for i in instances if i['label'] == 'floor']
    graph = session.scene_graph
    support = graph['furniture'][graph['task']['support']]
    support_ids = support.get('faces', {}).get('top', [])
    masks = session.masks()
    # Primary SAM proposals need the same independent support conflict check
    # as supplemental views. A large tabletop labelled floor can otherwise
    # collapse the inferred height separation despite many reliable tracks.
    from .plane_observations import select_floor
    independent_support_ids=sorted(set(support_ids)|{i['id'] for i in instances
        if i['label'] in {'table','desk','tabletop','counter','countertop'}})
    candidates=[{'mask':masks[i['id']],'confidence':i.get('score',0.),'prompt':i['label']}
                for i in instances if i['id'] in floor_ids]
    if candidates:
        floor_mask,floor_decisions=select_floor(candidates,[masks[i] for i in independent_support_ids],allow_empty=True)
        for identifier,decision in zip(floor_ids,floor_decisions):decision['instance_id']=identifier
    else:
        floor_mask=np.zeros(masks.shape[1:],bool);floor_decisions=[]
    original_floor_ids=list(floor_ids)
    floor_ids=[row['instance_id'] for row in floor_decisions if row['accepted']]
    semantic_review={'photo':name,'original_floor_ids':original_floor_ids,'accepted_floor_ids':floor_ids,
        'independent_support_ids':independent_support_ids,'floor_candidates':floor_decisions,
        'scope':'Current primary source labels only; no CAD, candidate residual or measured height used'}
    (session/'qa').mkdir(parents=True,exist_ok=True)
    (session/'qa/placement_plane_semantics.json').write_text(json.dumps(semantic_review,indent=2)+'\n')
    collected, extra_frames = [], {}
    for role, ids in [('floor', floor_ids), ('support', support_ids)]:
        # A whole-table instance may include legs, rails and the attached rig.
        # An explicitly acquired surface observation of this same photo takes
        # precedence for plane estimation; do not union the known broad label
        # back into it. Other perception consumers retain the original masks.
        primary_surface = session/'parsing/placement_plane_masks'/('support__'+name[:-4]+'.png')
        override = role == 'support' and primary_surface.exists()
        if override:
            surface = cv2.imread(str(primary_surface), cv2.IMREAD_GRAYSCALE)
            if surface is None:
                raise ValueError('Unreadable primary support surface observation')
            points = [masked_tracks(rec, photo, (surface < 128)[None], [0], minimum=0)]
        else:
            points = [masked_tracks(rec,photo,floor_mask[None],[0],minimum=0) if role=='floor'
                      else masked_tracks(rec,photo,masks,ids,minimum=0)]
        extra_frames[role] = []
        if override:
            extra_frames[role].append(name)
        by_name = {image.name: image for image in rec.images.values()}
        for path in sorted((session/'parsing/placement_plane_masks').glob(role+'__*.png')):
            filename = path.stem.split('__', 1)[1]+'.jpg'
            if override and filename == name:
                continue
            if filename not in by_name:
                raise ValueError('Placement-plane mask does not belong to this registered capture')
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) < 128
            points.append(masked_tracks(rec, by_name[filename], mask[None], [0], minimum=0))
            extra_frames[role].append(filename)
        # Multiple observations of the same triangulated track are one point.
        unique = np.unique(np.concatenate(points), axis=0)
        if len(unique) < 30:
            request = session/'qa/placement_plane_request.json'
            request.parent.mkdir(parents=True, exist_ok=True)
            candidates = [f'0_{index:05d}.jpg' for index in [kf['photo'], *kf['views']]]
            candidates = [frame for frame in candidates if frame in by_name and frame not in extra_frames[role]]
            if len(candidates) > 8:
                candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, 8).astype(int)]
            request.write_text(json.dumps({'role': role, 'points': len(unique), 'minimum': 30,
                               'frames': candidates,
                               'action': 'Acquire additional actual surface masks from this registered video'},
                              indent=2)+'\n')
            raise ValueError(f'{role} placement has only {len(unique)} independent reliable tracks; at least30 required')
        collected.append(unique)
    (session/'qa/placement_plane_request.json').unlink(missing_ok=True)
    return (*collected, {'photo': name, 'floor_ids': floor_ids, 'support_ids': support_ids,
                        'primary_floor_semantics':semantic_review,
                        'additional_mask_frames': extra_frames})
