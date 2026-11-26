"""Independent semantic part observations for native physical-state checks.

Targets come from archived raw part labels in the approved observation windows,
never from the model being evaluated. Existing identity and occlusion reviews
still apply; small flexible parts cannot hide behind whole-assembly averages.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.objects.blender.evidence import load as load_whole
from calibrated_scene_reconstruction.scene.objects.object_refine import pieces
from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts


def collection_primary(cache, prompts, camera_names):
    """Recover the independently acquired primary mask from collection anchors.

    Collection acquisition archives the original primary before unioning loose
    instances. Never subtract a candidate render or infer missing part labels.
    """
    manifest = json.loads(cache.with_suffix('.collections.json').read_text())
    names, labels = [], []
    with np.load(cache.with_suffix('.collections.acquired.npz')) as raw:
        for row in manifest:
            prefix = row['array_prefix']
            fields = prefix.split('_')
            if len(fields) != 4 or fields[0] != 'view' or fields[2] != 'part':
                raise ValueError('Invalid archived collection view/part index')
            index, part = int(fields[1]), int(fields[3])
            if (not 0 <= index < len(camera_names) or not 1 <= part < len(prompts)
                    or row['prompt'] != prompts[part]
                    or str(camera_names[index]) != row['frame']):
                raise ValueError('Collection anchor camera or prompt correspondence changed')
            label = raw[prefix+'_anchor']
            if label.ndim != 2 or label.dtype != np.bool_ or not label.any():
                raise ValueError('Independent primary anchor must be a nonempty native boolean mask')
            name = Path(row['frame']).name
            if name in names:
                if not np.array_equal(label, labels[names.index(name)]):
                    raise ValueError('Collection primary anchors disagree for the same view')
            else:
                names.append(name); labels.append(label.copy())
    if not labels:
        raise ValueError('No independent primary collection anchors')
    return names, np.stack(labels)


def collection_selected(cache, prompts, camera_names, prompt):
    """One independently acquired collection label; keep member identity unknown."""
    collection_primary(cache, prompts, camera_names)  # Validate archived view/prompt binding.
    manifest = json.loads(cache.with_suffix('.collections.json').read_text())
    names, labels = [], []
    with np.load(cache.with_suffix('.collections.acquired.npz')) as raw:
        for row in manifest:
            if row['prompt'] != prompt:
                continue
            prefix = row['array_prefix']; selected = np.zeros_like(raw[prefix+'_anchor'])
            for instance in row['instances']:
                if instance['selected']:
                    mask = raw[prefix+f'_proposal_{instance["index"]}']
                    if mask.dtype != np.bool_ or mask.shape != selected.shape:
                        raise ValueError('Collection instance must retain its native boolean image')
                    selected |= mask
            name = Path(row['frame']).name
            if name in names:
                raise ValueError('Duplicate collection label for one camera and prompt')
            names.append(name); labels.append(selected)
    if not labels:
        raise ValueError('No independently acquired collection with requested prompt')
    return names, np.stack(labels)


def observations(session, obj, prompt, *, exterior_to=None, min_pixels=32):
    z, frames, masks, windows = load_whole(session, obj)
    cache = session/'splat/object_fit'/f'{obj}.npz'
    with np.load(cache, allow_pickle=True) as data:
        prompts = str(data['prompt']).split('|')
    if prompts.count(prompt) != 1:
        raise ValueError('Select one independently acquired semantic part prompt')
    if exterior_to is not None and (prompts.count(exterior_to) != 1 or exterior_to == prompt):
        raise ValueError('Exterior comparison needs another independently acquired part prompt')
    if type(min_pixels) is not int or min_pixels < 16:
        raise ValueError('A positive source-pixel visibility budget is required')
    with np.load(cache.with_suffix('.acquired.npz'), allow_pickle=True) as acquired:
        if str(acquired['prompt']).split('|') != prompts:
            raise ValueError('Archived semantic part ordering differs from current acquisition')
    path = cache.with_suffix('.components.acquired.npz')
    if not path.exists() and exterior_to is None:
        names, labels = (collection_primary(cache, prompts, z['names']) if prompt == prompts[0]
                         else collection_selected(cache, prompts, z['names'], prompt))
        references = None
    else:
        with np.load(path) as raw:
            names = raw['names'].tolist()
            if len(set(names)) != len(names) or raw['part_masks'].shape[1] != len(prompts):
                raise ValueError('Part observations lack unique frame and prompt correspondence')
            labels = raw['part_masks'][:, prompts.index(prompt)]
            references = raw['part_masks'][:, prompts.index(exterior_to)] if exterior_to is not None else None
    from .part_label_review import load as reviewed_components
    labels,reviewed_names=reviewed_components(session,obj,prompt,names,labels,
                                             [Path(str(z['names'][frame])).name for frame in frames])
    selected, retained, domains, rejected = [], [], [], []
    for index, (frame, mask, window) in enumerate(zip(frames, masks, windows)):
        name = Path(str(z['names'][frame])).name
        if name not in names:
            raise ValueError('Part observations missing an approved view: '+name)
        x0, y0, width, height, _ = window
        if not np.allclose([x0, y0, width, height], np.rint([x0, y0, width, height])):
            raise ValueError('Raw part-label windows must be integral source-image crops')
        x0, y0, width, height = map(int, [x0, y0, width, height])
        part = labels[names.index(name)][y0:y0+height, x0:x0+width]
        if part.shape != (height, width):
            raise ValueError('Part label window lies outside its source frame')
        part = cv2.resize(part.astype('uint8'), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)>0
        part &= mask
        domain = z['_object_domains'][index].copy() if '_object_domains' in z else np.ones(mask.shape, bool)
        # A negative semantic query can accidentally select the task part
        # itself. Do not turn that contradictory label into an invisible part.
        # This source-only test precedes all model projections and fitting.
        conflict=float((part & ~domain).sum()/max(1,part.sum()))
        if conflict>.25:
            rejected.append({'frame':name,'reason':'Independent occluder/background labels conflict with the target part',
                             'conflicting_part_share':conflict})
            continue
        if references is not None:
            ref = references[names.index(name)][y0:y0+height, x0:x0+width]
            ref = cv2.resize(ref.astype('uint8'), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            ambiguous=cv2.dilate(ref,np.ones((5,5),np.uint8))>0
            if name in reviewed_names:
                # Broad enclosing queries can include the exterior sheet.
                # Reviewed visible foreground owns these pixels. Hidden or
                # merged parts cannot receive this source-only review status.
                ambiguous &= ~part
            overlap=float((part&ambiguous).sum()/max(1,part.sum()))
            if overlap>.25:
                rejected.append({'frame':name,'reason':'Exterior target and enclosing-part semantics are merged; inspect exposure alternatives or relabel source',
                                 'ambiguous_part_share':overlap})
                continue
            # White-on-white or overlapping labels do not identify a visible
            # free part. Evaluate the independently observed exterior only;
            # never use the simulated geometry to choose this domain.
            domain &= ~ambiguous
            part &= domain
        if part.sum() < min_pixels:
            rejected.append({'frame': name, 'reason': 'No sufficient independent visible target part in source labels'})
            continue
        selected.append(part)
        retained.append(index)
        domains.append(domain)
    if len(retained) < 8:
        raise ValueError('Need at least8 independently observed target-part views; inspect source-only visibility/label conflicts')
    z['_object_domains'] = domains
    z['_part_unobserved_views'] = rejected
    z['_part_scope'] = {'prompt': prompt, 'exterior_to': exterior_to,
                        'source_only_visibility_selection': True, 'min_pixels': min_pixels}
    return z, np.asarray(frames)[retained], selected, np.asarray(windows)[retained]


def compare(vertices, faces, target_faces, observed, *, device='cpu', renders=False,transmitting_faces=None):
    """Visibility and residuals for this physical state, without acceptance tuning."""
    z, frames, masks, windows = observed
    if z.get('_source_context'):
        import trimesh
        from .blender.context_visibility import compose
        faces=np.asarray(faces);target_faces=np.asarray(target_faces)
        if transmitting_faces is not None:
            keep=~np.asarray(transmitting_faces,bool);faces=faces[keep];target_faces=target_faces[keep]
        combined,target_faces=compose(trimesh.Trimesh(vertices,faces,process=False),z,target_faces)
        vertices,faces=combined.vertices,combined.faces;transmitting_faces=None
    state = VisibleParts(vertices, faces, target_faces, device=device,transmitting_faces=transmitting_faces)
    ious, local, predictions = [], [], []
    for index, (frame, mask, window) in enumerate(zip(frames, masks, windows)):
        predicted = state.mask(z['K'][frame], z['R_world_cam'][frame], z['centre'][frame], window, mask.shape)
        if '_object_domains' in z:
            domain = z['_object_domains'][index]
            predicted &= domain
            mask = mask & domain
        if not mask.any():
            raise ValueError('Occlusion domain leaves no observed target material')
        ious.append(float((mask & predicted).sum()/max((mask | predicted).sum(), 1)))
        local.append(sum(piece[0] for piece in pieces(mask, predicted, .003)))
        if renders:
            predictions.append(predicted)
    return {'frames': frames.tolist(), 'iou_median': float(np.median(ious)),
            'local_difference': float(np.median(local)), 'ious': ious, 'local_per_frame': local,
            'scope': 'Visible target part only; actual native triangles, source labels and calibrated cameras'}, predictions
