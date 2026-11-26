"""Exact silhouettes in the existing metric cameras; same raster and pieces as object_refine."""
from pathlib import Path
import json

import cv2
import numpy as np
import trimesh

from calibrated_scene_reconstruction.scene.objects.object_refine import _frame, pieces, raster
from calibrated_scene_reconstruction.scene.objects.blender.contract import write_json
from calibrated_scene_reconstruction.scene.objects.mask_cache import read_masks

IOU_MIN, LOCAL_MAX, PIECE_MIN, MIN_FRAMES = 0.85, 0.02, 0.003, 8


def search_objective(metrics):
    """Keep the search directed toward every unchanged numeric source gate.

    The reported observation loss remains independent of this proposal
    objective. Passing acceptance still requires the actual CPU measurements.
    """
    if 'boundary_search_loss' in metrics:
        # Search only. Candidate selection still prioritizes metrics['ok'];
        # the raw source gates and their thresholds below are unchanged.
        return float(metrics['boundary_search_loss'])
    limits=metrics.get('thresholds',{});iou_min=limits.get('iou_min',IOU_MIN);local_max=limits.get('local_max',LOCAL_MAX)
    constraints=[(metrics['iou_median'],metrics['local_difference'],iou_min,local_max)]
    constraints.extend((part['iou_median'],part['local_difference'],IOU_MIN,LOCAL_MAX)
                       for part in metrics.get('source_parts',[]))
    penalty=sum(max(0.,minimum-iou)/minimum+max(0.,local-maximum)/maximum
                for iou,local,minimum,maximum in constraints)
    return float(metrics['loss']+penalty)


def load(s, obj, *, parts=False):
    # A file-only core Session cannot expose scene observation policies. Do not
    # silently lose exclusions, visibility domains or functional-part evidence
    # when a diagnostic imports the wrong session adapter.
    if not hasattr(s, 'scene_graph'):
        raise TypeError('Object source evidence requires calibrated_scene_reconstruction.scene.session.Session; a file-only session omits scene policies')
    z = dict(np.load(s / 'parsing/posed_frames/cameras.npz'))
    path = s / 'splat/object_fit' / f'{obj}.npz'
    if not path.exists():
        raise ValueError(f'{obj}: no silhouette cache; run recon.object_fit --observations-only first')
    with np.load(path, allow_pickle=True) as d:
        from calibrated_scene_reconstruction.scene.objects.observation_history import check_policy, object_request
        request = object_request(s, obj)
        check_policy(d, request)
        from calibrated_scene_reconstruction.scene.objects.collection_observations import validate_context_cache
        context_inputs = validate_context_cache(s, d, request)
        from calibrated_scene_reconstruction.scene.objects.mask_components import policy as component_policy
        cached_policy = json.loads(str(d['mask_component_policy'])) if 'mask_component_policy' in d else None
        if cached_policy != component_policy(s, obj):
            raise ValueError('Semantic component policy changed; refresh object observations')
        frames = np.asarray(d['frames'], int)
        masks = read_masks(d)
        windows = np.asarray(d['windows'], float)
        from calibrated_scene_reconstruction.scene.objects.observation_history import color_inputs, check_color
        check_color(d, color_inputs(s, z, frames))
    if len(set(frames)) < MIN_FRAMES or len(masks) != len(frames) or len(windows) != len(frames):
        raise ValueError(f'{obj}: need at least {MIN_FRAMES} cached frames with masks and windows')
    if any(not m.any() for m in masks) or (frames < 0).any() or (frames >= len(z['K'])).any():
        raise ValueError('empty silhouette or invalid camera index')
    from calibrated_scene_reconstruction.scene.objects.collection_review import load as collection_review
    masks,review_inputs=collection_review(s,obj,z,frames,masks,windows)
    context_inputs={**context_inputs,**review_inputs}
    from calibrated_scene_reconstruction.scene.objects.object_label_union import load as object_identity
    masks,identity_inputs=object_identity(s,obj,z,frames,masks,windows)
    context_inputs={**context_inputs,**identity_inputs}
    from .observation_domain import load as domains
    masks, known, files = domains(s, obj, z, frames, masks, windows)
    if context_inputs:
        z['_object_domain_inputs'] = context_inputs
    if known is not None:
        z['_object_domains'] = known
        z['_object_domain_inputs'] = {**context_inputs, **files}
    search_metric=request.get('search_metric','gate_penalty')
    if search_metric not in {'gate_penalty','boundary_distance'}:
        raise ValueError('Choose an explicit supported source search metric')
    if search_metric=='boundary_distance':
        from .boundary_search import prepare
        z['_boundary_search']=[prepare(mask,None if known is None else known[k])
                               for k,mask in enumerate(masks)]
    from .context_visibility import load as source_context
    context=source_context(s,obj)
    if context is not None:
        z['_source_context']=context
        z['_object_domain_inputs']={**z.get('_object_domain_inputs',{}),**context['inputs']}
    if parts:
        from .part_evidence import attach
        attach(s,obj,z)
        from .material_reference import attach as material_attach
        material_attach(s,obj,z,frames)
    return z, frames, masks, windows


def placed(V, F, pose):
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    R = rotation(pose)
    return trimesh.Trimesh(np.asarray(V) @ R.T + [*pose['xy'], pose['support_z']], F, process=False)


def score(V, F, pose, observations, renders=False, rasterizer=None, model_parts=None):
    z, frames, masks, windows = observations
    excluded=z.get('_proposal_excluded_links',[])
    if excluded:
        if not model_parts or not set(excluded)<={p['name'] for p in model_parts}:
            raise ValueError('Rigid proposal requires preserved physical link faces')
        from .rigid_supervision import mesh_without_links
        F,model_parts=mesh_without_links(F,model_parts,excluded)
    mesh = placed(V, F, pose)
    context_mesh=context_target=context_native=None
    if z.get('_source_context'):
        from .context_visibility import compose
        context_mesh,context_target=compose(mesh,z)
        if rasterizer is None:
            from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts
            context_native=VisibleParts(context_mesh.vertices,context_mesh.faces,context_target,device='cpu')
    ious, shares, predicted, native_pixels, boundary_distances = [], [], [], [], []
    for k, (i, mask, win) in enumerate(zip(frames, masks, windows)):
        args=(z['K'][i],z['R_world_cam'][i],z['centre'][i],win,mask.shape)
        if context_mesh is None:ren=(rasterizer or raster)(mesh,*args)
        elif context_native is not None:ren=context_native.mask(*args)
        else:ren=rasterizer.visible(context_mesh,*args,target_faces=context_target)
        if '_object_domains' in z:
            ren &= z['_object_domains'][k]
        ious.append(float((ren & mask).sum() / max((ren | mask).sum(), 1)))
        shares.append(sum(p[0] for p in pieces(mask, ren, PIECE_MIN)))
        native_pixels.append(float(mask.sum()/win[4]**2))
        if '_boundary_search' in z:
            from .boundary_search import distance
            boundary_distances.append(distance(ren,z['_boundary_search'][k]))
        if renders:
            predicted.append(ren)
    iou, local = float(np.median(ious)), float(np.median(shares))
    # All cached views contribute; no new outlier filtering in the Blender route.
    result={'iou_median': iou, 'local_difference': local, 'ious': ious, 'local_per_frame': shares,
            'frames': frames.tolist(), 'ok': bool(iou >= IOU_MIN and local <= LOCAL_MAX and len(frames) >= MIN_FRAMES),
            'thresholds': {'iou_min': IOU_MIN, 'local_max': LOCAL_MAX, 'min_frames': MIN_FRAMES},
            'loss': float(1 - np.average(ious,weights=native_pixels) + np.average(shares,weights=native_pixels)),
            'loss_weighting': 'native observed object pixels; all cached views remain in the unweighted acceptance gates',
            'native_mask_pixels': native_pixels}
    if z.get('_source_parts'):
        from .part_evidence import score as part_score
        result['source_parts']=part_score(mesh,pose,z,model_parts,rasterizer)
        result['ok'] &= all(part['ok'] for part in result['source_parts'])
        result['loss'] += float(np.mean([part['loss'] for part in result['source_parts']]))
    if z.get('_source_material'):
        from .material_reference import score as material_score
        result['source_material']=material_score(z,model_parts)
        result['ok'] &= all(item['ok'] for item in result['source_material'])
        result['loss'] += sum(item['loss'] for item in result['source_material'])
    if excluded:
        result.update(ok=False,proposal_only=True,
                      scope='Body initialization only; every full-object and functional-part gate remains mandatory')
    if z.get('_source_context'):
        from .context_visibility import report
        result['source_context']=report(z['_source_context'])
        if not z['_source_context']['parent_source_geometry_pass']:
            result.update(ok=False,proposal_only=True,blocked_by_parent_source_geometry=True)
    if boundary_distances:
        result['boundary_search_loss']=float(np.average(boundary_distances,weights=native_pixels))
        result['boundary_search_per_frame']=boundary_distances
        result['boundary_search_scope']='Proposal objective only; unchanged full-object and part acceptance gates.'
    return result,predicted


def run(s, obj, mesh_path, pose, output, round_number):
    observations = load(s, obj,parts=True)
    with np.load(mesh_path) as d:
        from .material_reference import mesh_parts
        parts=mesh_parts(d)
        result, predicted = score(d['V'], d['F'], pose, observations, renders=True,model_parts=parts)
        from .source_acceptance import policy as acceptance_policy, controls as acceptance_controls
        requested = acceptance_policy(s, obj)
        if requested:
            result['source_pose_controls'] = acceptance_controls(d['V'], d['F'], pose, observations, parts, requested)
        if observations[0].get('_source_parts'):
            from .part_evidence import sheets
            result['source_part_visualizations']=sheets(s,placed(d['V'],d['F'],pose),observations[0],parts,
                                                        Path(output)/'source_parts',frame_reader=_frame)
    z, frames, masks, windows = observations
    order = np.argsort(result['local_per_frame'])
    selected = list(dict.fromkeys([*order[-4:][::-1], *order[::max(1, len(order) // 4)]]))[:8]
    rows = []
    for k in selected:
        mask, ren = masks[k], predicted[k]
        img = _frame(s, z, int(frames[k]), windows[k], mask.shape)
        overlay = img.copy()
        for m, colour in ((mask, (240, 40, 40)), (ren, (40, 220, 40))):
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, colour, 1)
        diff = (img * 0.3).astype(np.uint8)
        diff[mask & ren] = (235, 235, 235)
        diff[mask & ~ren] = (230, 40, 40)
        diff[ren & ~mask] = (40, 210, 40)
        if '_object_domains' in z:
            diff[~z['_object_domains'][k]] = (65, 80, 110)
        tiles = [cv2.resize(t, (256, 256)) for t in (img, overlay, diff)]
        cv2.putText(tiles[1], f'IoU {result["ious"][k]:.3f}', (4, 20), 0, 0.5, (255, 255, 0), 1)
        cv2.putText(tiles[2], f'local {100*result["local_per_frame"][k]:.2f}%', (4, 20), 0, 0.5, (255, 255, 0), 1)
        rows.append(np.hstack(tiles))
    qa = s / 'qa/objects' / obj / f'round_{round_number}.jpg'
    qa.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa), np.vstack(rows)[:, :, ::-1])
    result['sheet'] = str(qa.relative_to(s.dir))
    write_json(Path(output) / 'evidence.json', result)
    # Preserve the existing report and OBJ review interfaces.
    path = s / 'splat/object_refine.json'
    all_ = json.loads(path.read_text()) if path.exists() else {}
    local = {'pieces_share_median': result['local_difference'],
             'frames_with_piece': float(np.mean(np.asarray(result['local_per_frame']) > 0))}
    all_[obj] = {'route': 'blender', 'local_kit': local, 'local_carved': local, 'sheet': result['sheet'],
                 'frames': len(frames), 'round': round_number}
    write_json(path, all_)
    (s / 'qa/review').mkdir(parents=True, exist_ok=True)
    (s / 'qa/review' / f'OBJ.{obj}.ran').touch()
    return result


def opening_sheet(s, obj, pose, opening, output):
    """The declared opening's rim projected into observed frames; the agent checks visible rim geometry separately
    from the whole-object silhouette. Hidden depth is not measured by this overlay."""
    from calibrated_scene_reconstruction.scene.objects.object_hull import project
    if opening is None:
        return None
    z, frames, masks, windows = load(s, obj)
    angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
    circle = np.c_[opening['xy'][0]+opening['radius']*np.cos(angles),
                   opening['xy'][1]+opening['radius']*np.sin(angles), np.full(96,opening['rim_z'])]
    from calibrated_scene_reconstruction.assets.object_pose import rotation
    R = rotation(pose)
    circle = circle @ R.T + [*pose['xy'],pose['support_z']]
    middle = circle.mean(0)
    elevation = (z['centre'][frames,2]-middle[2]) / np.linalg.norm(z['centre'][frames]-middle,axis=1)
    tiles=[]
    for k in np.argsort(-elevation)[:8]:
        i,win=frames[k],windows[k]
        img=_frame(s,z,int(i),win,masks[k].shape)
        u,v,d=project(circle,z['K'][i],z['R_world_cam'][i],z['centre'][i])
        pts=np.c_[(u-win[0])*win[4],(v-win[1])*win[4]]
        overlay=img.copy()
        cv2.polylines(overlay,[np.round(pts).astype(np.int32)],True,(255,0,255),1)
        tiles.append(np.hstack([cv2.resize(img,(256,256)),cv2.resize(overlay,(256,256))]))
    path=Path(output)/'opening.jpg'
    cv2.imwrite(str(path),np.vstack(tiles)[:,:,::-1])
    return str(path.relative_to(s.dir))
