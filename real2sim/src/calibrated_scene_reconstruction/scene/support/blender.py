"""Agent-authored task-support geometry, fitted to observed support/occlusion masks.

Tables remain static supports. Their rendered and collision surfaces come from
the same Blender solid; no measured furniture dimension is an input or gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from scipy.optimize import minimize
import trimesh

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.core.session_code import source_path
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender
from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, write_json


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def require_mutable(session):
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: create a new support iteration session')


def target(session):
    graph = session.scene_graph
    name = graph['task']['support']
    return name, graph['furniture'][name]


def model_path(session):
    name, spec = target(session)
    model = (spec.get('fit') or {}).get('model')
    if not model:
        request = {'support': name, 'route': 'agent_blender', 'awaiting': 'agent-authored model.py',
                   'source_directory': f'configs/{session.name}/agent/support/{name}',
                   'evidence': ['qa/splat/support_box.jpg', 'splat/support_carve.json'],
                   'contract': 'Literal PARAMS/FIXED and build(params); metres,z up,bottom-centred origin. '
                               'Model visible top/edge/support geometry and disclose hidden construction. '
                               'Match the actual observed tabletop plane, including tilt; retain the independently observed gravity frame. '
                               'No true desk size or task dimensions may initialize or approve the model.'}
        write_json(session/'qa/support_model_request.json', request)
        raise ValueError('Task support needs agent Blender source; see qa/support_model_request.json')
    return source_path(session.config_dir, model)


def source_assets(session):
    _, support = target(session)
    return [source_path(session.config_dir, path) for path in (support.get('fit') or {}).get('assets', [])]


def collision_policy(support):
    value=(support.get('fit') or {}).get('collision_representation','surface')
    if value not in {'surface','source_convex'}:
        raise ValueError('Support collision representation must preserve the source surface or its exact convex source solids')
    return value


def evidence_policy(session):
    """Optional independently labelled negatives; never derive them from CAD/RGB black."""
    _, support = target(session)
    suffix = (support.get('fit') or {}).get('known_background_suffix')
    if suffix is not None and (not isinstance(suffix, str) or not re.fullmatch(r'_[A-Za-z0-9_]+', suffix)):
        raise ValueError('Support background suffix must name a local mask revision')
    return {'known_background_suffix': suffix}


def observed_domain(positive, background, ignored):
    """Resolve label conflicts as unknown, retaining independent negative pixels."""
    conflict = positive & background
    if (conflict & ~ignored).sum() > .25 * max(1, int((positive & ~ignored).sum())):
        raise ValueError('Support/background labels contradict over25% of observed foreground')
    positive = positive & ~ignored & ~conflict
    negative = background & ~ignored & ~conflict
    return positive, ignored | ~(positive | negative)


def observations(session, width=960):
    from .mask_inputs import settings as mask_settings
    masks = mask_settings(session)
    from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels, resize_valid
    validity=ValidPixels(session)
    camera_path = session/'parsing/posed_frames/cameras.npz'
    camera = dict(np.load(camera_path))
    directory = session/'splat/data/images'
    images, inputs = [], {rel(camera_path): sha(camera_path)}
    policy = evidence_policy(session)
    negative_suffix = policy['known_background_suffix']
    negative_margin = 0
    if negative_suffix:
        meta_path = session/'splat'/('masks'+negative_suffix+'.json')
        metadata = json.loads(meta_path.read_text())
        negative_margin = int(metadata.get('dilation_pixels', 0))
        if not 0 <= negative_margin <= 32:
            raise ValueError('Unsupported independent-background mask margin')
        inputs[rel(meta_path)] = sha(meta_path)
    associated={};manifest=None
    pointer=session/'support/observations.json'
    if pointer.exists():
        info=json.loads(pointer.read_text());manifest=resolve(info['path'])
        if sha(manifest)!=info['sha256']:raise ValueError('Support association manifest changed')
        metadata=json.loads(manifest.read_text())
        if metadata['support']!=session.scene_graph['task']['support']:
            raise ValueError('Associated observations belong to another support')
        for path,expected in metadata['inputs'].items():
            if sha(resolve(path))!=expected:raise ValueError('Support association evidence changed')
        associated={item['frame']:item for item in metadata['frames']}
        inputs.update({rel(pointer):sha(pointer),rel(manifest):sha(manifest),**metadata['inputs']})
    from .part_identity import load as source_part_identity
    identity_ignored,identity_inputs=source_part_identity(session,manifest,associated)
    inputs.update(identity_inputs)
    margin=0
    mask_metadata=session/('splat/masks'+masks['support_suffix']+'.json')
    if not associated and mask_metadata.exists():
        margin=int(json.loads(mask_metadata.read_text()).get('dilation_pixels',6))
        if not 0<=margin<=32:raise ValueError('Unsupported support-mask margin')
        inputs[rel(mask_metadata)]=sha(mask_metadata)
    for i, name in enumerate(camera['names']):
        if associated and str(name) not in associated:continue
        path = (resolve(associated[str(name)]['path']) if associated else
                directory/(Path(str(name)).stem+masks['support_suffix']+'.png'))
        if associated and sha(path)!=associated[str(name)]['sha256']:
            raise ValueError('Associated support mask changed')
        occluder_path = directory/(Path(str(name)).stem+masks['occluder_suffix']+'.png')
        if not path.exists() or not occluder_path.exists():
            continue
        support = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) < 128
        if margin:
            support=cv2.erode(support.astype('uint8'),np.ones((2*margin+1,2*margin+1),'uint8'))>0
        occluder = cv2.imread(str(occluder_path), cv2.IMREAD_GRAYSCALE) < 128
        # New source-visible marks or thin props can be added without changing
        # old acquired evidence or re-associating the same physical table.
        # Missing or malformed evidence must not become an all-unknown view.
        for suffix in masks.get('extra_occluder_suffixes',[]):
            extra_path=directory/(Path(str(name)).stem+suffix+'.png')
            extra=cv2.imread(str(extra_path),cv2.IMREAD_GRAYSCALE)
            if extra is None or extra.shape!=support.shape:
                raise ValueError('Additional source occlusion mask missing or in another pixel domain: '+str(extra_path))
            occluder|=extra<128
            inputs[rel(extra_path)]=sha(extra_path)
        if str(name) in identity_ignored:
            if identity_ignored[str(name)].shape != occluder.shape:
                raise ValueError('Reviewed support identity and camera resolution differ')
            occluder |= identity_ignored[str(name)]
        valid=validity.get(str(name),support.shape)
        if negative_suffix:
            negative_path = directory/(Path(str(name)).stem+negative_suffix+'.png')
            if not negative_path.exists():
                continue
            background = cv2.imread(str(negative_path), cv2.IMREAD_GRAYSCALE)
            if background is None or background.shape != support.shape:
                raise ValueError('Independent support-background mask has different image dimensions')
            background = background < 128
            if negative_margin:
                background = cv2.erode(background.astype('uint8'),
                                      np.ones((2*negative_margin+1,)*2,'uint8')) > 0
            inputs[rel(negative_path)] = sha(negative_path)
            try:
                support, occluder = observed_domain(support, background, occluder | ~valid)
            except ValueError:
                continue  # Ambiguous labels cannot turn nearly the entire support into unknown evidence.
            if int((background & ~occluder).sum()) < 200:
                continue  # A close-up without negative evidence cannot constrain the support extent.
        scale = min(1., width/max(support.shape))
        shape = tuple(max(1, round(v*scale)) for v in support.shape[::-1])
        support = cv2.resize(support.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST) > 0
        occluder = cv2.resize(occluder.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST) > 0
        occluder|=~resize_valid(valid,shape)
        support &= ~occluder
        if support.sum() < 100:
            continue
        part_masks={}; part_instances={}
        for group in associated.get(str(name),{}).get('attached_parts',[]):
            selected=[]
            for item in group['selected']:
                if 'path' not in item:continue  # Legacy rounds have endpoints only.
                part_path=resolve(item['path'])
                if sha(part_path)!=item['sha256']:raise ValueError('Associated support part mask changed')
                native_part=cv2.imread(str(part_path),cv2.IMREAD_GRAYSCALE)<128
                if native_part.shape!=tuple(cv2.imread(str(path),0).shape):
                    raise ValueError('Support part and whole-mask camera dimensions differ')
                selected.append(cv2.resize(native_part.astype('uint8'),shape,interpolation=cv2.INTER_NEAREST)>0)
                part_instances.setdefault(group['prompt'], []).append(
                    {'path': item['path'], 'mask': selected[-1]&~occluder})
                inputs[rel(part_path)]=sha(part_path)
            if selected:part_masks[group['prompt']]=np.logical_or.reduce(selected)&~occluder
        # Exact K scaling follows the actual mask shape, including portrait videos.
        K = camera['K'][i].copy()
        K[0] *= shape[0]/camera['size'][i, 0]
        K[1] *= shape[1]/camera['size'][i, 1]
        images.append({'name': str(name), 'index': i, 'K': K, 'R': camera['R_world_cam'][i],
                       'C': camera['centre'][i], 'mask': support, 'occluder': occluder,
                       'window': (0., 0., float(shape[0]), float(shape[1]), 1.),'parts':part_masks,
                       'part_instances': part_instances,
                       'evidence_policy': policy,
                       'known_pixels': int((~occluder).sum()), 'positive_pixels': int(support.sum())})
        inputs.update({rel(path): sha(path), rel(occluder_path): sha(occluder_path)})
    if len(images) < 8:
        raise ValueError('Support iteration needs at least8 observed support/occluder mask pairs')
    # Existing posed/masked frames only; this does not alter video frame selection.
    if len(images) > 32:
        images = [images[i] for i in np.linspace(0, len(images)-1, 32).astype(int)]
    inputs.update(validity.inputs)
    return images, inputs


def score(vertices, faces, pose, observations, observed_top, sheet=None, *,
          face_material=None, material_names=None, part_materials=None, part_loss_weight=1.,
          rasterizer=None):
    if not np.isfinite(part_loss_weight) or not 0<=part_loss_weight<=10:
        raise ValueError('Part training weight must be finite and between0 and10')
    from calibrated_scene_reconstruction.scene.objects.object_refine import pieces, raster
    a = np.radians(pose['yaw_deg'])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    world = vertices@R.T+[*pose['xy'], 0.]
    mesh = trimesh.Trimesh(world, faces, process=False)
    subsets={};part_materials=part_materials or {}
    part_samples={};part_faces={}
    for prompt,name in part_materials.items():
        if face_material is None or material_names is None or name not in material_names:
            raise ValueError('Declared support part material missing from generated mesh: '+name)
        selected=faces[np.asarray(face_material)==list(material_names).index(name)]
        if not len(selected):raise ValueError('Declared support material has no geometry: '+name)
        subsets[prompt]=trimesh.Trimesh(world,selected,process=False)
        part_faces[prompt]=np.asarray(face_material)==list(material_names).index(name)
        triangles=world[selected]
        edges=triangles[:,[1,2,0]]-triangles
        longest=np.linalg.norm(edges,axis=2).argmax(1)
        starts=triangles[np.arange(len(triangles)),longest]
        directions=edges[np.arange(len(triangles)),longest]
        part_samples[prompt]=(starts[:,None]+np.linspace(0,1,17)[None,:,None]*directions[:,None]).reshape(-1,3)
    visibility=None
    if subsets and rasterizer is None:
        from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts
        visibility=VisibleParts(world,faces,np.ones(len(faces),bool),device='cpu')
    ious, local, tiles, areas = [], [], [], []
    part_recalls,part_distances,part_false_positives,part_evidence=[],[],[],[]
    for observed in observations:
        rendered = raster(mesh, observed['K'], observed['R'], observed['C'],
                          observed['window'], observed['mask'].shape)
        rendered &= ~observed['occluder']
        mask = observed['mask']
        areas.append(max(1,int(mask.sum())))
        ious.append(float((rendered & mask).sum()/max(1, (rendered | mask).sum())))
        local.append(float(sum(piece[0] for piece in pieces(mask, rendered, .003))))
        face_ids=None
        for prompt in subsets:
            positive=(observed.get('parts',{}).get(prompt))
            if positive is None or positive.sum()<20:continue
            args=(observed['K'],observed['R'],observed['C'],observed['window'],mask.shape)
            if rasterizer is not None:
                predicted=rasterizer.visible(mesh,*args,target_faces=part_faces[prompt])
            else:
                if face_ids is None:face_ids=visibility.face_image(*args)
                predicted=face_ids>=0
                predicted[predicted]&=part_faces[prompt][face_ids[predicted]]
            predicted &= ~observed['occluder']
            recall=float((positive&predicted).sum()/positive.sum())
            outside=float((predicted&~mask).sum()/positive.sum())
            # Projected CAD samples retain a direction when a candidate leg
            # is entirely outside the image; an empty raster has no direction.
            from scipy.spatial import cKDTree
            from calibrated_scene_reconstruction.scene.objects.object_hull import project
            u,v,depth=project(part_samples[prompt],observed['K'],observed['R'],observed['C'])
            x0,y0,_,_,factor=observed['window']
            uv=np.c_[(u-x0)*factor,(v-y0)*factor][depth>.05]
            yy,xx=np.nonzero(positive);pixels=np.c_[xx,yy]
            if len(pixels)>400:pixels=pixels[np.linspace(0,len(pixels)-1,400).astype(int)]
            normalized=(float(cKDTree(uv).query(pixels)[0].mean()/np.sqrt(positive.sum())) if len(uv)
                        else float(np.hypot(*mask.shape)/np.sqrt(positive.sum())))
            part_recalls.append(recall);part_distances.append(normalized)
            part_false_positives.append(outside)
            part_evidence.append({'frame':observed['name'],'prompt':prompt,'positive_pixels':int(positive.sum()),
                                  'positive_recall':recall,'outside_support_fraction':outside,
                                  'normalized_contour_distance':normalized})
        if sheet is not None:
            image = np.full((*mask.shape, 3), 35, np.uint8)
            image[mask & rendered] = (225, 225, 225)
            image[mask & ~rendered] = (40, 40, 220)
            image[rendered & ~mask] = (40, 210, 40)
            image[observed['occluder']] = (80, 50, 35)
            tile = cv2.resize(image, (320, 320))
            cv2.putText(tile, observed['name'], (4, 18), 0, .45, (0, 255, 255), 1)
            tiles.append(tile)
    plane_review=None;plane_limit=.01
    if isinstance(observed_top,dict):
        from .observed_plane import model_error
        top_error,plane_review=model_error(world,faces,face_material,material_names,observed_top)
        # Tighter source evidence may narrow the existing centimetre cap;
        # this is a source-fit tolerance, never a true dimension or gravity estimate.
        plane_limit=max(.001,min(.01,3*float(observed_top['median_absolute_residual_m'])))
    else:top_error = float(abs(world[:, 2].max()-observed_top))
    median_iou, median_local = float(np.median(ious)), float(np.median(local))
    # A nearly occluded part can have only a few positive pixels. Keep its
    # diagnostics/gates, but do not let its tiny denominator dominate search.
    losses=1-np.asarray(part_recalls)+part_distances+np.asarray(part_false_positives)
    part_loss=float(np.average(losses,weights=[r['positive_pixels'] for r in part_evidence])) if part_recalls else 0.
    part_checks=[]
    for prompt in part_materials:
        rows=[r for r in part_evidence if r['prompt']==prompt]
        recall=float(np.median([r['positive_recall'] for r in rows])) if rows else None
        outside=float(np.median([r['outside_support_fraction'] for r in rows])) if rows else None
        part_checks.append({'prompt':prompt,'views':len(rows),'positive_recall_median':recall,
                            'outside_support_fraction_median':outside,
                            'pass':bool(rows and recall>=.85 and outside<=.02)})
    result = {'iou_median': median_iou, 'local_difference': median_local, 'views': len(ious),
              'ious': ious, 'local_per_frame': local, 'observed_top_plane_difference_m': top_error,
              'loss': float(1-np.average(ious,weights=areas)+np.average(local,weights=areas)+top_error/.02+part_loss_weight*part_loss),
              'part_loss_weight':part_loss_weight,
              'part_positive_evidence':part_evidence,'part_loss':part_loss,
              'part_loss_weighting':'observed part pixels; all per-view diagnostics retained',
              'part_visibility':'nearest actual whole-support surface; hidden faces cannot match source parts',
              'part_checks':part_checks,'part_acceptance_version':1,
              'part_positive_recall_median':float(np.median(part_recalls)) if part_recalls else None,
              'loss_weighting':'valid observed support pixels; every selected camera retained',
              'observation_domains': [{'frame': o['name'],
                                       'policy': o.get('evidence_policy', {'known_background_suffix': None}),
                                       'positive_pixels': int(o['mask'].sum()),
                                       'known_pixels': int((~o['occluder']).sum())} for o in observations],
              'ok': bool(median_iou >= .85 and median_local <= .02 and top_error <= plane_limit
                         and all(p['pass'] for p in part_checks)),
              'criteria': {'iou_min': .85, 'local_max': .02, 'frames_min': 8,
                           'observed_plane_difference_max_m': plane_limit,
                           'part_positive_recall_min':.85,'part_outside_support_max':.02},
              'note': 'Plane criterion compares to this capture geometry, never a measured furniture height.',
              'observed_surface_plane_review':plane_review}
    if sheet is not None:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    return result


def score_mesh(mesh,pose,observed,top,sheet=None,part_materials=None,part_loss_weight=1.,rasterizer=None):
    return score(mesh['V'],mesh['F'],pose,observed,top,sheet,
                 face_material=mesh['face_material'] if 'face_material' in mesh else None,
                 material_names=mesh['material_names'] if 'material_names' in mesh else None,
                 part_materials=part_materials,part_loss_weight=part_loss_weight,rasterizer=rasterizer)


def search_objective(metrics):
    """Direct proposal search toward every unchanged support acceptance gate.

    Mean overlap alone trades excess leg pixels for recall. That trade can
    indefinitely prefer an infeasible candidate even when a narrower leg
    meets both independently declared bounds. Keep raw loss and gates intact.
    """
    c=metrics['criteria']
    penalty=max(0.,c['iou_min']-metrics['iou_median'])/c['iou_min']
    penalty+=max(0.,metrics['local_difference']-c['local_max'])/c['local_max']
    limit=c['observed_plane_difference_max_m']
    penalty+=max(0.,metrics['observed_top_plane_difference_m']-limit)/limit
    for part in metrics.get('part_checks',[]):
        weight=metrics.get('part_loss_weight',1.)
        if part['positive_recall_median'] is None or part['outside_support_fraction_median'] is None:
            penalty+=weight*10.
            continue
        penalty+=weight*max(0.,c['part_positive_recall_min']-part['positive_recall_median'])/c['part_positive_recall_min']
        penalty+=weight*max(0.,part['outside_support_fraction_median']-c['part_outside_support_max'])/c['part_outside_support_max']
    return float(metrics['loss']+penalty)


def run(session, budget=300):
    require_mutable(session)
    if type(budget) is not int or not 10 <= budget <= 2000:
        raise ValueError('Support fit budget must be an integer between10 and2000')
    name, support = target(session)
    model = model_path(session)
    observed, inputs = observations(session)
    part_materials=(support.get('fit') or {}).get('part_materials',{})
    if not isinstance(part_materials,dict) or any(
            not isinstance(prompt,str) or not prompt.strip() or
            not isinstance(material,str) or not material.strip()
            for prompt,material in part_materials.items()):
        raise ValueError('Support part_materials must map each prompt to one material name')
    part_weight=float((support.get('fit') or {}).get('part_loss_weight',1.))
    if not np.isfinite(part_weight) or not 0<=part_weight<=10:
        raise ValueError('Part training weight must be finite and between0 and10')
    for prompt in part_materials:
        if not any(prompt in frame.get('parts',{}) for frame in observed):
            raise ValueError('Declared support part needs independent masks: '+prompt)
    from .plane_tracks import read as observed_plane_record
    plane_record=observed_plane_record(session)
    observed_surface=None
    if plane_record is not None:
        top_material=(support.get('fit') or {}).get('pose_anchor_material')
        if not top_material:raise ValueError('Observed tilted-plane support requires the actual tabletop material name')
        observed_surface={**plane_record['plane'],'top_material':top_material}
        inputs[rel(session/'support/surface_plane.json')]=sha(session/'support/surface_plane.json')
    assets = source_assets(session)
    inputs.update({rel(path): sha(path) for path in assets})
    root = session/'support'/name
    rounds = [int(path.name[6:]) for path in root.glob('round_*') if path.name[6:].isdigit()]
    number = max(rounds, default=0)+1
    out = root/f'round_{number}'
    out.mkdir(parents=True, exist_ok=False)
    engine = out/'engine'
    engine.mkdir()
    for filename in ('blender.py', 'observations.py', 'part_identity.py','pose_anchor.py','observed_plane.py','plane_tracks.py'):
        snapshot = engine/filename
        shutil.copyfile(Path(__file__).with_name(filename), snapshot)
        inputs[rel(snapshot)] = sha(snapshot)
    from calibrated_scene_reconstruction.assets import convex_source_solids
    snapshot=engine/'convex_source_solids.py';shutil.copyfile(Path(convex_source_solids.__file__),snapshot)
    inputs[rel(snapshot)]=sha(snapshot)
    if part_materials:
        from calibrated_scene_reconstruction.sim import state_visibility
        from calibrated_scene_reconstruction.scene.objects.blender import projection
        for source in (Path(state_visibility.__file__),Path(projection.__file__)):
            snapshot=engine/source.name
            shutil.copyfile(source,snapshot)
            inputs[rel(snapshot)]=sha(snapshot)
    shutil.copyfile(model, out/'model.py')
    snapshots=[]
    for asset in assets:
        try:relative=asset.resolve().relative_to(model.parent.resolve())
        except ValueError as exc:
            raise ValueError('Support source assets must be under the model directory') from exc
        if relative==Path('model.py'):
            raise ValueError('A support texture asset cannot replace model.py')
        snapshot=out/relative;snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(asset,snapshot);snapshots.append(snapshot)
        inputs[rel(snapshot)]=sha(snapshot)
    write_json(out/'input.json', {'inputs': inputs, 'model_sha256': sha(model), 'budget': budget,
                                 'declared_assets': [rel(path) for path in assets]})
    params = contract(model)['params']
    lo_box, hi_box = np.asarray(support['box'], float)
    centre = (lo_box[:2]+hi_box[:2])/2
    carve_path = session/'splat/support_carve.json'
    yaw = 0.
    if carve_path.exists():
        carve = json.loads(carve_path.read_text())
        if carve['support'] != name:
            raise ValueError('Support carve belongs to a different task support')
        hi_box[2] = carve['top_z']
        yaw = float(carve.get('top_yaw_deg', 0.))
        inputs[rel(carve_path)] = sha(carve_path)
        write_json(out/'input.json', {'inputs': inputs, 'model_sha256': sha(model), 'budget': budget,
                                     'declared_assets': [rel(path) for path in assets],
                                     'observed_top_m': float(hi_box[2])})
    optimize_shape=(support.get('fit') or {}).get('optimize_shape',True)
    if type(optimize_shape) is not bool:raise ValueError('Support optimize_shape must be boolean')
    names = list(params) if optimize_shape else []
    selected_parameters=(support.get('fit') or {}).get('optimize_parameters')
    if selected_parameters is not None:
        if (not isinstance(selected_parameters,list)
                or any(not isinstance(n,str) or n not in params for n in selected_parameters)
                or len(set(selected_parameters))!=len(selected_parameters)
                or selected_parameters and not optimize_shape):
            raise ValueError('Support optimize_parameters must name distinct declared shape parameters with shape optimization enabled')
        names=selected_parameters
    optimize_pose=(support.get('fit') or {}).get('optimize_pose',True)
    if type(optimize_pose) is not bool:raise ValueError('Support optimize_pose must be boolean')
    pose_lo=np.r_[centre-.05,yaw-15.];pose_hi=np.r_[centre+.05,yaw+15.]
    lo = np.array([params[n]['lo'] for n in names]+(pose_lo.tolist() if optimize_pose else []))
    hi = np.array([params[n]['hi'] for n in names]+(pose_hi.tolist() if optimize_pose else []))
    seed=(support.get('fit') or {}).get('initial_pose',{'xy':centre.tolist(),'yaw_deg':yaw})
    pose_values=np.array([*seed['xy'],seed['yaw_deg']],float)
    if (pose_values.shape!=(3,) or not np.isfinite(pose_values).all() or
            (pose_values<pose_lo).any() or (pose_values>pose_hi).any()):
        raise ValueError('Support initial pose must lie within the current observation search bounds')
    manifest=json.loads((out/'input.json').read_text())
    manifest['initial_pose']={'xy':pose_values[:2].tolist(),'yaw_deg':float(pose_values[2])}
    manifest['collision_representation']=collision_policy(support)
    manifest['part_materials']=part_materials
    manifest['part_acceptance_version']=1
    manifest['part_identity_review']=(support.get('fit') or {}).get('part_identity_review')
    manifest['optimized_parameters']=names
    manifest['part_loss_weight']=part_weight
    manifest['optimize_pose']=optimize_pose
    manifest['optimize_shape']=optimize_shape
    manifest['evidence_policy']=evidence_policy(session)
    starts=(support.get('fit') or {}).get('search_starts',1)
    if type(starts) is not int or not 1<=starts<=12 or starts*max(10,len(names)+4)>budget:
        raise ValueError('Support search_starts requires 1..12 starts and at least10 evaluations per start')
    manifest['search_starts']=starts
    write_json(out/'input.json',manifest)
    initial = np.array([params[n]['init'] for n in names]+(pose_values.tolist() if optimize_pose else []))
    best, evaluations = {}, []
    search_width=(support.get('fit') or {}).get('search_width',960)
    if type(search_width) is not int or not 240<=search_width<=960:
        raise ValueError('Support search_width must be an integer in240..960; final checks use960')
    search_observed=observed if search_width==960 else observations(session,width=search_width)[0]
    manifest['search_width']=search_width
    manifest['acceptance_width']=960
    write_json(out/'input.json',manifest)
    proposal=None
    if part_materials:
        from calibrated_scene_reconstruction.scene.objects.blender.projection import optional
        proposal=optional()
    with Blender(out/'model.py', root/'cache', max_extent_m=10., source_files=[*assets,*snapshots]) as builder:
        anchor_material=(support.get('fit') or {}).get('pose_anchor_material')
        reference_anchor=None
        if anchor_material is not None:
            if not isinstance(anchor_material,str) or not anchor_material:
                raise ValueError('Support pose anchor must name a generated material')
            from .pose_anchor import material_anchor,anchored_pose
            anchor_path,_=builder.evaluate({key:value['init'] for key,value in params.items()})
            with np.load(anchor_path/'mesh.npz') as initial_mesh:
                reference_anchor=material_anchor(initial_mesh,anchor_material)
            previous=support.get('fix') or {}
            preserve=(support.get('fit') or {}).get('preserve_exported_support_anchor',True)
            if type(preserve) is not bool:raise ValueError('Support anchor preservation must be boolean')
            if preserve and previous.get('mesh'):
                from .pose_anchor import carry_pose
                prior_path=resolve(previous['mesh'])
                if sha(prior_path)!=previous.get('mesh_sha256'):
                    raise ValueError('Previous support anchor geometry changed')
                with np.load(prior_path) as prior_mesh:
                    prior_anchor=material_anchor(prior_mesh,anchor_material)
                carried=carry_pose(previous,prior_anchor,
                    {'xy':pose_values[:2].tolist(),'yaw_deg':float(pose_values[2])},reference_anchor)
                pose_values[:2]=carried['xy']
                if optimize_pose:
                    lo[-3:]=pose_values-[.05,.05,15.];hi[-3:]=pose_values+[.05,.05,15.]
                initial=np.array([params[n]['init'] for n in names]+(pose_values.tolist() if optimize_pose else []))
                inputs[rel(prior_path)]=sha(prior_path)
                manifest.update(inputs=inputs,initial_pose=carried,
                    prior_anchor_reference={'mesh':rel(prior_path),'pose':{k:previous[k] for k in ('xy','yaw_deg')},
                                            'anchor_xy':prior_anchor.tolist(),'proposal_only':True})
        manifest['pose_anchor_material']=anchor_material
        manifest['reference_anchor_xy']=reference_anchor.tolist() if reference_anchor is not None else None
        write_json(out/'input.json',manifest)
        def evaluate(q):
            values = lo+np.clip(q, 0, 1)*(hi-lo)
            state = {key: value['init'] for key,value in params.items()}
            state.update(zip(names, values[:len(names)].tolist()))
            pv=values[-3:] if optimize_pose else pose_values
            pose = {'xy': pv[:2].tolist(), 'yaw_deg': float(pv[-1])}
            try:
                path, build = builder.evaluate(state)
                mesh = np.load(path/'mesh.npz')
                if reference_anchor is not None:
                    pose=anchored_pose(pose,reference_anchor,material_anchor(mesh,anchor_material))
                metrics = score_mesh(mesh,pose,search_observed,observed_surface or float(hi_box[2]),part_materials=part_materials,
                                     part_loss_weight=part_weight,rasterizer=proposal)
            except ValueError as exc:
                evaluations.append({'error': str(exc)})
                if len(evaluations)==1:
                    print('[support.fit] rejected geometry: '+str(exc), flush=True)
                if len(evaluations)%25==0:
                    write_json(out/'fit_progress.json',{'proposal_only':True,'evaluations':len(evaluations),
                               'model_sha256':sha(out/'model.py'),'best':best,'recent_errors':evaluations[-25:]})
                return 10.
            objective=search_objective(metrics)
            if not best or ((metrics['ok'], -objective) > (best['metrics']['ok'], -best['search_loss'])):
                best.update(params=state, pose=pose, metrics=metrics,search_loss=objective)
            evaluations.append({'iou': metrics['iou_median'], 'local': metrics['local_difference']})
            if len(evaluations) % 25 == 0:
                write_json(out/'fit_progress.json',{'proposal_only':True,'evaluations':len(evaluations),
                           'model_sha256':sha(out/'model.py'),'search_width':search_width,'best':best})
                print(f'[support.fit] {len(evaluations)} evaluations; IoU {best["metrics"]["iou_median"]:.4f}', flush=True)
            return objective
        q = (initial-lo)/(hi-lo)
        if len(q):
            from scipy.stats import qmc
            seeds=[q,*qmc.LatinHypercube(len(q),seed=0).random(starts-1)] if starts>1 else [q]
            converged=False
            for index,seed in enumerate(seeds):
                simplex=np.tile(seed,(len(q)+1,1))
                for i in range(len(q)):
                    simplex[i+1,i]+=-.025 if seed[i]>.95 else .025
                allowance=(budget-len(evaluations))//(starts-index)
                optimized = minimize(evaluate,seed,method='Nelder-Mead',bounds=[(0,1)]*len(q),
                                     options={'maxfev':allowance,'initial_simplex':simplex,'adaptive':True,
                                              'xatol':.0005,'fatol':.00001})
                converged|=bool(optimized.success)
        else:
            evaluate(q)
            converged=True
        if not best:
            write_json(out/'failure.json',{'ok':False,'evaluations':evaluations,
                       'reason':'No valid support model in the fit budget'})
            raise ValueError('No valid support model in the fit budget')
        path, build = builder.evaluate(best['params'], artifacts=True)
    for filename in ('mesh.npz', 'object.glb', 'scene.blend', 'preview_0.png', 'preview_1.png','source_solids.npz'):
        shutil.copyfile(path/filename, out/filename)
    mesh = np.load(out/'mesh.npz')
    if collision_policy(support)=='source_convex':
        from calibrated_scene_reconstruction.assets.convex_source_solids import load as load_convex_solids
        load_convex_solids(out/'source_solids.npz')  # Reject open/concave source pieces, never fill them.
    from .materials import extract
    textures = extract(out/'object.glb', out/'materials')
    sheet = session/f'qa/support/{name}/round_{number}.jpg'
    best['metrics'] = score_mesh(mesh,best['pose'],observed,observed_surface or float(hi_box[2]),sheet,part_materials,part_weight)
    manifest = json.loads((out/'input.json').read_text())
    if sha(model) != manifest['model_sha256'] or sha(out/'model.py') != manifest['model_sha256']:
        raise ValueError('Support source changed during fitting; retain this attempt and start a new round')
    for path, expected in inputs.items():
        if sha(resolve(path)) != expected:
            raise ValueError('Support observations changed during fitting')
    result = {**best, 'round': number, 'ok': best['metrics']['ok'], 'model_sha256': sha(model),
              'input_sha256': sha(out/'input.json'),
              'mesh_sha256': sha(out/'mesh.npz'), 'visual_sha256': sha(out/'object.glb'),
              'textures': textures,'source_solids_sha256':sha(out/'source_solids.npz'),
              'build': build, 'converged': converged, 'trace': evaluations,
              'sheet': rel(sheet), 'support': name}
    write_json(out/'fit.json', result)
    stamp = session/f'qa/review/SUP.{name}.ran'
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    return result


def verify(session, name, exported=True, number=None):
    spec = session.scene_graph['furniture'][name]
    fix = spec.get('fix') or {}
    number = fix['round'] if number is None else number
    out = session/'support'/name/f'round_{number}'
    result = json.loads((out/'fit.json').read_text())
    manifest = json.loads((out/'input.json').read_text())
    if sha(out/'input.json') != result['input_sha256']:
        raise ValueError('Support input manifest changed since fitting')
    if collision_policy(spec)!=manifest.get('collision_representation','surface'):
        raise ValueError('Support collision representation changed; qualify a new round')
    if result.get('source_solids_sha256') and sha(out/'source_solids.npz')!=result['source_solids_sha256']:
        raise ValueError('Authored support solids changed since their Boolean union')
    if [rel(path) for path in source_assets(session)] != manifest.get('declared_assets', []):
        raise ValueError('Declared support source assets changed since fitting')
    if (spec.get('fit') or {}).get('part_materials',{}) != manifest.get('part_materials',{}):
        raise ValueError('Support semantic part mapping changed since fitting')
    if (spec.get('fit') or {}).get('pose_anchor_material') != manifest.get('pose_anchor_material'):
        raise ValueError('Support pose anchor changed since fitting')
    if (spec.get('fit') or {}).get('part_identity_review') != manifest.get('part_identity_review'):
        raise ValueError('Support source part identity review changed; fit a new round')
    if ((spec.get('fit') or {}).get('part_acceptance_version') == 1
            and manifest.get('part_acceptance_version') != 1):
        raise ValueError('Support part acceptance requires a new measured round')
    if manifest.get('part_materials') and manifest.get('part_acceptance_version') == 1:
        checks=result.get('metrics',{}).get('part_checks',[])
        if (set(p['prompt'] for p in checks)!=set(manifest['part_materials'])
                or not all(p['pass'] for p in checks)):
            raise ValueError('Visible support parts did not pass their independent source checks')
    if (spec.get('fit') or {}).get('part_loss_weight',1.) != manifest.get('part_loss_weight',1.):
        raise ValueError('Support part training weight changed since fitting')
    if (spec.get('fit') or {}).get('optimize_pose',True) != manifest.get('optimize_pose',True):
        raise ValueError('Support pose refinement policy changed since fitting')
    if (spec.get('fit') or {}).get('optimize_shape',True) != manifest.get('optimize_shape',True):
        raise ValueError('Support shape refinement policy changed since fitting')
    if evidence_policy(session) != manifest.get('evidence_policy', {'known_background_suffix': None}):
        raise ValueError('Support positive/negative observation policy changed since fitting')
    if (sha(model_path(session)) != result['model_sha256'] or
            sha(out/'model.py') != result['model_sha256'] or
            manifest['model_sha256'] != result['model_sha256']):
        raise ValueError('Support Blender source changed since fitting')
    for path, expected in manifest['inputs'].items():
        if sha(resolve(path)) != expected:
            raise ValueError('Support camera or mask input changed since fitting')
    if sha(out/'mesh.npz') != result['mesh_sha256'] or sha(out/'object.glb') != result['visual_sha256']:
        raise ValueError('Support fitted geometry changed')
    for texture in result.get('textures', []):
        if sha(resolve(texture['path'])) != texture['sha256']:
            raise ValueError('Support image texture changed since fitting')
    if exported and (fix.get('model_sha256') != result['model_sha256'] or
                     fix.get('model') != (spec.get('fit') or {}).get('model') or
                     fix.get('mesh_sha256') != result['mesh_sha256'] or
                     fix.get('visual_sha256') != result['visual_sha256'] or
                     fix.get('textures', []) != result.get('textures', []) or
                     resolve(fix['mesh']) != (out/'mesh.npz').resolve() or
                     resolve(fix['visual_glb']) != (out/'object.glb').resolve() or
                     fix.get('xy') != result['pose']['xy'] or fix.get('yaw_deg') != result['pose']['yaw_deg']):
        raise ValueError('Support export differs from the fitted artifact')
    if exported and fix.get('collision_representation','surface')!=collision_policy(spec):
        raise ValueError('Support exported collision representation differs from its source policy')
    if exported and collision_policy(spec)=='source_convex':
        expected={'path':rel(out/'source_solids.npz'),'sha256':result['source_solids_sha256']}
        if fix.get('source_solids')!=expected:
            raise ValueError('Support exported convex cells are not bound to the authored source solids')
    if exported:
        mesh = np.load(out/'mesh.npz')
        a = np.radians(result['pose']['yaw_deg'])
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        world = mesh['V']@R.T+[*result['pose']['xy'], 0.]
        if not np.allclose(spec['box'], [world.min(0), world.max(0)], atol=1e-8, rtol=0):
            raise ValueError('Support task-contact box differs from the fitted geometry')
    if exported and result['metrics'].get('observed_surface_plane_review') is not None:
        plane=result['metrics']['observed_surface_plane_review']
        expected={k:plane[k] for k in ['normal','offset','top_material','maximum_planarity_error_m']}
        if fix.get('surface_plane')!=expected:raise ValueError('Exported support plane differs from the actual reviewed tabletop geometry')
    from .plane_tracks import read as observed_plane_record
    plane_record=observed_plane_record(session)
    if plane_record is not None:
        path=session/'support/surface_plane.json'
        if (rel(path) not in manifest['inputs'] or manifest['inputs'][rel(path)]!=sha(path)
                or result['metrics'].get('observed_surface_plane_review') is None):
            raise ValueError('Observed full support plane requires a fresh source-reviewed support model round')
    metrics = result['metrics']
    if (not result['ok'] or metrics['views'] < 8 or metrics['iou_median'] < .85 or
            metrics['local_difference'] > .02 or metrics['observed_top_plane_difference_m'] > .01):
        raise ValueError('Support fit has not met observation criteria')
    from calibrated_scene_reconstruction.scene.objects.object_understanding import current as semantic_review
    from calibrated_scene_reconstruction.scene.objects.structural_review import current as structure_review
    semantic_review(session,name,out,result['build'])
    structure_review(session,name,out)
    return result


def export(session, number):
    require_mutable(session)
    name, support = target(session)
    out = session/'support'/name/f'round_{number}'
    result = verify(session, name, exported=False, number=number)
    graph = session.scene_graph
    spec = graph['furniture'][name]
    spec['fix'] = {'model': spec['fit']['model'], 'mesh': rel(out/'mesh.npz'),
                   'visual_glb': rel(out/'object.glb'), 'round': number,
                   'model_sha256': result['model_sha256'], 'mesh_sha256': result['mesh_sha256'],
                   'visual_sha256': result['visual_sha256'], **result['pose']}
    spec['fix']['textures'] = result.get('textures', [])
    reviewed_plane=result['metrics'].get('observed_surface_plane_review')
    if reviewed_plane is not None:
        spec['fix']['surface_plane']={k:reviewed_plane[k] for k in ['normal','offset','top_material','maximum_planarity_error_m']}
    spec['fix']['collision_representation']=collision_policy(spec)
    if result.get('source_solids_sha256'):
        spec['fix']['source_solids']={'path':rel(out/'source_solids.npz'),'sha256':result['source_solids_sha256']}
    mesh = np.load(out/'mesh.npz')
    a = np.radians(result['pose']['yaw_deg'])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    world = mesh['V']@R.T+[*result['pose']['xy'], 0.]
    spec['box'] = [world.min(0).tolist(), world.max(0).tolist()]
    write_json(session.config_dir/'scene_graph.json', graph)
    return {'ok': True, 'support': name, 'round': number, 'mesh': rel(out/'mesh.npz')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('action', choices=['request', 'fit', 'export', 'refine'])
    parser.add_argument('--budget', type=int, default=300)
    parser.add_argument('--round', type=int)
    args = parser.parse_args()
    session = Session(args.session)
    if args.action == 'request':
        print(model_path(session))
    elif args.action in ('fit', 'refine'):
        result = run(session, args.budget)
        if args.action == 'refine' and result['ok']:
            result['export'] = export(session, result['round'])
        print(json.dumps({key: value for key, value in result.items() if key != 'trace'}, indent=2))
        if not result['ok']:
            raise SystemExit(2)
    else:
        print(json.dumps(export(session, args.round), indent=2))


if __name__ == '__main__':
    main()
