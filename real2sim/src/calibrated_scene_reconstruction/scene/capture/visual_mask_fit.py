"""Refine inferred articulation with current-view silhouettes and sparse 3D tracks.

Per-link convex silhouettes are registration proxies, never exported task meshes.
Independent occluder/hand masks prevent hidden CAD or unestimated fingers from
being interpreted as a size error. No measured joints or task dimensions enter.
"""
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from calibrated_scene_reconstruction.core.paths import resolve
from .visual_robot_fit import Settings, rotation, sample_surfaces


def load_views(session, reconstruction, sim, max_views=16, occluder_suffix='_visual_occluders',
               robot_suffix='_visual_robot', identity_points=None, background_suffix=None,
               require_boundary=True, rejected_frames=None, hands_suffix='_visual_hands'):
    from .valid_pixels import ValidPixels, resize_valid
    from .robot_visual import observation_rejections
    rejected_frames=observation_rejections(reconstruction,rejected_frames)
    validity=ValidPixels(session)
    suffixes=[occluder_suffix,robot_suffix,hands_suffix]+([background_suffix] if background_suffix else [])
    if not all(re.fullmatch(r'_[A-Za-z0-9_]+', suffix) for suffix in suffixes):
        raise ValueError('Local semantic-mask suffix required')
    images = session/'splat/data/images'
    candidates, rejected = [], []
    from .calibration_scope import allowed
    scope=allowed(session,[image.name for image in reconstruction.images.values()])
    for image in sorted(reconstruction.images.values(), key=lambda im: im.name):
        if image.name not in scope:
            rejected.append({'frame':image.name,'reason':'Outside declared metric source/interval','source':'input_plan'});continue
        if image.name in rejected_frames:
            rejected.append({'frame':image.name,'reason':rejected_frames[image.name],'source':'raw_image_review'})
            continue
        stem = Path(image.name).stem
        paths = {key: images/(stem+suffix+'.png') for key, suffix in (
            ('mask', robot_suffix), ('hands', hands_suffix), ('occluders', occluder_suffix))}
        if background_suffix:paths['background']=images/(stem+background_suffix+'.png')
        if not all(path.exists() for path in paths.values()):
            continue
        camera = reconstruction.cameras[image.camera_id]
        shape = (max(1, round(camera.width*min(1, 640/max(camera.width, camera.height)))),
                 max(1, round(camera.height*min(1, 640/max(camera.width, camera.height)))))
        native = {key: (cv2.imread(str(path), 0) < 128).astype('uint8') for key,path in paths.items()}
        image_valid=resize_valid(validity.get(image.name,native['mask'].shape),shape)
        paths.update({f'rectification_{i}':resolve(path) for i,path in enumerate(validity.inputs)})
        # Undo the background-mask margin at native resolution before resizing;
        # otherwise quantization can create an extra silhouette error band.
        native['mask'] = cv2.erode(native['mask'], np.ones((13,13),'uint8'))
        if background_suffix:native['background']=cv2.erode(native['background'],np.ones((13,13),'uint8'))
        masks = {key: cv2.resize(mask,shape,interpolation=cv2.INTER_NEAREST)>0
                 for key,mask in native.items()}
        soft_mask = cv2.resize(native['mask'].astype('float32'),shape,interpolation=cv2.INTER_AREA)
        # Missing remap pixels are outside the observation domain, rather than
        # contradictory robot/occluder labels or unlabeled robot interior.
        masks['mask']&=image_valid
        if masks['mask'].sum() < 100:
            continue
        if (masks['hands'] & masks['mask']).sum() > .25*masks['mask'].sum():
            rejected.append({'frame': image.name, 'reason': 'hand mask consumes over25% of robot; '
                             'closeup or ambiguous arm/hand semantics cannot constrain body calibration'})
            continue
        matrix = np.asarray(image.cam_from_world().matrix())
        Rsim = np.asarray(sim['R'])
        R = Rsim@matrix[:, :3].T
        C = float(sim['scale'])*Rsim@(-matrix[:, :3].T@matrix[:, 3])+sim['t']
        fx, fy, cx, cy = camera.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        K[0] *= shape[0]/camera.width
        K[1] *= shape[1]/camera.height
        ignored = masks['occluders'] | masks['hands'] | ~image_valid
        unknown_share=0.
        if background_suffix:
            # Segmentations are partial labels, not a guarantee that every
            # enclosed dark patch is background. Confirm negative interior
            # pixels with independent scene labels, without looking at CAD.
            contours=cv2.findContours(masks['mask'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            hull=np.zeros_like(masks['mask'],dtype='uint8')
            cv2.fillConvexPoly(hull,cv2.convexHull(np.concatenate(contours)),1)
            ambiguous=(hull>0)&~(masks['mask']|masks['background']|ignored)
            unknown_share=float(ambiguous.sum()/masks['mask'].sum())
            if unknown_share>.25:
                rejected.append({'frame':image.name,'reason':'unlabelled robot interior exceeds25%; insufficient image evidence',
                                 'interior_unknown_share':unknown_share})
                continue
            ignored|=ambiguous|(masks['mask']&masks['background'])
        # Contradictory labels are unknown evidence. Preferring 'robot' here
        # would force CAD to reproduce a camera boom mislabelled as an arm.
        conflict_share = float((ignored & masks['mask']).sum()/masks['mask'].sum())
        if conflict_share > .25:
            rejected.append({'frame': image.name, 'reason': 'robot/occluder semantic conflict exceeds25%',
                             'conflict_share': conflict_share})
            continue
        valid = masks['mask'] & ~ignored
        if valid.sum() < 100:
            continue
        identity_support = None
        if identity_points is not None:
            # Identify the observed instance from independent multi-view tracks,
            # never from the candidate CAD silhouette. A textureless image may
            # still see tracks reconstructed in other cameras. Projection here
            # is identity evidence only, never a new depth observation.
            q = (np.asarray(identity_points)-C)@R
            front = q[:, 2] > .05
            q = q[front]
            pixels = np.rint(q[:, :2]/q[:, 2:]*[K[0,0], K[1,1]]+K[:2,2]).astype(int)
            inside = ((pixels[:,0] >= 0) & (pixels[:,0] < shape[0]) &
                      (pixels[:,1] >= 0) & (pixels[:,1] < shape[1]))
            pixels = pixels[inside]
            identity_support = int(valid[pixels[:,1], pixels[:,0]].sum())
            if identity_support < 3:
                rejected.append({'frame': image.name,
                                 'reason': 'robot label lacks independent multi-view instance identity',
                                 'identity_track_projections': identity_support})
                continue
        distance = cv2.distanceTransform((~masks['mask']).astype('uint8'), cv2.DIST_L2, 5)
        contours = cv2.findContours(valid.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        boundary = np.concatenate(contours).reshape(-1, 2)
        away = cv2.dilate(ignored.astype('uint8'), np.ones((7, 7), 'uint8'))==0
        boundary = boundary[away[boundary[:, 1], boundary[:, 0]]]
        if require_boundary and len(boundary) < 30:
            continue
        if len(boundary) > 240:
            boundary = boundary[np.linspace(0, len(boundary)-1, 240).astype(int)]
        candidates.append({'name': image.name, 'K': K, 'R': R, 'C': C, 'mask': masks['mask'],
                           'ignored': ignored, 'distance': distance, 'boundary': boundary.astype(float),
                           'inputs': paths, 'conflict_share': conflict_share, 'soft_mask': soft_mask})
        candidates[-1]['interior_unknown_share']=unknown_share
        if identity_support is not None:
            candidates[-1]['identity_track_projections'] = identity_support
    if len(candidates) < 8:
        from collections import Counter
        reasons=Counter(item['reason'] for item in rejected)
        raise ValueError('Silhouette refinement needs at least8 current robot/hand/occluder mask triplets; '
                         f'accepted={len(candidates)}, rejections={dict(reasons)}')
    if len(candidates) > max_views:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, max_views).astype(int)]
    return candidates, {'rejected': rejected, 'selected': [view['name'] for view in candidates],
                        'boundary_proxy_required':require_boundary,
                        'boundary_samples':{v['name']:len(v['boundary']) for v in candidates},
                        'label_scope':'independent foreground/background; bounded unknown interior' if background_suffix else 'binary foreground',
                        'interior_unknown_share':{v['name']:v['interior_unknown_share'] for v in candidates},
                        'identity_track_projections': {view['name']: view.get('identity_track_projections')
                                                       for view in candidates},
                        'semantic_conflict_share': {view['name']: view['conflict_share'] for view in candidates}}


@threadpool_limits.wrap(limits=1, user_api='blas')
def refine(model, observed, initial, views, iterations=20, part_scenes=None):
    settings = Settings()
    joints = [j for j in model.movable if j.upper > j.lower and not any(
        re.search(pattern, j.name, re.I) for pattern in settings.exclude_joints)]
    zero = {j.name: 0. for j in model.movable}
    sampled = sample_surfaces(model, Settings(points_per_link=600))
    local, names = list(sampled.values()), list(sampled)
    spans = []
    offset = 0
    for points in local:
        spans.append(np.arange(offset, offset+len(points)))
        offset += len(points)
    floor = model.floor_height()
    k0 = initial['map_units_per_metre']
    start = np.r_[initial['base_xy_map'], np.radians(initial['yaw_deg']), np.log(k0),
                  [initial['estimated_joints_rad'][j.name] for j in joints]]
    lower = np.r_[start[:2]-.5*k0, start[2]-np.pi, np.log(k0*.6), [j.lower for j in joints]]
    upper = np.r_[start[:2]+.5*k0, start[2]+np.pi, np.log(k0*1.6), [j.upper for j in joints]]
    observed = np.asarray(observed, float)
    selected = observed[np.random.default_rng(0).permutation(len(observed))[:settings.max_scene_points]]
    observed = selected[np.arange(len(selected)) % 4 != 0][:3000]
    allowed_part_points = {tuple(point) for point in np.round(observed, 9)}
    observed_tree = cKDTree(observed)
    part_targets = []
    for pattern, cloud in (part_scenes or {}).items():
        cloud = np.asarray(cloud, float)
        cloud = cloud[[tuple(point) in allowed_part_points for point in np.round(cloud, 9)]]
        if len(cloud) < 100:
            continue
        indices = np.concatenate([span for name, span in zip(names, spans) if re.search(pattern, name, re.I)])
        part_targets.append((indices, cloud, cKDTree(cloud)))

    def points(values):
        state = {**zero, **{j.name: float(q) for j, q in zip(joints, values[4:])}}
        transforms = model.transforms(state)
        source = np.concatenate([p@transforms[n][:3, :3].T+transforms[n][:3, 3] for n,p in zip(names,local)])
        k = np.exp(values[3])
        return k*source@rotation(values[2]).T+[*values[:2], -k*floor]

    def project(source, view):
        q = (source-view['C'])@view['R']
        with np.errstate(divide='ignore', invalid='ignore'):
            pixel = q[:, :2]/q[:, 2:]*[view['K'][0,0],view['K'][1,1]]+view['K'][:2,2]
        return pixel,q[:,2]

    def outline(source, view):
        pixel,depth = project(source,view)
        ids=np.flatnonzero((depth>.05)&np.isfinite(pixel).all(1));h,w=view['mask'].shape
        if not len(ids): return ids,pixel,depth
        coords=pixel[ids]
        inside=(coords[:,0]>=0)&(coords[:,0]<w-1)&(coords[:,1]>=0)&(coords[:,1]<h-1)
        ids=ids[inside];coords=pixel[ids].astype(int)
        return ids[~view['ignored'][coords[:,1],coords[:,0]]],pixel,depth

    values=start.copy();trace=[]
    for step in range(iterations):
        source=points(values);k=np.exp(values[3])
        distance, nearest=cKDTree(source).query(observed)
        chosen=np.argsort(distance)[:max(100,int(.75*len(observed)))]
        target_ids=nearest[chosen]
        forward_distance, forward_nearest = observed_tree.query(source)
        by_link = []
        for span in spans:
            count = max(4, int(settings.keep*len(span)))
            selected = span[np.argsort(forward_distance[span])[:count]]
            by_link.append((float(np.median(forward_distance[selected])), selected))
        by_link.sort(key=lambda item: item[0])
        forward_ids = np.concatenate([item[1] for item in by_link[:max(2, round(.75*len(by_link)))]])
        forward_target = observed[forward_nearest[forward_ids]]
        parts = []
        for indices, cloud, part_tree in part_targets:
            dist, nearest = part_tree.query(source[indices])
            count = max(8, int(.65*len(indices)))
            keep = np.argsort(dist)[:count]
            back_dist, back_nearest = cKDTree(source[indices]).query(cloud)
            back_keep = np.argsort(back_dist)[:max(40, int(.75*len(cloud)))]
            parts.append((indices[keep], cloud[nearest[keep]], indices[back_nearest[back_keep]], cloud[back_keep]))
        correspondences=[]
        for view in views:
            ids,pixel,depth=outline(source,view)
            if len(ids)<8: continue
            distance2,nearest2=cKDTree(pixel[ids]).query(view['boundary'])
            keep=np.argsort(distance2)[:max(20,int(.8*len(distance2)))]
            reverse_ids=ids[nearest2[keep]]
            if len(ids)>800:
                ids=ids[np.linspace(0,len(ids)-1,800).astype(int)]
            # Pixel residuals are converted to robot metres by camera depth.
            factor=float(np.median(depth[ids])/view['K'][0,0]/k)
            correspondences.append((view,ids,reverse_ids,view['boundary'][keep],factor))
        def residual(candidate):
            predicted=points(candidate);scale=np.exp(candidate[3])
            result=[((predicted[target_ids]-observed[chosen])/scale).ravel()]
            result.append((np.sqrt(len(chosen)/len(forward_ids))*
                           (predicted[forward_ids]-forward_target)/scale).ravel())
            for ids, target, back_ids, back_target in parts:
                result.append((np.sqrt(len(chosen)/len(ids))*(predicted[ids]-target)/scale).ravel())
                result.append((np.sqrt(len(chosen)/len(back_ids))*(predicted[back_ids]-back_target)/scale).ravel())
            for view,ids,reverse_ids,boundary,factor in correspondences:
                pixel,_=project(predicted,view)
                dist=cv2.remap(view['distance'],pixel[ids,0].astype('float32'),
                               pixel[ids,1].astype('float32'),cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0).ravel()
                weight=np.sqrt(len(chosen)/len(correspondences))
                result.append(dist*factor*weight/np.sqrt(len(ids)))
                result.append(((pixel[reverse_ids]-boundary)*factor*weight/np.sqrt(len(boundary))).ravel())
            return np.concatenate(result)
        opt=least_squares(residual,values,bounds=(lower,upper),loss='soft_l1',f_scale=.01,
                          max_nfev=20,diff_step=1e-4,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        delta=np.linalg.norm(opt.x-values);values=opt.x
        trace.append(float(np.mean(residual(values)**2)))
        if (step+1)%5==0:print(f'[robot_visual_masks] refinement {step+1}/{iterations}; loss {trace[-1]:.7f}',flush=True)
        if delta<1e-5:break
    result={**initial,'map_units_per_metre':float(np.exp(values[3])),
            'metres_per_map_unit':float(np.exp(-values[3])), 'base_xy_map':values[:2].tolist(),
            'base_z_map':float(-np.exp(values[3])*floor),'yaw_deg':float(np.degrees(values[2])),
            'estimated_joints_rad':{j.name:float(q) for j,q in zip(joints,values[4:])},
            'mask_refinement':{'views':[view['name'] for view in views],'loss_trace':trace,
                               'proxy':'area-sampled CAD surface projections; no exported mesh change'}}
    return result


def score(model, observed, candidate, views, sheet=None, predicted_masks=None, point_sightings=None):
    """Score visible CAD surfaces, with a separate complete image-shape gate.

    Hidden undersides/backfaces are not sparse-map errors. A mismatched model
    cannot exploit visibility filtering: its entire observable silhouette must
    independently pass the ordinary IoU/local-difference criterion.
    """
    from calibrated_scene_reconstruction.scene.objects.object_refine import pieces
    settings = Settings(points_per_link=600 if point_sightings is not None else 160)
    sampled = sample_surfaces(model, settings)
    transforms = model.transforms(candidate['estimated_joints_rad'])
    yaw = rotation(np.radians(candidate['yaw_deg']))
    k = candidate['map_units_per_metre']
    points, normals, spans = [], [], []
    offset = 0
    for name, local in sampled.items():
        T = transforms[name]
        points.append(k*(local@T[:3, :3].T+T[:3, 3])@yaw.T+[
            *candidate['base_xy_map'], candidate['base_z_map']])
        ids = np.linspace(0, len(model.surfaces[name])-1, len(local)).astype(int)
        if name not in model.normals:
            raise ValueError('Visible CAD scoring requires area-sampled outward normals')
        normals.append(model.normals[name][ids]@T[:3, :3].T@yaw.T)
        spans.append(np.arange(offset, offset+len(local)))
        offset += len(local)
    points, normals = np.concatenate(points), np.concatenate(normals)
    triangles = []
    if predicted_masks is None:
        for name in sampled:
            if name not in model.triangles:
                raise ValueError('Complete silhouette scoring requires actual CAD triangles')
            T = transforms[name]
            local = model.triangles[name]
            triangles.append(k*(local@T[:3, :3].T+T[:3, 3])@yaw.T+[
                *candidate['base_xy_map'], candidate['base_z_map']])
        triangles = np.concatenate(triangles)
    elif len(predicted_masks) != len(views):
        raise ValueError('Predicted masks and camera views differ')
    observed=np.asarray(observed,float)
    selected_indices=np.random.default_rng(0).permutation(len(observed))[:settings.max_scene_points]
    validation_indices=selected_indices[::4]
    validation=observed[validation_indices]
    if point_sightings is not None:
        point_sightings=np.asarray(point_sightings,bool)
        if point_sightings.shape!=(len(observed),len(views)):
            raise ValueError('Actual track sightings must match the observed points and bound cameras')
        actual_sightings=point_sightings[validation_indices]
        depth_sightings=np.zeros(len(points),int)
    sightings = np.zeros(len(points), int)
    ious, local_differences, tiles = [], [], []
    for view_index, view in enumerate(views):
        q = (points-view['C'])@view['R']
        with np.errstate(divide='ignore', invalid='ignore'):
            pixel = q[:, :2]/q[:, 2:]*[view['K'][0,0],view['K'][1,1]]+view['K'][:2,2]
        h,w = view['mask'].shape
        valid = ((q[:, 2]>.05)&np.isfinite(pixel).all(1)&(pixel[:,0]>=0)&(pixel[:,0]<w)&
                 (pixel[:,1]>=0)&(pixel[:,1]<h))
        ids=np.flatnonzero(valid)
        coords=pixel[ids].astype(int)
        front=((view['C']-points[ids])*normals[ids]).sum(1)>0
        visible=front & view['mask'][coords[:,1],coords[:,0]] & ~view['ignored'][coords[:,1],coords[:,0]]
        sightings[ids[visible]] += 1
        if point_sightings is not None:
            measured=validation[actual_sightings[:,view_index]]
            measured_camera=(measured-view['C'])@view['R']
            measured_camera=measured_camera[measured_camera[:,2]>.05]
            if len(measured_camera):
                measured_uv=measured_camera[:,:2]/measured_camera[:,2:]*[view['K'][0,0],view['K'][1,1]]+view['K'][:2,2]
                pixel_distance=cKDTree(measured_uv).query(pixel[ids])[0]
                radius=max(view['K'][0,0],view['K'][1,1])*(.02*k)/q[ids,2]
                depth_sightings[ids[visible&(pixel_distance<=radius)]]+=1
        if predicted_masks is not None:
            predicted=np.asarray(predicted_masks[view_index],bool)
            if predicted.shape != (h,w):
                raise ValueError('Prediction shape differs from bound camera evidence')
        else:
            predicted=np.zeros((h,w),'uint8')
            # Raster the real CAD topology. Convex hulls fill articulated torso
            # openings and would falsely penalize an otherwise correct robot.
            tq=(triangles-view['C'])@view['R']
            valid_triangle=(tq[:,:,2]>.05).all(1)
            tq=tq[valid_triangle]
            tp=tq[:,:,:2]/tq[:,:,2:]*[view['K'][0,0],view['K'][1,1]]+view['K'][:2,2]
            in_image=(tp[:,:,0].max(1)>=0)&(tp[:,:,0].min(1)<w)&(tp[:,:,1].max(1)>=0)&(tp[:,:,1].min(1)<h)
            pixels=np.rint(np.clip(tp[in_image],-100000,100000)).astype('int32')
            for triangle in pixels:
                cv2.fillConvexPoly(predicted,triangle,1)
        actual=view['mask'] & ~view['ignored']
        predicted=(predicted>0) & ~view['ignored']
        ious.append(float((actual&predicted).sum()/max(1,(actual|predicted).sum())))
        local_differences.append(float(sum(p[0] for p in pieces(actual,predicted,.003))))
        if sheet is not None:
            rgb=cv2.imread(str(view['inputs']['mask'].parent/view['name']))
            rgb=cv2.resize(rgb,(w,h))
            rgb=(rgb*.3).astype('uint8')
            rgb[actual&predicted]=(225,225,225)
            rgb[actual&~predicted]=(45,45,230)
            rgb[predicted&~actual]=(40,220,40)
            cv2.putText(rgb,view['name']+f' IoU{ious[-1]:.2f}',(4,20),0,.5,(0,255,255),1)
            tiles.append(cv2.resize(rgb,(320,320)))
    all_visible=sightings>=2
    visible=all_visible if point_sightings is None else depth_sightings>=2
    distance=cKDTree(validation).query(points)[0]/k
    visible_distance=distance[visible]
    reasons=[]
    if len(visible_distance)<100:
        reasons.append('Fewer than100 CAD samples are visible in two independent labelled views')
    residual=float(np.median(np.sort(visible_distance)[:max(1,int(.65*len(visible_distance)))])) if len(visible_distance) else 1e6
    coverage=float((visible_distance<.02).mean()) if len(visible_distance) else 0.
    iou,local=float(np.median(ious)),float(np.median(local_differences))
    supported_links = [name for name, ids in zip(sampled, spans)
                       if visible[ids].sum() >= 20 and
                       (distance[ids][visible[ids]] < .02).mean() >= .2]
    if len(supported_links) < 2:
        reasons.append('Fewer than two visible CAD links have independent surface support')
    if residual>.02 or coverage<.3:
        reasons.append('Visible CAD surface residual/support fails the shared20mm/30% criterion')
    if iou<.85 or local>.02 or len(views)<8:
        reasons.append('Complete visible robot silhouettes fail IoU/local-difference/view-count criteria')
    if sheet is not None:
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        Path(sheet).parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(sheet),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    return {'ok':not reasons,'reasons':reasons,'residual_median_m':residual,'surface_coverage':coverage,
            'supported_links':supported_links,'validation_points':len(validation),
            'visible_model_points':int(visible.sum()),'all_model_points':len(points),
            'all_surface_coverage':float((distance<.02).mean()),
            'all_visible_surface_coverage':float((distance[all_visible]<.02).mean()) if all_visible.any() else 0.,
            'metric_criterion_version':'depth_observed_v2' if point_sightings is not None else'visible_v1',
            'depth_observed_model_points':int(visible.sum()),
            'silhouette_iou_median':iou,'silhouette_local_difference':local,
            'silhouette_ious':ious,'silhouette_local_per_view':local_differences,
            'views':len(views),'criterion_scope':('CAD samples with actual independent validation-depth observations in>=2 views'
                if point_sightings is not None else'CAD samples facing and labelled visible in>=2 views')+
                ';20mm/30%/two supported links unchanged; complete silhouettes gated separately; no measured sizes'}
