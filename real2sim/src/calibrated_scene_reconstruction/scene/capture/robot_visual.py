"""CAD anchor application and verification for visual-only and uncertain-command routes."""
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.scene.session import Session
from .articulated_model import from_usd
from .visual_robot_fit import Settings, fit, rotation


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def associate_robot_tracks(points, evidence):
    """Associate sparse components by shared actual sightings, before CAD fitting.

    A disconnected wheeled base remains valid when it is observed alongside the
    dominant body. A background device seen only in unrelated cameras cannot
    identify itself as this robot merely by receiving repeated semantic labels.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points=np.asarray(points,float)
    if len(points)<11:raise ValueError('Too few tracks to associate robot identity')
    tree=cKDTree(points)
    radius=10*float(np.median(tree.query(points,k=11)[0][:,-1]))
    if radius<=0:raise ValueError('Degenerate robot track geometry')
    pairs=tree.query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    count,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels);dominant=int(np.argmax(sizes))
    if sizes[dominant]<.6*len(points):
        raise ValueError('No dominant spatial robot identity; review semantic instance labels')
    sightings=[set(frames) for frames in evidence['positive_observation_frames']]
    body_frames=set().union(*(sightings[i] for i in np.flatnonzero(labels==dominant)))
    keep=np.zeros(len(points),bool);components=[]
    for component in range(count):
        ids=np.flatnonzero(labels==component)
        frames=set().union(*(sightings[i] for i in ids))
        shared=len(body_frames & frames)
        accepted=component==dominant or shared>=2
        if accepted:keep[ids]=True
        components.append({'points':len(ids),'shared_body_views':shared,'associated':accepted})
    evidence={**evidence,'raw_robot_points':len(points),'robot_points':int(keep.sum()),
              'track_identity':{'radius_map_units':radius,'components':components,
                                'rule':'dominant spatial component>=60%; other components share>=2 actual body views'}}
    for key in ('point_ids','positive_observation_frames'):
        evidence[key]=[value for value,selected in zip(evidence[key],keep) if selected]
    return points[keep],evidence


def observation_rejections(reconstruction, rejected_frames=None):
    """Named raw-image identity/occlusion review, independent of CAD fit errors."""
    rejected_frames={} if rejected_frames is None else rejected_frames
    if (not isinstance(rejected_frames, dict) or
            any(not isinstance(reason,str) or not reason.strip() for reason in rejected_frames.values())):
        raise ValueError('Observation rejections require frame names and nonempty semantic reasons')
    unknown=set(rejected_frames)-{image.name for image in reconstruction.images.values()}
    if unknown:raise ValueError('Rejected observation frames are not registered: '+str(sorted(unknown)))
    return rejected_frames


def observed_tracks(session, reconstruction, suffix='_visual_robot', max_robot_fraction=None,
                    occluder_suffix=None, excluded_suffixes=('_visual_hands',), rejected_frames=None,
                    require_exclusions=False):
    """Only actual robot-labelled observations vote, never projected hidden points."""
    from .valid_pixels import ValidPixels
    validity=ValidPixels(session)
    rejected_frames=observation_rejections(reconstruction,rejected_frames)
    votes, seen, inputs, positive_frames = {}, {}, {}, {}
    image_dir = session / 'splat/data/images'
    used, rejected = [], []
    from .calibration_scope import allowed
    scope=allowed(session,[image.name for image in reconstruction.images.values()])
    if hasattr(session,'config_dir') and (session.config_dir/'inputs.json').exists():
        inputs[rel(session.config_dir/'inputs.json')]=digest(session.config_dir/'inputs.json')
    for image in reconstruction.images.values():
        if image.name not in scope:
            rejected.append({'frame':image.name,'reason':'Outside declared metric source/interval','source':'input_plan'});continue
        if image.name in rejected_frames:
            rejected.append({'frame':image.name,'reason':rejected_frames[image.name],'source':'raw_image_review'})
            continue
        path = image_dir / (Path(image.name).stem+suffix+'.png')
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.erode((mask < 128).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if max_robot_fraction is not None:
            whole_path = image_dir/(Path(image.name).stem+'_visual_robot.png')
            if not whole_path.exists():
                continue
            whole = cv2.imread(str(whole_path), 0) < 128
            if (mask & whole).sum() > max_robot_fraction*whole.sum():
                rejected.append({'frame': image.name, 'reason': 'part mask consumes too much robot area'})
                continue
            mask &= whole
        camera = reconstruction.cameras[image.camera_id]
        unknown = ~validity.get(image.name,mask.shape)
        if occluder_suffix is not None:
            occluder_path = image_dir/(Path(image.name).stem+occluder_suffix+'.png')
            if not occluder_path.exists():
                continue
            unknown |= cv2.imread(str(occluder_path), 0) < 128
            inputs[rel(occluder_path)] = digest(occluder_path)
        ambiguous_part=False
        for excluded_suffix in excluded_suffixes:
            excluded_path=image_dir/(Path(image.name).stem+excluded_suffix+'.png')
            if not excluded_path.exists():
                if require_exclusions:
                    rejected.append({'frame':image.name,'reason':'Missing required unestimated-part observation'})
                    ambiguous_part=True
                    break
                continue
            excluded=cv2.imread(str(excluded_path),0)<128
            inputs[rel(excluded_path)]=digest(excluded_path)
            if (excluded & mask).sum() > .25*max(1,mask.sum()):
                rejected.append({'frame':image.name,'reason':'unestimated part consumes over25% of body'})
                ambiguous_part=True;break
            unknown=excluded if unknown is None else unknown|excluded
        if ambiguous_part:continue
        scale = np.array([mask.shape[1]/camera.width, mask.shape[0]/camera.height])
        for observation in image.points2D:
            if not observation.has_point3D():
                continue
            identity = observation.point3D_id
            point = reconstruction.points3D[identity]
            if point.track.length() < 3 or point.error >= 1.5:
                continue
            pixel = observation.xy*scale
            x, y = pixel.astype(int)
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if unknown is not None and unknown[y,x]:
                    continue  # contradictory/occluded pixels are unknown, not negative sightings
                seen[identity] = seen.get(identity, 0)+1
                votes[identity] = votes.get(identity, 0)+int(mask[y, x])
                if mask[y,x]:positive_frames.setdefault(identity,set()).add(image.name)
        inputs[rel(path)] = digest(path)
        used.append(image.name)
    accepted = [identity for identity in seen if votes[identity] >= 2
                and votes[identity]/seen[identity] >= .6]
    points = np.array([reconstruction.points3D[identity].xyz for identity in accepted], float).reshape(-1, 3)
    inputs.update(validity.inputs)
    return points, {'frames': sorted(used), 'robot_points': len(points),
                    'labelled_track_points': len(seen), 'mask_inputs': inputs,
                    'point_ids':[int(identity) for identity in accepted],
                    'positive_observation_frames':[sorted(positive_frames[identity]) for identity in accepted],
                    'semantic_rejections': rejected,
                    'criterion': 'track>=3,error<1.5px,>=2 actual robot-mask sightings,>=60% positive votes'}


def overlay(session, reconstruction, model, result, sim, directory):
    """Show inferred surfaces in original cameras; not a measured-size check."""
    c, R, t = float(sim['scale']), np.asarray(sim['R']), np.asarray(sim['t'])
    state = result['estimated_joints_rad']
    points, _, _ = model.points(state, exclude_links=Settings().exclude_links, per_link=100)
    k = result['map_units_per_metre']
    world = k*points@rotation(np.radians(result['yaw_deg'])).T + [
        *result['base_xy_map'], result['base_z_map']]
    mapped = (world-t)@R/c
    views = sorted(reconstruction.images.values(), key=lambda image: image.name)
    tiles = []
    for i in np.linspace(0, len(views)-1, min(8, len(views))).astype(int):
        image = views[i]
        frame = cv2.imread(str(session/'splat/data/images'/image.name))
        matrix = np.asarray(image.cam_from_world().matrix())
        camera = reconstruction.cameras[image.camera_id]
        fx, fy, cx, cy = np.asarray(camera.params[:4], float)
        q = mapped@matrix[:, :3].T+matrix[:, 3]
        valid = q[:, 2] > .1
        pixel = q[valid, :2]/q[valid, 2:]*[fx, fy]+[cx, cy]
        for x, y in np.rint(pixel).astype(int):
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 2, (0, 170, 255), -1)
        cv2.putText(frame, image.name+' inferred CAD', (12, 35), 0, .8, (0, 255, 255), 2)
        tiles.append(cv2.resize(frame, (640, 360)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory/'robot_visual_frames.jpg'), np.vstack([
        np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))


def rescale_graph(graph, scale, origin, floor_offset_m=0.):
    """Keep the provisional draft consistent with a floor-preserving similarity."""
    def point(value):
        result = (np.asarray(value, float)-origin)/scale+origin
        result[2] -= floor_offset_m
        return result.tolist()
    for furniture in graph.get('furniture', {}).values():
        if (furniture.get('fix') or {}).get('model'):
            raise ValueError('Calibrate before exporting task-support models')
        if 'box' in furniture:
            furniture['box'] = [point(corner) for corner in furniture['box']]
    for group in ('dynamic_objects', 'static_objects'):
        for obj in graph.get(group, {}).values():
            if (obj.get('fit') or {}).get('accepted'):
                raise ValueError('Recalibration cannot mutate an accepted object; start a new reconstruction')
            fix = obj.get('fix') or {}
            if fix.get('mesh') or fix.get('model'):
                raise ValueError('Calibrate before exporting task models')
            if 'xy' in fix:
                fix['xy'] = point([*fix['xy'], 0.])[:2]
            if 'size' in fix:
                fix['size'] = (np.asarray(fix['size'])/scale).tolist()
            if '_size' in obj:
                obj['_size'] = (np.asarray(obj['_size'])/scale).tolist()
            for key in ('support_z', 'local_top', 'down_offset'):
                if key in fix:
                    fix[key] /= scale
                    if key in ('support_z', 'local_top'):
                        fix[key] -= floor_offset_m
    return graph


def rescale_alignment(alignment, scale, origin, floor_offset_m=0.):
    """Transform metric diagnostics together with cameras, preserving floor zero."""
    sim = alignment['sim3']
    alignment['sim3_before_visual'] = sim
    alignment['sim3'] = {'scale': sim['scale']/scale, 'R': sim['R'],
                         't': ((np.asarray(sim['t'])-origin)/scale+origin).tolist()}
    alignment['sim3']['t'][2] -= floor_offset_m
    checks = alignment.get('checks', {})
    alignment['provisional_checks_before_visual'] = json.loads(json.dumps(checks))
    for key in ('floor_z', 'support_top_z', 'twin_support_top_z'):
        if key in checks:
            checks[key] /= scale
            checks[key] -= floor_offset_m
    if 'key_residual_m' in checks:
        checks['key_residual_m'] = (np.asarray(checks['key_residual_m'])/scale).tolist()
    for key in ('support_top_xy', 'twin_support_xy'):
        if key in checks:
            checks[key] = ((np.asarray(checks[key])-origin[:2])/scale+origin[:2]).tolist()
    return alignment


def verify_anchor(session, graph, joint_prior=False):
    result_name='robot_joint_fit.json' if joint_prior else 'robot_visual_fit.json'
    result = json.loads((session/'splat'/result_name).read_text())
    if (result.get('ok') is not True or result.get('applied') is not True or
            result.get('supplied_joint_config_read') is not joint_prior):
        route='Joint-prior' if joint_prior else 'No-joint'
        raise ValueError(route+' CAD anchor has not passed with matching provenance and been applied')
    if joint_prior:
        from .joint_prior import audit
        if result.get('calibration_mode')!='joint_prior' or not audit(
                result['joint_uncertainty'],result['estimated_joints_rad'],result.get('floor_offset_m',0.))['ok']:
            raise ValueError('Joint-prior calibration has not passed its execution uncertainty audit')
    if not result.get('input_sha256'):
        raise ValueError('Visual CAD anchor has no source fingerprints')
    paths = {**result['input_sha256'], **result['observation_evidence']['mask_inputs'],
             **result.get('fixed_base_evidence', {}).get('mask_inputs', {})}
    for path, expected in paths.items():
        if digest(resolve(path)) != expected:
            raise ValueError('Visual calibration evidence changed: '+path)
    for path, expected in result.get('derived_outputs_sha256', {}).items():
        if digest(session/path) != expected:
            raise ValueError('Calibrated camera output changed: '+path)
    alignment = json.loads((session/'splat/align.json').read_text())
    if alignment['sim3'] != result['applied_sim3']:
        raise ValueError('Camera alignment changed since visual calibration')
    if digest(session/'parsing/robot_visual_state.json') != result['visual_state_sha256']:
        raise ValueError('Inferred visual joint state changed since calibration')
    photo = graph.get('photo_robot', {})
    if photo.get('base') != result['applied_base'] or photo.get('spec') != result['robot']:
        raise ValueError('Rendered robot pose differs from its visual calibration')
    certificate = session/'splat/robot_visual_validation.json'
    if certificate.exists() and not joint_prior:
        binding = json.loads(certificate.read_text())
        if binding['anchor_sha256'] != digest(session/'splat/robot_visual_fit.json'):
            raise ValueError('Validation belongs to a different applied anchor')
        for path, expected in binding['input_sha256'].items():
            if digest(resolve(path)) != expected:
                raise ValueError('Applied-anchor validation evidence changed: '+path)
    return result


def apply_result(s, result):
    """Apply a verified visual-only calibration before any model export."""
    return _apply_result(s,result,joint_prior=False)


def apply_joint_result(s, result):
    """Apply only image-supported command corrections, never an exact-pose ICP."""
    from .joint_prior import audit
    if result.get('calibration_mode')!='joint_prior' or not audit(
            result['joint_uncertainty'],result['estimated_joints_rad'],result.get('floor_offset_m',0.))['ok']:
        raise ValueError('Joint-prior execution uncertainty audit did not pass')
    return _apply_result(s,result,joint_prior=True)


def _apply_result(s, result, joint_prior):
    if (s/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: preserve the approved calibration')
    if not result.get('ok') or result.get('supplied_joint_config_read') is not joint_prior:
        raise ValueError('Only a passing calibration with matching joint-input provenance may be applied')
    if result.get('measured_task_or_support_size_read') is not False:
        raise ValueError('Task/support truth is forbidden in deployment calibration')
    out=s/'splat'
    alignment=json.loads((out/'align.json').read_text())
    if alignment.get('robot_visual_anchor') or alignment.get('robot_joint_anchor'):
        raise ValueError('Visual calibration already applied')
    for path, expected in result['input_sha256'].items():
        if digest(resolve(path)) != expected:
            raise ValueError('Visual calibration evidence changed: '+path)
    presentation = None
    if joint_prior:
        from .joint_prior import presentation_state
        model = from_usd(load_spec(result['robot'])['usd'])
        presentation = presentation_state(model, s.config('robot_state.json'), result['estimated_joints_rad'])
    graph = s.scene_graph
    lo, hi = np.asarray(graph['furniture'][graph['task']['support']]['box'])
    origin = np.r_[(lo[:2]+hi[:2])/2, 0.]
    k = result['map_units_per_metre']
    floor_offset = float(result.get('floor_offset_m', 0.)) if result.get('floor_uncertainty') else 0.
    graph = rescale_graph(graph, k, origin, floor_offset)
    base = (np.r_[result['base_xy_map'], result['base_z_map']]-origin)/k+origin
    base[2] -= floor_offset
    alignment = rescale_alignment(alignment, k, origin, floor_offset)
    if floor_offset:
        alignment['floor_rebase'] = {'offset_m': floor_offset, 'source': result['floor_uncertainty'],
                                   'cad_wheel_floor_z': 0., 'note': 'Task geometry and cameras shifted together; no floating robot'}
    alignment['scale_from'] = ('visual CAD with uncertain commanded joints' if joint_prior else
                               'visual CAD articulation without supplied joint configuration')
    alignment['robot_joint_anchor' if joint_prior else 'robot_visual_anchor'] = {key: result[key] for key in (
        'residual_median_m', 'surface_coverage', 'plausible_scale_spread', 'estimated_joints_rad')}
    photo = graph.setdefault('photo_robot', {})
    photo.update(spec=result['robot'], base={'xy': base[:2].tolist(), 'yaw_deg': result['yaw_deg']},
                 render_visual_state=True, render_capture_state=False,
                 _base_from='Observed video/CAD articulation with uncertain command prior' if joint_prior else
                            'Observed video/CAD articulation; inferred, not a joint readout')
    state = {'robot': result['robot'], 'source': 'refined command prior' if joint_prior else 'inferred visual CAD articulation',
             'joints_rad': result['estimated_joints_rad'], 'unestimated_joints': result['unestimated_joints'],
             'base_height_m': float(base[2]), 'calibration_ok': True}
    if presentation is not None:
        state['presentation'] = presentation
    s.write_json('parsing/robot_visual_state.json', state)
    result['visual_state_sha256'] = digest(s/'parsing/robot_visual_state.json')
    result['applied_sim3'] = alignment['sim3']
    result['applied_base'] = photo['base']
    (out/'align.json').write_text(json.dumps(alignment, indent=2)+'\n')
    (s.config_dir/'scene_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    result['applied'] = True
    return result


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    s = Session(args.session)
    if (s/'qa/independent_freeze.json').exists():
        parser.error('Frozen reconstruction: create a new visual calibration session')
    config = s.config('robot_visual.json', {})
    if config.get('state_input', 'none') != 'none':
        parser.error('robot_visual accepts no supplied joint state')
    spec = load_spec(config.get('robot', 'vega_1p_sharpa_arm'))
    allowed = {field.name for field in fields(Settings)}
    options = config.get('fit', {})
    if set(options)-allowed:
        parser.error('Unknown visual-fit settings: '+str(sorted(set(options)-allowed)))
    settings = Settings(**options)
    out = s/'splat'
    alignment = json.loads((out/'align.json').read_text())
    if args.apply and alignment.get('robot_visual_anchor'):
        parser.error('Visual anchor already applied; use a new session for independent recalibration')
    sim = alignment.get('sim3_before_visual', alignment['sim3'])
    reconstruction = pycolmap.Reconstruction(str(out/'data/sparse/full'))
    observed, evidence = observed_tracks(s, reconstruction)
    c, R, t = float(sim['scale']), np.asarray(sim['R']), np.asarray(sim['t'])
    world = c*observed@R.T+t
    model = from_usd(spec['usd'])
    print(f'[robot_visual] {len(world)} actual robot tracks; no supplied state read', flush=True)
    base_points, base_evidence = observed_tracks(s, reconstruction, '_visual_fixed_base')
    base_world = c*base_points@R.T+t
    initial, base_result = {'home': spec.get('home', {})}, None
    if len(base_world) >= 100:
        rigid = model.fixed_root()
        rigid_points, _, _ = rigid.points()
        # Upper extent above the actual floor is an initialization hypothesis;
        # partial/occluded surfaces still have to pass the full evidence fit.
        height = rigid_points[:, 2].max()-rigid.floor_height()
        scale = max(.01, float(np.percentile(base_world[:, 2], 98)/height))
        base_result = fit(rigid, base_world, {'map_units_per_metre': scale},
                          replace(settings, min_scene_points=100, points_per_link=500))
        if base_result['ok']:
            initial.update(xy=base_result['base_xy_map'], yaw_deg=base_result['yaw_deg'],
                           map_units_per_metre=base_result['map_units_per_metre'])
        print(f'[robot_visual] fixed-base diagnostic '
              f'{"accepted as seed" if base_result["ok"] else "rejected, not a scale constraint"}; '
              f'residual {1000*base_result["residual_median_m"]:.2f}mm', flush=True)
    result = fit(model, world, initial, settings,
                 base_scene=base_world if base_result and base_result['ok'] else None)
    result.update(fixed_base_fit=base_result, fixed_base_evidence=base_evidence)
    result.update(robot=spec['name'], applied=False, observation_evidence=evidence,
                  input_sha256={rel(path): digest(path) for path in
                                [out/'data/sparse/full/points3D.bin', out/'data/sparse/full/images.bin',
                                 out/'data/sparse/full/cameras.bin', *model.dependencies]},
                  supplied_joint_config_read=False, board_input_read=False,
                  measured_task_or_support_size_read=False)
    overlay(s, reconstruction, model, result, sim, s/'qa/splat')
    if result['ok'] and args.apply:
        apply_result(s, result)
    s.write_json('splat/robot_visual_fit.json', result)
    print(f'[robot_visual] {"PASS" if result["ok"] else "ABSTAIN"}: '
          f'residual {1000*result["residual_median_m"]:.2f}mm, '
          f'coverage {100*result["surface_coverage"]:.1f}%, '
          f'scale spread {100*result["plausible_scale_spread"]:.2f}%', flush=True)
    if not result['ok']:
        raise SystemExit('; '.join(result['reasons']))


if __name__ == '__main__':
    main()
