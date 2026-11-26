"""Close native cutouts where the same GS leaks in checked free views.

Two opaque captured rays cannot certify a static opening for every direction.
Use real field/empty black-white controls in captured and displaced cameras.
Only close existing openings; do not change kernels, geometry or task assets.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background import clipped_projection, crop_gaps
from calibrated_scene_reconstruction.scene.background.causal_coverage import causal_mask
from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
from calibrated_scene_reconstruction.scene.background.gs_opacity import MEASUREMENT_KEYS
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def leaking(black, white, empty_black, empty_white):
    """An occluding frozen object is inapplicable, not a GS transparency veto."""
    opaque = causal_mask(black, white, empty_black, empty_white)
    reaches = abs(empty_white.astype(float)-empty_black).max(2)>32
    reaches = cv2.erode(reaches.astype('uint8'), np.ones((3, 3), 'uint8'))>0
    return reaches & ~opaque


def unsupported(samples, normal, views):
    """Veto only outward-facing, self-visible surface rays with actual leakage."""
    samples = np.asarray(samples); result = np.zeros(len(samples), bool)
    for view in views:
        centre, R, K, bad, depth = (view[k] for k in ('centre', 'R', 'K', 'bad', 'depth'))
        pc = (samples-centre)@R; uv = pc@K.T
        uv = uv[:, :2]/np.maximum(pc[:, 2, None], 1e-9); h, w = bad.shape
        valid = ((pc[:, 2]>.1)&(uv[:, 0]>=0)&(uv[:, 0]<w)&(uv[:, 1]>=0)&(uv[:, 1]<h)
                 &((centre-samples)@normal>1e-5))
        ids = np.flatnonzero(valid); xy = uv[ids].astype(int)
        z = depth[xy[:, 1], xy[:, 0]]
        visible = np.isfinite(z)&(abs(pc[ids, 2]-z)<=.025)
        result[ids] |= visible & bad[xy[:, 1], xy[:, 0]]
    return result


def close_mask(image, payload, pixel_m, inside):
    """Preserve UV charts and every already-native texel, including prior repairs."""
    bad, uv = crop_gaps.atlas(payload['world'], payload['counts'], payload['indices'],
                             inside, pixel_m, face_normals=True)
    if image.shape != bad.shape or not np.allclose(uv, payload['uv'], rtol=0, atol=1e-9):
        raise ValueError('Opacity feedback must keep the exact original atlas coordinates')
    new = np.maximum(image, np.flipud(bad))
    return new, int(np.count_nonzero(new>image))


def prepare(session, context_path, measurement, empty_controls, request_path, output):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current, extract
    context_path, measurement, empty_controls, request_path, out = [Path(p).resolve() for p in
        (context_path, measurement, empty_controls, request_path, output)]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen view-opacity proposal required')
    native = verify(session); physical, _ = current(session)
    context, request = [json.loads(p.read_text()) for p in (context_path, request_path)]
    composite_path = Path(request['composite_review']).resolve(); composite = json.loads(composite_path.read_text())
    if (native['splat_context']['evidence_sha256'] != digest(context_path)
            or context['protected_task_state'] != protected_state(session)
            or composite.get('diagnostic') or composite['usd'] != physical['usd']
            or composite['environment_context']['evidence_sha256'] != digest(context_path)):
        raise ValueError('Current physically checked hybrid and actual composed review required')
    paths = [measurement/'black/review.json', measurement/'white/review.json',
             empty_controls/'black/review.json', empty_controls/'white/review.json']
    reviews = [json.loads(p.read_text()) for p in paths]
    for review, mode, colour in zip(reviews, ['context_only']*2+['context_empty']*2,
                                    [[0.,0.,0.], [1.,1.,1.]]*2):
        if (review.get('diagnostic') != mode or review.get('context_background_override') != colour
                or review.get('static_scene_preserved') is not True
                or any(review.get(k) for k in ('background_field_override', 'background_transform_override', 'source_context_override', 'render_settings'))
                or review['environment_context']['asset_sha256'] != context['asset_sha256']
                or not np.allclose(json.loads(review['environment_context']['matrix_json']), context['matrix'], rtol=0, atol=1e-8)):
            raise ValueError('Actual same-field black/white field and empty controls required')
    if (any(any(r.get(k) != reviews[0].get(k) for k in MEASUREMENT_KEYS) for r in reviews[1:])
            or any(r['effective_photometry'] != composite['effective_photometry'] for r in reviews)):
        raise ValueError('Opacity feedback needs matching cameras and photometry')
    owners = request.get('owners')
    if not isinstance(owners, dict) or not 1 <= len(owners) <= 8:
        raise ValueError('Explicit bounded photographed background owners required')
    for owner, declaration in owners.items():
        role = context['appearance_ownership'].get(owner, {})
        if (role.get('role') != 'gap_fill' or role.get('preserve_gs') is not True
                or role.get('mesh_coverage', {}).get('compositing') != 'captured_projection'
                or not declaration.get('basis', '').strip()
                or len(set(declaration.get('frames', []))) != len(declaration.get('frames', []))
                or not 2 <= len(declaration.get('frames', [])) <= 16):
            raise ValueError('Name existing GS-preserving gap owners and source/free-view pairs')
    job_path = Path(json.loads((session/'qa/environment_physics/twin/current.json').read_text())['path']).parent/'job.json'
    job = json.loads(job_path.read_text()); signature = geometry_signature(job)
    if signature != context['selection_frozen']['geometry_signature']:
        raise ValueError('Current native geometry changed since coverage binding')
    inputs = {**context['input_sha256'], **composite['input_sha256']}
    for review in reviews:
        for p, sha in review['input_sha256'].items():
            if p in inputs and inputs[p] != sha: raise ValueError('Conflicting opacity-control inputs')
            inputs[p] = sha
    modules = [Path(__file__).resolve(), Path(causal_mask.__code__.co_filename).resolve(),
               Path(geometry_signature.__code__.co_filename).resolve(), Path(clipped_projection.__file__).resolve(),
               Path(crop_gaps.__file__).resolve(), Path(extract.__code__.co_filename).resolve()]
    for p in [context_path, request_path, composite_path, job_path, Path(job['geometry']), *paths, *modules]: inputs[str(p)] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Opacity feedback evidence changed')
    # A changed native build may reuse field-only controls only after proving
    # identical oriented world geometry, render settings and actual field.
    stages = [Usd.Stage.Open(p) for p in (reviews[0]['usd'], physical['usd'])]
    meta = [dict(stage.GetRootLayer().customLayerData) for stage in stages]
    if meta[0].get('r2s2r_environment_render') != meta[1].get('r2s2r_environment_render'):
        raise ValueError('Measured and composed render settings differ')
    out.mkdir(parents=True); parts, measured_geometry = extract(stages[0], session.scene_graph['task']['support'], session.scene_graph.get('dynamic_objects', {}))
    p = out/'measured_geometry.npz'; np.savez_compressed(p, **measured_geometry)
    if geometry_signature({'geometry':str(p), 'parts':parts}) != signature:
        raise ValueError('Measured and composed world geometry differ')
    inputs[str(p)] = digest(p)
    camera_path = Path(reviews[0]['camera_manifest']); cameras = json.loads(camera_path.read_text())
    current_cameras = json.loads(Path(composite['camera_manifest']).read_text())
    camera_index = {Path(a['source']).name:a for a in cameras['anchors']}
    for a in current_cameras['anchors']:
        old = camera_index.get(Path(a['source']).name)
        if old is None or a['K'] != old['K'] or a['views'] != old['views']:
            raise ValueError('Composite cameras differ from actual field control cameras')
    indexes = [{Path(v['source']).name:v for v in r['views']} for r in reviews]
    geometry = np.load(job['geometry']); result = copy.deepcopy(context); reports = []; cache = {}
    for owner, declaration in owners.items():
        V, F, offset = [], [], 0
        for i, row in enumerate(job['parts']):
            if row['entity'] != owner or row['external']: continue
            V.append(geometry['v'+str(i)]); F.append(geometry['f'+str(i)]+offset); offset += len(V[-1])
        V, F = np.concatenate(V), np.concatenate(F); views = []
        for frame in declaration['frames']:
            if frame not in camera_index or any(frame not in index for index in indexes): raise ValueError('Matching field control source views required')
            anchor = camera_index[frame]
            if len(anchor['views']) < 2 or anchor['views'][0]['offset_m'] != 0 or not any(v['offset_m'] != 0 for v in anchor['views']):
                raise ValueError('Captured and displaced control cameras required')
            for k, pose in enumerate(anchor['views']):
                key = (frame, k)
                if key not in cache:
                    files = [index[frame]['renders'][k] for index in indexes]
                    images = [cv2.imread(p) for p in files]
                    if any(im is None for im in images): raise ValueError('Actual backdrop control image missing')
                    bad = leaking(*images); h, w = bad.shape
                    width = min(640, w); height = round(h*width/w)
                    bad = cv2.resize(bad.astype('uint8'), (width,height), interpolation=cv2.INTER_NEAREST)>0
                    K = np.asarray(anchor['K'],float).copy(); K[0] *= width/anchor['source_size'][0]; K[1] *= height/anchor['source_size'][1]
                    cache[key] = {'centre':np.asarray(pose['centre']), 'R':np.asarray(pose['R_world_cam']), 'K':K, 'bad':bad}
                    for p in files: inputs[str(Path(p).resolve())] = digest(p)
                view = dict(cache[key]); view['depth'] = clipped_projection.depth_buffer(V, F, view['K'], view['R'], view['centre'], view['bad'].shape)
                views.append(view)
        changed = 0; mesh_reports = []
        for index, (path, payload) in enumerate(result['gap_masks'][owner]['meshes'].items()):
            if digest(payload['path']) != payload['sha256'] or digest(payload['uv_path']) != payload['uv_sha256']:
                raise ValueError('Actual native cutout payload changed')
            image = cv2.imread(payload['path'], 0)
            with np.load(payload['uv_path']) as data:
                repaired, count = close_mask(image, data,
                    result['appearance_ownership'][owner]['mesh_coverage'].get('pixel_m',.01),
                    lambda P,N: unsupported(P,N,views))
            if count:
                target = out/(owner+'_'+str(index)+'.png'); cv2.imwrite(str(target), repaired)
                payload.update(path=str(target), sha256=digest(target), visible_fraction=float((repaired>0).mean()))
                inputs[str(target)] = digest(target); changed += count
            mesh_reports.append({'mesh':path, 'closed_pixels':count})
        reports.append({'owner':owner, 'frames':declaration['frames'], 'tested_views':len(views), 'closed_pixels':changed, 'meshes':mesh_reports, 'basis':declaration['basis']})
    if not any(r['closed_pixels'] for r in reports): raise ValueError('No measured leaking openings to close')
    shutil.copyfile(context_path.parent/'context.ply', out/'context.ply'); inputs[str(out/'context.ply')] = digest(out/'context.ply')
    result.update(input_sha256=inputs, photorealism_approved=False,
        cross_view_opacity_completion={'owners':reports, 'field_rows_unchanged':True, 'only_closes_cutouts':True, 'appearance_pass':False, 'scope':__doc__.strip()})
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Opacity feedback changed during preparation')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'measurement', 'empty_controls', 'request'): p.add_argument(name)
    p.add_argument('--output', required=True); a = p.parse_args()
    result = prepare(Session(a.session), a.context, a.measurement, a.empty_controls, a.request, a.output)
    print('Unapproved view-opacity closures:', [(r['owner'],r['closed_pixels']) for r in result['cross_view_opacity_completion']['owners']])


if __name__ == '__main__': main()
