"""Restore nearby original GS kernels without changing their trained attributes.

Independent kernel filtering can destroy accumulated opacity and occlusion.
This is a bounded candidate, never an appearance pass: additional kernels need
two separated captured background observations and frozen foreground exclusion;
source, backdrop and novel-view rendering must still validate the result.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest, protected_geometry, protected_overlap
from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def neighborhood(points, core, radius):
    if isinstance(radius, bool) or not .003 <= radius <= .03:
        raise ValueError('Bounded original-kernel closure radius required')
    points, core = np.asarray(points), np.asarray(core)
    if (points.ndim != 2 or core.ndim != 2 or points.shape[1] != 3
            or core.shape[1] != 3 or not len(core)
            or not np.isfinite(points).all() or not np.isfinite(core).all()):
        raise ValueError('Finite original and retained world points required')
    return cKDTree(core).query(points, workers=4)[0] <= radius


def captured_votes(points, cameras, regions):
    """Latent support, not front visibility: permit occluded neighboring kernels."""
    names = list(map(str, cameras['names']))
    votes = np.zeros(len(points), np.uint16)
    last = np.full((len(points), 3), np.nan)
    seen = set()
    for row in regions['regions']:
        frame = row['frame']
        if frame in seen:
            raise ValueError('Exactly one background observation per frame required')
        seen.add(frame)
        i = names.index(frame)
        mask = cv2.imread(row['mask'], 0)
        if mask is None or list(cameras['size'][i]) != [mask.shape[1], mask.shape[0]]:
            raise ValueError('Calibrated background mask required')
        centre, rotation, K = cameras['centre'][i], cameras['R_world_cam'][i], cameras['K'][i]
        pc = (points-centre) @ rotation
        uv = pc @ K.T
        uv = uv[:, :2]/np.maximum(pc[:, 2, None], 1e-9)
        h, w = mask.shape
        ids = np.flatnonzero((pc[:, 2] > .1) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                            & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        xy = uv[ids].astype(int)
        before, after = points[ids]-last[ids], points[ids]-centre
        cosine = np.einsum('ij,ij->i', before, after)/np.maximum(1e-12, np.linalg.norm(before, axis=1)*np.linalg.norm(after, axis=1))
        independent = (votes[ids] == 0) | ((np.linalg.norm(last[ids]-centre, axis=1) > .05)
                                          & (np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= 2))
        take = ids[(mask[xy[:, 1], xy[:, 0]] > 128) & independent]
        votes[take] += 1
        last[take] = centre
    return votes


def prepare(session, source_session, context_manifest, geometry_job, regions, output, radius=.015):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.completion import inside
    out = Path(output).resolve()
    paths = [Path(p).resolve() for p in [context_manifest, geometry_job, regions]]
    base, job, support = [json.loads(p.read_text()) for p in paths]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen context candidate required')
    verify(session)
    native = session.config('environment.json')
    camera = session/'parsing/posed_frames/cameras.npz'
    if (base['camera_sha256'] != digest(camera) or digest(camera) != digest(source_session/'parsing/posed_frames/cameras.npz')
            or base['protected_task_state'] != protected_state(session) or base['opacity_gain'] != 1
            or base['parameters_sha256'] != parameter_digest(native['parameters'])
            or support['protected_task_state'] != protected_state(session)
            or support['parameters_sha256'] != base['parameters_sha256']
            or support['source_asset_sha256'] != base['source_asset_sha256']
            or support.get('claim_scope') != 'outside_known_furniture'):
        raise ValueError('Closure needs this exact unchanged capture, geometry and original field')
    if job['native_manifest_sha256'] != digest(session/'scene/environment/twin/current.json') or job['geometry_sha256'] != digest(job['geometry']):
        raise ValueError('Use current audited geometry')
    inputs = {**base['input_sha256'], **support['input_sha256']}
    old_jobs = [json.loads(Path(p).read_text()) for p in base['input_sha256'] if Path(p).name == 'job.json']
    old_jobs = [j for j in old_jobs if 'parts' in j and 'geometry' in j]
    if len(old_jobs) != 1 or geometry_signature(old_jobs[0]) != geometry_signature(job):
        raise ValueError('Native geometry changed after GS selection')
    cloud = source_session/'splat/export/background.ply'
    core_cloud = paths[0].parent/'context.ply'
    for p in [*paths, camera, cloud, core_cloud, Path(__file__).resolve(), Path(job['geometry'])]:
        inputs[str(p.resolve())] = digest(p)
    for p, sha in inputs.items():
        if digest(p) != sha:
            raise ValueError('Closure evidence changed: '+p)
    original = PlyData.read(str(cloud))['vertex'].data
    core = PlyData.read(str(core_cloud))['vertex'].data
    matrix = np.asarray(base['matrix'])
    world = lambda v: (np.c_[v['x'], v['y'], v['z'], np.ones(len(v))] @ matrix.T)[:, :3]
    points, core_points = world(original), world(core)
    candidates = neighborhood(points, core_points, radius)
    # Exact structured row membership prevents introducing altered/densified kernels.
    row_type = np.dtype((np.void, original.dtype.itemsize))
    if original.dtype != core.dtype:
        raise ValueError('Closure requires unchanged original kernel attributes')
    core_rows = np.isin(original.view(row_type).ravel(), core.view(row_type).ravel())
    if int(core_rows.sum()) != len(core):
        raise ValueError('Core rows are not an unchanged subset of the original model')
    ids = np.flatnonzero(candidates & ~core_rows)
    votes = captured_votes(points[ids], np.load(camera), support)
    scale = np.cbrt(np.linalg.det(matrix[:3, :3])); rotation = matrix[:3, :3]/scale
    sigma = scale*np.exp(np.c_[original['scale_0'][ids], original['scale_1'][ids], original['scale_2'][ids]]).max(1)
    enclosure = native['completion']
    floor_band = base['quality_filter']['settings']['floor_below_tolerance_m']
    eligible = (votes >= 2) & (sigma <= .065) & inside(points[ids], enclosure['footprint'])
    eligible &= (points[ids, 2] >= enclosure['floor_z']-floor_band) & (points[ids, 2] <= enclosure['ceiling_z'])
    mesh, parts, protected_inputs, robot = protected_geometry(job)
    inputs.update(protected_inputs)
    rejected, protected = protected_overlap(original[ids], points[ids], scale, rotation, mesh, parts)
    added = ids[eligible & ~rejected]
    keep = core_rows.copy(); keep[added] = True
    out.mkdir(parents=True)
    np.savez_compressed(out/'source_rows.npz', retained=np.flatnonzero(keep), added=added)
    PlyData([PlyElement.describe(original[keep], 'vertex')], text=False).write(str(out/'context.ply'))
    asset = out/'context.usdz'
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'), '-m', 'threedgrut.export.scripts.transcode',
                        str(out/'context.ply'), '-o', str(asset), '-f', 'nurec'], check=True, stdout=log, stderr=subprocess.STDOUT)
    result = copy.deepcopy(base)
    result.update(asset_sha256=digest(asset), retained=int(keep.sum()), removed=int((~keep).sum()),
                  photorealism_approved=False, input_sha256=inputs,
                  closure_support={'base_manifest': str(paths[0]), 'radius_m': radius, 'added': len(added),
                                   'nearby_candidates': len(ids), 'protected': protected, 'robot': robot,
                                   'scope': 'Original latent neighbors with two separated background observations. Not a visibility, opacity or appearance certificate.'})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Closure input changed during preparation')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['session', 'source_session', 'context_manifest', 'geometry_job', 'regions']:
        p.add_argument(key)
    p.add_argument('--output', required=True); p.add_argument('--radius', type=float, default=.015)
    a = p.parse_args()
    result = prepare(Session(a.session), Session(a.source_session), a.context_manifest, a.geometry_job, a.regions, a.output, a.radius)
    print('Original GS neighbors restored:', result['closure_support']['added'], 'total:', result['retained'])


if __name__ == '__main__':
    main()
