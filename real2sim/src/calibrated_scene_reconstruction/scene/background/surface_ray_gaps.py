"""Test photographic GS ownership with camera-space near-plane clipping.

Two separated, same-owner, measured opaque source rays are required. Complete
closed native back surfaces remain for unseen directions. This policy estimates
appearance ownership, not metric depth; free-view/whole-scene QA still decides.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from plyfile import PlyData

from calibrated_scene_reconstruction.assets.topology import closed_oriented_surface
from calibrated_scene_reconstruction.scene.background.crop_gaps import bake as bake_atlas
from calibrated_scene_reconstruction.scene.background.clipped_projection import depth_buffer
from calibrated_scene_reconstruction.scene.review.environment import digest


def supported(votes, previous, points, centre, good):
    before = points-previous; after = points-centre
    cosine = np.einsum('ij,ij->i', before, after)/np.maximum(1e-12, np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
    independent = (votes == 0)|((np.linalg.norm(previous-centre,axis=1) > .05)&(np.degrees(np.arccos(np.clip(cosine,-1,1))) >= 2.))
    selected = good&independent; votes[selected] += 1; previous[selected] = centre


def prepare(session, context_path, coverage_path, owners, output):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
    from calibrated_scene_reconstruction.scene.background import mesh_textures, crop_gaps, clipped_projection
    context_path, coverage_path, out = map(lambda p: Path(p).resolve(), [context_path, coverage_path, output])
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local source-ray proposal required')
    built = verify(session); current(session)
    context = json.loads(context_path.read_text()); coverage = json.loads(coverage_path.read_text())
    if digest(context_path) != built['splat_context']['evidence_sha256'] or context['input_sha256'].get(str(coverage_path)) != digest(coverage_path):
        raise ValueError('Use current field and already bound actual opaque coverage')
    job_path = Path(json.loads((session/'qa/environment_physics/twin/current.json').read_text())['path']).parent/'job.json'
    job = json.loads(job_path.read_text()); geometry = np.load(job['geometry'])
    if geometry_signature(job) != context['selection_frozen']['geometry_signature']:
        raise ValueError('Actual native geometry differs from the measured field binding')
    field = PlyData.read(str(context_path.parent/'context.ply'))['vertex'].data
    field_sha = hashlib.sha256(field.tobytes()).hexdigest()
    if coverage.get('version') != 2 or not coverage.get('opacity_evidence') or coverage['field_rows_sha256'] != field_sha:
        raise ValueError('Same-field backdrop-invariant opacity required')
    if not owners or len(set(owners)) != len(owners):raise ValueError('Explicit distinct photographed owners required')
    for owner in owners:
        row = context['appearance_ownership'].get(owner, {})
        if row.get('role') != 'gap_fill' or row.get('preserve_gs') is not True or row.get('mesh_coverage', {}).get('compositing') != 'captured_projection':
            raise ValueError('Source rays are only for existing GS-preserving photographed owners')
    inputs = {**context['input_sha256'], **coverage['input_sha256']}
    for p in [context_path, coverage_path, job_path, Path(job['geometry']), Path(__file__).resolve(),
              Path(mesh_textures.__file__).resolve(), Path(crop_gaps.__file__).resolve(), Path(clipped_projection.__file__).resolve()]:inputs[str(p)] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()):raise ValueError('Source-ray evidence changed')
    out.mkdir(parents=True); shutil.copyfile(context_path.parent/'context.ply', out/'context.ply')
    result = copy.deepcopy(context); cameras = np.load(session/'parsing/posed_frames/cameras.npz'); names = list(map(str,cameras['names']))
    for owner in owners:
        rows = [r for r in job['parts'] if r['entity'] == owner and not r['external']]
        V, F, offset = [], [], 0
        for i, row in enumerate(job['parts']):
            if row not in rows:continue
            vertices, faces = geometry['v'+str(i)], geometry['f'+str(i)]
            if not closed_oriented_surface(vertices, np.full(len(faces),3), faces.ravel()):
                raise ValueError('Source-ray layering needs complete closed oriented native solids')
            V.append(vertices); F.append(faces+offset); offset += len(vertices)
        V, F = np.concatenate(V), np.concatenate(F); views = []
        for row in coverage['rows']:
            if row['entity'] != owner or row['pixels'] < 32:continue
            i = names.index(row['frame']); w,h = row['size']; K = cameras['K'][i].copy(); K[0] *= w/cameras['size'][i,0]; K[1] *= h/cameras['size'][i,1]
            mask = cv2.imread(row['mask'],0)
            if digest(row['mask']) != row['mask_sha256']:raise ValueError('Measured source mask changed')
            depth = depth_buffer(V,F,K,cameras['R_world_cam'][i],cameras['centre'][i],(h,w))
            if np.count_nonzero((mask > 128)&np.isfinite(depth)) < 32:continue
            views.append((cameras['centre'][i],cameras['R_world_cam'][i],K,mask,depth))
        if len(views) < 2:raise ValueError('Two actual same-owner source projections required: '+owner)
        def inside(samples, normal):
            result = []
            for start in range(0,len(samples),32768):
                P = samples[start:start+32768]; votes = np.zeros(len(P),np.uint16); previous = np.full_like(P,np.nan)
                for centre,R,K,mask,depth in views:
                    pc = (P-centre)@R; uv = pc@K.T; uv = uv[:,:2]/np.maximum(pc[:,2,None],1e-9); h,w = mask.shape
                    valid = (pc[:,2] > .1)&(uv[:,0] >= 0)&(uv[:,0] < w)&(uv[:,1] >= 0)&(uv[:,1] < h)&((centre-P)@normal > 1e-5)
                    ids = np.flatnonzero(valid); xy = uv[ids].astype(int); good = np.zeros(len(P),bool)
                    good[ids] = (mask[xy[:,1],xy[:,0]] > 128)&(pc[ids,2]-depth[xy[:,1],xy[:,0]] >= -.02)
                    supported(votes,previous,P,centre,good)
                result.append(votes < 2)
            return np.concatenate(result)
        declaration = dict(result['appearance_ownership'][owner]['mesh_coverage'])
        declaration.pop('depth_band_m', None); declaration.pop('radius_m', None)
        declaration.update(evidence_mode='opaque_source_rays', ray_mode='opaque_layers', closed_shell=True)
        result['appearance_ownership'][owner]['mesh_coverage'] = declaration
        result['appearance_ownership'][owner]['basis'] += ' Source-ray appearance trial: Gaussian centres are not measured physical surfaces. Keep real opaque same-owner photography; hidden directions stay complete native geometry. Actual source/free-view review remains required.'
        directory = out/owner; directory.mkdir()
        masks, evidence = bake_atlas(job,owner,declaration,directory,inside,{'capture_coverage':{
            'field_rows_sha256':field_sha,'views':len(views),'coverage_entity':owner,'evidence_mode':'opaque_source_rays',
            'projection':'camera_space_near_plane_clipped', 'minimum_separated_views':2,'minimum_ray_angle_deg':2.,'ray_mode':'opaque_layers',
            'scope':__doc__.strip()}},inputs,face_normals=True)
        for payload in masks['meshes'].values():payload['one_sided_closed'] = True
        result['gap_masks'][owner] = masks; inputs.update(evidence)
    inputs[str(out/'context.ply')] = digest(out/'context.ply')
    result.update(input_sha256=inputs, photorealism_approved=False,
                  source_ray_trial={'owners':owners,'field_rows_unchanged':True,'appearance_pass':False,'scope':__doc__.strip()})
    if any(digest(p) != sha for p, sha in inputs.items()):raise ValueError('Source-ray evidence changed while baking')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('session'); p.add_argument('context'); p.add_argument('coverage')
    p.add_argument('--owners', nargs='+', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    result = prepare(Session(a.session),a.context,a.coverage,a.owners,a.output)
    print('Unapproved source-ray GS trial:',result['retained'],'unchanged rows')


if __name__ == '__main__':main()
