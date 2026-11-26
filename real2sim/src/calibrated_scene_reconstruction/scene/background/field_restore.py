"""Propose a jointly trained captured GS field with frozen foreground removed.

Use when independent kernel filtering damages source-supported appearance.
Preserving latent rows is a candidate policy, not proof that every region is
sharp or stable. New actual-field opacity, hybrid coverage, source preservation,
free-view artifact and whole-scene checks remain mandatory.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from plyfile import PlyData, PlyElement

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.blender_env import verify
from calibrated_scene_reconstruction.scene.background.context_refresh import backing
from calibrated_scene_reconstruction.scene.background.observed_context import (
    capture_reference, parameter_digest, protected_geometry, protected_overlap,
)
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state
from calibrated_scene_reconstruction.scene.session import Session


def prepare(session, source_session, geometry_job, output):
    out=Path(output).resolve();job_path=Path(geometry_job).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen captured-field candidate required')
    native=verify(session);job=json.loads(job_path.read_text());cfg=session.config('environment.json')
    if (job['native_manifest_sha256']!=digest(session/'scene/environment/twin/current.json')
            or job['geometry_sha256']!=digest(job['geometry'])):
        raise ValueError('Use current audited native geometry')
    if not native.get('splat_context'):raise ValueError('Bind the original captured context first')
    base_path=Path(cfg['assets'][cfg['splat_context']['evidence']]).resolve();base=json.loads(base_path.read_text())
    reference,reference_inputs=capture_reference(session,source_session,{})
    original=reference/'scene/splat/background.usdz';camera=session/'parsing/posed_frames/cameras.npz'
    if str(original.resolve())!=base['source_asset'] or digest(original)!=base['source_asset_sha256']:
        raise ValueError('Original field must be the same calibrated capture already bound by the scene')
    cloud=source_session/'splat/export/background.ply';vertices=PlyData.read(str(cloud))['vertex'].data
    if any(not np.isfinite(vertices[n]).all() for n in vertices.dtype.names if vertices[n].dtype.kind=='f'):
        raise ValueError('Original trained field contains nonfinite attributes')
    matrix=np.asarray(base['matrix']);points=(np.c_[vertices['x'],vertices['y'],vertices['z'],np.ones(len(vertices))]@matrix.T)[:,:3]
    scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rotation=matrix[:3,:3]/scale
    mesh,parts,inputs,robot=protected_geometry(job);inputs.update(reference_inputs)
    rejected,protection=protected_overlap(vertices,points,scale,rotation,mesh,parts);keep=~rejected
    entities=sorted({r['entity'] for r in job['parts'] if not r['external']});owners=backing(entities,cfg['completion'])
    for p in [job_path,Path(job['geometry']),cloud,camera,base_path,original,Path(__file__).resolve()]:inputs[str(p.resolve())]=digest(p)
    for masks in base['source_background_scope'].values():inputs.update(masks)
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Original field/protected evidence changed')
    out.mkdir(parents=True);np.savez_compressed(out/'source_rows.npz',retained=np.flatnonzero(keep))
    PlyData([PlyElement.describe(vertices[keep],'vertex')],text=False).write(str(out/'context.ply'))
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'),'-m','threedgrut.export.scripts.transcode',str(out/'context.ply'),'-o',str(out/'context.usdz'),'-f','nurec'],check=True,stdout=log,stderr=subprocess.STDOUT)
    result=copy.deepcopy(base)
    for key in ['selection_frozen','quality_filter','closure_support','capture_selection','coverage_reuse']:result.pop(key,None)
    result.update(asset_sha256=digest(out/'context.usdz'),parameters_sha256=parameter_digest(cfg['parameters']),
                  source_sha256={str(p.resolve()):digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')},
                  native_entities=entities,replaced_entities=[e for e,r in owners.items() if r['role']=='native_replacement'],
                  gap_fill_entities=[e for e,r in owners.items() if r['role']=='inferred_enclosure'],background_entities=entities,
                  appearance_ownership=owners,gap_masks={},input_sha256=inputs,source_gaussians=len(vertices),retained=int(keep.sum()),removed=int(rejected.sum()),
                  geometry_rebuild_required=True,photorealism_approved=False,
                  field_trial={'mode':'preserve_trained_rows_except_protected_foreground','protected_overlap':protection,'frozen_robot':robot,'quality_pass':False})
    result['appearance_plan']={'version':1,'routes':{e:{'route':'mesh','quality':'unresolved',
        'reason':'Provisional native backing; actual retained-field coverage and composed review required.',
        'scores':[],'free_view_review':{'pass':False,'note':'Candidate not reviewed.'}}
        for e,r in owners.items() if r['role']!='inferred_enclosure'},'input_sha256':{},'protected_task_state':protected_state(session)}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Captured-field input changed during export')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['session','source_session','geometry_job']:parser.add_argument(name)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=prepare(Session(args.session),Session(args.source_session),args.geometry_job,args.output)
    print('Unapproved captured field:',result['retained'],'original rows; fresh hybrid coverage required')


if __name__=='__main__':main()
