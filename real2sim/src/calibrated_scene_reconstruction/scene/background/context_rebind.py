"""Rebind native appearance to an unchanged, actually rendered GS selection.

Selecting kernels again while preparing gap masks changes the field whose
coverage was measured. Keep the field bytes fixed and require identical audited
geometry, capture, protected state and parameters. No scene acceptance is made.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest,protected_geometry
from calibrated_scene_reconstruction.scene.background.appearance_ownership import specification
from calibrated_scene_reconstruction.scene.background.appearance_plan import validate as validate_plan
from calibrated_scene_reconstruction.scene.background.capture_gaps import bake
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def geometry_signature(job):
    """Same oriented shape despite index/face enumeration changes; no rescaling."""
    mesh=np.load(job['geometry']);h=hashlib.sha256()
    for i,row in sorted(enumerate(job['parts']),key=lambda item:item[1]['path']):
        vertices=np.rint(mesh[f'v{i}']*1e6).astype('<i8');faces=mesh[f'f{i}'];triangles=[]
        for face in faces:
            tri=[tuple(v) for v in vertices[face]];start=min(range(3),key=lambda k:tri[k]);triangles.append(tuple(tri[start:]+tri[:start]))
        h.update(row['path'].encode());h.update(np.asarray(sorted(triangles),dtype='<i8').tobytes())
    return h.hexdigest()


def prepare(session,context_manifest,config,geometry_job,output,reuse_coverage_geometry_job=None):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    out=Path(output).resolve();context_manifest=Path(context_manifest).resolve();config=Path(config).resolve();geometry_job=Path(geometry_job).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen appearance candidate required')
    original=json.loads(context_manifest.read_text());cfg=json.loads(config.read_text());native=session.config('environment.json');job=json.loads(geometry_job.read_text());camera=session/'parsing/posed_frames/cameras.npz'
    verify(session)
    if (original['protected_task_state']!=protected_state(session) or original['camera_sha256']!=digest(camera)
            or original['parameters_sha256']!=parameter_digest(native['parameters'])):raise ValueError('Reprepare GS selection after capture, protected state or geometry parameters change')
    if job['native_manifest_sha256']!=digest(session/'scene/environment/twin/current.json') or job['geometry_sha256']!=digest(job['geometry']):raise ValueError('Use current audited geometry')
    jobs=[]
    for p,sha in original['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('GS selection evidence changed: '+p)
        if Path(p).name=='job.json':
            candidate=json.loads(Path(p).read_text())
            if 'geometry'in candidate and 'parts'in candidate:jobs.append(candidate)
    signature=geometry_signature(job)
    reuse_inputs={};reuse_record=None
    if original.get('geometry_rebuild_required'):
        coverage=json.loads(Path(cfg['projected_coverage']).read_text());audit=json.loads(Path(job['output']).read_text())
        built=json.loads(Path(json.loads((session/'scene/environment/twin/current.json').read_text())['path']).read_text())
        if reuse_coverage_geometry_job:
            from pxr import Usd,UsdGeom
            previous_path=Path(reuse_coverage_geometry_job).resolve();previous=json.loads(previous_path.read_text())
            before=json.loads(Path(previous['output']).read_text())
            if (coverage.get('version')!=2 or before.get('pass')is not True or coverage.get('render_usd')!=before['usd']
                    or coverage.get('input_sha256',{}).get(before['usd'])!=digest(before['usd'])
                    or previous['geometry_sha256']!=digest(previous['geometry']) or geometry_signature(previous)!=signature):
                raise ValueError('Material-only coverage reuse requires identical audited oriented world geometry')
            old_stage=Usd.Stage.Open(before['usd']);new_stage=Usd.Stage.Open(audit['usd'])
            old_meta=dict(old_stage.GetRootLayer().customLayerData)
            new_meta=dict(new_stage.GetRootLayer().customLayerData)
            if old_meta.get('r2s2r_environment_render')!=new_meta.get('r2s2r_environment_render'):
                raise ValueError('Material-only coverage cannot change render photometry')
            a,b=[meta.get('r2s2r_environment_context') or {} for meta in [old_meta,new_meta]]
            for key in ['asset_sha256','matrix_json','native_light_gain','background_color_linear_json','source_background_scope_json']:
                if a.get(key)!=b.get(key):raise ValueError('Material-only coverage changed its actual field, calibration or background')
            for usd in [before['usd'],audit['usd']]:
                stage=Usd.Stage.Open(usd);root=stage.GetPrimAtPath('/World/SplatBackground')
                if not root or not np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root)).T,np.asarray(original['matrix']),rtol=0,atol=1e-8):
                    raise ValueError('Material-only coverage changed the actual field world transform')
            for p in [previous_path,Path(previous['geometry']),Path(previous['output']),Path(before['usd'])]:reuse_inputs[str(p.resolve())]=digest(p)
            reuse_record={'from_usd':before['usd'],'to_usd':audit['usd'],'geometry_signature':signature,
                          'scope':'UV/material authoring only: unchanged oriented world geometry, field, cameras, protected state and photometry. Old cutouts are rebaked; actual new composite review remains required.'}
        elif (coverage.get('version')!=2 or coverage.get('render_usd')!=audit['usd']
                or coverage.get('environment_asset_sha256')!=digest(built['asset'])
                or coverage.get('input_sha256',{}).get(audit['usd'])!=digest(audit['usd'])):
            raise ValueError('Rebuilt geometry requires fresh actual-field/opacity coverage from this audited export')
    elif original.get('selection_frozen',{}).get('geometry_signature'):
        if signature!=original['selection_frozen']['geometry_signature']:raise ValueError('Native/task geometry changed; rerender and reprepare GS coverage')
    elif not jobs or any(signature!=geometry_signature(old) for old in jobs):raise ValueError('Native/task geometry changed; rerender and reprepare GS coverage')
    # The original row selection already excluded the robot. Require a fresh
    # unchanged audited robot/context readout; never fit or move it here.
    _,_,protected_inputs,_=protected_geometry(job,cfg.get('robot_exclusion_margin_m',.05))
    available={r['entity'] for r in job['parts'] if not r['external']};names=cfg['native_entities'];fills=cfg['gap_fill_entities']
    ownership=specification(cfg['appearance_ownership'],available,names,fills,native['completion'])
    plan,plan_inputs=validate_plan(session,cfg['appearance_plan'],ownership)
    if not cfg.get('projected_coverage'):raise ValueError('Actual rendered retained-field coverage required')
    cloud=context_manifest.parent/'context.ply';field=PlyData.read(str(cloud))['vertex'].data;matrix=np.asarray(original['matrix']);points=(np.c_[field['x'],field['y'],field['z'],np.ones(len(field))]@matrix.T)[:,:3]
    inputs={**original['input_sha256'],**protected_inputs,**plan_inputs,**reuse_inputs}
    inputs.update({str(p):digest(p) for p in [context_manifest,config,geometry_job,Path(job['geometry']),cloud,Path(__file__).resolve()]})
    out.mkdir(parents=True);shutil.copyfile(cloud,out/'context.ply');inputs[str(out/'context.ply')]=digest(out/'context.ply');masks={}
    for entity,row in ownership.items():
        if row['role']!='gap_fill':continue
        declaration=row.get('mesh_coverage')
        if declaration and declaration.get('compositing')=='captured_projection':
            masks[entity],new_inputs=bake(session,job,entity,declaration,field,points,cfg['projected_coverage'],out);inputs.update(new_inputs)
        elif row==original['appearance_ownership'].get(entity) and entity in original['gap_masks']:
            masks[entity]=original['gap_masks'][entity]
        else:raise ValueError('New/changed gap owner needs actual field coverage: '+entity)
    result=copy.deepcopy(original);result.update(appearance_ownership=ownership,appearance_plan=plan,gap_masks=masks,
        geometry_rebuild_required=False,
        native_entities=names+fills,replaced_entities=names,gap_fill_entities=fills,input_sha256=inputs,
        selection_frozen={'manifest':str(context_manifest),'manifest_sha256':digest(context_manifest),'field_rows_sha256':hashlib.sha256(field.tobytes()).hexdigest(),
                          'geometry_signature':geometry_signature(job),'scope':'Only appearance ownership/masks rebound. Original field asset and vertex bytes unchanged; full composite review still required.'},
        photorealism_approved=False)
    if reuse_record:
        result['source_sha256']={str(p.resolve()):digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')}
        result['coverage_reuse']=reuse_record
    # The package is intentionally reused. No point is filtered or transcoded.
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Appearance binding input changed')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('context_manifest');p.add_argument('config');p.add_argument('geometry_job');p.add_argument('--output',required=True);p.add_argument('--reuse-coverage-geometry-job',help='For UV/material edits only: verify identical audited world geometry and field/photometry before rebaking old coverage');a=p.parse_args()
    r=prepare(Session(a.session),a.context_manifest,a.config,a.geometry_job,a.output,a.reuse_coverage_geometry_job);print('Rebound appearance, unchanged GS rows:',r['retained'])


if __name__=='__main__':main()
