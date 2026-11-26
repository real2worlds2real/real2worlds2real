"""Propose local GS replacement around audited photographed native surfaces.

Only explicitly named background owners may be affected. Retained kernel rows
stay byte-identical; every removed row is saved. This does not certify that a
replacement is better: fresh actual-field coverage, original-GS preservation,
free-view, physical and full-scene appearance gates remain mandatory.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from plyfile import PlyData,PlyElement

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.context_refresh import backing
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def selection_scope(config, parts):
    entities=config.get('photographic_parts')
    if not isinstance(entities,dict) or not 1<=len(entities)<=16:
        raise ValueError('Explicit bounded background photographic replacements required')
    external={r['entity'] for r in parts if r['external']}
    for entity,proofs in entities.items():
        available={r['part'] for r in parts if r['entity']==entity and not r['external']}
        if entity in external or not available or not isinstance(proofs,dict) or not proofs or not set(proofs)<=available:
            raise ValueError('Photographic replacement must name actual background mesh parts; frozen geometry is protected')
    solids=config.get('solid_parts',{})
    if not isinstance(solids,dict) or not set(solids)<=set(entities):raise ValueError('Solid replacement must stay within photographed background owners')
    for entity,names in solids.items():
        available={r['part'] for r in parts if r['entity']==entity and not r['external']}
        if not isinstance(names,list) or not names or len(set(names))!=len(names) or not set(names)<=available:
            raise ValueError('Explicit actual closed solid parts required')
    replacements=config.get('replacement_parts',{})
    if not isinstance(replacements,dict) or not set(replacements)<=set(entities):
        raise ValueError('Part replacement must stay within photographed background owners')
    for entity,names in replacements.items():
        available={r['part'] for r in parts if r['entity']==entity and not r['external']}
        if (not isinstance(names,list) or not names or len(set(names))!=len(names)
                or not set(names)<=available or not set(entities[entity])<=set(names)
                or not set(solids.get(entity,[]))<=set(names)):
            raise ValueError('Replacement parts must include photographed and solid parts, and exclude unselected contents')
    source=config.get('source_replacements',{})
    if not isinstance(source,dict) or not set(source)<=set(entities):
        raise ValueError('Source replacement must stay within photographed background owners')
    source_only=config.get('source_only_owners',[])
    if (not isinstance(source_only,list) or len(set(source_only))!=len(source_only)
            or not set(source_only)<=set(source) or set(source_only)&set(solids)):
        raise ValueError('Source-only owners need reviewed source masks and cannot use solid volume replacement')
    margin=config.get('surface_margin_m',.035)
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not .01<=margin<=.10:
        raise ValueError('Bounded background replacement margin required')
    return entities,margin


def prepare(session,source_manifest,geometry_job,config,output):
    from pxr import Usd,UsdGeom,UsdShade
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.capture_backing import validate as validate_backing
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    out=Path(output).resolve();source_manifest,geometry_job,config=map(lambda p:Path(p).resolve(),(source_manifest,geometry_job,config))
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen background field proposal required')
    native=verify(session);physical,_=current(session);job=json.loads(geometry_job.read_text())
    if job['native_manifest_sha256']!=digest(session/'scene/environment/twin/current.json') or digest(job['geometry'])!=job['geometry_sha256']:
        raise ValueError('Use current audited geometry')
    if json.loads(Path(job['output']).read_text())['usd']!=physical['usd']:
        raise ValueError('Replacement geometry must bind the actual passed physical export')
    cfg=session.config('environment.json');source=json.loads(source_manifest.read_text())
    if source['camera_sha256']!=digest(session/'parsing/posed_frames/cameras.npz') or source['protected_task_state']!=protected_state(session):
        raise ValueError('Source field capture/frozen state changed')
    request_config=json.loads(config.read_text());selected,margin=selection_scope(request_config,job['parts'])
    inputs=dict(source['input_sha256']);stage=Usd.Stage.Open(physical['usd'])
    for entity,proofs in selected.items():
        for part,proof in proofs.items():
            targets=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and part in str(p.GetPath()).split('/')]
            if len(targets)!=1:raise ValueError('One actual photographed native backing mesh required')
            inputs.update(validate_backing(session,proof,UsdShade.MaterialBindingAPI(targets[0]).ComputeBoundMaterial()[0],targets[0]))
    cloud=source_manifest.parent/'context.ply';vertices=PlyData.read(str(cloud))['vertex'].data
    M=np.asarray(source['matrix']);scale=np.cbrt(np.linalg.det(M[:3,:3]))
    points=(np.c_[vertices['x'],vertices['y'],vertices['z'],np.ones(len(vertices))]@M.T)[:,:3]
    sigma=scale*np.exp(np.c_[vertices['scale_0'],vertices['scale_1'],vertices['scale_2']]).max(1)
    worker=Path(__file__).with_name('field_replace_worker.py').resolve()
    for path in [source_manifest,geometry_job,Path(job['geometry']),config,cloud,Path(__file__).resolve(),worker]:
        inputs[str(path)]=digest(path)
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Local replacement evidence changed')
    out.mkdir(parents=True);np.savez_compressed(out/'cloud_world.npz',points=points,sigma=sigma)
    request={'geometry':job['geometry'],'parts':job['parts'],'cloud':str(out/'cloud_world.npz'),
             'entities':[e for e in selected if e not in request_config.get('source_only_owners',[])],
             'replacement_parts':request_config.get('replacement_parts',{}),
             'margin_m':margin,'output':str(out/'removed.npy')}
    (out/'selection.json').write_text(json.dumps(request,indent=2)+'\n')
    with (out/'selection.log').open('w') as log:
        subprocess.run([str(ROOT/'tools/run_blender.sh'),'--python',str(worker),'--',str(out/'selection.json')],check=True,stdout=log,stderr=subprocess.STDOUT)
    removed=np.load(out/'removed.npy')
    source_report=[]
    if request_config.get('source_replacements'):
        from calibrated_scene_reconstruction.scene.background import source_replace
        semantic,new_inputs,source_report=source_replace.prepare(session,points,request_config['source_replacements'],selected)
        inputs.update(new_inputs);removed|=semantic
    if request_config.get('solid_parts'):
        from calibrated_scene_reconstruction.scene.background import convex_replacements
        geometry=np.load(job['geometry']);radii=margin+np.minimum(sigma*2,.10)
        inputs[str(Path(convex_replacements.__file__).resolve())]=digest(convex_replacements.__file__)
        for entity,names in request_config['solid_parts'].items():
            for i,part in enumerate(job['parts']):
                if part['entity']==entity and part['part'] in names:
                    removed |= convex_replacements.mask(geometry[f'v{i}'],geometry[f'f{i}'],points,radii)
    np.save(out/'removed.npy',removed)
    if removed.shape!=(len(vertices),) or removed.dtype!=bool or not removed.any() or removed.all():
        raise ValueError('Local replacement needs a nonempty strict subset of the trained field')
    np.savez_compressed(out/'source_rows.npz',retained=np.flatnonzero(~removed),removed=np.flatnonzero(removed))
    PlyData([PlyElement.describe(vertices[~removed],'vertex')],text=False).write(str(out/'context.ply'))
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'),'-m','threedgrut.export.scripts.transcode',str(out/'context.ply'),'-o',str(out/'context.usdz'),'-f','nurec'],check=True,stdout=log,stderr=subprocess.STDOUT)
    # No added kernels, colour edits or threshold changes. All previously frozen
    # foreground exclusions are retained by taking a subset of this input field.
    entities=sorted({p['entity'] for p in job['parts'] if not p['external']});owners=backing(entities,cfg['completion'])
    result=copy.deepcopy(source)
    for key in ['selection_frozen','coverage_reuse','quality_filter']:result.pop(key,None)
    for path in [out/'selection.json',out/'source_rows.npz',out/'removed.npy']:
        inputs[str(path)]=digest(path)
    result.update(asset_sha256=digest(out/'context.usdz'),parameters_sha256=parameter_digest(cfg['parameters']),
        source_sha256={str(p.resolve()):digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')},
        native_entities=entities,background_entities=entities,
        replaced_entities=[e for e,r in owners.items() if r['role']=='native_replacement'],
        gap_fill_entities=[e for e,r in owners.items() if r['role']=='inferred_enclosure'],
        appearance_ownership=owners,gap_masks={},input_sha256=inputs,geometry_rebuild_required=True,
        retained=int((~removed).sum()),removed=source.get('removed',0)+int(removed.sum()),photorealism_approved=False,
        native_surface_replacement={'source_manifest':str(source_manifest),'source_manifest_sha256':digest(source_manifest),
            'owners':list(selected),'removed_rows':int(removed.sum()),'row_indices_relative_to':str(cloud),
            'replacement_parts':request_config.get('replacement_parts',{}),'source_replacements':source_report,
            'quality_pass':False,'scope':__doc__.strip()})
    result['field_trial']={**result.get('field_trial',{}),'mode':'local_native_photo_replacement','quality_pass':False}
    result['appearance_plan']={'version':1,'routes':{e:{'route':'mesh','quality':'unresolved',
        'reason':'Provisional backing after local field replacement; fresh appearance evidence required.',
        'scores':[],'free_view_review':{'pass':False,'note':'Local replacement is not approved.'}}
        for e,r in owners.items() if r['role']!='inferred_enclosure'},'input_sha256':{},'protected_task_state':protected_state(session)}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Replacement input changed during export')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['session','source_manifest','geometry_job','config']:p.add_argument(name)
    p.add_argument('--output',required=True);a=p.parse_args()
    q=prepare(Session(a.session),a.source_manifest,a.geometry_job,a.config,a.output)
    print('Unapproved local GS replacement:',q['native_surface_replacement']['removed_rows'],'rows;',q['retained'],'preserved')


if __name__=='__main__':main()
