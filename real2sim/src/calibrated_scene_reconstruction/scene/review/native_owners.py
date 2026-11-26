"""Attribute appearance regressions to actual rendered background mesh owners.

This is a diagnostic, never an appearance approval. A private USD copy keeps
geometry, alpha networks, culling and frozen materials; background surfaces get
calibrated identification colours and the GS field is hidden.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


LABEL = 'Native appearance owner diagnostic; GS hidden; no quality approval.'


def identify(rgb, palette, tolerance=.075):
    """Leave blended, shaded or unrecognised pixels unassigned."""
    values=np.asarray(list(palette.values()),float)
    if values.ndim!=2 or values.shape[1]!=3 or not len(values):
        raise ValueError('Nonempty RGB owner palette required')
    if len(np.unique(values,axis=0))!=len(values):raise ValueError('Unique owner colours required')
    if not 0<tolerance<.1:raise ValueError('Bounded diagnostic colour tolerance required')
    data=np.asarray(rgb,float)/255
    if data.ndim!=3 or data.shape[2]!=3:raise ValueError('RGB owner image required')
    best=np.full(data.shape[:2],np.inf);second=best.copy();ids=np.full(data.shape[:2],-1,int)
    for i,value in enumerate(values):
        error=np.max(abs(data-value),axis=2);closer=error<best
        second=np.where(closer,best,np.minimum(second,error))
        best=np.minimum(best,error);ids[closer]=i
    ids[(best>tolerance)|(second-best<.04)]=-1
    return ids


def colour_stage(stage, response, group='entity'):
    from pxr import Gf,Sdf,UsdGeom,UsdShade
    from calibrated_scene_reconstruction.scene.background.display_response import invert
    if group not in ('entity','part'):raise ValueError('Known owner grouping required')
    field=stage.GetPrimAtPath('/World/SplatBackground')
    if field:UsdGeom.Imageable(field).MakeInvisible()
    targets=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith('/World/Environment/'):continue
        owner=None;parent=prim
        while parent and str(parent.GetPath()).startswith('/World/Environment/'):
            owner=parent.GetAttribute('userProperties:r2s2r_entity').Get()
            if owner:break
            parent=parent.GetParent()
        if not owner:raise ValueError('Native background mesh lacks an owner: '+str(prim.GetPath()))
        targets.append((prim,str(prim.GetPath()) if group=='part' else owner))
    owners=sorted({owner for _,owner in targets})
    if not owners:raise ValueError('No background owners in actual USD')
    # Four levels avoid saturated endpoints and stay well separated after RTX.
    levels=[.15,.36,.58,.80] if len(owners)<=64 else [.09,.24,.39,.54,.69,.84]
    if len(owners)>len(levels)**3:raise ValueError('Scope the diagnostic to at most 216 owners')
    palette=dict(zip(owners,itertools.product(levels,repeat=3)))
    for prim,owner in targets:
        subsets=UsdGeom.Subset.GetGeomSubsets(UsdGeom.Mesh(prim),UsdGeom.Tokens.face,'materialBind')
        for target in [prim]+[subset.GetPrim() for subset in subsets]:
            material=UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()[0]
            if not material:raise ValueError('Background diagnostic needs a complete material network')
            path='/World/NativeOwnerDiagnostic/m'+hashlib.sha256(str(target.GetPath()).encode()).hexdigest()[:16]
            if str(material.GetPath()).startswith('/World/NativeOwnerDiagnostic/'):
                raise ValueError('Use an original candidate, not an earlier owner diagnostic')
            clone=UsdShade.Material.Define(stage,path)
            clone.GetPrim().GetReferences().SetReferences([Sdf.Reference(primPath=material.GetPath())])
            original=material.ComputeSurfaceSource()[0]
            shader=UsdShade.Shader(stage.GetPrimAtPath(original.GetPath().ReplacePrefix(material.GetPath(),clone.GetPath())))
            if shader.GetIdAttr().Get()!='UsdPreviewSurface':raise ValueError('Explicit preview surface required')
            radiance=invert(np.array([palette[owner]]),response)[0]
            for name,value in [('diffuseColor',[0,0,0]),('specularColor',[0,0,0]),('emissiveColor',radiance)]:
                attr=shader.CreateInput(name,Sdf.ValueTypeNames.Color3f);attr.DisconnectSource();attr.Set(Gf.Vec3f(*map(float,value)))
            shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
            for name,value in [('ior',1.),('metallic',0.),('clearcoat',0.),('roughness',1.)]:
                attr=shader.CreateInput(name,Sdf.ValueTypeNames.Float);attr.DisconnectSource();attr.Set(value)
            UsdShade.MaterialBindingAPI.Apply(target).Bind(clone)
    return palette


def prepare(composite, preservation, output, maximum=8):
    from calibrated_scene_reconstruction.scene.review import ablation
    from calibrated_scene_reconstruction.scene.background import display_response
    composite,preservation=Path(composite).resolve(),Path(preservation).resolve();out=Path(output).resolve()
    if out.exists() or not 1<=maximum<=16:raise ValueError('New output and bounded camera count required')
    review=json.loads(composite.read_text());audit=json.loads(preservation.read_text())
    if review.get('diagnostic') or Path(audit['composite'])!=composite:raise ValueError('Original composite and its preservation audit required')
    for path,sha in audit['input_sha256'].items():
        if digest(path)!=sha:raise ValueError('Appearance evidence changed: '+path)
    failures=sorted((r for r in audit['scores'] if r['applicable'] and r['pass'] is False),
                    key=lambda r:r['overpainted_pixels'],reverse=True)[:maximum]
    if not failures:return None
    manifest=Path(review['environment']['asset']).parent/'manifest.json'
    # Use the exact immutable native build bound to this renderer, never the
    # current session configuration, which can already belong to a later round.
    native=json.loads(manifest.read_text())
    response_path=Path(native['display_response']['asset'])
    if digest(response_path)!=native['display_response']['sha256']:raise ValueError('Native display response changed')
    response=json.loads(response_path.read_text());inputs={str(p):digest(p) for p in [composite,preservation,manifest,response_path,Path(__file__).resolve(),Path(ablation.__file__).resolve(),Path(display_response.__file__).resolve()]}
    inputs.update(audit['input_sha256'])
    stage=ablation.copy_stage(review['usd'],out/'scene.usda',LABEL)
    palette=colour_stage(stage,response);stage.GetRootLayer().Save()
    cameras=json.loads(Path(review['camera_manifest']).read_text());selected={r['source'] for r in failures}
    cameras['anchors']=[r for r in cameras['anchors'] if str(Path(r['source']).resolve()) in selected]
    if len(cameras['anchors'])!=len(selected):raise ValueError('Owner diagnostic cameras missing source failures')
    for row in cameras['anchors']:row['views']=[v for v in row['views'] if v['offset_m']==0]
    if any(len(row['views'])!=1 for row in cameras['anchors']):raise ValueError('One captured camera required per anchor')
    (out/'cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
    for p in [out/'scene.usda',out/'cameras.json']:inputs[str(p)]=digest(p)
    result={'approved':False,'composite':str(composite),'preservation':str(preservation),
            'palette':palette,'input_sha256':inputs,'scope':LABEL}
    (out/'job.json').write_text(json.dumps(result,indent=2)+'\n');return result


def rank(job, diagnostic, output):
    from calibrated_scene_reconstruction.scene.review.frozen_scope import validate as frozen_scope
    job,diagnostic,out=map(lambda p:Path(p).resolve(),(job,diagnostic,output))
    if out.exists():raise ValueError('Use a new attribution output')
    spec=json.loads(job.read_text());shown=json.loads(diagnostic.read_text())
    if shown.get('diagnostic')!=LABEL:raise ValueError('Actual owner diagnostic render required')
    if shown.get('usd')!=str(job.parent/'scene.usda') or shown.get('camera_manifest')!=str(job.parent/'cameras.json'):
        raise ValueError('Render must bind the prepared owner scene and cameras')
    for p,sha in spec['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Diagnostic source changed: '+p)
    for p,sha in shown['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Rendered diagnostic changed: '+p)
    full=json.loads(Path(spec['composite']).read_text());audit=json.loads(Path(spec['preservation']).read_text())
    original=json.loads(Path(audit['reference']).read_text());empty=json.loads(Path(audit['empty']).read_text())
    for key in ('environment','environment_context','effective_photometry'):
        if full.get(key)!=shown.get(key):raise ValueError('Diagnostic scene/photometry must match candidate')
    frozen,_=frozen_scope(audit['frozen_exclusion'],full) if audit.get('frozen_exclusion') else ({},{})
    scope=json.loads(full['environment_context']['source_background_scope_json'])
    indexes=[{r['source']:r for r in q['views']} for q in (full,original,empty)]
    cameras=json.loads((job.parent/'cameras.json').read_text())
    if ({r['source'] for r in shown['views']}!={str(Path(r['source']).resolve()) for r in cameras['anchors']}
            or any(len(r['renders'])!=1 for r in shown['views'])):
        raise ValueError('Only the prepared captured camera views may be attributed')
    palette=spec['palette'];rows=[];inputs={str(job):digest(job),str(diagnostic):digest(diagnostic),**spec['input_sha256'],**shown['input_sha256']}
    for view in shown['views']:
        src=view['source'];paths=[src,*[index[src]['renders'][0] for index in indexes],view['renders'][0]]
        raw,composed,captured,zero,owner=[cv2.imread(p) for p in paths]
        for p in paths:inputs[p]=digest(p)
        if any(im is None for im in (raw,composed,captured,zero,owner)):raise ValueError('Rendered source images missing')
        mask=np.ones(raw.shape[:2],bool)
        for p in scope[Path(src).name]:mask &=cv2.imread(p,0)>128
        if frozen:mask &=cv2.imread(frozen[Path(src).name]['mask'],0)>128
        mask=cv2.resize(mask.astype('uint8'),(composed.shape[1],composed.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=cv2.erode(mask,np.ones((5,5),'uint8'))>0
        raw=cv2.resize(raw,(composed.shape[1],composed.shape[0]),interpolation=cv2.INTER_AREA)
        base_error=abs(raw.astype(float)-captured).mean(2)/255;error=abs(raw.astype(float)-composed).mean(2)/255
        lost=mask&(base_error<=.10)&(abs(captured.astype(float)-zero).max(2)>8)&(error>base_error+.08)&(error>.18)
        ids=identify(cv2.cvtColor(owner,cv2.COLOR_BGR2RGB),palette)
        counts=[{'owner':key,'lost_pixels':int(np.count_nonzero(lost&(ids==i)))} for i,key in enumerate(palette)]
        rows.append({'source':src,'lost_pixels':int(lost.sum()),'unassigned':int(np.count_nonzero(lost&(ids<0))),
                     'owners':sorted((r for r in counts if r['lost_pixels']),key=lambda r:r['lost_pixels'],reverse=True)})
    result={'approved':False,'rows':rows,'input_sha256':inputs,'scope':LABEL+' Pixel attribution is triage, not proof of geometric or semantic correctness.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Owner attribution input changed')
    out.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('composite');a.add_argument('preservation');a.add_argument('--output',required=True)
    a=sub.add_parser('rank');a.add_argument('job');a.add_argument('diagnostic');a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':print('Native owner diagnostic prepared' if prepare(a.composite,a.preservation,a.output) else 'No failing source views')
    else:print('Native owner attribution views',len(rank(a.job,a.diagnostic,a.output)['rows']))


if __name__=='__main__':main()
