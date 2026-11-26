"""Native appearance gaps measured against the actual retained GS render.

Spatial point density alone cannot certify opaque coverage on a novel ray.
Only camera-facing, self-visible mesh samples with repeated photographic GS
support become transparent; unseen surfaces retain their complete PBR mesh.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def prepare(session,context_manifest,review,regions,output,empty_review=None,opacity_evidence=None):
    from plyfile import PlyData
    out=Path(output).resolve();context_manifest=Path(context_manifest).resolve();review=Path(review).resolve();regions=Path(regions).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('Use a new unfrozen coverage proposal')
    context=json.loads(context_manifest.read_text());render=json.loads(review.read_text());original=json.loads(regions.read_text());shown=render.get('environment_context') or {}
    from calibrated_scene_reconstruction.scene.background.gs_opacity import validate_field,MEASUREMENT_KEYS
    validate_field(render,context,context_manifest)
    if render.get('diagnostic')!='context_only' or render.get('context_background_override') is not None:
        raise ValueError('Coverage must show this actual retained GS field')
    if render.get('static_scene_preserved')is not True:raise ValueError('Static scene preservation required')
    empty=None
    if empty_review:
        empty=json.loads(Path(empty_review).read_text())
        if (empty.get('diagnostic')!='context_empty' or any(empty.get(k)!=render.get(k) for k in ['usd','camera_manifest','environment','environment_context','effective_photometry','context_background_override','background_field_override','background_world_matrix'])
                or empty.get('static_scene_preserved')is not True):raise ValueError('Empty coverage diagnostic must match scene, cameras and photometry')
    elif any(json.loads(shown.get('background_color_linear_json','[0,0,0]'))):raise ValueError('Nonblack coverage background needs a matching empty diagnostic')
    camera=session/'parsing/posed_frames/cameras.npz';source_cloud=context_manifest.parent/'context.ply'
    if context['camera_sha256']!=digest(camera) or original['protected_task_state']!=protected_state(session):raise ValueError('Coverage calibration/task state changed')
    for p,sha in original['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Original regional evidence changed')
    inputs={str(p):digest(p) for p in [context_manifest,review,regions,camera,source_cloud,Path(__file__).resolve()]};inputs.update(original['input_sha256'])
    field=PlyData.read(str(source_cloud))['vertex'].data;views={Path(v['source']).name:v for v in render['views']}
    if not opacity_evidence:raise ValueError('Measured backdrop-invariant coverage required; colour agreement cannot establish opacity')
    opacity_path=Path(opacity_evidence).resolve();opacity=json.loads(opacity_path.read_text())
    if opacity.get('version')!=2 or opacity.get('measurement')!={k:render.get(k) for k in MEASUREMENT_KEYS}:
        raise ValueError('Opacity measurement must use the same scene, actual field, cameras and photometry')
    if (opacity.get('field_rows_sha256')!=hashlib.sha256(field.tobytes()).hexdigest()
            or opacity.get('camera_sha256')!=digest(camera) or opacity.get('protected_task_state')!=protected_state(session)):
        raise ValueError('Opacity evidence must bind the same field, capture and protected state')
    inputs.update(opacity['input_sha256']);inputs[str(opacity_path)]=digest(opacity_path)
    opaque={r['frame']:r for r in opacity['rows']}
    empties={Path(v['source']).name:v for v in empty['views']} if empty else {}
    if empty_review:inputs[str(Path(empty_review).resolve())]=digest(empty_review)
    scope=json.loads(shown['source_background_scope_json']);out.mkdir(parents=True);rows=[]
    for row in original['regions']:
        frame=row['frame']
        if frame not in views:continue
        view=views[frame];raw=cv2.imread(row['source']);image=cv2.imread(view['renders'][0]);h,w=image.shape[:2]
        trusted=cv2.resize(cv2.imread(row['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128
        for p,sha in scope[frame].items():
            if digest(p)!=sha:raise ValueError('Frozen task/background exclusion changed')
            trusted&=cv2.resize(cv2.imread(p,0),(w,h),interpolation=cv2.INTER_NEAREST)>128;inputs[p]=sha
        raw=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
        zero=cv2.imread(empties[frame]['renders'][0]) if empty else np.zeros_like(image)
        if empty:
            for p in empties[frame]['renders']:inputs[str(Path(p).resolve())]=digest(p)
        trusted=sharp_mask(raw,image,trusted)&(abs(image.astype(float)-zero).max(2)>16)
        if frame not in opaque:trusted[:]=False
        else:
            record=opaque[frame]
            if digest(record['mask'])!=record['mask_sha256']:raise ValueError('Opacity mask changed')
            trusted &= cv2.resize(cv2.imread(record['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128
        width=min(640,w);height=round(h*width/w);mask=out/(row['entity']+'_'+Path(frame).stem+'.png')
        cv2.imwrite(str(mask),cv2.resize(trusted.astype('uint8')*255,(width,height),interpolation=cv2.INTER_NEAREST))
        for p in [*view['renders'],str(mask)]:inputs[str(Path(p).resolve())]=digest(p)
        rows.append({'entity':row['entity'],'frame':frame,'mask':str(mask),'mask_sha256':digest(mask),'size':[width,height],'pixels':int(trusted.sum())})
    result={'version':2,'opacity_evidence':str(opacity_path),'rows':rows,'field_rows_sha256':hashlib.sha256(field.tobytes()).hexdigest(),
            'render_usd':render['usd'],'environment_asset_sha256':(render.get('environment') or {}).get('asset_sha256'),
            'parameters_sha256':context['parameters_sha256'],'input_sha256':inputs,'protected_task_state':protected_state(session),
            'scope':'Appearance-coverage proposal from the actual retained field. Native support and physical geometry stay complete; source/free-view acceptance still required.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Coverage input changed')
    (out/'coverage.json').write_text(json.dumps(result,indent=2)+'\n');return result


def bake(session,job,entity,declaration,field,points,evidence,out):
    from scipy.spatial import cKDTree
    from calibrated_scene_reconstruction.scene.background.crop_gaps import bake as bake_atlas
    from calibrated_scene_reconstruction.scene.background.mesh_textures import depth_buffer
    evidence=Path(evidence).resolve();record=json.loads(evidence.read_text())
    if record.get('version')!=2 or not record.get('opacity_evidence'):raise ValueError('Legacy colour-only cutouts are unsafe; measure backdrop-invariant coverage')
    if (record['protected_task_state']!=protected_state(session) or record['parameters_sha256']!=parameter_digest(session.config('environment.json')['parameters'])
            or record['field_rows_sha256']!=hashlib.sha256(field.tobytes()).hexdigest()):raise ValueError('Coverage belongs to another geometry or GS selection; rerender the retained field')
    inputs={str(evidence):digest(evidence),**record['input_sha256'],str(Path(__file__).resolve()):digest(__file__)}
    from calibrated_scene_reconstruction.scene.background import mesh_textures
    inputs[str(Path(mesh_textures.__file__).resolve())]=digest(mesh_textures.__file__)
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Projected GS coverage evidence changed')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']));geometry=np.load(job['geometry'])
    vertices=[];faces=[];offset=0
    for i,row in enumerate(job['parts']):
        if row['entity']!=entity:continue
        vertices.append(geometry[f'v{i}']);faces.append(geometry[f'f{i}']+offset);offset+=len(vertices[-1])
    vertices=np.concatenate(vertices);faces=np.concatenate(faces);views=[]
    depth_band=declaration.get('depth_band_m')
    if depth_band is not None:
        from calibrated_scene_reconstruction.scene.background import capture_depth,surface_textures
        # Validate even when there are no eligible source rays.
        capture_depth.compatible(np.array([1.]),np.array([[0,0]]),np.array([[1.]]),depth_band)
        for module in [capture_depth,surface_textures]:inputs[str(Path(module.__file__).resolve())]=digest(module.__file__)
        opacity=1/(1+np.exp(-np.clip(field['opacity'].astype(float),-80,80)))
    ray_mode=declaration.get('ray_mode','self_visible')
    if ray_mode not in ('self_visible','opaque_layers'):raise ValueError('Known measured coverage ray mode required')
    closed_shell=declaration.get('closed_shell',True)
    if type(closed_shell)is not bool:raise ValueError('Closed-shell coverage declaration must be boolean')
    if ray_mode=='opaque_layers' or closed_shell:
        from calibrated_scene_reconstruction.assets import topology
        closed_oriented_surface=topology.closed_oriented_surface
        inputs[str(Path(topology.__file__).resolve())]=digest(topology.__file__)
        for i,row in enumerate(job['parts']):
            if row['entity']==entity and not closed_oriented_surface(geometry[f'v{i}'],np.full(len(geometry[f'f{i}']),3),geometry[f'f{i}'].ravel()):
                raise ValueError('Closed-shell coverage requires closed oriented native mesh parts')
    coverage_entity=declaration.get('coverage_entity',entity)
    if not isinstance(coverage_entity,str) or not coverage_entity:raise ValueError('Explicit capture coverage owner required')
    frames=declaration.get('frames')
    if frames is not None and (not isinstance(frames,list) or len(set(frames))!=len(frames) or len(frames)<2 or not set(frames)<=set(names)):
        raise ValueError('At least two distinct registered source observation frames required')
    for row in record['rows']:
        if row['entity']!=coverage_entity or row['pixels']<32 or frames is not None and row['frame'] not in frames:continue
        i=names.index(row['frame']);w,h=row['size'];K=C['K'][i].copy();K[0]*=w/C['size'][i,0];K[1]*=h/C['size'][i,1]
        pc=(vertices-C['centre'][i])@C['R_world_cam'][i];front=pc[:,2]>.1
        if not front.any():continue
        uv=pc[front]@K.T;uv=uv[:,:2]/uv[:,2,None];lo=np.maximum(np.floor(uv.min(0)),[0,0]).astype(int);hi=np.minimum(np.ceil(uv.max(0)),[w,h]).astype(int)
        mask=cv2.imread(row['mask'],0)
        if np.any(hi<=lo) or np.count_nonzero(mask[lo[1]:hi[1],lo[0]:hi[0]]>128)<32:continue
        depth=depth_buffer(vertices,faces,K,C['R_world_cam'][i],C['centre'][i],(h,w))
        captured_depth=surface_textures.relief_depth(points,opacity,K,C['R_world_cam'][i],C['centre'][i],mask.shape,cell=2) if depth_band is not None else None
        views.append((C['centre'][i],C['R_world_cam'][i],K,mask,depth,captured_depth))
    tree=cKDTree(points);radius=declaration.get('radius_m',.08)
    if not .01<=radius<=.2:raise ValueError('Bounded native/GS compatibility radius required')
    def inside(samples,normal):
        output=[]
        for start in range(0,len(samples),32768):
            P=samples[start:start+32768];votes=np.zeros(len(P),np.uint8);last=np.full((len(P),3),np.nan)
            for centre,R,K,mask,depth,captured_depth in views:
                pc=(P-centre)@R;uv=pc@K.T;uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9);h,w=mask.shape
                visible=(pc[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&((centre-P)@normal>1e-5)
                ids=np.flatnonzero(visible);xy=uv[ids].astype(int)
                delta=pc[ids,2]-depth[xy[:,1],xy[:,0]]
                visible_layer=abs(delta)<=.02 if ray_mode=='self_visible' else delta>=-.02
                ok=(mask[xy[:,1],xy[:,0]]>128)&visible_layer
                if captured_depth is not None:ok &=capture_depth.compatible(pc[ids,2],xy,captured_depth,depth_band)
                before=P[ids]-last[ids];after=P[ids]-centre;cos=np.einsum('ij,ij->i',before,after)/np.maximum(1e-12,np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
                independent=(votes[ids]==0)|((np.linalg.norm(last[ids]-centre,axis=1)>.05)&(np.degrees(np.arccos(np.clip(cos,-1,1)))>=2))
                take=ids[ok&independent];votes[take]+=1;last[take]=centre
            near=tree.query(P,workers=1)[0]<=radius;output.append(~((votes>=2)&near))
        return np.concatenate(output)
    result,inputs=bake_atlas(job,entity,declaration,out,inside,{'capture_coverage':{'field_rows_sha256':record['field_rows_sha256'],'views':len(views),'radius_m':radius,'coverage_entity':coverage_entity,'ray_mode':ray_mode,'depth_band_m':depth_band,
             'scope':'Two separated source rays with actual opaque GS contribution; camera-facing native surfaces only. Layered mode requires closed one-sided solids; unseen outward rear faces stay native.'}},inputs,face_normals=True)
    if ray_mode=='opaque_layers' or closed_shell:
        for payload in result['meshes'].values():payload['one_sided_closed']=True
    backing=declaration.get('captured_backing_parts',{})
    if not isinstance(backing,dict):raise ValueError('Explicit captured part-to-proof declarations required')
    if backing:
        from pxr import Usd,UsdShade
        from calibrated_scene_reconstruction.scene.background.capture_backing import validate as validate_backing
        stage=Usd.Stage.Open(json.loads(Path(job['output']).read_text())['usd'])
        for part,proof in backing.items():
            found=[(name,payload) for name,payload in result['meshes'].items() if part in name.split('/')]
            if not found:raise ValueError('Captured backing part absent from current audited geometry: '+part)
            for name,payload in found:
                material=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(name)).ComputeBoundMaterial()[0]
                inputs.update(validate_backing(session,proof,material,stage.GetPrimAtPath(name)))
                payload.update(compositing='solid_backing',captured_backing_proof=str(Path(proof).resolve()))
    return result,inputs


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('context_manifest');p.add_argument('review');p.add_argument('regions');p.add_argument('--output',required=True);p.add_argument('--empty-review');p.add_argument('--opacity-evidence',required=True)
    a=p.parse_args();q=prepare(Session(a.session),a.context_manifest,a.review,a.regions,a.output,a.empty_review,a.opacity_evidence);print('Projected coverage candidate',len(q['rows']),'observations')


if __name__=='__main__':main()
