"""Exclude actual frozen task/support/robot silhouettes from background scoring.

Only named frozen geometry in the audited export is projected, including its
declared GS protection margin. Environment meshes, shadows and image-error
regions cannot define these exclusions.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def project(vertices, faces, K, R, centre, shape):
    from calibrated_scene_reconstruction.core.geometry.raster import triangle_mask
    camera=(np.asarray(vertices)-centre)@R;faces=np.asarray(faces,int);tri=camera[faces]
    front=(tri[:,:,2]>.02).all(1);valid=faces[front]
    uv=camera@K.T;uv=uv[:,:2]/np.maximum(camera[:,2,None],.02)
    uv=np.clip(uv,[-shape[1]*4,-shape[0]*4],[shape[1]*5,shape[0]*5])
    projected=uv[valid]
    valid=valid[(projected[:,:,0].max(1)>=0)&(projected[:,:,0].min(1)<shape[1])
                &(projected[:,:,1].max(1)>=0)&(projected[:,:,1].min(1)<shape[0])]
    result=triangle_mask(uv,valid,shape)
    # Clip triangles crossing the near plane instead of projecting negative Z.
    for points in tri[(tri[:,:,2]>.02).any(1)&~front]:
        output=[]
        for a,b in zip(points,np.roll(points,-1,axis=0)):
            if a[2]>=.02:output.append(a)
            if (a[2]>=.02)!=(b[2]>=.02):output.append(a+(b-a)*(.02-a[2])/(b[2]-a[2]))
        if len(output)<3:continue
        p=np.asarray(output)@K.T;p=p[:,:2]/p[:,2,None]
        p=np.clip(p,[-shape[1]*4,-shape[0]*4],[shape[1]*5,shape[0]*5])
        cv2.fillConvexPoly(result.view('uint8'),np.rint(p*16).astype('int32'),1,shift=4)
    return result


def meshes(stage, support, objects):
    from pxr import Usd,UsdGeom
    prefixes=['/World/Room/'+support,*['/World/Objects/'+n for n in objects],'/World/Robot']
    cache=UsdGeom.XformCache();result=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
        path=str(prim.GetPath())
        frozen=any(path==p or path.startswith(p+'/') for p in prefixes) or path.startswith('/World/Room/'+support+'_')
        if not prim.IsA(UsdGeom.Mesh) or not frozen:continue
        mesh=UsdGeom.Mesh(prim);v=np.asarray(mesh.GetPointsAttr().Get(),float)
        world=(np.c_[v,np.ones(len(v))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:,:3]
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int);offset=0;faces=[]
        for n in mesh.GetFaceVertexCountsAttr().Get():
            p=indices[offset:offset+n];faces.extend([[p[0],p[j],p[j+1]] for j in range(1,n-1)]);offset+=n
        if faces:result.append((path,world,np.asarray(faces,int)))
    if not result:raise ValueError('No actual frozen task/support/robot meshes in this export')
    return result


def prepare(session, usd, cameras, output, variant='twin'):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    out=Path(output).resolve();usd=Path(usd).resolve();cameras=Path(cameras).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()):raise ValueError('New session-local background scoring scope required')
    audit,_=current(session,variant)
    if audit['usd']!=str(usd) or audit['input_sha256'].get(str(usd))!=digest(usd):raise ValueError('Use actual current audited export')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']));manifest=json.loads(cameras.read_text())
    graph=session.scene_graph;parts=meshes(Usd.Stage.Open(str(usd)),graph['task']['support'],graph.get('dynamic_objects',{}))
    inputs={str(p.resolve()):digest(p) for p in [usd,cameras,camera,Path(__file__).resolve()]};out.mkdir(parents=True);rows=[]
    margins={'task_and_support':0.,'robot':0.}
    config=session.config('environment.json',{})
    if config.get('splat_context'):
        proof=Path(config['assets'][config['splat_context']['evidence']]).resolve();data=json.loads(proof.read_text())
        protection=data.get('field_trial') or data.get('capture_selection') or data.get('closure_support') or {}
        margins={'task_and_support':float((protection.get('protected_overlap') or protection.get('protected') or {}).get('margin_m',.015)),
                 'robot':float((protection.get('frozen_robot') or protection.get('robot') or {}).get('margin_m',.05))}
        if not 0<=margins['task_and_support']<=.02 or not 0<=margins['robot']<=.1:raise ValueError('Bounded frozen Gaussian protection margins required')
        inputs[str(proof)]=digest(proof)
    for anchor in manifest['anchors']:
        source=Path(anchor['source']).resolve();i=names.index(source.name);h,w=map(int,C['size'][i][::-1]);view=anchor['views'][0]
        if (not np.allclose(anchor['K'],C['K'][i],atol=1e-5,rtol=0)
                or not np.allclose(view['centre'],C['centre'][i],atol=1e-5,rtol=0)
                or not np.allclose(view['R_world_cam'],C['R_world_cam'][i],atol=1e-5,rtol=0)
                or source!=(session/'splat/data/images'/source.name).resolve()):
            raise ValueError('Background scope requires unchanged registered source cameras')
        excluded=np.zeros((h,w),bool)
        for name,v,f in parts:
            mask=project(v,f,C['K'][i],C['R_world_cam'][i],C['centre'][i],(h,w))
            if mask.any():
                # Match the existing metric GS foreground exclusion instead of
                # demanding preservation of deliberately removed boundary GS.
                margin=margins['robot' if name.startswith('/World/Robot/') else 'task_and_support']
                depth=((v-C['centre'][i])@C['R_world_cam'][i])[:,2];positive=depth[depth>.02]
                radius=int(np.ceil(max(C['K'][i,0,0],C['K'][i,1,1])*margin/max(.02,float(np.median(positive))))) if len(positive) else 0
                radius=min(radius,round(min(h,w)*.05))
                if radius:mask=cv2.dilate(mask.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))>0
            excluded|=mask
        excluded=cv2.dilate(excluded.astype('uint8'),np.ones((3,3),'uint8'))>0
        path=out/(source.stem+'.png');cv2.imwrite(str(path),(~excluded).astype('uint8')*255)
        inputs[str(source)]=digest(source);inputs[str(path)]=digest(path)
        rows.append({'source':str(source),'mask':str(path),'mask_sha256':digest(path),'excluded_pixels':int(excluded.sum()),'pixels':h*w})
    result={'version':1,'usd':str(usd),'camera_manifest':str(cameras),'rows':rows,'frozen_meshes':[p for p,_,_ in parts],
            'protected_task_state':protected_state(session),'projection_uncertainty_m':margins,'input_sha256':inputs,'scope':__doc__.strip()}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Frozen scoring geometry changed')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def validate(path, review):
    path=Path(path).resolve();q=json.loads(path.read_text())
    if (q.get('version')!=1 or q['usd']!=review['usd'] or q['camera_manifest']!=review['camera_manifest']
            or q['input_sha256'].get(q['usd'])!=review['input_sha256'].get(q['usd'])
            or q['input_sha256'].get(q['camera_manifest'])!=review['input_sha256'].get(q['camera_manifest'])
            or not q.get('frozen_meshes') or any(p.startswith('/World/Environment') for p in q['frozen_meshes'])):
        raise ValueError('Frozen exclusion must project this exact audited scene and cameras')
    for p,sha in q['input_sha256'].items():
        if digest(p)!=sha:raise ValueError('Frozen exclusion changed: '+p)
    if q['input_sha256'].get(str(Path(__file__).resolve()))!=digest(__file__):raise ValueError('Frozen exclusion implementation changed')
    return {Path(r['source']).name:r for r in q['rows']},{str(path):digest(path),**q['input_sha256']}


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['session','usd','cameras']:p.add_argument(n)
    p.add_argument('--output',required=True);p.add_argument('--variant',default='twin');a=p.parse_args();r=prepare(Session(a.session),a.usd,a.cameras,a.output,a.variant)
    print('Frozen silhouettes excluded in',len(r['rows']),'source views')


if __name__=='__main__':main()
