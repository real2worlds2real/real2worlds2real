"""Bake masked capture appearance onto editable native background meshes.

Geometry comes from an audited agent Blender export, not a Gaussian surface or
task CAD. Self-occlusion and capture masks constrain every sampled texel. Hidden
surfaces receive an explicit plain hypothesis. Radiance is source-lit, not a
recovered relightable material, and requires actual novel-view review.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.background.surface_textures import linear,srgb
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
from calibrated_scene_reconstruction.scene.session import Session


def triangle_pixels(points,shape):
    """Yield pixel centres and barycentric coordinates for a finite triangle."""
    p=np.asarray(points,float);h,w=shape
    if p.shape!=(3,2) or not np.isfinite(p).all():return
    lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int),[w-1,h-1])
    if np.any(hi<lo):return
    x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
    a,b,c=p;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
    if abs(den)<1e-10:return
    u=((b[1]-c[1])*(x+.5-c[0])+(c[0]-b[0])*(y+.5-c[1]))/den
    v=((c[1]-a[1])*(x+.5-c[0])+(a[0]-c[0])*(y+.5-c[1]))/den
    weights=np.stack([u,v,1-u-v],axis=-1);valid=(weights>=-1e-7).all(-1)
    if valid.any():yield y[valid],x[valid],weights[valid]


def depth_buffer(vertices,faces,K,R,centre,shape):
    """Perspective-correct nearest native surface depth, including hidden parts."""
    pc=(vertices-centre)@R;q=pc@K.T;uv=q[:,:2]/np.maximum(pc[:,2,None],1e-10)
    depth=np.full(shape,np.inf,np.float32)
    for tri in faces:
        z=pc[tri,2]
        if (z<=.05).any():continue
        for yy,xx,bary in triangle_pixels(uv[tri],shape):
            value=1/(bary/z).sum(1)
            depth[yy,xx]=np.minimum(depth[yy,xx],value)
    return depth


def sample_pixels(image,pixels,interpolation):
    """OpenCV limits each remap dimension to SHRT_MAX; an atlas can exceed it."""
    chunks=[]
    for start in range(0,len(pixels),16384):
        p=pixels[start:start+16384].astype('float32')
        values=cv2.remap(image,p[:,0,None],p[:,1,None],interpolation,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        chunks.append(values[:,0])
    return np.concatenate(chunks)


def uv_surface(vertices,faces,uv,size):
    w,h=size;position=np.zeros((h,w,3),np.float32);normal=np.zeros_like(position);owner=np.full((h,w),-1,np.int32)
    pixels=uv*np.array([w,-h])+[0,h]
    for k,tri in enumerate(faces):
        a,b,c=vertices[tri];n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-12:continue
        for yy,xx,bary in triangle_pixels(pixels[tri],(h,w)):
            position[yy,xx]=bary@vertices[tri];normal[yy,xx]=n/length;owner[yy,xx]=k
    return position,normal,owner


def export_geometry(path,vertices,faces,uv,part_ids):
    from pxr import Sdf,Usd,UsdGeom,Vt
    import trimesh
    stage=Usd.Stage.CreateNew(str(path));root=UsdGeom.Xform.Define(stage,'/CapturedSurface');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals
    for k in np.unique(part_ids):
        selected=faces[part_ids==k];ids,remap=np.unique(selected,return_inverse=True)
        mesh=UsdGeom.Mesh.Define(stage,f'/CapturedSurface/part_{k}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices[ids].astype('float32')))
        mesh.CreateFaceVertexCountsAttr([3]*len(selected));mesh.CreateFaceVertexIndicesAttr(remap.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals[ids].astype('float32')));mesh.SetNormalsInterpolation('vertex')
        st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
        st.Set(Vt.Vec2fArray.FromNumpy(uv[ids].astype('float32')))
    stage.GetRootLayer().Save()


def blend_views(samples,score,fallback,power=12.,max_gain=1.5):
    """Bound exposure differences on shared visible surfaces, then feather views."""
    if not samples or not any(len(s['ids'])>=64 for s in samples) or not 1<=power<=32 or not 1<=max_gain<=2:raise ValueError('Observed samples and bounded view blending required')
    count=len(score);anchor=max(range(len(samples)),key=lambda k:len(samples[k]['ids']))
    reference=np.zeros((count,3),np.float32);known=np.zeros(count,bool);row=samples[anchor]
    reference[row['ids']]=row['rgb'];known[row['ids']]=True
    accum=np.zeros((count,3),np.float64);weights=np.zeros(count);gains=[]
    for sample in samples:
        ids=sample['ids'];rgb=sample['rgb'];overlap=known[ids]&(rgb.min(1)>.015)&(reference[ids].min(1)>.015)
        gain=np.ones(3)
        if overlap.sum()>=64:
            ratio=np.median(np.log(reference[ids[overlap]]+.001)-np.log(rgb[overlap]+.001),axis=0)
            gain=np.exp(np.clip(ratio,-np.log(max_gain),np.log(max_gain)))
        w=np.clip(sample['quality']/np.maximum(score[ids],1e-12),0,1)**power
        accum[ids]+=rgb*gain*w[:,None];weights[ids]+=w;gains.append(gain.tolist())
    colour=np.array(fallback,copy=True);valid=weights>0;colour[valid]=accum[valid]/weights[valid,None]
    return colour,{'reference_view_index':anchor,'linear_rgb_gains':gains,'blend_power':power,'max_gain':max_gain,
                   'scope':'Robust overlap display harmonization, not recovered illumination or reflectance.'}


def run(session,config,geometry_job,output):
    import xatlas
    from scipy.ndimage import distance_transform_edt
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    cfg=json.loads(Path(config).read_text());job=json.loads(Path(geometry_job).read_text());entity=cfg['entity']
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    verify(session)
    pointer=session/'scene/environment/twin/current.json'
    if job.get('native_manifest_sha256')!=digest(pointer) or job.get('geometry_sha256')!=digest(job['geometry']):
        raise ValueError('Bake only geometry bound to the current native build')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Mesh texture evidence belongs in this session')
    out.mkdir(parents=True,exist_ok=False)
    definition=next(e for e in session.config('environment.json')['parameters']['entities'] if e['id']==entity)
    if entity in {session.scene_graph['task']['support'],*session.scene_graph.get('dynamic_objects',{})}:raise ValueError('Background textures cannot replace protected task/support')
    z=np.load(job['geometry']);parts=[];V=[];F=[];part_ids=[];offset=0
    for i,row in enumerate(job['parts']):
        if row['entity']!=entity or row['external']:continue
        v,f=z[f'v{i}'],z[f'f{i}'];V.append(v);F.append(f+offset);offset+=len(v);part_ids.extend([len(parts)]*len(f));parts.append(row)
    if not parts:raise ValueError('Selected background entity has no exported native geometry')
    V=np.concatenate(V).astype('float32');F=np.concatenate(F).astype('uint32');part_ids=np.asarray(part_ids)
    resolution=cfg.get('resolution',2048);gain=float(cfg.get('emission_gain',.001));fallback=np.asarray(cfg.get('fallback_srgb',[.7,.7,.7]),float)
    if type(resolution)is not int or not 128<=resolution<=4096 or not 0<gain<=10 or fallback.shape!=(3,) or not np.isfinite(fallback).all() or np.any((fallback<0)|(fallback>1)):
        raise ValueError('Bounded atlas size, radiance gain and plain fallback required')
    camera=session/'parsing/posed_frames/cameras.npz';observations=Path(cfg['observations']);C=np.load(camera);names=list(map(str,C['names']))
    inputs={str(p.resolve()):digest(p) for p in [Path(config),Path(geometry_job),Path(job['geometry']),pointer,camera,observations,Path(__file__)]}
    atlas=xatlas.Atlas();atlas.add_mesh(V,F);pack=xatlas.PackOptions();pack.resolution=resolution;pack.padding=6;atlas.generate(pack_options=pack)
    mapping,triangles,uv=atlas[0];vertices=V[mapping]
    # Atlas vertex splits are allowed, but every output triangle must be exactly
    # one original triangle. This preserves native shape and component identity.
    lookup={tuple(sorted(map(int,tri))):k for k,tri in enumerate(F)}
    face_rows=np.array([lookup[tuple(sorted(map(int,tri)))] for tri in mapping[triangles]])
    if sorted(face_rows.tolist())!=list(range(len(F))):raise ValueError('UV unwrapping changed native triangles')
    part_ids=part_ids[face_rows]
    size=(atlas.width,atlas.height)
    if max(size)>8192:raise ValueError('UV atlas grew beyond bounded memory limit')
    P,N,owner=uv_surface(vertices,triangles,uv,size);occupied=owner>=0;yy,xx=np.nonzero(occupied);Q=P[yy,xx];normals=N[yy,xx]
    colour=np.broadcast_to(linear(fallback),(len(Q),3)).copy();score=np.zeros(len(Q));chosen=np.full(len(Q),-1,np.int16)
    import fnmatch
    overrides=cfg.get('part_fallback_srgb',{})
    texel_parts=part_ids[owner[yy,xx]]
    for pattern,rgb in overrides.items():
        rgb=np.asarray(rgb,float)
        if rgb.shape!=(3,) or not np.isfinite(rgb).all() or np.any((rgb<0)|(rgb>1)):raise ValueError('Bounded per-part fallback colors required')
        matching=[i for i,part in enumerate(parts) if fnmatch.fnmatch(part['part'],pattern)]
        if not matching:raise ValueError('Fallback pattern matched no original native part: '+pattern)
        colour[np.isin(texel_parts,matching)]=linear(rgb)
    records=list({r['frame']:r for r in json.loads(observations.read_text())['records'] if r['entity']==entity and r.get('mask')}.values())
    if len(records)<2:raise ValueError('Mesh appearance requires at least two distinct observed views')
    validity=ValidPixels(session);votes=np.zeros(len(Q),np.int16);samples=[]
    for k,row in enumerate(records):
        for path,sha in [(row['source'],row['source_sha256']),(row['mask'],row['mask_sha256'])]:
            if digest(path)!=sha:raise ValueError('Mesh observation changed: '+path)
            inputs[str(Path(path).resolve())]=sha
        rgb=np.asarray(Image.open(row['source']).convert('RGB'))/255.;mask=np.asarray(Image.open(row['mask']))>128;h,w=mask.shape
        i=names.index(row['frame'])
        if list(map(int,C['size'][i]))!=[w,h] or rgb.shape[:2]!=(h,w):raise ValueError('Capture/mask/calibration size mismatch')
        mask &= validity.get(row['frame'],(h,w));mask=cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8'))
        depth=depth_buffer(vertices,triangles,C['K'][i],C['R_world_cam'][i],C['centre'][i],(h,w))
        pc=(Q-C['centre'][i])@C['R_world_cam'][i];q=pc@C['K'][i].T;pixels=q[:,:2]/np.maximum(pc[:,2,None],1e-9);maps=pixels.astype('float32')
        camera_ray=C['centre'][i]-Q;distance=np.linalg.norm(camera_ray,axis=1);angle=(normals*camera_ray).sum(1)/np.maximum(distance,1e-9)
        surface_mask=sample_pixels(mask.astype('float32'),maps,cv2.INTER_LINEAR)>=1.-1e-6
        front=sample_pixels(depth,maps,cv2.INTER_NEAREST)
        visible=surface_mask&(pc[:,2]>.05)&(angle>.05)&np.isfinite(front)&(abs(pc[:,2]-front)<.004+pc[:,2]*.002)
        quality=np.maximum(angle,0)**2*C['K'][i,0,0]/np.maximum(distance,.05)
        take=visible&(quality>score);sample=sample_pixels(rgb.astype('float32'),maps,cv2.INTER_LINEAR)
        score[take]=quality[take];chosen[take]=k;votes+=visible
        ids=np.flatnonzero(visible);samples.append({'ids':ids,'rgb':linear(sample[ids]).astype('float32'),'quality':quality[ids]})
        print('[mesh.textures]',entity,row['frame'],int(visible.sum()),'visible texels',flush=True)
    colour,blending=blend_views(samples,score,colour,float(cfg.get('blend_power',12.)),float(cfg.get('max_view_gain',1.5)))
    image=np.zeros((*owner.shape,3),np.float32);image[yy,xx]=colour
    # Pad only atlas gutters; never fill a hidden surface with another surface's
    # pixels. Hidden occupied texels retain the declared plain fallback.
    dist,nearest=distance_transform_edt(~occupied,return_indices=True);gutter=(~occupied)&(dist<=6)
    image[gutter]=image[nearest[0][gutter],nearest[1][gutter]]
    texture=out/'appearance.png';Image.fromarray(np.round(srgb(image)*255).astype('uint8')).save(texture)
    radiance=out/'radiance.hdr'
    if not cv2.imwrite(str(radiance),(image[...,::-1]*gain).astype('float32')):raise ValueError('Cannot write native mesh radiance')
    mesh=out/'geometry.usdc';export_geometry(mesh,vertices,triangles,uv,part_ids)
    observed=np.zeros(owner.shape,'uint8');observed[yy,xx]=(chosen>=0)*255;Image.fromarray(observed).save(out/'observed.png')
    np.savez_compressed(out/'texel_evidence.npz',y=yy,x=xx,view=chosen,votes=votes)
    inputs.update({str(Path(path).resolve()):sha for path,sha in validity.inputs.items()})
    for path,sha in inputs.items():
        if digest(path)!=sha:raise ValueError('Mesh baking input changed: '+path)
    proof={'version':1,'entity':entity,'entity_definition':definition,'entity_parameters_sha256':parameter_digest(definition),
           'mesh':str(mesh),'mesh_sha256':digest(mesh),'radiance':str(radiance),'radiance_sha256':digest(radiance),
           'texture':str(texture),'texture_sha256':digest(texture),'source_parts':parts,'triangles':len(triangles),
           'size_px':list(size),'observed_fraction':float((chosen>=0).mean()),'multiview_fraction':float((votes>=2).mean()),
           'fallback_srgb':fallback.tolist(),'part_fallback_srgb':overrides,'view_blending':blending,
           'emission_gain':gain,'frames':[r['frame'] for r in records],
           'input_sha256':inputs,'protected_task_state':protected_state(session),'photorealism_approved':False,
           'scope':'Native agent mesh, exact geometry retained by UV split. Masked self-visible capture radiance; hidden plain fill. Not relightable PBR or automatic realism approval.'}
    (out/'manifest.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('config');p.add_argument('geometry_job');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(Session(a.session),a.config,a.geometry_job,a.output),indent=2))


if __name__=='__main__':main()
