"""Source-colour atlas on a verified planar support, with visible-pixel provenance.

Preserve actual grain and marks instead of repeating a small patch everywhere.
Only independently labelled clear support pixels contribute. Hidden material is
explicitly inferred from an observed repeating patch; this is not an intrinsic
albedo measurement and still needs whole-scene appearance review.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.paths import rel,resolve
from calibrated_scene_reconstruction.scene.session import Session
from .blender import observations,target,verify,model_path
from .texture_patch import surface_points


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(session,patch_source,resolution=2048,output_name='wood_atlas.png'):
    if type(resolution) is not int or not 256<=resolution<=4096:
        raise ValueError('Bounded support texture resolution required')
    if Path(output_name).name!=output_name or Path(output_name).suffix!='.png':
        raise ValueError('A new local PNG atlas revision is required')
    name,spec=target(session);fit=verify(session,name)
    out=model_path(session).parent/output_name
    if out.exists():raise FileExistsError('Preserve existing texture evidence; choose a new revision')
    patch_source=Path(patch_source);patch=json.loads(patch_source.read_text())
    tile_path=resolve(patch['output'])
    if digest(tile_path)!=patch['output_sha256'] or patch['support_mesh_sha256']!=fit['mesh_sha256']:
        raise ValueError('Patch must belong to this reviewed support geometry')
    material=spec['fit']['pose_anchor_material']
    with np.load(resolve(spec['fix']['mesh'])) as mesh:
        selected=mesh['face_material']==list(mesh['material_names']).index(material)
        vertices=mesh['V'][np.unique(mesh['F'][selected])];lo=vertices[:,:2].min(0);hi=vertices[:,:2].max(0)
    size=hi-lo;w=resolution;h=max(128,round(w*size[1]/size[0]))
    # Image rows run opposite to UV V. Pixel centres preserve the actual scale.
    x=lo[0]+(np.arange(w)+.5)/w*size[0]
    y=hi[1]-(np.arange(h)+.5)/h*size[1]
    grid=np.stack(np.meshgrid(x,y),axis=-1)
    a=np.radians(fit['pose']['yaw_deg']);axes=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    points=surface_points(grid,axes,fit['pose']['xy'],spec['box'][1][2],spec['fix'].get('surface_plane'))
    normal=np.asarray(spec['fix'].get('surface_plane',{}).get('normal',[0,0,1.]),float)
    cameras=np.load(session/'parsing/posed_frames/cameras.npz');observed,inputs=observations(session)
    texture=np.zeros((h,w,3),'uint8');best=np.zeros((h,w),'float32');owners=np.full((h,w),-1,'int16')
    records=[];warps=[];validity=[];quality=[]
    for index,row in enumerate(observed):
        source=session/'splat/data/images'/row['name'];rgb=cv2.imread(str(source))
        i=row['index']
        if rgb is None or rgb.shape[:2]!=tuple(cameras['size'][i][::-1]):
            raise ValueError('Texture source resolution differs from its camera')
        camera=(points-row['C'])@row['R'];q=camera@row['K'].T
        uv=q[...,:2]/q[...,2,None]
        # Erode all target/occluder boundaries independently of the model.
        clear=cv2.erode((row['mask']&~row['occluder']).astype('uint8'),np.ones((5,5),'uint8'))
        valid=cv2.remap(clear,uv[...,0].astype('float32'),uv[...,1].astype('float32'),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        K=cameras['K'][i];native=camera@K.T;native=native[...,:2]/native[...,2,None]
        samples=cv2.remap(rgb,native[...,0].astype('float32'),native[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        view=np.asarray(row['C'])-points;distance=np.linalg.norm(view,axis=-1)
        cosine=(view@normal)/np.maximum(distance,1e-9)
        score=(cosine.clip(0,1)**2*K[0,0]*K[1,1]/np.maximum(distance,1e-9)**2).astype('float32')
        valid&=(camera[...,2]>.05)&(cosine>.15)&(samples.max(-1)<250)&(samples.mean(-1)>8)
        take=valid&(score>best)
        texture[take]=samples[take];best[take]=score[take];owners[take]=index
        warps.append(samples);validity.append(valid);quality.append(score)
        records.append({'frame':row['name'],'rgb_sha256':digest(source),'independently_valid_texels':int(valid.sum())})
    # Different viewing angles carry exposure/white-balance changes. Measure
    # gains on actual overlapping source texels, then reject colour outliers
    # before weighting detail. A winning single view can otherwise paste an
    # overlooked leaf, specular highlight, or exposure seam onto the table.
    reference=int(np.argmax([v.sum()for v in validity]));gains=[]
    aligned=[]
    for image,valid in zip(warps,validity):
        overlap=valid&validity[reference]
        if overlap.sum()<1000:
            gains.append([1.,1.,1.]);aligned.append(image.astype('float32'));continue
        gain=np.median((warps[reference][overlap].astype(float)+1)/(image[overlap].astype(float)+1),axis=0).clip(.5,2.)
        gains.append(gain.tolist());aligned.append((image.astype('float32')*gain).astype('float32'))
    stack=np.stack(aligned);valid=np.stack(validity)
    # Unobserved texels receive a temporary zero only to keep all-NaN medians
    # out of the statistics; they remain explicitly unknown in the atlas mask.
    visible_count=valid.sum(0);temporary=valid.copy();temporary[0,visible_count==0]=True
    stack[~temporary]=np.nan
    median=np.nanmedian(stack,axis=0);error=np.nanmean(abs(stack-median),axis=-1)
    deviation=np.nanmedian(error,axis=0)
    consistent=valid&((error<=np.maximum(12.,3*deviation)[None])|(visible_count[None]<3))
    weights=np.stack(quality)*consistent
    weights/=np.maximum(weights.max(0),1e-12)[None]
    weights=np.square(weights)
    total=weights.sum(0);known=total>0
    texture=np.rint(np.nansum(stack*weights[...,None],axis=0)/np.maximum(total,1e-12)[...,None]).clip(0,255).astype('uint8')
    owners=np.argmax(weights,axis=0).astype('int16');owners[~known]=-1
    if known.mean()<.25:raise ValueError('Insufficient clear source coverage for a support atlas')
    observed_path=out.with_name(out.stem+'_observed.png');mask_path=out.with_name(out.stem+'_observed_mask.png')
    cv2.imwrite(str(observed_path),texture);cv2.imwrite(str(mask_path),known.astype('uint8')*255)
    tile=cv2.imread(str(tile_path));period=float(patch['patch_m'])
    tx=((grid[...,0]/period)%1*(tile.shape[1]-1)).astype('float32')
    ty=((-grid[...,1]/period)%1*(tile.shape[0]-1)).astype('float32')
    fallback=cv2.remap(tile,tx,ty,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    # Only inferred missing pixels are filled; the separate observed image and
    # mask remain the immutable source-backed diagnostic.
    completed=texture.copy()
    low=cv2.inpaint(texture,(~known).astype('uint8')*255,12,cv2.INPAINT_TELEA)
    # Continue the observed low-frequency appearance across hidden footprints,
    # retaining only source-patch grain at those inferred pixels.
    grain=fallback.astype('float32')-cv2.GaussianBlur(fallback,(0,0),8).astype('float32')
    fill=(low.astype('float32')+grain).clip(0,255).astype('uint8')
    completed[~known]=fill[~known]
    cv2.imwrite(str(out),completed)
    np.savez_compressed(out.with_name(out.stem+'_owners.npz'),owner=owners,local_bounds=np.stack([lo,hi]))
    record={'version':1,'output':rel(out),'output_sha256':digest(out),'support_mesh_sha256':fit['mesh_sha256'],
            'cameras_sha256':digest(session/'parsing/posed_frames/cameras.npz'),'producer_sha256':digest(__file__),
            'independent_label_inputs':inputs,'source_frames':records,'local_bounds_m':[lo.tolist(),hi.tolist()],
            'observed_fraction':float(known.mean()),'observed_image':rel(observed_path),'observed_image_sha256':digest(observed_path),
            'observed_mask':rel(mask_path),'observed_mask_sha256':digest(mask_path),'patch_source':rel(patch_source),
            'patch_source_sha256':digest(patch_source),'uv_policy':'local XY bounds; image row0 is maximum local Y',
            'photometric_alignment':{'reference_frame':records[reference]['frame'],'bgr_gains':gains,
                'method':'Median per-channel overlap ratios; robust multi-view colour consistency and resolution-weighted blend'},
            'hidden_material':'Inpainted low-frequency continuation plus observed patch grain at scene scale, only in unobserved footprints.',
            'scope':'Observed appearance with capture illumination, not identified intrinsic albedo; final render review required.'}
    out.with_name(out.stem+'_source.json').write_text(json.dumps(record,indent=2)+'\n')
    return {'output':str(out),'observed_fraction':record['observed_fraction']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('patch_source')
    p.add_argument('--resolution',type=int,default=2048);p.add_argument('--output-name',default='wood_atlas.png')
    a=p.parse_args();print(json.dumps(run(Session(a.session),a.patch_source,a.resolution,a.output_name),indent=2))


if __name__=='__main__':main()
