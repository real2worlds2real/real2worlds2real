"""Rectify an unoccluded captured support patch at its reconstructed metric scale.

No task truth or model-rendered labels select the patch. Independent full-frame
support/occluder labels must cover every sample; inference is limited to tiling
that observed material on unobserved areas of the same support.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.scene.session import Session
from .blender import observations, target, verify, model_path


def surface_points(local_xy, axes, centre_xy, top_z, surface_plane=None):
    """Place sampling pixels on the verified surface, including its tilt.

    A support box maximum is not its work surface except for a level top.
    Using it for tilted tables displaces source texture and occlusion checks.
    """
    xy=np.asarray(local_xy,float)@np.asarray(axes,float).T+np.asarray(centre_xy,float)
    if xy.shape[-1:]!=(2,) or not np.isfinite(xy).all():
        raise ValueError('Finite local material sample coordinates required')
    if surface_plane is None:
        if not np.isfinite(top_z):raise ValueError('Finite support height required')
        z=np.full(xy.shape[:-1],float(top_z))
    else:
        from .observed_plane import height
        height(surface_plane,xy.reshape(-1,2)[0])  # Validate the actual reviewed plane.
        normal=np.asarray(surface_plane['normal'],float)
        z=(float(surface_plane['offset'])-xy@normal[:2])/normal[2]
    return np.concatenate([xy,z[...,None]],axis=-1)


def run(session, patch_m=.18, resolution=512, output_name='wood.png', *,
        source_frame=None, selection_basis=None):
    if not .02 <= patch_m <= .5 or not 128 <= resolution <= 2048:
        raise ValueError('Use a finite material patch of .02.. .5m and 128..2048 pixels')
    name, spec=target(session);fit=verify(session,name)
    observed, observation_inputs=observations(session)
    if source_frame is not None:
        if (not isinstance(source_frame,str) or Path(source_frame).name!=source_frame
                or not isinstance(selection_basis,str) or not selection_basis.strip()):
            raise ValueError('A reviewed source frame requires a local image name and selection basis')
        observed=[row for row in observed if row['name']==source_frame]
        if not observed:
            raise ValueError('Reviewed material frame must have independent support observations')
    elif selection_basis is not None:
        raise ValueError('A selection basis requires an explicit reviewed source frame')
    # Agent models need not use particular parameter names. The verified local
    # mesh supplies a search envelope; actual masks still validate every sample.
    with np.load(resolve(spec['fix']['mesh'])) as mesh:
        length,width=np.ptp(mesh['V'][:,:2],axis=0)
    yaw=np.radians(fit['pose']['yaw_deg']);axes=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=np.asarray(fit['pose']['xy']);z=float(spec['box'][1][2])
    surface_plane=spec['fix'].get('surface_plane')
    plane=np.stack(np.meshgrid(np.linspace(-patch_m/2,patch_m/2,64),np.linspace(-patch_m/2,patch_m/2,64)),axis=-1)
    if Path(output_name).name!=output_name or Path(output_name).suffix.lower()!='.png':
        raise ValueError('Choose a local PNG filename for this source texture revision')
    directory=model_path(session).parent;destination=directory/output_name
    if destination.exists():raise ValueError('Preserve the existing source texture; choose a new agent asset revision')
    candidates=[]
    for local in np.stack(np.meshgrid(np.linspace(-length*.35,length*.35,7),np.linspace(-width*.30,width*.30,5)),axis=-1).reshape(-1,2):
        points=surface_points(plane+local,axes,xy,z,surface_plane)
        for row in observed:
            camera=(points-row['C'])@row['R'];q=camera@row['K'].T;uv=q[...,:2]/q[...,2,None];ij=np.rint(uv).astype(int);h,w=row['mask'].shape
            inside=(camera[...,2]>.05)&(ij[...,0]>=0)&(ij[...,0]<w)&(ij[...,1]>=0)&(ij[...,1]<h)
            if not inside.all():continue
            if not (row['mask'][ij[...,1],ij[...,0]]&~row['occluder'][ij[...,1],ij[...,0]]).all():continue
            area=float(abs(np.cross(uv[0,-1]-uv[0,0],uv[-1,0]-uv[0,0])))
            candidates.append((area,local,row))
    if not candidates:raise ValueError('No independently labelled clear support-material patch; do not paint over occluders')
    from .texture_quality import repeatability
    cameras=np.load(session/'parsing/posed_frames/cameras.npz');rejected=[];images={};selected=None
    grid=np.stack(np.meshgrid(np.linspace(-patch_m/2,patch_m/2,resolution),np.linspace(-patch_m/2,patch_m/2,resolution)),axis=-1)
    for area,local,row in sorted(candidates,key=lambda item:item[0],reverse=True):
        index=row['index'];K=cameras['K'][index];C=cameras['centre'][index];R=cameras['R_world_cam'][index]
        points=surface_points(grid+local,axes,xy,z,surface_plane)
        camera=(points-C)@R;q=camera@K.T;uv=q[...,:2]/q[...,2,None]
        maskq=camera@row['K'].T;ij=np.rint(maskq[...,:2]/maskq[...,2,None]).astype(int);h,w=row['mask'].shape
        if not ((ij[...,0]>=0)&(ij[...,0]<w)&(ij[...,1]>=0)&(ij[...,1]<h)).all():continue
        if not (row['mask'][ij[...,1],ij[...,0]]&~row['occluder'][ij[...,1],ij[...,0]]).all():continue
        source=session/'splat/data/images'/row['name']
        if row['name'] not in images:images[row['name']]=cv2.imread(str(source))
        patch=cv2.remap(images[row['name']],uv[...,0].astype('float32'),uv[...,1].astype('float32'),cv2.INTER_CUBIC)
        quality=repeatability(patch)
        if quality['suitable_for_repetition']:
            selected=True;break
        rejected.append({'image':row['name'],'world_centre_xy':(local@axes.T+xy).tolist(),'quality':quality})
    if not selected:raise ValueError('No observed patch suitable for repetition; retain source evidence and request material review')
    # Clear source pixels can still make a visible grid when the crop's opposite
    # edges differ. Keep the measured crop and remove only its smooth mismatch
    # in the inferred repeating material.
    from .texture_quality import periodic_tile
    original=directory/(Path(output_name).stem+'_observed.png')
    cv2.imwrite(str(original),patch)
    patch,seams=periodic_tile(patch)
    cv2.imwrite(str(destination),patch)
    manifest={'source':'Current independently labelled visible support only','image':rel(source),'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'support_mesh_sha256':fit['mesh_sha256'],'cameras_sha256':hashlib.sha256((session/'parsing/posed_frames/cameras.npz').read_bytes()).hexdigest(),'patch_m':patch_m,'world_centre_xy':(local@axes.T+xy).tolist(),'support_yaw_deg':fit['pose']['yaw_deg'],'output':rel(destination),'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'hidden_material':'Same observed wood patch tiled at reconstructed physical scale; unobserved grain is inferred'}
    manifest.update(quality=quality,rejected_candidates=rejected,seam_correction=seams,
                    observed_crop=rel(original),observed_crop_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    sampling_surface_plane=surface_plane,
                    producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_observation_inputs=observation_inputs,
                    selection={'mode':'reviewed_source_frame' if source_frame is not None else 'projected_area',
                               'frame':source_frame,'basis':selection_basis,
                               'scope':'Material-patch selection only; no reconstruction observation or source-acceptance frame is removed'})
    (directory/(Path(output_name).stem+'_source.json')).write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('--patch-m',type=float,default=.18);p.add_argument('--output-name',default='wood.png')
    p.add_argument('--source-frame',help='Reviewed current image with independently labelled support; material sampling only')
    p.add_argument('--selection-basis',help='Source-only reason for selecting this material view, such as resolved grain')
    a=p.parse_args();print(json.dumps(run(Session(a.session),a.patch_m,output_name=a.output_name,
        source_frame=a.source_frame,selection_basis=a.selection_basis),indent=2))

if __name__=='__main__':main()
