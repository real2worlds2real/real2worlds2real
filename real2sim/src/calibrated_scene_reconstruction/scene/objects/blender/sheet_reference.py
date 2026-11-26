"""Recover complete flat material from reviewed source evidence; never a task pose."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from calibrated_scene_reconstruction.assets import cloth_outline,cloth_sheet
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.observation_exclusions import source_inputs,plane_camera
from . import evidence,material_reference


def run(s,obj,link,out,*,spacing_m=.001,resolution=1024):
    s=Session(s.dir.resolve());out=Path(out).resolve()
    if not out.is_relative_to(s.dir/'qa') or type(resolution) is not int or not 128<=resolution<=2048:
        raise ValueError('New current-session QA output and bounded source texture resolution required')
    cameras,frames,_,_=evidence.load(s,obj,parts=True)
    ref=next((r for r in cameras.get('_source_material',[]) if r['link']==obj+'/'+link),None)
    if not ref or ref['relationship']!='same_sheet_geometry':
        raise ValueError('Reviewed complete unfolded source reference required')
    source=next(x for x in source_inputs(s,obj,cameras) if x['region']['id']==ref['region'])
    sheet=cloth_sheet.build(ref['source_unfolded_geometry'],spacing_m=spacing_m)
    geometry=material_reference.intrinsic_geometry(sheet['rest'],sheet['faces'],sheet['report']['material_area_m2'])
    measured=material_reference.score({'_source_material':[ref]},[{'name':link,'geometry':geometry}])
    selected=next(x for x in measured if x['link']==obj+'/'+link)
    if not selected['ok']:raise ValueError('Generated complete sheet fails original source-material constraints')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py')
    photo_uv=sheet['uv'].copy();photo_uv[:,1]=1-photo_uv[:,1]
    np.savez_compressed(out/'material_sheet.npz',points=sheet['points'],faces=sheet['faces'],uv=sheet['uv'],photo_uv=photo_uv,
                        cloth_rest_json=np.array(json.dumps({link:sheet['rest']})))
    shape=ref['source_unfolded_geometry'];axes=np.asarray(shape['plane_axes']);centre=np.asarray(shape['plane_centre_m'])
    loops=[np.asarray(c)*np.sqrt(shape['area_m2']) for c in shape['area_normalized_contours']]
    low=np.concatenate(loops).min(0);high=np.concatenate(loops).max(0);scale=resolution/(high-low).max()
    size=np.ceil((high-low)*scale).astype(int)+1
    plane=np.eye(3);plane[:2,:2]=axes;plane[:2,2]=-axes@centre
    pixels=np.diag([scale,scale,1.]);pixels[:2,2]=-scale*low
    a=source['anchor'];H=plane_camera(cameras['K'][a],cameras['R_world_cam'][a],cameras['centre'][a],source['top_z'])
    transform=pixels@plane@np.linalg.inv(H)
    image_path=s.dir/source['region']['source_image'];rgb=cv2.imread(str(image_path))
    image=cv2.warpPerspective(rgb,transform,tuple(size),flags=cv2.INTER_LINEAR)
    mask=cv2.warpPerspective(source['mask'].astype('uint8'),transform,tuple(size),flags=cv2.INTER_NEAREST)
    photo=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA);photo[:,:,3]=mask*255;photo[mask==0,:3]=0
    cv2.imwrite(str(out/'reference.png'),photo)
    files=[image_path,s.config_dir/source['region']['mask'],s/'parsing/posed_frames/cameras.npz',
           s.config_dir/'agent/material_references.json',s.config_dir/'agent/physics_priors.json']
    for module in [cloth_outline,cloth_sheet,material_reference]:
        path=Path(module.__file__);shutil.copy2(path,out/path.name);files.append(path)
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report={'ok':True,'source_ok':True,'proposal_only':True,'task_approved':False,'world_pose_approved':False,
            'instance_relationship':ref.get('instance_relationship','unspecified'),
            'exact_instance_boundary_or_pattern':ref.get('instance_relationship')=='same_instance',
            'link':obj+'/'+link,'reference':ref,'material':sheet['report'],'geometry':geometry,'source_check':selected,
            'material_mesh':str(out/'material_sheet.npz'),'reference_photo':str(out/'reference.png'),
            'evidence_images':[str(out/'reference.png')],
            'source_to_reference_pixels':transform.tolist(),
            'photo_uv_convention':'Bottom-left texture origin; geometric uv remains material y increasing. Use photo_uv for reference.png.',
            'input_sha256':{str(p.resolve()):sha(p) for p in [*files,Path(__file__)]},
            'output_sha256':{p.name:sha(p) for p in [out/'material_sheet.npz',out/'reference.png']},
            'scope':'Complete source-derived flat cloth template and photographic fabric reference only. A same-type separate specimen constrains size/shape/material within declared uncertainty; its frayed boundary and motif phase are not exact correspondences to the target sheet. Source lighting remains in RGB; this is not calibrated albedo. Packing, fold count, visible material correspondence, native mechanics and task feasibility remain unapproved.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');p.add_argument('link')
    p.add_argument('--output',required=True);p.add_argument('--spacing-m',type=float,default=.001)
    args=p.parse_args();result=run(Session(args.session),args.object,args.link,args.output,spacing_m=args.spacing_m)
    print(json.dumps({k:result[k] for k in ['ok','proposal_only','task_approved','material_mesh','reference_photo']}))


if __name__=='__main__':main()
