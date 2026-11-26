"""Agent-declared finish priors on named PBR regions, preserving observed graphics.

Uniform paint and view-dependent metal radiance need not be asserted as a
multiview colour collage. Policies and detail masks are source-bound agent inputs;
ordinary patterned materials retain the standard multiview reconstruction.
"""
import hashlib
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.core.session_code import source_path
from calibrated_scene_reconstruction.assets.metric_mesh import asset
from calibrated_scene_reconstruction.scene.objects.object_texture import to_srgb


def atlas_hash(uv, faces):
    return hashlib.sha256(np.ascontiguousarray(uv,dtype='<f8').tobytes()+
                          np.ascontiguousarray(faces,dtype='<i8').tobytes()).hexdigest()


def apply(session, obj, fix, baked, policies, directory):
    if not isinstance(policies,dict):raise ValueError('Named material finish policies must be an object')
    with np.load(resolve(fix['mesh'])) as raw:
        names=raw['material_names'].tolist();ids=raw['face_material']
    parts=asset(fix)['parts'];materials=parts[0]['materials']
    image=np.array(baked['albedo'],copy=True);coverage=np.asarray(baked['coverage'])>0
    h,w=image.shape[:2];owner=np.full((h,w),-1,'int32')
    for face,idx in zip(baked['UV'][baked['F']],ids):
        cv2.fillConvexPoly(owner,np.rint(face*[w,h]-.5).astype('int32'),int(idx))
    owner=owner[::-1];report={}
    for name,policy in policies.items():
        if name not in names or name not in materials:raise ValueError('Unknown physical material region: '+name)
        if not isinstance(policy,dict) or policy.get('mode') not in {'uniform','source','source_tint'} or not policy.get('basis'):
            raise ValueError('Finish prior needs uniform/source/source_tint mode and an observed material basis')
        mask=owner==names.index(name);preserve=np.zeros((h,w),bool)
        if policy.get('preserve_mask'):
            if atlas_hash(baked['UV'],baked['F'])!=policy.get('atlas_sha256'):
                raise ValueError('Reviewed detail mask belongs to a different texture atlas')
            path=source_path(session.config_dir,policy['preserve_mask']);preserve=np.asarray(Image.open(path).convert('L'))>128
            if preserve.shape!=(h,w):raise ValueError('Detail mask image dimensions differ from the atlas')
            preserve &= coverage
            Image.fromarray(preserve.astype('uint8')*255).save(directory/f'details_{names.index(name)}.png')
        samples=mask&coverage&~preserve
        if policy['mode']=='uniform':
            if samples.sum()<20:raise ValueError('Uniform finish needs observed unprinted material samples')
            colour=np.median(image[samples],axis=0)
        else:
            linear=np.asarray(policy.get('base_color',materials[name]['color']),float)
            if linear.shape!=(3,) or not np.isfinite(linear).all() or (linear<0).any() or (linear>1).any():
                raise ValueError('Source PBR base colour requires finite linear RGB0..1')
            colour=to_srgb(linear)*255
        correction=None
        if policy['mode']=='source_tint':
            if samples.sum()<20:raise ValueError('Source tint requires observed detail samples')
            correction=np.asarray(colour)/np.maximum(np.median(image[samples],axis=0),1.)
            if (correction>4).any() or (correction<.25).any():
                raise ValueError('Large tint correction needs source exposure/material review')
            image[mask&~preserve]=np.rint(image[mask&~preserve]*correction).clip(0,255).astype('uint8')
        else:
            image[mask&~preserve]=np.rint(colour).clip(0,255).astype('uint8')
        report[name]={'policy':policy,'base_srgb':(np.asarray(colour)/255).tolist(),'observed_sample_pixels':int(samples.sum()),
                      'preserved_graphic_pixels':int((mask&preserve).sum()),'not_absolute_reflectance_identification':True}
        if correction is not None:report[name]['source_detail_rgb_gain']=correction.tolist()
        if policy.get('preserve_mask'):
            report[name]['preserve_mask_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    known=owner>=0
    if known.any():
        indices=distance_transform_edt(~known,return_distances=False,return_indices=True)
        image[~known]=image[indices[0][~known],indices[1][~known]]
    return image,report


def refinish(session, obj, policies):
    """Append a material-only look using existing calibrated samples and atlas."""
    import json
    import shutil
    from calibrated_scene_reconstruction.scene.objects.blender.contract import write_json
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen reconstruction')
    graph=session.scene_graph;fix=graph['dynamic_objects'][obj]['fix'];look=fix['look']
    if hashlib.sha256(resolve(fix['mesh']).read_bytes()).hexdigest()!=look.get('source_mesh_sha256'):
        raise ValueError('Appearance belongs to a different physical surface')
    source=look.get('sampled_source') or {'mesh':look['mesh'],'albedo':look['albedo'],
                                       'coverage':str((session/look['albedo']).with_name('coverage.png').relative_to(session.dir))}
    data=np.load(session/source['mesh']);baked={key:data[key] for key in data}
    baked.update(albedo=np.asarray(Image.open(session/source['albedo'])),coverage=np.asarray(Image.open(session/source['coverage'])))
    root=session/'objects'/obj/f"round_{fix['round']}";number=1;out=root/'look'
    while out.exists():number+=1;out=root/f'look_{number}'
    out.mkdir(parents=True,exist_ok=False)
    image,report=apply(session,obj,fix,baked,policies,out)
    shutil.copyfile(session/source['mesh'],out/'mesh.npz');shutil.copyfile(session/source['coverage'],out/'coverage.png')
    Image.fromarray(image).save(out/'albedo.png');write_json(out/'material_finishes.json',report)
    graph['dynamic_objects'][obj]['fit']['texture_materials']=policies
    fix['look'].update(mesh=str((out/'mesh.npz').relative_to(session.dir)),albedo=str((out/'albedo.png').relative_to(session.dir)),
                       material_finishes=str((out/'material_finishes.json').relative_to(session.dir)),material_policy=policies,
                       sampled_source=source)
    write_json(session.config_dir/'scene_graph.json',graph)
    (session/'qa/review/S4.ran').touch()
    return fix['look']
