"""Source-labelled functional parts constrain fitting as well as native checks."""
import numpy as np


def requirements(session,obj):
    if not hasattr(session,'config_dir'):return []
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
    from .surface_observations import requirements as regions
    plan=declaration(session)
    return [item for item in (plan or {}).get('observations',[])
            if item.get('part_prompt') and item['link'].split('/')[0]==obj] + regions(session,obj)


def attach(session,obj,cameras):
    requested=requirements(session,obj)
    from .source_pose_groups import declaration as groups
    pose_groups=groups(session,obj)
    if not requested and not pose_groups:return
    from calibrated_scene_reconstruction.scene.objects.physical_observations import observations
    sources=[]
    for item in requested:
        observed=observations(session,obj,item['part_prompt'],exterior_to=item.get('exterior_to'))
        sources.append({'id':item['id'],'link':item['link'].split('/')[1],
                        **({'links':[ref.split('/')[1] for ref in item['link_group']]} if 'link_group' in item else {}),
                        **({'materials':item['materials']} if 'materials' in item else {}),
                        'prompt':item['part_prompt'],'observations':observed,'object':obj,
                        'transmitting_links':item.get('transmitting_links',[]),
                        'transmission_basis':item.get('transmission_basis')})
    for item in pose_groups:
        observed=observations(session,obj,item['prompt'])
        sources.append({**item,'object':obj,'observations':observed})
    if sources:cameras['_source_parts']=sources


def target_faces(source, model_parts, count):
    if 'materials' in source:
        from calibrated_scene_reconstruction.assets.surface_regions import select
        return select(model_parts,source['materials'],count)
    names=source.get('links') or [source['link']]
    target=np.zeros(count,bool)
    for name in names:
        part=next((part for part in model_parts if part['name']==name),None)
        if part is None:raise ValueError('Missing independently observed Blender link: '+name)
        first,size=part['face_start'],part['triangle_count'];target[first:first+size]=True
    if not target.any():raise ValueError('Observed Blender links have no actual surface')
    return target


def score(mesh,pose,cameras,model_parts,rasterizer=None):
    sources=cameras.get('_source_parts',[])
    if not sources:return []
    if not model_parts:raise ValueError('Source part constraints require preserved Blender link faces')
    from calibrated_scene_reconstruction.core.geometry.visibility import VisibleParts
    from calibrated_scene_reconstruction.scene.objects.object_refine import pieces
    results=[]
    for source in sources:
        target=target_faces(source,model_parts,len(mesh.faces))
        gpu=rasterizer is not None and hasattr(rasterizer,'visible')
        from calibrated_scene_reconstruction.scene.objects.optical_visibility import transmitting_faces
        optical=transmitting_faces(source.get('object',''),source,model_parts,len(mesh.faces))
        optical_mesh=mesh;optical_target=target
        if gpu and optical is not None:
            import trimesh
            optical_mesh=trimesh.Trimesh(mesh.vertices,mesh.faces[~optical],process=False);optical_target=target[~optical]
        from .context_visibility import compose
        if optical is not None and not gpu:
            import trimesh
            optical_mesh=trimesh.Trimesh(mesh.vertices,mesh.faces[~optical],process=False);optical_target=target[~optical]
        optical_mesh,optical_target=compose(optical_mesh,cameras,optical_target)
        native=None if gpu else VisibleParts(optical_mesh.vertices,optical_mesh.faces,optical_target,device='cpu')
        z,frames,masks,windows=source['observations'];ious=[];local=[];native_pixels=[];coverage=[]
        for k,(i,mask,window) in enumerate(zip(frames,masks,windows)):
            args=(z['K'][i],z['R_world_cam'][i],z['centre'][i],window,mask.shape)
            if gpu:
                predicted=rasterizer.visible(optical_mesh,*args,target_faces=optical_target)
            else:
                # Reuse exact visibility hits to distinguish a missing surface
                # from another region occluding/mislabelling the observed part.
                # This diagnoses where to inspect; it never relabels evidence
                # or changes the acceptance criteria.
                ids=native.face_image(*args)
                hit=ids>=0;predicted=hit.copy()
                predicted[hit]&=optical_target[ids[hit]]
                domain=z['_object_domains'][k] if '_object_domains' in z else np.ones(mask.shape,bool)
                observed=mask&domain
                coverage.append({'frame':int(i),'observed_pixels':int(observed.sum()),
                    'correct_region_pixels':int((observed&predicted).sum()),
                    'other_region_pixels':int((observed&hit&~predicted).sum()),
                    'missing_geometry_pixels':int((observed&~hit).sum())})
            if '_object_domains' in z:predicted &= z['_object_domains'][k]
            ious.append(float((predicted&mask).sum()/max(1,(predicted|mask).sum())))
            local.append(float(sum(piece[0] for piece in pieces(mask,predicted,.003))))
            native_pixels.append(float(mask.sum()/window[4]**2))
        iou=float(np.median(ious));difference=float(np.median(local))
        diagnostic={}
        if coverage:
            total=sum(row['observed_pixels'] for row in coverage)
            diagnostic={'coverage_diagnostic':{
                'per_frame':coverage,
                **{key+'_fraction':sum(row[key+'_pixels'] for row in coverage)/max(1,total)
                   for key in ('correct_region','other_region','missing_geometry')},
                'scope':'Exact source-visible triangle hits. Other-region coverage may be a wrong label, wrong shape or wrong occlusion; inspect source views before changing geometry. Diagnostic only; no acceptance waiver.'}}
        results.append({'id':source['id'],**diagnostic,
                        'link':source.get('link','group:'+source['id']),
                        **({'links':source['links'],**({'gate':source['gate']} if 'gate' in source else {})} if 'links' in source else {}),
                        'prompt':source['prompt'],
                        'iou_median':iou,'local_difference':difference,'views':len(frames),
                        'ok':bool(iou>=.85 and difference<=.02 and len(frames)>=8),
                        'loss':float(1-np.average(ious,weights=native_pixels)+np.average(local,weights=native_pixels)),
                        'loss_weighting':'Native observed part pixels; unchanged unweighted acceptance gates',
                        'ious':ious,'local_per_frame':local,'native_mask_pixels':native_pixels,
                        'unobserved_source_views':z.get('_part_unobserved_views',[]),
                        'optical_visibility':{'transmitting_links':source.get('transmitting_links',[]),
                            'transmitting_faces':int(optical.sum()) if optical is not None else 0,
                            'model':'Straight-ray optical proxy; source verification and actual transmitting materials required; all collision faces retained'},
                        'scope':'Source part labels and visibility through all actual object triangles'})
    return results


def sheets(session,mesh,cameras,model_parts,output,*,frame_reader=None):
    """All-view part comparisons for the agent's next geometry iteration.

    These are fit diagnostics. Source identities are reviewed separately from
    source-only evidence, never corrected by copying this rendered boundary.
    """
    from pathlib import Path
    import cv2
    from calibrated_scene_reconstruction.scene.objects.physical_observations import compare
    from calibrated_scene_reconstruction.scene.objects.object_refine import _frame
    frame_reader=frame_reader or _frame
    root=Path(output);root.mkdir(parents=True,exist_ok=True);result={}
    for source in cameras.get('_source_parts',[]):
        target=target_faces(source,model_parts,len(mesh.faces))
        from calibrated_scene_reconstruction.scene.objects.optical_visibility import transmitting_faces
        optical=transmitting_faces(source.get('object',''),source,model_parts,len(mesh.faces))
        metrics,predictions=compare(mesh.vertices,mesh.faces,target,source['observations'],renders=True,transmitting_faces=optical)
        z,frames,masks,windows=source['observations'];tiles=[]
        def frame_name(i):return str(z['names'][i]) if 'names' in z else f'frame_{i}'
        for k,(i,mask,predicted,window) in enumerate(zip(frames,masks,predictions,windows)):
            image=frame_reader(session,z,int(i),window,mask.shape);overlay=image.copy()
            for region,colour in [(mask,(240,40,40)),(predicted,(40,220,40))]:
                contours,_=cv2.findContours(region.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,colour,2)
            diff=(image*.25).astype('uint8');diff[mask&predicted]=(235,235,235)
            diff[mask&~predicted]=(230,40,40);diff[predicted&~mask]=(40,210,40)
            if '_object_domains' in z:diff[~z['_object_domains'][k]]=(65,80,110)
            row=[cv2.resize(tile,(256,256)) for tile in [image,overlay,diff]]
            cv2.putText(row[0],frame_name(i),(4,18),0,.45,(255,255,0),1)
            cv2.putText(row[1],f'IoU {metrics["ious"][k]:.3f}',(4,18),0,.45,(255,255,0),1)
            cv2.putText(row[2],f'local {metrics["local_per_frame"][k]:.3f}',(4,18),0,.45,(255,255,0),1)
            tiles.append(np.hstack(row))
        paths=[]
        for start in range(0,len(tiles),8):
            path=root/f'{source["id"]}_{start//8+1}.jpg';cv2.imwrite(str(path),np.vstack(tiles[start:start+8])[:,:,::-1])
            paths.append(str(path.resolve().relative_to(Path(session.dir).resolve())))
        result[source['id']]={'sheets':paths,'frames':[frame_name(i) for i in frames],
                              'scope':'Actual agent geometry and independent source labels; not source-mask approval'}
    return result
