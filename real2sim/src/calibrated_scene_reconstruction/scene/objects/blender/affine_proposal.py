"""Bounded differentiable initialization of independent rigid body collections.

This operates on actual candidate triangles, not primitives or target dimensions.
It cannot export an approved asset or replace final Blender/source/native checks.
"""
import json
from pathlib import Path
import shutil
import numpy as np


def transform(points,owner,pivots,values,translation_limit,angle_limit,log_scale_limit,
              *,scale_frames=None,transverse_tied=False):
    import torch
    bounded=torch.tanh(values)
    delta=bounded[:,:3]*translation_limit
    angle=bounded[:,3:6]*angle_limit
    scale_values=bounded[:,6:9]
    if transverse_tied:scale_values=torch.stack([scale_values[:,0],scale_values[:,0],scale_values[:,2]],dim=1)
    scale=torch.exp(scale_values*log_scale_limit)
    x,y,z=angle.unbind(1);zero=torch.zeros_like(x)
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],dim=1).reshape(-1,3,3)
    rotation=torch.matrix_exp(skew)
    local=points-pivots[owner]
    if scale_frames is not None:
        frames=scale_frames[owner]
        local=torch.einsum('nij,nj->ni',frames.transpose(1,2),local)
        local=torch.einsum('nij,nj->ni',frames,local*scale[owner])
    else:local=local*scale[owner]
    result=(rotation[owner]@local[:,:,None])[:,:,0]+pivots[owner]+delta[owner]
    return result,delta,rotation,scale


def scale_basis(frames,count,*,transverse_tied=False):
    """Explicit orthonormal body axes keep circular shafts circular.

    Columns are the two transverse axes and longitudinal axis, in world
    coordinates. This basis is a source-reviewed construction hypothesis;
    it does not identify a cylinder or approve semantic geometry by itself.
    """
    if type(transverse_tied) is not bool:raise ValueError('Explicit transverse scale policy required')
    if frames is None:
        if transverse_tied:raise ValueError('Tied transverse scale needs explicit body axes')
        return None
    frames=np.asarray(frames,float)
    if (frames.shape!=(count,3,3) or not np.isfinite(frames).all()
            or not np.allclose(frames.transpose(0,2,1)@frames,np.eye(3),rtol=0,atol=1e-7)
            or not np.allclose(np.linalg.det(frames),1.,rtol=0,atol=1e-7)):
        raise ValueError('One proper orthonormal scale frame per actual body required')
    return {'frames':frames.tolist(),'transverse_tied':transverse_tied,
            'mode':'body_frame_scale'}


def radial_basis(points,owner,pivots,controls=32,concentration=8.):
    """Smooth bounded angular cage, suitable only for reviewed star-shaped bodies."""
    if (type(controls) is not int or not 16<=controls<=128
            or not np.isfinite(concentration) or not 1<=concentration<=32):
        raise ValueError('Bounded angular cage resolution and concentration required')
    direction=np.asarray(points)-np.asarray(pivots)[np.asarray(owner)]
    norms=np.linalg.norm(direction,axis=1)
    if np.any(norms<1e-9):raise ValueError('Radial cage cannot contain a vertex at its pivot')
    direction/=norms[:,None]
    k=np.arange(controls);z=1-2*(k+.5)/controls;phi=k*np.pi*(3-np.sqrt(5))
    centres=np.c_[np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z]
    weights=np.exp(concentration*(direction@centres.T-1))
    return weights/weights.sum(1,keepdims=True)


def validate_radial_body(vertices,faces,pivot):
    """Require a single closed, outward mesh with the pivot in its kernel.

    Each outward face half-space must contain the pivot. For a simple closed
    polyhedron this makes every radial segment interior; open cups, rings and
    cavity shells cannot silently use a solid radial deformation model.
    """
    import trimesh
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0
            or len(mesh.split(only_watertight=False))!=1):
        raise ValueError('Radial cage requires one closed outward solid per body')
    extent=float(np.ptp(mesh.vertices,axis=0).max())
    distance=np.einsum('ij,ij->i',mesh.triangles_center-np.asarray(pivot),mesh.face_normals)
    if (not np.isfinite(distance).all() or np.min(distance)<-extent*1e-8):
        raise ValueError('Radial cage pivot is outside the star-shaped solid kernel')
    return {'closed_outward_solid':True,'kernel_margin_m':float(distance.min())}


def warm_parameters(path,root,parts,*,radial_cage,cage_controls,cage_concentration,scale_basis=None):
    """Load bounded variables only when their original geometry/evidence still match.

    Engine code may evolve between diagnostic proposals. All non-code inputs,
    body ordering, shape basis and original coordinate bounds must remain the
    same; no previously proposed world mesh becomes a new unconstrained base.
    """
    from .store import digest
    path,root=Path(path),Path(root)
    report=json.loads((path/'report.json').read_text())
    inputs=json.loads((path/'inputs.json').read_text())
    if (report.get('input_mesh_sha256')!=digest(root/'mesh.npz')
            or report.get('input_fit_sha256')!=digest(root/'fit.json')
            or report.get('proposal_sha256')!=digest(path/'proposal.npz')
            or report.get('body_names')!=[p['name']for p in parts]
            or report.get('source_inputs_unchanged') is not True
            or report.get('radial_cage') is not radial_cage
            or report.get('angle_limit_rad')!=.7 or report.get('log_scale_limit')!=.3
            or any(not Path(p).exists() or digest(p)!=h for p,h in inputs.items()
                   if Path(p).suffix!='.py')):
        raise ValueError('Warm proposal geometry, body identity or source evidence changed')
    if radial_cage and (report.get('cage_controls')!=cage_controls
                       or report.get('cage_concentration')!=cage_concentration):
        raise ValueError('Warm proposal radial basis changed')
    if report.get('scale_basis')!=scale_basis:
        raise ValueError('Warm proposal scale basis or tied degrees of freedom changed')
    with np.load(path/'proposal.npz') as source:
        values=np.array(source['affine_parameters'],copy=True)
        coefficients=np.array(source['cage_coefficients'],copy=True) if radial_cage else None
        predicted=np.array(source['V_world'],copy=True)
    if (values.shape!=(len(parts),9) or not np.isfinite(values).all()
            or radial_cage and (coefficients.shape!=(len(parts),cage_controls)
                               or not np.isfinite(coefficients).all())
            or predicted.ndim!=2 or predicted.shape[1]!=3 or not np.isfinite(predicted).all()):
        raise ValueError('Invalid warm proposal variables')
    return values,coefficients,predicted


def soft_local_discrepancy(predicted,target,domain):
    """Differentiable opening of both error signs, a proposal loss only.

    Full CPU acceptance additionally applies its original connected-component
    area cutoff. Isolated subpixel/border noise must not displace broad shape
    errors as it can with a mean pixel objective alone.
    """
    import torch.nn.functional as functional
    total=predicted.sum()*0
    for difference in (predicted-target,target-predicted):
        positive=difference.clamp(min=0)*domain
        eroded=-functional.max_pool2d(-positive[None,None],3,stride=1,padding=1)
        opened=functional.max_pool2d(eroded,3,stride=1,padding=1)
        total=total+opened.sum()
    return total/target.sum().clamp(min=1)


def run(session,obj,number,output,*,iterations=600,radial_cage=False,cage_controls=32,cage_concentration=8.,
        warm_start=None,learning_rate=.035,final_full_view_steps=0,cpu_checkpoint_interval=0,
        local_discrepancy_weight=0.,local_discrepancy_aggregate="mean",scale_frames=None,transverse_tied=False,
        local_discrepancy_mode='components'):
    import cv2,torch,nvdiffrast.torch as dr
    from . import evidence
    from .store import digest
    from .projection import ProposalRaster
    if type(iterations) is not int or not 50<=iterations<=2000:
        raise ValueError('Bounded50..2000 iteration proposal required')
    if (not np.isfinite(learning_rate) or not 0<learning_rate<=.1
            or type(final_full_view_steps) is not int or not 0<=final_full_view_steps<=iterations
            or type(cpu_checkpoint_interval) is not int or not 0<=cpu_checkpoint_interval<=iterations
            or not np.isfinite(local_discrepancy_weight) or not 0<=local_discrepancy_weight<=10):
        raise ValueError('Bounded learning rate and final full-view step count required')
    if local_discrepancy_aggregate not in {'mean','median'}:
        raise ValueError('Local proposal aggregate must be explicit mean or median')
    if local_discrepancy_mode not in {'components','soft'}:
        raise ValueError('Choose original component-gate or legacy soft local proposal loss')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Proposal must belong to the active session')
    out.mkdir(parents=True,exist_ok=False)
    root=session/'objects'/obj/f'round_{number}'
    fitted=json.loads((root/'fit.json').read_text())
    with np.load(root/'mesh.npz') as raw:
        mesh={k:raw[k] for k in raw.files}
    parts=json.loads(str(mesh['links_json']))
    if (not parts or any(p['kind']!='rigid' for p in parts)
            or json.loads(str(mesh.get('joints_json','[]')))):
        raise ValueError('Affine proposal is limited to independently movable rigid bodies without joints')
    scale_contract=scale_basis(scale_frames,len(parts),transverse_tied=transverse_tied)
    if scale_contract and radial_cage:
        raise ValueError('Explicit axial scale cannot silently combine with a radial shape cage')
    warm=None
    if warm_start is not None:
        warm_start=Path(warm_start).resolve()
        if not warm_start.is_relative_to(session.dir.resolve()):
            raise ValueError('Warm proposal must belong to the active session')
        warm=warm_parameters(warm_start,root,parts,radial_cage=radial_cage,
                             cage_controls=cage_controls,cage_concentration=cage_concentration,
                             scale_basis=scale_contract)
    z,frames,masks,windows=evidence.load(session,obj,parts=True)
    inputs={str(p.resolve()):digest(p) for p in [root/'mesh.npz',root/'fit.json',
        session/'parsing/posed_frames/cameras.npz',session/'splat/align.json',
        session/'splat/object_fit'/f'{obj}.npz',Path(__file__)]}
    if local_discrepancy_weight and local_discrepancy_mode=='components':
        helper=Path(__file__).with_name('proposal_loss.py')
        inputs[str(helper)]=digest(helper)
    inputs.update({str((session.dir/k).resolve()):v for k,v in z.get('_object_domain_inputs',{}).items()})
    if warm is not None:
        snapshot=out/'warm_start_inputs';snapshot.mkdir()
        for name in ['proposal.npz','report.json','inputs.json','source.py']:
            source_path=warm_start/name
            inputs[str(source_path)]=digest(source_path)
            shutil.copyfile(source_path,snapshot/name)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    source=evidence.placed(mesh['V'],mesh['F'],fitted['pose'])
    owner=np.full(len(source.vertices),-1,int);pivots=[]
    for i,p in enumerate(parts):
        a,n=p['vertex_start'],p['vertex_count']
        if (owner[a:a+n]>=0).any():raise ValueError('Independent bodies cannot share vertices')
        owner[a:a+n]=i;pivots.append(source.vertices[a:a+n].mean(0))
    if (owner<0).any():raise ValueError('Every vertex requires explicit body ownership')
    radial_checks=[]
    if type(radial_cage) is not bool:raise ValueError('Radial cage selection must be explicit boolean')
    if radial_cage:
        for p,pivot0 in zip(parts,pivots):
            a,n=p['vertex_start'],p['vertex_count'];f,t=p['face_start'],p['triangle_count']
            radial_checks.append(validate_radial_body(source.vertices[a:a+n],source.faces[f:f+t]-a,pivot0))
    tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype,device='cuda')
    v=tensor(source.vertices);owners=tensor(owner,torch.long);pivot=tensor(np.array(pivots));face=source.faces.copy()
    context=z.get('_source_context');context_v=None
    if context:
        parent=context['mesh'];context_v=tensor(parent.vertices);face=np.r_[face,parent.faces+len(v)]
    face=tensor(face,torch.int32);labels=tensor(np.r_[np.ones(len(source.faces)),np.zeros(len(face)-len(source.faces))])
    extents=np.array([np.ptp(source.vertices[p['vertex_start']:p['vertex_start']+p['vertex_count']],axis=0).max() for p in parts])
    limits=tensor(extents*.35)[:,None];angle_limit=.7;log_scale_limit=.3
    scale_options={'scale_frames':tensor(scale_contract['frames']) if scale_contract else None,
                   'transverse_tied':transverse_tied}
    raster=ProposalRaster();topology=dr.antialias_construct_topology_hash(face);views=[]
    for k,(i,mask,window) in enumerate(zip(frames,masks,windows)):
        x,y,_,_,scale=window;K=np.array([[scale,0,-x*scale],[0,scale,-y*scale],[0,0,1]])@z['K'][i]
        domain=z['_object_domains'][k] if '_object_domains' in z else np.ones_like(mask)
        distance=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,3)
        views.append((tensor(K),tensor(z['R_world_cam'][i]),tensor(z['centre'][i]),tensor(mask),tensor(domain),tensor(1+np.minimum(distance,20)/10)))
    values=torch.nn.Parameter(torch.zeros((len(parts),9),device='cuda') if warm is None else tensor(warm[0]))
    basis=tensor(radial_basis(source.vertices,owner,pivots,cage_controls,cage_concentration)) if radial_cage else None
    coefficients=None
    if radial_cage:
        coefficients=torch.nn.Parameter(torch.zeros((len(parts),cage_controls),device='cuda')
                                        if warm is None else tensor(warm[1]))
    parameters=[values]+([coefficients]if radial_cage else [])
    optimizer=torch.optim.Adam(parameters,lr=learning_rate);best=(float('inf'),None,None);history=[]
    best_cpu=None;cpu_history=[]
    def deformed(cage):
        if cage is None:return v
        radial=1+.18*(basis*torch.tanh(cage)[owners]).sum(1)
        return pivot[owners]+(v-pivot[owners])*radial[:,None]
    def render(vertices,view):
        K,R,C,gt,domain,weight=view;h,w=gt.shape;camera=(vertices-C)@R;depth=camera[:,2]
        x=2*K[0,0]*camera[:,0]/w+(2*(K[0,2]+.5)/w-1)*depth
        y=2*K[1,1]*camera[:,1]/h+(2*(K[1,2]+.5)/h-1)*depth
        clip=torch.stack([x,y,100.05/99.95*depth-10/99.95,depth],dim=-1)[None].contiguous()
        result,_=dr.rasterize(raster.context,clip,face,(h,w),grad_db=False)
        ids=result[...,3].long()-1;coverage=torch.where(ids>=0,labels[ids.clamp(min=0)],0.)[...,None]
        image=dr.antialias(coverage,result,clip,face,topology_hash=topology)[0,...,0]
        loss=((image-gt).abs()*domain*weight).sum()/gt.sum().clamp(min=1)
        local=loss*0
        if local_discrepancy_weight:
            if local_discrepancy_mode=='components':
                from .proposal_loss import local_components
                local=local_components(image,gt,domain,hard_prediction=coverage[0,...,0]>0,
                                       share=evidence.PIECE_MIN)
            else:local=soft_local_discrepancy(image,gt,domain)
        return loss,local
    if warm is not None:
        reconstructed=transform(deformed(coefficients),owners,pivot,values,limits,angle_limit,log_scale_limit,**scale_options)[0]
        if not np.allclose(reconstructed.detach().cpu().numpy(),warm[2],rtol=0,atol=2e-7):
            raise ValueError('Warm variables do not reproduce the saved actual candidate mesh')
    for step in range(iterations+1):
        points,_,_,_=transform(deformed(coefficients),owners,pivot,values,limits,angle_limit,log_scale_limit,**scale_options)
        if context_v is not None:points=torch.cat([points,context_v])
        check_cpu=bool(cpu_checkpoint_interval and (step%cpu_checkpoint_interval==0 or step==iterations))
        selected=(list(range(len(views))) if step%40==0 or step>=iterations-final_full_view_steps or check_cpu or local_discrepancy_aggregate=='median'
                  else [(step*4+j)%len(views)for j in range(min(4,len(views)))])
        terms=[render(points,views[k])for k in selected]
        losses=torch.stack([term[0]for term in terms]);locals_=torch.stack([term[1]for term in terms])
        local_term=locals_.mean() if local_discrepancy_aggregate=='mean' else locals_.median()
        loss=losses.mean()+local_discrepancy_weight*local_term+values.square().mean()*.0001
        if radial_cage:loss=loss+coefficients.square().mean()*.0005
        if len(selected)==len(views):
            numeric=float(loss.detach());history.append({'iteration':step,'all_view_loss':numeric})
            if numeric<best[0]:best=(numeric,values.detach().clone(),coefficients.detach().clone()if radial_cage else None)
            print('AFFINE',step,numeric,'best',best[0],flush=True)
        if check_cpu:
            # CPU uses actual current triangles, all original observations,
            # unchanged local gates and the parent-support context. The
            # proposal cannot turn a failed parent or native check into a pass.
            current=evidence.score(points[:len(v)].detach().cpu().numpy(),source.faces,
                {'xy':[0.,0.],'support_z':0.,'yaw_deg':0.},(z,frames,masks,windows),model_parts=parts)[0]
            objective=evidence.search_objective(current)
            key=(bool(current['ok']),-objective)
            cpu_history.append({'iteration':step,'source_objective':objective,'metrics':current})
            if best_cpu is None or key>best_cpu[0]:
                best_cpu=(key,numeric,values.detach().clone(),
                          coefficients.detach().clone()if radial_cage else None,current,step)
            print('CPU',step,current['iou_median'],current['local_difference'],current['ok'],flush=True)
        if step<iterations:
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,2.);optimizer.step()
    gpu_best_loss=best[0]
    if best_cpu is not None:best=(best_cpu[1],best_cpu[2],best_cpu[3])
    points,delta,rotation,scale=transform(deformed(best[2]),owners,pivot,best[1],limits,angle_limit,log_scale_limit,**scale_options)
    arrays=[a.detach().cpu().numpy()for a in [points,delta,rotation,scale]]
    np.savez_compressed(out/'proposal.npz',V_world=arrays[0],F=source.faces,pivots=pivots,delta=arrays[1],rotation=arrays[2],scale=arrays[3],
        affine_parameters=best[1].cpu().numpy(),**({'cage_coefficients':best[2].cpu().numpy(),'radial_fraction_limit':.18,
        'cage_concentration':cage_concentration}if radial_cage else {}))
    report={'accepted':False,'task_approved':False,'native_tested':False,'object':obj,'round':number,'body_names':[p['name']for p in parts],
        'input_mesh_sha256':digest(root/'mesh.npz'),'input_fit_sha256':digest(root/'fit.json'),
        'proposal_sha256':digest(out/'proposal.npz'),'history':history,'loss':best[0],
        'translation_limit_m':(extents*.35).tolist(),'angle_limit_rad':angle_limit,'log_scale_limit':log_scale_limit,
        'scale_basis':scale_contract,
        'radial_cage':radial_cage,'cage_controls':cage_controls if radial_cage else None,
        'cage_concentration':cage_concentration if radial_cage else None,
        'warm_start':str(warm_start)if warm is not None else None,'learning_rate':learning_rate,
        'final_full_view_steps':final_full_view_steps,
        'cpu_checkpoint_interval':cpu_checkpoint_interval,'cpu_checkpoints':cpu_history,
        'selection_policy':'unchanged full-source CPU gates/objective'if best_cpu is not None else 'GPU mean proposal loss',
        'selected_cpu_iteration':best_cpu[5]if best_cpu is not None else None,
        'selected_cpu_metrics':best_cpu[4]if best_cpu is not None else None,'gpu_best_loss':gpu_best_loss,
        'local_discrepancy_weight':local_discrepancy_weight,'local_discrepancy_aggregate':local_discrepancy_aggregate,
        'local_discrepancy_mode':local_discrepancy_mode,
        'radial_body_checks':radial_checks,'source_inputs_unchanged':all(digest(p)==v for p,v in inputs.items()),
        'scope':__doc__.strip()+' Optional angular radial cage is limited to18% of local radius; it needs an independent star-shaped semantic/geometry review and final closed-mesh/contact/source validation.'}
    if not report['source_inputs_unchanged']:raise ValueError('Source inputs changed during geometric proposal')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report
