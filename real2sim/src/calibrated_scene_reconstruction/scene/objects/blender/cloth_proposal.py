"""Source-conditioned flexible-geometry proposals; never native or task approval.

CUDA optional tool: python -m calibrated_scene_reconstruction.scene.objects.blender.cloth_proposal --help.
Material chart freedom, actual solid distances, full triangle/self/material
screens and unchanged exact multi-view gates are retained. The agent must
rebuild proposals through its model before accepted rounds or delivery.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
import nvdiffrast.torch as dr

from calibrated_scene_reconstruction.scene.session import Session
from . import evidence
from .material_reference import mesh_parts
from .part_evidence import sheets
from calibrated_scene_reconstruction.scene.objects.physical_observations import observations
from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
from calibrated_scene_reconstruction.scene.objects.optical_visibility import transmitting_faces
from calibrated_scene_reconstruction.assets.cloth_metric import material_inverse
from calibrated_scene_reconstruction.assets.cloth_state import check


def run(s,obj,candidate,observation,out,*,steps=400,learning_rate=2e-6,membrane_weight=10.,mode='field'):
    s=Session(s.dir.resolve())
    candidate=Path(candidate).resolve();out=Path(out).resolve()
    if not candidate.is_relative_to(s.dir.resolve()) or not out.is_relative_to(s.dir.resolve()):
        raise ValueError('Candidate and output must belong to the current session')
    if (type(steps) is not int or not 10<=steps<=5000 or not 0<learning_rate<=1e-4
            or not 0<=membrane_weight<=1000 or mode not in {'field','nodes','spatial'}):
        raise ValueError('Bounded steps, learning rate and membrane objective weight required')
    info=json.loads(candidate.read_text());mesh_input=(Path(info['path'])/'mesh.npz').resolve()
    if not mesh_input.is_relative_to(s.dir.resolve()):raise ValueError('Candidate mesh must belong to the current session')
    plan=declaration(s);item=next(i for i in plan['observations'] if i['id']==observation)
    owner,link=item['link'].split('/')
    if owner!=obj or not item.get('part_prompt'):raise ValueError('Independent source part observation for this object required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    for name in ['contact_field.py','source_objective.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    inputs={'candidate':str(candidate),'candidate_sha256':digest(candidate),'mesh':str(mesh_input),'mesh_sha256':digest(mesh_input),
            'object':obj,'observation':observation,'steps':steps,'learning_rate':learning_rate,'membrane_weight':membrane_weight,'mode':mode,
            'source_sha256':digest(__file__),'contact_field_sha256':digest(Path(__file__).with_name('contact_field.py')),
            'source_objective_sha256':digest(Path(__file__).with_name('source_objective.py')),
            'physics_priors_sha256':digest(s.config_dir/'agent/physics_priors.json'),
            'mechanisms_sha256':digest(s.config_dir/'agent/mechanisms.json'),
            'source_files':s.config('agent/mechanisms.json')['observations']}
    (out/'input.json').write_text(json.dumps(inputs,indent=2)+'\n')
    with np.load(mesh_input) as data:mesh={k:data[k].copy() for k in data.files}
    parts=mesh_parts(mesh);part=next(p for p in parts if p['name']==link);a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
    rest=json.loads(str(mesh['cloth_rest_json']))[link];plan=declaration(s);item=next(i for i in plan['observations'] if i['id']==observation and i['link']==obj+'/'+link)
    optical=transmitting_faces(obj,item,parts,len(mesh['F']))
    if optical is None:optical=np.zeros(len(mesh['F']),bool)
    faces=torch.as_tensor(mesh['F'][~optical].astype('int32'),device='cuda:0')
    original=torch.as_tensor(mesh['V'],dtype=torch.float32,device='cuda:0');cloth=original[a:a+n];low=cloth.min(0).values-.005;high=cloth.max(0).values+.005
    # Material-coordinate proposal freedom allows folded layers to slide; no native pins.
    rest_for_grid=np.array(rest['points']);centred=rest_for_grid-rest_for_grid.mean(0)
    _,_,axes=np.linalg.svd(centred,full_matrices=False);uv=centred@axes[:2].T
    if np.max(abs(centred@axes[2]))>1e-8:
        raise ValueError('This proposal tool requires a planar intrinsic material chart; retain other construction routes')
    lower_uv=uv.min(0);upper_uv=uv.max(0)
    grid=torch.tensor(((uv-lower_uv)/(upper_uv-lower_uv)*2-1)[None,:,None],dtype=torch.float32,device='cuda:0')
    if mode=='spatial':
        grid=((cloth-low)/(high-low)*2-1)[None,:,None,None]
    field_shape={'field':(1,3,49,37),'nodes':(n,3),'spatial':(1,3,17,25,25)}[mode]
    field=torch.nn.Parameter(torch.zeros(field_shape,device='cuda:0'))
    optimizer=torch.optim.Adam([field],lr=learning_rate)
    from calibrated_scene_reconstruction.assets.object_pose import rotation as object_rotation
    pose=info['pose'];rotation=torch.tensor(object_rotation(pose),device='cuda:0',dtype=torch.float32)
    translation=torch.tensor([*pose['xy'],pose['support_z']],device='cuda:0',dtype=torch.float32)
    classes=torch.zeros((len(original),1),device='cuda:0');classes[a:a+n]=1
    z,frames,masks,windows=observations(s,obj,item['part_prompt'],exterior_to=item.get('exterior_to'))
    snapshot={'frames':frames,'windows':windows,'K':z['K'][frames],
              'R_world_cam':z['R_world_cam'][frames],'centre':z['centre'][frames]}
    for k,mask in enumerate(masks):
        snapshot[f'mask_{k}']=mask
        snapshot[f'domain_{k}']=z.get('_object_domains',[np.ones_like(m) for m in masks])[k]
    np.savez_compressed(out/'source_observations.npz',**snapshot)
    inputs['source_observations_sha256']=digest(out/'source_observations.npz')
    (out/'input.json').write_text(json.dumps(inputs,indent=2)+'\n')
    groups={}
    for k,(i,mask,window) in enumerate(zip(frames,masks,windows)):
        groups.setdefault(mask.shape,[]).append((int(i),mask,window,z['_object_domains'][k] if '_object_domains' in z else np.ones_like(mask)))
    views=[]
    native_weights=[]
    for shape,rows in groups.items():
        Ks=[];Rs=[];Cs=[];targets=[];domains=[];weights=[]
        for i,mask,window,domain in rows:
            x,y,_,_,scale=window;weights.append(1./scale**2);Ks.append(np.array([[scale,0,-x*scale],[0,scale,-y*scale],[0,0,1.]])@z['K'][i]);Rs.append(z['R_world_cam'][i]);Cs.append(z['centre'][i]);targets.append(mask);domains.append(domain)
        native_weights.append(torch.tensor(weights,dtype=torch.float32,device='cuda:0'))
        views.append((shape,*[torch.tensor(np.array(x),dtype=torch.float32,device='cuda:0') for x in [Ks,Rs,Cs,targets,domains]]))
    context=dr.RasterizeCudaContext();topology=dr.antialias_construct_topology_hash(faces)
    cf=torch.tensor(mesh['F'][b:b+count]-a,dtype=torch.long,device='cuda:0')
    from calibrated_scene_reconstruction.assets.cloth_rest import triangle_edges
    smooth_edges=torch.tensor(triangle_edges(cf.cpu().numpy()),device='cuda:0')
    inverse=torch.tensor(material_inverse(np.array(rest['points']),cf.cpu().numpy()),dtype=torch.float32,device='cuda:0')
    rest_points=np.array(rest['points']);tri=rest_points[cf.cpu().numpy()];area=torch.tensor(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2,dtype=torch.float32,device='cuda:0')
    from .initial_state import policy
    contract=policy(s,obj)
    if not contract or link not in contract['links'] or link not in contract.get('self_clearance_m',{}):raise ValueError('Material validity and physical thickness declarations required')
    from calibrated_scene_reconstruction.scene.objects.blender.contact_field import ContactField
    projection_fields=[ContactField(mesh['V'][part['vertex_start']:part['vertex_start']+part['vertex_count']],mesh['F'][part['face_start']:part['face_start']+part['triangle_count']]-part['vertex_start'],device='cuda:0') for part in parts if part['kind']=='rigid']
    from calibrated_scene_reconstruction.assets.physical_priors import expand_families
    ply=expand_families(s.config('agent/physics_priors.json'))['bodies'][obj+'/'+link]['parameters']['thickness']['nominal']
    # Proposal-only contact barrier; it never creates native attachments.
    from scipy.spatial import cKDTree
    import trimesh
    base=cloth.cpu().numpy();face_np=cf.cpu().numpy();tri=base[face_np]
    material_tri=rest_points[face_np];edge=max(np.linalg.norm(material_tri[:,1]-material_tri[:,0],axis=1).max(),np.linalg.norm(material_tri[:,2]-material_tri[:,0],axis=1).max(),np.linalg.norm(material_tri[:,2]-material_tri[:,1],axis=1).max())
    neighbours_per_node=min(48,len(tri))
    distance,neighbours=cKDTree(tri.mean(1)).query(base,k=neighbours_per_node)
    indices=np.repeat(np.arange(len(base)),neighbours_per_node);targets=neighbours.ravel()
    far=np.linalg.norm(rest_points[indices]-material_tri.mean(1)[targets],axis=1)>3*edge
    indices,targets=indices[far],targets[far]
    closest=trimesh.triangles.closest_point(tri[targets],base[indices]);delta=base[indices]-closest
    gaps=np.linalg.norm(delta,axis=1);keep=(gaps<.0015)&(gaps>1e-8)
    indices,targets,closest,delta,gaps=[x[keep] for x in [indices,targets,closest,delta,gaps]]
    weights=trimesh.triangles.points_to_barycentric(tri[targets],closest)
    np.savez_compressed(out/'proposal_contacts.npz',indices=indices,faces=face_np[targets],weights=weights,normals=delta/gaps[:,None],gaps=gaps)
    contact_i=torch.tensor(indices,device='cuda:0');contact_f=torch.tensor(face_np[targets],device='cuda:0');contact_w=torch.tensor(weights,dtype=torch.float32,device='cuda:0')
    contact_normal=torch.tensor(delta/gaps[:,None],dtype=torch.float32,device='cuda:0')
    from calibrated_scene_reconstruction.assets.cloth_clearance import contact_witnesses
    extra={}
    def update_witnesses(candidate,row,valid_seed):
        examples=row.get('links',{}).get(link,{}).get('self_clearance',{}).get('examples',[])
        for example in examples:
            key=tuple(example['triangles'])
            witness=contact_witnesses(candidate['V'][a:a+n],face_np,np.array([key]))
            extra[key]={k:torch.tensor(v,device='cuda:0',dtype=torch.long if k.startswith('faces') else torch.float32) for k,v in witness.items()}
            extra[key]['clearance_m']=1.1*ply
        # Intersections stop the independent screen before a clearance report.
        # Use the last valid surface to determine the approach side, never
        # infer a separating normal from already-crossed/coincident features.
        crossings=row.get('links',{}).get(link,{}).get('self_intersection',{}).get('example_face_pairs',[])
        for pair in crossings:
            key=tuple(pair)
            witness=contact_witnesses(valid_seed['V'][a:a+n],face_np,np.array([key]))
            extra[key]={k:torch.tensor(v,device='cuda:0',dtype=torch.long if k.startswith('faces') else torch.float32) for k,v in witness.items()}
            extra[key]['clearance_m']=min(1.1*ply,.5*float(witness['gap_m'][0]))
    # Use rejected previous proposals only to discover constraints, never approval.
    def clearance_loss(value):
        state=value[a:a+n];other=(state[contact_f]*contact_w[:,:,None]).sum(1)
        gap=((state[contact_i]-other)*contact_normal).sum(1)
        penalty=(torch.relu((1.1*ply-gap)/ply).square().topk(min(1024,len(gap))).values.mean()
                 if len(gap) else state.sum()*0)
        for witness in extra.values():
            left=(state[witness['faces_a']]*witness['weights_a'][:,:,None]).sum(1)
            right=(state[witness['faces_b']]*witness['weights_b'][:,:,None]).sum(1)
            signed=((left-right)*witness['normal']).sum(1)
            penalty+=torch.relu((witness['clearance_m']-signed)/ply).square().mean()/max(1,len(extra))
        return penalty
    def vertices():
        if mode=='spatial':
            delta=nnf.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=True)[0,:,:,0,0].T
        else:
            delta=(nnf.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=True)[0,:,:,0].T
                   if mode=='field' else field)
        state=cloth+delta
        for _ in range(2):
            for projection in projection_fields:state=projection.project(state,.00001)
        return torch.cat([original[:a],state,original[a+n:]],0)
    def source_loss(value):
        world=value@rotation.T+translation;errors=[];local_errors=[]
        for (shape,K,R,C,target,domain),weights in zip(views,native_weights):
            camera=torch.bmm(world[None]-C[:,None],R);h,w=shape;depth=camera[:,:,2]
            x=2*K[:,None,0,0]*camera[:,:,0]/w+(2*(K[:,None,0,2]+.5)/w-1)*depth
            y=2*K[:,None,1,1]*camera[:,:,1]/h+(2*(K[:,None,1,2]+.5)/h-1)*depth
            clip=torch.stack([x,y,(100.05/99.95)*depth-10./99.95,depth],-1).contiguous()
            raster,_=dr.rasterize(context,clip,faces,shape,grad_db=False)
            prediction,_=dr.interpolate(classes[None].expand(len(K),-1,-1).contiguous(),raster,faces)
            prediction=dr.antialias(prediction,raster,clip,faces,topology_hash=topology)[...,0]
            errors.append(((prediction-target).abs()*domain).sum((1,2))/(target*domain).sum((1,2)).clamp_min(1.))
            from .source_objective import local_part_residual
            local_errors.append(local_part_residual(prediction,target,domain))
        from .source_objective import balanced_part_loss
        return balanced_part_loss(torch.cat(errors))+2*balanced_part_loss(torch.cat(local_errors))
    def material_loss(value):
        p=value[a:a+n][cf];dx=torch.stack([p[:,1]-p[:,0],p[:,2]-p[:,0]],-1);deformation=dx@inverse
        strain=deformation.transpose(1,2)@deformation-torch.eye(2,device='cuda:0')
        return (strain.square().sum((1,2))*area).sum()/area.sum()
    initial=check(mesh,contract,device='cuda:0')
    if not initial['pass']:raise ValueError('Source proposal input fails unchanged material/self/solid checks')
    best=None;minimum_energy=None;accepted=field.detach().clone();accepted_mesh=mesh;trace=[];start=time.time()
    for step in range(steps+1):
        optimizer.zero_grad();value=vertices();image=source_loss(value);metric=material_loss(value)
        smooth=(sum((field.diff(dim=axis)).square().mean()/.005**2 for axis in ([2,3,4] if mode=='spatial' else [2,3])) if mode in {'field','spatial'} else
                (field[smooth_edges[:,0]]-field[smooth_edges[:,1]]).square().mean()/.005**2)
        barrier=clearance_loss(value)
        triangles=value[a:a+n][cf]
        samples=torch.cat([triangles.mean(1),(triangles[:,0]+triangles[:,1])/2,(triangles[:,1]+triangles[:,2])/2,(triangles[:,2]+triangles[:,0])/2])
        solid=sum(torch.relu((.000008-projection(samples))/.00001).square().topk(min(256,len(samples))).values.mean() for projection in projection_fields)
        loss=image+100.*solid+100.*barrier+membrane_weight*metric+.003*smooth+.003*(field/.005).square().mean()
        if step%10==0 or step==steps:
            candidate={**mesh,'V':value.detach().cpu().numpy()};validity=check(candidate,contract,device='cuda:0')
            row={'step':step,'source_loss':float(image.detach()),'material_energy_proxy':float(metric.detach()),'clearance_barrier':float(barrier.detach()),'objective':float(loss.detach()),'admissibility':validity,'extra_contact_witnesses':len(extra)}
            trace.append(row);(out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
            print('VERTEX_PROPOSAL',step,row['source_loss'],row['material_energy_proxy'],validity['pass'],flush=True)
            if validity['pass']:
                accepted=field.detach().clone()
                accepted_mesh=candidate
                if minimum_energy is None or row['material_energy_proxy']<minimum_energy['material_energy_proxy']:
                    minimum_energy=row
                    np.savez_compressed(out/'minimum_energy_proposal.npz',**candidate)
                    (out/'minimum_energy.json').write_text(json.dumps({**row,'proposal_only':True,
                        'task_approved':False,'source_approved':False},indent=2)+'\n')
                if step%50==0 or step==steps:
                    full_metrics,_=evidence.score(candidate['V'],candidate['F'],pose,evidence.load(s,obj,parts=True),model_parts=parts)
                    score=evidence.search_objective(full_metrics)
                    if best is None or (full_metrics['ok'],-score)>(best['full_source_metrics']['ok'],-best['source_score']):
                        best={**row,'source_score':score,'full_source_metrics':full_metrics,'field':field.detach().cpu().numpy().copy(),'V':candidate['V'].copy()}
                        np.savez_compressed(out/'best_proposal.npz',**candidate)
                    print('VERTEX_SOURCE',step,[(x['iou_median'],x['local_difference']) for x in full_metrics['source_parts']],flush=True)
            else:
                update_witnesses(candidate,validity,accepted_mesh)
                thickness_only=not validity['links'][link].get('self_clearance',{'pass':True})['pass']
                with torch.no_grad():field.copy_(accepted)
                optimizer.state.clear()
                for group in optimizer.param_groups:group['lr']*=1. if thickness_only else .95
                continue
        if step==steps:break
        loss.backward();torch.nn.utils.clip_grad_norm_([field],10.);optimizer.step()
        with torch.no_grad():
            norm=torch.linalg.vector_norm(field,dim=1,keepdim=True);field.mul_(torch.clamp(.008/norm.clamp_min(1e-10),max=1.))
    if best is None:raise ValueError('No material/contact-valid proposal')
    mesh['V']=best.pop('V');np.savez_compressed(out/'mesh.npz',**mesh)
    np.savez_compressed(out/'deformation_field.npz',field=best.pop('field'),low=low.cpu().numpy(),high=high.cpu().numpy())
    metrics,_=evidence.score(mesh['V'],mesh['F'],pose,evidence.load(s,obj,parts=True),model_parts=parts)
    (out/'candidate.json').write_text(json.dumps({**info,'path':str(out),'pose':pose,
        'proposal_only':True,'task_approved':False,'requires_agent_rebuild':True},indent=2)+'\n')
    (out/'numeric_result.json').write_text(json.dumps({'metrics':metrics,'best':best,
        'proposal_only':True,'task_approved':False},indent=2)+'\n')
    visual=sheets(s,evidence.placed(mesh['V'],mesh['F'],pose),evidence.load(s,obj,parts=True)[0],parts,out/'source_comparison')
    result={'proposal_only':True,'task_approved':False,'native_constraints_created':False,'source_views_removed':False,
     'best':best,'trace':trace,'duration_s':time.time()-start,'pose':pose,'metrics':metrics,'visual':visual,
     'scope':'Proposal only. Original physical/source gates, all source views and rigid geometry retained. No native constraints, rest metric or material parameter changes. Rebuild with the agent model and run actual native/task gates before acceptance.'}
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VERTEX_PROPOSAL_RESULT',[(r['iou_median'],r['local_difference']) for r in metrics['source_parts']],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('object');parser.add_argument('candidate')
    parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--learning-rate',type=float,default=2e-6)
    parser.add_argument('--membrane-weight',type=float,default=10.)
    parser.add_argument('--mode',choices=['field','nodes','spatial'],default='field')
    args=parser.parse_args()
    run(Session(args.session),args.object,args.candidate,args.observation,args.output,
        steps=args.steps,learning_rate=args.learning_rate,membrane_weight=args.membrane_weight,mode=args.mode)


if __name__=='__main__':main()
