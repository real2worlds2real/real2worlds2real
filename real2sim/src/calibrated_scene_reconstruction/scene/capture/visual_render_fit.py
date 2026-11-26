"""Fit legal articulated CAD directly to current video silhouettes on CUDA.

The input directory is prepared from the current capture and legal robot asset.
Joint-free packages infer articulation; joint-prior packages refine bounded
command errors. Task dimensions and earlier-session fits are never inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .differentiable_robot import TorchModel
from .visual_model_io import load
from .visual_render_seed import resume_hypotheses, root_hypothesis
from .visual_robot_fit import Settings, sample_surfaces


def mesh_arrays(model, names):
    vertices, faces, spans = {}, [], {}
    offset = 0
    for name in names:
        raw = model.triangles[name]
        points, inverse = np.unique(raw.reshape(-1, 3), axis=0, return_inverse=True)
        vertices[name] = points
        faces.append(inverse.reshape(-1, 3)+offset)
        spans[name] = (offset, offset+len(points))
        offset += len(points)
    return vertices, np.concatenate(faces).astype('int32'), spans


def image_rank(ious,local):
    """Keep a candidate satisfying all image criteria before maximizing IoU."""
    if not len(ious) or len(ious)!=len(local):
        raise ValueError('Matching nonempty image and local-difference evidence required')
    iou=float(np.median(ious));difference=float(np.median(local))
    heldout=float(np.median(np.asarray(ious)[::4]))
    violation=max(0.,(.85-iou)/.85)+max(0.,(difference-.02)/.02)+max(0.,(.85-heldout)/.85)
    return violation==0.,-violation,heldout,iou


class Renderer:
    def __init__(self, model, names, data, device='cuda'):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        self.model, self.device = model, device
        self.kinematics = TorchModel(model, names, device=device)
        mesh, faces, self.spans = mesh_arrays(model, names)
        self.mesh = {name:self.kinematics.tensor(values) for name,values in mesh.items()}
        self.faces = torch.as_tensor(faces,device=device,dtype=torch.int32)
        self.context = dr.RasterizeCudaContext(device=device)
        self.topology = dr.antialias_construct_topology_hash(self.faces)
        self.R = self.kinematics.tensor(data['R'])
        self.C = self.kinematics.tensor(data['C'])
        self.K = self.kinematics.tensor(data['K'])
        self.shape = data['masks'].shape[1:]
        self.mask = self.kinematics.tensor(data.get('soft_masks',data['masks']))
        self.valid = self.kinematics.tensor(~data['ignored'])
        limit=float(np.hypot(*self.shape))
        self.distance_out=self.kinematics.tensor(np.stack([
            np.minimum(cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,0),limit)
            for m in data['masks']]))
        self.distance_in=self.kinematics.tensor(np.stack([
            np.minimum(cv2.distanceTransform(m.astype('uint8'),cv2.DIST_L2,0),limit)
            for m in data['masks']]))
        self.sample = {name:self.kinematics.tensor(value) for name,value in
                       sample_surfaces(model,Settings(points_per_link=600)).items() if name in names}
        self.floor = model.floor_height()

    def world(self, q, xy, yaw, log_scale, surfaces):
        torch = self.torch
        source = self.kinematics.points(q,surfaces)
        zero = yaw*0
        R = torch.stack([torch.cos(yaw),-torch.sin(yaw),zero,
                         torch.sin(yaw),torch.cos(yaw),zero,zero,zero,zero+1]).reshape(3,3)
        k = torch.exp(log_scale)
        offset=getattr(self,'height_offset_m',None)
        z=-k*self.floor if offset is None else k*(offset-self.floor)
        return k*source@R.T+torch.cat([xy,z.reshape(1)])

    def render(self, vertices, indices):
        torch = self.torch
        q = torch.matmul((vertices[None]-self.C[indices,None]),self.R[indices])
        K = self.K[indices]
        h,w = self.shape
        z = q[:,:,2]
        # Positive image y and bottom-up raster storage cancel: output tensor
        # row0 corresponds to image row0 without a post-render image flip.
        x = 2*K[:,0,0,None]*q[:,:,0]/w+(2*K[:,0,2,None]/w-1)*z
        y = 2*K[:,1,1,None]*q[:,:,1]/h+(2*K[:,1,2,None]/h-1)*z
        near,far = .01,100.
        depth = ((far+near)/(far-near))*z-2*far*near/(far-near)
        clip = torch.stack([x,y,depth,z],dim=-1).contiguous()
        raster,_ = self.dr.rasterize(self.context,clip,self.faces,self.shape,grad_db=False)
        coverage = (raster[:,:,:,3:4]>0).to(dtype=torch.float32)
        return self.dr.antialias(coverage,raster,clip,self.faces,topology_hash=self.topology)[:,:,:,0]


def run(directory, output, steps=1500, batch=4, seeds=None,
        metric_weight=0., resume_from=None, contour_weight=0.):
    import torch
    from calibrated_scene_reconstruction.scene.objects.object_refine import pieces
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if not 5<=steps<=10000 or not 1<=batch<=8:
        raise ValueError('Render fit needs5..10000 steps and batch1..8')
    if not 0<=contour_weight<=10:raise ValueError('Contour weight must be between0 and10')
    def sha(path):
        value=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
        return value.hexdigest()
    manifest=json.loads((directory/'manifest.json').read_text())
    floor_prior = manifest.get('floor_prior')
    seeds=seeds or manifest.get('seeds',('inferred','cad_home_hypothesis','midpoint_hypothesis'))
    prior=None
    if manifest.get('calibration_mode')=='joint_prior':
        from .joint_prior import seed as prior_seed
        prior=json.loads((directory/'joint_prior.json').read_text())
    fingerprints={**manifest['source_inputs'],**{str(path):sha(path) for path in directory.iterdir() if path.is_file()}}
    fingerprints.update({str(path):sha(path) for path in
                         (Path(__file__),Path(__file__).with_name('differentiable_robot.py'),
                          Path(__file__).with_name('visual_render_seed.py'))})
    for path,expected in fingerprints.items():
        if sha(path)!=expected:raise ValueError('Render-fit source input changed: '+path)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('*.json')):
        raise ValueError('Preserve existing render-fit attempts; prepare a new round')
    model=load(directory)
    data=dict(np.load(directory/'observations.npz'))
    initial=json.loads((directory/'initial.json').read_text())
    resumed=resume_hypotheses(directory,resume_from,data,model) if resume_from else {}
    fingerprints.update({item['path']:item['sha256'] for item in resumed.values()})
    settings=Settings()
    names=list(sample_surfaces(model,settings))
    renderer=Renderer(model,names,data)
    joints=[joint for joint in model.movable if joint.upper>joint.lower and not any(
        re.search(pattern,joint.name,re.I) for pattern in settings.exclude_joints)]
    limits=torch.tensor([[joint.lower,joint.upper] for joint in joints],device='cuda',dtype=torch.float32)
    if prior is not None:
        limits=torch.tensor([[prior['lower_rad'][joint.name],prior['upper_rad'][joint.name]] for joint in joints],
                            device='cuda',dtype=torch.float32)
        nominal=torch.tensor([prior['nominal_rad'][j.name] for j in joints],device='cuda',dtype=torch.float32)
        sigmas=torch.tensor([prior['sigma_rad'][j.name] for j in joints],device='cuda',dtype=torch.float32)
    observed=np.asarray(data['points'],float)
    selected=observed[np.random.default_rng(0).permutation(len(observed))[:settings.max_scene_points]]
    training=selected[np.arange(len(selected))%4!=0][:3000]
    training_tensor=torch.tensor(training,device='cuda',dtype=torch.float32)
    nviews=len(data['names'])
    if nviews<8:raise ValueError('At least8 current-view silhouette observations required')
    all_indices=list(range(nviews))
    # Hold out every fourth actual camera for pixel validation; sparse-map
    # validation tracks are also excluded from the 3D fitting objective.
    train_indices=[i for i in all_indices if i%4!=0]
    reference_area=(renderer.mask[train_indices]*renderer.valid[train_indices]).sum((1,2)).mean().clamp(min=1)
    results=[]
    for seed in seeds:
        start=time.monotonic()
        starting=resumed[seed]['candidate'] if seed in resumed else initial
        state=copy.deepcopy(starting['estimated_joints_rad'])
        seed_source='coarse3D current-capture fit'
        if prior is not None and seed not in resumed:
            state=prior_seed(prior,seed)
            seed_source='current command plus bounded execution uncertainty; not measured joint truth'
        elif prior is None and not starting.get('ok') and not starting.get('render_fit'):
            # An invalid 3D seed may fold the torso to explain a partial surface.
            # Neutral CAD is another hypothesis, never a measured joint state.
            state={j.name:float(np.clip(0.,j.lower,j.upper)) for j in joints}
            seed_source='neutral CAD fallback after rejected coarse3D initialization'
        if prior is None and seed not in resumed:
            from .visual_render_seed import visual_seed
            state=visual_seed(state,initial['cad_home_hypothesis'],seed)
        state={j.name:float(np.clip(state[j.name],j.lower,j.upper)) for j in joints}
        lineage_start=(starting['render_fit'].get('articulation_lineage_start',state)
                       if seed in resumed else state.copy())
        if seed in resumed:seed_source='same-capture independent rendered lineage; no cross-session seed'
        renderer.height_offset_m=(renderer.kinematics.tensor(starting.get('floor_offset_m',0.))
                                  if prior is not None or floor_prior is not None else None)
        root=root_hypothesis(renderer,state,training,train_indices,starting)
        print(f'[robot_render_fit] {seed} broad root IoU{root["training_iou"]:.4f}, '
              f'scale{root["scale"]:.4f}, yaw{np.degrees(root["yaw"]):.1f}',flush=True)
        k0=root['scale']
        xy=torch.nn.Parameter(torch.tensor(root['xy'],device='cuda',dtype=torch.float32))
        yaw=torch.nn.Parameter(torch.tensor(root['yaw'],device='cuda',dtype=torch.float32))
        log_scale=torch.nn.Parameter(torch.tensor(np.log(k0),device='cuda',dtype=torch.float32))
        angles=torch.nn.Parameter(torch.tensor([state[j.name] for j in joints],device='cuda',dtype=torch.float32))
        if prior is not None and seed=='joint_fixed':angles.requires_grad_(False)
        optimizer=torch.optim.Adam([{'params':[xy],'lr':.003*k0},{'params':[yaw],'lr':.005},
                                    {'params':[log_scale],'lr':.0015},{'params':[angles],'lr':.008}])
        height_offset=None
        if prior is not None or floor_prior is not None:
            height_offset=torch.nn.Parameter(torch.tensor(starting.get('floor_offset_m',0.),device='cuda',dtype=torch.float32))
            renderer.height_offset_m=height_offset
            optimizer.add_param_group({'params':[height_offset],'lr':.0005})
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps,eta_min=0.)
        with torch.no_grad():angles.clamp_(limits[:,0],limits[:,1])
        def evaluate(vertices):
            ious=[];local=[]
            for first in range(0,nviews,batch):
                ids=all_indices[first:first+batch]
                predictions=(renderer.render(vertices,ids)>.5).cpu().numpy()
                for index,p in zip(ids,predictions):
                    valid=~data['ignored'][index];actual=data['masks'][index]&valid;p=p&valid
                    ious.append(float((actual&p).sum()/max(1,(actual|p).sum())))
                    local.append(float(sum(item[0] for item in pieces(actual,p,.003))))
            return ious,local
        with torch.no_grad():
            q={j.name:value for j,value in zip(joints,angles.unbind())}
            baseline,baseline_local=evaluate(renderer.world(q,xy,yaw,log_scale,renderer.mesh))
            best={'step':0,'heldout_iou':float(np.median(np.array(baseline)[::4])),
                  'xy':xy.detach().cpu().tolist(),'yaw':float(yaw),
                  'angles':angles.detach().cpu().tolist(),'log_scale':float(log_scale),
                  'height_offset_m':float(height_offset) if height_offset is not None else 0.,
                  'rank':image_rank(baseline,baseline_local)}
            print(f'[robot_render_fit] {seed} baseline IoU{np.median(baseline):.4f}',flush=True)
        trace=[]
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            q={j.name:value for j,value in zip(joints,angles.unbind())}
            vertices=renderer.world(q,xy,yaw,log_scale,renderer.mesh)
            if metric_weight:
                sample=renderer.world(q,xy,yaw,log_scale,renderer.sample)
                if step%10==0:
                    points=sample.detach().cpu().numpy()
                    dist,nearest=cKDTree(points).query(training)
                    keep_count=max(100,int(.75*len(training)))
                    chosen=np.argsort(dist)[:keep_count]
                    obs_ids=torch.tensor(chosen,device='cuda',dtype=torch.long)
                    source_ids=torch.tensor(nearest[chosen],device='cuda',dtype=torch.long)
            indices=[train_indices[(step*batch+i)%len(train_indices)] for i in range(min(batch,len(train_indices)))]
            predicted=renderer.render(vertices,indices)
            mask=renderer.mask[indices];valid=renderer.valid[indices]
            intersection=(predicted*mask*valid).sum((1,2))
            union=((predicted+mask-predicted*mask)*valid).sum((1,2)).clamp(min=1)
            area=(mask*valid).sum((1,2))
            image_loss=((1-intersection/union)*area).mean()/reference_area
            # A signed distance field supplies a direction even where target
            # and CAD contours do not overlap. Normalize by observed extent,
            # so the same objective works across image scales and captures.
            distance_error=((predicted*renderer.distance_out[indices]+
                             (1-predicted)*renderer.distance_in[indices])*valid).sum((1,2))
            contour_loss=(distance_error/area.clamp(min=1).sqrt()).mean()/reference_area
            metric_loss=angles.sum()*0
            if metric_weight:
                residual=(sample[source_ids]-training_tensor[obs_ids])/k0
                metric_loss=torch.nn.functional.smooth_l1_loss(residual,torch.zeros_like(residual),beta=.01)
            prior_loss=angles.sum()*0
            if prior is not None:
                standardized=(angles-nominal)/sigmas
                prior_loss=(torch.sqrt(1+standardized**2)-1).mean()
                prior_loss=prior_loss+.5*(height_offset/prior['floor_sigma_m'])**2
            loss=image_loss+contour_weight*contour_loss+metric_weight*metric_loss/.02
            if prior is not None:loss=loss+prior['weight']*prior_loss
            if floor_prior is not None:
                floor_residual=torch.exp(log_scale)*height_offset/floor_prior['sigma_map']
                loss=loss+floor_prior['weight']*.5*floor_residual**2
            loss.backward();optimizer.step();scheduler.step()
            with torch.no_grad():
                angles.clamp_(limits[:,0],limits[:,1])
                log_scale.clamp_(np.log(k0*.6),np.log(k0*1.6))
                if height_offset is not None:
                    limit = (prior['max_floor_offset_m'] if prior is not None else
                             floor_prior['max_offset_map']/float(torch.exp(log_scale)))
                    height_offset.clamp_(-limit,limit)
                centre=renderer.kinematics.tensor(root['xy'])
                xy.copy_(torch.maximum(centre-.5*k0,torch.minimum(centre+.5*k0,xy)))
            if (step+1)%100==0 or step==steps-1:
                with torch.no_grad():
                    q={j.name:value for j,value in zip(joints,angles.unbind())}
                    current=renderer.world(q,xy,yaw,log_scale,renderer.mesh)
                    ious,local=evaluate(current)
                    validation_iou=float(np.median(np.array(ious)[::4]))
                    row={'step':step+1,'train_loss':float(loss.detach()),
                         'image_loss':float(image_loss.detach()),'metric_loss':float(metric_loss.detach()),
                         'contour_loss':float(contour_loss.detach()),
                         'joint_prior_loss':float(prior_loss.detach()),
                         'median_iou':float(np.median(ious)),
                         'median_local':float(np.median(local)),
                         'heldout_iou':validation_iou,'scale':float(torch.exp(log_scale))}
                    trace.append(row)
                    rank=image_rank(ious,local)
                    if rank>best['rank']:
                        best={**row,'xy':xy.detach().cpu().tolist(),'yaw':float(yaw),
                              'angles':angles.detach().cpu().tolist(),'log_scale':float(log_scale),'rank':rank,
                              'height_offset_m':float(height_offset) if height_offset is not None else 0.}
                    print(f'[robot_render_fit] {seed} {step+1}/{steps}: IoU{row["median_iou"]:.4f}, '
                          f'heldout{validation_iou:.4f}, local{row["median_local"]:.4f}, scale{row["scale"]:.4f}',flush=True)
        with torch.no_grad():
            xy.copy_(renderer.kinematics.tensor(best['xy']));yaw.copy_(renderer.kinematics.tensor(best['yaw']))
            angles.copy_(renderer.kinematics.tensor(best['angles']));log_scale.copy_(renderer.kinematics.tensor(best['log_scale']))
            if height_offset is not None:height_offset.copy_(renderer.kinematics.tensor(best['height_offset_m']))
            q={j.name:value for j,value in zip(joints,angles.unbind())}
            vertices=renderer.world(q,xy,yaw,log_scale,renderer.mesh)
            masks=[]
            for first in range(0,nviews,batch):
                masks.extend((renderer.render(vertices,all_indices[first:first+batch])>.5).cpu().numpy())
        mask_array=np.array(masks)
        np.savez_compressed(output/(seed+'_masks.npz'),names=data['names'],predicted=mask_array)
        ious=[];local=[];tiles=[]
        for i,p in enumerate(mask_array):
            valid=~data['ignored'][i];actual=data['masks'][i]&valid;p=p&valid
            ious.append(float((actual&p).sum()/max(1,(actual|p).sum())))
            local.append(float(sum(item[0] for item in pieces(actual,p,.003))))
            image=np.zeros((*p.shape,3),'uint8');image[actual&p]=(225,225,225)
            image[actual&~p]=(45,45,230);image[p&~actual]=(40,220,40)
            cv2.putText(image,str(data['names'][i])+f' {ious[-1]:.3f}',(4,20),0,.45,(0,255,255),1)
            tiles.append(cv2.resize(image,(320,320)))
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(output/(seed+'_evidence.jpg')),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
        k=float(np.exp(best['log_scale']))
        result={**initial,'seed':seed,'applied':False,'ok':False,
                'reasons':['Rendered candidate awaits metric, ambiguity and source verification before application'],
                'map_units_per_metre':k,'metres_per_map_unit':1/k,
                'base_xy_map':best['xy'],'base_z_map':float(k*(-renderer.floor+best['height_offset_m'])),
                'floor_offset_m':best['height_offset_m'],
                'floor_uncertainty': floor_prior,
                'yaw_deg':float(np.degrees(best['yaw'])),
                'estimated_joints_rad':{joint.name:float(value) for joint,value in zip(joints,best['angles'])},
                'render_fit':{'backend':'nvdiffrast actual CAD triangles','trace':trace,'ious':ious,'local':local,
                              'iou_median':float(np.median(ious)),'local_median':float(np.median(local)),
                              'heldout_iou':float(np.median(np.array(ious)[::4])),
                              'training_views':[str(data['names'][i]) for i in train_indices],
                              'heldout_views':[str(data['names'][i]) for i in all_indices[::4]],
                              'seconds':time.monotonic()-start}}
        result['render_fit'].update(mask_sha256=sha(output/(seed+'_masks.npz')),
                                    evidence_sha256=sha(output/(seed+'_evidence.jpg')),input_sha256=fingerprints,torch_version=torch.__version__,
                                    cuda_version=torch.version.cuda,steps=steps,batch=batch,
                                    metric_weight=metric_weight,best_step=best['step'],
                                    contour_weight=contour_weight,
                                    articulation_lineage_start=lineage_start,
                                    root_search=root,articulation_seed_source=seed_source,
                                    image_loss_weighting='valid robot pixels across all training cameras')
        result['render_fit']['checkpoint_selection']='minimize unchanged complete-image criterion violations, then maximize held-out IoU; initial pose retained'
        for path,expected in fingerprints.items():
            if sha(path)!=expected:raise ValueError('Render-fit input changed during optimization: '+path)
        (output/(seed+'.json')).write_text(json.dumps(result,indent=2)+'\n');results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs');parser.add_argument('output')
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--seeds',nargs='+',default=None)
    parser.add_argument('--resume-from',help='previous fit directory from the same unchanged capture')
    parser.add_argument('--metric-weight',type=float,default=0.,help='optional3D training weight; default reserves3D for verification')
    parser.add_argument('--contour-weight',type=float,default=0.,help='experimental capture-independent distance-field loss weight')
    args=parser.parse_args()
    run(args.inputs,args.output,args.steps,args.batch,args.seeds,args.metric_weight,args.resume_from,args.contour_weight)


if __name__=='__main__':main()
