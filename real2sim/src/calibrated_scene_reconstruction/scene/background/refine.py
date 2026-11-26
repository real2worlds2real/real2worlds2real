"""Fit agent entity-list geometry proposals to same-capture silhouettes, never size truth."""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.optimize import least_squares,minimize

from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background import occluded_edges


def rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def bounds(vertices,K,R,C,size):
    p=(vertices-C)@R
    if np.any(p[:,2]<.05):return None
    uv=p@K.T;uv=uv[:,:2]/uv[:,2,None]
    lo,hi=uv.min(0),uv.max(0)
    return np.r_[np.maximum(lo,0),np.minimum(hi,size)]


def part_indices(parts, entity, selection):
    """A panel-only mask must constrain named panel meshes, not its stand."""
    available = {row.get('part') for row in parts if row['entity'] == entity}
    if selection is not None and (not isinstance(selection, list) or not selection
            or len(set(selection)) != len(selection) or not set(selection) <= available):
        raise ValueError('Distinct exported mesh part names required for a partial silhouette')
    return [i for i, row in enumerate(parts) if row['entity'] == entity
            and (selection is None or row.get('part') in selection)]


def surface_outside(points, position, yaw, size):
    """Metric source points must remain on a fitted horizontal support."""
    local=(np.asarray(points)[:,:2]-np.asarray(position)[:2])@rotation(yaw)[:2,:2]
    return np.maximum(abs(local)-np.asarray(size)[:2]/2-.02,0)


def enclose_surface_samples(samples, position, yaw, size):
    """Complete hidden edges to include independent source-point lower bounds.

    Keep at least 90% of qualified GS and 95% of sparse tracks. The full lists
    remain in evidence and the resulting coverage is reported, never erased.
    """
    result=np.asarray(size,float).copy()
    for key,points in samples.items():
        keep=int(np.ceil(len(points)*(.9 if key=='gs' else .95)))
        distance=np.linalg.norm(surface_outside(points,position,yaw,size),axis=1)
        selected=points[np.argsort(distance,kind='stable')[:keep]]
        local=(selected[:,:2]-np.asarray(position)[:2])@rotation(yaw)[:2,:2]
        result[:2]=np.maximum(result[:2],2*np.maximum(0,abs(local).max(0)-.018))
    return result


def run(session, observation_file, geometry_job, output, entities=None, yaw_hypotheses=12, lock=(), part_selection=None, support_surfaces=None, front_hypotheses=None):
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Refinement proposals belong in this session')
    out.mkdir(parents=True,exist_ok=False)
    cfgpath=session.config_dir/'environment.json';cfg=json.loads(cfgpath.read_text())
    observations=json.loads(Path(observation_file).read_text())['records'];job=json.loads(Path(geometry_job).read_text())
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    verify(session)
    native_pointer=session/'scene/environment/twin/current.json'
    if job.get('geometry_sha256')!=digest(job['geometry']) or job.get('native_manifest_sha256')!=digest(native_pointer):
        raise ValueError('Refinement requires geometry bound to the current native build')
    mesh=np.load(job['geometry']);camerapath=session/'parsing/posed_frames/cameras.npz';camera=np.load(camerapath);names=list(map(str,camera['names']))
    # Applying a proposal changes the active config and native pointer. Preserve
    # the actual inputs in the proposal, rather than making its own successful
    # application invalidate the historical source evidence.
    mutable_inputs={str(p.resolve()):digest(p) for p in [cfgpath,native_pointer]}
    config_snapshot=out/'environment_input.json';config_snapshot.write_bytes(cfgpath.read_bytes())
    native_snapshot=out/'native_input.json';native_snapshot.write_bytes(native_pointer.read_bytes())
    inputs={str(p.resolve()):digest(p) for p in [config_snapshot,native_snapshot,Path(observation_file),Path(geometry_job),Path(job['geometry']),camerapath,Path(__file__)]}
    inputs[str(Path(occluded_edges.__file__).resolve())] = digest(occluded_edges.__file__)
    if front_hypotheses is not None:
        from calibrated_scene_reconstruction.scene.background import front_hypotheses as source_directions
        hypothesis = Path(front_hypotheses).resolve(); request = json.loads(hypothesis.read_text())
        checks, source_inputs = source_directions.validate(request,
            set(entities or [e['id'] for e in cfg['parameters']['entities']]), job['parts'], names, session/'splat/data/images')
        owners = {c['entity'] for c in checks}
        cfg['front_view_checks'] = [c for c in cfg.get('front_view_checks', []) if c['entity'] not in owners]+checks
        snapshot = out/'front_hypothesis.json'; snapshot.write_bytes(hypothesis.read_bytes())
        inputs.update(source_inputs)
        for file in (snapshot, Path(source_directions.__file__).resolve()):
            inputs[str(file)] = digest(file)
    try:
        native_source=Path(json.loads(native_snapshot.read_text())['path'])
    except (json.JSONDecodeError,KeyError,TypeError):
        native_source=None
    if native_source is not None:
        inputs[str(native_source.resolve())]=digest(native_source)
    floor=cfg['physics']['ground_height'];results=[]
    if type(yaw_hypotheses)is not int or not 1<=yaw_hypotheses<=48:raise ValueError('Use1..48 orientation hypotheses')
    fields=['x','y','yaw','width','depth','height']
    if len(set(lock))!=len(lock) or not set(lock)<=set(fields):raise ValueError('Known distinct refinement coordinates required')
    locked=[fields.index(k) for k in lock]
    part_selection = part_selection or {}
    support_surfaces = support_surfaces or {}
    if not isinstance(support_surfaces,dict) or not set(support_surfaces)<={e['id'] for e in cfg['parameters']['entities']}:
        raise ValueError('Explicit background support-to-source-evidence declarations required')
    if not isinstance(part_selection, dict) or not set(part_selection) <= {e['id'] for e in cfg['parameters']['entities']}:
        raise ValueError('Partial silhouettes require explicit background entity-to-parts declarations')
    for entity in cfg['parameters']['entities']:
        eid=entity['id']
        if entities and eid not in entities:continue
        selected = part_indices(job['parts'], eid, part_selection.get(eid))
        parts=[mesh[f'v{i}'] for i in selected]
        if not parts or entity['kind'] not in ('desk','cabinet','cabinet_open','chair','stool','sofa','monitor','laptop','partition','box','open_box'):continue
        anchor=floor;support=None
        if entity['kind'] in ('monitor','laptop'):
            supports=cfg['physics']['entities'].get(eid,{}).get('on',[])
            if len(supports)!=1:continue
            support=supports[0];support_parts=[mesh[f'v{i}'] for i,row in enumerate(job['parts']) if row['entity']==support]
            if not support_parts:continue
            anchor=float(np.concatenate(support_parts)[:,2].max())
        if abs(entity['position'][2]-anchor)>.03:continue
        records=list({r['frame']:r for r in observations if r['entity']==eid and r.get('mask')}.values());data=[];image_masks=[]
        hidden_edges = {r['frame']: occluded_edges.hidden(r) for r in records}
        partial_frames={frame for frame, flags in hidden_edges.items() if flags.any()}
        for record in records:
            from calibrated_scene_reconstruction.scene.background.observations import geometry_review
            if 'partially_occluded' in record and (type(record['partially_occluded'])is not bool or eid not in part_selection):
                raise ValueError('Part occlusion is an explicit boolean for a named-part silhouette')
            if eid in part_selection:
                if (record.get('geometry_fit_scope') != 'surface_part'
                        or not isinstance(record.get('geometry_fit_review'), str)
                        or not record['geometry_fit_review'].strip()):
                    raise ValueError('Named-part fitting requires a reviewed surface-part silhouette')
            else:
                geometry_review(record)
            if digest(record['mask'])!=record['mask_sha256']:raise ValueError('Observed mask changed')
            if record.get('source'):
                if digest(record['source'])!=record['source_sha256']:raise ValueError('Observed source changed')
                inputs[str(Path(record['source']).resolve())]=record['source_sha256']
            mask=np.array(Image.open(record['mask']))>128
            yy,xx=np.nonzero(mask)
            if len(xx)<1000:continue
            i=names.index(record['frame']);h,w=mask.shape
            target=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],float)
            edge=max(12,w*.012)
            weights=np.array([target[0]>edge,target[1]>edge,target[2]<w-edge,target[3]<h-edge],float)
            if weights.sum()<2:continue
            data.append((i,target,weights,np.array([w,h,w,h]),record['frame']))
            image_masks.append(cv2.resize(mask.astype('float32'),(w//4,h//4),interpolation=cv2.INTER_AREA)>.5)
            inputs[str(Path(record['mask']).resolve())]=digest(record['mask'])
        if len(data)<2:continue
        vertices=np.concatenate(parts);origin=np.array(entity['position'],float);yaw=np.radians(entity.get('yaw_deg',0))
        metric_samples={}
        if eid in support_surfaces:
            from calibrated_scene_reconstruction.scene.review.environment import protected_state
            evidence=Path(support_surfaces[eid]).resolve();surface=json.loads(evidence.read_text())
            samples=Path(surface['samples']).resolve()
            if (surface['entity']!=eid or surface['protected_task_state']!=protected_state(session)
                    or not samples.is_relative_to(session.dir.resolve()) or digest(samples)!=surface['samples_sha256']
                    or abs(surface['height_m']-(origin[2]+entity['size'][2]))>.03):
                raise ValueError('Support points must bind the current background plane')
            for path,sha in surface['input_sha256'].items():
                if digest(path)!=sha:raise ValueError('Source support evidence changed')
            inputs.update(surface['input_sha256']);inputs[str(evidence)]=digest(evidence);inputs[str(samples)]=digest(samples)
            cloud=np.load(samples)
            for key,minimum in [('gs',10),('sparse',3)]:
                points=np.asarray(cloud[key],float)
                if points.ndim!=2 or points.shape[1]!=3 or len(points)<minimum or not np.isfinite(points).all():
                    raise ValueError('Finite independent source surface points required')
                metric_samples[key]=points
        local=(vertices-origin)@rotation(yaw)
        front_evidence=[]
        for spec in cfg.get('front_view_checks',[]):
            if spec['entity']!=eid:continue
            def centre_of_parts(required):
                selected_parts=[i for i,p in enumerate(job['parts']) if p['entity']==eid and (required is None or p['part'] in required)]
                if required is not None and set(required)!={job['parts'][i]['part'] for i in selected_parts}:
                    raise ValueError('Front orientation fitting requires all declared exported parts')
                vv=np.concatenate([mesh[f'v{i}'] for i in selected_parts]);return (((vv.min(0)+vv.max(0))/2)-origin)@rotation(yaw)
            face=centre_of_parts(spec['front_parts']);rear=centre_of_parts(spec.get('reference_parts'))
            axis=np.asarray(spec['local_front'],float)
            if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-6:raise ValueError('Finite nonzero front axis required')
            if len(set(spec['frames']))<2:raise ValueError('Two source front views required')
            for frame in spec['frames']:
                # Bind the same original pictures used by the declared build check.
                path=session/'splat/data/images'/frame
                inputs[str(path.resolve())]=digest(path)
                front_evidence.append((axis,face,rear,camera['centre'][names.index(frame)],spec.get('minimum_facing_cosine',.1)))
        def front_valid(x):
            from calibrated_scene_reconstruction.scene.background.front_views import facing
            turn=rotation(yaw+x[2]);scale=np.exp(x[3:6]);translation=origin+[x[0],x[1],0]
            return all((lambda result:result[0]>1e-4 and result[1]>=limit)(
                facing((axis/scale)@turn.T,rear*scale@turn.T+translation,face*scale@turn.T+translation,view))
                for axis,face,rear,view,limit in front_evidence)
        local_full=local.copy();faces=[];offset=0
        for i,row in enumerate(job['parts']):
            if i in selected:
                faces.append(mesh[f'f{i}']+offset);offset+=len(mesh[f'v{i}'])
        faces=np.concatenate(faces)
        local=local[::max(1,len(local)//500)]
        def transformed(x):return local*np.exp(x[3:6])@rotation(yaw+x[2]).T+origin+[x[0],x[1],0]
        def residual(x,prior=True):
            v=transformed(x);values=[]
            for i,target,weights,size,frame in data:
                predicted=bounds(v,camera['K'][i],camera['R_world_cam'][i],camera['centre'][i],size[:2])
                if predicted is None:values.extend([2.]*4);continue
                delta=(predicted-target)/size
                # Visible masks are lower bounds when partially occluded. Penalize
                # undershoot strongly; modest overshoot is allowed for hidden feet.
                # Explicitly occluded masks are lower bounds. Unknown pixels
                # behind another object cannot shrink the complete body.
                factor=occluded_edges.residual_factor(delta,hidden_edges[frame])
                values.extend(delta*weights*factor)
            if prior:values.extend(np.r_[x[:2]*.015,np.arctan2(np.sin(x[2]),np.cos(x[2]))*.001,x[3:6]*.012])
            ratio=(entity.get('shape_prior') or {}).get('width_depth_ratio')
            if prior and ratio:
                current=entity['size'][0]*np.exp(x[3])/(entity['size'][1]*np.exp(x[4]))
                values.extend([max(0,ratio[0]-current)*.1,max(0,current-ratio[1])*.1])
            for points in metric_samples.values():
                # A mask omits areas covered by equipment. Those holes cannot
                # justify shrinking the table past independent observed points.
                outside=surface_outside(points,origin+[x[0],x[1],0],yaw+x[2],np.asarray(entity['size'])*np.exp(x[3:6]))
                values.extend((outside*.6/np.sqrt(len(points))).ravel())
            return values
        def mask_score(x):
            from calibrated_scene_reconstruction.core.geometry.raster import triangle_mask
            V=local_full*np.exp(x[3:6])@rotation(yaw+x[2]).T+origin+[x[0],x[1],0]
            costs=[];ious=[];weights=[]
            for (i,_,_,_,frame),observed in zip(data,image_masks):
                pc=(V-camera['centre'][i])@camera['R_world_cam'][i];uv=pc@camera['K'][i].T
                predicted=triangle_mask(uv[:,:2]/np.maximum(pc[:,2,None],1e-9)/4,faces,observed.shape)
                area=max(1,int(observed.sum()));intersection=int((observed&predicted).sum())
                outside_weight=0. if frame in partial_frames else .35
                costs.append(((observed&~predicted).sum()+outside_weight*(predicted&~observed).sum())/area)
                ious.append(intersection/max(1,int((observed|predicted).sum())));weights.append(area)
            return float(np.average(costs,weights=weights)),ious
        initial=np.zeros(6);before=float(np.mean(np.square(residual(initial,False))))
        # A wrong PCA axis must not be repaired by shrinking width and inflating
        # depth. Search full-circle orientations before selecting a proposal.
        candidates=[]
        for seed in np.linspace(-np.pi,np.pi,yaw_hypotheses,endpoint=False) if yaw_hypotheses>1 and 'yaw' not in lock else [0.]:
            x0=initial.copy();x0[2]=seed;half=np.pi/yaw_hypotheses if yaw_hypotheses>1 else .55
            lo=np.array([-.7,-.7,seed-half,*([np.log(.35)]*3)]);hi=np.array([.7,.7,seed+half,*([np.log(1.8)]*3)])
            lo[locked]=-1e-12;hi[locked]=1e-12
            proposal=least_squares(residual,x0,bounds=(lo,hi),
                          loss='soft_l1',f_scale=.02,max_nfev=120)
            candidates.append(proposal)
        candidate_masks=[mask_score(f.x) for f in candidates]
        eligible=[i for i,f in enumerate(candidates) if front_valid(f.x)]
        if not eligible:raise ValueError('No orientation hypothesis faces all declared source front views: '+eid)
        chosen=min(eligible,key=lambda i:np.mean(np.square(residual(candidates[i].x)))+.005*candidate_masks[i][0])
        fit=candidates[chosen]
        # Refine across yaw partition boundaries against actual triangle masks.
        lower=np.array([-.7,-.7,fit.x[2]-np.pi/8,*([np.log(.35)]*3)])
        upper=np.array([.7,.7,fit.x[2]+np.pi/8,*([np.log(1.8)]*3)])
        lower[locked]=-1e-12;upper[locked]=1e-12
        def full_cost(x):return float(np.mean(np.square(residual(x)))+.005*mask_score(x)[0]) if front_valid(x) else 1e6
        refined=minimize(full_cost,fit.x,method='Powell',bounds=list(zip(lower,upper)),
                         options={'maxfev':350,'maxiter':20,'xtol':.002,'ftol':1e-4})
        if full_cost(refined.x)<full_cost(fit.x):fit.x=refined.x
        if metric_samples:
            completed=enclose_surface_samples(metric_samples,origin+[fit.x[0],fit.x[1],0],
                                              yaw+fit.x[2],np.asarray(entity['size'])*np.exp(fit.x[3:6]))
            fit.x[3:5]=np.minimum(upper[3:5],np.maximum(fit.x[3:5],np.log(completed[:2]/np.asarray(entity['size'][:2]))))
        fit.active_mask=np.isclose(fit.x,lower,atol=1e-4)|np.isclose(fit.x,upper,atol=1e-4)
        fit.active_mask[locked]=False
        after=float(np.mean(np.square(residual(fit.x,False))))
        position=origin+[fit.x[0],fit.x[1],0];size=np.asarray(entity['size'])*np.exp(fit.x[3:6])
        results.append({'entity':eid,'frames':[row[-1] for row in data],'before':before,'after':after,
            'position':position.tolist(),'size':size.tolist(),'yaw_deg':float(np.degrees(yaw+fit.x[2])),
            'bound_hit':bool(np.any(np.abs(fit.active_mask)>0)),'approved':False,
            'support':support,'anchor_height_m':anchor,'locked_coordinates':list(lock),
            'front_back_approved':False})
        results[-1]['declared_front_views_pass']=bool(front_evidence and front_valid(fit.x))
        results[-1]['fitted_parts'] = part_selection.get(eid)
        if front_hypotheses is not None:
            results[-1]['source_front_hypothesis'] = str(out/'front_hypothesis.json')
        results[-1]['occluded_edges'] = {frame: [edge for edge, flag in zip(occluded_edges.EDGES, hidden_edges[frame]) if flag]
                                         for *_, frame in data}
        results[-1]['whole_object_geometry_approved'] = False
        if metric_samples:
            coverage={key:float(np.mean(np.max(surface_outside(points,position,yaw+fit.x[2],size),axis=1)<1e-5))
                      for key,points in metric_samples.items()}
            results[-1]['source_surface_coverage']=coverage
            results[-1]['source_surface_pass']=coverage['gs']>=.9 and coverage['sparse']>=.95
        results[-1]['mask_iou_before']=mask_score(initial)[1];results[-1]['mask_iou_after']=mask_score(fit.x)[1]
        results[-1]['orientation_candidates']=[{'yaw_deg':float(np.degrees(yaw+f.x[2])),'cost':float(np.mean(np.square(residual(f.x)))),
            'mask_cost':candidate_masks[i][0],'declared_front_views_pass':bool(front_evidence and front_valid(f.x)),
            'size':(np.asarray(entity['size'])*np.exp(f.x[3:6])).tolist()} for i,f in enumerate(candidates)]
        print('[environment.refine]',eid,len(data),'views',round(before,5),'->',round(after,5),flush=True)
    for path,expected in {**inputs,**mutable_inputs}.items():
        if digest(path)!=expected:raise ValueError('Refinement source changed: '+path)
    (out/'proposals.json').write_text(json.dumps({'version':1,'results':results,'input_sha256':inputs,
        'scope':'Unapproved geometry proposals from calibrated image masks. Bounding-image agreement is not photorealism, support stability or actual-size accuracy. Review masks and render after applying.'},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session');parser.add_argument('observations');parser.add_argument('geometry_job');parser.add_argument('--output',required=True)
    parser.add_argument('--entities',nargs='+');parser.add_argument('--yaw-hypotheses',type=int,default=12)
    parser.add_argument('--parts-json',help='Entity-to-exported-mesh-parts JSON for reviewed surface-part masks; never a whole-object approval')
    parser.add_argument('--support-surfaces-json',help='Background support-to-source-surface-proposal JSON; preserve independent GS/SfM surface points')
    parser.add_argument('--front-hypotheses',help='Source-bound direction hypothesis for selected owners; does not change the active native configuration')
    parser.add_argument('--lock',nargs='+',default=[],choices=['x','y','yaw','width','depth','height'],help='Hold current hypotheses fixed for a staged fit; these are not measured dimensions')
    a=parser.parse_args();run(Session(a.session),a.observations,a.geometry_job,a.output,a.entities,a.yaw_hypotheses,a.lock,
                             json.loads(Path(a.parts_json).read_text()) if a.parts_json else None,
                             json.loads(Path(a.support_surfaces_json).read_text()) if a.support_surfaces_json else None,a.front_hypotheses)


if __name__=='__main__':main()
