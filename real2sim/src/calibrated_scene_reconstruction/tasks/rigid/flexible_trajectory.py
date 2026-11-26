"""Independent geometry checks of recorded native mixed-assembly surfaces.

Rigid poses never stand in for flexible nodes. Source indices, files and
sampling intervals are checked; reported clearance is a sampled screen, not a
claim of continuous collision detection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def sample_indices(times,maximum_step_s):
    t=np.asarray(times,float)
    if (t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or (np.diff(t)<=0).any()
            or not np.isfinite(maximum_step_s) or maximum_step_s<=0
            or np.diff(t).max()>maximum_step_s+1e-9):
        raise ValueError('Complete increasing native trace within the declared temporal budget required')
    ids=[0]
    while ids[-1]<len(t)-1:
        end=min(len(t)-1,int(np.searchsorted(t,t[ids[-1]]+maximum_step_s+1e-10,side='right'))-1)
        if end<=ids[-1]:raise ValueError('Native sampling budget cannot be met')
        ids.append(end)
    return np.asarray(ids)


def measure(ctx,native,output,rigid_parts,poses,times,*,maximum_step_s,tolerance_m,root_tolerance_m,device='cuda:0'):
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    from calibrated_scene_reconstruction.assets.physical_priors import expand_families
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface,triangle_samples
    from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
    from calibrated_scene_reconstruction.assets.cloth_clearance import measure as clearance
    from .assembly_measurements import rotations
    report=native.get('flexible_trace');output=Path(output)
    expected={n for n in getattr(ctx,'_cloth',{}) if n.split('/')[0] in {p['name'].split('/')[0] for p in rigid_parts}}
    if not expected:
        if report:raise ValueError('Unexpected flexible trace')
        return {'pass':True,'applicable':False}
    if not report or set(report['names'])!=expected:raise ValueError('Every actual flexible body needs a native trace')
    source=output/report['positions']
    with source.open('rb') as f:actual_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if actual_hash!=report['positions_sha256']:raise ValueError('Native surface trace changed')
    nodes=np.load(source,mmap_mode='r');steps=np.asarray(report['sample_steps']);times=np.asarray(times)
    if len(nodes)!=len(steps) or steps[0]!=0 or steps[-1]!=len(times)-1:
        raise ValueError('Native surface trace must span the complete rigid assay')
    selected=sample_indices(times[steps],maximum_step_s);R=rotations(poses)
    rigid_ids={p['name']:i for i,p in enumerate(rigid_parts)}
    solids={p['name']:SignedSurface(p['V'],p['F'],device=device) for p in rigid_parts}
    priors=expand_families(ctx.s.config('agent/physics_priors.json'))['bodies']
    kits={obj:asset(ctx.sg['dynamic_objects'][obj]['fix']) for obj in {n.split('/')[0] for n in expected}}
    faces=[];groups=[];rest=[];bonds=[];thickness=[];names=[];vertex_owner=[];face_owner=[];detail_budgets={}
    from calibrated_scene_reconstruction.assets.response_policy import secondary_contact_budgets,contact_limit
    from calibrated_scene_reconstruction.assets.object_understanding import declaration as understand
    for obj,kit in kits.items():
        detail_budgets.update({obj+'/'+name:value for name,value in secondary_contact_budgets(
            kit['parts'],understand(ctx.s,obj),ctx.task.get('flexible_accuracy')).items()})
    for name in report['names']:
        obj,link=name.split('/');kit=kits[obj];p=next(p for p in kit['parts'] if p['name']==link)
        start,count=report['vertex_ranges'][name]
        if start!=len(rest) or count!=len(p['V']):raise ValueError('Flexible trace topology/range mismatch')
        f=np.asarray(p['F'])+start;face_start=len(faces);faces.extend(f);rest.extend(p['cloth_rest']['points'])
        vertex_owner.extend([name]*count);face_owner.extend([name]*len(f))
        components=p['cloth_rest'].get('components',[list(range(count))])
        owners=np.full(len(f),-1)
        for component in components:
            belongs=np.isin(f,np.asarray(component)+start).all(1);owners[belongs]=len(names);names.append(name)
        if (owners<0).any():raise ValueError('Unowned material triangles')
        groups.extend(owners)
        h=ctx.ts.physical_estimates['bodies'][name]['thickness'];thickness.append(h)
        binding=next((b for b in kit['surface_bindings'] if b['surface']==link),None)
        if binding:
            rigid=obj+'/'+binding['rigid']
            if rigid not in rigid_ids:raise ValueError('Attachment owner absent from native rigid trace')
            ids=np.asarray(binding['vertices']);root_faces=np.isin(p['F'],ids).any(1)
            bonds.append({'name':name,'rigid':rigid,'indices':ids+start,'local':p['V'][ids],
                          'face_ids':np.flatnonzero(root_faces)+face_start})
    faces=np.asarray(faces);rest=np.asarray(rest);groups=np.asarray(groups);rows=[]
    vertex_owner=np.asarray(vertex_owner);face_owner=np.asarray(face_owner)
    detail_only=expected<=detail_budgets.keys()
    exposed=np.ones(len(faces),bool)
    for bond in bonds:exposed[bond['face_ids']]=False
    if not exposed.any():raise ValueError('Flexible assembly has no exposed material surface')
    for row in selected:
        step=int(steps[row]);q=np.asarray(nodes[row],float);root_max=0.;roots_pass=True;root_errors={}
        for bond in bonds:
            i=rigid_ids[bond['rigid']];target=bond['local']@R[step,i].T+poses[step,i,:3]
            error=float(np.linalg.norm(q[bond['indices']]-target,axis=1).max())
            root_errors[bond['name']]=error;root_max=max(root_max,error)
            roots_pass &= error<=contact_limit(root_tolerance_m,detail_budgets.get(bond['name']))
        crossing=self_intersections(q,faces[exposed],device=device)
        finite=clearance(q,faces[exposed],rest,max(thickness),face_groups=groups[exposed],cross_only=False,device=device)
        penetration={};by_link={};contacts_pass=True
        for name,solid in solids.items():
            excluded=np.concatenate([b['face_ids'] for b in bonds if b['rigid']==name]) if any(b['rigid']==name for b in bonds) else []
            use=np.ones(len(faces),bool);use[np.asarray(excluded,int)]=False
            free=faces[use];ids,inverse=np.unique(free,return_inverse=True)
            samples=triangle_samples(q[ids],inverse.reshape(-1,3));i=rigid_ids[name]
            local=(samples-poses[step,i,:3])@R[step,i]
            depth=np.zeros(len(local));inside=((local>=solid.bounds[0])&(local<=solid.bounds[1])).all(1)
            if inside.any():depth[inside]=np.maximum(0.,-solid(local[inside]))
            labels=np.r_[vertex_owner[ids],np.tile(face_owner[use],4)]
            by_link[name]={link:float(depth[labels==link].max(initial=0.)) for link in report['names']}
            penetration[name]=float(depth.max(initial=0.))
            contacts_pass &= all(value<=contact_limit(tolerance_m,detail_budgets.get(link)) for link,value in by_link[name].items())
        contact_shape=(crossing['pass'] and finite['pass']) or detail_only
        numeric_pass=roots_pass and crossing['pass'] and finite['pass'] and contacts_pass
        # Approximate secondary sheets still obey their explicit finite root
        # and rigid-contact budgets. Semantic review can assess small local
        # fold/contact details; it cannot approve detached roots or unbounded
        # penetration merely because the sheet is task-secondary.
        passed=bool(roots_pass and contacts_pass and contact_shape)
        rows.append({'physics_step':step,'time_s':float(times[step]),'pass':bool(passed),
                     'maximum_root_error_m':root_max,'intersection':crossing,'clearance':finite,
                     'rigid_penetration_m':penetration,'rigid_penetration_by_link_m':by_link,
                     'root_errors_m':root_errors,'root_accuracy_pass':bool(roots_pass),
                     'root_connections_pass':bool(roots_pass),
                     'numeric_response_pass':bool(numeric_pass),
                     'effective_contact_pass':bool(contacts_pass),
                     'effective_self_contact_pass':bool(contact_shape)})
        if not passed:break  # Archive a failed screen; never certify unexamined states.
    complete=len(rows)==len(selected)
    return {'pass':bool(report['pass'] and complete and all(r['pass'] for r in rows)),
            'applicable':True,'native_material_pass':report['pass'],'complete':complete,'rows':rows,
            'planned_samples':len(selected),'maximum_step_s':maximum_step_s,
            'penetration_tolerance_m':tolerance_m,'root_tolerance_m':root_tolerance_m,
            'positions_sha256':actual_hash,'thickness_basis':'Conservative maximum actually sampled native material thickness',
            'bonded_faces':int((~exposed).sum()),'exposed_faces':int(exposed.sum()),
            'secondary_contact_budgets_m':detail_budgets,
            'qualitative_secondary_only':bool(detail_only),
            'native_semantic_review_required':bool(detail_budgets),
            'scope':'Actual sampled FEM triangles, all rigid source surfaces, real bonds and finite thickness. Root faces excluded only against their declared rigid owner; all flexible pairs retained. No continuous-contact claim.'}
