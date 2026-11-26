"""Qualitative flexible-assembly evidence alongside a native rigid task.

Exact local response is retained as diagnostic data. This does not approve
semantics, identify materials, or replace contact and final delivery checks.
"""
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.assets.cloth_rest import validate
from calibrated_scene_reconstruction.tasks.mechanisms.cloth import nonrigid_change


def assess(ctx,obj,parts,trials,*,task_pass,evidence_files):
    selected={p['name']:p for p in parts if p['kind']=='cloth'}
    expected={obj+'/'+n for n in selected};present=expected<=set(ctx._cloth)
    native_parts=bool(expected and present)
    for name in expected:
        p=ctx.stage.GetPrimAtPath(str(ctx.ts.objects[obj].GetPath())+'/'+name.split('/')[1]+'/surface')
        part=selected[name.split('/')[1]]
        if part.get('backend','particle')=='surface_fem':
            native_parts &= bool(p and p.HasAPI('OmniPhysicsSurfaceDeformableSimAPI'))
        else:
            from pxr import PhysxSchema
            native_parts &= bool(p and p.HasAPI(PhysxSchema.PhysxParticleClothAPI)
                and getattr(ctx._cloth.get(name),'index_registration',{}).get('indexed_particle_order_preserved'))
    for p in selected.values():validate(p['cloth_rest'],p['V'],p['F'])
    complete=True;contacts=True;roots=True;valid=True;peak=0.;quantitative=True;initial_contact=True;initial_self=True;self_contact=True;qualitative=True
    for trial in trials.values():
        native=trial['native'];trace=native.get('flexible_trace');physical=trial.get('flexible',{})
        qualitative &= physical.get('qualitative_secondary_only') is True
        complete &= bool(trace and expected<=set(trace['names']))
        contacts &= physical.get('pass') is True
        roots &= bool(physical.get('rows')) and all(r.get('root_connections_pass',r['maximum_root_error_m']<=physical['root_tolerance_m']) for r in physical.get('rows',[]))
        rows=physical.get('rows',[])
        # The first assay sample can follow earlier passive settling. Initial
        # authored geometry was audited before World.reset, independently.
        authored=getattr(ctx,'authored_mechanism_audit',{}) or {}
        initial_contact &= authored.get('pass') is True
        initial_self &= authored.get('pass') is True
        self_contact &= bool(rows) and all(r.get('intersection',{}).get('pass') is True and r.get('clearance',{}).get('pass') is True for r in rows)
        valid &= native.get('finite_poses') is True and native.get('native_physics',{}).get('pass') is True
        if not trace:continue
        quantitative &= trace.get('quantitative_material_pass',False)
        array=np.load(Path(trial['directory'])/trace['positions'],mmap_mode='r')
        for name in expected:
            link=name.split('/')[1];part=selected[link];a,n=trace['vertex_ranges'][name]
            bond=next(b for b in trial['bindings'] if b['surface']==link)
            free=np.setdiff1d(np.arange(n),bond['vertices'])
            if len(free)<4:continue
            start=np.asarray(array[0,a:a+n][free],float)
            for frame in [len(array)//2,len(array)-1]:
                peak=max(peak,nonrigid_change(start,np.asarray(array[frame,a:a+n][free],float))['maximum_m'])
    checks={'geometry':bool(selected),'native_flexible_parts':native_parts,
            'task_interaction':bool(task_pass and complete),'physical_plausibility':bool(valid and complete and contacts and roots),
            'root_attachment':roots,'contact_preserved':contacts,'rest_metric':True,
            'no_initial_interpenetration':initial_contact,'no_initial_self_intersection':initial_self,
            'no_self_intersection':self_contact,'self_contact_recorded':bool(complete and all(t.get('flexible',{}).get('complete') for t in trials.values()))}
    return {'pass':all(v for k,v in checks.items() if not (qualitative and k=='no_self_intersection')),'representation':'native_deformable',
            'native_flexible_parts':native_parts,'physical_plausibility_pass':checks['physical_plausibility'],
            'solver_deformation_observed':peak>1e-5,'maximum_observed_free_surface_nonrigid_change_m':float(peak),
            'quantitative_material_pass':bool(quantitative),'response_accuracy':'task_and_semantics',
            'independent_native_semantic_review_required':bool(qualitative),
            'checks':{k:{'pass':bool(v)} for k,v in checks.items()},'evidence_files':evidence_files,
            'scope':__doc__.strip()}
