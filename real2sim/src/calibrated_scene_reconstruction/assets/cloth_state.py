"""Shared sheet-state measurements for authored and actual native geometry."""
import json

import numpy as np


def check(mesh,contract,*,device=None,binding_targets=None):
    from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric
    parts=json.loads(str(mesh['links_json']));rest=json.loads(str(mesh['cloth_rest_json']))
    result={};passed=True;solid_surfaces={};actual_device=device or 'cpu'
    bindings=json.loads(str(mesh['surface_bindings_json'])) if 'surface_bindings_json' in mesh else []
    if binding_targets is not None and (not isinstance(binding_targets,dict)
            or set(binding_targets)!={b['surface'] for b in bindings}):
        raise ValueError('Supply actual rigid-carried targets for every declared physical bond')
    for name,limits in contract['links'].items():
        detail_budget=contract.get('secondary_contact_budgets_m',{}).get(name)
        part=next(part for part in parts if part['name']==name)
        a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
        points,faces=mesh['V'][a:a+n],mesh['F'][b:b+count]-a
        metric=MaterialMetric(points,faces,rest[name],max_edge_strain=limits['max_edge_strain'],
                                max_area_change=limits['max_area_change'])
        material=metric.check(points)
        from .response_policy import material_verdict
        response=material_verdict(material,contract.get('material_accuracy_mode','quantitative'))
        row={'pass':response['pass'],'material':material,'material_response':response}
        bond=next((b for b in bindings if b['surface']==name),None)
        if binding_targets is not None and bond:
            row['bond_resolution_diagnostic']=metric.bond_diagnostic(points,bond['vertices'],binding_targets[name])
        parameters=contract.get('release_parameters',{}).get(name)
        particle_parameters=contract.get('particle_release_parameters',{}).get(name)
        if row['pass'] and particle_parameters:
            from .spring_energy import measure as spring_energy
            mass=particle_parameters.get('mass',particle_parameters.get('areal_density',0.)*material['rest_area_m2'])
            row['initial_spring_release_diagnostic']=spring_energy(points,rest[name],particle_parameters,
                mass_kg=mass if mass>0 else None)
        if row['pass'] and parameters:
            from .deformable_release import bending_budget,membrane_budget
            mass=parameters.get('mass',parameters.get('areal_density',0.)*material['rest_area_m2'])
            row['initial_release_diagnostic']={
                'membrane':membrane_budget(points,faces,rest[name]['points'],parameters['youngs_modulus'],
                    parameters['poissons_ratio'],parameters['thickness'],mass),
                'bending':bending_budget(points,faces,rest[name]['points'],parameters['flexural_rigidity'],mass,
                    natural=rest[name].get('bend_reference',{}).get('points')),
                'scope':'Prestrain/curvature estimates, not native energy or acceptance. Unexplained preload requires source/history review and actual unforced-release checks; prestressed objects may be intentional.'}
        support=contract.get('initial_support',{}).get(name)
        if row['pass'] and support:
            from .material_support import check as support_state
            rigid=next(p for p in parts if p['name']==support['link'] and p['kind']=='rigid')
            x,v,y,f=[rigid[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
            relation=support_state(points,rest[name]['points'],faces,mesh['V'][x:x+v],mesh['F'][y:y+f]-x,
                support,thickness_m=contract.get('self_clearance_m',{}).get(name),
                tolerance_m=contract['maximum_authored_solid_penetration_m'])
            row['initial_support']=relation['routing']
            if relation['material_contact'] is not None:row['initial_support_contact']=relation['material_contact']
            row['pass'] &= relation['pass']
        # Do not spend native/contact work on a sheet already outside its
        # declared material validity. The failed measurement remains archived.
        if row['pass']:
            import warp as wp
            wp.init();actual_device=device or ('cuda:0' if wp.is_cuda_available() else 'cpu')
            if contract['material_state_contract']>=2:
                from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
                row['self_intersection']=self_intersections(points,faces,device=actual_device)
                row['pass'] &= row['self_intersection']['pass'] or detail_budget is not None
                clearance=contract.get('self_clearance_m',{}).get(name)
                if row['pass'] and clearance is not None:
                    from calibrated_scene_reconstruction.assets.cloth_clearance import measure
                    row['self_clearance']=measure(points,faces,rest[name]['points'],clearance,device=actual_device,
                        local_bending_review=contract.get('material_accuracy_mode','quantitative')=='task_and_semantics')
                    row['pass'] &= row['self_clearance']['pass']
            if row['pass']:
                from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface,triangle_samples
                samples=triangle_samples(points,faces);penetration={};solid_clearance={}
                bond=next((b for b in bindings if b['surface']==name),None)
                clearance=contract.get('solid_clearance_m',{}).get(name)
                if clearance is not None and (not np.isfinite(clearance) or clearance<=0):
                    raise ValueError('Positive finite physical sheet/solid clearance required')
                for rigid in parts:
                    if rigid['kind'] not in {'rigid','deformable','rope'}:continue
                    x,v=rigid['vertex_start'],rigid['vertex_count'];y,f=rigid['face_start'],rigid['triangle_count']
                    if rigid['name'] not in solid_surfaces:
                        solid_surfaces[rigid['name']]=SignedSurface(mesh['V'][x:x+v],mesh['F'][y:y+f]-x,device=actual_device)
                    surface=solid_surfaces[rigid['name']];queried=samples
                    if bond and bond['rigid']==rigid['name']:
                        indices=np.asarray(bond['vertices'],int)
                        gap=float(max(0.,surface(points[indices]).max()))
                        from .response_policy import contact_limit
                        root_tolerance=contact_limit(contract.get('root_connection_tolerance_m',.00005),detail_budget)
                        if not np.isfinite(root_tolerance) or not 0<root_tolerance<=(.002 if detail_budget is not None else .0001):
                            raise ValueError('Root gap tolerance must remain within the native100micrometre connection budget')
                        row['root_connection']={'pass':gap<=root_tolerance,'maximum_root_gap_m':gap,'tolerance_m':root_tolerance}
                        row['pass'] &= row['root_connection']['pass']
                        free=faces[~np.isin(faces,indices).any(axis=1)]
                        if not len(free):raise ValueError('Bonded sheet has no exposed free surface')
                        used,inverse=np.unique(free,return_inverse=True)
                        queried=triangle_samples(points[used],inverse.reshape(-1,3))
                        row['bonded_root_faces']=len(faces)-len(free)
                    penetration[rigid['name']]=surface.max_penetration(queried)
                    if clearance is not None:
                        tolerance=contract['maximum_authored_solid_penetration_m']
                        near=((queried>=surface.bounds[0]-clearance-tolerance)&
                              (queried<=surface.bounds[1]+clearance+tolerance)).all(1)
                        signed=surface(queried[near]) if near.any() else np.empty(0)
                        deficit=float(max(0.,clearance-signed.min())) if len(signed) else 0.
                        allowance=contract.get('secondary_thickness_tolerances_m',{}).get(name,0.)
                        if not np.isfinite(allowance) or not 0<=allowance<=clearance:
                            raise ValueError('Secondary thickness approximation cannot exceed physical half-thickness')
                        exact=deficit<=tolerance
                        solid_clearance[rigid['name']]={'pass':deficit<=max(tolerance,allowance),
                            'physical_thickness_clearance_pass':exact,
                            'secondary_thickness_approximation_m':float(allowance),
                            'native_semantic_review_required':bool(not exact and allowance>0),
                            'required_mid_surface_clearance_m':float(clearance),'tolerance_m':float(tolerance),
                            'maximum_deficit_m':deficit,'below_clearance_samples':int((signed<clearance-tolerance).sum()),
                            'samples_queried':int(near.sum()),
                            'minimum_queried_signed_distance_m':float(signed.min()) if len(signed) else None}
                row['rigid_penetration_m']=penetration
                tolerance=contract.get('solid_penetration_tolerances_m',{}).get(name,contract['maximum_authored_solid_penetration_m'])
                from .response_policy import contact_limit
                tolerance=contact_limit(tolerance,detail_budget)
                row['solid_penetration_tolerance_m']=tolerance
                row['pass'] &= all(value<=tolerance for value in penetration.values())
                if clearance is not None:
                    row['solid_clearance']=solid_clearance
                    row['pass'] &= all(value['pass'] for value in solid_clearance.values())
        if detail_budget is not None and binding_targets is not None:
            # Qualitative secondary response is reviewed from the actual
            # native states. Do not prevent testing the task on a local
            # accuracy diagnostic; final semantic/physical review is separate.
            row['secondary_numeric_accuracy_pass']=bool(row['pass'])
            row['native_semantic_review_required']=True
            row['task_accepted']=False
            row['acceptance_scope']='Finite secondary native state only; raw response diagnostics and final semantic/physical review remain separate'
            row['pass']=bool(response['geometry_valid'] and row.get('initial_support',{}).get('pass',True))
        result[name]=row;passed &= bool(row['pass'])
    collective=None;collective_clearance=None
    if contract['material_state_contract']>=2 and all(row['material_response']['pass'] for row in result.values()):
        selected=[];owners=[];names=[];thickness=[]
        for part in parts:
            if part['kind']!='cloth':continue
            start,count=part['face_start'],part['triangle_count'];actual_faces=mesh['F'][start:start+count]
            bond=next((b for b in bindings if b['surface']==part['name']),None)
            if bond:
                rooted=np.asarray(bond['vertices'],int)+part['vertex_start']
                actual_faces=actual_faces[~np.isin(actual_faces,rooted).any(axis=1)]
            material=rest[part['name']]
            components=material.get('components',[list(range(part['vertex_count']))])
            for index,component in enumerate(components):
                inside=np.isin(actual_faces,np.asarray(component)+part['vertex_start']).all(axis=1)
                group_faces=actual_faces[inside]
                if not len(group_faces):continue
                selected.extend(group_faces);owners.extend([len(names)]*len(group_faces))
                names.append(part['name']+':component_'+str(index))
                thickness.append(contract.get('self_clearance_m',{}).get(part['name']))
        if len(names)>1 and selected:
            from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
            collective=self_intersections(mesh['V'],np.asarray(selected),device=device or actual_device,face_groups=np.asarray(owners))
            collective['example_link_pairs']=[[names[owners[a]],names[owners[b]]] for a,b in collective['example_face_pairs']]
            collective['scope']='Exposed surfaces of different physical sheets; only declared physical root faces excluded. Individual-sheet checks remain mandatory.'
            secondary=set(contract.get('secondary_contact_budgets_m',{}))
            detail_only=all(part['name'] in secondary for part in parts if part['kind']=='cloth')
            collective['qualitative_secondary_detail']=detail_only
            passed &= collective['pass'] or detail_only
            if all(value is not None for value in thickness):
                from calibrated_scene_reconstruction.assets.cloth_clearance import measure
                # Conservative for mixed materials; no nearby-material
                # exclusion is valid between separate physical components.
                collective_clearance=measure(mesh['V'],np.asarray(selected),mesh['V'],max(thickness),
                    device=device or actual_device,face_groups=np.asarray(owners))
                collective_clearance['component_names']=names
                collective_clearance['threshold_basis']='Maximum declared material thickness, conservative for different-thickness sheets; rooted faces excluded, all free components retained'
                passed &= collective_clearance['pass']
    return {'pass':passed,'links':result,'cross_surface_intersection':collective,
            'cross_surface_clearance':collective_clearance,'contract':contract,
            'scope':'Actual supplied triangle state against intrinsic material, self-contact and solid geometry; not task approval'}
