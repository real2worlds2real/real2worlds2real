"""Reject invalid authored sheet states before silhouette optimization.

Use the existing material-state limits and exported-stage penetration criterion.
This is geometry admissibility, never a substitute for actual native dynamics.
"""
import json
import hashlib
import os
from copy import deepcopy
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.cloth_state import check


from calibrated_scene_reconstruction.core.environment import setting

def _preserve_source_conditions(source,candidate):
    """A numerical hypothesis cannot silently discard declared input behavior."""
    for obj,spec in source.get('objects',{}).items():
        links=candidate['objects'][obj]['links']
        for name,body in spec['links'].items():
            proposed=links.get(name,{})
            if proposed.get('kind')!=body['kind']:
                raise ValueError('Candidate hypotheses cannot drop or replace source physical links')
            support=body.get('initial_support')
            if support is None:continue
            other=proposed.get('initial_support',{})
            if (any(other.get(k)!=support[k] for k in ['link','relation','surface_region'])
                    or not np.allclose(other.get('up',[]),support['up'],atol=1e-8)):
                raise ValueError('Candidate hypotheses cannot drop source initial support routing')
            contact=support.get('material_contact')
            if contact:
                new=other.get('material_contact',{})
                if (any(new.get(k)!=contact[k] for k in ['axis','end','fraction'])
                        or new.get('maximum_gap_m',float('inf'))>contact['maximum_gap_m']
                        or new.get('minimum_area_fraction',0)<contact['minimum_area_fraction']):
                    raise ValueError('Candidate hypotheses cannot weaken source material-end support')
    mechanisms={m['id']:m for m in candidate.get('mechanisms',[])}
    for old in source.get('mechanisms',[]):
        new=mechanisms.get(old['id'],{})
        if (any(new.get(k)!=old[k] for k in ['type','role','links','representation'])
                or not set(old['checks'])<=set(new.get('checks',[]))):
            raise ValueError('Candidate hypotheses cannot drop source mechanisms or required checks')
        if old.get('initial_state'):
            bounds=new.get('initial_state',{}).get('position_interval',[])
            original=old['initial_state']['position_interval']
            if len(bounds)!=2 or bounds[0]<original[0] or bounds[1]>original[1]:
                raise ValueError('Candidate hypotheses cannot widen the source initial articulation interval')


def policy(s,obj,*,mechanism_hypothesis=None,physics_hypothesis=None):
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
    plan=declaration(s)
    hypotheses={}
    if (mechanism_hypothesis is None)!=(physics_hypothesis is None):
        raise ValueError('Candidate representation and physical hypotheses must be supplied together')
    if mechanism_hypothesis is not None:
        from calibrated_scene_reconstruction.assets.mechanisms import validate as validate_mechanisms,digest
        from calibrated_scene_reconstruction.assets.physical_priors import validate as validate_priors
        source=plan or {};plan=validate_mechanisms(deepcopy(mechanism_hypothesis))
        if (set(plan['objects'])!=set(source.get('objects',{}))
                or plan.get('material_state_contract',0)<source.get('material_state_contract',0)):
            raise ValueError('Candidate hypotheses cannot drop source object scope or material-state checks')
        _preserve_source_conditions(source,plan)
        for name,body in source.get('objects',{}).items():
            for link,request in body['links'].items():
                effects=set(request.get('required_contact_effects',[]))
                candidate=plan['objects'][name]['links'].get(link,{})
                if not effects<=set(candidate.get('required_contact_effects',[])):
                    raise ValueError('Candidate hypotheses cannot drop required native contact effects')
        physical=validate_priors(deepcopy(physics_hypothesis),plan)
        hypotheses={'candidate_hypotheses':{'mechanisms_sha256':digest(plan),
                                          'physical_priors_sha256':digest(physical),'task_approved':False}}
    version=(plan or {}).get('material_state_contract',0)
    links=(plan or {}).get('objects',{}).get(obj,{}).get('links',{})
    selected=[name for name,link in links.items() if link['kind']=='cloth']
    if not version or not selected:return None
    from calibrated_scene_reconstruction.assets.physical_priors import expand_families
    priors=(physical if hypotheses else expand_families(s.config('agent/physics_priors.json')))['bodies']
    # Use the candidate's declared material thickness, never a product-size
    # reference or a fitted-to-success thickness. Particle-only priors without
    # this parameter retain their separate native sampling/contact checks.
    clearance={}
    release={}
    particle_release={}
    if version>=2:
        for name in selected:
            thickness=priors[obj+'/'+name]['parameters'].get('thickness')
            if thickness is not None:
                if thickness.get('unit')!='m':
                    raise ValueError('Sheet self-clearance requires thickness in metres')
                clearance[name]=thickness['nominal']
            parameters=priors[obj+'/'+name]['parameters']
            if (links[name].get('backend')=='surface_fem' and
                    {'youngs_modulus','poissons_ratio','thickness','flexural_rigidity'}<=parameters.keys()):
                release[name]={k:v['nominal'] for k,v in parameters.items()
                               if k in {'youngs_modulus','poissons_ratio','thickness','flexural_rigidity','mass','areal_density'}}
            if (links[name].get('backend','particle')=='particle' and
                    {'stretch_stiffness','shear_stiffness','bend_stiffness'}<=parameters.keys()):
                particle_release[name]={k:v['nominal'] for k,v in parameters.items()
                    if k in {'stretch_stiffness','shear_stiffness','bend_stiffness','mass','areal_density'}}
    from calibrated_scene_reconstruction.assets.response_policy import mode,approximate_part
    accuracy=mode(s.config('task.json',{}).get('flexible_accuracy'))
    eligible=[]
    if (accuracy=='task_and_semantics' and hasattr(s,'config_dir')
            and (Path(s.config_dir)/'agent/object_understanding.json').is_file()):
        from calibrated_scene_reconstruction.assets.object_understanding import declaration as understanding
        primary={link for mechanism in (plan or {}).get('mechanisms',[]) if mechanism.get('role')=='primary'
                 for link in mechanism.get('links',[])}
        eligible=[link for part in understanding(s,obj)['parts']
                  if approximate_part(part) and part.get('task_primary') is False for link in part['links']
                  if obj+'/'+link not in primary]
    return {**hypotheses,'material_state_contract':version,
            'material_accuracy_mode':accuracy,'secondary_thickness_approximation_links':eligible,
            'links':{name:priors[obj+'/'+name]['validity_hypothesis'] for name in selected},
            'self_clearance_m':clearance,
            'solid_clearance_m':{name:thickness/2 for name,thickness in clearance.items()},
            'release_parameters':release,
            'particle_release_parameters':particle_release,
            'initial_support':{name:links[name]['initial_support'] for name in selected if links[name].get('initial_support')},
            'maximum_authored_solid_penetration_m':1e-6}


class Checked:
    def __init__(self,builder,contract,output,*,construction_plan=None):
        self.construction=construction_plan is not None
        if self.construction:
            # An intermediate jig need not already reproduce final support.
            # Its physical geometry remains strict and its report cannot pass
            # a source/task consumer expecting initial_state_admissibility.
            plan=deepcopy(construction_plan);supported=set()
            if (not isinstance(contract,dict) or not isinstance(plan,dict) or plan.get('version')!=1 or not plan.get('basis')
                    or plan.get('cloth_link') not in contract.get('links',{})):
                raise ValueError('Explicit temporary material-construction plan required')
            for grip in plan.get('grips',[]):
                active=np.asarray(grip.get('active_s',[]),float)
                force=grip.get('force_limit_N')
                if (active.shape!=(2,) or not np.isfinite(active).all() or active[0]!=0 or active[1]<=0
                        or type(force) not in (int,float) or not np.isfinite(force) or not 0<force<=10):
                    raise ValueError('Construction requires a finite-force grip active from initialization')
                supported.add(plan['cloth_link'])
            if not supported:raise ValueError('Temporary construction support needs an actual finite-force grip')
            contract=deepcopy(contract)
            contract['construction_only']={'source_initial_conditions_evaluated':False,
                'temporary_fixture_plan':plan,'task_approved':False,
                'source_initial_support':deepcopy(contract.get('initial_support',{}))}
            for name in supported:contract.get('initial_support',{}).pop(name,None)
        self.builder,self.contract,self.output=builder,contract,Path(output)
        self.spec=builder.spec;self.cache={}

    def support_offset(self,mesh,**kwargs):
        from calibrated_scene_reconstruction.assets.support_placement import offset
        return offset(mesh,self.contract.get('self_clearance_m',{}),**kwargs)

    def evaluate(self,*args,**kwargs):
        path,report=self.builder.evaluate(*args,**kwargs)
        from .store import digest
        contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        key=(path,digest(path/'mesh.npz'),contract_sha)
        if key not in self.cache:
            with np.load(path/'mesh.npz') as mesh:
                device=setting('GEOMETRY_DEVICE')
                if device is not None and device!='cpu' and not device.startswith('cuda:'):raise ValueError('Explicit CPU or CUDA geometry device required')
                contract=deepcopy(self.contract)
                eligible=contract.get('secondary_thickness_approximation_links',[])
                if eligible:
                    from calibrated_scene_reconstruction.assets.metric_mesh import asset
                    from calibrated_scene_reconstruction.assets.response_policy import secondary_thickness_tolerances
                    kit=asset({'mesh':str(path/'mesh.npz')})
                    contract['secondary_thickness_tolerances_m']=secondary_thickness_tolerances(
                        kit['parts'],kit['surface_bindings'],eligible,contract.get('self_clearance_m',{}))
                self.cache[key]=check(mesh,contract,device=device)
                # Review natural folds at the same topology as the authored
                # state. An admissible loaded mesh can still spring toward an
                # unresolved, sub-thickness rest crease after initialization.
                from calibrated_scene_reconstruction.assets.cloth_curvature import reference_geometry
                charts=json.loads(str(mesh['cloth_rest_json']))
                parts=json.loads(str(mesh['links_json']))
                for name,thickness in contract.get('self_clearance_m',{}).items():
                    part=next(p for p in parts if p['name']==name)
                    start,count=part['face_start'],part['triangle_count']
                    review=reference_geometry(charts[name],mesh['F'][start:start+count]-part['vertex_start'],
                                              thickness,device=device or 'cpu')
                    if review is not None:
                        self.cache[key]['links'][name]['natural_reference_geometry']=review
            self.output.mkdir(parents=True,exist_ok=True)
            (self.output/(path.name+'_'+key[1][:16]+'_'+contract_sha[:12]+'.json')).write_text(json.dumps(self.cache[key],indent=2)+'\n')
        result=self.cache[key]
        if not result['pass']:
            failures={name:{'material':row['material']['pass'],
                            'crossings':row.get('self_intersection',{}).get('intersecting_pairs'),
                            'thickness_overlaps':row.get('self_clearance',{}).get('violating_pairs'),
                            'penetration_m':row.get('rigid_penetration_m'),
                            'solid_clearance':row.get('solid_clearance'),
                            'initial_support_contact':row.get('initial_support_contact'),
                            'initial_support':row.get('initial_support')} for name,row in result['links'].items() if not row['pass']}
            for key in ('cross_surface_intersection','cross_surface_clearance'):
                if result.get(key) and not result[key]['pass']:
                    failures[key]=result[key]
            raise ValueError('Authored initial material state failed: '+json.dumps(failures))
        report=deepcopy(report)
        if self.construction:
            report.pop('initial_state_admissibility',None)
            report['construction_state_admissibility']={**result,
                'source_initial_conditions_evaluated':False,'task_approved':False,
                'temporary_fixture_plan':self.contract['construction_only']['temporary_fixture_plan']}
        else:report['initial_state_admissibility']=result
        return path,report




def wrap(s,obj,builder,output):
    from .articulation_path import Checked as ArticulationChecked
    builder=ArticulationChecked(builder,Path(output)/'articulation_path')
    geometry_plan=s.config('agent/geometry_checks.json',{})
    if geometry_plan:
        if geometry_plan.get('version')!=1:raise ValueError('Source geometry check version must be1')
        checks=geometry_plan.get('objects',{}).get(obj,[])
        if checks:
            from .geometry_checks import Checked as GeometryChecked
            from calibrated_scene_reconstruction.core.paths import ROOT
            builder=GeometryChecked(builder,checks,Path(output)/'source_geometry',ROOT)
    contract=policy(s,obj)
    return Checked(builder,contract,output) if contract else builder
