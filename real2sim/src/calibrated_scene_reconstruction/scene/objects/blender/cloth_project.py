"""Globally reduce an admissible sheet proposal's initial membrane strain."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from calibrated_scene_reconstruction.assets.cloth_projection import project
from calibrated_scene_reconstruction.assets.cloth_state import check
from calibrated_scene_reconstruction.scene.session import Session
from . import evidence
from .cloth_pose_projection import Contacts,solid_geometry
from .initial_state import policy
from .material_reference import mesh_parts
from .part_evidence import sheets
from .store import digest


def authored_state(mesh):
    """Use Blender's centered float32 vertices in the original proposal frame.

    Iteration screening and the saved proposal must use the same coordinates.
    Otherwise rounding can put a near-contact pair on opposite sides of the
    very same clearance threshold. This changes neither that threshold nor
    the intrinsic material metric. The Blender rebuild remains independent.
    """
    vertices = np.asarray(mesh['V'], float)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Finite proposal vertices required')
    shift = (vertices.min(0)+vertices.max(0))/2
    shift[2] = vertices[:, 2].min()
    return {**mesh, 'V': (vertices-shift).astype('float32').astype(float)+shift}


def preparation_contract(contract, link, backend, cloth_settings, rigid_rest_offsets, *, contact_envelope='native_rest'):
    """Preserve physical limits and reserve the authored native rigid margin.

    Numerical padding is not a new material thickness. Preparing a smaller gap
    than the actual rest envelope lets reset inject separation into a valid
    physical sheet. Keep both contracts for independent review.
    """
    from copy import deepcopy
    if contact_envelope not in {'native_rest', 'physical'}:
        raise ValueError('Explicit physical or native_rest contact envelope required')
    prepared = deepcopy(contract)
    if contact_envelope == 'physical':
        # A loaded contact state need not have the unloaded solver's padding.
        # Preserve every physical check; this is only a geometry proposal and
        # must be rebuilt and released afresh in the native solver.
        return prepared
    if backend not in {'surface_fem','particle'}:
        return prepared
    rest = float(cloth_settings.get('rest_offset', 0.))
    offsets = np.asarray(rigid_rest_offsets, float)
    if not np.isfinite(rest) or rest < 0 or not np.isfinite(offsets).all() or (offsets < 0).any():
        raise ValueError('Finite nonnegative native rest offsets required')
    if backend=='surface_fem':
        prepared['self_clearance_m'][link] = max(prepared['self_clearance_m'][link], 2*rest)
    # Particle self-contact is a vertex-sphere representation with a separate
    # solid-rest radius. Do not relabel that radius as continuous sheet
    # thickness. Coverage, local filtering and native release need their own
    # checks; only the particle/rigid rest margin belongs in this contract.
    solid = prepared.setdefault('solid_clearance_m', {})
    solid[link] = max(solid.get(link, 0.), rest + float(offsets.max(initial=0.)))
    return prepared


def repair_guard(contract, link, physical_clearance):
    """Search guard only; final proposals still obey the entire contract.

    A released native surface can have valid centre geometry while its finite
    thickness intersects a solid. Explicit bounded repair may start there;
    material, self thickness, support and centre penetration remain hard guards.
    """
    from copy import deepcopy
    guard=deepcopy(contract)
    if physical_clearance:
        guard.get('solid_clearance_m',{}).pop(link,None)
    return guard


def run(s,obj,link,candidate,out,*,iterations=100,cg_iterations=80,linear_solver='cg',device='cuda:0',envelope_repair_limit_m=0.,contact_pairs='all',maximum_step_m=.0003,contact_weight=20.,repair_physical_clearance=False,mechanism_plan=None,physics_plan=None,hotspot_strain_scale=None,contact_envelope='native_rest',clearance_solver='jacobi',release_height_goal_m=None,objective='edges'):
    if objective not in {'edges','triangle_metric'} or objective=='triangle_metric' and hotspot_strain_scale is not None:
        raise ValueError('Choose edges with optional edge-hotspot weighting, or triangle_metric')
    if device not in {'cpu','cuda:0'} or (device=='cpu' and linear_solver=='cuda_direct'):
        raise ValueError('Choose CPU or CUDA with a compatible projection solver')
    if clearance_solver not in {'jacobi','coupled'}:raise ValueError('Explicit supported clearance proposal solver required')
    if release_height_goal_m is not None and(not np.isfinite(release_height_goal_m)or release_height_goal_m<=0):
        raise ValueError('Positive finite optional prestrain objective required; it is not an acceptance limit')
    if not np.isfinite(envelope_repair_limit_m) or not 0<=envelope_repair_limit_m<=.002:
        raise ValueError('Offline envelope repair must be explicitly bounded to at most 2 mm')
    if type(repair_physical_clearance) is not bool or repair_physical_clearance and envelope_repair_limit_m<=0:
        raise ValueError('Physical-clearance repair requires an explicit positive displacement bound')
    if not np.isfinite(maximum_step_m) or not 1e-6<=maximum_step_m<=.003:
        raise ValueError('Geometry proposal step must be between1micrometre and3mm')
    if not np.isfinite(contact_weight) or not 1<=contact_weight<=10000:
        raise ValueError('Bounded positive offline contact penalty required')
    if hotspot_strain_scale is not None and (not np.isfinite(hotspot_strain_scale) or not 1e-8<=hotspot_strain_scale<=.1):
        raise ValueError('Finite bounded hotspot strain scale required')
    s=Session(s.dir.resolve());candidate=Path(candidate).resolve();out=Path(out).resolve()
    if not candidate.is_relative_to(s.dir) or not out.is_relative_to(s.dir):
        raise ValueError('Current-session candidate and output required')
    info=json.loads(candidate.read_text());source=(Path(info['path'])/'mesh.npz').resolve()
    if not source.is_relative_to(s.dir):raise ValueError('Current-capture geometry required')
    with np.load(source) as data:mesh={k:data[k].copy() for k in data.files}
    parts=mesh_parts(mesh);part=next(p for p in parts if p['name']==link and p['kind']=='cloth')
    a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
    original=mesh['V'][a:a+n].copy();faces=mesh['F'][b:b+count]-a
    chart=json.loads(str(mesh['cloth_rest_json']))[link];rest=np.asarray(chart['points'])
    if (mechanism_plan is None)!=(physics_plan is None):
        raise ValueError('Candidate mechanism and physical hypotheses must be supplied together')
    hypothesis_files=[]
    if mechanism_plan is not None:
        hypothesis_files=[Path(mechanism_plan).resolve(),Path(physics_plan).resolve()]
        if any(not p.is_relative_to(s.dir) or not p.is_file() for p in hypothesis_files):
            raise ValueError('Explicit current-session candidate hypothesis files required')
        contract=policy(s,obj,mechanism_hypothesis=json.loads(hypothesis_files[0].read_text()),
                        physics_hypothesis=json.loads(hypothesis_files[1].read_text()))
    else:contract=policy(s,obj)
    if not contract or link not in contract.get('self_clearance_m',{}):
        raise ValueError('Declared material validity and physical layer thickness required')
    physical_contract=contract
    fit=s.scene_graph['dynamic_objects'][obj]['fit']
    settings={**fit.get('link_physics',{}).get(link,{}),**info.get('cloth_physics',{})}
    default_sdf=info.get('sdf',fit.get('sdf',{}))
    rigid_offsets=[fit.get('link_physics',{}).get(p['name'],{}).get('sdf',default_sdf).get('rest_offset',0.)
                   for p in parts if p['kind']=='rigid']
    contract=preparation_contract(contract,link,part.get('backend','particle'),settings,rigid_offsets,contact_envelope=contact_envelope)
    h=contract['self_clearance_m'][link]
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    from calibrated_scene_reconstruction.assets import cloth_projection
    shutil.copy2(cloth_projection.__file__,out/'cloth_projection.py')
    from calibrated_scene_reconstruction.assets import cloth_contact_rows
    shutil.copy2(cloth_contact_rows.__file__,out/'cloth_contact_rows.py')
    from . import cloth_pose_projection
    from calibrated_scene_reconstruction.sim import cloth_clearance
    for module in [cloth_pose_projection,cloth_clearance]:
        shutil.copy2(module.__file__,out/Path(module.__file__).name)
    inputs={'candidate':str(candidate),'candidate_sha256':digest(candidate),
        'mesh':str(source),'mesh_sha256':digest(source),'iterations':iterations,'cg_iterations':cg_iterations,'linear_solver':linear_solver,'device':device,
        'material_policy':physical_contract,'preparation_contact_policy':contract,'contact_pairs':contact_pairs,'maximum_step_m':maximum_step_m,'contact_weight':contact_weight,
        'envelope_repair_limit_m':envelope_repair_limit_m,'repair_physical_clearance':repair_physical_clearance,'source_sha256':digest(__file__),
        'hotspot_strain_scale':hotspot_strain_scale,'contact_envelope':contact_envelope,'clearance_solver':clearance_solver,
        'release_height_goal_m':release_height_goal_m,'geometry_objective':objective,
        'projection_sha256':digest(cloth_projection.__file__),
        'contact_rows_sha256':digest(cloth_contact_rows.__file__),
        'contact_proposal_sha256':digest(cloth_pose_projection.__file__),
        'clearance_sha256':digest(cloth_clearance.__file__),
        'physics_priors_sha256':digest(hypothesis_files[1] if hypothesis_files else s.config_dir/'agent/physics_priors.json'),
        'candidate_hypotheses_sha256':{str(p):digest(p) for p in hypothesis_files}}
    (out/'input.json').write_text(json.dumps(inputs,indent=2)+'\n')
    rigids=solid_geometry(mesh,parts)
    # Proposal active sets may be reduced; admissibility always screens all pairs.
    # An old material-neighbour exception could miss the thickness of a fold.
    maximum_step=maximum_step_m
    solid_gap=max(.5*h,contract.get('solid_clearance_m',{}).get(link,0.))+1e-5
    contacts=Contacts(faces,rest,rigids,h,solid_gap,device=device,weight=contact_weight,maximum_step=maximum_step,pair_selection=contact_pairs)
    from .cloth_pose_projection import material_budget,within_material_budget,MaterialBounds,combined_constraints
    # Use the same representable seed as every admissibility trial. Measuring
    # the budget in float64 but checking centred float32 can reject the seed
    # itself solely from rounding of a narrow material triangle.
    initial_metrics=check(authored_state(mesh),physical_contract,device=device)['links'][link]['material']
    repair_budget=material_budget(initial_metrics,physical_contract['links'][link])
    (out/'material_repair_budget.json').write_text(json.dumps(repair_budget,indent=2)+'\n')
    guard_limit=.95*min(repair_budget['maximum_absolute_edge_strain'],repair_budget['maximum_principal_material_strain'])
    constraints=combined_constraints(contacts,MaterialBounds(faces,rest,guard_limit,weight=contact_weight)) if 0<guard_limit<1 else contacts
    last_report=None
    rejected=[]
    def admissible(points):
        nonlocal last_report
        vertices=mesh['V'].astype(float);vertices[a:a+n]=points
        # Blender recentres then writes float32 vertices. A valid double-
        # precision proposal at a near-zero gap is insufficient for rebuilding.
        last_report=check(authored_state({**mesh,'V':vertices}),contract,device=device)
        last_report['offline_material_budget_pass']=within_material_budget(last_report['links'][link]['material'],repair_budget)
        last_report['pass'] &= last_report['offline_material_budget_pass']
        if not last_report['pass']:rejected.append(last_report)
        return last_report['pass']
    def record(points,row):
        payload={**row,'admissibility':last_report,'contact_constraints':contacts.latest,
                 'rejected_trials':len(rejected),'first_rejected_trial':rejected[0] if rejected else None,
                 'proposal_only':True,'task_approved':False}
        rejected.clear()
        (out/'progress.json').write_text(json.dumps(payload,indent=2)+'\n')
        if row['accepted']:
            candidate_mesh={**mesh,'V':mesh['V'].astype(float)};candidate_mesh['V'][a:a+n]=points
            np.savez_compressed(out/'latest_proposal.npz',**candidate_mesh)
        print('GLOBAL_SHEET_PROJECT',row['iteration'],row['accepted'],row['maximum_edge_strain'],
              row['rms_edge_strain'],row['maximum_displacement_m'],flush=True)
    if envelope_repair_limit_m and not admissible(original):
        from .cloth_pose_projection import repair_clearances
        search_contract=repair_guard(physical_contract,link,repair_physical_clearance)
        def physically_admissible(points):
            vertices=mesh['V'].astype(float);vertices[a:a+n]=points
            measured=check(authored_state({**mesh,'V':vertices}),search_contract,device=device)
            return measured['pass'] and within_material_budget(measured['links'][link]['material'],repair_budget)
        def repair_progress(points,row):
            (out/'envelope_repair_progress.json').write_text(json.dumps(row,indent=2)+'\n')
            if row['accepted']:
                # A checkpoint is still only a proposal. Save atomically so an
                # interrupted repair does not discard all accepted CPU work.
                import os
                checkpoint={**mesh,'V':mesh['V'].astype(float)};checkpoint['V'][a:a+n]=points
                temporary=out/'latest_envelope_proposal.tmp.npz'
                np.savez_compressed(temporary,**checkpoint)
                os.replace(temporary,out/'latest_envelope_proposal.npz')
            print('SHEET_CLEARANCE_REPAIR',row,flush=True)
        original,repair=repair_clearances(original,constraints,admissible=physically_admissible,
            maximum_displacement=envelope_repair_limit_m,iterations=128,on_iteration=repair_progress,
            satisfied=admissible,solver=clearance_solver)
        (out/'envelope_repair.json').write_text(json.dumps(repair,indent=2)+'\n')
        if not admissible(original):
            (out/'envelope_repair_failure.json').write_text(json.dumps(last_report,indent=2)+'\n')
            raise ValueError('Bounded geometry repair did not satisfy the selected contact envelope')
    def release_goal(points,row):
        if release_height_goal_m is None or not row['accepted']:return False
        return last_report['links'][link]['initial_release_diagnostic']['membrane']['equivalent_gravity_height_m']<=release_height_goal_m
    points,report=project(rest,original,faces,iterations=iterations,cg_iterations=cg_iterations,
        device=device,maximum_step_m=maximum_step,linear_contacts=constraints,admissible=admissible,on_iteration=record,linear_solver=linear_solver,
        hotspot_strain_scale=hotspot_strain_scale,stop_condition=release_goal if release_height_goal_m is not None else None,objective=objective)
    mesh['V']=mesh['V'].astype(float);mesh['V'][a:a+n]=points
    mesh=authored_state(mesh)
    final=check(mesh,contract,device=device)
    final['offline_material_budget_pass']=within_material_budget(final['links'][link]['material'],repair_budget)
    final['pass'] &= final['offline_material_budget_pass']
    (out/'final_admissibility.json').write_text(json.dumps(final,indent=2)+'\n')
    if not final['pass']:raise ValueError('Projected geometry fails independent initial-state checks')
    np.savez_compressed(out/'mesh.npz',**mesh)
    (out/'candidate.json').write_text(json.dumps({**info,'path':str(out),'proposal_only':True,
        'task_approved':False,'native_approved':False,'source_pose_approved':False,
        'contact_envelope':contact_envelope,'requires_agent_rebuild':True,
        'requires_fresh_native_release':True},indent=2)+'\n')
    observed=evidence.load(s,obj,parts=True)
    metrics,_=evidence.score(mesh['V'],mesh['F'],info['pose'],observed,model_parts=parts)
    report.update(admissibility=final,metrics=metrics)
    if release_height_goal_m is not None:
        height=final['links'][link]['initial_release_diagnostic']['membrane']['equivalent_gravity_height_m']
        report['prestrain_objective']={'requested_height_m':release_height_goal_m,'estimated_height_m':height,
            'achieved':height<=release_height_goal_m,'scope':'Proposal stopping objective only; no native energy measurement, physical limit, source stability or task approval.'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    visual=sheets(s,evidence.placed(mesh['V'],mesh['F'],info['pose']),observed[0],parts,out/'source_comparison')
    (out/'visual.json').write_text(json.dumps(visual,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session');p.add_argument('object');p.add_argument('link');p.add_argument('candidate')
    p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--cg-iterations',type=int,default=80)
    p.add_argument('--linear-solver',choices=['cg','sparse_direct','cuda_direct'],default='cg')
    p.add_argument('--objective',choices=['edges','triangle_metric'],default='edges',help='Explicit area-weighted material metric objective also resolves in-plane shear in narrow triangles; all independent geometry guards remain active')
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    p.add_argument('--contact-pairs',choices=['all','closest'],default='all')
    p.add_argument('--maximum-step-m',type=float,default=.0003)
    p.add_argument('--contact-weight',type=float,default=20.)
    p.add_argument('--clearance-solver',choices=['jacobi','coupled'],default='jacobi',help='Coupled sparse inequalities can move contacting plies together; full nonlinear guards remain strict')
    p.add_argument('--release-height-goal-m',type=float,help='Optional membrane-prestrain stopping objective in equivalent gravity height, not a native energy or acceptance threshold')
    p.add_argument('--contact-envelope',choices=['native_rest','physical'],default='native_rest',help='Default reserves unloaded native padding; physical preserves true thickness for loaded-state geometry proposals and still requires fresh native release')
    p.add_argument('--hotspot-strain-scale',type=float,help='Emphasize sparse intrinsic-strain hotspots instead of letting low RMS hide them; immutable material chart and all contact checks remain active')
    p.add_argument('--envelope-repair-limit-m',type=float,default=0.)
    p.add_argument('--repair-physical-clearance',action='store_true',help='Bounded offline repair of finite-thickness solid gaps; final full checks remain strict')
    p.add_argument('--mechanism-plan',help='Explicit paired current-session candidate representation hypothesis')
    p.add_argument('--physics-plan',help='Explicit paired current-session candidate physical hypothesis')
    a=p.parse_args()
    run(Session(a.session),a.object,a.link,a.candidate,a.output,iterations=a.iterations,cg_iterations=a.cg_iterations,linear_solver=a.linear_solver,device=a.device,envelope_repair_limit_m=a.envelope_repair_limit_m,contact_pairs=a.contact_pairs,maximum_step_m=a.maximum_step_m,contact_weight=a.contact_weight,repair_physical_clearance=a.repair_physical_clearance,mechanism_plan=a.mechanism_plan,physics_plan=a.physics_plan,hotspot_strain_scale=a.hotspot_strain_scale,contact_envelope=a.contact_envelope,clearance_solver=a.clearance_solver,release_height_goal_m=a.release_height_goal_m,objective=a.objective)


if __name__=='__main__':main()
