"""Refit a same-capture candidate pose against every original source gate."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import evidence
from .fit import optimize
from .projection import optional
from .material_reference import mesh_parts
from .store import digest
from calibrated_scene_reconstruction.scene.session import Session


def run(s,obj,candidate,out,*,budget=300):
    s=Session(s.dir.resolve());candidate=Path(candidate).resolve();out=Path(out).resolve()
    if (not candidate.is_relative_to(s.dir) or not out.is_relative_to(s.dir)
            or type(budget) is not int or not 10<=budget<=1000):
        raise ValueError('Current-session candidate/output and bounded pose budget required')
    info=json.loads(candidate.read_text());mesh_dir=Path(info['path']).resolve()
    if not mesh_dir.is_relative_to(s.dir):raise ValueError('Current-capture candidate mesh required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
    with np.load(mesh_dir/'mesh.npz') as data:mesh={k:data[k].copy() for k in data.files}
    parts=mesh_parts(mesh)
    class FixedGeometry:
        def evaluate(self,params):
            if params:raise ValueError('Pose-only proposal cannot change agent geometry')
            return mesh_dir,{'parts':parts}
    # Height remains fixed by the pose-parameter selection. Do not invent
    # a support verification label for an unapproved candidate.
    initial={**info['pose'],'local_top':info['pose']['support_z']}
    observed=evidence.load(s,obj,parts=True)
    result=optimize(FixedGeometry(),{'params':{}},observed,initial,budget,optional(),
                    parameter_names=[],pose_parameters=['xy','yaw_deg'],anchor_body=True)
    metrics,_=evidence.score(mesh['V'],mesh['F'],result['pose'],observed,model_parts=parts)
    # The search raster and exact CPU raster do not have identical boundary
    # cells. A subpixel-scale CPU polish can resolve the final pose instead of
    # loosening gates or changing labels to absorb that numerical difference.
    # Height and geometry remain immutable; all source views still score.
    if not metrics['ok'] and evidence.search_objective(metrics)-metrics['loss']<.2:
        from .pose_polish import run as polish
        result['pose'],metrics,result['cpu_pose_polish']=polish(
            mesh['V'],mesh['F'],result['pose'],observed,model_parts=parts,
            initial_metrics=metrics,checkpoint=out/'cpu_progress.json')
    # GPU search termination and its metrics cannot approve a CPU source gate.
    result.update(metrics=metrics,ok=bool(metrics['ok']),proposal_only=True,task_approved=False,
                  input_candidate_sha256=digest(candidate),input_mesh_sha256=digest(mesh_dir/'mesh.npz'))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'candidate.json').write_text(json.dumps({**info,'pose':result['pose'],
        'proposal_only':True,'task_approved':False},indent=2)+'\n')
    from .part_evidence import sheets
    visual=sheets(s,evidence.placed(mesh['V'],mesh['F'],result['pose']),observed[0],parts,out/'source_comparison')
    (out/'visual.json').write_text(json.dumps(visual,indent=2)+'\n')
    print('POSE_PROPOSAL',metrics['ok'],[(r['iou_median'],r['local_difference']) for r in metrics.get('source_parts',[])],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('object');parser.add_argument('candidate')
    parser.add_argument('--output',required=True);parser.add_argument('--budget',type=int,default=300)
    args=parser.parse_args()
    run(Session(args.session),args.object,args.candidate,args.output,budget=args.budget)


if __name__=='__main__':main()
