"""Bounded native IK proposal search for an explicitly initialized pregrasp.

This qualification tool moves only the isolated IK robot and never reports
task success. The actual robot/object source geometry and base remain fixed.
Every selected proposal still requires native contact and released-outcome tests.
"""
import argparse
import importlib
import itertools
from pathlib import Path
import traceback

import numpy as np

from bound_insertion_config import policy_io
from bound_training import read, write_new
from astra_e2e.oversight.evidence import file_hash


def rotations():
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.),repeat=3):
            matrix=np.eye(3)[:,order]*signs
            if np.linalg.det(matrix)>.5:
                yield matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('session','config','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=read(args.config)
    binding={'schema':'astra_unbound_native_probe_v1','world_sha256':'unbound_ik_scan',
             'task_id':'source_pregrasp_ik_proposal','backend_id':config['backend_id'],
             'native_session':str(Path(args.session).resolve()),'policy_io':config.get('policy_io',policy_io())}
    module,name=config['factory'].split(':')
    environment=None;rows=[]
    report={'schema':'astra_native_pregrasp_scan_v1','is_policy_receipt':False,'qualified':False,
            'config':str(Path(args.config).resolve()),'config_sha256':file_hash(args.config),
            'scope':'Native IK proposals only; no scene geometry/base changes or physical task outcomes.'}
    try:
        environment=getattr(importlib.import_module(module),name)(binding,{'operation':'qualification_probe','diagnostic_directory':str(output)},config)
        opening=environment.context.opening(environment.receptacle)
        bounds=np.asarray(config['environment']['initial_offset_bounds_m'])
        initial_z=float(bounds[2].mean())
        origin=np.r_[opening['xy']-environment.center_xy,opening['rim_z']+initial_z-environment.bottom_offset]
        upright_points=environment.local_points@environment.upright.T
        choices=[environment.rotation,*rotations()]
        # Centre pose only; initial-distribution endpoints are verified later.
        for above in (.03,.06,.09):
            offset=np.array([0.,0.,float(upright_points[:,2].max())+above])
            for index,rotation in enumerate(choices):
                q,error=environment.ik.solve_ik(environment.side,origin+offset+environment.ik_offset,
                       rotation,environment.home,iters=450,tol=(.0005,.005))
                row={'tcp_above_object_top_m':above,'rotation_index':index,'tcp_rotation':rotation.tolist(),
                     'tcp_from_object_origin_m':offset.tolist(),'error_m_rad':list(error),
                     'joint_positions':q.tolist(),'joint_names':environment.ik.names,
                     'passes_existing_tolerance':error[0]<=config['environment']['reset_position_tolerance_m'] and error[1]<=config['environment']['reset_rotation_tolerance_rad']}
                rows.append(row)
                if row['passes_existing_tolerance']:
                    print({'reachable':len(rows),'above':above,'error_m_rad':error},flush=True)
            write_new(output/('offset_'+str(above)+'.json'),{'candidates':rows[-len(choices):]})
        report['native_physics']=environment.diagnostics()['native_physics_errors']
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error);report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    finally:
        report['candidates']=rows
        report['reachable']=sum(row['passes_existing_tolerance'] for row in rows)
        write_new(output/'scan_report.json',report)
        print({'scan_report':str(output/'scan_report.json'),'reachable':report['reachable'],'total':len(rows)},flush=True)
        if environment is not None:environment.close()


if __name__=='__main__':main()
