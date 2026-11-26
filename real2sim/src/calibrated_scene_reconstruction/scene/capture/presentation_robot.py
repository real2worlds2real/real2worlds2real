"""Prepare rendering standpoints without grasp, IK, reach or execution checks."""
import argparse
import json
import shutil

import numpy as np

from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame
from calibrated_scene_reconstruction.scene.session import Session


def prepare(session, robot):
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: preserve its rendering placements')
    graph=session.scene_graph
    spec=load_spec(robot)
    support=np.asarray(graph['furniture'][graph['task']['support']]['box'],float)
    target=(graph.get('dynamic_objects',{}).get(graph['task'].get('target')) or {}).get('fix') or {}
    point=np.r_[target.get('xy',support.mean(0)[:2]),float(support[1,2])]
    photo=graph.get('photo_robot') or {}
    base=photo.get('base')
    if base and 'xy' in base:
        pose=[*map(float,base['xy']),float(base['yaw_deg'])]
        source='captured/calibrated robot pose'
    else:
        x,y,yaw=TaskFrame(graph,point).sample_base(spec,np.random.default_rng(0))
        pose=[x,y,float(np.degrees(yaw))]
        source='inferred presentation standpoint; no measured robot pose available'
    record={'robot':robot,'object':point.tolist(),'scope':'presentation_only',
            'task_validation':'object_model','robot_execution_checked':False,
            'candidates':[{'id':0,'base':pose,'accepted':False,'presentation_only':True,'source':source}]}
    path=session/'placements'/('placements.json' if robot=='vega_1p_sharpa' else robot+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous==record:return path
        archive=path.parent/'prior_checks';archive.mkdir(exist_ok=True)
        number=1
        while (archive/f'{robot}_{number}.json').exists():number+=1
        shutil.copyfile(path,archive/f'{robot}_{number}.json')
    path.write_text(json.dumps(record,indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('--robot',required=True)
    args=parser.parse_args()
    print(prepare(Session(args.session),args.robot))


if __name__=='__main__':main()
