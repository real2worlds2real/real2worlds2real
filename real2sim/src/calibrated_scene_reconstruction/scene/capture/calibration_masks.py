"""Acquire reproducible semantic observation revisions for either robot-scale route."""
import argparse
import re
import subprocess
import sys

from calibrated_scene_reconstruction.scene.session import Session


def prompt_argument(value, name):
    if (not isinstance(value, list) or not value or
            any(not isinstance(p, str) or not p.strip() or any(c in p for c in ',\r\n') for p in value)):
        raise ValueError(name+' must be a nonempty list of individual prompt strings')
    return ','.join(value)


def plan(session, config_name):
    if config_name not in ('robot_visual.json', 'robot_calibration.json'):
        raise ValueError('Select visual-only or uncertain-joint calibration configuration')
    observations=session.config(config_name, {}).get('observations', {})
    if not isinstance(observations, dict):raise ValueError('observations must be an object')
    jobs=[dict(name='robot', suffix=observations.get('robot_suffix', '_visual_robot_v2'),
               prompts=observations.get('robot_prompts',
                   ['humanoid robot','robot torso','robot head','wheeled robot base'])),
          dict(name='hands', suffix=observations.get('hands_suffix', '_visual_hands'),
               prompts=observations.get('hands_prompts', ['robot hand','robot fingers']),
               annotation=observations.get('hands_annotations')),
          dict(name='occluder', suffix=observations.get('occluder_suffix', '_visual_occluders_v3'),
               prompts=observations.get('occluder_prompts'), role='calibration_occluders')]
    if observations.get('background_suffix') is not None:
        jobs.append(dict(name='background', suffix=observations['background_suffix'],
                         prompts=observations.get('background_prompts',
                             ['floor','wall','ceiling','desk','cabinet','monitor','chair','door','partition'])))
    suffixes=[]
    for job in jobs:
        suffix=job['suffix']
        if not isinstance(suffix, str) or not re.fullmatch(r'_[A-Za-z0-9_]+', suffix):
            raise ValueError('Calibration mask suffix must be a local observation revision')
        if suffix in suffixes:raise ValueError('Calibration observation roles require distinct mask suffixes')
        suffixes.append(suffix)
        job['arguments']=['--suffix', suffix, '--threshold', '0.3', '--every', '4', '--inside', '0', '--reuse',
                          '--all-capture-frames']
        if job['prompts'] is not None:
            job['arguments']+=['--prompts', prompt_argument(job['prompts'], job['name']+'_prompts')]
        elif job.get('role'):job['arguments']+=['--role', job['role']]
        else:raise ValueError(job['name']+'_prompts must be a nonempty list')
    return jobs


def run(session, config_name):
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: acquire observations in a new session')
    jobs=plan(session, config_name)
    logs=session/'logs';logs.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if job.get('annotation'):
            from .annotation_masks import write
            path = session.config_dir / job['annotation']
            if not path.resolve().is_relative_to(session.config_dir.resolve()):
                raise ValueError('Observation annotations must belong to this capture configuration')
            write(session, path, job['suffix'])
            print('[calibration.masks] '+job['name']+': reviewed source regions', flush=True)
            continue
        with (logs/('visual_'+job['name']+'_masks.log')).open('w') as log:
            subprocess.run([sys.executable, '-m', 'calibrated_scene_reconstruction.scene.background.splat.masks',
                            str(session.dir), *job['arguments']], stdout=log, stderr=subprocess.STDOUT, check=True)
        print('[calibration.masks] '+job['name']+': '+job['suffix'], flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--config', choices=['robot_visual.json','robot_calibration.json'], required=True)
    args=parser.parse_args()
    run(Session(args.session), args.config)


if __name__=='__main__':main()
