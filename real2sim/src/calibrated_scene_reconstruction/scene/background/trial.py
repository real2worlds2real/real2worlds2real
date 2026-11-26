"""Export and visually compare an isolated background training trial.

Waits for an explicitly named training job, then produces diagnostics only.
Never updates the active environment configuration or approves a scene.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT,resolve
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
from calibrated_scene_reconstruction.scene.session import Session


def unchanged_world(actual,expected):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,atol=1e-7,rtol=0):
        raise ValueError('Background trial changed the calibrated coordinate frame')


def run(session,training_exit,usd,cameras,reference,output):
    from calibrated_scene_reconstruction.scene.background.splat.training_data import validate
    from calibrated_scene_reconstruction.scene.background.splat.config import settings
    from calibrated_scene_reconstruction.scene.review.field_comparison import compare
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a background derivative')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Background trial output belongs in its session')
    out.mkdir(parents=True,exist_ok=False)
    paths=[resolve(p).resolve() for p in (usd,cameras,reference)]
    usd,cameras,reference=paths;baseline=json.loads(reference.read_text());protected=protected_state(session)
    inputs={str(p):digest(p) for p in [*paths,session.config_dir/'splat.json',Path(__file__).resolve()]}
    status={'pass':False,'photorealism_approved':False,'protected_task_state':protected,'input_sha256':inputs,
            'scope':'Isolated training/export/actual-view diagnostics only. No active environment or task model promotion.'}
    def update(phase,**fields):
        status.update(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**fields)
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def check():
        if protected_state(session)!=protected:raise ValueError('Protected task changed during background trial')
        for p,sha in inputs.items():
            if digest(p)!=sha:raise ValueError('Background trial input changed: '+p)
    def command(argv,name):
        with (out/(name+'.log')).open('w') as log:subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        update('waiting_for_training',training_exit=str(Path(training_exit).resolve()))
        while not Path(training_exit).is_file():time.sleep(10)
        if Path(training_exit).read_text().strip()!='0':raise ValueError('Background training failed; no export or promotion')
        check();record=session/'splat/train.json';training=json.loads(record.read_text())
        if training['settings']!=settings(session)['train']:raise ValueError('Training result belongs to different settings')
        data=resolve(session.dir/training['data']).resolve()
        if not data.is_relative_to(session.dir.resolve()):raise ValueError('Training snapshot belongs in this session')
        snapshot=validate(data)
        checkpoint=resolve(session.dir/training['checkpoint']).resolve()
        if not checkpoint.is_relative_to(session.dir.resolve()) or not checkpoint.is_file():raise ValueError('Current session training checkpoint required')
        inputs[str(record.resolve())]=digest(record);inputs[str(checkpoint)]=digest(checkpoint)
        update('exporting',training_record=str(record.resolve()),held_out=training['held_out'],training_snapshot=snapshot)
        command(['tools/run_splat.sh','-m','calibrated_scene_reconstruction.scene.background.splat.export',str(session.dir)],'export')
        check();transform=json.loads((session/'scene/splat/transform.json').read_text())
        unchanged_world(transform['matrix'],json.loads(baseline['environment_context']['matrix_json']))
        package=session/'scene/splat/background.usdz';field=out/'trial.nurec'
        with zipfile.ZipFile(package) as archive:
            members=[n for n in archive.namelist() if n.endswith('.nurec')]
            if len(members)!=1:raise ValueError('Exactly one exported NuRec payload required')
            with archive.open(members[0]) as source,field.open('wb') as target:shutil.copyfileobj(source,target)
        update('rendering',exported_asset=str(package.resolve()),exported_sha256=digest(package))
        command(['tools/run_isaac.sh','apps/scene/review_usd_cameras.py',str(usd),str(cameras),'--renderer','rt','--context-only',
                 '--background-field',str(field),'--output',str(out/'renders')],'render')
        check();result=compare(out/'renders/review.json',reference,out/'comparison')
        update('complete_unapproved',comparison=str(out/'comparison/comparison.json'),mean=result['mean'])
        return status
    except BaseException as exc:
        update('failed',error=str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('--training-exit',required=True)
    p.add_argument('--usd',required=True);p.add_argument('--cameras',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(Session(a.session),a.training_exit,a.usd,a.cameras,a.reference,a.output)


if __name__=='__main__':main()
