"""Use only the agent-declared metric source and observation interval."""
import json
import re
from pathlib import Path
from calibrated_scene_reconstruction.core.media import frame_timestamp
from calibrated_scene_reconstruction.core.paths import ROOT
from .supplementary import load


def select(plan,videos,names,*,timestamp=frame_timestamp):
    if plan is None:return set(names)
    scale=plan['metric_scale'];asset=next(item for item in plan['assets'] if item['id']==scale['source'])
    source=(ROOT/Path(asset['path'])).resolve()
    uses=[use for use in asset['uses'] if use['role']=='metric_calibration' and use.get('method')==scale['method']]
    allowed=set()
    for name in names:
        match=re.fullmatch(r'(\d+)_(\d+)\.jpg',name)
        if match is None:raise ValueError('Calibration frame has no registered media/frame identity: '+name)
        video,frame=map(int,match.groups())
        if video>=len(videos):raise ValueError('Calibration frame refers to an unknown registered stream')
        path=(ROOT/Path(videos[video])).resolve()
        if path!=source:continue
        seconds=timestamp(path,frame)
        if any(lo<=seconds<hi for lo,hi in [use['interval_s'] for use in uses]):allowed.add(name)
    return allowed


def allowed(session,names):
    plan=load(session)
    if plan is None:return set(names)
    video=json.loads((session/'capture/keyframes.json').read_text())['video']
    extra=session.config('splat.json',{}).get('videos',[])
    return select(plan,[video,*[path for path in extra if path!=video]],names)
