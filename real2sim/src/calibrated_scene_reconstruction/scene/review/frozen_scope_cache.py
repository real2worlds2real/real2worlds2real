"""Reuse frozen silhouettes only when actual exported foreground is identical.

Background edits need new appearance review, but cannot change a projection of
unchanged task/support/robot triangles from unchanged registered cameras.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from calibrated_scene_reconstruction.scene.review import frozen_scope
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def same_geometry(before, after):
    a={name:(v,f) for name,v,f in before};b={name:(v,f) for name,v,f in after}
    return (a.keys()==b.keys() and all(np.array_equal(a[k][0],b[k][0])
            and np.array_equal(a[k][1],b[k][1]) for k in a))


def margins(session):
    cfg=session.config('environment.json',{})
    if not cfg.get('splat_context'):return {'task_and_support':0.,'robot':0.}
    p=Path(cfg['assets'][cfg['splat_context']['evidence']]).resolve()
    data=json.loads(p.read_text())
    protection=data.get('field_trial') or data.get('capture_selection') or data.get('closure_support') or {}
    return {'task_and_support':float((protection.get('protected_overlap') or protection.get('protected') or {}).get('margin_m',.015)),
            'robot':float((protection.get('frozen_robot') or protection.get('robot') or {}).get('margin_m',.05))}


def reuse(session, previous, usd, cameras, output, variant='twin'):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    previous=Path(previous).resolve();usd=Path(usd).resolve();cameras=Path(cameras).resolve();out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()):raise ValueError('New session-local scope required')
    old=json.loads(previous.read_text())
    # Validate original bytes, cameras, implementation and masks before reuse.
    frozen_scope.validate(previous,{'usd':old['usd'],'camera_manifest':old['camera_manifest'],
                                   'input_sha256':old['input_sha256']})
    if (old['protected_task_state']!=protected_state(session) or old['camera_manifest']!=str(cameras)
            or old.get('projection_uncertainty_m')!=margins(session)):
        raise ValueError('Frozen state, cameras or GS protection margins changed')
    audit,_=current(session,variant)
    if audit['usd']!=str(usd) or audit['input_sha256'].get(str(usd))!=digest(usd):
        raise ValueError('Use the current actual audited export')
    graph=session.scene_graph;support=graph['task']['support'];objects=graph.get('dynamic_objects',{})
    before=frozen_scope.meshes(Usd.Stage.Open(old['usd']),support,objects)
    after=frozen_scope.meshes(Usd.Stage.Open(str(usd)),support,objects)
    if not same_geometry(before,after):raise ValueError('Actual frozen world triangles or instance paths changed')
    inputs={**old['input_sha256'],str(previous):digest(previous),str(usd):digest(usd),
            str(cameras):digest(cameras),str(Path(__file__).resolve()):digest(__file__)}
    cfg=session.config('environment.json',{})
    if cfg.get('splat_context'):
        p=Path(cfg['assets'][cfg['splat_context']['evidence']]).resolve();inputs[str(p)]=digest(p)
    out.mkdir(parents=True);rows=[]
    for old_row in old['rows']:
        row=dict(old_row);target=out/Path(row['mask']).name;shutil.copyfile(row['mask'],target)
        row.update(mask=str(target),mask_sha256=digest(target));inputs[str(target)]=digest(target);rows.append(row)
    result={**old,'usd':str(usd),'rows':rows,'input_sha256':inputs,
            'reuse':{'previous':str(previous),'exact_world_geometry':True,
                     'scope':'Identical frozen triangles, instance transforms, cameras and protection margins. Background appearance still requires a new actual render.'}}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Frozen projection reuse input changed')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


def prepare(session, usd, cameras, output, variant='twin'):
    root=session/'qa/environment_renders'/variant
    candidates=sorted(root.glob('round_*/frozen_scope/manifest.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    reasons=[]
    for previous in candidates:
        try:return reuse(session,previous,usd,cameras,output,variant)
        except (ValueError,KeyError,FileNotFoundError) as error:
            if Path(output).exists():raise
            reasons.append(str(error))
    result=frozen_scope.prepare(session,usd,cameras,output,variant)
    # Record a missed cache separately; do not alter or reinterpret its proof.
    (Path(output)/'cache_miss.json').write_text(json.dumps({'reasons':reasons},indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['session','usd','cameras']:p.add_argument(n)
    p.add_argument('--output',required=True);p.add_argument('--variant',default='twin')
    p.add_argument('--reuse-from');a=p.parse_args();s=Session(a.session)
    q=reuse(s,a.reuse_from,a.usd,a.cameras,a.output,a.variant) if a.reuse_from else prepare(s,a.usd,a.cameras,a.output,a.variant)
    print('Frozen scope:',len(q['rows']),'views;', 'exact geometry reused' if q.get('reuse') else 'projected anew')


if __name__=='__main__':main()
