"""Route unresolved background appearance to agent mesh work, not GS coverage.

Image diagnostics propose work; an agent still reviews source and free views.
A planned mesh is not automatically a correct or photorealistic mesh.
"""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def detail(source,captured,mask):
    if source.shape!=captured.shape or mask.shape!=source.shape[:2]:raise ValueError('Aligned source/render/mask required')
    raw=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY).astype('float32')/255
    rendered=cv2.cvtColor(captured,cv2.COLOR_BGR2GRAY).astype('float32')/255
    high=lambda a:cv2.GaussianBlur(abs(a-cv2.GaussianBlur(a,(0,0),1.2)),(0,0),2.)
    expected,actual=high(raw),high(rendered);textured=mask&(expected>.012)
    blurred=textured&(actual<expected*.6)
    return {'pixels':int(mask.sum()),'detail_pixels':int(textured.sum()),'blur_fraction':float(blurred.sum()/max(1,textured.sum())),
            'rgb_mae':float(abs(source.astype(float)-captured).mean(2)[mask].mean()/255) if mask.any() else None}


def propose(session,review,observations,output):
    review=Path(review).resolve();render=json.loads(review.read_text())
    if render.get('diagnostic')!='context_source':raise ValueError('Use the fixed original GS capture/free-view review')
    cfg=session.config('environment.json');definitions=cfg['parameters']['entities'];names={e['id'] for e in definitions}
    aliases={n:n for n in names}
    # Cross-session identity associations must be explicit. No class-name merge.
    for entity,row in cfg.get('observation_aliases',{}).items():
        for alias in row:
            if alias in aliases and aliases[alias]!=entity:raise ValueError('Ambiguous observation alias')
            aliases[alias]=entity
    inputs={str(review):digest(review),str(Path(__file__).resolve()):digest(__file__)};views={Path(v['source']).name:v for v in render['views']};scores={n:[] for n in names}
    for path in observations:
        path=Path(path).resolve();inputs[str(path)]=digest(path)
        for row in json.loads(path.read_text())['records']:
            entity=aliases.get(row['entity']);view=views.get(Path(row['source']).name)
            if entity not in names or view is None or not row.get('mask'):continue
            source=Path(row['source']).resolve();mask=Path(row['mask']).resolve()
            for p,sha in [(source,row['source_sha256']),(mask,row['mask_sha256'])]:
                if digest(p)!=sha:raise ValueError('Appearance observation changed')
                inputs[str(p)]=sha
            image=cv2.imread(view['renders'][0]);raw=cv2.resize(cv2.imread(str(source)),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA)
            region=cv2.resize(cv2.imread(str(mask),0),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)>128
            for p in view['renders']:inputs[str(Path(p).resolve())]=digest(p)
            scores[entity].append({'source':str(source),'captured':view['renders'][0],'novel':view['renders'][1],**detail(raw,image,region)})
    routes={e['id']:{'route':'mesh','quality':'unresolved','reason':'Blur, coverage and unseen structure require agent mesh work until capture AND free-view fidelity are accepted.',
                         'scores':scores[e['id']],'free_view_review':{'pass':False,'note':'Agent review required'}} for e in definitions}
    q={'version':1,'routes':routes,'input_sha256':inputs,'protected_task_state':protected_state(session),
       'policy':{'blurred':'agent_mesh','unobserved':'agent_complete_mesh','unknown':'agent_mesh','sharp_observed':'gs_only_after_review'},
       'scope':'Source-backed appearance work plan, not shape/physics/photorealism acceptance. No true dimensions.'}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('Use a new appearance work plan')
    path.write_text(json.dumps(q,indent=2)+'\n');return q


def validate(session,path,ownership):
    path=Path(path).resolve();q=json.loads(path.read_text())
    if q.get('version')!=1 or q.get('protected_task_state')!=protected_state(session):raise ValueError('Appearance plan belongs to a different task state')
    expected=set(ownership)-{n for n,r in ownership.items() if r['role']=='inferred_enclosure'}
    if set(q.get('routes',{}))!=expected:raise ValueError('Appearance plan must cover every observed background entity')
    for entity,row in q['routes'].items():
        if row.get('route') not in ('gs','hybrid','mesh') or not row.get('reason','').strip():raise ValueError('Explicit GS/mesh decision and basis required')
        role=ownership[entity]['role'];review=row.get('free_view_review',{})
        if row['route']=='mesh' and role=='gs':raise ValueError('Blurred/unresolved appearance cannot remain GS-owned: '+entity)
        if row['route'] in ('gs','hybrid'):
            scores=row.get('scores',[])
            if (role!=('gs' if row['route']=='gs' else 'gap_fill') or row.get('quality')!='reliable' or review.get('pass') is not True or not review.get('note','').strip()
                    or len({r['source'] for r in scores})<2):raise ValueError('Retained GS needs source and free-view reliability evidence: '+entity)
            sharp={r['source'] for r in scores if r.get('detail_pixels',0)>=32 and r.get('blur_fraction',1)<=.2 and r.get('rgb_mae',1)<=.12}
            if len(sharp)<2:raise ValueError('Retained GS must resolve source detail in two views: '+entity)
            if any(p not in q['input_sha256'] for r in scores for p in (r['source'],r['captured'],r['novel'])):
                raise ValueError('Retained GS review images must be bound')
    inputs={str(path):digest(path),**q['input_sha256']}
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Appearance plan evidence changed: '+p)
    return q,inputs


def required_parts(requirements,rows,geometry=None):
    if not isinstance(requirements,dict):raise ValueError('Assembly completion requirements must be an entity dictionary')
    for entity,groups in requirements.items():
        if not isinstance(groups,list) or not groups:raise ValueError('Explicit required subassemblies needed')
        names={name for r in rows if r['entity']==entity for name in ([r['object']] if 'object' in r else r['path'].split('/'))}
        for group in groups:
            if (group.get('provenance') not in ('observed','inferred') or not group.get('basis','').strip()
                    or not group.get('parts') or not set(group['parts'])<=names):
                raise ValueError('Required observed/inferred assembly parts absent from actual mesh: '+entity)
            if geometry is not None:
                for part,ends in group.get('segments',{}).items():
                    ends=np.asarray(ends,float);tol=group.get('tolerance_m',.05)
                    if ends.shape!=(2,3) or not np.isfinite(ends).all() or not .001<=tol<=.1:
                        raise ValueError('Bounded world-space structural endpoints required')
                    arrays=[geometry[f'v{i}'] for i,r in enumerate(rows) if r['entity']==entity and part in r['path'].split('/')]
                    if not arrays:raise ValueError('Structural segment missing from exported geometry')
                    vertices=np.concatenate(arrays)
                    length=np.linalg.norm(ends[1]-ends[0])
                    if length<=1e-6:raise ValueError('Nonzero structural member required')
                    projected=(vertices-ends[0])@((ends[1]-ends[0])/length)
                    if abs(projected.min())>.005 or abs(projected.max()-length)>.005:
                        raise ValueError('Structural beam stops short of its declared connections: '+part)
                    if any(np.linalg.norm(vertices-e,axis=1).min()>tol for e in ends):
                        raise ValueError('Structural beam does not reach its declared connections: '+part)
    return True


def entity_scope(parameters, exported_rows=None):
    """Enforce this capture's explicit exclusions on plans and actual meshes.

    Exclusion IDs are scoped to the requested environment, not a global ban on
    equipment or people. An agent still resolves source-instance identities;
    unrelated fixed luminaires do not inherit a capture-light exclusion.
    """
    excluded=parameters.get('excluded_entities',[])
    if (not isinstance(excluded,list) or any(not isinstance(n,str) or
            not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',n) for n in excluded)
            or len(set(excluded))!=len(excluded)):
        raise ValueError('Explicit exclusions must be distinct environment entity IDs')
    present={r.get('id') for r in parameters.get('entities',[])}
    if exported_rows is not None:present.update(r.get('entity') for r in exported_rows)
    forbidden=sorted(set(excluded)&present)
    if forbidden:raise ValueError('Explicitly excluded environment entities were reconstructed: '+', '.join(forbidden))
    return excluded


def material_continuity(parameters, exported_rows=None):
    """Check agent-declared shared surfaces across separately built wall pieces.

    Instance membership and material identity are explicit capture decisions;
    this does not impose one material on all walls or certify their appearance.
    """
    groups=parameters.get('surface_material_groups',[])
    if not isinstance(groups,list):raise ValueError('Surface material groups must be a list')
    entities={r['id']:r for r in parameters.get('entities',[])}
    for group in groups:
        members=group.get('members',[])
        if (not group.get('basis','').strip() or len(group.get('source_views',[]))<2
                or not group.get('material') or not group.get('exported_material')
                or len(members)<2 or any(not m.get('parts') for m in members)):
            raise ValueError('Shared surface requires explicit members, parts, material and source basis')
        for member in members:
            entity=member['entity']
            if entity not in entities or entities[entity].get('material')!=group['material']:
                raise ValueError('Continuous source surface changed material: '+entity)
            if exported_rows is not None:
                for part in member['parts']:
                    rows=[r for r in exported_rows if r['entity']==entity and r['object']==part]
                    if len(rows)!=1 or rows[0].get('materials')!=[group['exported_material']]:
                        raise ValueError('Exported continuous surface lost its declared material: '+entity+'/'+part)
    return groups


def assembly_policy(parameters,session=None):
    entity_scope(parameters)
    material_continuity(parameters)
    requirements=parameters.get('assembly_requirements',{})
    for entity in parameters.get('entities',[]):
        if entity.get('kind')!='frame':continue
        exception=parameters.get('completion_exceptions',{}).get(entity['id'])
        if exception:
            evidence=exception.get('source_evidence',[])
            if (exception.get('open_top') is not True or not exception.get('basis','').strip()
                    or len({r.get('path') for r in evidence})<2):raise ValueError('Observed open frame needs two bound source views')
            for row in evidence:
                if digest(row['path'])!=row.get('sha256'):raise ValueError('Open-frame evidence changed')
                if session is not None and digest(session/'splat/data/images'/Path(row['path']).name)!=row['sha256']:
                    raise ValueError('Open-frame evidence must come from this capture')
            continue
        groups=requirements.get(entity['id'],[])
        if not any(g.get('id')=='upper_perimeter' and len(g.get('parts',[]))>=len(entity.get('columns',[]))>=3 for g in groups):
            raise ValueError('Frame requires an observed/inferred upper perimeter completion contract; unknown is not permission to omit it')
    return requirements


def semantic_keep(claims,allowed):
    if not isinstance(allowed,list) or not allowed or any(type(i)is not int or i<0 for i in allowed) or len(set(allowed))!=len(allowed):
        raise ValueError('Distinct approved GS claim IDs required')
    return np.isin(np.asarray(claims),allowed)


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('review');p.add_argument('--observations',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();propose(Session(a.session),a.review,a.observations,a.output)


if __name__=='__main__':main()
