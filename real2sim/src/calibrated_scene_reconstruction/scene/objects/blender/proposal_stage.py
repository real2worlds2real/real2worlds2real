"""Optional source-body initialization before jointly checking every part.

The actual agent Blender mesh is always used. Independently observed flexible
parts are unknown during this proposal only; the final full observations,
functional-part gates and native task checks remain mandatory.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def settings(s,obj):
    value=s.scene_graph['dynamic_objects'][obj]['fit'].get('initial_rigid_stage')
    if value is None:return None
    if (not isinstance(value,dict) or set(value)!={'parameters','exclude_links','budget'}
            or type(value['budget']) is not int or not 10<=value['budget']<=500
            or any(not isinstance(value[k],list) or not value[k] or len(set(value[k]))!=len(value[k])
                   or any(not isinstance(n,str) or not n for n in value[k]) for k in ['parameters','exclude_links'])):
        raise ValueError('Initial rigid proposal requires named parameters/links and a bounded budget')
    return value


def observations(s,obj,full,exclude_links):
    z,frames,masks,windows=full;cameras=dict(z);cameras.pop('_source_parts',None)
    cameras.pop('_source_material',None)
    source_parts={}
    for item in z.get('_source_parts',[]):
        names=item.get('links') or [item['link']]
        overlap=set(names)&set(exclude_links)
        if overlap and overlap!=set(names):
            raise ValueError('A jointly labelled source group must be excluded as a whole during rigid initialization')
        source_parts.update({name:item['prompt']for name in names})
    if not set(exclude_links)<=source_parts.keys():raise ValueError('Proposal links need independent source-part semantics')
    path=s/'splat/object_fit'/f'{obj}.npz'
    with np.load(path,allow_pickle=True) as data:prompts=str(data['prompt']).split('|')
    with np.load(path.with_suffix('.components.acquired.npz')) as data:
        names=list(map(str,data['names']));raw=data['part_masks']
    # A corrected source identity is authoritative during initialization too.
    # Reusing the raw union here can remove a neighboring object's pixels or
    # hide the very rigid structure that the proposal is meant to constrain.
    from ..part_label_review import load as reviewed_labels
    expected=[Path(str(z['names'][i])).name for i in frames]
    excluded_parts={};by_prompt={}
    for link in exclude_links:
        prompt=source_parts[link]
        if prompt not in by_prompt:
            by_prompt[prompt],_=reviewed_labels(
                s,obj,prompt,names,raw[:,prompts.index(prompt)],expected)
        excluded_parts[link]=by_prompt[prompt]
    # Subtraction alone is insufficient when the whole-object and flexible
    # queries disagree: leftover leaf/sheet pixels are not rigid evidence.
    # Prefer independently labelled retained parts when available. Unlabelled
    # regions inside the full source silhouette remain unknown for this
    # initializer; the final whole-object loop still evaluates them all.
    retained = [item for item in z.get('_source_parts', [])
                if not set(item.get('links') or [item['link']])&set(exclude_links)
                and item.get('observations') is not None]
    known=[];positive=[];selected=[];unobserved=[]
    for k,(i,mask,window) in enumerate(zip(frames,masks,windows)):
        excluded=np.zeros(raw.shape[-2:],bool);name=Path(str(z['names'][i])).name
        parts=[excluded_parts[link][names.index(name)] for link in exclude_links]
        # An unobserved flexible label is unknown, not proof of rigid material.
        # Using the whole positive silhouette in that view would enlarge the
        # rigid body to explain a loose sheet or foliage. This bounded proposal
        # uses only separable views; final full-object evidence keeps every view.
        if any(not part.any() for part in parts):
            unobserved.append(name);continue
        for part in parts:excluded|=part
        x,y,w,h,_=window;x,y,w,h=map(int,[x,y,w,h])
        region=cv2.resize(excluded[y:y+h,x:x+w].astype('uint8'),mask.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
        source_positive = mask
        domain=(z['_object_domains'][k].copy() if '_object_domains' in z else np.ones_like(mask))&~region
        if retained:
            source_positive = retained_positive(retained, name, window, mask.shape)
            if not source_positive.any():
                unobserved.append(name);continue
            domain=(z['_object_domains'][k].copy() if '_object_domains' in z else np.ones_like(mask))&(~region|source_positive)
            domain &= source_positive | ~mask
        known.append(domain);positive.append(source_positive&domain);selected.append(k)
    if sum(int(m.sum())>=50 for m in positive)<8:raise ValueError('Rigid proposal needs eight independently visible body views')
    cameras['_object_domains']=known;cameras['_proposal_excluded_links']=list(exclude_links)
    cameras['_proposal_unobserved_part_frames']=unobserved
    cameras['_proposal_positive_basis']='independent_retained_parts' if retained else 'full_minus_observed_flexible_parts'
    if retained:
        from .rigid_supervision import retained_sources
        occlusions={}
        for index,name in enumerate(names):
            parts=[excluded_parts[link][index] for link in exclude_links]
            occlusions[name]=np.logical_or.reduce(parts) if all(part.any() for part in parts) else None
        cameras['_source_parts']=retained_sources(retained,exclude_links,occlusions)
    return cameras,np.asarray(frames)[selected],positive,np.asarray(windows)[selected]


def retained_positive(parts, name, window, shape):
    """Map independently acquired source labels between native-image crops."""
    result=np.zeros(shape,bool)
    x,y,w,h,_=window
    for part in parts:
        cameras,frames,masks,windows=part['observations']
        for frame,mask,source_window in zip(frames,masks,windows):
            if Path(str(cameras['names'][frame])).name != name:continue
            sx,sy,sw,sh,_=source_window
            transform=np.array([[shape[1]/w*sw/mask.shape[1],0.,(sx-x)*shape[1]/w],
                                [0.,shape[0]/h*sh/mask.shape[0],(sy-y)*shape[0]/h]])
            result |= cv2.warpAffine(mask.astype('uint8'),transform,shape[::-1],
                                     flags=cv2.INTER_NEAREST,borderValue=0)>0
    return result


def shape_hypotheses(builder,spec,observed,initial,values,rasterizer=None):
    """Bounded geometry initialization scored against all source constraints."""
    if not isinstance(values,list) or not 1<=len(values)<=32:
        raise ValueError('Use1..32 bounded source-geometry parameter hypotheses')
    from . import evidence
    from .fit import rigid_anchor
    from .contract import parameters
    defaults={name:p['init'] for name,p in spec['params'].items()}
    path,report=builder.evaluate(defaults)
    with np.load(path/'mesh.npz') as mesh:anchor=rigid_anchor(mesh['V'],report.get('parts'))
    angle=np.radians(initial['yaw_deg']);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    centre=np.asarray(initial['xy'])+rotation@anchor
    rows=[];best=None
    for choice in [{}]+values:
        if not isinstance(choice,dict) or not choice.keys()<=spec['params'].keys():
            raise ValueError('Geometry hypotheses must name declared model parameters')
        candidate={**defaults,**choice};parameters(spec,candidate)
        try:
            path,report=builder.evaluate(candidate)
            with np.load(path/'mesh.npz') as mesh:
                pose={**initial,'xy':(centre-rotation@rigid_anchor(mesh['V'],report.get('parts'))).tolist()}
                metrics,_=evidence.score(mesh['V'],mesh['F'],pose,observed,rasterizer=rasterizer,model_parts=report.get('parts'))
            row={'params':candidate,'pose':pose,'metrics':metrics,'search_loss':evidence.search_objective(metrics)}
            if best is None or (metrics['ok'],-row['search_loss'])>(best['metrics']['ok'],-best['search_loss']):best=row
        except ValueError as error:row={'params':candidate,'error':str(error)}
        rows.append(row)
    if best is None:raise ValueError('No valid source-geometry hypothesis')
    return best,{'hypotheses':rows,'selected_params':best['params'],'proposal_only':True,
                 'scope':'Geometry initialization only; every source, native response and task gate remains required'}
