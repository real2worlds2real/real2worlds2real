"""Robust relative exposure/WB alignment using overlapping surface observations.

This estimates image-to-image radiance consistency, not true intrinsic albedo.
Disconnected view groups remain explicitly uncalibrated relative to each other.
"""
import numpy as np


def align(samples, weights, *, max_samples=20000, minimum_overlap=64):
    S,W=np.asarray(samples,float),np.asarray(weights,float)
    if S.ndim!=3 or S.shape[2]!=3 or W.shape!=S.shape[:2]:
        raise ValueError('Surface samples need texels x views x RGB and matching weights')
    n=S.shape[1];chosen=np.linspace(0,len(S)-1,min(len(S),max_samples)).astype(int)
    values=S[chosen];valid=(W[chosen]>.02)&np.isfinite(values).all(2)&(values.min(2)>.005)
    edges=[];adj=[set() for _ in range(n)]
    for a in range(n):
        for b in range(a+1,n):
            common=valid[:,a]&valid[:,b]
            if common.sum()<minimum_overlap:continue
            difference=np.log(values[common,b])-np.log(values[common,a])
            shift=np.median(difference,axis=0)
            scatter=float(np.median(abs(difference-shift)))
            edges.append((a,b,shift,float(np.sqrt(common.sum())/(1+10*scatter))))
            adj[a].add(b);adj[b].add(a)
    components=[];remaining=set(range(n))
    while remaining:
        group={remaining.pop()};front=list(group)
        while front:
            found=adj[front.pop()]&remaining;remaining-=found;group|=found;front.extend(found)
        components.append(sorted(group))
    gains=np.ones((n,3))
    for group in components:
        if len(group)<2:continue
        indices={j:i for i,j in enumerate(group)};A=[];B=[]
        for a,b,shift,weight in edges:
            if a not in indices:continue
            row=np.zeros(len(group));row[indices[a]]=weight;row[indices[b]]=-weight;A.append(row);B.append(shift*weight)
        A.append(np.ones(len(group)));B.append(np.zeros(3))
        x=np.linalg.lstsq(np.asarray(A),np.asarray(B),rcond=None)[0]
        gains[group]=np.exp(np.clip(x,-np.log(2),np.log(2)))
    return S*gains[None],{'method':'robust overlapping-surface log RGB ratios; zero-mean log-gain gauge per connected group','view_gains':gains.tolist(),'connected_view_groups':components,'overlap_edges':len(edges),'absolute_albedo_identified':False}
