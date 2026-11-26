"""Source-ray proposals through one closed transparent shell, without changing geometry.

Uses Snell's law (PBRT4, Specular Reflection and Transmission, section9.3.4).
This is a two-interface geometric diagnostic, not a photometric renderer or a
shape/material certificate. Opaque labels must be included in the source face
mask; uncertain thickness/IOR require separate sensitivity hypotheses. Multiple
reflections, scattering and caustic brightness are not modeled.
"""
import numpy as np


def refract(direction, normal_against_ray, n_from, n_to):
    d,n=np.asarray(direction,float),np.asarray(normal_against_ray,float)
    if (d.ndim!=2 or d.shape[1:]!=(3,) or n.shape!=d.shape
            or not np.isfinite([d,n]).all() or not np.isfinite([n_from,n_to]).all()
            or min(n_from,n_to)<=0 or not np.allclose(np.linalg.norm(d,axis=1),1.,atol=1e-7)
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-7)):
        raise ValueError('Unit rays/normals and positive finite refractive indices required')
    cosine=-(d*n).sum(1)
    if (cosine < -1e-7).any():raise ValueError('Interface normal must oppose the incident ray')
    ratio=n_from/n_to;radicand=1-ratio**2*np.maximum(0,1-cosine**2)
    valid=radicand>=0
    result=ratio*d+(ratio*cosine-np.sqrt(np.maximum(0,radicand)))[:,None]*n
    result[~valid]=np.nan
    return result,valid


def first_hit(vertices,faces,origins,directions,*,near=1e-8):
    """Bounded-memory nearest positive triangle hit; preserve source face IDs."""
    t=np.asarray(vertices,float)[np.asarray(faces,int)]
    e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
    distances=np.full(len(origins),np.inf);ids=np.full(len(origins),-1,int)
    for start in range(0,len(origins),64):
        o=np.asarray(origins[start:start+64]);d=np.asarray(directions[start:start+64])
        p=np.cross(d[:,None],e2[None]);det=(e1[None]*p).sum(2)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=abs(det)>1e-12)
        delta=o[:,None]-t[None,:,0];u=(delta*p).sum(2)*inv
        q=np.cross(delta,e1[None]);v=(d[:,None]*q).sum(2)*inv
        distance=(e2[None]*q).sum(2)*inv
        good=(abs(det)>1e-12)&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(distance>near)
        candidate=np.where(good,distance,np.inf);index=candidate.argmin(1)
        best=candidate[np.arange(len(o)),index];distances[start:start+len(o)]=best
        ids[start:start+len(o)]=np.where(np.isfinite(best),index,-1)
    return distances,ids


def through_shell(vertices,faces,transmitting,origins,directions,*,ior=1.5,ambient_ior=1.):
    from calibrated_scene_reconstruction.assets.metric_mesh import geometry
    vertices=np.asarray(vertices,float);faces=np.asarray(faces);mask=np.asarray(transmitting)
    geometry(vertices,faces,max_extent_m=10.,min_extent_m=1e-5,require_bottom_origin=False)
    origins=np.asarray(origins,float);directions=np.asarray(directions,float)
    if (origins.ndim!=2 or origins.shape[1:]!=(3,) or not len(origins) or directions.shape!=origins.shape
            or not np.isfinite([origins,directions]).all() or (np.linalg.norm(directions,axis=1)<=0).any()
            or mask.shape!=(len(faces),) or mask.dtype!=bool or not np.isfinite([ior,ambient_ior]).all()
            or min(ior,ambient_ior)<=0):raise ValueError('Finite source rays, a source-face optical mask and positive IORs required')
    direction=directions/np.linalg.norm(directions,axis=1)[:,None]
    tri=vertices[faces];normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.linalg.norm(normals,axis=1)
    if (areas<=1e-14).any():raise ValueError('Degenerate optical faces cannot define interface normals')
    normals/=areas[:,None]
    points=np.full_like(origins,np.nan);rays=np.full_like(origins,np.nan);valid=np.zeros(len(origins),bool)
    first_t,first=first_hit(vertices,faces,origins,direction)
    selected=np.flatnonzero(first>=0)
    selected=selected[mask[first[selected]] & ((direction[selected]*normals[first[selected]]).sum(1)<0)]
    if len(selected):
        entry=origins[selected]+first_t[selected,None]*direction[selected]
        inside,ok=refract(direction[selected],normals[first[selected]],ambient_ior,ior)
        selected=selected[ok];entry=entry[ok];inside=inside[ok]
        second_t,second=first_hit(vertices,faces,entry+inside*1e-7,inside)
        hit=second>=0;hit[hit] &= mask[second[hit]] & ((inside[hit]*normals[second[hit]]).sum(1)>0)
        chosen=selected[hit]
        if len(chosen):
            outgoing,ok=refract(inside[hit],-normals[second[hit]],ior,ambient_ior)
            exit_point=entry[hit]+inside[hit]*(second_t[hit,None]+1e-7)
            points[chosen[ok]]=exit_point[ok];rays[chosen[ok]]=outgoing[ok];valid[chosen[ok]]=True
    return {'origins':points,'directions':rays,'valid':valid,'first_face':first,
            'proposal_only':True,'model':'Two-interface direct transmission; no reflection/scattering or source approval'}
