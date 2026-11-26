"""Conservative split contact solve between independently materialized particles.

Normal impulses prevent approaching particles from interpenetrating. Coulomb
tangent impulses transfer sliding momentum. No robot pose or target is accepted.
"""
import numpy as np
from scipy.spatial import cKDTree


def sheet_contact_geometry(fluid_p,cloth_p,faces):
    """Closest sheet points with barycentric reaction weights, in world space.

    Local candidates come from triangle centroids. Plane interiors and all
    three edges are tested; a cloth vertex is never treated as a rigid bead.
    """
    p=np.asarray(fluid_p,float);q=np.asarray(cloth_p,float);faces=np.asarray(faces,int)
    tri=q[faces];k=min(24,len(tri))
    if not k:raise ValueError('A sheet requires nonempty triangles')
    _,candidates=cKDTree(tri.mean(1)).query(p,k=k)
    candidates=np.asarray(candidates).reshape(len(p),k)
    t=tri[candidates];a,b,c=t[:,:,0],t[:,:,1],t[:,:,2]
    ab=b-a;ac=c-a;ap=p[:,None,:]-a
    dot=lambda x,y:np.sum(x*y,axis=-1)
    d00=dot(ab,ab);d01=dot(ab,ac);d11=dot(ac,ac)
    d20=dot(ap,ab);d21=dot(ap,ac);den=d00*d11-d01*d01
    u=(d11*d20-d01*d21)/np.maximum(den,1e-30)
    v=(d00*d21-d01*d20)/np.maximum(den,1e-30)
    weight=np.stack([1-u-v,u,v],axis=-1)
    closest=a+u[...,None]*ab+v[...,None]*ac
    distance2=dot(p[:,None,:]-closest,p[:,None,:]-closest)
    interior=(den>1e-24)&(weight>=-1e-10).all(-1)
    distance2=np.where(interior,distance2,np.inf)
    for i,j in [(0,1),(1,2),(2,0)]:
        start=t[:,:,i];edge=t[:,:,j]-start
        alpha=np.clip(dot(p[:,None,:]-start,edge)/np.maximum(dot(edge,edge),1e-30),0,1)
        point=start+alpha[...,None]*edge
        dist=dot(p[:,None,:]-point,p[:,None,:]-point)
        better=dist<distance2
        w=np.zeros_like(weight);w[:,:,i]=1-alpha;w[:,:,j]=alpha
        weight=np.where(better[...,None],w,weight)
        closest=np.where(better[...,None],point,closest)
        distance2=np.minimum(dist,distance2)
    row=np.arange(len(p));chosen=np.argmin(distance2,axis=1)
    nodes=faces[candidates[row,chosen]]
    return nodes,weight[row,chosen],closest[row,chosen],np.sqrt(distance2[row,chosen])


def contact_impulses(fluid_p,fluid_v,fluid_mass,cloth_p,cloth_v,cloth_mass,dt,
                     distance=.0022,friction=.10,faces=None,geometry=None):
    fv=np.asarray(fluid_v,float).copy();cv=np.asarray(cloth_v,float).copy()
    if not len(fluid_p):return fv,cv,dict(contacts=0,momentum_error=0.,impulse_Ns=0.)
    if faces is None:
        distances,nodes=cKDTree(cloth_p).query(fluid_p)
        nodes=nodes[:,None];weights=np.ones_like(nodes,dtype=float);closest=np.asarray(cloth_p)[nodes[:,0]]
    else:
        nodes,weights,closest,distances=sheet_contact_geometry(fluid_p,cloth_p,faces) if geometry is None else geometry
    contacts=np.flatnonzero(distances<distance)
    total=0.;maximum=0.
    before=fluid_mass*fv.sum(0)+(np.asarray(cloth_mass)[:,None]*cv).sum(0)
    for i in contacts:
        j=nodes[i];w=weights[i];n=(fluid_p[i]-closest[i])/max(distances[i],1e-12)
        if distances[i]<1e-12:
            if len(j)==3:
                n=np.cross(cloth_p[j[1]]-cloth_p[j[0]],cloth_p[j[2]]-cloth_p[j[0]])
                n=n/max(np.linalg.norm(n),1e-12)
            if np.linalg.norm(n)<1e-12 or len(j)!=3:n=np.array([0.,0.,1.])
        relative=fv[i]-(w[:,None]*cv[j]).sum(0);vn=float(relative@n)
        # Bounded Baumgarte stabilization adds separation velocity only where
        # actual particle volumes overlap, without prescribing any position.
        separation=min(.003,.03*(distance-distances[i])/dt)
        inverse_mass=1/fluid_mass+np.sum(w*w/cloth_mass[j])
        normal=max(0.,separation-vn)/inverse_mass
        tangent=relative-vn*n;speed=float(np.linalg.norm(tangent))
        impulse=normal*n
        if speed>1e-12:impulse-=min(speed/inverse_mass,friction*normal)*tangent/speed
        fv[i]+=impulse/fluid_mass;cv[j]-=w[:,None]*impulse/cloth_mass[j,None]
        magnitude=float(np.linalg.norm(impulse));total+=magnitude;maximum=max(maximum,magnitude)
    after=fluid_mass*fv.sum(0)+(np.asarray(cloth_mass)[:,None]*cv).sum(0)
    return fv,cv,dict(contacts=len(contacts),momentum_error=float(np.linalg.norm(after-before)),
                      impulse_Ns=total,maximum_pair_impulse_Ns=maximum)
