"""Necessary aggregate gravity balance on a planar support.

This diagnoses an observed unactuated pose before material fitting. It does
not identify masses, certify each loose part, or replace native contact tests.
"""
import numpy as np
from scipy.optimize import linprog,minimize


def check(points,center_of_mass,*,normal=(0.,0.,1.),plane_offset=0.,gravity=(0.,0.,-9.81),contact_tolerance_m=1e-4,friction=None):
    p=np.asarray(points,float);c=np.asarray(center_of_mass,float);n=np.asarray(normal,float);g=np.asarray(gravity,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not len(p) or c.shape!=(3,) or n.shape!=(3,) or g.shape!=(3,)
            or not np.isfinite(np.r_[p.ravel(),c,n,g,plane_offset,contact_tolerance_m]).all()
            or not np.isclose(n@n,1.,rtol=1e-6) or not contact_tolerance_m>0 or not np.linalg.norm(g)>0
            or friction is not None and (not np.isfinite(friction) or friction<0)):
        raise ValueError('Finite physical surface, mass centroid, unit support normal and bounded contact assumptions required')
    distance=p@n-plane_offset;selected=np.flatnonzero(abs(distance)<=contact_tolerance_m);down=float(g@n)
    result={'pass':False,'center_of_mass':c.tolist(),'contact_points':len(selected),
            'maximum_initial_penetration_m':max(0.,float(-distance.min())),
            'contact_tolerance_m':float(contact_tolerance_m),
            'scope':'Necessary aggregate planar-support balance for the declared mass hypothesis, gravity and no other external forces. It neither certifies individual loose parts nor identifies real material properties.'}
    if down>=-1e-12 or not len(selected):
        result['reason']='No compressive gravity support on the declared plane';return result
    projection=c+g*((plane_offset-c@n)/down);result['gravity_projection']=projection.tolist()
    tangent=g-n*down;required=float(np.linalg.norm(tangent)/(-down));result['minimum_global_friction']=required
    contacts=p[selected];contacts-=((contacts@n-plane_offset)[:,None])*n
    direction=np.eye(3)[np.argmin(abs(n))];u=direction-n*(direction@n);u/=np.linalg.norm(u);v=np.cross(n,u)
    xy=np.c_[contacts@u,contacts@v];target=np.array([projection@u,projection@v])
    A=np.vstack([np.ones(len(xy)),xy.T]);b=np.r_[1.,target]
    proof=linprog(np.zeros(len(xy)),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    feasible=bool(proof.success and np.linalg.norm(A@proof.x-b)<1e-7)
    result['force_balance_feasible']=feasible
    if feasible:
        ids=np.flatnonzero(proof.x>1e-9);result['reaction_weights']=[{'source_point':int(selected[i]),'weight':float(proof.x[i])} for i in ids]
        result['gravity_projection_distance_to_support_m']=0.
    else:
        objective=lambda w:float(np.sum((w@xy-target)**2))
        nearest=minimize(objective,np.full(len(xy),1/len(xy)),bounds=[(0,1)]*len(xy),
                         constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],method='SLSQP',options={'ftol':1e-15,'maxiter':200})
        result['gravity_projection_distance_to_support_m']=float(np.linalg.norm(nearest.x@xy-target)) if nearest.success else None
    result['pass']=bool(feasible and result['maximum_initial_penetration_m']<=contact_tolerance_m
                        and (friction is None or required<=friction+1e-9))
    if not result['pass']:result['reason']='Declared geometry, initial pose and mass hypothesis do not admit the required planar gravity balance; inspect support and material composition before adjusting rest curvature or stiffness'
    return result
