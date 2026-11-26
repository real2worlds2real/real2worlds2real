"""Bounded free-sheet geometry initialization against an actual rigid solid.

Roots and intrinsic material coordinates stay fixed. This proposes authored
geometry before simulation; it never changes native states or approves a task.
"""
import numpy as np
from scipy.optimize import minimize
from .cloth_rest import triangle_edges


def project(points,faces,roots,query,*,clearance_m,maximum_displacement_m,
            maximum_edge_strain=.03,maximum_iterations=200,rest_points=None,exclude_root_faces=True):
    p=np.asarray(points,float);f=np.asarray(faces,int);root=np.asarray(roots,int)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,)
            or not len(f) or not np.isfinite(p).all() or f.min()<0 or f.max()>=len(p)
            or root.ndim!=1 or len(set(root))!=len(root) or root.size and (root.min()<0 or root.max()>=len(p))):
        raise ValueError('Actual finite sheet triangles and distinct real root nodes required')
    if (not 0<clearance_m<=.01 or not 0<maximum_displacement_m<=.002
            or not 0<maximum_edge_strain<=.1 or not 1<=maximum_iterations<=1000):
        raise ValueError('Explicit bounded geometry, thickness and material budgets required')
    rest=p if rest_points is None else np.asarray(rest_points,float)
    if rest.shape!=p.shape or not np.isfinite(rest).all():raise ValueError('Unchanged finite material coordinates required')
    free=np.setdiff1d(np.arange(len(p)),root);mapping=np.full(len(p),-1,int);mapping[free]=np.arange(len(free))
    selected=f[~np.isin(f,root).any(1)] if exclude_root_faces else f
    if not len(selected) or not len(free):raise ValueError('An actual free sheet surface is required')
    # Exactly the vertex, triangle-centre and edge-midpoint screen used by the
    # independent authored-state check. No unsampled-face acceptance is claimed.
    templates=np.r_[np.eye(3),np.full((1,3),1/3),[[.5,.5,0],[0,.5,.5],[.5,0,.5]]]
    ids=np.repeat(selected,len(templates),axis=0);weights=np.tile(templates,(len(selected),1))
    edges=triangle_edges(f);lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1)
    if lengths.min()<=1e-10:raise ValueError('Resolved material edges required')
    unit=maximum_displacement_m;cache={};size=len(free)*3
    def evaluate(x):
        if 'x' in cache and np.array_equal(cache['x'],x):return cache['value']
        q=p.copy();q[free]+=x.reshape(-1,3)*unit
        sample=(q[ids]*weights[:,:,None]).sum(1);distance,normal=query(sample)
        distance=np.asarray(distance,float);normal=np.asarray(normal,float)
        if (distance.shape!=(len(sample),) or normal.shape!=sample.shape or not np.isfinite(distance).all()
                or not np.isfinite(normal).all() or not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-3)):
            raise ValueError('Actual finite signed distances and unit solid normals required')
        jac=np.zeros((len(sample),size))
        for corner in range(3):
            index=mapping[ids[:,corner]];valid=np.flatnonzero(index>=0)
            for axis in range(3):jac[valid,3*index[valid]+axis]+=weights[valid,corner]*normal[valid,axis]
        delta=q[edges[:,1]]-q[edges[:,0]];current=np.linalg.norm(delta,axis=1)
        if current.min()<=1e-12:raise ValueError('Geometry proposal collapsed a material edge')
        edge_jac=np.zeros((len(edges),size));direction=delta/current[:,None]
        for side,sign in [(0,-1),(1,1)]:
            index=mapping[edges[:,side]];valid=np.flatnonzero(index>=0)
            for axis in range(3):edge_jac[valid,3*index[valid]+axis]+=sign*direction[valid,axis]
        xrows=x.reshape(-1,3);bound_jac=np.zeros((len(free),size))
        for axis in range(3):bound_jac[np.arange(len(free)),3*np.arange(len(free))+axis]=-2*xrows[:,axis]
        values=np.r_[(distance-clearance_m)/unit,
            (current-lengths*(1-maximum_edge_strain))/unit,
            (lengths*(1+maximum_edge_strain)-current)/unit,1-(xrows*xrows).sum(1)]
        gradient=np.r_[jac,edge_jac,-edge_jac,bound_jac]
        value=(q,values,gradient,distance,current);cache.update(x=x.copy(),value=value)
        return value
    result=minimize(lambda x:.5*float(x@x),np.zeros(size),jac=lambda x:x,method='SLSQP',
        bounds=[(-1.,1.)]*size,constraints=[{'type':'ineq','fun':lambda x:evaluate(x)[1],
                                         'jac':lambda x:evaluate(x)[2]}],
        options={'maxiter':maximum_iterations,'ftol':1e-10})
    q,values,_,distance,current=evaluate(result.x)
    maximum=float(np.linalg.norm(q-p,axis=1).max());strain=float(abs(current/lengths-1).max())
    passed=bool(values.min()>=-1e-5 and maximum<=maximum_displacement_m*(1+1e-6))
    return q,{'pass':passed,'task_approved':False,'native_states_changed':False,
        'root_positions_changed':False,'material_coordinates_changed':False,
        'solver_converged':bool(result.success),'solver_message':str(result.message),
        'iterations':int(result.nit),'minimum_sampled_clearance_m':float(distance.min()),
        'maximum_vertex_displacement_m':maximum,'maximum_absolute_edge_strain':strain,
        'budgets':{'clearance_m':clearance_m,'displacement_m':maximum_displacement_m,
                   'maximum_edge_strain':maximum_edge_strain},
        'scope':'Authored geometry proposal only. Full material, finite thickness, self-contact, source, native and task checks remain required.'}
