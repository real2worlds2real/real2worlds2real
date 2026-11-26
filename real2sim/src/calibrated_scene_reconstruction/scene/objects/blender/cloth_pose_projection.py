"""Material-length and contact constraints for an offline cloth-pose proposal.

All geometry changes need fresh Blender, source and native/task validation.
This module does not install forces, pins, anchors or material changes.
"""
import numpy as np


def solid_geometry(mesh,parts):
    """Current solid boundaries for offline geometry construction only.

    Compliant contents obstruct the initial sheet just as rigid surfaces do.
    This observation does not freeze them or assign support forces in physics;
    their original volume reference and fresh native release remain required.
    """
    solids=[]
    for part in parts:
        if part['kind'] not in {'rigid','deformable','rope'}:continue
        a,n,b,m=[part[key] for key in ['vertex_start','vertex_count','face_start','triangle_count']]
        solids.append({'V':np.asarray(mesh['V'][a:a+n],float).copy(),
                       'F':np.asarray(mesh['F'][b:b+m],int).copy()-a})
    return solids


def material_budget(measurement,limits):
    """Bound offline contact repairs independently of final response waivers.

    Preserve an existing approximate seed, but never use that policy as a
    license to add large stretch while resolving a geometric collision.
    """
    fields=[
        ('maximum_absolute_edge_strain','max_edge_strain'),
        ('maximum_principal_material_strain','max_edge_strain'),
        ('relative_area_change','max_area_change')]
    if any(not np.isfinite([measurement[key],limits[limit]]).all() or limits[limit]<=0 for key,limit in fields):
        raise ValueError('Finite measured material and positive declared budgets required')
    return {key:max(abs(float(measurement[key])),float(limits[limit])) for key,limit in fields}


def within_material_budget(measurement,budget):
    return bool(measurement.get('geometry_valid') is True and all(
        np.isfinite(measurement[key]) and abs(measurement[key])<=limit+1e-6
        for key,limit in budget.items()))


def combined_constraints(*terms):
    """Combine offline inequality proposals, retaining all independent rows."""
    def evaluate(points):
        rows=[row for term in terms if (row:=term(points)) is not None]
        return tuple(np.concatenate([row[column] for row in rows]) for column in range(3)) if rows else None
    return evaluate


def coupled_clearance_step(rows,shape,scale):
    """Minimum-motion proposal for coupled linearized contact inequalities.

    A solid correction on one ply can require its neighbours to move too.
    Solve the nonnegative sparse dual instead of averaging independent pushes.
    Nonlinear thickness/material/source guards remain the caller's authority.
    """
    from scipy import sparse
    from scipy.optimize import minimize
    ids,gradient,residual=rows
    norm=np.sqrt((gradient**2).sum((1,2)))
    if (len(shape)!=2 or shape[1]!=3 or not np.isfinite(scale)or scale<=0
            or not np.isfinite(norm).all()or(norm<=1e-10).any()):
        raise ValueError('Resolved contact rows and finite positive proposal scale required')
    unit=gradient/norm[:,None,None];rhs=residual/norm/scale
    columns=(ids[:,:,None]*3+np.arange(3)).reshape(-1)
    matrix=sparse.coo_matrix((unit.reshape(-1),(np.repeat(np.arange(len(ids)),unit.shape[1]*3),columns)),
                            shape=(len(ids),shape[0]*3)).tocsr();matrix.eliminate_zeros()
    def dual(load):
        move=matrix.T@load
        return .5*float(move@move)+float(rhs@load),matrix@move+rhs
    solved=minimize(dual,np.zeros(len(ids)),jac=True,bounds=[(0.,None)]*len(ids),method='L-BFGS-B',
                    options={'maxiter':500,'ftol':1e-14,'gtol':1e-9,'maxls':40})
    move=(matrix.T@solved.x).reshape(shape)*scale
    if not np.isfinite(move).all():raise ValueError('Nonfinite coupled clearance proposal')
    remaining=np.maximum(0.,-(matrix@move.ravel()+residual/norm))
    return move,{'solver':'coupled_sparse_dual','linear_solver_converged':bool(solved.success),
                 'iterations':int(solved.nit),'maximum_linearized_remaining_deficit_m':float(remaining.max(initial=0.)),
                 'task_approved':False}


def repair_clearances(points,contacts,*,admissible,maximum_displacement,
                      maximum_step=2e-5,iterations=128,on_iteration=None,satisfied=None,solver='jacobi'):
    """Satisfy offline contact inequalities without pulling material to zero strain.

    The existing material validity remains a hard constraint supplied by the
    caller. This is a bounded geometry proposal, never a native state update.
    """
    q=np.array(points,float,copy=True);original=q.copy();history=[]
    if (not np.isfinite(q).all() or not np.isfinite([maximum_displacement,maximum_step]).all()
            or min(maximum_displacement,maximum_step)<=0 or type(iterations) is not int or not 1<=iterations<=512):
        raise ValueError('Finite bounded offline clearance repair required')
    if not admissible(q):raise ValueError('Material-valid nonintersecting seed required')
    if satisfied is not None and not callable(satisfied):raise ValueError('Independent final-geometry predicate required')
    if solver not in {'jacobi','coupled'}:raise ValueError('Explicit supported clearance proposal solver required')
    converged=False;reason=None
    for iteration in range(iterations):
        if satisfied is not None and satisfied(q):
            converged=True;reason='independent_final_geometry';break
        rows=contacts(q)
        if rows is None:converged=True;break
        ids,gradient,residual=rows
        norm2=(gradient**2).sum((1,2))
        if (norm2<=1e-20).any():raise ValueError('Unresolved clearance normal')
        deficit=np.maximum(0.,-residual)/np.sqrt(norm2)
        if deficit.max(initial=0.)<=5e-7:converged=True;break
        solve_report=None
        if solver=='coupled':move,solve_report=coupled_clearance_step(rows,q.shape,maximum_step)
        else:
            active=residual<0;ids=ids[active];gradient=gradient[active]
            correction=-residual[active,None,None]*gradient/norm2[active,None,None]
            move=np.zeros_like(q);count=np.zeros(len(q))
            np.add.at(move,ids.ravel(),correction.reshape(-1,3))
            np.add.at(count,ids.ravel(),(np.linalg.norm(gradient,axis=2)>1e-12).ravel())
            move*=.8/np.maximum(count[:,None],1.)
        magnitude=np.linalg.norm(move,axis=1).max()
        if magnitude>maximum_step:move*=maximum_step/magnitude
        fraction=1.;accepted=False
        for _ in range(12):
            trial=q+fraction*move
            if np.linalg.norm(trial-original,axis=1).max()<=maximum_displacement and admissible(trial):
                q=trial;accepted=True;break
            fraction*=.5
        row={'iteration':iteration,'accepted':accepted,'maximum_linearized_deficit_m':float(deficit.max()),
             'maximum_displacement_m':float(np.linalg.norm(q-original,axis=1).max())}
        if solve_report is not None:row['coupled_proposal']=solve_report
        history.append(row)
        if on_iteration:on_iteration(q.copy(),row)
        if not accepted:break
    return q,{'converged':converged,'convergence_reason':reason,'iterations':history,'native_state_written':False,
              'material_coordinates_changed':False,'task_approved':False,
              'scope':'Offline contact proposal; full independent geometry, source and native checks remain required.'}


class MaterialBounds:
    """Linearized principal-stretch inequalities for offline proposals only."""
    def __init__(self,faces,rest,limit,*,weight=40.):
        from calibrated_scene_reconstruction.assets.cloth_rest import material_inverse
        if not np.isfinite([limit,weight]).all() or not 0<limit<1 or weight<=0:
            raise ValueError('Finite principal strain limit in0..1 and positive proposal weight required')
        self.faces=np.asarray(faces,int);self.rest=np.asarray(rest,float)
        self.inverse=material_inverse(self.rest,self.faces);self.limit=limit;self.weight=weight
        tri=self.rest[self.faces]
        self.scale=np.sqrt(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2)
        self.latest={}

    def __call__(self,points):
        tri=np.asarray(points,float)[self.faces]
        deformation=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)@self.inverse
        left,stretch,right=np.linalg.svd(deformation,full_matrices=False)
        if not np.isfinite(stretch).all() or (stretch<=1e-12).any():raise ValueError('Collapsed material triangle needs a new pose seed')
        self.latest={'principal_bound_violations':int((abs(stretch-1)>self.limit).sum()),
                     'maximum_principal_strain':float(abs(stretch-1).max()),'limit':self.limit,
                     'native_forces_installed':False,'task_approved':False}
        near=abs(stretch-1)>self.limit*.9;triangle,axis=np.nonzero(near)
        if not len(triangle):return None
        # d singular value / d F = u v^T. Material chart coordinates remain
        # immutable; derivatives map through the two actual rest edge vectors.
        u=left[triangle,:,axis];v=right[triangle,axis,:]
        chart=np.einsum('nij,nj->ni',self.inverse[triangle],v)
        derivative=np.stack([-u*chart.sum(1)[:,None],u*chart[:,0,None],u*chart[:,1,None]],axis=1)
        sign=np.where(stretch[triangle,axis]>1,-1.,1.)
        residual=self.limit-abs(stretch[triangle,axis]-1)
        scale=self.scale[triangle]*self.weight
        indices=np.zeros((len(triangle),6),int);indices[:,:3]=self.faces[triangle]
        gradient=np.zeros((len(triangle),6,3));gradient[:,:3]=derivative*(sign*scale)[:,None,None]
        return indices,gradient,residual*scale


class ParticleEnvelopes:
    """Actual particle-centre envelope constraints, separate from sheet thickness."""
    def __init__(self,rest,radius,*,weight=20.):
        if not np.isfinite([radius,weight]).all() or min(radius,weight)<=0:
            raise ValueError('Explicit positive particle radius and proposal weight required')
        self.rest=np.asarray(rest,float);self.radius=radius;self.weight=weight;self.latest={}

    def __call__(self,points):
        from calibrated_scene_reconstruction.assets.cloth_sampling import initial_overlaps
        report=initial_overlaps(points,self.rest,self.radius,proposal_candidates=True)
        pairs=report.pop('nearby_nonlocal_pairs',None);self.latest=report
        if pairs is None:raise ValueError('Particle envelope proposal exceeds its query budget')
        if not len(pairs):return None
        delta=np.asarray(points)[pairs[:,0]]-np.asarray(points)[pairs[:,1]];distance=np.linalg.norm(delta,axis=1)
        if (distance<=1e-10).any():raise ValueError('Coincident nonlocal particles need a new pose seed')
        ids=np.zeros((len(pairs),6),int);ids[:,:2]=pairs
        gradient=np.zeros((len(pairs),6,3));gradient[:,0]=delta/distance[:,None];gradient[:,1]=-gradient[:,0]
        return ids,self.weight*gradient,self.weight*(distance-2*self.radius-2e-6)


class Contacts:
    def __init__(self,faces,rest,rigids,clearance,solid_clearance,*,device='cpu',weight=20.,maximum_step=0.,pair_selection='all',candidate_capacity=1000000):
        from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
        if not np.isfinite([clearance,solid_clearance,weight]).all() or min(clearance,solid_clearance,weight)<=0:
            raise ValueError('Positive explicit geometry clearances and proposal penalty weight required')
        self.faces=np.asarray(faces,int);self.rest=np.asarray(rest,float)
        self.surfaces=[SignedSurface(p['V'],p['F'],device=device) for p in rigids]
        self.clearance=clearance;self.solid_clearance=solid_clearance;self.device=device;self.weight=weight
        if not np.isfinite(maximum_step) or maximum_step<0:raise ValueError('Nonnegative proposal step bound required')
        self.maximum_step=maximum_step
        if pair_selection not in {'all','closest'}:raise ValueError('Choose all or closest-per-face proposal contacts')
        self.pair_selection=pair_selection
        if type(candidate_capacity) is not int or not 1<=candidate_capacity<=2000000:
            raise ValueError('Bounded contact proposal capacity in1..2000000 required')
        self.candidate_capacity=candidate_capacity
        self.latest={}

    def __call__(self,points):
        from calibrated_scene_reconstruction.assets.cloth_clearance import measure,contact_witnesses
        # Anticipate closing local folds before their normals become opposed.
        # Otherwise a tiny step across90degrees introduces a previously absent
        # finite-thickness constraint and can stall the independent line search.
        # This conservative proposal guard changes neither physical thickness
        # nor the independent/default clearance verdict.
        report=measure(points,self.faces,self.rest,self.clearance,device=self.device,proposal_candidates=True,
                       search_padding=2*self.maximum_step,collect_all_pairs=self.pair_selection=='all',opposition_cosine=.5,
                       candidate_capacity=self.candidate_capacity)
        key='all_nearby_pairs' if self.pair_selection=='all' else 'closest_nearby_pairs'
        pairs=np.asarray(report[key],int).reshape(-1,2)
        indices=[];gradients=[];residuals=[]
        angular_count=0
        if len(pairs):
            witness=contact_witnesses(points,self.faces,pairs)
            tri=np.asarray(points)[self.faces];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals=cross/np.linalg.norm(cross,axis=1)[:,None]
            cosine=(normals[pairs[:,0]]*normals[pairs[:,1]]).sum(1)
            centres=self.rest[self.faces].mean(1)
            local=np.linalg.norm(centres[pairs[:,0]]-centres[pairs[:,1]],axis=1)<=report['material_neighbour_exclusion_m']
            ordinary=(~local)|(cosine<0.)
            # Local, non-opposed neighbours are one continuous sheet. Keep
            # their normals from crossing the opposition boundary instead of
            # spuriously forcing all such neighbouring triangles apart.
            angular=(~ordinary)&(witness['gap_m']<self.clearance+2e-6)
            angular_count=int(angular.sum())
            if angular_count:
                from calibrated_scene_reconstruction.assets.cloth_contact_rows import normal_cosine_rows
                ii,gg,rr=normal_cosine_rows(points,self.faces,pairs[angular],
                                           minimum_cosine=np.cos(np.radians(85.)),scale=self.clearance)
                indices.append(ii);gradients.append(gg);residuals.append(rr)
            if ordinary.any():
                indices.append(np.c_[witness['faces_a'],witness['faces_b']][ordinary])
                gradients.append(np.concatenate([witness['weights_a'][:,:,None]*witness['normal'][:,None],
                    -witness['weights_b'][:,:,None]*witness['normal'][:,None]],axis=1)[ordinary])
                residuals.append((witness['gap_m']-self.clearance-2e-6)[ordinary])
        count=0
        # Match the independent rigid-clearance screen: vertices alone miss
        # a small solid beneath a triangle interior or an edge midpoint.
        from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
        samples=triangle_samples(points,self.faces)
        sample_ids=np.concatenate([np.c_[np.arange(len(points)),np.zeros((len(points),2),int)],
                                   np.tile(self.faces,(4,1))])
        weights=np.concatenate([np.tile([1.,0.,0.],(len(points),1)),
            *[np.tile(w,(len(self.faces),1)) for w in [[1/3,1/3,1/3],[.5,.5,0.],[0.,.5,.5],[.5,0.,.5]]]])
        for surface in self.surfaces:
            signed=surface(samples);active=np.flatnonzero(signed<1.2*self.solid_clearance+self.maximum_step)
            count+=int((signed<self.solid_clearance).sum())
            if len(active):
                delta=surface.project_out(samples[active],1.3*self.solid_clearance+self.maximum_step)-samples[active]
                distance=np.linalg.norm(delta,axis=1)
                if (distance<=1e-8).any():raise ValueError('Unresolved rigid contact normal for geometry proposal')
                # Pad with zero-weight entries to the triangle-pair convention.
                ids=np.zeros((len(active),6),int);ids[:,:3]=sample_ids[active]
                gradient=np.zeros((len(active),6,3))
                gradient[:,:3]=weights[active,:,None]*(delta/distance[:,None])[:,None,:]
                indices.append(ids);gradients.append(gradient);residuals.append(signed[active]-self.solid_clearance-2e-6)
        self.latest={'self_clearance_pass':report['pass'],'violating_pairs':report['violating_pairs'],
                     'conservative_proposal_normal_cosine':.5,
                     'local_angular_guard_rows':angular_count,
                     'proposal_pair_selection':self.pair_selection,'full_clearance_scan':True,
                     'minimum_gap_m':report['minimum_gap_within_search_m'],'nearby_proposal_pairs':len(pairs),
                     'rigid_sample_clearance_deficits':count,'task_approved':False}
        if not indices:return None
        return np.concatenate(indices),self.weight*np.concatenate(gradients),self.weight*np.concatenate(residuals)
