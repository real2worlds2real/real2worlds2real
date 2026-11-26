"""Source-constrained refinement of native seated shaft contact hypotheses."""
import itertools

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .contract import require
from .initial_state import project, segment_distance,aperture_clearance


def fit_network(segments, cameras, tips, directions, *, center, rim_height,
                fit_indices, validation_indices, contact_pairs=(), tilt_bound=.04,
                tip_limit=2.5, axis_limit=.095, penetration_limit=1e-5,support_radius=None,
                obstacle_segments=(),obstacle_margin=0.,assignments=None,
                rim_radius=None,floor_z=None,apertures=(),bottom_profile=None,diagnostics=None,
                line_observations=None,line_probe_y=None,line_limit=3.,container_refinement=None,
                settling_offsets=None):
    """Refine a small native contact network without pins or runtime pose targets.

    Preserve contact azimuth hypotheses, permit bounded tilt/length changes in
    modeling, and re-seat finite lower ends on their existing support plane.
    Native execution is required afterwards; these constraints alone do not
    prove frictional equilibrium. Identical source branches may exchange IDs.
    """
    n=len(segments);require(1<n<=4,'Plan larger contact networks in explicit groups')
    settling_offsets=np.zeros((n,2,3)) if settling_offsets is None else np.asarray(settling_offsets,float)
    require(settling_offsets.shape==(n,2,3) and np.isfinite(settling_offsets).all(),
            'Native preparation feedback must contain finite bottom/top endpoint offsets')
    center=np.asarray(center,float);tips=np.asarray(tips);directions=np.asarray(directions)
    free_contact=rim_radius is not None
    require(not free_contact or rim_radius>max(s[2] for s in segments) and floor_z is not None,
            'Free contact fitting requires reviewed circular rim and floor geometry')
    stride=4 if free_contact else 3
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    require(len(fit)>=3 and len(held)>=2 and not set(fit)&set(held),'Disjoint source views required')
    if line_observations is not None:
        line_observations=np.asarray(line_observations,float);line_probe_y=np.asarray(line_probe_y,float)
        require(line_observations.shape==(len(cameras),n,2) and line_probe_y.shape==(len(cameras),2)
                and line_limit>0,'Explicit source-visible line probes and uncertainty required')
    if container_refinement is not None:
        cp=np.asarray(container_refinement['points'],float);co=np.asarray(container_refinement['observations'],float)
        cs=np.asarray(container_refinement['sigma_px'],float);cb=np.asarray(container_refinement['bounds'],float)
        require(cp.ndim==2 and cp.shape[1]==3 and co.shape==(len(cameras),len(cp),2)
                and cs.shape==(len(cp),) and (cs>0).all() and cb.shape==(3,2)
                and (cb[:,0]<0).all() and (cb[:,1]>0).all(),
                'Bounded container translation needs independent source features and uncertainties')

    def camera_at(x,i):
        if container_refinement is None:return cameras[i]
        K,D,T=cameras[i];T=np.array(T,copy=True);T[:3,3]-=x[-3:]
        return K,D,T

    def container_errors(x,indices):
        return np.array([project(cp,camera_at(x,i))-co[i] for i in indices])
    seeds=[]
    for a,b,r in segments:
        a,b=np.asarray(a),np.asarray(b);d=b-a;L=np.linalg.norm(d);d/=L
        theta=np.arccos(d[2]);phi=np.arctan2(d[1],d[0]);entry=a+(b-a)*(rim_height-a[2])/(b[2]-a[2])
        seeds.append((a,b,r,L,theta,phi,entry,a[2]-r*np.sin(theta)))

    def geometry(x):
        axes=[]
        for i,(a,b,r,L,theta,phi,entry,floor) in enumerate(seeds):
            if free_contact:
                sx,sy,entry_phi,ratio=x[i*stride:i*stride+stride];direction=np.array([sx,sy,1.]);direction/=np.linalg.norm(direction)
                z=floor_z+r*np.linalg.norm(direction[:2])+1e-5
                pivot=center+np.r_[(rim_radius-r/direction[2]-2e-5)*np.array([np.cos(entry_phi),np.sin(entry_phi)]),rim_height-center[2]]
            else:
                spin,ratio,tilt=x[i*stride:i*stride+stride];angle=theta+tilt
                direction=np.array([np.sin(angle)*np.cos(phi+spin),np.sin(angle)*np.sin(phi+spin),np.cos(angle)])
                pivot=center+Rotation.from_euler('z',spin).apply(entry-center)
                z=floor+r*np.sin(angle)
            bottom=pivot-direction*(rim_height-z)/direction[2]
            axes.append((bottom,bottom+L*ratio*direction,r))
        return axes

    def solid_margins(x):
        axes=geometry(x);margins=[]
        for i,(a,b,r) in enumerate(axes):
            if support_radius is not None:margins.append(support_radius-r-np.linalg.norm(a[:2]-center[:2]))
            for opening in apertures:
                margins.append(aperture_clearance(a,b,r,center=opening['center'],height=opening['height'],opening_radius=opening['radius']))
            if bottom_profile is not None:
                axis=(b-a)/np.linalg.norm(b-a);u=np.cross(axis,np.eye(3)[np.argmin(abs(axis))]);u/=np.linalg.norm(u);v=np.cross(axis,u)
                angles=np.arange(32)*2*np.pi/32
                ring=a+r*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
                allowed=np.interp(ring[:,2],bottom_profile['heights'],bottom_profile['radii'])
                margins.extend(allowed-np.linalg.norm(ring[:,:2]-np.asarray(bottom_profile.get('center',center[:2])),axis=1))
            margins.extend(segment_distance(a,b,c,d)-r-s-obstacle_margin for c,d,s in obstacle_segments)
            for j in range(i+1,n):
                c,d,s=axes[j];gap=segment_distance(a,b,c,d)-r-s
                margins.append(gap+penetration_limit)
                if (i,j) in contact_pairs or (j,i) in contact_pairs:margins.append(5e-5-gap)
        return np.asarray(margins)

    bounds=[b for _ in range(n) for b in ([(-1.2,1.2),(-1.2,1.2),(-2*np.pi,2*np.pi),(.8,1.2)] if free_contact else
                                        [(-np.pi,np.pi),(.8,1.2),(-tilt_bound,tilt_bound)])]
    if container_refinement is not None:bounds.extend(cb.tolist())
    solutions=[]
    if diagnostics is not None:
        diagnostics.update({'accepted':False,'attempts':[]})
    identities=list(itertools.permutations(range(n))) if assignments is None else list(assignments)
    require(identities and all(len(row)==n and set(row)==set(range(n)) for row in identities),
            'Each permitted source identity assignment must be a complete permutation')
    for assignment in identities:
        def errors(x,indices):
            rows=[]
            for j,(a,b,r) in enumerate(geometry(x)):
                tip=[];axis=[];line=[];k=assignment[j]
                for i in indices:
                    uv=project(np.array([b+settling_offsets[j,1],a+settling_offsets[j,0]]),camera_at(x,i));delta=uv[1]-uv[0];delta/=np.linalg.norm(delta)
                    if np.isfinite(tips[i,k]).all():tip.extend(uv[0]-tips[i,k])
                    if np.isfinite(directions[i,k]).all():
                        target=directions[i,k];axis.append(np.arctan2(delta[0]*target[1]-delta[1]*target[0],delta@target))
                    if line_observations is not None and np.isfinite(line_observations[i,k]).all():
                        from .line_features import line_probes
                        line.extend(line_probes(uv,line_probe_y[i])-line_observations[i,k])
                rows.append((np.asarray(tip),np.asarray(axis),np.asarray(line)))
            return rows
        def residual(x):
            # Each feature channel has its own uncertainty. Dense tip tracks
            # must not overwhelm sparsely visible shaft fragments by count.
            rows=[v/(s*np.sqrt(max(1,len(v)))) for t,a,l in errors(x,fit)
                  for v,s in [(t,3.),(a,.1),(l,line_limit)]]
            if container_refinement is not None:
                e=container_errors(x,fit)
                rows += [e[:,j][np.isfinite(e[:,j]).all(1)].ravel()/cs[j]/np.sqrt(max(1,np.isfinite(e[:,j]).all(1).sum()*2)) for j in range(len(cp))]
            return np.concatenate(rows)
        def source_margins(x):
            values=[v for t,a,l in errors(x,fit) for v in
                [tip_limit-np.sqrt(np.mean(t*t)),axis_limit-np.sqrt(np.mean(a*a)),
                 *([line_limit-np.sqrt(np.mean(l*l))] if len(l) else [])]]
            if container_refinement is not None:values.extend(cs-np.sqrt(np.nanmean(container_errors(x,fit)**2,axis=(0,2))))
            return np.asarray(values)
        zero=(np.concatenate([np.r_[(b-a)[:2]/(b-a)[2],np.arctan2(entry[1]-center[1],entry[0]-center[0]),1.]
                              for a,b,r,L,theta,phi,entry,floor in seeds]) if free_contact else np.tile([0.,1.,0.],n))
        if container_refinement is not None:zero=np.r_[zero,np.zeros(3)]
        guess=least_squares(residual,zero,bounds=np.array(bounds).T,max_nfev=80).x
        if diagnostics is not None:
            diagnostics['attempts'].append({'assignment':list(assignment),'kind':'source_only',
                'fit_loss':float(np.mean(residual(guess)**2)),
                'source_margins':source_margins(guess).tolist(),
                'solid_margins_m':solid_margins(guess).tolist(),
                'segments':[{'bottom':a.tolist(),'top':b.tolist(),'radius':r} for a,b,r in geometry(guess)]})
        for start in [guess,zero]:
            solved=minimize(lambda x:float(np.mean(residual(x)**2)),start,method='SLSQP',bounds=bounds,
                constraints=[{'type':'ineq','fun':lambda x:solid_margins(x)/.001},
                             {'type':'ineq','fun':source_margins}],
                options={'maxiter':180,'ftol':1e-8})
            if diagnostics is not None:
                diagnostics['attempts'].append({'assignment':list(assignment),'kind':'constrained',
                    'optimizer_success':bool(solved.success),'optimizer_message':str(solved.message),
                    'fit_loss':float(np.mean(residual(solved.x)**2)),
                    'source_margins':source_margins(solved.x).tolist(),
                    'solid_margins_m':solid_margins(solved.x).tolist(),
                    'segments':[{'bottom':a.tolist(),'top':b.tolist(),'radius':r} for a,b,r in geometry(solved.x)]})
            if min(solid_margins(solved.x).min(),source_margins(solved.x).min())>=-1e-6:
                solutions.append((float(np.mean(residual(solved.x)**2)),assignment,solved.x))
    require(bool(solutions),'No source-constrained contact network; revise geometry/contact hypotheses')
    if diagnostics is not None:diagnostics['accepted']=True
    loss,assignment,x=min(solutions,key=lambda row:row[0]);rows=[]
    for j,(a,b,r) in enumerate(geometry(x)):
        k=assignment[j];metrics={}
        for label,ids in [('fit',fit),('validation',held)]:
            uv=np.array([project(np.array([b+settling_offsets[j,1],a+settling_offsets[j,0]]),camera_at(x,i)) for i in ids]);d=uv[:,1]-uv[:,0]
            e=uv[:,0]-tips[ids,k];valid=np.isfinite(e).all(1)
            metrics[label+'_tip_rmse_px']=float(np.sqrt(np.mean(e[valid]**2)))
            angles=np.arctan2(d[:,0],d[:,1])-np.arctan2(directions[ids,k,0],directions[ids,k,1]);angles=np.arctan2(np.sin(angles),np.cos(angles))
            metrics[label+'_axis_rmse_rad']=float(np.sqrt(np.nanmean(angles**2)))
            if line_observations is not None:
                from .line_features import line_probes
                predicted=np.array([line_probes(pair,line_probe_y[i]) for i,pair in zip(ids,uv)])
                metrics[label+'_visible_line_rmse_px']=float(np.sqrt(np.nanmean((predicted-line_observations[ids,k])**2)))
        axis=(b-a)/np.linalg.norm(b-a)
        rows.append({'bottom':a,'top':b,'radius_m':r,'source_index':k,'length_ratio':float(x[j*stride+(3 if free_contact else 1)]),
            'azimuth':float(np.arctan2(axis[1],axis[0])-seeds[j][5]) if free_contact else float(x[j*stride]),
            'tilt_delta':float(np.arccos(axis[2])-seeds[j][4]) if free_contact else float(x[j*stride+2]),
            'contact_model':'free slopes on reviewed circular rim' if free_contact else 'bounded native seated hypothesis',
            'metrics':metrics,'native_checked':False,
            'settling_endpoint_offsets_m':settling_offsets[j].tolist(),
            'prediction_scope':'Source-static proposal with native preparation feedback; fresh native execution is mandatory.'})
        if container_refinement is not None:
            rows[-1]['container_translation']=x[-3:].copy()
            rows[-1]['container_fit_rmse_px']=np.sqrt(np.nanmean(container_errors(x,fit)**2,axis=(0,2))).tolist()
            rows[-1]['container_validation_rmse_px']=np.sqrt(np.nanmean(container_errors(x,held)**2,axis=(0,2))).tolist()
    return rows
