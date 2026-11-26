"""Bounded common placement from native initial states and source observations."""
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation

from .contract import require
from .initial_state import project
from .line_features import line_probes


def refine_translation(features,cameras,*,fit_indices,validation_indices,bounds_m,strategy='minimax',rotation=None,endpoints=None):
    """Translate every actor and its support together; keep relative contacts.

    Material points, axes and line probes use exact projection. Region terms
    use a local perspective proxy for proposal generation only; the next free
    native run must recompute the complete original mask operators and gates.
    """
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int);bounds=np.asarray(bounds_m,float)
    require(features and len(fit)>=3 and len(held)>=2 and not set(fit)&set(held)
            and bounds.shape==(3,2) and (bounds[:,0]<0).all() and (bounds[:,1]>0).all(),
            'Independent static source views and explicit placement uncertainty required')
    require(strategy in {'balanced','minimax'},'Explicit supported placement objective required')
    moving=set()
    if rotation is not None:
        axis=np.asarray(rotation['axis_world'],float);pivot=np.asarray(rotation['pivot_world_m'],float)
        rb=np.asarray(rotation['bounds_rad'],float);moving=set(rotation['feature_ids'])
        require(axis.shape==pivot.shape==(3,) and np.isfinite(axis).all() and np.isfinite(pivot).all()
                and abs(np.linalg.norm(axis)-1)<1e-6 and rb.shape==(2,) and rb[0]<0<rb[1]
                and moving and moving<=set(f['id'] for f in features) and rotation.get('reason'),
                'Bounded initial group rotation needs explicit source identities, pivot, axis and contact hypothesis')
        bounds=np.vstack([bounds,rb])
    scales=np.r_[np.full(3,.001),[.01] if rotation is not None else []];endpoint_indices={}
    for key,item in (endpoints or {}).items():
        feature=next((f for f in features if f['id']==key),None)
        directions=np.asarray(item['directions_world'],float);eb=np.asarray(item['bounds_m'],float)
        require(feature is not None and feature['kind']=='point' and np.asarray(feature['xyz']).shape[1:]==(1,3)
                and directions.shape==(len(cameras),3) and np.isfinite(directions).all()
                and np.allclose(np.linalg.norm(directions,axis=1),1) and eb.shape==(2,) and eb[0]<0<eb[1]
                and item.get('reason'),'Endpoint dimension refinement needs an explicit bounded geometric hypothesis and unit directions')
        endpoint_indices[key]=len(scales);bounds=np.vstack([bounds,eb]);scales=np.r_[scales,.001]
    dim=len(scales)
    def predict(feature,translation,indices):
        rows=[];xyz=np.asarray(feature['xyz'],float);kind=feature['kind']
        for i in indices:
            moved=xyz[i]
            if feature['id'] in endpoint_indices:
                moved=moved+translation[endpoint_indices[feature['id']]]*np.asarray(endpoints[feature['id']]['directions_world'])[i]
            if feature['id'] in moving:
                R=Rotation.from_rotvec(axis*translation[3]).as_matrix();moved=(moved-pivot)@R.T+pivot
            moved=moved+translation[:3];uv=project(moved,cameras[i])
            if kind=='point':value=uv[0]
            elif kind=='axis':
                d=uv[1]-uv[0];value=np.array([np.arctan2(d[0],d[1])])
            elif kind=='line_probes':value=line_probes(uv,feature['probe_y'][i])
            elif kind=='region_center':value=np.asarray(feature['baseline'])[i]+uv[0]-project(xyz[i],cameras[i])[0]
            elif kind=='region_extent':
                T=cameras[i][2];z0=((xyz[i,0]-T[:3,3])@T[:3,:3])[2]
                z1=((moved[0]-T[:3,3])@T[:3,:3])[2]
                require(z0>0 and z1>0,'Placement left the visible volume');value=np.asarray(feature['baseline'])[i]*z0/z1
            else:raise ValueError('Unsupported initial placement feature: '+kind)
            rows.append(value)
        return np.asarray(rows)
    def error(feature,x,indices):
        e=predict(feature,x,indices)-np.asarray(feature['observed'])[indices]
        if feature['kind']=='axis':e=np.arctan2(np.sin(e),np.cos(e))
        return e[np.isfinite(e).all(1)].ravel()
    def residual(x):
        rows=[]
        for feature in features:
            e=error(feature,x,fit);require(len(e)>=2 and feature['sigma']>0,'Missing static source feature support')
            rows.append(e/feature['sigma']/np.sqrt(len(e)))
        return np.concatenate(rows)
    def margins(x):
        return np.array([1-np.sqrt(np.mean(error(f,x,fit)**2))/f['sigma'] for f in features])
    # Millimetre variables give the optimizer a useful common geometric scale.
    lower,upper=bounds[:,0]/scales,bounds[:,1]/scales
    seed=least_squares(lambda u:residual(u*scales),np.zeros(dim),bounds=(lower,upper),max_nfev=100).x
    if strategy=='balanced':
        result=minimize(lambda u:float(np.sum(residual(u*scales)**2)),seed,method='SLSQP',
            bounds=list(zip(lower,upper)),constraints=[{'type':'ineq','fun':lambda u:margins(u*scales)}],
            options={'maxiter':100,'ftol':1e-10})
    else:
        # Seek interior margin across channels, rather than improving already
        # precise landmarks while leaving a sparse visible part at its limit.
        initial=np.r_[seed,max(0.,1-margins(seed*scales).min())]
        result=minimize(lambda u:float(u[dim]+.01*np.sum(residual(u[:dim]*scales)**2)),initial,method='SLSQP',
            bounds=[*zip(lower,upper),(0.,10.)],
            constraints=[{'type':'ineq','fun':lambda u:u[dim]-1+margins(u[:dim]*scales)}],
            options={'maxiter':150,'ftol':1e-10})
    delta=result.x[:dim]*scales
    metrics={}
    for feature in features:
        metrics[feature['id']]={split+'_rmse':float(np.sqrt(np.mean(error(feature,delta,ids)**2)))
                                 for split,ids in [('fit',fit),('validation',held)]}
    return {'translation_world_m':delta[:3],**({'rotation':{**rotation,'angle_rad':float(delta[3])}} if rotation is not None else {}),
            **({'endpoint_deltas_m':{key:float(delta[index]) for key,index in endpoint_indices.items()}} if endpoints else {}),
            'metrics':metrics,'fit_feasible':bool(margins(delta).min()>=-1e-6),
            'objective':strategy,'largest_normalized_fit_error':float(1-margins(delta).min()),
            'native_checked':False,'optimizer_success':bool(result.success),
            'scope':'Common source-static placement proposal within registration uncertainty. '+
                ('One declared rigid group may rotate; internal relations are preserved, external contact compatibility needs native verification. ' if rotation is not None else
                 'Body frames retain their relative poses; dimension changes require contact verification. ' if endpoints else 'Relative geometry and contacts unchanged. ')+
                ('Declared endpoint dimensions may change; rendering, collision geometry and density-based mass/inertia must be updated together in a derived hypothesis. ' if endpoints else '')+
                'Region proxies are not final evidence. No response poses or observations changed.'}
