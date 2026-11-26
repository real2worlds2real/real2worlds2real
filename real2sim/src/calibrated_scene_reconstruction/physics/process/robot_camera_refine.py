"""Refine independent camera sequences with static tracks and measured robot CAD.

No response-object positions, forces or measured object sizes enter this fit.
The supplied clocks stay fixed in this step. Held-out robot landmarks expose
camera/trajectory errors that a static bundle alone cannot detect.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


def solve(cameras, groups, static, robot, *, static_sigma_px=2., robot_sigma_px=4., budget=400):
    names=list(map(str,cameras['names']));n=len(names)
    R=np.asarray(cameras['R_world_cam'],float);C=np.asarray(cameras['centre'],float);K=np.asarray(cameras['K'],float)
    if (len(set(names))!=n or set(groups)!=set(names) or R.shape!=(n,3,3) or C.shape!=(n,3)
            or K.shape!=(n,3,3) or not all(np.isfinite(x).all()for x in [R,C,K])
            or min(static_sigma_px,robot_sigma_px)<=0 or type(budget)is not int or budget<10):
        raise ValueError('Independent calibrated cameras, groups and bounded uncertainty required')
    labels=list(dict.fromkeys(groups[name]for name in names));group=np.array([labels.index(groups[name])for name in names]);ng=len(labels)
    points=np.asarray(static['world'],float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<6 or not np.isfinite(points).all():
        raise ValueError('Finite static world tracks required')
    def observation(value,world=None):
        ci=np.asarray(value['observation_camera']);pixel=np.asarray(value['pixels'],float);training=np.asarray(value['training'])
        if (ci.ndim!=1 or ci.dtype.kind not in 'iu' or np.any(ci<0)or np.any(ci>=n)
                or pixel.shape!=(len(ci),2)or not np.isfinite(pixel).all()
                or training.shape!=(len(ci),)or training.dtype.kind!='b' or training.sum()<6 or (~training).sum()<3):
            raise ValueError('Explicit finite training and held-out observations required')
        if world is not None and (world.shape!=(len(ci),3) or not np.isfinite(world).all()):
            raise ValueError('Finite measured robot CAD points required')
        return ci,pixel,training
    rc,ru,rt=observation(robot,np.asarray(robot['world'],float));rw=np.asarray(robot['world'],float)
    sc,su,st=observation(static);sp=np.asarray(static['observation_point'])
    if sp.shape!=sc.shape or sp.dtype.kind not in 'iu' or np.any(sp<0)or np.any(sp>=len(points)):
        raise ValueError('Static observations must retain distinct point identities')
    for i in range(n):
        if np.sum((sc==i)&st)<6:raise ValueError('Every camera needs at least six static training tracks')
    for g in range(ng):
        if np.sum((group[rc]==g)&rt)<6 or np.sum((group[rc]==g)&~rt)<3:
            raise ValueError('Every independent camera needs training and held-out robot evidence')
    # Refine one rigid correction per physical camera. Preserve independently
    # recovered small camera motion: planar static tracks alone cannot safely
    # re-estimate every frame's metric depth without overfitting robot views.
    focal=np.array([K[np.where(group==g)[0][0],[0,1],[0,1]]for g in range(ng)])
    for g in range(ng):
        if not np.allclose(K[group==g],K[np.where(group==g)[0][0]]):
            raise ValueError('This refinement requires constant intrinsics within each camera')
    nr=6*ng;nf=2*ng
    anchors=np.array([C[np.where(group==g)[0][0]]for g in range(ng)])
    x0=np.r_[np.zeros(nr),np.log(focal).ravel(),points.ravel()]
    lower=x0.copy();upper=x0.copy()
    delta=np.tile([.15,.15,.15,.15,.15,.15],ng)
    lower[:nr]-=delta;upper[:nr]+=delta
    lower[nr:nr+nf]+=np.log(.8);upper[nr:nr+nf]+=np.log(1.2)
    lower[nr+nf:]-=.5;upper[nr+nf:]+=.5
    def unpack(x):
        q=x[:nr].reshape(ng,6);correction=Rotation.from_rotvec(q[:,:3]).as_matrix()
        rr=correction[group]@R
        cc=np.einsum('nij,nj->ni',correction[group],C-anchors[group])+anchors[group]+q[group,3:]
        kk=K.copy();ff=np.exp(x[nr:nr+nf].reshape(ng,2));kk[:,0,0]=ff[group,0];kk[:,1,1]=ff[group,1]
        return rr,cc,kk,x[nr+nf:].reshape(-1,3)
    def projection(world,ci,rr,cc,kk):
        pc=np.einsum('ni,nij->nj',world-cc[ci],rr[ci]);uv=np.einsum('nij,nj->ni',kk[ci],pc)
        return uv[:,:2]/np.maximum(uv[:,2,None],1e-6),pc[:,2]
    def predictions(x):
        rr,cc,kk,pp=unpack(x)
        a,za=projection(pp[sp],sc,rr,cc,kk);b,zb=projection(rw,rc,rr,cc,kk)
        return a,b,za,zb
    def residual(x):
        a,b,za,zb=predictions(x)
        return np.r_[((a[st]-su[st])/static_sigma_px).ravel(),((b[rt]-ru[rt])/robot_sigma_px).ravel()]
    sparsity=lil_matrix((2*(st.sum()+rt.sum()),len(x0)),dtype=int)
    index=0
    for ci,pi in zip(sc[st],sp[st]):
        sparsity[index:index+2,6*group[ci]:6*group[ci]+6]=1;sparsity[index:index+2,nr+2*group[ci]:nr+2*group[ci]+2]=1
        sparsity[index:index+2,nr+nf+3*pi:nr+nf+3*pi+3]=1;index+=2
    for ci in rc[rt]:
        sparsity[index:index+2,6*group[ci]:6*group[ci]+6]=1;sparsity[index:index+2,nr+2*group[ci]:nr+2*group[ci]+2]=1;index+=2
    fit=least_squares(residual,x0,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),
        x_scale='jac',loss='soft_l1',f_scale=1.,max_nfev=budget,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    rr,cc,kk,pp=unpack(fit.x);a,b,za,zb=predictions(fit.x)
    def stats(error):return np.percentile(error,[50,90,95,100]).tolist()
    se=np.linalg.norm(a-su,axis=1);re=np.linalg.norm(b-ru,axis=1)
    report=dict(accepted=False,task_approved=False,optimizer_converged=bool(fit.success),evaluations=fit.nfev,
                bound_hit=bool(np.any(fit.active_mask)),behind_camera=int((za<=0).sum()+(zb<=0).sum()),
                static_training_error_px=stats(se[st]),static_held_out_error_px=stats(se[~st]),
                robot_training_error_px=stats(re[rt]),robot_held_out_error_px=stats(re[~rt]),
                camera_translation_change_m=np.linalg.norm(cc-C,axis=1).tolist(),
                scope=__doc__.strip(),groups={name:dict(focal=kk[np.where(group==g)[0][0],[0,1],[0,1]].tolist(),
                    robot_training_error_px=stats(re[rt&(group[rc]==g)]),robot_held_out_error_px=stats(re[~rt&(group[rc]==g)]))for g,name in enumerate(labels)})
    return dict(names=np.array(names),K=kk,R_world_cam=rr,centre=cc),dict(world=pp,robot_predicted=b,static_predicted=a),report
