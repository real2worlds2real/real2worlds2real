"""Guard model callbacks without copying every soft vertex each physics step."""
import numpy as np
from .contract import require


class DriverGuard:
    """Observe USD response edits and immediate native rigid pose changes.

    Physics-step USD readback happens outside this guard. Prescribed drivers
    remain free to move. Force application is allowed; response pose/geometry
    or velocity authoring by a callback is not a simulated response.
    """
    def __init__(self,stage,response_paths,rigid_views):
        from pxr import Tf,Usd
        self.paths=tuple(response_paths);self.rigid=dict(rigid_views)
        self.active=False;self.changed=[]
        self.listener=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._notice,stage)

    def _notice(self,notice,sender):
        if not self.active:return
        for path in notice.GetResyncedPaths():
            p=str(path.GetPrimPath())
            if any(p==r or p.startswith(r+'/') or r.startswith(p+'/') for r in self.paths):self.changed.append(str(path))
        geometry={'points','positions','orientations','velocities','physics:velocity','physics:angularVelocity',
                  'physics:kinematicEnabled','physics:rigidBodyEnabled','physxParticle:particleEnabled',
                  'physxDeformable:deformableEnabled','xformOpOrder','faceVertexCounts','faceVertexIndices',
                  'size','radius','height','width','length','axis'}
        for path in notice.GetChangedInfoOnlyPaths():
            p=str(path.GetPrimPath());name=path.name;transform=name=='xformOpOrder' or name.startswith('xformOp:')
            inside=any(p==r or p.startswith(r+'/') for r in self.paths)
            ancestor=any(r.startswith(p+'/') for r in self.paths)
            if (inside and (name in geometry or transform)) or (ancestor and transform):self.changed.append(str(path))

    @staticmethod
    def _pose(view):
        p,q=view.get_world_pose()
        def array(v):return np.asarray(v.detach().cpu().numpy() if hasattr(v,'detach') else v,float).copy()
        return array(p),array(q)

    def run(self,callback,*args):
        before={k:self._pose(v) for k,v in self.rigid.items()};self.changed=[];self.active=True
        try:result=callback(*args)
        finally:self.active=False
        require(not self.changed,'Driver teleported or authored a dynamic response: '+', '.join(self.changed[:3]))
        for k,view in self.rigid.items():
            p,q=self._pose(view);a,b=before[k]
            if np.dot(q,b)<0:q=-q
            require(np.allclose(a,p,rtol=0,atol=1e-7) and np.allclose(b,q,rtol=0,atol=1e-7),
                    'Driver teleported a dynamic response body: '+k)
        return result

    def close(self):
        if self.listener is not None:self.listener.Revoke();self.listener=None
