"""Temporary finite native joint drive with explicit, verified release.

Targets are solver drives, never joint positions or body pose writes. The drive
is a construction fixture and must not be exported as an object's mechanics.
"""
import numpy as np
from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy


class JointDriveFixture:
    def __init__(self,ctx,assembly,parent,child,*,stiffness,damping,effort_limit):
        if not np.isfinite([stiffness,damping,effort_limit]).all() or min(stiffness,damping,effort_limit)<=0:
            raise ValueError('Finite positive temporary drive gains and effort bound required')
        self.ctx,self.names,self.limit=ctx,(assembly,parent,child),float(effort_limit)
        ctx.apply_joint_effort(*self.names,0.,effort_limit=self.limit)
        path,self.index,kind=ctx._joint_handles[self.names]
        if kind!='revolute':raise ValueError('This preparation drive requires a revolute joint')
        self.view=ctx._joint_views[path][0];self.controller=self.view.get_articulation_controller()
        self.original_gains=[as_numpy(v).copy() for v in self.controller.get_gains()]
        self.original_limits=as_numpy(self.controller.get_max_efforts()).copy()
        if any(abs(g[self.index])>1e-12 for g in self.original_gains):
            raise ValueError('Preparation may not replace an existing physical joint spring or damper')
        self.released=False
        self.target(float(as_numpy(self.view.get_joint_positions())[self.index]))
        limits=self.original_limits.copy();limits[self.index]=self.limit
        self.controller.set_max_efforts(limits.tolist())
        kp,kd=[v.copy() for v in self.original_gains];kp[self.index]=stiffness;kd[self.index]=damping
        self.set_gains(kp,kd)
        actual=[as_numpy(v)[self.index] for v in self.controller.get_gains()]
        if not np.allclose(actual,[stiffness,damping],rtol=1e-5,atol=1e-10) or not np.isclose(
                as_numpy(self.controller.get_max_efforts())[self.index],self.limit,rtol=1e-5):
            raise ValueError('Native preparation drive did not retain its finite gains and effort cap')
        self.initial_readback={'stiffness_Nm_rad':float(actual[0]),'damping_Nms_rad':float(actual[1]),
            'effort_limit_Nm':self.limit,'positions_prescribed':False,'native_drive':True}

    def set_gains(self,kp,kd):
        if self.ctx.world.backend=='torch':
            import torch
            kp=torch.as_tensor(kp,dtype=torch.float32,device=self.ctx.world.device)
            kd=torch.as_tensor(kd,dtype=torch.float32,device=self.ctx.world.device)
        self.controller.set_gains(kps=kp,kds=kd,save_to_usd=False)

    def target(self,angle):
        if self.released or not np.isfinite(angle):raise ValueError('Active drive and finite angle required')
        from isaacsim.core.utils.types import ArticulationAction
        self.view.apply_action(ArticulationAction(joint_positions=np.array([angle],np.float32),
                                                 joint_indices=[self.index]))
        self.last_target_rad=float(angle)

    def release(self):
        # Restore every untouched DOF and zero this temporary drive before
        # clearing the explicit effort channel. Stopping target writes is not
        # release: position drives persist in the native solver.
        self.set_gains(*self.original_gains)
        self.controller.set_max_efforts(self.original_limits.tolist())
        receipt=self.ctx.apply_joint_effort(*self.names,0.,effort_limit=self.limit)
        gains=[float(as_numpy(v)[self.index]) for v in self.controller.get_gains()]
        if any(abs(v)>1e-12 for v in gains):raise ValueError('Temporary native drive remains active after release')
        self.released=True
        return {**receipt,'native_drive_gains_zero':True,'native_drive_gains_readback':gains,
                'temporary_drive_removed':True,'positions_prescribed':False}

    def verify_released(self):
        gains=[float(as_numpy(v)[self.index]) for v in self.controller.get_gains()]
        effort=float(as_numpy(self.view.get_applied_joint_efforts())[self.index])
        if not self.released or any(abs(v)>1e-12 for v in [*gains,effort]):
            raise ValueError('Released preparation drive or explicit effort became active')
        return {'native_drive_gains_zero':True,'native_drive_gains_readback':gains,
                'native_command_readback':effort,'readback_after_release':True}
