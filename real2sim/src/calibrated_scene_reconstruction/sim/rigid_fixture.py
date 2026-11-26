"""Temporary kinematic holding without the unsupported kinematic+CCD state."""


class KinematicFixture:
    """Preserve CCD and kinematic opinions; never change material or geometry.

    Holding is an explicit test intervention, not contact or grasp evidence.
    Release switches back to dynamic before restoring the original CCD value.
    """

    def __init__(self, prim):
        from pxr import UsdPhysics
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('A native rigid body is required for a temporary fixture')
        api = UsdPhysics.RigidBodyAPI(prim)
        self.kinematic = api.CreateKinematicEnabledAttr()
        if self.kinematic.Get():
            raise ValueError('Temporary fixture requires an initially dynamic body')
        self.original_kinematic = self.kinematic.Get()
        self.kinematic_authored = self.kinematic.HasAuthoredValueOpinion()
        self.ccd = prim.GetAttribute('physxRigidBody:enableCCD')
        self.original_ccd = self.ccd.Get() if self.ccd else None
        self.ccd_authored = self.ccd.HasAuthoredValueOpinion() if self.ccd else False
        # PhysX rejects kinematic+CCD. Disable only for this disclosed hold;
        # contact offsets, SDF, mass, friction and all dynamic bodies are intact.
        if self.ccd and self.original_ccd:
            self.ccd.Set(False)
        self.kinematic.Set(True)
        self.active = True

    def release(self):
        if not self.active:
            raise ValueError('Temporary fixture was already released')
        self.kinematic.Set(False)
        if self.ccd:
            if self.ccd_authored:
                self.ccd.Set(self.original_ccd)
            else:
                self.ccd.Clear()
        if self.kinematic_authored:
            self.kinematic.Set(self.original_kinematic)
        else:
            self.kinematic.Clear()
        self.active = False

    def report(self):
        return {'kind': 'temporary_kinematic_fixture', 'active': self.active,
                'original_kinematic': self.original_kinematic, 'original_ccd': self.original_ccd,
                'ccd_disabled_only_while_kinematic': bool(self.original_ccd),
                'geometry_material_and_collision_precision_changed': False,
                'scope': 'Disclosed test hold; not autonomous grasp/contact evidence.'}
