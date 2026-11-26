"""Rebind readback handles after native actor retirement, without changing state."""


class WipeViewRegistry:
    def __init__(self, robot, cloth, contacts):
        self.robot, self.cloth, self.contacts = robot, cloth, contacts
        self.simulation = robot.world.physics_sim_view

    @staticmethod
    def stale(wrapper):
        native = getattr(wrapper, '_physics_view', None)
        return native is None or getattr(native, '_backend', None) is None

    def refresh(self):
        simulation = self.robot._ensure_views()
        changed = simulation is not self.simulation
        if hasattr(self.cloth, 'panel_view'):
            panel = self.cloth.panel_view
            if changed or self.stale(panel):
                panel._physics_view = None
                panel.initialize(simulation)
                paths = list(panel._physics_view.prim_paths)
                self.cloth.panel_order = [paths.index(p) for p in self.cloth.panel_paths]
        elif changed:
            self.cloth.initialize()
        if self.contacts is not None:
            contact = self.contacts._contact_view
            if changed or self.stale(contact):
                contact._physics_view = None
                contact.initialize(simulation)
        self.simulation = simulation
