"""Scene-domain preparation for runtime authoring; importing this module needs no Isaac."""
from dataclasses import dataclass
from functools import cached_property

from calibrated_scene_reconstruction.core.contracts.scene import SceneInputs
from calibrated_scene_reconstruction.core.contracts.physics import PhysicalParameters
from calibrated_scene_reconstruction.scene.background.splat.config import settings, task_zone
from calibrated_scene_reconstruction.scene.graph import cousin_parts, structure_role
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.twin.gates import meshed_structures
from .setpiece_inputs import furniture_kits, furniture_overrides, kit_for_object


def _session(value):
    return value if isinstance(value, Session) else Session(value)


@dataclass
class SceneRuntimeInputs:
    session: SceneInputs

    @cached_property
    def simulation(self):
        from calibrated_scene_reconstruction.scene.simulation import settings
        return settings(self.session)

    @cached_property
    def mechanisms(self):
        from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
        return declaration(self.session)

    @cached_property
    def physical_priors(self):
        from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration
        return declaration(self.session)

    @cached_property
    def physics(self) -> PhysicalParameters:
        return self.session.config("physics.json", {})

    @cached_property
    def support_parts(self):
        return cousin_parts(self.session.scene_graph, self.session.manifest)["counter_parts"]

    @cached_property
    def windows(self):
        return [k for k, v in self.session.scene_graph["structure"].items() if structure_role(k, v) == "window"]

    @cached_property
    def furniture_kits(self):
        return furniture_kits(self.session)

    @cached_property
    def support_models(self):
        from calibrated_scene_reconstruction.scene.support.runtime_inputs import models
        return models(self.session)

    @cached_property
    def furniture_overrides(self):
        return furniture_overrides(self.session)

    def object_kit(self, name, fix, vertices, synthetic_boxes):
        boxes = {**self.session.manifest["collision_boxes"], **synthetic_boxes}
        return kit_for_object(self.session, boxes, name, fix, vertices)


@dataclass
class BackgroundRuntimeInputs:
    session: SceneInputs

    @cached_property
    def settings(self):
        return settings(self.session)

    @cached_property
    def inside(self):
        inside, _ = task_zone(self.session, self.settings["crop"])
        return inside

    @cached_property
    def meshed_structures(self):
        return meshed_structures(self.session)

    def environment(self, variant='twin'):
        from calibrated_scene_reconstruction.scene.background.blender_env import verify
        return verify(self.session, variant)


def prepare_scene(session):
    if getattr(session, "qualification_only", False):
        raise ValueError("Unaccepted intrinsic assay inputs cannot be promoted through ordinary prepare_scene")
    return session if isinstance(session, SceneRuntimeInputs) else SceneRuntimeInputs(_session(session))


def prepare_background(session):
    return session if isinstance(session, BackgroundRuntimeInputs) else BackgroundRuntimeInputs(_session(session))
