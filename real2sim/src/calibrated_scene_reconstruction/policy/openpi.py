"""OpenPI joint-space data adapters and JSON-configured training registrations.

Launch the upstream training/normalization scripts through tools/policy/openpi_run.py.
Task and dataset identifiers live in configs/policy/openpi.json. Legacy class and
configuration identifiers remain available for existing training jobs.
"""
import dataclasses
import pathlib
import json

from calibrated_scene_reconstruction.core.paths import CONFIGS as CONFIG_ROOT

import einops
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders

N_ARM, N_HAND = 7, 22


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class JointInputs(_transforms.DataTransformFn):
    """openpi's LiberoInputs with one camera: the head image the base view, both wrist views zero and masked."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        pad = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": data["observation/state"],
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": np.zeros_like(base),
                      "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": pad, "right_wrist_0_rgb": pad},
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class JointOutputs(_transforms.DataTransformFn):
    arm_dofs: int = N_ARM
    hand_dofs: int = N_HAND

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :self.arm_dofs + self.hand_dofs])}


@dataclasses.dataclass(frozen=True)
class LeRobotJointDataConfig(_config.DataConfigFactory):
    """openpi's LeRobotLiberoDataConfig for tools/export_openpi.py's dataset, with the delta transform its comments
    prescribe for absolute joint targets (the arm relative, the hand absolute, as a gripper)."""

    arm_dofs: int = N_ARM
    hand_dofs: int = N_HAND

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "observation.images.head",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })])
        mask = _transforms.make_bool_mask(self.arm_dofs, -self.hand_dofs)
        data_transforms = _transforms.Group(
            inputs=[JointInputs(model_type=model_config.model_type), _transforms.DeltaActions(mask)],
            outputs=[_transforms.AbsoluteActions(mask), JointOutputs(self.arm_dofs, self.hand_dofs)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


# Existing checkpoint and training integrations may still import these names.
Lab01Inputs = JointInputs
Lab01Outputs = JointOutputs
LeRobotLab01DataConfig = LeRobotJointDataConfig


def register_configs(path=None):
    """Register application configurations without embedding a capture identifier in code."""
    source = pathlib.Path(path) if path is not None else CONFIG_ROOT / "policy" / "openpi.json"
    records = json.loads(source.read_text())["train_configs"]
    configs = []
    for entry in records:
        data = entry["data"]
        base = dict(data.get("base_config", {}))
        if "action_sequence_keys" in base:
            base["action_sequence_keys"] = tuple(base["action_sequence_keys"])
        config = _config.TrainConfig(
            name=entry["name"],
            model=pi0_config.Pi0Config(**entry["model"]),
            data=LeRobotJointDataConfig(
                repo_id=data["repo_id"],
                arm_dofs=data.get("arm_dofs", N_ARM), hand_dofs=data.get("hand_dofs", N_HAND),
                base_config=_config.DataConfig(**base),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(entry["weights"]),
            **entry.get("training", {}),
        )
        _config._CONFIGS_DICT[config.name] = config
        configs.append(config)
    return configs


CONFIGS = register_configs()
