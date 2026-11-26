"""Isaac Lab asset configs for generated bundles; import after runtime launch."""

from __future__ import annotations

import json
from pathlib import Path

from .sim_parameters import load_parameters


def configure_surface_environment(folder):
    """Call once after reset to install the world's optional still-air drag force."""
    rate = load_parameters(Path(folder))["environment"]["air_drag_rate_per_second"]
    if rate:
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        from .newton_forces import still_air_drag

        model = NewtonManager.backend.model

        def drag_force(state):
            wp.launch(still_air_drag, dim=model.particle_count,
                      inputs=[state.particle_qd, model.particle_inv_mass, rate, state.particle_f],
                      device=model.device)

        NewtonManager.register_state_force_callback(drag_force)
    return {"rate_per_second": rate,
            "model": "Still-air linear force F=-mass*rate*velocity; applied inside every solver substep, not postprocessed motion.",
            "calibrated": False}


def asset_configs(folder, *, physics_path="/World/Generated/Physics", visual_path="/World/Generated/Visuals"):
    """Return scene entries using standard Isaac Lab configs, with no task/controller."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg

    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    parameters = load_parameters(folder)
    kind = manifest["asset_kind"]
    if kind == "deformable":
        from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
        from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

        return {
            "asset": DeformableObjectCfg(
                prim_path=physics_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(folder / "newton_surface.usda"),
                    deformable_props=NewtonDeformableBodyPropertiesCfg(),
                    physics_material=NewtonSurfaceDeformableBodyMaterialCfg(**parameters["surface"]),
                ),
            ),
            "visuals": AssetBaseCfg(prim_path=visual_path, spawn=sim_utils.UsdFileCfg(usd_path=str(folder / "visual.usdc"))),
        }
    spawn = sim_utils.UsdFileCfg(usd_path=str(folder / "physx.usda"))
    if kind == "articulated":
        return {"asset": ArticulationCfg(prim_path=physics_path, spawn=spawn, actuators={})}
    if kind == "rigid":
        return {"asset": RigidObjectCfg(prim_path=physics_path, spawn=spawn)}
    raise ValueError(f"No verified runtime loader for {kind!r}")


def bind_surface_visuals(folder, stage, *, physics_path="/World/Generated/Physics", visual_path="/World/Generated/Visuals"):
    """Create a visual follower; call its update with world-space Newton nodes."""
    from pxr import UsdGeom

    from .usd_surface_binding import UsdSurfaceBinding

    binding = UsdSurfaceBinding(folder, stage, root=visual_path)
    mesh = stage.GetPrimAtPath(physics_path + "/SimMesh")
    if not mesh:
        raise ValueError(f"No control mesh under {physics_path}")
    UsdGeom.Imageable(mesh).MakeInvisible()
    return binding
