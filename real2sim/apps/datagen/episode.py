"""Compose a rigid episode recorder or policy evaluation after backend startup."""
from calibrated_scene_reconstruction.tasks.rigid.cli import parse_args, W, H

args = parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from calibrated_scene_reconstruction.grasping.place import PLACE_SEG, PlaceCfg, PlaceGoal, fixed_style, keep_out, place_outcome, place_succeeded, receptacle_weights, sample_goal, support_obstacles, task_string, task_type, tilt_deg  # noqa: E402
from calibrated_scene_reconstruction.grasping.targets import arm_direction, collision_parts, grasp_key, hand_name, load_records, local_mesh, mesh_key, scene_env, scene_env_world, target_spec  # noqa: E402
from calibrated_scene_reconstruction.grasping.trajgen import SEG, Traj, TrajCfg, TrajectoryPlanner, analytic_target, bodex_target, hull_planes  # noqa: E402
from calibrated_scene_reconstruction.grasping.trajgen import fixed_style as traj_fixed_style  # noqa: E402
from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_cameras, add_robot, clear_support, hand_links, hand_points, place_robot  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.receptacle import add_receptacle, remove_receptacle  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import VideoWriter, encode_video, grab, load_variant, rgb_annotator, warm  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import room_obstacles  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import DEPTH_OK, RobotSweep, children, clear_robot, summarize, support_collisions  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole  # noqa: E402
from calibrated_scene_reconstruction.scene.cousins.variants import fit_target  # noqa: E402
from pxr import Sdf, UsdLux  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.batch_kinematics import BatchKinematics  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.runtime import EpisodeRuntime  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.services import EpisodeServices  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.evaluate import evaluate_policy  # noqa: E402
from calibrated_scene_reconstruction.datagen.recording import record_step, record, write_episode, record_batches, capture_frame  # noqa: E402
from calibrated_scene_reconstruction.policy.rollout import make_runner, run_rollout  # noqa: E402

services = EpisodeServices(
    BatchKinematics=BatchKinematics,
    DEPTH_OK=DEPTH_OK,
    PLACE_SEG=PLACE_SEG,
    PlaceCfg=PlaceCfg,
    PlaceGoal=PlaceGoal,
    Robot=Robot,
    RobotSweep=RobotSweep,
    SEG=SEG,
    Sdf=Sdf,
    Session=Session,
    SingleRigidPrim=SingleRigidPrim,
    Traj=Traj,
    TrajCfg=TrajCfg,
    TrajectoryPlanner=TrajectoryPlanner,
    Usd=Usd,
    UsdGeom=UsdGeom,
    UsdLux=UsdLux,
    VideoWriter=VideoWriter,
    World=World,
    add_camera=add_camera,
    add_cameras=add_cameras,
    add_receptacle=add_receptacle,
    add_robot=add_robot,
    analytic_target=analytic_target,
    arm_direction=arm_direction,
    bodex_target=bodex_target,
    build_scene=build_scene,
    children=children,
    clear_robot=clear_robot,
    clear_support=clear_support,
    collision_parts=collision_parts,
    configure_render=configure_render,
    encode_video=encode_video,
    evaluate_policy=evaluate_policy,
    fit_target=fit_target,
    fixed_style=fixed_style,
    grab=grab,
    grasp_key=grasp_key,
    hand_links=hand_links,
    hand_name=hand_name,
    hand_points=hand_points,
    hull_planes=hull_planes,
    keep_out=keep_out,
    load_records=load_records,
    load_variant=load_variant,
    local_mesh=local_mesh,
    mesh_key=mesh_key,
    omni=omni,
    pinhole=pinhole,
    place_outcome=place_outcome,
    place_robot=place_robot,
    place_succeeded=place_succeeded,
    prepare_scene=prepare_scene,
    prepare_variant=prepare_variant,
    receptacle_weights=receptacle_weights,
    record=record,
    record_batches=record_batches,
    record_step=record_step,
    remove_receptacle=remove_receptacle,
    rgb_annotator=rgb_annotator,
    room_obstacles=room_obstacles,
    sample_goal=sample_goal,
    scene_env=scene_env,
    scene_env_world=scene_env_world,
    set_pose=set_pose,
    summarize=summarize,
    support_collisions=support_collisions,
    support_obstacles=support_obstacles,
    target_spec=target_spec,
    task_string=task_string,
    task_type=task_type,
    tilt_deg=tilt_deg,
    traj_fixed_style=traj_fixed_style,
    warm=warm,
    write_episode=write_episode,
    capture_frame=capture_frame,
    make_runner=make_runner,
    run_rollout=run_rollout,
)
EpisodeRuntime(args, app, services).run()
