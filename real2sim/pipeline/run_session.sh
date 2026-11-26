#!/usr/bin/env bash
# End-to-end: photo -> instance-level relightable twin -> photoreal cousins -> robot episodes -> dataset + video.
#
# Usage: pipeline/run_session.sh <name> [photo.jpg | walkaround.mp4] [--views "v1.jpg v2.jpg ..."] [--photo-at SEC] [--max-views N]
#                                [--from STAGE] [--to STAGE] [--cousins N] [--robots "R1 R2 ..."]
#                                [--ep-cousins N] [--ep-per-scene N] [--batch N] [--task pick|place] [--auto] [--video] [--splat]
#                                [--demo real/<demo> [--demo-cfg configs/demo_tasks/<task>.json] [--demo-n N]]
#   STAGE: calib robot_cloud perceive multiview graph poses anchor support support_model objects align furniture library retrieve audit inpaint
#          panorama layout irradiance albedo surfaces materials task_zone views cousins cousin_rooms walls completion_review
#          room_objects clearance object_review placements robot_views room_views splat_capture splat_masks splat_train splat_fix
#          splat_export splat_review environment_build deliver fly grasps receptacles demo episodes
#   configs/<name>/twin.json (docs/twin_pipeline.md): the twin's background (splat | mesh | completion) and scale (board |
#              robot_joints | robot_parts | prior); with it the stages S0-S6 run -- calib, robot_cloud, poses,
#              anchor, support, task_zone, deliver -- and each stage's review gate is its record in qa/review/
#          export video
#   --splat    also the twin's Gaussian-splat background (or configs/<name>/splat.json: more videos, settings): what the
#              session's videos saw around the task as a NuRec volume beside the meshes (envs/splat, recon.splat_*,
#              sim.splat); the apps take it with --background splat; off, the twin is the mesh twin unchanged
#   --views    more photos of the same scene (capture/views/): recon.multiview puts them in the photo's frame (Depth
#              Anything 3), takes the scale from the robot in the picture (configs/robots/scale_refs) and measures
#              the objects from several sides; without views the stage does nothing unless configs/<name>/multiview.json
#              asks for the robot scale on the photo alone
#   a video    (walked around the scene, the robot facing the camera at --photo-at seconds, default 0.5) gives the
#              photo and the views at once: calibrated_scene_reconstruction.scene.capture.keyframes takes the sharpest frame near --photo-at as the photo and
#              the sharpest of --max-views (default 16) equal stretches of the walk as the views (capture/keyframes.jpg)
#   --robots   robot specs (configs/robots/*.json) to collect episodes with; default vega_1p_sharpa
#   --ep-cousins  record episodes in the twin and the first N cousins only (default: all)
#   --batch N     validate N planned trajectories at once in N copies of the task scene in one stage (default 6, 0: one by one)
#   --ep-per-scene  successful episodes per scene and robot (default 1): each a different grasp / random trajectory
#   --task     pick (grasp + lift) or place (+ transport to a sampled goal, place, release, retreat:
#              calibrated_scene_reconstruction.grasping.place); default: the scene graph's task.type, else pick. Place datasets go to
#              outputs/lerobot/<name>/place/<robot>
# grasps: BODex synthesis (envs/grasp, calibrated_scene_reconstruction.grasping.synth) for robots whose hand has a BODex asset (ee.bodex);
# object_review: the photo measurements of the agent's set pieces (calibrated_scene_reconstruction.scene.objects.setpiece_observations), then apps/scene/review_objects.py
#   on the twin and every cousin: every object and table / counter top checked (floating, sunk, intersecting, size,
#   glass / metal built as an opaque shell) and rendered beside the photo; the agent judges qa/objects/<tag>/ as a
#   photograph would be judged, writes fixes into the scene graph (docs/object_review.md) and a top-level
#   `_object_review` note; without the note (and without --auto) the run stops here: rerun with --from clearance
#   (the fixes act where scenes are built; from `views` the cousins and their room objects would be made again)
# receptacles (place task only): apps/review_receptacles.py samples each place-target kind of the scene's room as the
#   episodes will, checks it automatically and renders it; the agent reviews qa/receptacles/sheet.jpg and writes its
#   decision into the scene graph's task.place with a `_review` note (docs/pick_place_policy.md); without that note
#   (and without --auto) the run stops here: rerun with --from episodes.
# episodes: apps/datagen/episode.py -- synthesized or analytic grasps, randomized arm trajectories (calibrated_scene_reconstruction.grasping.trajgen).
#   --to       stop after this stage (a review point: e.g. --to views, look at the twin, then --from cousins)
#   --auto     accept the drafted scene graph without review (configs/<name>/scene_graph.auto.json)
#   --demo     one demonstration of an unseen task (docs/demo_datagen.md): the demo stage re-targets it onto the twin
#              and the cousins (calibrated_scene_reconstruction.demo.generate: N TCP plans, default 200); their Isaac Sim playback is TODO
#   --video    also render the robot fly-through and compose the demo video (outputs/<name>.mp4): a visualization
#              for one or two showcase photos, not part of data generation (off by default)
# Environment: MAX_OBJECTS (default 10: objects kept around the task by autograph), LIBRARY_FETCH (default
# "stool=8 bench=4 table=8": visually diverse ABO models for the cousins' furniture and the completion's synthetic
# pieces), PANO_MODEL (sdxl | flux), BATCH.
#
# configs/<name>/: scene_graph.json is drafted by calibrated_scene_reconstruction.scene.twin.graph after `perceive` (review it: the overlay is
# sessions/<name>/parsing/qa_autograph.png); perception.json (SAM 3 prompts, default kitchen vocabulary),
# completion.json (panorama prompts), views.json and lights.json (calibrated_scene_reconstruction.scene.review.views) and cousins.json (optional
# part overrides / static swaps) are optional. README.md lists what the agent decides at each of these points.
set -euo pipefail
# Handle help before parsing a session name, loading configuration or creating output directories.
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: pipeline/run_session.sh <name> [photo.jpg | video.mp4] [options]
Options: --views FILES --photo-at SEC --max-views N --from STAGE --to STAGE
         --robots ROBOTS --cousins N --ep-cousins N --ep-per-scene N --batch N
         --task pick|place --auto --video --splat --demo DIR --demo-cfg FILE --demo-n N
         --object-route agent_blender|template (default: agent_blender)
         --reuse-camera-model --pose-workers N
         --robot-execution (optional robot data generation; not reconstruction QA)
See docs/pipeline.md for stages and required inputs.
HELP
      exit 0
      ;;
  esac
done
[ "$#" -gt 0 ] || { echo "A session name is required; use --help" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NAME="$1"; shift
PHOTO=""; VIEWS=""; PHOTO_AT=""; MAX_VIEWS=""; FROM="calib"; NCOUSINS=6; ROBOTS="vega_1p_sharpa"; AUTO=0; EPC=999; EPS=1; VIDEO=0; TASK=""
SPLAT=0
OBJECT_ROUTE=agent_blender
REUSE_CAMERA_MODEL=0
POSE_WORKERS=1
ROBOT_EXECUTION=0
DEMO=""; DEMO_CFG=""; DEMO_N=200
BATCH="${BATCH:-6}"  # trajectories validated at once, each in its own copy of the task scene (apps/datagen/episode.py --batch)
while [ $# -gt 0 ]; do
  case "$1" in
    --object-route) OBJECT_ROUTE="$2"; shift 2 ;;
    --reuse-camera-model) REUSE_CAMERA_MODEL=1; shift ;;
    --pose-workers) POSE_WORKERS="$2"; shift 2 ;;
    --robot-execution) ROBOT_EXECUTION=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --views) VIEWS="$2"; shift 2 ;;
    --photo-at) PHOTO_AT="$2"; shift 2 ;;
    --max-views) MAX_VIEWS="$2"; shift 2 ;;
    --cousins) NCOUSINS="$2"; shift 2 ;;
    --robots) ROBOTS="$2"; shift 2 ;;
    --ep-cousins) EPC="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --ep-per-scene) EPS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --auto) AUTO=1; shift ;;
    --video) VIDEO=1; shift ;;
    --splat) SPLAT=1; shift ;;
    --demo) DEMO="$2"; shift 2 ;;
    --demo-cfg) DEMO_CFG="--cfg $2"; shift 2 ;;
    --demo-n) DEMO_N="$2"; shift 2 ;;
    *) PHOTO="$1"; shift ;;
  esac
done
case "$OBJECT_ROUTE" in agent_blender|template) ;; *) echo "Invalid --object-route: $OBJECT_ROUTE" >&2; exit 2 ;; esac
[[ "$POSE_WORKERS" = -1 || "$POSE_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "--pose-workers must be -1 or a positive integer" >&2; exit 2; }
WORKER_ARGS=(); [ "$POSE_WORKERS" = 1 ] || WORKER_ARGS=(--workers "$POSE_WORKERS")
S="sessions/$NAME"
C="configs/$NAME"
require_camera_model() {
  [ -d "$S/splat/data/sparse/full" ] && [ -d "$S/splat/data/images" ] || {
    echo "--reuse-camera-model requires this session's COLMAP model and images; no reconstruction cache exists" >&2; return 2; }
}
# Resuming at poses can reject a missing reuse cache before loading the reconstruction runtime.
if [ "$REUSE_CAMERA_MODEL" = 1 ] && [ "$FROM" = poses ]; then require_camera_model; fi
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.capture.inputs --check-inputs "$NAME"
LOG="$S/logs"; mkdir -p "$S/capture" "$LOG" "$C"
RECON="tools/run_capped.sh envs/recon/bin/python -m"
ISAAC="tools/run_isaac.sh"
object_route() {
  $RECON calibrated_scene_reconstruction.scene.objects.routes "$S" --object-route "$OBJECT_ROUTE" --apply "$@"
}
run=0
step() {  # --to: the stages after it are skipped
  [ "$1" = "$FROM" ] && run=1; [ $run = 1 ] && [ "${past_to:-0}" = 0 ] || return 1; [ "$1" = "${TO:-}" ] && past_to=1; return 0; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
# a failed step says where it stopped (set -e ended runs without a word: a layout's KeyError looked like a gate)
trap 'say "FAILED at line $LINENO (exit $?): $BASH_COMMAND -- see $LOG"' ERR
tagged() { [ "$1" = vega_1p_sharpa ] && echo "$2" || echo "$2__$1"; }  # episodes/<tag>[__<robot>][__e<k>]
clear_episodes() {  # this robot's episode directories only (result.json names the robot)
  python3 - "$S" "$1" <<'PY'
import json, shutil, sys
from pathlib import Path
for d in Path(sys.argv[1], "episodes").glob("*/"):
    try:
        robot = json.loads((d / "result.json").read_text()).get("robot")
    except (OSError, ValueError):
        continue
    if robot == sys.argv[2]:
        shutil.rmtree(d)
PY
}

# the photo (and the views) upright by their EXIF orientation, at most 2400 px on the long side; a video's key frames
# as the photo and the views (calibrated_scene_reconstruction.scene.capture.capture)
if [ -n "$PHOTO" ]; then old=$(md5sum "$S/capture/rgb.png" 2>/dev/null | cut -c1-32 || true)
  PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.capture.capture "$S" "$PHOTO" ${VIEWS:+--views $VIEWS} \
  ${PHOTO_AT:+--photo-at $PHOTO_AT} ${MAX_VIEWS:+--max-views $MAX_VIEWS}
  # a new photo: the world frame drawn from the old one goes (autograph reused it: a photo taken 10 s later lost its
  # desk and task)
  [ "$old" = "$(md5sum "$S/capture/rgb.png" | cut -c1-32)" ] || rm -f "$S/parsing/world_frame.json"; fi
[ -f "$S/capture/rgb.png" ] || { echo "no photo: pass one or put $S/capture/rgb.png"; exit 1; }

# the twin's options (configs/<name>/twin.json, docs/twin_pipeline.md): the background and the scale; with them the
# stages S0-S6 and their review gates (qa/review/<stage>_r<k>.json). Without twin.json the pipeline runs as before.
SPL="tools/run_splat.sh -m"
TWIN=0; BACKGROUND=""; SCALE=""; TASK_LOOK=kit; ZONE=0; SUPPORT_ROUTE=carve
if [ -f "$C/twin.json" ]; then
  TWIN=1
  TASK_LOOK=$(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" get task_look)
  BACKGROUND=$(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" get background)
  SCALE=$(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" get scale)
  SUPPORT_ROUTE=$(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" get support_route)
  NCOUSINS=$(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" get cousins)
  [ "$BACKGROUND" = splat ] && SPLAT=1
  [ "$BACKGROUND" = zone ] && { ZONE=1; NCOUSINS=0; }  # the task zone alone: no room, no splat, no cousins
  # Native agent environments replace legacy room reconstruction and task-changing cousins.
  [ "$BACKGROUND" = blender ] && { ZONE=1; NCOUSINS=0; }
  say "twin: background $BACKGROUND, scale $SCALE (configs/$NAME/twin.json)"
fi
VIDEOCAP=0; [ -f "$S/capture/keyframes.json" ] && VIDEOCAP=1
gate() {  # gate STAGE "what the agent reviews": a twin's stage passes on its review record (docs/twin_pipeline.md)
  [ $TWIN = 1 ] || return 0
  # Review evidence must describe the scene produced by the current stage.
  mkdir -p "$S/qa/review"; touch "$S/qa/review/$1.ran"
  PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.gates "$S" gate "$1" && return 0
  say "agent ($1): $2; record the round in $S/qa/review/$1_r<k>.json (docs/twin_pipeline.md section 4), then rerun with --from $3"
  [ $AUTO = 1 ] || exit 0; }

# S0: the session video's camera (its own frames, when there is no calibration) and the robot at its joint readout
if step calib && [ $TWIN = 1 ] && [ $VIDEOCAP = 1 ]; then say "S0 camera: self-calibration of the session video"
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.capture "$S" --calibrate > "$LOG/calib.log" 2>&1; grep -h "^done" "$LOG/calib.log" | sed 's/^/  /' || true
  MEM=16G $RECON calibrated_scene_reconstruction.scene.capture.rectification "$S" > "$LOG/capture_rectification.log" 2>&1; fi
if step robot_cloud && [ "$SCALE" != robot_visual ] && [ "$SCALE" != robot_joints ] && [ -f "$C/robot_state.json" ]; then say "S0 the robot's surface at the capture's joint readout"
  $ISAAC apps/robot_cloud.py "$S" --robot "${ROBOTS%% *}" > "$LOG/robot_cloud.log" 2>&1; grep -h "^done" "$LOG/robot_cloud.log" | sed 's/^/  /' || true; fi

if step perceive; then say perceive; MEM=24G $RECON calibrated_scene_reconstruction.scene.capture.perceive "$S" > "$LOG/perceive.log" 2>&1; grep -h "^done" "$LOG/perceive.log"; fi
if step perceive && [ $TWIN = 1 ] && [ $VIDEOCAP = 0 ]; then
  gate S0 "the source photo, its camera inputs and task-region visibility ($S/capture, $S/parsing/overlay.png)" perceive; fi
if step multiview; then say multiview; MEM=40G $RECON calibrated_scene_reconstruction.scene.capture.multiview "$S" > "$LOG/multiview.log" 2>&1; grep -h "^done\|^\[scale\] x" "$LOG/multiview.log"; fi
if step graph; then
  # a world frame newer than the reviewed scene graph (recon.support_carve turned it to the support): the graph's
  # boxes are of the old world -- drafted again and reviewed again (the old one kept beside it)
  if [ -f "$C/scene_graph.json" ] && [ "$S/parsing/world_frame.json" -nt "$C/scene_graph.json" ]; then
    mv "$C/scene_graph.json" "$C/scene_graph.prev.json"; say "the world frame changed: the scene graph drafted again"; fi
  if [ ! -f "$C/scene_graph.json" ]; then
    say "scene graph draft (autograph)"
    MEM=16G $RECON calibrated_scene_reconstruction.scene.twin.graph "$S" --max-objects "${MAX_OBJECTS:-10}" > "$LOG/autograph.log" 2>&1
    grep -A4 '"task"' "$LOG/autograph.log" | tr -d '\n' | sed 's/  */ /g'; echo
    if [ $AUTO = 1 ]; then cp "$C/scene_graph.auto.json" "$C/scene_graph.json"
    else say "review $C/scene_graph.auto.json against $S/parsing/qa_autograph.png, save it as scene_graph.json, then rerun with --from $([ $TWIN = 1 ] && echo poses || echo objects)"; exit 0; fi
  fi
fi
if step graph && [ $TWIN = 1 ] && [ $VIDEOCAP = 0 ]; then
  gate S1 "the metric scale and world frame ($LOG/multiview.log, $S/parsing/world_frame.json), checked against permitted calibration evidence" graph; fi
# S0 / S1 with a video: COLMAP poses of every frame in the twin's world (any background: the task zone is measured
# on them), the scale's place -- robot_joints refines uncertain commanded articulation against current images;
# sparse points, no splat), the robot's base -- and every frame's camera (recon.posed_frames select)
if step poses && { [ $TWIN = 1 ] || [ $SPLAT = 1 ]; } && [ $VIDEOCAP = 1 ]; then say "S0 poses: COLMAP of the session's videos, the twin's world, coverage"
  CAMERA_ARGS=()
  if [ "$REUSE_CAMERA_MODEL" = 1 ]; then
    require_camera_model
    CAMERA_ARGS=(--reuse-model)
  fi
  [ $SPLAT = 1 ] && [ ! -f "$C/splat.json" ] && echo '{}' > "$C/splat.json"
  # Sparse texture on the primary support may need other actual registered
  # observations. Keep the point-count gate and reuse only this capture's map.
  rm -f "$S/qa/placement_plane_request.json"
  pose_rc=0
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.capture "$S" "${CAMERA_ARGS[@]}" > "$LOG/splat_capture.log" 2>&1 || pose_rc=$?
  if [ "$pose_rc" != 0 ] && [ -f "$S/qa/placement_plane_request.json" ]; then
    cp "$LOG/splat_capture.log" "$LOG/splat_capture_plane_request.log"
    MEM=24G $RECON calibrated_scene_reconstruction.scene.capture.plane_observations "$S" --request > "$LOG/placement_plane_masks.log" 2>&1
    MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.capture "$S" --reuse-model > "$LOG/splat_capture.log" 2>&1
  elif [ "$pose_rc" != 0 ]; then exit "$pose_rc"; fi
  grep -h "^\[colmap\] registered\|^\[align\]\|^\[board\]\|^\[coverage\]\|^\[zone\]" "$LOG/splat_capture.log" | sed 's/^/  /' || true
  gate S0 "the capture: registered frames ($LOG/splat_capture.log), the task zone's coverage ($S/splat/coverage.png), one capture state (the task zone the same in every video)" poses; fi
if step anchor && [ $TWIN = 1 ] && [ $VIDEOCAP = 1 ]; then say "S1 the frames placed by the scale ($SCALE), the robot's base, every frame's camera"
  # (no measured size is read: a deployed capture has none -- recon.scale_refs is not a twin's stage)
  if [ "$SCALE" = robot_visual ] || [ "$SCALE" = robot_joints ]; then
    # Segment the current robot and independent occluders. Unestimated fingers
    # remain unknown. Only robot_joints reads a command prior; neither route
    # reads board scale or task/support dimensions.
    VISUAL_CONFIG=robot_visual.json
    [ "$SCALE" != robot_joints ] || VISUAL_CONFIG=robot_calibration.json
    MEM=24G $RECON calibrated_scene_reconstruction.scene.capture.calibration_masks "$S" --config "$VISUAL_CONFIG" > "$LOG/calibration_masks.log" 2>&1
    JOINT_PRIOR_ARGS=()
    [ "$SCALE" != robot_joints ] || JOINT_PRIOR_ARGS=(--joint-prior)
    MEM=24G $SPL calibrated_scene_reconstruction.scene.capture.visual_render_pipeline "$S" prepare "${JOINT_PRIOR_ARGS[@]}" > "$LOG/robot_visual_prepare.log" 2>&1
    read -r VISUAL_INPUT VISUAL_OUTPUT < <(PYTHONPATH=src envs/recon/bin/python -c 'import json,sys; from pathlib import Path; d=json.load(open(Path(sys.argv[1])/"splat/robot_visual_run.json")); print(Path(d["input"]).resolve(),Path(d["output"]).resolve())' "$S")
    MEM=24G $RECON calibrated_scene_reconstruction.scene.capture.visual_render_fit "$VISUAL_INPUT" "$VISUAL_OUTPUT" > "$LOG/robot_visual_render.log" 2>&1
    MEM=24G $SPL calibrated_scene_reconstruction.scene.capture.visual_render_pipeline "$S" assess --apply > "$LOG/robot_visual.log" 2>&1
    tail -30 "$LOG/robot_visual.log"
  elif [ -f "$S/parsing/robot_cloud.npz" ]; then  # the robot against the placed frames: the base, and the scale's check
    MEM=16G $SPL calibrated_scene_reconstruction.scene.capture.robot_fit "$S" --points sparse --apply "${WORKER_ARGS[@]}" > "$LOG/robot_fit.log" 2>&1; grep -h "^\[robot\]" "$LOG/robot_fit.log" | sed 's/^/  /' || true
  else  # no readout: the robot's place from its silhouettes in the frames (its torso carved as a column)
    $ISAAC apps/robot_cloud.py "$S" --robot "${ROBOTS%% *}" --state home > "$LOG/robot_cloud_home.log" 2>&1 || true
    MEM=24G $RECON calibrated_scene_reconstruction.scene.background.splat.masks "$S" --prompts "humanoid robot,robot torso,robot head,mobile robot base" --every 4 --suffix _robot --threshold 0.3 --inside 0 > "$LOG/splat_masks_robot.log" 2>&1
    MEM=16G $SPL calibrated_scene_reconstruction.scene.capture.robot_carve "$S" --apply > "$LOG/robot_carve.log" 2>&1; grep -h "^\[robot_carve\]" "$LOG/robot_carve.log" | sed 's/^/  /' || true; fi
  MEM=8G $SPL calibrated_scene_reconstruction.scene.capture.posed_frames "$S" select > "$LOG/posed_select.log" 2>&1; grep -h "^done" "$LOG/posed_select.log" | sed 's/^/  /' || true
  gate S1 "the scale against its check ($LOG/multiview.log [scale], $LOG/robot_anchor.log, $LOG/robot_fit.log, $S/qa/splat/robot_fit.png), the robot and the support overlaid on the frames" anchor; fi
# S2a the support's footprint carved from the frames (the support itself, then what stands between, in every 4th frame)
if step support && [ $TWIN = 1 ] && [ $VIDEOCAP = 1 ]; then say "S2 the support carved from the frames"
  read -r SUPPORT_MASK_SUFFIX SUPPORT_OCCLUDER_SUFFIX < <(PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.support.mask_inputs "$S")
  MEM=24G $RECON calibrated_scene_reconstruction.scene.background.splat.masks "$S" --role support --every 4 --suffix "$SUPPORT_MASK_SUFFIX" --threshold 0.3 --reuse > "$LOG/splat_masks_carve.log" 2>&1
  MEM=24G $RECON calibrated_scene_reconstruction.scene.background.splat.masks "$S" --role occluders --every 4 --suffix "$SUPPORT_OCCLUDER_SUFFIX" --threshold 0.3 --reuse > "$LOG/splat_masks_occ.log" 2>&1
  rc=0; MEM=16G $SPL calibrated_scene_reconstruction.scene.support.support_carve "$S" --apply > "$LOG/support_carve.log" 2>&1 || rc=$?
  grep -h "^\[carve\]\|^done" "$LOG/support_carve.log" | sed 's/^/  /' || true
  if [ $rc = 3 ]; then  # background zone, the support askew in the room: the world turned to its edges
    say "the world turned to the support's edges (parsing/world_frame.json): rerun with --from graph"; exit 0
  elif [ $rc != 0 ]; then exit $rc; fi; fi
if step support_model && [ "$SUPPORT_ROUTE" = agent_blender ]; then say "S2 the task support's agent Blender iteration"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.support.observations "$S" > "$LOG/support_observations.log" 2>&1
  if [ $TWIN = 1 ] && [ $VIDEOCAP = 1 ]; then
    MEM=16G $SPL calibrated_scene_reconstruction.scene.support.plane_tracks "$S" > "$LOG/support_plane.log" 2>&1
  fi
  MEM=24G $RECON calibrated_scene_reconstruction.scene.support.blender "$S" refine > "$LOG/support_model.log" 2>&1
  SUPPORT_NAME=$(PYTHONPATH=src envs/recon/bin/python -c 'import sys; from calibrated_scene_reconstruction.scene.session import Session; print(Session(sys.argv[1]).scene_graph["task"]["support"])' "$S")
  gate "SUP.$SUPPORT_NAME" "the support model's current multi-view masks, observed top, visible edges/legs and Blender previews ($LOG/support_model.log, $S/qa/support)" support_model; fi
if step objects; then object_route; say "objects (SAM 3D initialization; task models use $OBJECT_ROUTE)"; MEM=30G $RECON calibrated_scene_reconstruction.scene.objects.reconstruct "$S" > "$LOG/objects.log" 2>&1; fi
if step align; then say align; MEM=16G $RECON calibrated_scene_reconstruction.scene.objects.align "$S" > "$LOG/align.log" 2>&1; grep "IoU" "$LOG/align.log" || true
  $RECON calibrated_scene_reconstruction.scene.review.views "$S" > "$LOG/autoviews.log" 2>&1; cat "$LOG/autoviews.log" | grep -v run_capped; fi
if step furniture && [ $ZONE = 0 ]; then say "furniture (shared SAM 3D reconstruction)"; MEM=30G $RECON calibrated_scene_reconstruction.scene.support.furniture "$S" > "$LOG/furniture.log" 2>&1; grep "^\[" "$LOG/furniture.log" | grep -v "near-best" || true; fi
if step library && [ $ZONE = 0 ]; then say "library (ABO catalog + cousin furniture set)"
  MEM=16G $RECON calibrated_scene_reconstruction.assets.abo --fetch ${LIBRARY_FETCH:-stool=8 bench=4 table=8} > "$LOG/library.log" 2>&1; grep -E "models" "$LOG/library.log" || true; fi
if step retrieve && [ $ZONE = 0 ]; then say "retrieval fallback + USD conversion and drop test of library models"
  MEM=16G $RECON calibrated_scene_reconstruction.scene.objects.retrieve "$S" > "$LOG/retrieve.log" 2>&1; grep -E "^\[.*->|^nothing" "$LOG/retrieve.log" || true
  $ISAAC apps/gltf_to_usd.py > "$LOG/library_usd.log" 2>&1
  $ISAAC apps/library_drop_test.py > "$LOG/library_drop.log" 2>&1; grep -h "^drop test" "$LOG/library_drop.log" | sed 's/^/  /' || true; fi
if step audit && [ $ZONE = 0 ]; then say "reconstruction audit (photo evidence; suspect ones replaced by a library model of their class)"
  MEM=16G $RECON calibrated_scene_reconstruction.scene.review.audit "$S" --fix > "$LOG/audit.log" 2>&1; grep "^\[audit\]" "$LOG/audit.log" | grep -v "| ok" | sed 's/^/  /' || true; fi
# inpaint runs with background zone too: the layout builds the support from the photo's plate
if step inpaint; then say inpaint; MEM=12G $RECON calibrated_scene_reconstruction.scene.background.inpaint "$S" > "$LOG/inpaint.log" 2>&1; fi
# configs/<name>/completion.json "source": "captured": the scene as captured, nothing generated -- no panorama, no
# cousins, no wall bays; the twin's room objects come from the photo and the session's other views only
CAPTURED=0; grep -q '"source": *"captured"' "$C/completion.json" 2>/dev/null && { CAPTURED=1; NCOUSINS=0; }
if [ $TWIN = 1 ]; then CAPTURED=1; [ "$BACKGROUND" = completion ] && CAPTURED=0; fi  # twin.json's background decides
if [ $CAPTURED = 1 ] && step panorama; then say "captured scene: no completion (completion.json source)"
elif step panorama; then say "panorama completion of what the photo did not see (${PANO_MODEL:-sdxl})"
  [ -f "$C/completion.json" ] || say "no $C/completion.json: default prompt from the scene graph labels (the agent should write one)"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.background.panorama "$S" --model "${PANO_MODEL:-sdxl}" --apply > "$LOG/panorama.log" 2>&1
  tr '\r' '\n' < "$LOG/panorama.log" | grep -E "^\[pano\] (unseen|.*depth scale|scale constraint)" | sed 's/^/  /' || true
  $RECON calibrated_scene_reconstruction.scene.review.views "$S" --force --only views.json > "$LOG/autoviews_pano.log" 2>&1; fi
if step layout; then say "layout (photo textures)"; rm -f "$S/scene/irradiance_photo.npy"; MEM=16G $RECON calibrated_scene_reconstruction.scene.background.layout "$S" > "$LOG/layout.log" 2>&1; fi
if step irradiance; then say "irradiance probe"; $ISAAC apps/scene/irradiance.py "$S" > "$LOG/irradiance.log" 2>&1; fi
if step albedo; then say "layout (albedo = photo / irradiance)"; MEM=16G $RECON calibrated_scene_reconstruction.scene.background.layout "$S" > "$LOG/albedo.log" 2>&1; fi
if step surfaces; then say "surface review (what the photo bakes into the surfaces the robot works at)"
  MEM=16G $RECON calibrated_scene_reconstruction.scene.support.surfaces "$S" > "$LOG/surfaces.log" 2>&1; grep -h "^\[surfaces\]" "$LOG/surfaces.log" | sed 's/^/  /' || true
  if [ $TWIN = 1 ]; then
    gate S4 "each surface in $S/qa/surfaces/*.jpg and qa/surfaces.json: its look (furniture groups / structure planes, in $C/scene_graph.json); a colour the frames clip from recon.support_marks --look (scene.objects.frame_color)" materials
  elif python3 -c "import json, sys; sys.exit('_surface_review' in json.load(open('$C/scene_graph.json')))"; then
    say "agent: review $S/qa/surfaces/*.jpg and qa/surfaces.json, decide each surface's look (furniture groups / structure planes) and write _surface_review in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from materials"; exit 0; }
  fi; fi
if step materials; then say materials; $RECON calibrated_scene_reconstruction.scene.background.materials "$S" $([ $AUTO = 1 ] && echo --auto-surfaces) > "$LOG/materials.log" 2>&1; grep -v run_capped "$LOG/materials.log" | tail -1; fi
# S2 the task zone against the walk's frames: agent-authored Blender models fitted (dynamic_objects.<obj>.fit), the marks
# on the support and its colour (support_marks), every task object beside its close-ups; the review's gate
if step task_zone; then
  object_route
  if [ ! -f "$S/parsing/posed_frames/cameras.npz" ] && PYTHONPATH=src envs/recon/bin/python -c "import json,sys; sys.exit(not json.load(open('$S/qa/model_requests.json'))['objects'])"; then
    echo "Task-object accuracy iteration requires calibrated posed frames; acquire observations before acceptance" >&2; exit 2
  fi
  if [ -f "$S/parsing/posed_frames/cameras.npz" ]; then
  say "S2 task zone: fits, marks, close-ups"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.objects.object_hull "$S" --from-config --apply > "$LOG/object_hull.log" 2>&1; grep -h "^\[hull\]\|^done" "$LOG/object_hull.log" | sed 's/^/  /' || true
  MEM=24G $RECON calibrated_scene_reconstruction.scene.objects.object_fit "$S" --from-config --apply --object-route "$OBJECT_ROUTE" > "$LOG/object_fit.log" 2>&1; grep -h "^\[object_fit\]\|^done" "$LOG/object_fit.log" | sed 's/^/  /' || true
  MEM=40G $RECON calibrated_scene_reconstruction.scene.objects.blender.pipeline "$S" --object-route "$OBJECT_ROUTE" > "$LOG/objects_blender.log" 2>&1
  # the work zone's structures (scene graph `structures`: a frame, a fixture, a shelf) carved from the frames as meshes
  MEM=24G $RECON calibrated_scene_reconstruction.scene.objects.structure_hull "$S" --apply > "$LOG/structure_hull.log" 2>&1; grep -h "^\[structure\]\|^done" "$LOG/structure_hull.log" | sed 's/^/  /' || true
  # each task object's own check and evidence (its loop: the agent's OBJ.<obj> rounds, docs/twin_agent.md)
  for o in $(PYTHONPATH=src envs/recon/bin/python -c "import json,sys; g=json.load(open('$C/scene_graph.json')); print(' '.join(n for n, v in g.get('dynamic_objects', {}).items() if (v.get('fit') or {}).get('kit')))"); do
    MEM=16G $RECON calibrated_scene_reconstruction.scene.objects.object_refine "$S" run "$o" > "$LOG/object_refine_$o.log" 2>&1; grep -h "^\[object_refine\]" "$LOG/object_refine_$o.log" | cut -c1-220 | sed 's/^/  /' || true; done
  MEM=24G $RECON calibrated_scene_reconstruction.scene.support.support_marks "$S" --from-config --apply > "$LOG/support_marks.log" 2>&1; grep -h "^\[support_marks\]\|^done" "$LOG/support_marks.log" | sed 's/^/  /' || true
  if [ "$TASK_LOOK" = frames ]; then  # the fitted kits textured from the walk's frames (twin.json task_look)
    MEM=16G $RECON calibrated_scene_reconstruction.scene.objects.object_texture "$S" --apply > "$LOG/object_texture.log" 2>&1
    MEM=40G $RECON calibrated_scene_reconstruction.scene.objects.blender.pipeline "$S" --object-route "$OBJECT_ROUTE" --texture > "$LOG/object_texture_blender.log" 2>&1
    grep -h "^\[object_texture\]\|^done" "$LOG/object_texture.log" | sed 's/^/  /' || true; fi
  # the rigs' boxes and the support's look are the scene graph's furniture: into the layout and its materials again
  # (built before this stage, the twin had no frame on its desk)
  MEM=16G $RECON calibrated_scene_reconstruction.scene.background.layout "$S" > "$LOG/albedo.log" 2>&1; tail -n 2 "$LOG/albedo.log" | grep -v run_capped | sed 's/^/  /' || true
  $RECON calibrated_scene_reconstruction.scene.background.materials "$S" $([ $AUTO = 1 ] && echo --auto-surfaces) > "$LOG/materials.log" 2>&1
  # a splat twin renders at a fixed exposure (sim.splat.fixed_exposure): its close-ups too, before the splat exists
  $ISAAC apps/scene/object_closeups.py "$S" $([ $SPLAT = 1 ] && echo --fixed-exposure) $([ $ZONE = 1 ] && echo --background zone) > "$LOG/object_closeups.log" 2>&1; grep -h "^\[closeups\]" "$LOG/object_closeups.log" | sed 's/^/  /' || true
  gate S2 "each task object in $S/qa/object_closeups/ (shape, orientation, size, place, colour, what lies beside it) and $S/qa/support_marks, $S/qa/object_fit; a wrong shape -> revise agent/objects/<obj>/model.py and repeat the Blender fit/evidence/OBJ review; template fitting requires explicit fallback, a mark -> support_marks in $C/scene_graph.json" task_zone; fi; fi
if step task_zone && [ $TWIN = 1 ] && [ $VIDEOCAP = 0 ] && [ ! -f "$S/parsing/posed_frames/cameras.npz" ]; then
  gate S2 "the photo-aligned task objects, support and their evidence ($S/objects, $C/scene_graph.json); review every applicable object loop" task_zone; fi
PANO=0; [ -f "$S/scene/pano/views.npz" ] && PANO=1
audit() {  # automatic audit of the renders of what the photo did not see -> qa/unseen_audit.json
  $RECON calibrated_scene_reconstruction.scene.review.unseen_audit "$S" --tags "$1" > "$LOG/unseen_audit.log" 2>&1
  grep "^\[audit\]" "$LOG/unseen_audit.log" | grep -v ": ok$" | sed 's/^/  /' || true; }
if step views && [ "$BACKGROUND" != blender ]; then say "twin views"; $ISAAC apps/scene/render_views.py "$S" --tag twin --novel $([ $PANO = 1 ] && echo --unseen) > "$LOG/views_twin.log" 2>&1
  [ $PANO = 1 ] && audit twin; fi
if step cousins && [ "$BACKGROUND" != blender ]; then say "cousins x$NCOUSINS (style themes)"; $RECON calibrated_scene_reconstruction.scene.cousins.variants "$S" --n "$NCOUSINS" > "$LOG/cousins.log" 2>&1
  { grep -v run_capped "$LOG/cousins.log" || true; } | sed 's/^/  /' || true  # empty with no cousins (a captured scene)
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/render_views.py "$S" --tag "$t" --variant "$f" > "$LOG/views_$t.log" 2>&1; done; fi
if step cousin_rooms && [ -f "$S/scene/pano/views.npz" ]; then say "cousin rooms: what the photo did not see completed again per cousin (theme style, own seed)"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.cousins.rooms "$S" --model "${PANO_MODEL:-sdxl}" > "$LOG/cousin_rooms.log" 2>&1
  grep "^\[cousin_rooms\]" "$LOG/cousin_rooms.log" | sed 's/^/  /' || true; fi  # none with no cousins
reviewed_all() {  # every tag (twin, each cousin) has an entry under scene graph key $1 with "ok": true
  python3 -c "
import json, glob, os, sys
sg = json.load(open('$C/scene_graph.json')); r = sg.get('$1', {})
tags = ['twin'] + sorted(os.path.basename(f)[:-5] for f in glob.glob('$S/cousins/cousin_*.json'))
miss = [t for t in tags if not (r.get(t) or {}).get('ok')]
print(' '.join(miss)); sys.exit(1 if miss else 0)"; }
if step walls && [ $CAPTURED = 0 ]; then say "wall completion (every free wall of the twin and of each cousin, furnished straight on)"
  $ISAAC apps/scene/wall_views.py "$S" --tag twin > "$LOG/wall_views_twin.log" 2>&1
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/wall_views.py "$S" --tag "$t" --variant "$f" > "$LOG/wall_views_$t.log" 2>&1; done
  grep -h "bays ->" "$LOG"/wall_views_*.log | sed 's/^/  /' || true
  MEM=30G $RECON calibrated_scene_reconstruction.scene.background.wall_completion "$S" --model "${PANO_MODEL:-sdxl}" > "$LOG/walls.log" 2>&1
  echo "  $(grep -c '^\[walls\]' "$LOG/walls.log") bays painted"; fi
if step completion_review && [ -f "$S/scene/pano/views.npz" ]; then say "completion review (every completed view of the twin and of every cousin)"
  $RECON calibrated_scene_reconstruction.scene.review.completion_review "$S" > "$LOG/completion_review.log" 2>&1; grep "^\[completion\]" "$LOG/completion_review.log" | sed 's/^/  /' || true
  if ! miss=$(reviewed_all _completion_review); then
    say "agent: review $S/qa/completion/<tag>.jpg and $S/qa/walls/<tag>.jpg for $miss -- a plausible, realistic room of this kind, consistent with the photos (no gibberish text, nothing glued to a wall or floating, no painted floor stripes, no product renders); re-roll / re-prompt a rejected view in $C/completion.json \"tags\" (a wall bay: \"walls\" / \"wall_items\"), rerun from panorama / cousin_rooms / walls; write _completion_review {tag: {ok, views, note}} in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from room_objects"; exit 0; }
  fi
  gate S3 "every completed view in $S/qa/completion/twin.jpg and $S/qa/walls/twin.jpg against the walk's frames (what the room is, what stands where the frames saw it)" room_objects; fi
if step room_objects && [ $ZONE = 0 ] && { [ -f "$S/scene/pano/views.npz" ] || [ $CAPTURED = 1 ]; }; then say "room objects (what each completion shows, reconstructed in 3D: SAM 3D, library only where it fails)"
  MEM=30G $RECON calibrated_scene_reconstruction.scene.background.room_objects "$S" --apply > "$LOG/room_objects.log" 2>&1
  grep -h "^\[room_objects\] [a-z_0-9]*: [0-9]* pieces" "$LOG/room_objects.log" | sed 's/^/  /' || true
  $RECON calibrated_scene_reconstruction.scene.background.materials "$S" $([ $AUTO = 1 ] && echo --auto-surfaces) > "$LOG/materials.log" 2>&1; fi
if step clearance; then say "clearance QA (penetration + physics settle)"
  $ISAAC apps/scene/check_clearance.py "$S" --tag twin > "$LOG/clearance_twin.log" 2>&1; grep -h "^\[twin\]" "$LOG/clearance_twin.log" | sed 's/^/  /' || true
  # the task's end state set up and let go (a twin: task.json): do its objects support the task
  if [ $TWIN = 1 ]; then
    rm -f "$S/qa/task_check.json"  # a launcher/startup failure must not preserve an earlier passing result
    $ISAAC apps/scene/task_check.py "$S" $([ $ZONE = 1 ] && echo "--background zone") > "$LOG/task_check.log" 2>&1
    test -s "$S/qa/task_check.json"  # Isaac shutdown can return zero after a Python exception; require a fresh result
    grep -h "^\[task_check\]" "$LOG/task_check.log" | sed 's/^/  /' || true; fi
  if [ -f "$C/code_scene.json" ]; then  # the room written as code: its pieces checked (geometry) and settled (PhysX)
    $RECON calibrated_scene_reconstruction.scene.background.code_room "$S" check > "$LOG/code_check.log" 2>&1; grep -h "^done" "$LOG/code_check.log" | sed 's/^/  code room: /' || true
    $ISAAC apps/scene/check_clearance.py "$S" --tag code --background code > "$LOG/clearance_code.log" 2>&1
    grep -h "^\[code\]" "$LOG/clearance_code.log" | sed 's/^/  /' || true; fi
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/check_clearance.py "$S" --tag "$t" --variant "$f" > "$LOG/clearance_$t.log" 2>&1; grep -h "^\[$t\]" "$LOG/clearance_$t.log" | sed 's/^/  /' || true; done; fi
if step object_review; then say "object review (set pieces measured in the photo, every object and surface checked)"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.objects.setpiece_observations "$S" > "$LOG/setpieces.log" 2>&1; grep -h "^\[setpieces\]" "$LOG/setpieces.log" | sed 's/^/  /' || true
  $ISAAC apps/scene/review_objects.py "$S" --tag twin > "$LOG/objects_twin.log" 2>&1; grep -h "^\[objects\] twin:" "$LOG/objects_twin.log" | sed 's/^/  /' || true
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/review_objects.py "$S" --tag "$t" --variant "$f" > "$LOG/objects_$t.log" 2>&1
    grep -h "^\[objects\] $t:" "$LOG/objects_$t.log" | sed 's/^/  /' || true; done
  if [ -f "$S/parsing/posed_frames/cameras.npz" ]; then  # walk frames posed: each object beside its close-ups (a twin: every item)
    $ISAAC apps/scene/object_closeups.py "$S" $([ $TWIN = 1 ] && [ $ZONE = 0 ] && echo --all) $([ $ZONE = 1 ] && echo --background zone) > "$LOG/object_closeups.log" 2>&1; grep -h "^\[closeups\]" "$LOG/object_closeups.log" | grep -c . | sed 's/^/  close-ups of /;s/$/ items/'; fi
  if [ $TWIN = 1 ]; then
    gate S5 "every item in $S/qa/objects/twin/ and $S/qa/object_closeups/ (class, shape, upright, on its support, place, size; duplicates, merged pieces, missing things) and $S/qa/clearance_twin.json; fixes in $C/scene_graph.json (docs/object_review.md)" clearance
  elif python3 -c "import json, sys; sys.exit('_object_review' in json.load(open('$C/scene_graph.json')))"; then
    say "agent: review $S/qa/objects/<tag>/ (sheet_*.jpg, surfaces_*.jpg, overview.jpg; qa/objects_<tag>.json) against the photo and $S/qa/object_closeups/ (each task object's shape beside the walk's frames: recon.object_fit, recon.support_marks), write fixes and an _object_review note in $C/scene_graph.json (docs/object_review.md)"
    [ $AUTO = 1 ] || { say "then rerun with --from clearance"; exit 0; }
  fi; fi
if step placements; then for R in $ROBOTS; do
  if [ "$ROBOT_EXECUTION" = 1 ]; then say "optional robot execution placements ($R)"
    $ISAAC apps/placements.py "$S" --robot "$R" > "$LOG/placements_$R.log" 2>&1
  else say "presentation pose ($R; object-model task validation)"
    $RECON calibrated_scene_reconstruction.scene.capture.presentation_robot "$S" --robot "$R" > "$LOG/placements_$R.log" 2>&1
  fi; done; fi
if step robot_views; then R="${ROBOTS%% *}"; say "robot views ($R: head camera and over the shoulder at every accepted base)"
  $ISAAC apps/robot_views.py "$S" --robot "$R" --tag twin $([ $ZONE = 1 ] && echo --background zone) > "$LOG/robot_views_twin.log" 2>&1
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/robot_views.py "$S" --robot "$R" --tag "$t" --variant "$f" > "$LOG/robot_views_$t.log" 2>&1; done
  grep -h "^\[robot_views\].*bases" "$LOG"/robot_views_*.log | sed 's/^/  /' || true
  if python3 -c "import json, sys; sys.exit('_robot_view_review' in json.load(open('$C/scene_graph.json')))"; then
    say "agent: review $S/qa/robot_views/*_sheet.jpg as the policy will see them (baked light or shadows, smeared or flat surfaces, seams, a theme that does not fit the room), fix and rerun from the stage at fault, then write _robot_view_review in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from room_views"; exit 0; }
  fi; fi
if step room_views && [ $ZONE = 0 ]; then say "room views (the whole room of the twin and of every cousin, from the robot and the room's middle)"
  $ISAAC apps/scene/room_views.py "$S" --robot "${ROBOTS%% *}" --tag twin > "$LOG/room_views_twin.log" 2>&1
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/room_views.py "$S" --robot "${ROBOTS%% *}" --tag "$t" --variant "$f" > "$LOG/room_views_$t.log" 2>&1; done
  if ! miss=$(reviewed_all _room_review); then
    say "agent: review $S/qa/room_views/<tag>.jpg and qa/room_objects/<tag>.jpg for $miss -- every object solid and believable, nothing painted on a wall, no blurred / stretched texture, seam or hole; fix at the stage at fault and write _room_review {tag: {ok, note}} in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from splat_capture"; exit 0; }  # the next stage (from fly skipped delivery)
  fi; fi
if step room_views && [ $TWIN = 1 ] && [ "$BACKGROUND" = mesh ]; then
  gate S3 "the captured mesh background in $S/qa/room_views and its correspondence to the capture" room_views; fi
# the optional Gaussian-splat background: off unless --splat or configs/<name>/splat.json; the stages keep --from / --to
# working either way (step is asked first)
[ $TWIN = 0 ] && [ -f "$C/splat.json" ] && SPLAT=1  # (a twin: twin.json's background decides; splat.json then only sets the frames)
if [ $SPLAT = 1 ] && [ ! -f "$C/splat.json" ]; then echo '{}' > "$C/splat.json"; fi
if step splat_capture && [ $SPLAT = 1 ] && [ $TWIN = 0 ]; then say "splat: the frames of the session's videos, COLMAP poses, the twin's world, coverage"
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.capture "$S" > "$LOG/splat_capture.log" 2>&1  # (a twin: the poses stage did it)
  grep -h "^\[colmap\] registered\|^\[align\]\|^\[board\]\|^\[coverage\]\|^\[zone\]" "$LOG/splat_capture.log" | sed 's/^/  /' || true; fi
if step splat_masks && [ $SPLAT = 1 ]; then say "splat: the task zone's things out of every frame (SAM 3 in the zone's projection)"
  MEM=24G $RECON calibrated_scene_reconstruction.scene.background.splat.masks "$S" > "$LOG/splat_masks.log" 2>&1; grep -h "^done" "$LOG/splat_masks.log" | sed 's/^/  /' || true; fi
if step splat_train && [ $SPLAT = 1 ]; then say "splat: 3DGRUT training (3DGUT, MCMC), held-out frames scored"
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.train "$S" > "$LOG/splat_train.log" 2>&1; grep -h "^done" "$LOG/splat_train.log" | sed 's/^/  /' || true; fi
# the splat sharpened where the walk did not look from (splat.json "fix": true; setup/setup_difix_env.sh): novel views
# round the task zone, Difix, trained back in (recon.splat_fix)
if step splat_fix && [ $SPLAT = 1 ] && grep -q '"fix": *true' "$C/splat.json"; then say "splat: novel views fixed by Difix, trained back in"
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.fix "$S" prepare > "$LOG/splat_fix.log" 2>&1
  PYTHONPATH=src MEM=40G tools/run_capped.sh envs/difix/bin/python -m calibrated_scene_reconstruction.scene.background.splat.fix "$S" fix >> "$LOG/splat_fix.log" 2>&1
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.fix "$S" train >> "$LOG/splat_fix.log" 2>&1; grep -h "^done" "$LOG/splat_fix.log" | sed 's/^/  /' || true; fi
if step splat_export && [ $SPLAT = 1 ]; then say "splat: the task zone, unseen, faint and huge Gaussians out, NuRec USDZ"
  MEM=40G $SPL calibrated_scene_reconstruction.scene.background.splat.export "$S" > "$LOG/splat_export.log" 2>&1; grep -h "^done" "$LOG/splat_export.log" | sed 's/^/  /' || true; fi
if step splat_review && [ $SPLAT = 1 ]; then say "splat: review renders with the background (scene-level stills; the robot's views)"
  $ISAAC apps/scene/scene_tour.py "$S" --robot "${ROBOTS%% *}" --background splat --frames-only 16 > "$LOG/splat_stills.log" 2>&1
  $ISAAC apps/robot_views.py "$S" --robot "${ROBOTS%% *}" --tag twin --background splat > "$LOG/robot_views_splat.log" 2>&1
  if python3 -c "import json, sys; sys.exit('_splat_review' in json.load(open('$C/scene_graph.json')))"; then
    say "agent: review $S/qa/splat/ (held_out.jpg, coverage.png), $S/renders/tour/splat/stills and $S/qa/robot_views/twin_sheet_splat.jpg as the scene, not as a camera's view -- needles, holes, a seam to the meshes, doubled pieces, colour; fix at the stage at fault (splat.json) and write _splat_review in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from deliver"; exit 0; }  # the next stage (from fly skipped delivery)
  fi; fi
if step splat_review && [ $TWIN = 1 ] && [ "$BACKGROUND" = splat ]; then
  gate S3 "the held-out capture comparisons, coverage and splat renders ($S/qa/splat, $S/renders/tour/splat/stills)" splat_review; fi
if step environment_build && [ "$BACKGROUND" = blender ]; then
  say 'Agent Blender environment: native USD, full-scene captured and novel view review'
  bash pipeline/environment.sh "$NAME" --robot "${ROBOTS%% *}" > "$LOG/environment.log" 2>&1
  gate S3 "the current whole-scene environment review: materials, lighting, shadows, occlusion, layout, detail and view consistency" environment_build
fi
# S6 delivery (a twin): the tour, the USD export, the single USDZ opened alone in Isaac Sim
if step deliver && [ $TWIN = 1 ]; then object_route --require-reviewed; R="${ROBOTS%% *}"; say "S6 delivery: tour, USD, single USDZ"
  if [ "$ROBOT_EXECUTION" = 0 ]; then
    $RECON calibrated_scene_reconstruction.scene.capture.presentation_robot "$S" --robot "$R" > "$LOG/presentation_$R.log" 2>&1
  fi
  BG=$([ $SPLAT = 1 ] && echo "--background splat" || true); SUB=$([ $SPLAT = 1 ] && echo twin_splat || echo twin)
  [ $ZONE = 1 ] && { BG="--background zone"; SUB=twin_zone; }
  [ "$BACKGROUND" = blender ] && { BG="--background blender"; SUB=twin_blender; }
  # Start with the captured camera path for splat and zone scenes, then show the task objects close up.
  $ISAAC apps/scene/scene_tour.py "$S" --robot "$R" --seconds 90 $BG $([ $SPLAT = 1 ] || [ $ZONE = 1 ] && echo "--path capture_free --fov 60") > "$LOG/tour.log" 2>&1
  grep -h "^saved\|^free view" "$LOG/tour.log" | sed 's/^/  /' || true
  $ISAAC apps/scene/export_usd.py "$S" --robot "$R" $BG > "$LOG/export_usd.log" 2>&1; grep -h "^saved" "$LOG/export_usd.log" | cut -c1-160 | sed 's/^/  /' || true
  envs/splat/3dgrut/.venv/bin/python apps/scene/pack_usdz.py "$S/export/usd/$SUB" > "$LOG/pack_usdz.log" 2>&1; grep -h "^done" "$LOG/pack_usdz.log" | sed 's/^/  /' || true
  if [ "$BACKGROUND" = blender ]; then
    $RECON calibrated_scene_reconstruction.scene.background.environment_physics "$S" "$S/export/usd/${NAME}_${SUB}.usdz" \
      --delivery > "$LOG/environment_delivery_physics.log" 2>&1
  fi
  U=$(ls "$S/export/usd/"*"_$SUB.usdz" 2>/dev/null | head -1)
  if [ -n "$U" ]; then
    $ISAAC apps/scene/check_package_physics.py "$S" "$U" > "$LOG/package_physics.log" 2>&1
  fi
  [ -n "$U" ] && $ISAAC apps/scene/view_usd.py "$S" "$U" > "$LOG/view_usdz.log" 2>&1 && grep -h "^done" "$LOG/view_usdz.log" | sed 's/^/  /' || true
  gate S6 "the tour, the USDZ's previews ($S/export/usd/*_preview_*.jpg) and the report (options, scale and its check, task objects' sizes and places, limitations, inferences)" deliver; fi
if step fly && [ $VIDEO = 1 ]; then say "fly-through (twin and every cousin, ending at the robot's view)"
  $ISAAC apps/scene/flythrough.py "$S" --robot "${ROBOTS%% *}" > "$LOG/fly.log" 2>&1
  for f in "$S"/cousins/cousin_*.json; do [ -f "$f" ] || continue; t=$(basename "$f" .json)
    $ISAAC apps/scene/flythrough.py "$S" --robot "${ROBOTS%% *}" --tag "$t" --variant "$f" > "$LOG/fly_$t.log" 2>&1; done; fi
placed() {  # an accepted base in placements/<robot>.json (Vega: placements.json)
  f="$S/placements/$1.json"; [ "$1" = vega_1p_sharpa ] && f="$S/placements/placements.json"
  [ -f "$f" ] && python3 -c "import json,sys; sys.exit(not any(c['accepted'] for c in json.load(open('$f'))['candidates']))"; }
EPTAGS="twin $({ ls "$S"/cousins/cousin_*.json 2>/dev/null || true; } | head -n "$EPC" | xargs -rn1 basename | sed 's/\.json$//' | tr '\n' ' ')"
if step grasps && [ "$ROBOT_EXECUTION" = 1 ]; then say "scene environment around the target (walls, furniture, objects)"
  $ISAAC apps/scene_obstacles.py "$S" --tags $EPTAGS > "$LOG/scene_obstacles.log" 2>&1
  grep -h "^\[" "$LOG/scene_obstacles.log" | sed 's/^/  /' || true
  for R in $ROBOTS; do placed "$R" || continue
  if grep -q '"bodex"' "configs/robots/$R.json"; then say "grasps ($R: BODex synthesis, environment-aware)"
    PYTHONPATH=src MEM=16G tools/run_capped.sh envs/grasp/bin/python -m calibrated_scene_reconstruction.grasping.synth "$S" --robot "$R" \
      --variants $EPTAGS > "$LOG/grasps_$R.log" 2>&1; grep -h "^\[" "$LOG/grasps_$R.log" | grep -v anchored_bodex | sed 's/^/  /' || true; fi
  done; fi
PLACE=0; [ "${TASK:-$(python3 -c "import json; print((json.load(open('$C/scene_graph.json')).get('task') or {}).get('type') or 'pick')")}" = place ] && PLACE=1
if step receptacles && [ $PLACE = 1 ] && [ "$ROBOT_EXECUTION" = 1 ]; then say "place targets: sampled per kind, checked and rendered (${ROBOTS%% *})"
  $ISAAC apps/review_receptacles.py "$S" --robot "${ROBOTS%% *}" > "$LOG/review_receptacles.log" 2>&1
  grep -h "^\[receptacles\]" "$LOG/review_receptacles.log" | sed 's/^/  /' || true
  if python3 -c "import json, sys; sys.exit('_review' in (json.load(open('$C/scene_graph.json'))['task'].get('place') or {}))"; then
    say "agent: review $S/qa/receptacles/sheet.jpg (and qa/receptacles/<kind>.jpg, qa/receptacles.json), write task.place (room, receptacles weights, keep_out, _review) in $C/scene_graph.json"
    [ $AUTO = 1 ] || { say "then rerun with --from episodes"; exit 0; }
  fi; fi
if step demo && [ -n "$DEMO" ]; then say "demo re-targeting ($DEMO -> $DEMO_N plans)"
  MEM=16G $RECON calibrated_scene_reconstruction.demo.generate "$S" --demo "$DEMO" $DEMO_CFG --n "$DEMO_N" > "$LOG/demo.log" 2>&1
  tail -1 "$LOG/demo.log" | sed 's/^/  /' || true; fi  # TODO: Isaac Sim playback + PhysX validation of the plans (apps/datagen/episode.py)
if step episodes && [ "$ROBOT_EXECUTION" = 1 ]; then for R in $ROBOTS; do say "episodes ($R)"
  if ! placed "$R"; then echo "  no accepted placement for $R: skipped"; continue; fi
  clear_episodes "$R"
  for f in twin $(ls "$S"/cousins/cousin_*.json 2>/dev/null | head -n "$EPC"); do
    if [ "$f" = twin ]; then t=twin; v=""; else t=$(basename "$f" .json); v="--variant $f"; fi
    $ISAAC apps/datagen/episode.py "$S" --tag "$t" $v --robot "$R" --episodes "$EPS" --batch "$BATCH" ${TASK:+--task-type "$TASK"} > "$LOG/episode_$(tagged "$R" "$t").log" 2>&1 || true
    grep -hE "object lift|^recorded robot" "$LOG/episode_$(tagged "$R" "$t").log" | sed "s/^/  $t: /"; done; done; fi
if step export && [ "$ROBOT_EXECUTION" = 1 ]; then say "export (LeRobot v2.1)"; for R in $ROBOTS; do
  MEM=16G $RECON calibrated_scene_reconstruction.export.lerobot "outputs/lerobot/$NAME${TASK:+/$TASK}/$R" "$S" --robot "$R" \
    ${TASK:+--task-type "$TASK"} > "$LOG/export_$R.log" 2>&1 \
    && tail -1 "$LOG/export_$R.log" | sed 's/^/  /' || true || echo "  $R: nothing to export"; done; fi
if step video && [ $VIDEO = 1 ]; then say video; mkdir -p outputs; MEM=16G $RECON calibrated_scene_reconstruction.video "$S" "outputs/$NAME.mp4" > "$LOG/video.log" 2>&1 || true; say "-> outputs/$NAME.mp4"; fi
