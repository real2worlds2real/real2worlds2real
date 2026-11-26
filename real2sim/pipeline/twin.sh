#!/usr/bin/env bash
# The twin's default pass (docs/twin_agent.md): the session's stages straight through, then the check report the
# agent's loop starts from.
#
# Usage: pipeline/twin.sh <name> [video.mp4 | photo.jpg] [--views "a.jpg b.jpg"] [--robots R] [--from STAGE] [--to STAGE]
#   configs/<name>/twin.json (background, scale, scale_check, task_look) and configs/<name>/task.json (the task, its
#   objects, the support) are written first (skills/twin-capture/SKILL.md).
#   Phase 1 (no configs/<name>/scene_graph.json yet): up to the scene graph's draft, then it stops -- the agent reviews
#   the draft against task.json and the photo (which instances are the task's objects and support) and saves
#   scene_graph.json.
#   Phase 2: every later stage straight through (run_session.sh --auto, no gate stopping it) up to `deliver`, once
#   more from the graph when the world turns to the support, and qa/report.md.
set -euo pipefail
# Handle help before parsing a session name, loading configuration or creating output directories.
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: pipeline/twin.sh <name> [video.mp4 | photo.jpg] [--views "a.jpg b.jpg"] [--robots R] [--from STAGE] [--to STAGE]
Build and review a twin using configs/<name>/twin.json and task.json.
Task objects default to agent Blender iteration; --object-route template selects the fallback explicitly.
Optional: --reuse-camera-model reuses existing COLMAP inputs; --pose-workers N sets CPU alignment workers.
HELP
      exit 0
      ;;
  esac
done
[ "$#" -gt 0 ] || { echo "A session name is required; use --help" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NAME="$1"; shift
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.capture.inputs --check-inputs "$NAME"
ARGS=(); TO=deliver; ROBOTS="vega_1p_sharpa_arm"; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --to) TO="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --robots) ROBOTS="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ -f "configs/$NAME/twin.json" ] || { echo "configs/$NAME/twin.json first (skills/twin-capture/SKILL.md)"; exit 1; }
mkdir -p "sessions/$NAME"
OUT="sessions/$NAME/twin_pass.log"
if [ ! -f "configs/$NAME/scene_graph.json" ]; then
  pipeline/run_session.sh "$NAME" "${ARGS[@]}" ${FROM:+--from "$FROM"} --robots "$ROBOTS" --to graph 2>&1 | tee "$OUT"
  echo "review configs/$NAME/scene_graph.auto.json (docs/twin_agent.md), save it as scene_graph.json, then: pipeline/twin.sh $NAME ${ARGS[*]} --from poses"
  exit 0
fi
pipeline/run_session.sh "$NAME" "${ARGS[@]}" ${FROM:+--from "$FROM"} --auto --robots "$ROBOTS" --to "$TO" 2>&1 | tee -a "$OUT"
# the support askew in the room (background zone): the world was turned to it -- the graph again, reviewed again
if tail -3 "$OUT" | grep -q "the world turned to the support's edges"; then
  pipeline/run_session.sh "$NAME" "${ARGS[@]}" --from graph --to graph --robots "$ROBOTS" 2>&1 | tee -a "$OUT"
  echo "the world turned to the support: review the redrafted configs/$NAME/scene_graph.auto.json and save it as scene_graph.json (the earlier review: scene_graph.prev.json), then: pipeline/twin.sh $NAME ${ARGS[*]} --from poses"
  exit 0
fi
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.twin.report "sessions/$NAME"
