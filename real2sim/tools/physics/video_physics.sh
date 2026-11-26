#!/usr/bin/env bash
# One video recording (real/<name>/ with its video and meta.json) through the whole of docs/video_physics.md:
# perception, track, the object's mesh, the fit, the PhysX calibration, and the pictures of the result
# (meta.json "motion": slide, roll, drop, incline, pendulum, turntable). An object on a support that is not level or
# moves, or on a pivot, is a primitive in its twin and gets no mesh (its track is in the support's own frame, which
# the mesh stage does not keep); MESH=0 leaves the mesh out for any other. A body that deforms ("motion": beam,
# strain, sheet) has its own perception and fit (vidphys.soft, and vidphys.beam | strain | sheet); a beam its own
# PhysX replay (apps/physics/soft.py). "carried" and "stickslip" are read from the track or the masks alone.
# Usage: tools/video_physics.sh real/<name> [...]        (STAGES="perceive track mesh fit sim view" picks stages)
set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "usage: tools/physics/video_physics.sh <recording> [options]"
  exit 0
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY=envs/recon/bin/python
STAGES="${STAGES:-perceive track mesh fit sim view}"
OBJECT_ROUTE="${OBJECT_ROUTE:-agent_blender}"
case "$OBJECT_ROUTE" in agent_blender|template|sam3d) ;; *) echo "Unknown OBJECT_ROUTE: $OBJECT_ROUTE" >&2; exit 2 ;; esac
has() { [[ " $STAGES " == *" $1 "* ]]; }
one() {
  local rec="$1"
  echo "== $rec"
  # meta.json as the agent wrote it: a key or a motion that would be read as if it were not there is said, not run
  local said; if ! said=$($PY -m calibrated_scene_reconstruction.physics.video.meta "$rec" 2>&1); then echo "$said"; return 1; fi
  for soft in strain sheet; do  # bodies that deform, measured without a PhysX twin
    if grep -q "\"motion\": *\"$soft\"" "$rec/meta.json"; then
      if has perceive; then MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.soft "$rec" 2>&1 | grep -E "^real/|FAILED"; fi
      if has fit; then $PY -m calibrated_scene_reconstruction.physics.video.$soft "$rec" 2>/dev/null | grep "^real/"; fi
      if has view; then $PY -m calibrated_scene_reconstruction.physics.video.$soft --summary "$rec" 2>/dev/null | tail -1; fi
      return
    fi
  done
  if grep -q '"motion": *"beam"' "$rec/meta.json"; then
    if has perceive; then MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.soft "$rec" 2>&1 | grep -E "^real/|FAILED"; fi
    if has fit; then $PY -m calibrated_scene_reconstruction.physics.video.beam "$rec" 2>&1 | grep "^real/"; fi
    if has sim; then tools/run_isaac.sh apps/physics/soft.py "$rec" --render 2>&1 | grep -E "^\[replay\]|^\[render\]|Traceback|^[A-Za-z]*Error"; fi
    if has view; then $PY -m calibrated_scene_reconstruction.physics.video.beam --summary "$rec" 2>&1 | tail -1; fi
    return
  fi
  if grep -q '"motion": *"stickslip"' "$rec/meta.json"; then  # read off the body's own line: no track, no mesh, no replay
    if has perceive; then MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.perceive "$rec" ${PERCEIVE_ARGS:-} 2>&1 | grep -E "^real/|FAILED"; fi
    if has fit; then $PY -m calibrated_scene_reconstruction.physics.video.stickslip "$rec" 2>&1 | grep "^real/"; fi
    if has view; then $PY -m calibrated_scene_reconstruction.physics.video.stickslip --summary "$rec" 2>/dev/null | tail -1; fi
    return
  fi
  if grep -q '"motion": *"carried"' "$rec/meta.json"; then  # a bound from the track alone: no mesh, no replay
    if has perceive; then MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.perceive "$rec" ${PERCEIVE_ARGS:-} 2>&1 | grep -E "^real/|FAILED"; fi
    if has track; then $PY -m calibrated_scene_reconstruction.physics.video.track "$rec" 2>&1 | grep "^real/"; fi
    if has fit; then $PY -m calibrated_scene_reconstruction.physics.video.fit "$rec" 2>&1 | grep "^real/"; fi
    if has view; then $PY -m calibrated_scene_reconstruction.physics.video.visualize "$rec" 2>&1 | tail -1; fi
    return
  fi
  if has perceive; then MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.perceive "$rec" ${PERCEIVE_ARGS:-} 2>&1 | grep -E "^real/|FAILED"; fi
  if has track; then $PY -m calibrated_scene_reconstruction.physics.video.track "$rec" 2>&1 | grep "^real/"; fi
  MESH_HERE="${MESH:-1}"
  if grep -Eq '"support_moves": *true|"motion": *"(incline|turntable|pendulum)"' "$rec/meta.json"; then MESH_HERE=0; fi
  if [ "$OBJECT_ROUTE" = agent_blender ]; then
    if [ "$MESH_HERE" = 0 ]; then
      echo "Agent Blender geometry is required by default; this motion needs object-relative calibrated observations. Explicit OBJECT_ROUTE=template selects primitive fallback." >&2
      return 2
    fi
    if ! has mesh && has sim; then
      $PY -m calibrated_scene_reconstruction.agent.physics_models status "$rec" > /dev/null || return 2
    fi
  fi
  # a drop's fit sets the scene's scale by gravity: before the mesh is stood in the scene (and again after, as for all)
  if has mesh && [ "$MESH_HERE" != 0 ] && has fit && grep -q '"motion": *"drop"' "$rec/meta.json"; then $PY -m calibrated_scene_reconstruction.physics.video.fit "$rec" >/dev/null 2>&1; fi
  if has mesh && [ "$MESH_HERE" != 0 ] && [ "$OBJECT_ROUTE" != template ]; then
    MEM=40G tools/run_capped.sh $PY -m calibrated_scene_reconstruction.physics.video.mesh "$rec" --object-route "$OBJECT_ROUTE" ${MESH_ARGS:-} 2>&1 | grep -E "^real/|FAILED" || return 2
  fi
  if has fit; then $PY -m calibrated_scene_reconstruction.physics.video.fit "$rec" 2>&1 | grep "^real/"; fi
  if has sim; then
    if grep -q '"motion": *"pendulum"' "$rec/meta.json"; then  # replayed, not fitted; the first 40 s are drawn
      tools/run_isaac.sh apps/physics/video.py "$rec" --object-route "$OBJECT_ROUTE" 2>&1 | grep -E "^\[replay\]|Traceback|^[A-Za-z]*Error|no free phase"
      tools/run_isaac.sh apps/physics/video.py "$rec" --object-route "$OBJECT_ROUTE" --render --seconds 40 2>&1 | grep -E "^\[render\]|Traceback|^[A-Za-z]*Error"
    else
      tools/run_isaac.sh apps/physics/video.py "$rec" --object-route "$OBJECT_ROUTE" --calibrate 2>&1 | grep -E "^\[calibrate\] (loss|the twin)|Traceback|^[A-Za-z]*Error|: motion|no free phase|nothing to replay"
      tools/run_isaac.sh apps/physics/video.py "$rec" --object-route "$OBJECT_ROUTE" --render 2>&1 | grep -E "^\[render\]|Traceback|^[A-Za-z]*Error"
    fi
  fi
  if has view; then $PY -m calibrated_scene_reconstruction.physics.video.visualize "$rec" 2>&1 | tail -1; fi
}
failed=0
for rec in "$@"; do
  # then what was made of what the agent wrote, for the agent to read before it writes again
  if one "$rec"; then
    if has fit || has view; then $PY -m calibrated_scene_reconstruction.physics.video.meta --after "$rec" 2>/dev/null | tail -n +2; fi
  else failed=1; fi
done
exit "$failed"
