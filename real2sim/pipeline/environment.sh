#!/usr/bin/env bash
# Agent Blender environment build -> actual full-scene export -> fixed-camera review.
set -euo pipefail
for arg in "$@"; do
  case "$arg" in -h|--help)
    cat <<'HELP'
Usage: pipeline/environment.sh SESSION [--variant twin|NAME] [--cameras MANIFEST]
       [--robot NAME | --no-robot] [--reuse-build] [--review-request JSON]
       [--composition-observations JSON] (repeat for additional instance masks)
Requires configs/<SESSION>/environment.json and agent/environment/model.py.
Builds an unapproved native USD environment without changing task geometry.
Each variant needs its own full-scene photorealism review; no robot operation.
HELP
    exit 0 ;;
  esac
done
[ "$#" -gt 0 ] || { echo 'Session required; use --help' >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NAME="${1#sessions/}"; shift
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Simple session name required' >&2; exit 2; }
VARIANT=twin; CAMERAS=""; REUSE=0; REVIEW=""; ROBOT_ARGS=(); COMPOSITION_OBS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --cameras) CAMERAS="$2"; shift 2 ;;
    --robot) ROBOT_ARGS=(--robot "$2"); shift 2 ;;
    --no-robot) ROBOT_ARGS=(--no-robot); shift ;;
    --reuse-build) REUSE=1; shift ;;
    --review-request) REVIEW="$2"; shift 2 ;;
    --composition-observations) COMPOSITION_OBS+=("$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$VARIANT" =~ ^[A-Za-z][A-Za-z0-9_-]{0,63}$ ]] || { echo 'Simple variant identifier required' >&2; exit 2; }
S="sessions/$NAME"
[ ! -f "$S/qa/independent_freeze.json" ] || { echo 'Frozen session: create an environment derivative' >&2; exit 2; }
export PYTHONPATH=src
P=envs/recon/bin/python
PYTHONPATH=src "$P" -m calibrated_scene_reconstruction.scene.capture.inputs --check-inputs "$NAME"
LOG="$S/logs/environment_$VARIANT"; mkdir -p "$LOG"
if [ "$REUSE" = 1 ]; then
  "$P" -m calibrated_scene_reconstruction.scene.background.blender_env "$S" verify --variant "$VARIANT" > "$LOG/verify.log" 2>&1
else
  "$P" -m calibrated_scene_reconstruction.scene.background.blender_env "$S" build --variant "$VARIANT" > "$LOG/build.log" 2>&1
fi
if [ -z "$CAMERAS" ]; then
  CAMERAS="$S/qa/environment_cameras.json"
  [ -f "$CAMERAS" ] || "$P" -m calibrated_scene_reconstruction.scene.review.cameras "$S" --output qa/environment_cameras.json
fi
OUT=$($P - "$S" "$VARIANT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])/'qa/environment_renders'/sys.argv[2]
rounds=[int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
print(root/f'round_{max(rounds,default=0)+1}')
PY
)
TAG="environment_${VARIANT}_review_${OUT##*/}"
tools/run_isaac.sh apps/scene/export_usd.py "$S" --background blender --environment-variant "$VARIANT" \
  --tag "$TAG" "${ROBOT_ARGS[@]}" > "$LOG/export.log" 2>&1
# Kit can return zero after a Python exception. Require a completed export,
# rather than continuing with a missing or partial scene.
"$P" - "$S/export/usd/${TAG}_blender" "$NAME" "${TAG}_blender" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);name,tag=sys.argv[2:]
if not (out/f'{name}_{tag}.usd').is_file() or not (out/'export.json').is_file():
    raise SystemExit('Environment export incomplete; inspect export.log')
meta=json.loads((out/'export.json').read_text())
if meta.get('session')!=name or meta.get('tag')!=tag or meta.get('background')!='blender':
    raise SystemExit('Environment export metadata does not match this candidate')
PY
"$P" -m calibrated_scene_reconstruction.scene.background.environment_physics "$S" \
  "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" --variant "$VARIANT" > "$LOG/physics.log" 2>&1
"$P" -m calibrated_scene_reconstruction.scene.background.completion "$S" \
  "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" --variant "$VARIANT" > "$LOG/completion.log" 2>&1
envs/splat/3dgrut/.venv/bin/python apps/scene/pack_usdz.py "$S/export/usd/${TAG}_blender" > "$LOG/pack.log" 2>&1
tools/run_isaac.sh apps/scene/review_usd_cameras.py \
  "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" --renderer "$("$P" - "$S" "$VARIANT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'scene/environment'/sys.argv[2]/'current.json'
print(json.loads(Path(json.loads(p.read_text())['path']).read_text())['render']['mode'] if p.exists() else 'pt')
PY
)" --output "$OUT" > "$LOG/review.log" 2>&1
"$P" -m calibrated_scene_reconstruction.scene.review.frozen_scope_cache "$S" \
  "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" \
  --variant "$VARIANT" --output "$OUT/frozen_scope" > "$LOG/frozen_scope.log" 2>&1
CONTEXT_MANIFEST=$("$P" - "$S" "$VARIANT" <<'PY'
import json,sys
from pathlib import Path
pointer=json.loads((Path(sys.argv[1])/'scene/environment'/sys.argv[2]/'current.json').read_text())
manifest=json.loads(Path(pointer['path']).read_text())
if manifest.get('splat_context'):print(pointer['path'])
PY
)
if [ -n "$CONTEXT_MANIFEST" ]; then
  "$P" -m calibrated_scene_reconstruction.scene.background.capture_scope "$CONTEXT_MANIFEST" \
    --output "$OUT/capture_scope.json" > "$LOG/capture_scope.log" 2>&1
  tools/run_isaac.sh apps/scene/review_usd_cameras.py \
    "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" \
    --renderer rt --context-only --output "$OUT/context_only" > "$LOG/context_only.log" 2>&1
  tools/run_isaac.sh apps/scene/review_usd_cameras.py \
    "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" \
    --renderer rt --context-only --context-empty --output "$OUT/context_empty" > "$LOG/context_empty.log" 2>&1
  tools/run_isaac.sh apps/scene/review_usd_cameras.py \
    "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" \
    --renderer rt --context-disabled --output "$OUT/context_disabled" > "$LOG/context_disabled.log" 2>&1
  "$P" -m calibrated_scene_reconstruction.scene.review.gs_visibility "$OUT/review.json" "$OUT/context_only/review.json" \
    "$OUT/context_empty/review.json" "$OUT/context_disabled/review.json" --output "$OUT/gs_visibility.json" > "$LOG/gs_visibility.log" 2>&1
  SOURCE_CONTEXT=$("$P" - "$CONTEXT_MANIFEST" <<'PY'
import json,sys
from pathlib import Path
print((json.loads(Path(sys.argv[1]).read_text()).get('splat_context') or {}).get('source_asset') or '')
PY
)
  if [ -n "$SOURCE_CONTEXT" ]; then
    tools/run_isaac.sh apps/scene/review_usd_cameras.py \
      "$S/export/usd/${TAG}_blender/${NAME}_${TAG}_blender.usd" "$CAMERAS" \
      --renderer rt --context-only --source-context "$SOURCE_CONTEXT" --output "$OUT/context_source" > "$LOG/context_source.log" 2>&1
    "$P" -m calibrated_scene_reconstruction.scene.review.source_preservation "$OUT/review.json" "$OUT/context_source/review.json" \
      "$OUT/context_empty/review.json" --frozen-exclusion "$OUT/frozen_scope/manifest.json" --output "$OUT/source_preservation.json" > "$LOG/source_preservation.log" 2>&1
    "$P" -m calibrated_scene_reconstruction.scene.review.residuals "$OUT/review.json" "$OUT/context_source/review.json" \
      --output "$OUT/appearance_residuals" > "$LOG/appearance_residuals.log" 2>&1
    "$P" -m calibrated_scene_reconstruction.scene.review.native_owners prepare "$OUT/review.json" "$OUT/source_preservation.json" \
      --output "$OUT/native_owners" > "$LOG/native_owners_prepare.log" 2>&1
    if [ -f "$OUT/native_owners/job.json" ]; then
      tools/run_isaac.sh apps/scene/review_usd_cameras.py "$OUT/native_owners/scene.usda" \
        "$OUT/native_owners/cameras.json" --renderer rt --output "$OUT/native_owners/render" > "$LOG/native_owners_render.log" 2>&1
      "$P" -m calibrated_scene_reconstruction.scene.review.native_owners rank "$OUT/native_owners/job.json" \
        "$OUT/native_owners/render/review.json" --output "$OUT/native_owners/ranking.json" > "$LOG/native_owners_rank.log" 2>&1
    fi
  else
    echo 'Original GS preservation pending: reprepare context to bind its same-capture source asset.'
  fi
  if [ "${#COMPOSITION_OBS[@]}" -gt 0 ]; then
    if [ -n "$SOURCE_CONTEXT" ]; then
      "$P" -m calibrated_scene_reconstruction.scene.background.appearance_plan "$S" "$OUT/context_source/review.json"         --observations "${COMPOSITION_OBS[@]}" --output "$OUT/appearance_work_plan.json" > "$LOG/appearance_work_plan.log" 2>&1
    fi
    "$P" -m calibrated_scene_reconstruction.scene.review.hybrid_composition "$OUT/review.json" "$OUT/context_only/review.json" \
      "$CONTEXT_MANIFEST" --empty "$OUT/context_empty/review.json" --frozen-exclusion "$OUT/frozen_scope/manifest.json" --observations "${COMPOSITION_OBS[@]}" --output "$OUT/composition.json" > "$LOG/composition.log" 2>&1
    if [ -f "$OUT/appearance_work_plan.json" ]; then
      "$P" -m calibrated_scene_reconstruction.scene.background.gs_restore_plan "$OUT/appearance_work_plan.json" \
        --composition "$OUT/composition.json" --output "$OUT/gs_restore_work.json" > "$LOG/gs_restore_work.log" 2>&1
    fi
  else
    echo 'GS preservation pending: provide --composition-observations instance masks before quality approval.'
  fi
fi
PRESENCE_OBS=("${COMPOSITION_OBS[@]}")
PRESENCE_CONFIG="configs/$NAME/environment_source_presence.json"
if [ -f "$PRESENCE_CONFIG" ]; then
  PRESENCE_PATHS=$("$P" - "$PRESENCE_CONFIG" <<'PY'
import json,sys
q=json.load(open(sys.argv[1]))
if not isinstance(q.get('observations'),list) or not q['observations']:
    raise ValueError('Nonempty explicit source observation list required')
for path in q['observations']:
    if not isinstance(path,str) or not path or '\n' in path:raise ValueError('Explicit observation paths required')
    print(path)
PY
)
  mapfile -t CONFIG_PRESENCE_OBS <<< "$PRESENCE_PATHS"
  PRESENCE_OBS+=("${CONFIG_PRESENCE_OBS[@]}")
fi
if [ "${#PRESENCE_OBS[@]}" -gt 0 ]; then
  # Unknown source identities must not disappear while routing appearances.
  # Keep rendered comparisons available, but block quality approval until the
  # agent models/associates every reviewed instance or explicitly rejects it.
  "$P" -m calibrated_scene_reconstruction.scene.review.source_presence "$S" \
    --observations "${PRESENCE_OBS[@]}" --output "$OUT/source_presence.json" \
    --require-pass > "$LOG/source_presence.log" 2>&1
fi
if [ -n "$REVIEW" ]; then
  # The request must name this exact rendered round, not a previous candidate.
  "$P" - "$REVIEW" "$OUT" "$VARIANT" <<'PY'
import json,sys
from pathlib import Path
q=json.loads(Path(sys.argv[1]).read_text())
if Path(q['render_review']).resolve() != (Path(sys.argv[2])/'review.json').resolve() or q.get('variant','twin') != sys.argv[3]:
    raise SystemExit('Quality request must bind this exact variant/render round')
PY
  "$P" -m calibrated_scene_reconstruction.scene.review.environment "$S" "$REVIEW"
fi
echo "Unapproved environment candidate and full-scene comparisons: $OUT"
echo 'Inspect source/matched/novel views and record all whole-scene criteria before approval.'
