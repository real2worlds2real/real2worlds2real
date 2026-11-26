#!/usr/bin/env bash
# Start one station server on a simulation host:  station_up.sh <task> <run_dir> <socket> [num_envs] [extra serve.py args...]
# The task spec's `station` block (task registry, $STATION_TASKS) names the backend, its runtime and environment variables.
# Host facts come from the environment or station_host.env next to this script: STATION_PYTHON (the interpreter with the
# simulator), ASTRA_ROOT (the simulator's task code, e.g. an Isaac Lab task checkout), STATION_TASKS (the task registry on this host).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/station_host.env" ] && . "$HERE/station_host.env"
TASK=$1; RUN=$2; SOCK=$3; NENV=${4:-16}; shift 4 || shift $#
: "${STATION_PYTHON:?set STATION_PYTHON}"; : "${STATION_TASKS:?set STATION_TASKS}"; export ASTRA_ROOT="${ASTRA_ROOT:-}"
SPEC="$STATION_TASKS/$TASK/task.json"
[ -f "$SPEC" ] || { echo "no task spec $SPEC" >&2; exit 2; }
# backend (resolved against the spec's directory), runtime and env vars of the task
eval "$(python3 - "$SPEC" <<'PY'
import json, os, shlex, sys
from pathlib import Path
spec = Path(sys.argv[1]); st = json.loads(spec.read_text()).get("station") or {}
path, _, cls = st["backend"].partition(":")
print(f"BACKEND={shlex.quote(str((spec.parent / path).resolve()) + ':' + cls)}")
print(f"RUNTIME={shlex.quote(st.get('runtime', 'python'))}")
for k, v in (st.get("env") or {}).items():
    print(f"export {k}={shlex.quote(os.path.expandvars(v))}")
# backend options of the task ({spec} = the task spec file, {spec_dir} = its directory), e.g. a twin's session and factory
args = [os.path.expandvars(str(x)).format(spec=spec, spec_dir=spec.parent) for x in st.get("args", [])]
print("TASK_ARGS=(" + " ".join(shlex.quote(x) for x in args) + ")")
PY
)"
mkdir -p "$RUN"
[ -n "$ASTRA_ROOT" ] && cd "$ASTRA_ROOT"
EXTRA=(); [ "$RUNTIME" = isaac ] && EXTRA=(--headless --enable_cameras)
exec "$STATION_PYTHON" -u "$HERE/serve.py" --backend "$BACKEND" --runtime "$RUNTIME" "${EXTRA[@]}" \
  --task "$TASK" --num_envs "$NENV" --socket "$SOCK" --run "$RUN" "${TASK_ARGS[@]}" "$@"
