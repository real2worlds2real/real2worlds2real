#!/usr/bin/env bash
# openpi environment (envs/openpi): Physical Intelligence's openpi (third_party/openpi) for fine-tuning pi0.5 with its
# own JAX training code (scripts/train.py) and serving it; kept apart from the other environments (its jax 0.5.3,
# torch 2.7.1, transformers 4.53.2, numpy 1 pins).
#
#   GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive third_party/openpi
#   setup/setup_openpi_env.sh        # ~10 min, needs network and uv
#
# Python 3.11 and the repository's uv.lock (jax[cuda12] wheels: no system CUDA). The weights download on first use
# from gs://openpi-assets (pi05_base, ~12 GB) and the PaliGemma tokenizer from gs://big_vision into $OPENPI_DATA_HOME
# (default ~/.cache/openpi).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/third_party/openpi"

GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$ROOT/envs/openpi" uv sync --python 3.11 --frozen

"$ROOT/envs/openpi/bin/python" - <<'PYEOF'
import jax
import openpi
print("jax", jax.__version__, jax.devices(), "| openpi", openpi.__file__)
PYEOF
