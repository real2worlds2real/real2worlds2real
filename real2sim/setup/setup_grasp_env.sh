#!/usr/bin/env bash
# Grasp synthesis environment (envs/grasp) for a separately supplied BODex port on cuRobo v2. It is kept apart from
# envs/recon so the reconstruction stack's pins stay untouched.
#
#   setup/setup_grasp_env.sh        # ~5 min, needs network and uv
#
# cuRobo is pinned to the commit the upstream codebase validated the port against; warp / coacd / trimesh match that
# stack (warp 1.14 ships a CUDA 12.9 runtime, which a 580 driver loads; warp >= 1.15 needs a CUDA 13.x driver).
# coal comes from PyPI (its wheels need numpy 2.2-2.3; the port runs unchanged on numpy 2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python3 -c 'from calibrated_scene_reconstruction.grasping import require_bodex; require_bodex()'
CUROBO="a35a708ecfbb26eb9ab2d7ef22c65919c4fae4a9"

[ -x envs/grasp/bin/python ] || uv venv envs/grasp --python 3.11
PY=envs/grasp/bin/python
uv pip install --python $PY torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@$CUROBO" \
  "warp-lang==1.14.0" "coacd==1.0.11" "trimesh==4.5.1" "numpy>=2.2,<2.4" "scipy==1.15.3" "coal==3.0.3" "scikit-image==0.25.2" pyyaml

"$ROOT/setup/install_package.sh" "$PY"

PYTHONPATH=src $PY - <<'EOF'
from calibrated_scene_reconstruction.grasping.bodex.engine import check_exact_bodex_v2_backend
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| cuRobo", check_exact_bodex_v2_backend().curobo_path)
EOF
