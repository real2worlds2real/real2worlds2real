# Installation

Run the commands below from the `real2sim/` directory of your checkout. The host setup uses an editable installation so reconstruction jobs can find this checkout’s scripts, skills and runner configuration. For a basic installation of all three Python projects, follow the [repository quickstart](../../README.md#start).

For optional scene orchestration, also install `pip install -e '.[scene]'` in the host environment. [Scene documentation](SCENES.md) describes the version-independent simulator adapter protocol. The bundled example uses an explicit Lab runtime profile; other releases can supply their own compatible adapter.

For optional scene video, use `pip install -e '.[scene,video]'`. The Sim 5.1 scene adapter uses mesh backgrounds. CPU Gaussian preparation is separately available with the `gaussian` extra; [Gaussian backgrounds](GAUSSIAN_BACKGROUNDS.md) describes its additional training and rendering requirements. Installing this extra does not add ParticleField/NuRec rendering to Sim 5.1.

## Host and Blender

The supported full deployment is Linux. Install `bubblewrap` with your system
package manager; generated Blender/Python tools use it for task-scoped execution.
The host must allow its user-namespace operations. See [task access](MODEL_CONNECTION.md#source-and-task-access).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vision,video,simulation,scene,fitting,dev]'
export REAL2SIM_ISAAC_PROFILE=5.1
```

That is the agent harness's host environment. The scene pipeline's GPU stages run in their own pinned environments, built by `setup/*.sh` under `envs/`
([environments](ENVIRONMENTS.md)).

Install [Blender 4.5.15 LTS](https://download.blender.org/release/Blender4.5/) from the official distribution. Use its executable through `--blender /path/to/blender` or `PATH`. The host dependencies do not include Blender or Isaac packages. See `requirements/host-validated.txt` for the recorded direct dependency versions.

Install Codex separately, using the [official CLI installation instructions](https://developers.openai.com/codex/cli). The persistent backend expects stdio app-server, strict configuration, dynamic tools and custom permission profiles. Check compatibility with your installed CLI; the precise protocol is evolving. Model authorization is explained separately in [model connection](MODEL_CONNECTION.md).

## Isaac Sim 5.1 with Isaac Lab 2.3

Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html). Use the simulator's Python 3.11 separately from the host Python 3.12 environment. The bundled Kit-based PhysX/RTX adapters need the full runtime; a kit-less installation is insufficient for those adapters. The framework can instead use a compatible operator-supplied adapter via the [scene runner protocol](SCENES.md#simulator-adapter-protocol). Using the bundled container launcher accepts the upstream image's EULA (`ACCEPT_EULA=Y`).

Choose an existing compatible installation or configure the image appropriate to your adapter. These commands reproduce the tested reference setup; they do not require every collaborator to adopt the same Sim/Lab version. `REAL2SIM_ISAAC_IMAGE` selects the image, and `REAL2SIM_ISAACLAB_ROOT` selects a local installation. Changing those settings does not automatically port version-specific APIs: adapt and validate the affected conversion/runtime entrypoints while retaining the common asset and feedback contracts.

### Docker

Install Docker and the NVIDIA Container Toolkit on the host, then:

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
export REAL2SIM_SIM_RUNTIME=docker
export REAL2SIM_ISAAC_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
scripts/run_isaac_sim51.sh /workspace/real2sim_harness/scripts/isaac_lab3_versions.py
```

For repeatability, replace the tag with the digest of your downloaded image. Run the version inventory and smoke checks with your selected container runtime before using it for asset acceptance.

### Apptainer

```bash
mkdir -p containers
apptainer pull containers/isaac-lab-2.3.0.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0
export REAL2SIM_SIM_RUNTIME=apptainer
export REAL2SIM_ISAAC_IMAGE="$PWD/containers/isaac-lab-2.3.0.sif"
# Enable only if your Apptainer installation has fakeroot configured:
export REAL2SIM_APPTAINER_FAKEROOT=1
scripts/run_isaac_sim51.sh /workspace/real2sim_harness/scripts/isaac_lab3_versions.py
```

### Existing local Lab installation

```bash
export REAL2SIM_SIM_RUNTIME=local
export REAL2SIM_ISAACLAB_ROOT=/path/to/IsaacLab-2.3
scripts/run_isaac_sim51.sh /workspace/real2sim_harness/scripts/isaac_lab3_versions.py
```

The local launcher translates the stable project mount prefix to the current checkout. Activate your simulator environment before invoking it. It requires an existing compatible Lab installation and `isaaclab.sh`; it does not install or change that environment. Run the smoke checks below to validate your installation.

All branches keep the internal project prefix `/workspace/real2sim_harness` to match existing converters. This is a virtual runtime path, not a required host location. Container outputs must be within the checkout. `REAL2SIM_SIM_CACHE` can select a separate cache location. Docker may create root-owned output depending on its host configuration; use a suitable container/user setup on your workstation.

For the bundled object-conversion checks, `REAL2SIM_PHYSICS_DEVICE=cpu` selects
CPU physics; the default is `cuda:0`, and `cuda:<index>` selects another device.
The bundle records the selected physics device. RTX visual checks still use GPU
rendering. Scene adapters select their device independently through their
operator-configured arguments. Choose a supported collision/physics combination:
the current GPU dynamics adapter rejects requested sweep CCD rather than
allowing the engine to silently disable it. See the optional
[rigid-contact guidance](../skills/review-simulation/rigid-contact.md).

## Validate the 5.1 installation

See [Isaac Sim 5.1](ISAAC_SIM_51.md) for the Lab 2.3 profile, separate cache,
startup commands, supported workflows and runtime validation requirements.
Select `REAL2SIM_ISAAC_PROFILE=5.1` for existing framework commands. Run the installation and source-asset checks described in that guide.

```bash
scripts/run_isaac_sim51.sh /workspace/real2sim_harness/scripts/isaac_physx_smoke.py --viz none --device cuda:0
```

The retained Sim 6.2 / Lab 3 adapter is optional. Using it requires an explicit
`REAL2SIM_ISAAC_PROFILE=6.2` and its own compatible image or local installation;
see [runtime environments](ENVIRONMENTS.md). The default on this branch is 5.1.

## Native Newton for hybrid, strands and mixed scenes

Install a separate target with the activated host Python. No simulator environment is overwritten:

```bash
python -m pip install --no-deps --target .tools/newton-hybrid -r requirements/native-newton.txt
```

Both packages and the host numerical dependencies must be present. The hybrid entrypoint also selects this overlay inside the Lab process through the project mount. Strand simulation selects it in a host subprocess. Keep the pair pinned; replacing only one component is not covered by the recorded baseline.

The optional mixed-scene route uses this same overlay only in its physics subprocess. Configure [scene-newton-runner.json](../examples/scene-newton-runner.json) with your actual runtime paths; container launchers must translate host scene/output/manifest/review paths into their mount. RTX replays saved solver states in a separate subprocess with the renderer's own dependencies. See [scene support and limits](SCENES.md#native-newton-mixed-adapter).

For particle-material reviews without Lab 3, configure [the native MPM runner](../examples/mpm-native-physics-runner.json)
with `scripts/simulate_particle_materials_native.py` and
`--native-runtime /path/to/newton-hybrid`. Its Python needs NumPy and SciPy.
Set `REAL2SIM_MPM_RUNNER` to that operator JSON for `run_fluid`; the scene fluid
adapter uses its existing `--physics-runner` option. The native launcher preserves
the particle, transition, surface and review-report formats. It supports fixed
axis-aligned support boxes and prescribed material changes, not thermal prediction
or dynamic pan/fluid coupling. Blender renders the retained surfaces independently
of Isaac Sim. The original Lab-based launcher remains the default.

## Cell Fracture

The patched upstream add-on required for thin-wall fracture is included under `vendor/cell-fracture/`, with its original headers, GPL license and patch provenance. The Blender exporter imports this copy. No separate add-on installation is needed. See [third-party notices](../THIRD_PARTY.md).

## Optional reference tools

COLMAP can be installed using `python -m pip install -e '.[geometry]'`. SIFT is the basic route. ALIKED/LightGlue requires separately supplied compatible ONNX models and provider support; consult `skills/recover-multiview-geometry/SKILL.md`. No model weights are included.

VGGT-Omega uses an isolated environment:

```bash
git clone https://github.com/facebookresearch/vggt-omega.git .tools/vggt-omega
git -C .tools/vggt-omega checkout 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
python3.12 -m venv .envs/vggt-omega
.envs/vggt-omega/bin/python -m pip install -e .tools/vggt-omega
```

Install a compatible PyTorch build in that environment using upstream instructions. Obtain `vggt_omega_1b_512.pt` under the upstream license and place it in `.tools/checkpoints/vggt-omega/`. Access approvals and download credentials belong to the collaborator. Environment variables `REAL2SIM_VGGT_REPO`, `REAL2SIM_VGGT_PYTHON` and `REAL2SIM_VGGT_CHECKPOINT` can override these locations.

The adapter also requires a successful validation record matching the actual checkpoint file's size and modification time. Use `scripts/validate_vggt_checkpoint.py` to produce that record from a strict state-dictionary load in the isolated environment. Never copy a previous machine's validation report and call it validated.

```bash
.envs/vggt-omega/bin/python scripts/validate_vggt_checkpoint.py
```

If an optional tool is unavailable, its status reports that fact. It is not necessary to download optional reconstruction models to use the code-generation path.
