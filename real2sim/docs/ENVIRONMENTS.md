# Runtime environments

Keep host processing, Blender, simulator execution and optional learned models
in separate environments. The harness protocol is independent of their versions;
backend adapters must be checked against the actual installed runtime.

| Layer | Dependency specification | Purpose |
| --- | --- | --- |
| Host Python 3.12 | `pyproject.toml`, `requirements/host-validated.txt` | Task orchestration, geometry, observations and evidence |
| Blender 4.5 LTS | Separate Blender executable | Editable source construction and retained-state rendering |
| Isaac Sim 5.1 / Lab 2.3 | Simulator Python 3.11 | This branch's PhysX compatibility profile |
| Isaac Sim 6.2 / Lab 3 | Separate simulator environment | Optional retained adapter, selected explicitly with `REAL2SIM_ISAAC_PROFILE=6.2` |
| Native Newton | `requirements/native-newton.txt` | Standalone native surface, solid, strand, hybrid and contact routes |
| Optional perception | `requirements/segmentation.txt` and specialist setup guides | SAM3, tracking and other learned observations |

Requirements snapshots are reference dependency sets, not universal transitive
locks. Pin container digests when reproducibility matters and inspect the runtime
version report after installation. PhysX is supplied with Isaac Sim; do not
install an unrelated PhysX package as a replacement.

## Host extras

The `scene` extra supplies USD composition, `fitting` supplies numerical fitting,
`video` supplies encoding and decoding, and `gaussian` supplies Gaussian data
preparation. The `solid` extra adds tetrahedralization. These dependencies do not
install simulator binaries or model weights. See [installation](INSTALLATION.md)
and the relevant capability documentation for optional requirements.

Run `bpy` scripts with Blender's own Python. Never replace it with an arbitrary
host `bpy` wheel. Expose framework source to simulator scripts through the
launcher; do not upgrade simulator USD, NumPy, torch or Newton in place to satisfy
host requirements. Load USD-dependent modules after Kit starts.

Native Newton uses its isolated environment or package overlay. The supplied
reference pairing is Newton 1.7.0 / Warp 1.17.0. A standalone Newton run is not
an Isaac Lab integration. Source-bound surface conversion preserves control
indices; volume conversion uses MeshPy/TetGen and remaps authored selections.

VGGT, 3DGRUT and optional rendered-view refinement require separate compatible
upstream environments. Install weights separately and preserve their provenance.
Do not downgrade host or simulator dependencies to satisfy an optional model.

## Asset routes

| Asset | Provided runtime adapter | Scope |
| --- | --- | --- |
| Rigid and articulated | PhysX | Editable collision geometry, material and supported joint structure |
| Cloth | PhysX particle cloth by default; optional Newton Lab 3 VBD or standalone native surface | PhysX uses the native construction/evidence API; the generic deformable exporter uses Newton |
| Solid | Native Newton FEM | One homogeneous tetrahedral control and visual bindings |
| Strand | Native Newton | Unbranched rods and supported endpoint attachments |
| Hybrid | Native Newton | Supported rigid/cloth coupling and attachments |
| Fluid | Newton MPM | Seed volumes and supported fixed collision fixtures |
| Fracture | PhysX | Prefractured pieces and impact-triggered release |

Cloth and soft-body guidance prefers PhysX unless the user selects another
backend. Sim 5.1 cloth uses the PhysX construction API; the implemented
tetrahedral solid, strand and hybrid routes above use standalone Newton.
Select the supported adapter for the requested representation and preserve
explicit user backend requirements. These are supplied implementations, not a
restriction on the user's engine choice. An agent may use a configured adapter
or task-scoped native simulator code for a requested backend while retaining
editable parameters, actual state records and source-matched review. A backend
change still requires checking its parameter conventions and actual behavior.
These routes have different adapter contracts. Check the specialist skill before
choosing one. Real material identification and robot-policy validation are
separate from motion sanity checks. See [5.1 compatibility](ISAAC_SIM_51.md) for
its narrower simulator scope and [deformable behavior](DEFORMABLE_BEHAVIOR.md)
for standalone surface loading and limitations.

## Check an installation

```bash
python --version
blender --version
export REAL2SIM_ISAAC_PROFILE=5.1
scripts/run_isaac_lab3.sh /workspace/real2sim_harness/scripts/isaac_lab3_versions.py
scripts/run_isaac_lab3.sh /workspace/real2sim_harness/scripts/isaac_physx_smoke.py --viz none --device cuda:0
PYTHONPATH=.tools/newton-hybrid python -c 'import newton, warp; print(newton.__version__, warp.__version__)'
```

Keep generated environment reports outside published source. Rendering and
physics have different device and driver requirements; CPU dynamics alone does
not satisfy the RTX review path. Consult the official requirements for the
selected simulator release and test representative source assets.

Optional concave rigid contact uses native Newton SDF cooking and SolverXPBD.
`run_rigid_contact_behavior` is diagnostic by default; source-bound acceptance
requires original-material replay, a complete video, passive configuration and
source/cache consistency. See `skills/simulate-mating-rigid-parts/SKILL.md`.
