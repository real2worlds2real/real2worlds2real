# Isaac Sim 5.1 compatibility

This profile targets Isaac Sim **5.1**, Isaac Lab **2.3** and simulator Python
**3.11**. Keep the host application in its separate Python **3.12** environment.
Do not install host USD, NumPy or Newton packages into the simulator environment.
The harness and asset contracts remain independent of a particular Sim release;
the backend adapter supplies version-specific behavior.

## Select the profile

The shared dispatcher defaults to Sim 5.1 / Lab 2.3 on this branch. You may also
select it explicitly in the environment that launches framework commands:

```bash
export REAL2SIM_ISAAC_PROFILE=5.1
```

`scripts/run_isaac_sim51.sh` selects 5.1 for a single invocation. It does not
change its parent shell. `run_isaac_lab3.sh` is the shared dispatcher despite its
historical filename. Version checks reject a mismatched simulator runtime.

## Local installation

Install Sim 5.1 and Lab 2.3 following the
[Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html).
Activate that installation's compatible Python environment, then configure:

```bash
export REAL2SIM_SIM_RUNTIME=local
export REAL2SIM_ISAAC_PROFILE=5.1
export REAL2SIM_ISAACLAB_ROOT=/path/to/IsaacLab-2.3
scripts/run_isaac_sim51.sh /workspace/real2sim_harness/scripts/isaac_physx_smoke.py --viz none --device cuda:0
```

The launcher translates `/workspace/real2sim_harness/...` to this checkout. It does not
create or upgrade the installation. Host task directories may be outside the
checkout when using the local runtime.

## Containers

With Docker and NVIDIA Container Toolkit configured:

```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
export REAL2SIM_SIM_RUNTIME=docker
export REAL2SIM_ISAAC_PROFILE=5.1
unset REAL2SIM_ISAAC_IMAGE REAL2SIM_SIM_CACHE
scripts/run_isaac_lab3.sh /workspace/real2sim_harness/scripts/isaac_lab3_versions.py
scripts/run_isaac_lab3.sh /workspace/real2sim_harness/scripts/isaac_physx_smoke.py --viz none --device cuda:0
```

The default image is `nvcr.io/nvidia/isaac-lab:2.3.0`; its default cache is
`.cache/isaac-sim-5.1/runtime`, separate from the 6.2 cache. Explicit overrides
still take precedence. Container use is subject to the upstream license terms.
For Apptainer, select `REAL2SIM_SIM_RUNTIME=apptainer` and set
`REAL2SIM_ISAAC_IMAGE=/path/to/isaac-lab-2.3.0.sif`. Container task paths must fit
the launcher's checkout mount mapping. See [installation](INSTALLATION.md).

## Adapter scope

| Workflow | 5.1 route |
| --- | --- |
| Rigid and articulated assets | PhysX conversion, contact review, joint control and retained trajectories |
| Prefractured rigid assets | PhysX contact and fragment handoff |
| Granular solids | Optional [solid particle API](PHYSX_MATERIAL_CAPABILITIES.md#solid-grains); GPU PBD grains sharing native rigid colliders |
| Capsule-chain ropes | Optional [native D6 chain API](PHYSX_MATERIAL_CAPABILITIES.md#capsule-chain-rope); complements the standalone Newton rod route |
| Mesh scenes and asset views | PhysX scene feedback and RTX rendering |
| Physics experiments | Native material/drive bindings, reset, parameter readback and state recording |
| Thin elastic surfaces / cloth | Native PhysX particle-cloth construction and [scene integration](SCENES.md#physx-particle-cloth-scenes-on-sim-51) with rigid participants, configurable contact/solver settings and source-bound evidence |
| Gaussian NuRec volumes | Native RTX mixed rendering with PhysX mesh participants; separate support/collision geometry |
| Standard Gaussian ParticleField | Transcode to NuRec for 5.1; direct rendering requires a compatible newer runtime |
| Lab 3 Newton integration | Not ported to Lab 2.3 |
| Standalone Newton surfaces, solids and other native routes | Separate runtime; not an Isaac Lab Newton integration |

Each asset still needs source-matched numerical and visual review. Adapter
availability does not certify every geometry, material or interaction.

The generic deformable `run_simulation` exporter still builds Newton assets;
use the native PhysX cloth API for this profile. Volumetric soft bodies, strands
and rigid/cloth coupling remain available through their standalone Newton
routes; see [runtime routes](ENVIRONMENTS.md#asset-routes).

## State and control conventions

The Lab 2 adapter starts `AppLauncher` before simulator imports, uses torch
state tensors and converts between Lab WXYZ poses and XYZW interchange at
explicit boundaries. Raw PhysX tensor views already use XYZW. Scene files
`rigid_trajectories.npz` retain legacy XYZ+XYZW poses; hashed
`rigid_states.npz` use the documented XYZ+WXYZ convention. Do not mix them.
Nested rigid links receive a live-stage transform reset after scene placement,
preserving world transforms and the visual/joint hierarchy. Portable source
bundles retain placement transforms.

Scene reviews accept `--width`, `--height`, camera paths and rendering sample
settings independently of physical time. The manifest's physics timestep and
substeps control integration. The repeatable scene argument
`--rigid-kick OBJECT SECONDS VX VY VZ WX WY WZ` applies a world-space velocity
increment in m/s and rad/s at the requested time; subsequent motion is simulated.
The report retains requested and applied times. It is an intervention, not
measured real-world force input.

Rigid ground review computes support from enabled authored colliders at the
measured body pose. `visual_aabb_min_z_m` remains a diagnostic; the
`collision_support` record identifies the geometry used for `min_z_m`. This
geometric check does not measure solver cooking, SDF/decomposition approximation
or contact offsets. Falling, settling and penetration tolerances still apply.

## Check your installation

The network-independent smoke script constructs a falling cube and driven joint.
Require finite states, support response and joint tracking, not merely startup.
Run the physics experiment check in the simulator Python environment:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_physx_experiment.py --output /path/to/new-trial-directory
```

Use a new output directory. The check starts Kit; CPU dynamics does not eliminate
Kit's rendering requirements. Use `examples/scene-physx51-runner.json` for scene
integration, substituting the checkout path and exporting the profile in the
orchestration process. Review `runtime_report.json`, retained states and images.

## Gaussian backgrounds

Use `nurec_volume` with the [5.1 scene runner](../examples/scene-physx51-runner.json).
The adapter enables UJITSO geometry processing before Kit startup and verifies
NuRec field relationships and payload resolution before rendering. Gaussian
appearance does not supply contact geometry. Preserve separate mesh colliders,
source alignment and removal of captured operation targets. Review the actual
mixed frames and movement; a successful load does not establish visual acceptance.
See [conversion and scene setup](GAUSSIAN_BACKGROUNDS.md#isaac-sim-51-native-nurec-scenes).

## Transmissive material export

The standard object USD exporter and mesh-background exporter preserve supported
constant Principled glass through a native MDL material context. Set
`REAL2SIM_MDL_ROOT` to the renderer installation's `kit/mdl/core/Base` directory.
The export copies OmniGlass and its two local library dependencies into the
asset's `shaders/` directory, retaining their original notices. No library is
downloaded or committed to this repository. Keep these files when relocating
the asset. The target renderer must support MDL and its standard library.

Supported source glass connects one Principled BSDF directly to Material Output,
with Transmission Weight 1, Alpha 1, constant Base Color/Roughness/IOR, default
specular/weight settings and no additional scattering lobes, thin-film effect,
coating, anisotropy, emission, volume shader, displacement or normal map.
`material['real2sim_thin_walled'] = True` explicitly describes a single glass sheet;
the default is a modeled thick dielectric. Mixed/partial-transmission or textured
glass requires a corresponding adapter and fails explicitly. Ordinary opaque
materials retain their previous export path.

`visual.materials.json` records source material identities, native parameters and
shader hashes. Conversion audits check these against the actual USD material;
bundle inventories retain the receipt and shader files. The universal Preview
Surface remains available but does not establish equivalent transmission.
Neither these static checks nor a successful shader compilation certifies visual
fidelity: inspect native RTX images through the actual glass, including its
contents and highlights, before accepting. Do not lower Alpha or distort IOR to
hide lost transmission. This adapter is scoped to glass; arbitrary Blender node
graphs are not automatically portable.

## Normal-detail transfer checks

Object and mesh-background USD export now writes `visual.normals.json` or
`environment.normals.json`. When a reachable Principled shader uses a connected
normal input, export requires a corresponding USD normal connection with local
UV texture dependencies. A dropped procedural Noise/Bump path fails explicitly
and retains the failing receipt. Unused nodes do not trigger this check.

Use Blender's native normal baking to preserve required procedural bump detail
as a UV image, retain the image in editable source, connect it through a Normal
Map node and re-export. The exporter does not automatically bake or replace
the source material. The conversion audit rechecks normal connections and
texture hashes; the simulator bundle inventories the receipt. Existing legacy
bundles require re-export to acquire this check.

This is a transfer-loss check for connected Principled normals, not a general
shader translator. It does not validate roughness/color graph equivalence,
normal-map scale or UV accuracy, non-Principled shading, material identity in a
photograph, or final appearance. Inspect source and target renders at useful
views and lighting before judging those properties.

`REAL2SIM_RENDER_DEVICE=cuda:<index>` selects the native review GPU. Without it,
bundle rendering follows a selected CUDA physics device, or uses the default
CUDA renderer when physics is on the CPU. USD authoring/audit startup honors
the same selection instead of initializing an unrelated default GPU.
