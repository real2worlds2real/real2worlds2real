# Robots and grasp synthesis

`robots/` contains robot specifications and backend-independent geometry. Isaac articulation setup, live joint control and stage inspection belong to `sim/robot_articulation.py`. Import the latter after simulator startup. The former `robots.articulation` module remains a deprecated identity-preserving alias.

The pure `robots.kinematics.SlotRobot` accepts a batched request provider; `sim.batch_kinematics.BatchKinematics` serves those requests on live articulations. `robots.sweep_geometry.Hull` computes convex containment and translation; `sim.sweep` reads the stage and inspects robot motion. Existing class imports from the two sim modules remain available.

## Synthesis entry points

The optional tabletop workflow requires a compatible solver that you have permission to install. Supply the port in its existing filesystem layout. Point the following variable to the absolute directory named `bodex` that contains its `__init__.py`. Start in `real2sim/`.

```bash
export SCENE_RECONSTRUCTION_BODEX_ROOT=/absolute/path/to/grasping/bodex
setup/setup_grasp_env.sh
source envs/grasp/bin/activate
```

The adapter imports `assets.load_end_effector`, `clearance.ClearanceChecker`, `tabletop.scene.RestingPose`, `tabletop.scene.build_scene` and `tabletop.synthesize.solve_tabletop_grasps` from that package. Setup checks the port's `engine.check_exact_bodex_v2_backend`. The public BODex research checkout has a different interface. Use the hand assets for your robot and keep any relative paths required by your solver. The cup mesh examples and base CLI installation run independently of this solver.

```bash
python tools/grasping/synthesize.py sessions/<name> --robot <spec> --seeds 96 --top-k 16
python -m calibrated_scene_reconstruction.grasping.synth sessions/<name> --robot <spec> --seeds 96 --top-k 16
```

The module command forwards to the same composition entry point. The tool constructs the scene session, resolves stored variants and provides demonstration callbacks. For solver attribution and license information see the [component notices](../../THIRD_PARTY.md).

For `--demo`, the tool supplies `seated_mesh(session, object_name)` and `seed_region(region)` explicitly to `synth_demo`. Library callers provide those callbacks as keyword arguments. A functional thin-wall hold preserves the mesh opening; a generic body hold retains the filled-solid behavior. Seed counts, thresholds, mesh keys and output directories retain their existing semantics.

`grasping.synth.variants(session, names)` retains its two-argument interface and shallow overlay: a variant replaces nested values as whole entries. It is distinct from scene assembly's nested light and room overrides. Shared `core.artifacts.tagged_json_path` only looks up files from caller-supplied directories and an identity tag; scene and grasping both use it. It contains no scene-generation or grasp-selection policy.

## Receptacle boundary

`assets.receptacles` owns receptacle descriptions, meshes and textures. `sim.receptacle` authors those inputs as USD geometry, materials and collision properties. These modules implement different responsibilities and share the asset implementation rather than duplicate its geometry. Backend stage authoring stays outside assets.
