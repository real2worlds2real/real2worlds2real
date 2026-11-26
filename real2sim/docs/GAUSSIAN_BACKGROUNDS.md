# Optional captured Gaussian backgrounds

Astra can choose a full-mesh environment or a captured Gaussian background with editable mesh participants. Gaussian appearance does not supply collision geometry or make captured objects operable. The shared input/video/COLMAP tools serve both branches; there is no separate scene-draft module.

## Installation and entrypoints

```bash
python -m pip install -e '.[scene,video,gaussian]'
python scripts/prepare_gaussian_background.py /path/to/request.json --output /path/to/new-run
```

In a scene session, Astra uses `run_gaussian_background(request_path, timeout_seconds)`. Requests must be JSON files retained in its workspace. Standalone output and session experiments remain outside the Git source tree. The native CPU route needs NumPy, Pillow, plyfile and OpenUSD with `ParticleField3DGaussianSplat` (host range 26.11–26.x); it needs neither Torch nor a simulator. This host exporter requirement does not pin the target simulator.

## Prepare an existing trained PLY on CPU

Start from [gaussian-prepare.json](../examples/gaussian-prepare.json). The PLY must contain ordinary pre-activation 3DGS properties: positions, log scales, WXYZ rotations, opacity logits, RGB DC coefficients and optionally contiguous `f_rest_*` coefficients for complete SH degree 1–3. An XYZ point cloud or mesh PLY is not a Gaussian model. Other representations need an explicit importer.

The required `frame` specifies a finite, positive uniform similarity `source_to_scene`, mapping source coordinates into scene metres/Z-up with column vectors. State the actual `scale_source` and whether it is verified. Estimates are allowed when stated. Shear, anisotropic rescaling and reflection are rejected. The USD keeps this similarity as a root transform, preserving the local relationship between anisotropic ellipsoids and directional appearance rather than rotating positions alone. The cropped PLY retains the original source frame; do not assume it has already been baked into scene coordinates.

`observed` defaults to true for this captured-background route; set it false for generated or inferred Gaussian content. Cropping and export do not change inferred content into a real observation.

Optional `crop` fields:

| Field | Meaning |
| --- | --- |
| `exclude_boxes` | Scene-frame boxes with `size_m` and rigid `transform`; cover operation targets and their swept motion |
| `keep_bounds_m` | `[minimum_xyz, maximum_xyz]` in scene metres |
| `sigma_extent` | Conservative spherical particle extent multiplier, default 3 |
| `min_opacity` | Discard low-opacity particles after sigmoid, default 0 |
| `max_sigma_m` | Optional largest-axis scale cutoff in scene metres |
| `min_frustum_views` | Optional frustum-count filter; requires actual posed cameras |
| `near_camera_m` | Optional removal of particles intersecting this radius around actual capture-camera centers; default 0, requires posed cameras |

Filters act on complete particles. Exclusion boxes remove intersecting conservative bounding spheres; keep bounds require the sphere inside the bounds. Gaussian tails are unbounded, so finite sigma bounds are a controlled approximation. This may remove more background near large particles; review holes and adjust the explicit zones. There is no automatic hole filling or claim that removed geometry was observed.

Optional `cameras_path` points to JSON:

```json
{
  "cameras": [{
    "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "intrinsics": [[500,0,320],[0,500,240],[0,0,1]],
    "width": 640,
    "height": 480
  }]
}
```

These are undistorted OpenCV pinhole cameras (+Z forward), in the same metric scene frame as the crop. Use actual reconstructed/supplied camera evidence and its justified scene alignment. Coverage reports per-particle frustum inclusion and a direction-diversity diagnostic. It does **not** account for occlusion or certify rendered quality, precise geometry, or unseen-room coverage. No camera evidence returns `not_supplied`; it never fabricates a score.

Outputs include the retained request, cropped `background.ply`, native `background.usdc`, a top-down keep/remove diagnostic, counts and file hashes. Source input is unchanged. The returned `background` can be placed directly in a scene's `backgrounds` list after adding an ID. Composition packages the USD and requires the `particle_field` runtime capability. Program export/composition success is separate from Astra's scene acceptance.

## Optional training and legacy conversion backend

Install the official [3DGRUT](https://github.com/nv-tlabs/3dgrut) in its own environment according to upstream instructions. No checkout, CUDA environment or trained asset is bundled. Record your installed revision. Copy [gaussian-runner.json](../examples/gaussian-runner.json) to a local operator configuration and replace the paths. The backend is an argv process protocol, so another implementation may replace it.

```bash
real2sim-harness generate --scene --reference /path/to/capture.mp4 \
  --objective 'Keep the captured room background and make the selected cabinet operable.' \
  --gaussian-runner /path/to/local-gaussian-runner.json \
  --scene-runner /path/to/local-scene-runner.json --blender /path/to/blender --auth api-key
```

`REAL2SIM_GAUSSIAN_RUNNER` is an alternative. Without a backend, CPU preparation remains available and training/conversion returns an actionable error. Backend configuration contains `schema_version:1`, supported operation names in `capabilities`, and `argv` with `{request}` and `{output}` placeholders. Commands are executed without shell interpolation. It must return `background_report.json` with `ok:true`, actual `backend.name`/`backend.version`, and output-relative artifact paths. Failed subprocesses, missing/wrong output representations and escaped artifact paths are rejected.

Training request:

```json
{"schema_version":1,"operation":"train","dataset":"/path/to/colmap-data",
 "iterations":30000,"downsample_factor":2}
```

The supplied bridge uses an existing COLMAP dataset (`images/` and `sparse/0`). It also accepts the shared COLMAP tool's `sparse/selected` output, staging its images and selected model under the backend output's `capture_dataset` without changing the original capture. Camera recovery, frame selection and optional image masks belong to the shared evidence/capture stage; training does not invent camera poses. The bridge disables dataset normalization and exports PLY directly to retain dataset coordinates, returning `source_ply` for a subsequent explicit alignment/crop request. For large/poorly conditioned captures, replace the backend with a properly normalized route that reports its complete inverse transform. Training iterations and downsampling are configurable; no hardware assumptions are embedded.

Conversion request:

```json
{"schema_version":1,"operation":"transcode","source":"/path/to/prepared.ply",
 "format":"nurec_volume",
 "frame":{"source_to_scene":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
          "scale_source":"Replace with actual alignment evidence","scale_verified":false}}
```

`format` can also be `particle_field`. The bridge delegates conversion to the official [3DGRUT transcoder](https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md), then wraps its output with the explicitly supplied similarity. Specify that transform relative to the converted asset's retained coordinate frame, including any upstream normalization/axis changes; do not apply alignment twice. The runtime adapter must actually support its selected representation. NuRec is available for legacy integrations; new exports use standard ParticleField by default, following upstream guidance. There is no blanket compatibility promise for every simulator version.

[scene-gaussian-video-runner.json](../examples/scene-gaussian-video-runner.json) shows a newer adapter configuration for plain ParticleField/NuRec fields. The [5.1 runner](../examples/scene-physx51-runner.json) declares `nurec_volume`, not `particle_field`. The shared adapter configures geometry processing and disables multi-device rendering before Kit startup, inspects the composed representation, and rejects unresolved NuRec payloads. SPG/PPISP exports still require a dedicated renderer adapter. Setup receipts do not imply visual acceptance.

For standard ParticleFields without an authored material, the example Kit adapter binds its built-in `ParticleFieldEmissive.mdl` radiance material at runtime. Existing materials remain. This renderer dependency is not inserted into the portable CPU-exported USD; other renderers provide their own appropriate particle-field material.

The example adapter selects `RaytracedLighting` for Gaussian scenes. Its hook disables frame generation and averages 64 display samples at the same physical state; `--render-mode` and `--render-samples` override those choices. No physics steps or motion blending occur during capture. Other renderer versions require their own appearance checks.

## Isaac Sim 5.1 native NuRec scenes

Isaac Sim 5.1 uses the NuRec `Volume` / `OmniNuRecFieldAsset` representation for native Gaussian rendering. Standard `ParticleField3DGaussianSplat` assets require a newer compatible renderer. The scene protocol accepts both, but the 5.1 adapter rejects direct ParticleField input with a conversion hint.

1. Prepare and crop the Gaussian source with the existing tool, retaining the original source and cleanup evidence.
2. Transcode with the configured official 3DGRUT bridge, using `operation: "transcode"` and `format: "nurec_volume"` as above. Keep the exporter in a separate environment; do not replace simulator USD, NumPy or torch packages. NuRec payloads use the exporter's legacy precision and representation, so inspect converted appearance rather than assuming a lossless transfer.
3. Compose the returned background entry with editable physical participants and separate support/collision meshes. Use the 5.1 scene runner declaring `nurec_volume`. Assets remain in one PhysX simulation; the Gaussian background is a native RTX visual layer, not a second physics engine or a screen-space overlay.
4. Review actual rendered views and motion. Confirm that mesh objects in front occlude the Gaussian background and objects behind are occluded; verify that moving an editable target does not reveal a captured static duplicate. Keep camera, metric registration, removal-region evidence and the real states used for each frame.

For a prepared `background.ply`, the preparation transform has **not** been baked into its points: pass the same source-to-scene similarity when transcoding. For an already aligned ParticleField USD, retain its authored transform and use an identity outer wrapper. Never apply alignment twice.

The adapter enables `/UJITSO/geometry=true` before `SimulationApp` starts. On the 5.1 NuRec route it disables histogram auto-exposure and enables registered-compositing inverse tone/color correction for captured radiance. Mesh materials remain authored assets. It records the effective settings, detected volume paths and resolved payload references in `gaussian_setup`; `visual_acceptance: false` means an agent still needs to review the frames. Native Gaussian radiance does not imply relightable captured materials, Gaussian collision geometry, or arbitrary transparency/shadow correctness.

See the [official 5.1 NuRec documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html) and [3DGRUT export formats](https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut/export/README.md).

## Optional rendered-view refinement

`refine` is an optional branch of the same Gaussian tool. It requires a separately configured worker environment for official 3DGRUT and an image-restoration environment for official [Fixer](https://github.com/nv-tlabs/Fixer). No weights, accounts or runtime packages are bundled. [gaussian-refinement-runner.json](../examples/gaussian-refinement-runner.json) shows a combined backend that delegates ordinary training/transcoding to the existing runner. The worker and restoration argv contracts are illustrated in the adjacent example files.

A refinement request includes `schema_version:1`, `operation:"refine"`, `source` (Gaussian PLY), `observations` (the shared [posed-observation schema](OBSERVATION_TOOLS.md) with at least two RGB capture views), and `novel_views` (1–32 camera records in the same frame). Cameras use the shared OpenCV integer-pixel-center convention; the worker adapts it explicitly to the renderer. No dataset normalization or scene transform is silently applied. Select cameras near actual coverage; `max_novel_distance_m` defaults to 0.5 in the source frame and rejects greater distance to every capture camera. This is a bounded-neighborhood check, not a complete visibility guarantee.

Options include `anchors` (1–8, bounded by novel count), `iterations` (1–10000, default 200), `learning_rate` (default 0.0025), `min_opacity` (default 0.95), `depth_tolerance_m` (default 0.02), and `relative_depth_tolerance` (default 0.02). Low-opacity geometry and depth discontinuities are excluded from propagation. Gaussian expected depth remains a model estimate; tolerances and scale must be justified for the input. Captured originals remain separate training targets. Generated anchors are propagated only where both view depths agree; remaining pixels are masked out rather than fabricated as observations.

The default `geometry_mode:"color_only"` trains spherical-harmonic appearance with fixed positions, scale, rotation and opacity, verifying the frozen tensors after optimization. `joint_refine` permits small geometry/opacity updates without densification and reports motion; it is not a complete hole-filling or geometry-reconstruction method. Outputs include actual `refined.ply`, original/repaired/propagated views, masks, coverage, original-view residuals and before/after renders. A completed operation does not certify improved quality. Astra may keep the original, adjust views/parameters, rerun or reject the proposed repair, then prepare/export and review the composed scene as usual.

`fixer_view_adapter.py` uses official base and pretrained weights, checks finite same-domain outputs and records the actual model revision/checkpoint hash. Prefer upstream's documented environment. On newer Transformer Engine versions the adapter preserves the official rotary-embedding function after its public import moved. If the image-only Cosmos path imports optional Qwen encoders unnecessarily, `requirements/fixer-text-encoder-lazy.patch` defers those imports until real use; it changes no Fixer image network or weights. Apply only to the matching Cosmos 1.0.9 source in an isolated environment. Extension builds must use that environment's actual Torch headers/libraries. The upstream tokenizer training patch is unnecessary for inference.

Optional CPU `operation:"cousin"` edits standard ParticleField USD/USDZ appearance globally or regionally, preserving native directional appearance. Scene cousins retain layout by default; spatial region edits are permitted for user-requested layout variation. This branch grades existing appearance and moves existing regions, without generating new semantic textures or unseen geometry. See [whole-scene cousins](SCENE_COUSINS.md) for its schema, functional-object coordination and final acceptance boundary.
