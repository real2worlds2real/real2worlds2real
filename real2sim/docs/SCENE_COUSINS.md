# Whole-scene cousins

An accepted twin scene can seed user-requested cousins. Astra interprets the natural-language quantity, shared style, individual variants and scope. The same entrypoint handles accepted object and scene parents. No fixed count, prompt grammar or separate user-facing layout switch is required.

## Default scope and requested layout variation

Preserve function. Articulated participants retain joint roles, types, connectivity and mechanism loops; coupled cloth retains attachment ownership. Shapes, proportions, metric joint frames, dimensions and materials may change through the editable object source. Revalidate the resulting motion rather than blindly scaling a composed asset.

By default keep the scene recognizable: preserve room structure, relative placement, support relationships and operation space. Numeric gates retain object, support and background placement anchors and support proxy dimensions. Astra reviews actual internal background geometry, visual room structure, changed object fit and clearances. Fixed manifest transforms alone cannot prove these semantic properties.

Layout variation is valid when requested by the user. The planner interprets that request and records its scope in an internal per-variant `layout_change_request`; otherwise it records null. The child can then modify placement and support dimensions within the user's scope, with anchor differences reported at acceptance. Astra judges conformity to that scope, and the current target simulation must still pass. A child cannot enable layout changes merely by changing its local intent record. Participant IDs and simulation kinds remain fixed in this initial interface; unsupported changes must be reported rather than silently dropping an operation target.

All-mesh scenes support the same cousin process without Gaussian tools. A scene using captured Gaussian appearance may optionally restyle that background while operation objects use the functional object-cousin workers. Background appearance never supplies physical support or cross-engine coupling.

## Use and recovery

Install the host `scene` extra, Blender and a tested simulator runner. The parent must have current accepted scene evidence and an unchanged accepted package.

```bash
real2sim-harness cousins /path/to/accepted-scene-run \
  --prompt 'Generate two cousins with different furniture shapes and materials, keeping the room layout and all operation functions.' \
  --blender /path/to/blender --auth api-key

real2sim-harness cousins /path/to/accepted-scene-run \
  --prompt 'Generate one cousin with a wider working aisle by moving the desk; retain the operable objects and their functions.' \
  --blender /path/to/blender --auth api-key

real2sim-harness cousins /path/to/retained-family --resume \
  --blender /path/to/blender --auth api-key
```

Optional `--count` overrides the natural-language count. `--scene-runner` and `--gaussian-runner` override the operator adapters for new or unfinished children. On `generate --scene`, a natural-language cousin request or explicit cousin arguments can defer this stage until twin acceptance. A planning-only request is retained without generating assets; a later explicit execution request can start that plan.

The family stores the exact accepted parent scene package, participant source library, editable structural contracts, accepted images/videos, raw request and internal plan. Workers reuse unchanged parent bundles, or call `build_scene_cousin_object` for a redesigned operation participant. Each functional worker retains its own source and review lineage; arbitrary independently packaged replacements cannot bypass that contract. A participant without retained editable source can be reused but cannot be redesigned until its source is supplied.

Child scenes retain failed attempts and sessions. Resume preserves counters and accepted siblings. Object-worker success and background success are evidence, not whole-scene delivery. Final `accept_scene` requires current target-runtime evidence, intact parent/library lineage, the layout policy and Astra's final visual/functional judgment. Failed feedback returns to the same scene repair loop. Exact geometry fingerprints detect reuse; they are not semantic diversity scores.

## Optional Gaussian background cousin edits

The CPU `run_gaussian_background` cousin branch requires the host `gaussian` extra and an existing standard Gaussian ParticleField USD/USDZ. It grades existing spherical-harmonic appearance globally or by regions; it does not synthesize new semantic textures or unseen geometry. Opaque NuRec assets require a suitable converter or an alternative background representation.

```json
{
  "schema_version": 1,
  "operation": "cousin",
  "source": "/path/to/parent-background.usdz",
  "color": {
    "matrix": [[1.05,0,0],[0,1,0],[0,0,0.95]],
    "offset": [0.01,0,0]
  },
  "edits": [{
    "bounds_m": [[0,0,0],[2,2,2]],
    "color": {"offset": [0.02,0,0]}
  }]
}
```

The optional `asset_to_scene` is a rigid column-vector transform mapping the background asset to the scene; default identity. Region `bounds_m` are ordered scene-frame corners selecting particle centers. Up to 64 edits are supported, with first-match priority and overlapping-selection counts. The optional `max_particles` defaults to 1,000,000 and is bounded at 5,000,000 for CPU memory control. RGB grading uses a finite bounded 3×3 matrix and offset; global grading is applied before each regional grading.

For a user-requested layout change, an edit may supply a finite positive uniform-similarity `transform` acting on scene-frame positions. Ordinary scene cousins reject nonidentity spatial region transforms. Geometry, anisotropic ellipsoids and directional appearance move together through USD transforms; positions-only PLY rotation would be incorrect. Moved regions may expose holes, overlap objects or disagree with supports. Astra must repair/review those effects and update separate physical geometry. Generated appearance remains marked `observed:false`.

The source is retained unchanged. Output is a portable USDZ plus numeric reports, without a simulator-success claim. New semantic background generation needs an installed, validated backend or mesh authoring; neither the reconstruction-refinement tool nor affine color grading establishes that capability. Rendering and whole-scene acceptance are still required.
