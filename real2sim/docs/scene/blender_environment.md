# Agent-authored Blender environments

This route builds editable surrounding environments while keeping the task,
support, robot and calibrated cameras fixed. It is separate from the task-object
solid-mesh harness, which would merge identities and lose open scene geometry.
Native Blender USD export retains separate entities, materials, UVs and lights.
The native environment defaults to RTX path tracing at 128 samples and 8 bounces
with fixed exposure. The configured renderer is retained in the exported USD and
used for review. A more expensive renderer alone does not satisfy the quality gate.

`environment.json.render` can override mode, samples, bounces, auto exposure and
`photometry` (`tonemap_op`, `film_iso`, `exposure_time_s`, `f_number`,
`cm2_factor`). New builds record all exposure fields explicitly rather than
inheriting Kit's camera preset. Defaults are display-exposure hypotheses
(ISO 100, 1 s, f/1, ACES), not measured capture settings. Tune against source RGB
when metadata is unavailable and retain brightness/clipping findings in the
review. Older built assets without these fields retain their existing behavior.

In `configs/<session>/twin.json` select `background: "blender"` and
`environment_quality_version: 1`. Existing frozen sessions need a derivative.
Use `python -m calibrated_scene_reconstruction.scene.background.derive sessions/<frozen> <new_name>` to copy
the frozen task, model sources and legal capture/calibration into separate regular
files. It verifies the parent freeze, refuses existing destinations, excludes old
capture automation helpers and does not inherit environment approval records.
Write `environment.json` alongside it:

```json
{
  "version": 1,
  "model": "agent/environment/model.py",
  "parameters": {
    "cabinet": {"shape": "rounded", "roughness": 0.45},
    "layout": {"cabinet_position": [2.0, 0.0, 0.5]}
  },
  "assets": {"wood": "data/<capture>/environment/wood.jpg"},
  "physics": {
    "ground": "background_floor",
    "ground_height": 0.0,
    "entities": {
      "background_floor": {"mode": "foundation"},
      "cabinet": {"mode": "resting", "on": ["background_floor"]}
    }
  },
  "variants": {
    "rearranged": {
      "parameters": {
        "cabinet": {"shape": "curved", "roughness": 0.25, "friction": 0.4},
        "layout": {"cabinet_position": [-2.0, 1.0, 0.5]}
      }
    }
  }
}
```

The example numbers illustrate the interface, not a room template or measured
scene. Runtime has no filename/task-specific modeling policies. The agent writes
`agent/environment/model.py` with `build(parameters, context)`, using Blender to
assemble geometry in the session's metric Z-up world. Parameters may contain
arbitrary nested JSON; variants recursively override dictionaries and replace
lists. Variants can also override the model or asset map. Task overrides are
rejected. No room template is substituted when agent source is missing.

Every visible mesh needs `object["r2s2r_entity"]` naming its background instance.
Optional `parameters.excluded_entities` lists instances the user excludes from
reconstruction, such as capture equipment or people. The harness checks both
the planned entities and the actual exported entities; an excluded instance
cannot remain in the exported scene. This does not exclude fixed room fixtures
or other similarly shaped objects unless their instance IDs are also listed.
When one source surface is split into several entities, optional
`parameters.surface_material_groups` records its `material`, `exported_material`,
`basis`, at least two `source_views`, and `members` with `entity` and mesh `parts`.
The build checks the planned material and the actual exported material names on
each listed part. A distant wall segment cannot silently revert to a default
white material. Source image review still determines the correct groups and
material appearance; unrelated wall finishes remain independent.
For a source-visible furniture instance, `background.mesh_proposal.solve`
can refine the actual exported triangles using bounded translation, yaw and
anisotropic scale. Supply at least three source views and explicit visibility
domains; keep unobserved or occluded pixels out of the silhouette loss.
The optional GPU rasterizer proposes parameters, then an independent CPU
rasterizer remeasures the same source masks. This is fitting evidence, not
held-out validation or appearance approval. After applying the proposal to an
agent model, rebuild and remeasure the exported geometry: procedural rebuilding
need not reproduce an affine mesh transform exactly. Reconcile support height
and retain normal contact, structure, material and full-scene checks.
`context["reserved_entities"]` identifies protected task/support instances that
must not be rebuilt. `context["assets"]` maps logical asset IDs to copied files.
Textures/HDRs must come from this snapshot or be baked into the round. Unsupported
procedural material nodes are rejected: bake them rather than silently exporting
flat materials. Principled PBR/image/UV/normal-map networks are supported by the
worker; successful conversion still needs final-export image review.
An image-linked Principled emission color with a non-unit/linked emission
strength is rejected: the native exporter otherwise drops that multiplier and
can turn captured screen detail into clipped white. Bake the multiplier into a
separate texture before exporting an emissive surface.

Blender rigid-body metadata is carried to USD collision, mass, kinematic state,
friction and restitution by the simulation adapter. New environment builds also
require a `physics` contract: `ground`, its capture-estimated `ground_height`,
and an `entities` dictionary. Declare the ground as `mode: "foundation"`; ordinary
assemblies use `mode: "resting", on: ["support_entity"]`. A fixed attachment needs
named `anchors` with a `part` and capture `evidence`, plus actual contact geometry.
`part` is the exported object's full name, as recorded in the audit's components.
Task objects and the protected support can be read-only collision references.
Every background mesh needs a declaration. Exported surroundings have colliders.

`pipeline/environment.sh` always audits the actual composed USD before review,
including when `--reuse-build` is used. World-space triangle geometry is checked
for disconnected members (including islands within one mesh), closed and
consistently wound physical members, actual support
contacts, overhangs, tipping margins and interpenetration. Support chains must
reach the ground or the protected support; mutually supporting floaters fail.
Static success is followed by a two-second gravity release of all resting
background assemblies together in an isolated Isaac scene. The robot and task
objects are not simulated; their source poses are never corrected by this check.
The experiment assumes rigid members, conservative convex colliders, shared
300 kg/m³ validation density (10 g minimum body mass), friction 0.6/0.5 and a
uniform-density gravity centre. Declared validation masses are recorded explicitly.
It verifies modeled stability,
not material strength or the truth of hidden attachments.

Numbered results live in `qa/environment_physics/<variant>/round_N/`, including
geometry, worker snapshots and passive-simulation evidence. Failed candidates may
still render for diagnosis, but cannot pass whole-scene review, S3 or S6. The
approval binds the physical result to the exact USD, native build and source
hashes. Changing geometry, the physical checker or its configuration invalidates
approval. A visual review cannot override missing/failed physical checks.

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.blender_env sessions/<session> request
bash pipeline/environment.sh <session> --variant twin --robot <robot>
bash pipeline/environment.sh <session> --variant rearranged --robot <robot>
```

Each invocation builds an immutable `scene/environment/<variant>/round_N/` with
source/assets/worker snapshots, native USD, Blender scene, entity metadata and
input/output hashes. `current.json` points to the latest built round. Changed
sources, task state or output files invalidate verification. Captured and novel
review cameras are fixed once; `--cameras` accepts an existing manifest and
`--reuse-build` verifies rather than rebuilding. Exports and review images use
new numbered directories.

Review the actual exported full scene, not isolated attractive objects or a
Blender-only preview. Submit all twelve criteria through
`scene.review.environment`. Use `kind: "twin", variant: "twin"` for the base or
`kind: "cousin", variant: "rearranged"` for that cousin. Include the built native
USD in `candidate_inputs`; the review binds its current build manifest. Cousin
quality records live separately under `qa/environment_quality/cousins/<variant>`.
They cannot approve the twin's S3. All criteria must pass and all findings must be
resolved. No actual task/table dimensions or task CAD are permitted as fitting or
acceptance inputs.
Controlled render-setting/background-payload/transform overrides are diagnostic
ablations. They cannot approve an export with different settings: bake the chosen
configuration into a new candidate, export it and render that actual asset.

Whole-scene photorealism includes coherent materials/texture scale, lighting,
contact shadows, occlusion, relative layout, fine detail and cross-view behavior.
Novel views have no paired capture truth. Rich cousins can change shapes,
appearance, properties and layout, but must remain physically and visually
coherent. Pixel agreement with the unchanged original is not a cousin criterion.

`pipeline/run_session.sh` has an `environment_build` stage before delivery, uses this route
for `background: blender`, and skips legacy task-changing cousins. It requires
S3's current whole-scene quality approval before delivery can pass.
Its delivery stage also reruns the mandatory audit on the actual
packaged `export/usd/<session>_twin_blender.usdz`, after packaging rather than
only checking the unpacked source. `--delivery` keeps a separate
`delivery_current.json`, so checking a later delivery export does not replace
S3's reviewed-export evidence. S6 binds the delivery audit hash and rejects a
missing, failed, changed or different-export result even when S3 was approved.

`scene.background.splat.inventory` prepares unreviewed instance proposals from
registered captures and the calibrated background GS. Set
`environment_inventory.json` prompts (`vocabulary`, `stuff`) and `n_frames`, or
explicit registered `frames`; use `--output` with a new session directory. The
review view list does not change video sampling or training frames. Same-frame
duplicate masks contribute only one vote and cannot prove multi-view support.
Inspect crops and point extents before modeling; broad masks, duplicate classes
and partial furniture are not automatically accepted geometry.

The experimental background iteration helpers also operate on arbitrary captures:
`scene.background.observations` caches configured semantic masks, source crops and
rectification validity; `--config` selects a separate request list.
`scene.background.structure_fit` proposes vertical planes, connected posts and
source-contour top profiles using the same calibrated camera frame. Mask votes
are restricted to front-most cloud points and distinct views. Color filters and
ROI choices belong to capture configuration. `scene.background.refine` proposes
bounded image-silhouette extents; degenerate and partially observed fits need
agent review and are never auto-approved. Each command requires a fresh
session-local `--output` and refuses a frozen delivery.
For entity-list models, refinement now searches full-circle yaw hypotheses and
ranks actual triangle silhouettes across distinct observed masks. A bounded
Powell refinement can cross initial yaw partitions. `--entities` restricts the
proposal list; `--yaw-hypotheses` accepts 1–48 starts (default 12). Optional
per-entity `shape_prior.width_depth_ratio: [min,max]` is an explicit structural
prior, not measured dimensions. Image-edge clipping reduces bounding-box weights;
mask overlap, bound hits and all orientation candidates remain in the proposal.
The shared `structure_fit.triangulate` helper also validates agent-matched image
landmarks: it rejects inconsistent reprojections, points behind cameras and weak
baselines before fitting geometry. This can refine textureless/partly occluded
objects whose cloud PCA extents are misleading; it does not consume real sizes.

`scene.background.surface_textures SESSION CONFIG --output NEW_DIR` bakes masked
calibrated surface pixels onto a configured world quad or extracts a captured
material patch. Every bilinear sample must be inside the observed surface.
Coverage masks mark hidden texels, which receive only a declared plain fill.
White-reference albedo normalization and patch tiling are appearance hypotheses,
not calibrated inverse rendering. These tools do not alter video sampling,
reconstruction scale, protected task models or their positions.
An optional `emission_gain` writes a separate float RGBE radiance texture, with
gain baked into its pixels. This is captured, source-lit appearance rather than
recovered relightable reflectance. Use unit emission strength in the agent model.
For a planar captured surface, `enclosed_appearance_holes_max_px` can explicitly
include small enclosed mounted details in photographic appearance. It does not
infer their depth. Optional `captured_relief` supplies a calibrated world-point
context (`cloud`, `camera_sha256`, `band_m`, at most 0.3m): nearest context depth
can validate additional source pixels close to the plane while rejecting closer
foreground and unknown gaps. Rectification validity remains mandatory after all
mask operations. These options require capture review; neither creates solid relief.

For nonplanar native geometry, use `scene.background.mesh_textures`:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.mesh_textures \
  sessions/<session> configs/<session>/mesh_texture_request.json \
  sessions/<session>/qa/environment_physics/twin/round_N/job.json \
  --output sessions/<session>/qa/environment_mesh_textures/round_N
```

The request names an `entity`, masked `observations`, an atlas `resolution`
(128–4096), `emission_gain` and plain `fallback_srgb`. Optional
`part_fallback_srgb` maps original part-name globs to inferred colors, for example
hidden wooden feet rather than upholstery. Every pattern must match a part.
The source geometry job must bind the current native build. UV unwrapping splits
vertices but retains every audited source triangle and member identity.
Texture samples require a complete valid bilinear mask footprint, front-facing
geometry and a perspective-correct native self-depth test. Multiple parts can
occlude one another. Hidden surfaces retain explicit plain colors; padding is
restricted to atlas gutters. At least two distinct observed views are required.
View overlap supports bounded RGB display harmonization (`max_view_gain`, 1–2,
default 1.5), followed by resolution/angle-weighted feathering (`blend_power`,
1–32, default 12). This reduces exposure seams without claiming inverse lighting.
The manifest records coverage, contributing frames, gains and exact geometry,
radiance, source and calibration hashes. Coverage includes hidden/interior faces;
a low total is not automatically a failure or a pass for camera-visible surfaces.

Declare the output mesh, radiance and proof as logical assets, then set:

```json
"surface_atlases": {
  "background_entity": {
    "mesh": "geometry_asset_id",
    "radiance": "radiance_asset_id",
    "evidence": "proof_asset_id"
  }
}
```

The worker first builds the agent's actual native geometry, checks capture/task
identity and payload hashes, then compares imported surface vertices within
0.02mm. It separately bounds symmetric surface differences to 0.5mm because USD
and Blender may triangulate slightly nonplanar bevel quads differently; actual
maxima are recorded in `surface_atlases.json`. Changed shapes must be rebaked.
Only static background meshes with the external physical contract are supported;
protected task/support and per-object rigid-body overrides are rejected.
The imported UV members retain the same entity and undergo the normal exported
geometry/gravity checks. A cousin may set an entity's atlas binding to `null` to
disable captured appearance before changing its shape or use a newly baked atlas.
Photographic radiance is not a relightable PBR material. Actual exported renders,
not a texture preview or preservation check, decide appearance acceptance.
The native export adapter also preserves explicitly zero-specular photographic
emission. Blender exports a nonstandard PreviewSurface `specular` scalar; RTX's
default metalness workflow otherwise keeps IOR 1.5 and reflects scene lights over
the captured pixels. For image-linked emission with black diffuse, authored
zero specular, zero metallic and zero clearcoat, the adapter writes IOR 1 and an
explicit zero-color specular workflow. Other PBR materials are retained. Changed
shaders are recorded in the build manifest; diagnostic and actual RTX renders
are both needed to validate the result. Display tone compression still applies;
this correction does not recover capture exposure or inverse lighting.

An experimental hybrid route retains same-capture GS appearance around selected
native replacements. Hybrid builds now require a `completion` contract in
`environment.json`. The agent must assemble the declared architecture in its
Blender source; this contract does not silently insert a box-room template:

```json
"completion": {
  "version": 1,
  "footprint": [[-4,-3],[5,-3],[5,4],[-4,4]],
  "floor_z": 0.0,
  "ceiling_z": 3.2,
  "surfaces": [
    {"entity":"room_floor", "role":"floor", "provenance":"observed_and_inferred", "basis":"Capture ground plane; hidden carpet extended."},
    {"entity":"room_walls", "role":"wall", "provenance":"inferred", "basis":"Perimeter beyond visible furniture; uncertain hidden openings."},
    {"entity":"room_ceiling", "role":"ceiling", "provenance":"inferred", "basis":"Partial ceiling evidence plus an explicit height/structure hypothesis."}
  ]
}
```

These example dimensions illustrate the interface, not actual building sizes.
The current validator accepts convex counterclockwise footprints, including
rotated/irregular convex rooms. Multiple/concave room envelopes need extension;
do not certify them by declaring an unrelated rectangle. Surface provenance is
`observed`, `observed_and_inferred` or `inferred`, with an explicit basis.
Unseen geometry is plausible completion, never claimed as measured truth.
Every declared surface must have a physical support contract and remain visible
in the final hybrid (`native_entities` or `gap_fill_entities`). Add native
furniture/wall/floor gap fillers wherever filtering leaves missing appearance.

`pipeline/environment.sh` audits actual composed triangles after physical QA.
`scene.background.completion` casts spherical rays from a floor-plan grid at
three heights and from registered camera centres. Only declared architectural
triangles can close rays; furniture, field opacity and the background display
color cannot hide an open room. Vertical rays also require real floor/roof
geometry at the declared planes, continuing past beams/fixtures. These finite
ray samples detect broad holes; they are not a proof against arbitrarily small
cracks. Results and immutable geometry/worker snapshots live in
`qa/environment_completion/<variant>/round_N/`. Missing, hidden, stale or failing
completion prevents hybrid visual acceptance and the presentation-video app.
The video checks the exact exported USD that passed completion.

First build/export the native geometry (with completion and `splat_context:
null`), then prepare context:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.observed_context \
  sessions/<derivative> sessions/<same_capture_gs> configs/<derivative>/observed_context.json \
  sessions/<derivative>/qa/environment_physics/twin/round_N/job.json \
  --output sessions/<derivative>/scene/observed_context/round_N
```

Context config selects existing background `native_entities`; their nearby
Gaussians are removed by actual triangle distance. The geometry job must bind
the current native build and geometry hash. Optional `replace_claim_ids` remove
their old semantic appearance as well; disjoint `preserve_claim_ids` protect
identified neighboring chairs, partitions and props from the metric margin.
These IDs come from the capture inventory, with no runtime class/task presets.
The preparation config is snapshotted alongside its hashes.
Completed environments apply a quality filter by default; `quality_filter`
overrides its settings. It culls nonfinite/out-of-envelope, oversized, faint and
isolated rows, then requires DC-color agreement with multiple spatially distinct
registered photo views. Its defaults are `max_sigma_m: 0.065`, `min_opacity:
0.08`, `neighbor_radius_m: 0.10`, `min_neighbors: 4`, `photo_views: 64`,
`min_photo_views: 2`, `max_rgb_error: 0.35`, `max_sigma_px: 12`,
`surface_band_m: null`. The pixel bound is tested only in observing camera views
with matching source appearance; proximity to a camera behind the surface does
not erase otherwise supported detail. Non-null `surface_band_m` is rejected:
distance from an unapproved collision proxy cannot determine GS validity.
Explicit settings and rejection counts
are written to `quality_filter.json`; `quality_rows.npz` preserves selection.
This is conservative appearance rejection, not occlusion proof, sharpening or a
photorealism pass. Filtering creates gaps, so native gap fillers are necessary.
Source training/sampling and frozen task geometry remain unchanged.
Optional `volume_entities` selects native replacements whose old captured
interiors must also be removed, rather than leaving Gaussian ghosts between
surface bands. Neighbor claim preservation still applies. Small explicit
`surface_margins_m` (at most 0.3m) can match a declared photographic relief band;
other entities retain the default small replacement margin.
`gap_fill_entities` remain camera-visible without removing the captured field;
their ownership role must require either a GS coverage mask or an explicitly
inferred enclosure surface. They cannot all be opaque furniture proxies.
All other native entities retain physical proxies but are hidden from the camera.
Their shapes still require the mandatory physical audit; Gaussian appearance is
not solid geometry or evidence of material strength.

Declare the generated `context.usdz` and `manifest.json` as logical `assets`, then
set `splat_context: {"asset": "context_id", "evidence": "proof_id"}` and
`render.mode: "rt"`. The source and derivative must have identical calibrated
camera files. Prepared context binds the native model sources, all parameters,
protected task state and payload hash; changed geometry requires re-preparation.
The optional `splat_context.native_light_gain` calibrates only native lights
against captured display appearance (default 1), not measured illumination.
Source GS tone mapping is restored explicitly. Uniform Blender world HDRs are
converted to equivalent constant-color domes, avoiding unsupported tiny texture
uploads while preserving their radiance; nonuniform HDRs retain their texture.
Native dome illumination is hidden from primary rays, so a light gain cannot
turn sparse captured surfaces white. `splat_context.background_color_linear`
sets an explicit neutral display fill (three values in 0–1; default black).
It supplies no missing geometry or detail and cannot satisfy coverage review.
World-quad source-lit baking can also cover native floor gaps with captured
carpet appearance; unseen texels remain plainly marked in its coverage map.

Optional `ground_completion` can restore trained near-ground Gaussians rejected
only by the floater filter. It requires matching source row IDs, conservative
ground/size bands, color agreement and at least two distinct captured floor
masks. Unseen, undrawn, faint, oversized, near-camera and task-zone rows remain
excluded. This is coverage repair requiring visual review, not generated content.

`pipeline/environment.sh` builds, composes, audits, packages and renders this
hybrid with the configured RT renderer. Whole-scene review must declare
`backend: "hybrid"` and bind the actual context payload and preparation proof.
Approval cannot substitute a native-only render for the delivered hybrid.
Packaging retains the NuRec payload and calibrated transform inside standalone
USDZ. GS blur, ghosts, holes and unsupported novel-view appearance remain explicit
whole-scene rejection reasons even when the native physics checks pass.

Validation covers native Blender twin/cousin builds and actual Isaac composition
with changed shape, material, friction and layout. The actual-capture peg-room
derivative now includes connected furniture, captured whiteboard writing and
material patches, a triangulated workcell frame, sloped wire fence and inferred
architectural completion. Its earlier 51 background entities passed the exported-geometry
audit and 44 released assemblies pass passive gravity verification. Hidden wall
attachments remain declared architectural assumptions. Source/matched/novel
renders of the native-only baseline still show coarse props, lighting/layout
differences and missing distant architecture. The experimental hybrid restores
captured distant context. The sofa now renders as an actual native mesh with
capture UV appearance, a six-view silhouette fit, continuous back, layered
cushions and explicit support feet; the mounted-detail wood wall is a native
plane with photographic surface detail. These replace their old Gaussian
appearance. The enclosure-completion candidate adds four inferred perimeter
walls and a ceiling with connected beams, service tray and suspended fixtures,
for 56 background entities. The first completion candidate made all native
props camera-visible and over-filtered GS against those coarse proxies; that
candidate was rejected. The revised route assigns appearance ownership per
entity and only exposes capture-supported gap masks or explicitly inferred
architecture. Hidden architectural details remain hypotheses.
Camera-visible native-only ablations distinguish these contributions.
Remaining cloth smoothness, atlas seams, photographic relief, display/exposure
differences and distant GS artifacts mean that whole-scene photorealism remains
failed.

For a single uninterrupted presentation, prepare `scene.review.roam_path` with
`--show-ceiling` and render it using `apps/scene/render_review_video.py`:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.review.roam_path sessions/<session> \
  --output sessions/<session>/qa/environment_video/roam_N.json --seconds 90 --fps 24 --show-ceiling
PYTHONPATH=src tools/run_isaac.sh apps/scene/render_review_video.py <completed-export.usd> \
  sessions/<session>/qa/environment_video/roam_N.json \
  --output sessions/<session>/renders/environment_review/roam_N
```

This smooths one capture corridor with a wide lens and upward looks; it does not
select reconstruction frames or train a model. The video app disables dynamics
in an anonymous session layer, uses zero-time rendering and checks all existing
world transforms before/after. It records continuous-take/cut counts, camera
poses, input hashes and preservation evidence. Captured/novel still review also
disables dynamics and guards world transforms. A complete enclosure and a video
do not imply whole-scene photorealism acceptance.

Hybrid appearance ownership is mandatory for newly prepared contexts. Declare
every background entity in `appearance_ownership` with a `role` and `basis`:

- `gs`: camera-hidden native collision proxy; captured GS supplies appearance.
- `native_replacement`: explicitly selected and reviewed mesh; remove only its
  corresponding GS rows, preserving neighboring semantic claims.
- `gap_fill`: native mesh fills missing appearance on a declared
  metric plane. `plane` specifies `origin`, two orthonormal `axes`, `extent_m`,
  `pixel_m`, `band_m` and `max_radius_m`. Capture-supported rows near that plane
  create a coverage mask; overhead furniture cannot punch holes in the floor.
  The default `compositing: "cutout"` uses alpha testing, not physical
  translucency. Fractional mesh transparency can obscure foreground NuRec GS.
  `compositing: "depth_backing"` keeps a solid capture-textured plane behind
  foreground GS, avoiding black cutout gaps around noisy near-floor rows. It
  requires `gap_radiance` evidence; below-plane GS is occluded by that plane.
- `inferred_enclosure`: unmasked mesh is allowed only for a completion surface
  explicitly declared `provenance: "inferred"` (such as an unseen roof).

The mask changes appearance only; collision meshes stay solid. USD export binds
a separate material per masked mesh and preserves the original shared material.
Context preparation binds the ownership code, selection inputs and mask hashes.
Changing any of them requires re-preparation.

Context filtering treats thin sheets separately from spherical blobs and
needles, while retaining the requirement for spatially separated photographic
support. The default `planar_filter: "capture_supported"` retains thin sheets backed by
at least two spatially separated source views. The experimental
`bounded_projection` mode additionally bounds metric size and projected oriented
covariance and requires neighborhood support. It can delete real walls and is
not the default. Thinness alone is not photographic support. With aligned semantic
floor rows, the lower rejection band is estimated
from repeatedly observed floor residuals (bounded to15–100mm). This does not move
the calibrated floor. After filtering, the three-sigma oriented extent of every
Gaussian is checked against each exported protected task/support mesh part,
with a15mm margin. Overlapping rows are excluded so field tails cannot paint
the frozen task meshes. `protected_rows.npz` and the manifest record this
conservative rejection; they do not certify an unchanged rendered appearance.

`opacity_gain` in the context preparation config is an experimental appearance
option (default1, bounded1–4). It increases accumulated opacity without changing
centres, covariance, SH or task geometry. It can strengthen sparse distant
surfaces and also amplify artifacts; source and free-view review remain required.

For a source-lit plane, declare `gap_radiance` as an entity-to-binding dictionary,
with `evidence` pointing to a `surface_textures` result and `surface` naming its
surface ID. Preparation verifies the exact world quad, entity, camera, protected
task state and input/texture hashes. Optional `fallback_texture` and
`fallback_tile_m` in a planar bake tile a declared material in unseen texels;
these remain excluded from observed coverage. The result is captured lighting,
not recovered relightable reflectance.
`observation_entity` can explicitly associate a modeled plane with its parent
surface mask; the output records both identities. A bake with zero observed
pixels cannot be bound as captured radiance. Its fallback remains an unapproved
appearance hypothesis.

Hybrid pipeline runs render four versions of the same exported scene:
the delivered composite, `--context-only` (background meshes hidden), and
`--context-only --context-empty` (GS also hidden), plus the original same-capture
GS with `--context-only --source-context ASSET`. Diagnostics edit only the USD
session layer and cannot approve a candidate. Pass instance-mask evidence with
repeatable `--composition-observations` arguments, for example:

```bash
pipeline/environment.sh <session> \
  --cameras sessions/<session>/qa/environment_cameras.json \
  --composition-observations sessions/<session>/qa/environment_observations/<round>/observations.json
```

`scene.review.hybrid_composition` compares matched source pixels in GS-owned
instance masks. The field must contribute relative to the empty render and
match the source before that pixel counts as reliable. More than5% overpainting
in any scored object/view fails; fewer than64 reliable pixels leaves that view
unscored. Two scored source cameras and evidence for every GS-owned entity are
required. Missing instance masks therefore keep this check pending/failed.
This checks captured-view preservation only; it cannot prove novel-view
completeness or photorealism. Whole-scene requests must name the current
`composition_audit` JSON. Review validation recomputes its scores and checks
all image, observation, scene and camera hashes. Normal pipeline completion
still produces an **unapproved candidate**, even when physics and enclosure pass.

Whole-scene hybrid acceptance additionally requires `hybrid_review`: `objects`
must review every monitor, laptop, chair and stool for `complete` and
`front_back_correct`, each with a boolean `pass`, visual `note` and at least two
current `{source, matched, novel}` comparisons in that object's observed source
frames. `gs_free_views` reviews each current novel render for holes, floaters and
double images; `appearance_boundary` reviews GS/mesh overpainting and seams.
Missing records or failed results keep the whole scene rejected even if physics
and enclosure geometry pass. These are explicit image-backed semantic reviews,
not an automatic claim that geometry or RGB agreement proves photorealism.

A `gap_fill` may instead declare `source_crop: {"pixel_m": 0.01}`. Preparation
verifies all proposed crop decisions against the original GS row exclusions
and hashes the original cloud, selection masks and crop code. Additional
original exclusions (for example a separately removed robot surface) are not
filled by this declaration. Per-face alpha atlases show native surfaces only
inside that verified crop. The original mesh points, faces, UVs and collision
remain unchanged; new face-varying coverage UVs and an isolated material are
added. Export rejects changed geometry, payloads or unsupported material subsets.
This addresses known crop holes; it does not certify the replacement shape.

Twin review also requires `source_preservation_audit`, comparing the delivered
composite with the original GS rather than its newly filtered version. Its
background scope comes from the original task-zone, semantic and valid-pixel
masks. The check binds those fixed masks and all render inputs; deleting a
source region from the candidate cannot redefine the comparison baseline.

For an original export that explicitly recorded a raw-readout robot-surface crop,
`source_crop` may set `include_robot_surface: true` and its recorded sampling
resolution (`robot_surface_voxel_m`, default0.005). Preparation checks the raw
robot-cloud hash, declared sample count, calibrated pose and exclusion margin;
the resulting predicate must reproduce **all** original row decisions exactly.
Without that record the extra crop is refused. This consumes calibration input,
not robot manipulation tests or task ground-truth dimensions.

For holes not caused by a known crop, `gap_fill` can declare
`mesh_coverage: {"pixel_m": 0.01, "radius_m": 0.05, "coverage": 0.9}` instead
of `plane` or `source_crop` (exactly one is required). A per-face atlas samples
the actual exported mesh surface, then estimates local opacity support from
up to32 retained, source-backed GS neighbors. Gaussian widths are capped at
half the declared radius so a large distant sheet cannot erase a small native
part. Native appearance is shown where support falls below the declared
threshold. This is a local coverage hypothesis, not a ray-transmittance or
shape-accuracy certificate. The original-source preservation check and visual
free-view review remain mandatory. Collision and topology do not change.

`mesh_coverage.close_gaps_m` optionally closes very small holes in each surface
atlas (0–0.03m, default0). This affects the appearance hypothesis only and must
be assessed for overpainting; strict original-crop masks never use this closing.

### Background training

`train.masked: true` in `splat.json` excludes the task/robot masks from GS loss.
This can allocate the point budget to the background instead of modeling parts
that will later be cropped. It is an experiment until held-out and actual
composed free-view checks pass; training metrics cannot approve the scene.

Every training run now copies its registered images, masks and COLMAP poses to
`splat/training_data/round_N/`. Derived loss masks are written and hashed there;
training never renames or overwrites the capture's mask files. Missing masks
reject a masked run. `train.mask_source_session` can reuse masks from another
session only when camera bytes, task instance identities and every source image
match. This does not select or downsample the input video. The snapshot and
training record bind the observations used by that run. Background/task metrics
use independent camera validity; task pixels excluded from loss remain visible
to evaluation, and empty regions do not turn the aggregate into NaN.

After retraining, context preparation can set `reference_session` to the original
same-capture GS session. It verifies the cameras and all image bytes and retains
that original field and fixed masks as the preservation baseline. A new field
must not redefine its own comparison baseline. `native_entities` may be empty
when explicit gap fillers are present; a hybrid need not replace a whole object.

`scene.background.trial` waits for a named training exit file, verifies the
private snapshot and unchanged task/calibration, exports the trial, and renders
it against a fixed original-GS review. Example (all outputs stay unapproved):

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.trial sessions/<session> \
  --training-exit /tmp/<training-job>.exit \
  --usd sessions/<session>/export/usd/<baseline>/<baseline>.usd \
  --cameras sessions/<session>/qa/<cameras>.json \
  --reference sessions/<session>/qa/environment_renders/twin/<round>/context_source/review.json \
  --output sessions/<session>/qa/environment_trials/<trial>
```

The trial writes `status.json`, export/render logs, original/trial RGB metrics,
and source/original/trial/free-view comparison images. Source masks are fixed;
the candidate cannot select only its own good pixels. Photometry or coordinate
changes reject the comparison. Even an RGB improvement leaves `pass: false`:
completeness, physical support, GS floaters and full-scene appearance still need
the ordinary environment gates. This command does not change `environment.json`.

### Blur and missing structure

Hybrid now defaults to agent mesh work for blurred, missing or unverified
appearance. Spatial GS coverage is insufficient. The pipeline writes
`appearance_work_plan.json` from the fixed original-GS/source/free-view
comparisons and instance masks. Its detail-loss diagnostic is image-relative;
it does not read actual object dimensions or certify shape accuracy.

Context preparation accepts `appearance_plan`. Retained GS needs two bound
source views with resolved detail and an explicit agent free-view review. A
`hybrid` entry additionally uses a native gap filler for unseen sides. The
optional `retain_claim_ids` keeps only those reviewed semantic owners; unassigned
rows cannot remain as unexplained blurry background. Re-segment a retrained
field before using semantic IDs: rows from another field cannot be reused.

Agent parameters may declare `assembly_requirements`: required mesh part names,
observed/inferred provenance and structural segment endpoints. A frame needs an
upper-perimeter contract by default. An observed open-top frame instead needs
an explicit exception with two hashed source views from the same capture.
Unknown upper structure is completion work, not a reason to omit it.

Requirements are checked in the Blender output and again in actual exported
world geometry. A beam must reach its declared endpoints along its length within
5mm; this is a connection tolerance against the agent's own model, not error
against an unavailable true dimension. Existing contact, penetration, gravity
and passive-stability checks still apply. Presentation cameras include a smooth
look and bounded pullback that frame the complete assembly, in addition to the
room/ceiling route. Pullbacks check exported background/task mesh bounds for
clearance; framing is not proof that no other object occludes the assembly.

For a rectangular beam, its centreline alone does not determine the orientation
of its broad face. An agent model can use `background.structural_axes.section_frame`
with a source-reviewed face normal to obtain X-width, Y-depth and Z-length axes.
This prevents an arbitrary roll from swapping width and thickness. An undefined
or axis-parallel normal is rejected; matching endpoints alone still does not
establish correct cross-section geometry or appearance.

Whole-mesh captured-radiance atlases with less than 25% observed surface are
rejected at binding. Sparse projection over a complete object can leave texture
patches or plain unlit fallback over most of it. Retain PBR completion or refine
the geometry and source coverage first. Neither the routing plan nor a sharp
mesh bypasses source fidelity, semantic completeness or photorealism review.

### Instance counts and local GS retention

Inventory association now treats disjoint same-class masks in the same image as
different instances. A broad proposal in another frame cannot merge those
instances through a transitive link. Proposal-to-Gaussian row membership and
rejected bridge links are saved for inspection; inventory counts still require
agent review, particularly where masks are nested or an object is mostly hidden.

`scene.background.gs_regions` prepares local sharpness masks from the original
captured-GS diagnostics and source instance masks. For example:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.gs_regions \
  sessions/<session> sessions/<session>/qa/<review>/context_source/review.json \
  --observations sessions/<session>/qa/<observations>/observations.json \
  --entities <background_entity> --output sessions/<session>/qa/gs_regions/<round>
```

Set `capture_regions` in the observed-context configuration to the resulting
`regions.json`. Retention needs two camera positions separated by more than
5cm, front-visible point support, resolved source detail and acceptable local
RGB agreement. Regions select original rows; they do not create detail or
authorize reusing a different field's semantic IDs. `retain_claim_ids` selects
reviewed thing owners; `retain_stuff_ids` additionally selects unclaimed surface
labels bound by each entity's `source_stuff`. The unassigned label is distinct
from every surface label. Native gap fillers provide the remaining appearance;
the full collision mesh stays present. The whole-scene preservation masks and
visual acceptance criteria remain unchanged, even when only local regions are
retained. A larger GS count is not a quality pass.

Retention also requires at least 2 degrees of ray parallax. The complete projected
3-sigma Gaussian footprint must fit inside trusted pixels; testing only its
centre can leak blur across mask boundaries. An unfrozen context additionally
uses the audited, read-only frozen robot meshes with a bounded surface margin.
This is background exclusion, not robot pose refinement or operation. A wrong-
depth photographic robot ghost needs source-image exclusion as well:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.source_exclusions \
  sessions/<session> sessions/<session>/qa/gs_regions/<proposal>/regions.json \
  --observations sessions/<session>/qa/<foreground_observations>/observations.json \
  --entities <foreground_observation_entity> \
  --output sessions/<session>/qa/gs_regions/<excluded_proposal>
```

This creates new support masks and preserves original capture/loss masks. Each
support view needs an explicit foreground observation, including model
abstentions. Excluded pixels cannot vote for background GS; other views may
still support the true background behind the foreground. Abstentions and
semantic mistakes still require visual inspection.

### Actual GS visibility and appearance feedback

`solid_backing` keeps a complete native surface, but may hide a useful captured
field. For measured local gaps, first render the actual retained field using
`--context-only` and a matching `--context-empty` diagnostic. A nonblack
background requires the empty render; gray empty pixels are not GS coverage.
Prepare `scene.background.capture_gaps` with that context manifest, retained-
field review, original regional evidence, `--empty-review` and
`--opacity-evidence`. Measure opacity support first with
`scene.background.gs_opacity`: render the same field/cameras/photometry twice,
using `--context-only --context-background 0 0 0` and `1 1 1`.
Only backdrop-invariant pixels can cut native appearance. RGB agreement on
one gray background cannot establish opacity; legacy colour-only coverage is
rejected. Unmeasured views retain native backing. Set the context
configuration's `projected_coverage` to its `coverage.json`, and the relevant
owner's `mesh_coverage.compositing` to `captured_projection`.

After changing background geometry, remeasure the retained field with the
current physically audited export before rebinding. This command runs fresh
black, white, empty and retained-field diagnostics, then writes opacity and
same-owner coverage. It preserves the field and task geometry and rejects
missing native render proofs even when Kit exits with code zero.

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.background.measure_context \
  sessions/<session> sessions/<session>/scene/observed_context/<bootstrap>/manifest.json \
  sessions/<session>/qa/<cameras>.json sessions/<session>/qa/<regions>/regions.json \
  --output sessions/<session>/qa/environment_refinement/<new-round>/measurement
```

Use `measurement/coverage/coverage.json` in the new ownership configuration,
then run `context_rebind`. Use a new output directory for each measurement;
existing proofs are never overwritten. Measurement does not approve appearance:
the actual composed source/free views, visibility, physical and whole-scene
checks remain required.

For an explicitly rejected `native_replacement` owner with a complete PBR mesh,
`scene.background.owned_part_replace` proposes a local field subset without
requiring a photographed texture. Pass `SESSION CONTEXT_MANIFEST GEOMETRY_JOB
CONFIG --output NEW_DIR`. The config names one background `owner`, exact exported
`parts`, reviewed source `observations`, a `reason`, bound `evidence_sha256`, and
a bounded `surface_margin_m`. Removal requires both repeated source ownership
and proximity to those actual audited parts. Reliable GS/gap owners and task
parts are rejected. Surviving kernel attributes remain byte-identical; original
row indices and every removed row index are saved. This only proposes a field:
it does not establish that the native shape, material or new composition is good.

When only the field changed, `measure_context --background-field
<proposal>/context.usdz` can measure it on the current audited export without
rebuilding task geometry. This explicit mode checks the package, current agent
geometry sources, parameters, capture and protected state. The actual render
must also prove the overridden field bytes and calibrated transform. A concurrent
native-pointer change invalidates measurement. Rebind the measured candidate,
then rebuild and audit the actual composite before acceptance.

For a newly observed object absent from the old GS instance inventory,
`scene.background.coverage_regions` prepares coverage masks from reviewed
same-object source annotations and original GS comparisons. Each annotation
must have an explicit `appearance_owner_review`; class labels alone are
insufficient. Supply `SESSION ORIGINAL_GS_REVIEW --observations JSON ...
--entities ID ... --output NEW_DIR`. Its `regions.json` is for measured native
gaps only, with no kernel claim IDs. It cannot assign or delete GS rows, qualify
the whole object's geometry, or approve appearance. Source/free-view reliability,
two separated opaque source rays, native depth compatibility and final composed
review remain separate requirements.

Coverage binds the exact retained vertex bytes, agent parameters, cameras,
protected state, photometry and source masks. Changed geometry or selection
requires a fresh diagnostic. Native samples become transparent only when two
independent, camera-facing, self-visible source observations support them near
the actual retained field. Hidden surfaces remain native. Physical geometry is
always complete. The composed render still decides whether GS is visible and
whether the boundary has holes, overpainting or ghosts.

Each mesh's gap material keeps the complete original image/UV/shader network
inside its own material. Cross-material shader links are not used. Exporter
face reordering permits remapping identical oriented faces; changed vertices,
faces or winding invalidate coverage.

The wrapper also writes `appearance_residuals/worklist.json`, with cropped
source/render/original-GS comparisons. It flags textured regions where captured
appearance was lost or where both reconstructions need mesh work. The agent
reviews object count, missing parts, facing, shape and material, then collects
additional source masks or annotations. This list stays unapproved and never
substitutes for the whole-scene gate.

The wrapper also writes `gs_restore_work.json`: clear photographed furniture
with at least two usable source observations goes back to GS-preservation
review, even when it already has a native proxy. Mesh existence alone never
justifies discarding reliable GS appearance.

`scene.background.gs_closure` proposes restoring original kernels near a
retained core when per-kernel filtering destroys trained opacity/occlusion.
It keeps original kernel attributes, requires two separated background-mask
observations, bounds the restoration radius, and applies frozen task/robot
exclusion. These are latent-neighbor proposals, not front-visibility proof.
They need actual source/free-view and black/white coverage diagnostics before
new native cutouts. Nothing approves the scene or modifies task geometry.

`scene.background.supported_props` fits a background prop envelope to at least
three calibrated silhouettes on a declared desk/cabinet/shelf. It requires
independent baselines and rejects repeated near-identical views. The agent
still writes the semantic subassemblies in its session model, records observed
versus inferred parts, and reruns support, gravity and full-scene review. The fit
uses no real prop dimensions or task CAD and does not approve its geometry.
The fit checks projected silhouette coverage as well as bounding boxes; a thin
diagonal strip with the right image bounds is insufficient. It reports each
source silhouette IoU and active fitting bounds. A `part` declaration requires
the same reviewed `geometry_fit_part` name and `surface_part` scope on every
observation, and certifies only that named part's envelope. Other parts remain
separate hypotheses. Each new proposal saves its solver and initial environment
snapshots, so later unrelated code edits do not change its original evidence.

For a thin part with a concave footprint, `scene.background.support_outline`
projects reviewed original masks onto a declared support/height hypothesis.
Two independent rays must support each retained sample. It reports source
silhouette overlap, disconnected material and inferred raster cleanup. The
cleanup is off by default; an explicit review can complete sampling artifacts
within at most 3 mm. Real openings remain explicit rings and grid cells.
`solid` creates a closed oriented mesh while preserving those holes and rejects
nonmanifold diagonal contacts. A rectangular or convex cap must not fill an
observed crescent opening. Neither the plane/height hypothesis nor the outline
approves the complete parent object, its support stability or its appearance.
Use plausible hidden construction for manufactured enclosures and test it:
a filled solid box can give a misleading centre of mass for a hollow display
case. Record shell thickness and internal supports as inferred, retain the
observed outer pose, and rerun actual contacts and passive stability. Do not
invent a visible oversized base or a fixed anchor to bypass those checks.

When a neighbor occludes a straight edge, its intersection is not a matching
physical corner. `scene.background.surface_landmarks` fits fixed physical
features together with an **unbounded** observed edge. Endpoints of that edge
may differ between views. It requires at least four noncollinear corresponding
features, calibrated separated fitting views, and at least two held-out views;
both point and line reprojection must pass the unchanged six-pixel limit.
`upright_edge_horizontal` is an explicit structural hypothesis, never a pose
measurement. The proposal deliberately does not return complete width/height:
hidden boundaries, support, material and novel-view appearance need separate
review. Inputs and immutable solver/dependency snapshots bind each proposal;
failed held-out checks retain their diagnostics and cannot approve the surface.

For a named part surrounded by other contents, use
`scene.background.part_source_replace` with one `photographic_parts` owner,
explicit `replacement_parts`, and that owner's `source_replacements`. It
intersects the existing audited 3D proximity proposal with independent source
mask ownership. A nearby wrist rest fails source ownership; remote geometry on
the same image ray fails proximity. Neither is removed. The intermediate
distance-only proposal is retained for audit and is never adopted. Surviving
kernel rows remain byte-identical. This requires current physical/backing
proofs and fresh retained-field opacity, coverage and composed reviews; it
does not approve part geometry or photorealism.

Source observations must not vanish during appearance routing merely because
their instance IDs are absent from the current assembly. The environment shell
pipeline now audits all `--composition-observations` plus observations listed
in optional `configs/<session>/environment_source_presence.json` before quality
approval. `scene.review.source_presence` matches exact entity IDs and explicit
`observation_aliases`, never class names. An unmatched observed chair blocks
approval even when another chair is already modeled. A false detection needs a
recorded `rejected_instance_review` on that source observation. The report keeps
per-frame identities, unmatched instances and immutable source/config snapshots.
It checks coverage of the reviewed observations; it does not detect all objects
or certify the shapes/materials of matched objects.

`scene.background.material_fit` makes isolated diagnostic PBR color/image trials
and compares the actual renderer response under fixed cameras and photometry.
Nonresponsive views and insufficient source coverage cannot propose a fit.
`scene.background.source_masks` records explicit, hashed agent polygons when
semantic segmentation misses a visible background part. Neither helper turns a
source-lit photo into measured albedo or relightable reflectance. Bake a chosen
trial into a new normal export and review the entire scene again.

`scene.background.material_channels SESSION PROBES --output JSON` reanalyses
the same immutable renderer probes per RGB channel. A clipped channel does not
discard valid observations in the other channels. Each fitted channel needs
two responsive, unclipped source views bracketed by the measured response;
unidentified or boundary gains stay at one and are labelled explicitly. New
reviewed source patches can be passed with `--observations`, while the original
probe geometry, photometry, assets and source inputs remain hash-verified.
These proposals still need an actual full-scene render; partial channel evidence
cannot establish a complete material or measured reflectance.

A grayscale display response cannot invert a luminance-coupled color operator
by treating RGB channels separately. `scene.background.display_validation`
renders a color cube, can fit positive luminance weights with `colour-model`,
then requires a separate cube (`prepare --response <proposal> --independent`)
before the response can be bound. The independent cube uses different interior
colors; passing needs mean RGB error at most 0.025 and maximum channel error at
most 0.08. These are synthetic display-reproduction limits, not scene fidelity
or object size limits. A fitted response is explicitly unapproved until that
check passes. The diagnostic root copies the actual scene's photometry metadata
and checks the effective settings against the neutral measurement.

Calibrated photographic textures use float32 TIFF. Radiance RGBE's shared
exponent can quantize weaker channels after inverse display mapping. The actual
renderer must validate the chosen format and color transform; neither neutral
ramp agreement nor file-export success is sufficient. This calibration does not
correct wrong geometry, missing objects, lighting hypotheses or GS artifacts.
Converted radiance filenames include a hash of their float pixels, preventing
distinct photographs or calibration revisions from sharing a renderer cache
name. Consume the returned `radiance` path from the manifest rather than assuming
a fixed `radiance.tiff` basename.

Declare sRGB photograph assets in `captured_photos: ["asset_name", ...]` and
bind a validated `display_response` asset. Each build converts those 8-bit
photographs to float32 linear TIFF using that response before passing them to
the agent model. The manifest records source, response and converted-image
hashes. Bind the converted texture as zero-specular emission with strength 1;
do not apply another gain or sRGB conversion. PBR albedo maps and already
converted radiance must stay outside this list. Transparent photographs need
a separate surface/coverage proof and are rejected here. Variants can override
the list; a cousin that moves objects or changes lighting must replace affected
source-lit photographs with appropriate materials or newly validated captures.

Semantic prompt names also do not establish instance identity. The source-mask
selector rejects proposals that do not intersect a requested ROI. Review source
crops before associating objects across views, especially when a prompt can
refer to two different devices. Do not fit their combined silhouettes as one
object. Required subassemblies and source/free-view checks remain mandatory.

GS override diagnostics accept either a complete USD/USDZ context reference or a
raw `.nurec` payload. They are different representations: a package must replace
the background reference, including its field bounds, and must never be written
into the density/color `filePath`. Other payload suffixes and USD files without
the two expected fields are rejected. The review records the override kind and
preserves the calibrated background root transform. Overrides remain diagnostic
and cannot approve delivery. Inspect actual field contribution; row counts and
an empty render cannot demonstrate successful GS preservation.

Context preparation also has an explicit experimental
`projection_support: "render_candidate"`. It proposes centre-supported kernels
for joint render evaluation, while the default `full_footprint` keeps the strict
3-sigma test. A trained field's foreground and background kernels interact:
deleting edge/occluder kernels can expose bright rear kernels and create streaks
that did not exist in the original render. Neither projection rule establishes
quality. Compare the actual retained-field render with the fixed original and
source images before binding native gaps; reject new holes, streaks and floaters.
Use existing source-preservation and composition checks and the full free-view
review. Native proxy existence alone is not a reason to erase photographed
furniture; retain image-backed GS where it remains reliable.

For staged geometry fitting, `scene.background.refine --lock yaw height depth`
can hold current hypotheses while fitting remaining coordinates. These are
coordinate-descent constraints, not supplied real dimensions. Open cabinets are
supported. Silhouette fits can otherwise collapse unseen depth or move a cabinet
through floor objects. Inspect proposals and rerun actual exported contact,
penetration and passive stability checks before context preparation. A lower
image residual cannot override a physical failure.

Once the actual retained field has been rendered, use
`scene.background.context_rebind` to update native appearance without selecting
kernels again. Supply the original context manifest, the new ownership config,
the current audited geometry job and a new output directory. The command checks
all original selection inputs, exact capture/protected state/parameters, current
physical audit and an oriented world-triangle geometry signature. Reordering
vertices/faces is allowed; changed geometry or winding is rejected. It copies
the same PLY bytes and reuses the same GS package. The manifest records this
frozen selection and the new coverage masks. Binding it cannot approve the scene.

A `captured_projection` declaration must use evidence scoped to that same
object. Its `coverage_entity` and the owner recorded in the baked coverage
must both match the native entity. Shared `background_capture` masks cannot
justify an object's appearance cutouts: an opaque adjacent object on the same
ray is insufficient. `capture_scope` audits both declarations and actual masks,
and the quality gate rejects mismatches. Omit `frames` to use every registered
view in that object's measured coverage. Use an explicit subset only when
observations require it, and revisit it when capture coverage expands. Native
samples still need repeated, separated, front-facing source support. The
compatibility radius is explicit and bounded to 0.01–0.2m; it is a background
proxy hypothesis, not a measured surface accuracy. Geometry remains complete.
Review the resulting actual GS contribution, source fidelity and free views;
pixel confidence is not semantic-instance or whole-scene approval.

For an owner that declares `preserve_gs: true`, the appearance gate also checks
`capture_coverage.views`: fewer than two usable same-owner views fails through
`coverage_support`. A correctly labeled zero-view mask is complete native
backing, not verified hybrid preservation. Review the missing source masks,
measure their opaque retained-field coverage, rebind and inspect the composite.
`apps/scene/preview_environment_gaps.py` can separately test `opaque_layers`
against an existing measured gap owner without changing the active model, field
rows, metric tolerances or frozen geometry. Its result remains diagnostic.

An observation request may specify `inference_crop: [x0, y0, x1, y1]` in original
source pixels for a small distant object. SAM sees that crop; masks are pasted
back into the full calibrated raster without changing camera intrinsics or
video frame selection. Inspect `inference_clipped_edges`; a crop-clipped mask
cannot be reviewed as a complete whole-object silhouette. A crop or model
projection proposes an acquisition region and never establishes object identity.

When source preservation fails, the environment pipeline also runs
`scene.review.native_owners`: a private diagnostic USD assigns calibrated
colours to background owners, preserves their actual opacity and culling,
leaves frozen materials unchanged and hides the GS field. Up to eight failed
source views produce a ranking of meshes responsible for overpainted reliable
GS pixels. Unknown or blended colours remain unassigned. This is a repair
worklist, not an appearance approval; it cannot excuse an incorrect object
count, silhouette, material or physical arrangement.

For a photographed planar replacement whose old GS still produces double
edges or surface ghosts, `scene.background.field_replace` prepares a separate
field candidate. Its configuration explicitly maps background owners and
photographed native parts to current `capture_backing` proofs. It requires the
current passed physical export, rejects frozen/unknown parts and bounds the
replacement band. The worker selects only near-surface kernels; all removed
indices and unchanged retained rows are saved. No opacity or colour is adjusted.
After changing the field, rerender its opacity and coverage, rebind appearance,
and repeat original-GS preservation, free-view and whole-scene checks. Local
removal is not permission to drop reliable GS or approve a photographic plane
as a complete object. If a wall strip was misidentified as a floor-standing
board, fix the identity and remove unsupported stands before fitting its shape.
Optional `solid_parts` stays within those same photographed owners. Each named
part must be an actual closed convex body; its triangle volume must match its
convex hull. The selection uses that rotated body and a bounded surface margin,
not its larger axis-aligned bounding box. Open or concave parts and disconnected
solids are rejected, so the operation cannot fill unobserved cavities or erase
captured neighbours merely inside a loose bounding box.

Optional `replacement_parts` restricts a photographed owner's distance selection
to named actual background parts. Its photographic and solid parts must be
included. Use this to replace a cardboard shell while preserving independent
captured contents. `source_replacements` maps the same qualified owners to
original registered observation files. Each mask needs an explicit
`appearance_owner_review`, unchanged source/mask hashes and two independent
source rays. Material-chroma narrowing is an optional proposal tool, not an
identity check. `source_only_owners` disables distance selection for owners with
those source masks; it cannot be combined with solid volume replacement.
Retained PLY rows remain byte-identical, and every field change discards old
opacity/coverage and scene approval.

For curved or layered furniture, `mesh_coverage.depth_band_m` can additionally
require agreement between the native ray depth and the frontmost opaque
retained-GS centre. It accepts 0.005–0.08 metres and rejects unknown depth.
This is an approximate compatibility test, not a measured depth map or a true
dimension reference. It prevents one opaque seat from opening a different-depth
backrest merely because both belong to the same chair. Rebuild and render the
actual field after adding the constraint; tighter masks alone do not pass an
appearance gate.

`scene.background.photo_views` ranks literal source photographs for each plane
and creates nonoverlapping appearance masks. Clear, unclipped details take
priority; other views fill only previously unobserved texels. Writing and texture
are not generated. Small enclosed mask holes can be explicitly included for
photographed mounted details, using the existing bounded
`enclosed_appearance_holes_max_px` option. Exterior-connected foreground regions
remain excluded. World/UV backing proofs, actual source/free-view inspection and
the full appearance gate remain required.

During silhouette fitting, an `occluded_object` observation is a lower bound:
visible source pixels must be covered, but unknown pixels behind an occluder
cannot shrink the full body. Complete views and declared priors still constrain
the inferred extent. Review instance identity first; disconnected masks from
two different boxes are not valid multi-view evidence for a single box.
For a nearly square box, a silhouette can match while the face orientation is
wrong. `scene.background.box_landmarks` fits corresponding visible body corners
from independent registered source views, keeping the existing floor height.
Corner indices follow the Cartesian product `x=(-.5,.5), y=(-.5,.5), z=(0,1)`;
the agent must distinguish a body rim from raised flaps or an internal fold.
The proposal binds its original images, cameras and an immutable initial scene
snapshot. Failed fits retain projected/observed corner evidence and remain
rejected; bounded image error cannot be replaced by a dimensions reference.
Even a successful corner fit requires actual rebuilt support, source and
free-view appearance checks before adoption.

Gap binding supports meshes with face material subsets. Every effective subset
keeps a private copy of its complete original material network and receives the
same verified mesh coverage coordinates; subset membership and geometry remain
unchanged. A shared mesh-level material must not flatten photographed fronts
and hidden PBR faces into one appearance. Missing/unsupported surface materials
still reject export instead of silently leaving opaque faces over GS.
Mask filenames include the mesh identity and pixel-content digest. Different
rounds or reordered exports must not reuse an ambiguous basename: renderer
texture caches can otherwise substitute an older mask and make a diagnostic
look better than the exported candidate. The atlas generator is included in
the evidence hashes, and actual composed renders still decide acceptance.

Background edits can reuse exact captured GS bytes with
`scene.background.context_refresh`. Cameras and protected task/robot state must
be unchanged. Declare every background entity. This bootstrap discards all old
cutouts and keeps complete native backing; it cannot transfer acceptance.
Export and audit the new background, render fresh field/empty/black/white
diagnostics, then `context_rebind` requires coverage from that audited export.

If kernel filtering damages an otherwise useful trained surface,
`scene.background.field_restore SESSION SOURCE_SESSION GEOMETRY_JOB --output DIR`
proposes the original jointly trained rows with the frozen task/support/robot
overlap removed. It requires identical source pixels and cameras, keeps every
retained kernel attribute unchanged, discards old native cutouts and transfers
no approval. It may also restore blur and floaters: compare the actual field,
rerender backdrop/coverage diagnostics, and pass the complete composite gates
before adopting it. This is an alternative candidate policy, not permission to
keep every original GS region.

`scene.background.appearance_regions.prepare` narrows an observed instance mask
to source-supported, sharp photographic subregions. Keep the original mask and
record the retained fraction; inspect the original source and free views before
declaring hybrid ownership. The narrowed observations are marked `surface_part`
and cannot be used as whole-object geometry evidence. Complete native geometry
and fresh actual-field opacity still supply every unsupported region. The full
source-preservation gate continues to score the original background scope.

Before scoring background preservation, `scene.review.frozen_scope` projects
only the actual audited task, support and robot meshes at the registered source
cameras. It includes the already declared metric GS foreground protection
margin, preserves the frozen models and excludes no environment furniture.
The pipeline uses these hashed masks in both preservation audits. Views with
no scorable background outside frozen silhouettes are recorded as inapplicable;
at least two applicable views and the existing error limits are still required.
Model projection, not image error or a chosen failure region, defines this scope.
The wrapper uses `scene.review.frozen_scope_cache` to avoid repeating an
identical projection. It validates the previous masks and compares the actual
old/new frozen world vertices, oriented faces, instance paths, registered
cameras and protection margins exactly. Any change requires a fresh projection;
background appearance is always reviewed again. The copied masks and both
exported scenes remain bound in the new evidence.

`captured_backing_parts` in a `captured_projection` declaration maps native part
names to photographic backing proofs. Each proof binds a registered source
quad, camera/task state, calibrated radiance and input hashes. The validator also
checks that this radiance is connected to that part's actual emission shader.
Those parts keep their opaque photographic skin while the other surfaces use
measured GS cutouts. A plain dark inner wall is not photographic backing.
Source-lit skins are not recovered albedo; composite/free-view review remains
mandatory.

Use `scene.background.capture_backing` through its module CLI to bind
a freshly baked plane, registered source frame and named exported mesh part.
The CLI arguments are `SESSION SURFACES_JSON ASSET FRAME MESH_PART --output PATH`.
New proofs use version 2: actual world-space vertices and photographic UVs must
still match the baked world quad. Moving the object or flipping its UVs requires
a new bake. The separate coverage mask records photographed versus fallback
texels; the clipped source footprint does not claim complete photography.

`scene.background.support_surface` estimates a background support top from
separated source views and checks it against independent sparse points.
`scene.background.support_adjust.propose` then moves its supported objects and
their photographic world quads together. It rejects frozen descendants, cycles,
ambiguous support parents and side photographs that need explicit deformation.
Rebake the returned photo planes, rebuild, rerun physics, remeasure GS coverage
and review the whole scene. A source-derived height is not a true-size input.

For a reviewed `surface_part` silhouette, `scene.background.refine --parts-json`
accepts a JSON mapping from entity IDs to actual exported mesh part names. Only
those parts contribute to the fit. For example, a display-panel mask must not
make the solver shrink the stand because it is missing from that mask. The
complete assembly still needs independent support, shape, front/back and
appearance checks. Proposals retain snapshots of the active configuration and
native pointer so applying a candidate does not erase its historical inputs.
An observation may declare `occluded_edges` using `left`, `top`, `right`, and
`bottom`, with an explicit `occlusion_review` of the original source. A hidden
bottom edge is a lower bound; it must not erase constraints from visible top and
side edges. This avoids shrinking a backrest behind a partition or moving a
panel to fit an occluder's boundary. Existing partial observations without edge
annotations retain their conservative policy. Edge visibility and the helper
source are recorded in the proposal; a better silhouette still requires fresh
physical and source/free-view review before adoption.

If source images contradict an existing front/back declaration, use
`--front-hypotheses` with an explicit reason, exact front/reference mesh names,
a normalized local front axis and at least two registered source images bound
by SHA256. This replaces those declarations only inside the proposed fit;
the active configuration remains unchanged. A good silhouette can still face
the wrong way, so inspect visible assembly details as well as the score.
`scene.background.refine_preview` creates a separate diagnostic USD from a
current proposal, applies the proposed pose to the complete owner assembly,
and verifies unchanged frozen foreground triangles. Its physical job and
source/free-view renders must pass review before updating the canonical model.

For an occluded table, `--support-surfaces-json` maps the background entity to a
fresh `support_surface` proposal. The proposal binds its GS and sparse samples
by hash. The solver retains at least 90% of qualified GS and 95% of independent
sparse tabletop points, completing hidden edges as needed instead of shrinking
past observed surfaces. Both full point lists and resulting coverage remain in
the proposal. Support contact, other views and whole-scene appearance still
decide whether this inferred footprint is usable.

`scene.background.support_association` verifies and retains the source support
proposal while associating its sparse validation tracks with the observed GS
patch. Distant coplanar tracks stay in a separate audit; they cannot enlarge a
different tabletop simply because they project inside an occluded mask. The
association radius is bounded and at least three tracks must remain. This does
not resolve a mask that combines multiple objects: review and annotate the
actual surface part first, then fit its named exported meshes.

Captured-projection cutouts now default to `closed_shell: true`. The baker and
runtime both verify actual closed, consistently oriented geometry before
marking it one sided. This keeps an unseen rear's inward face from blocking GS
through an observed front opening. Camera-facing/self-visibility evidence is
unchanged; enabling this does not enable the more permissive `opaque_layers`
ray mode. An intentional open surface needs an explicit `closed_shell: false`.

RTX needs `singleSided` on those meshes as well as its face-culling renderer
setting; `doubleSided: false` alone is insufficient in the tested runtime.
The setting is recorded in the composed environment metadata and restored by
the normal render adapter. Composition rejects an external mesh already using
`singleSided`, so changing the environment cannot silently change frozen
robot/task rendering. Collision geometry remains complete. Check original and
novel views after rebaking; this prevents interior occlusion and does not approve
missing surfaces or the appearance of the resulting cutout boundaries.

Optional `shape_profile_checks` run on the exported background before a native
build can become current. They use agent-declared named surface parts, a front
axis and a lateral axis. `concavity` checks whether both edges curve toward the
declared front; `rear_support` checks that a named support reaches the surface
and stays behind its visible skin. These relations must come from source review;
they are not an automatic chair template or an appearance pass. The resulting
`shape_profiles.json` and actual asset are bound into the native manifest.

`front_view_checks` can name distinct `reference_parts` behind a front skin.
This tests a chair back against its rear spine instead of the centre of the
entire seat/base. Missing references fail; at least two registered front
photographs must still face the surface. The silhouette refinement rejects
orientation candidates that contradict these source front views. This does
not certify appearance or a complete object from a partial mask.

Agent assemblies should expose independent mechanical degrees of freedom.
For example, a swivel seat can change yaw without rotating the caster base;
rebuild and test the actual exported contacts after either change. Remove a
source-contradicted inferred support instead of inventing furniture to pass
physics. Any replacement hidden feet or post ends remain explicit completion
hypotheses and require gravity/contact and source/free-view checks. Frozen
task supports and robot assets stay unchanged.

Optional `texture_scale_checks` can be declared on an environment or variant:

```json
[{"asset": "carpet_patch", "tile_m": [0.35, 0.13], "relative_tolerance": 0.1}]
```

The builder checks texture repeat using actual exported triangle positions and
UVs, and rejects a patch stretched across the room before making the build
current. Up to 10% surface area is allowed for bevel projection. This tests an
explicit agent UV contract, possibly a source estimate or appearance prior; it
does not measure true material dimensions or approve appearance.

New hybrid builds require visible GS contribution. `--context-disabled` renders
the same delivered USD with native meshes visible and GS disabled.
`scene.review.gs_visibility` compares that image with the normal composite,
field-only and empty diagnostics. Photographic mesh matches and nonempty field
files alone cannot prove GS is visible. Captured/free views from at least two
source frames must show contribution. Pixel counts do not certify scene
quality. Physics CLI failures return nonzero and stop the environment pipeline.

The optional `scene.background.source_ray_gaps` proposal uses measured opaque,
sharp, same-owner source regions to decide native appearance cutouts. Each
surface sample needs two separated source rays (at least 50 mm camera baseline
and 2 degrees ray angle). Gaussian centres are not treated as measured physical
surfaces. The proposal preserves every retained field row, original mesh
geometry and collision; unseen outward surfaces retain native backing. This is
an appearance experiment. A better source image cannot approve novel views,
furniture completeness or the whole-scene gate. Collect additional reviewed
source regions when two views leave fragmented cutouts; do not certify an owner
with no usable source support.

`scene.background.material_fit` supports `probe_mode: "absolute"` for an
explicit constant colour parameter. Probe values are bounded to [0.001, 0.95];
the resulting `reflectance_rgb` is a proposal, and a boundary hit remains visible
in its report. Relative probes keep their existing [0.001, 8] gain range and
preserve texture detail. Absolute probes reject connected base-colour textures.
The actual candidate still needs source/free-view rendering; fitted RGB alone
does not identify a physically measured albedo.

`scene.background.mesh_radiance` calibrates a source-baked background atlas
through the current measured display response. Captured lighting stays baked;
these textures are not relightable recovered albedos. The diagnostic
`apps/scene/preview_source_atlases.py` attaches observed appearance to the
original exported mesh without changing vertices, polygons, semantic part names
or original UVs. Its pixel mixture preserves original PBR on unknown pixels and
reapplies existing GS cutouts. Outputs remain separate, unapproved diagnostic
USDs; this partial-atlas experiment is not a default native Blender build path.
The existing whole-mesh texture coverage requirement remains in force.

Normal environment builds can opt into that partial-photo path explicitly:

```json
"surface_atlases": {
  "background_owner": {
    "attachment": "usd_partial",
    "mesh": "owner_atlas_mesh",
    "radiance": "owner_calibrated_radiance",
    "observed_mask": "owner_observed_pixels",
    "evidence": "owner_atlas_evidence"
  }
}
```

All four values are logical IDs declared in `assets`. The evidence must bind the
exact current owner parameters, exported parts, calibration and protected task
state, and include measured display inversion. The builder retains the original
Blender geometry and part paths, attaches the source atlas after USD export,
packages the observed-pixel mask and records `partial_photos.json`. Unknown
pixels retain native PBR; richer connected materials retain their original
network where a constant-material mixture cannot preserve it. Geometry changes
invalidate the owner's atlas. This explicit partial mode does not weaken the
existing 25% whole-mesh atlas requirement. GS cutouts are applied later using
fresh coverage; partial photos neither remove GS nor approve appearance.

A Blender-generated mesh data suffix can change when a different owner is
added. Partial photo attachment can resolve a single mesh inside the same
explicitly named assembly part; it still verifies every vertex and oriented
triangle. Missing or ambiguous parts fail. A suffix change cannot authorize
moving a photo to another part or changed geometry.

Source-fit preview results may declare `fixed_parts` and an
`articulation_review`. The preview preserves those actual mesh parts while
transforming the remaining assembly. For example, a source-reviewed swivel
seat can change yaw while its caster base stays in the same world frame. This
is an unapproved pose hypothesis; contact, gravity stability and source/free
views still decide whether the agent should rebuild it.

Supported-prop fitting now carries reviewed image-edge visibility into its
solver. Occluded boundaries are lower bounds, and a candidate cannot win by
shrinking hidden geometry into the visible fragment. `source_visible_edges`
and `prior_guided_hidden_extent` expose the remaining uncertainty. Edge names
refer to image extrema: a hidden perspective corner may remove the top or
bottom extremum as well as the left edge. Inspect the original before labeling
those edges. These priors do not establish true dimensions or approve a prop.

If an old zone/semantic exclusion also masks a distinct background object,
`scene.background.owner_coverage` can make a supplementary coverage proposal.
The request names `field_review`, `empty_review`, and explicit `owners`, each
with `observations` and `background_identity_review`. Each observation must also
contain an `appearance_owner_review`. Supply the original context and measured
coverage plus an actual `frozen_scope` projection from that measured export.
A camera subset is accepted only for identical captured rays and unchanged
protection margins. The proposal intersects reviewed source masks with actual
frozen-geometry exclusions, measured opacity, source detail and GS contribution.
It does not edit old masks, field rows or task geometry, and leaves other owner
rows unchanged. Bind this new coverage, rebuild source-ray cutouts and repeat
composition/free-view checks. A source mask that misses the projected native
owner still needs geometry/identity repair; source pixel counts alone cannot
qualify a source-ray opening.
A frozen projection from a later material/background export may be reused only
when the actual frozen world triangles and instance paths are exactly identical
to those in the field measurement export. Captured camera rays and protection
margins must still match. A changed robot/support/object invalidates this path.

`scene.background.native_surface_gaps` prepares a narrower repair when a GS
subregion is sharp in captured views but distorts in free views. Its request
binds the current `render_review` and explicit `surfaces`: background `owner`,
semantic `part`, world `outward_axis`, `min_cosine` (0.8–1), `basis`, and two
complete captured/free-view evidence pairs. Only matching polygon facets become
opaque in that part's existing cutout atlas. Other charts and field rows remain
unchanged; overlapping atlas charts are rejected. For example, native top facets
can keep photographed vent detail while side facets continue to expose GS.
This changes neither geometry nor physical properties. Preview first, then bind
an accepted proposal and rerun actual hybrid composition, visibility and visual
checks. A proposal is never an appearance approval.

For dark surfaces, `scene.background.causal_coverage` also renders empty black
and white background controls. A usable ray must respond to the background
without GS, stay opaque with GS, and retain an unclipped colour channel. This
avoids rejecting real dark GS merely because it resembles the default empty
background; static foreground and clipped highlights cannot establish coverage.
The proposal still requires reviewed source ownership, source detail, matching
field rows, actual frozen-geometry exclusions and the existing opacity evidence.
Bind the coverage and rerun the composed source/free-view checks.

Use `scene.background.native_field_replace` when GS remains distorted over a
reviewed native part. First compare the same native geometry with GS enabled
and disabled, using two source views. The request identifies closed background
parts and independently reviewed source masks. The default selection intersects
source ownership with a bounded native surface distance. If a reviewed component
has unreliable depth, explicit `selection_mode: source_component` also requires
`bad_depth_review`; Gaussian centres must not be treated as measured surfaces.
Unselected kernel attributes remain byte-identical. The resulting field starts
with complete native backing and an unapproved appearance state: measure its
actual opacity/coverage, rebind cutouts, and repeat whole-scene GS preservation,
composition, physical and frozen-task checks before accepting it.

`scene.background.surface_ray_gaps` is the near-clipping version of the
source-ray cutout step. It clips triangles in camera space before projection,
so a large floor or wall crossing the camera plane keeps its visible surface.
It retains the two separated same-owner source-ray requirement and complete
hidden native solids. The earlier `source_ray_gaps` engine stays available for
reproducing its hash-bound historical candidates; new trials should use the
clipped projection. This does not change cameras or loosen coverage criteria.

When new observations qualify a previously native owner, an
`scene.background.extend_coverage` proposal can add only those openings. Its
request names up to eight current native owners, their `basis` and optional
`pixel_m`, plus the existing `appearance_plan`. It checks the actual measured
field, photometry and oriented world geometry against the current audited
export. Existing owners and cutout payloads remain unchanged, including local
facets deliberately kept native. After binding and physically checking the
proposal, use `surface_ray_gaps` on the added owners and render the result.
This path does not approve source/free-view appearance or change geometry.

Before changing shape to fix an appearance artifact, use
`scene.review.native_material_controls` on an actual composed review. It writes
private photographed/native material variants for named background owners.
Both restore those owners' complete opaque native surfaces; the native variant
disables only generated capture-atlas subsets and keeps intrinsic face
materials. Render each variant with GS enabled and `--context-disabled` to
separate texture projection, native geometry and residual GS artifacts. The
controls use the immutable rendered build and never edit the active scene.

`scene.background.view_opacity_gaps` adds a conservative free-view check to
existing source-ray openings. Supply actual same-field black/white field and
empty controls, plus a current physical/composed review. Each requested owner
names at least two captured/free-view pairs. The proposal projects the actual
self-visible native facets into those cameras. If a background ray reaches the
empty backdrop but GS does not remain opaque, that atlas texel becomes native.
Frozen foreground occlusion is inapplicable. Existing native texels and UV
charts stay fixed; this step can only close openings, never create them or
remove kernels. Repeat the composed source/free-view, physical and GS visibility
checks after binding it. Opacity alone does not establish sharpness or shape.
