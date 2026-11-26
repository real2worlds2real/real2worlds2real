# Background rendering and whole-scene acceptance

Use `scene.review.cameras` to fix captured and nearby novel review cameras, then
`apps/scene/review_usd_cameras.py` to render the actual exported USD. Review
cameras do not alter video sampling or training inputs. Novel views have no
paired ground truth. `scene.review.background_scores` scores matched cameras
with independent background labels and geometric rectification validity.

For new full-environment sessions set `environment_quality_version: 1` in
`twin.json`. Submit a review through `scene.review.environment`. All twelve
criteria need explanations and current rendered evidence: shape, materials and
texture scale, lighting, contact shadows, occlusion, layout, view consistency,
detail, coverage, ghosts, task preservation and export rendering. This applies
to GS, Blender and hybrid environments, including cousins. Passing object
geometry or a PSNR threshold cannot approve the whole scene. Changed cousins
are judged for coherent realism, not pixel equality to the original capture.

Gaussian coordinate conversions must transform view-dependent SH coefficients
together with positions and covariance orientations. `splat.radiance` uses the
renderer's own SH basis when undoing training normalization; DC, density and
geometry fields remain unchanged. Export snapshots include this implementation
and declare `radiance_frame: colmap` in selection metadata.

NuRec assets may declare photometric settings in root-layer `renderSettings`.
The simulation adapter applies only the supported tone-mapping/compositing
settings and records them in `r2s2r_background_render`. Export and review apps
restore this profile when opening an asset. Existing assets without a profile
retain the earlier fixed-exposure behavior. Renderer and physics settings are
not imported through this adapter.

Re-export preserves prior owned assets under `scene/splat/asset_history/` and
prior selection evidence under `splat/export_history/`. Cousin folders are
preserved. Historical renders remain bound to their original hashes; replaying
a missing diagnostic asset creates new provenance, never a claim of identical
bytes or an altered timestamp.

The current 60k/3M peg-background validation candidate improves photometric
agreement after these corrections but still fails whole-scene review for blur,
disocclusion ghosts and incomplete boundaries. See that session's
`qa/background_diagnostics/production_sh_reframed_3m/report.md`. No photorealistic
delivery is claimed. Frozen task zones remain unchanged.
