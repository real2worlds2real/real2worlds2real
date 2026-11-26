---
name: improve-visual-realism
description: Review and improve reference-based Blender appearance, including local shape and normal errors, materials, markings, lighting, and exported appearance.
---

# Improve visual realism

Use this skill during draft and final reference review, including shape and
proportion diagnosis. Reference-based appearance is a delivery requirement;
having the required parts or a structurally valid asset does not establish it.
Compare useful draft views early, choose tools for demonstrated discrepancies,
and preserve supported structural priors when revising geometry. Keep fast
`run_blender` renders for iteration; reserve `render_final` for promising candidates.

If viewpoint mismatch prevents a useful comparison, first consult [align-reference-camera](../align-reference-camera/SKILL.md) or the available calibration evidence. For observed printed/surface detail, consider [bake-reference-texture](../bake-reference-texture/SKILL.md); for cloth-like folds and surface relief, use [refine-deformable-surface](../refine-deformable-surface/SKILL.md). Compare reference, retained candidate and revision under corresponding views: keep an earlier candidate when a change improves one cue but worsens the requested overall appearance. Return the selected source to the task's target-render and acceptance checks; changing material or lighting alone is not evidence of improvement.

Check the reference's identity-bearing visible features individually, including
features inside the outer silhouette. A strong overall resemblance or mask IoU
can conceal a missing crease, rounded transition, seam or opening. Use a local
reference/render comparison when a whole-object view cannot resolve the feature.
When its cause is uncertain, choose a diagnostic such as a neutral clay view,
grazing light, wireframe or normal visualization, or a controlled flat-versus-
smooth shading comparison. Keep camera, geometry, material and lighting fixed
except for the variable being tested.

Separate mesh shape from shading normals: smooth normals interpolated across a
real slope break can make an angular profile look rounded even though its
vertices are correct. Preserve reference-supported creases using appropriate
sharp edges, split normals or controlled bevel geometry while keeping curved
regions smooth; do not apply one angle or global flat shading to every asset.
After repair, compare the actual appearance again and inspect the exported
target-rendered feature. A diagnostic image is not updated delivery evidence;
follow the normal affected-check and acceptance requirements.

Describe surface identity from visible cues before assigning material labels:
highlight width and contrast, spatial variation, pits or facets, machining
direction, coatings and supported wear. Different parts of a metal assembly may
have different finishes. A silver metallic base with one roughness value does
not reproduce a rough forged/coated body and smooth machined parts. Distinguish
manufacturing texture from age; do not invent rust, dirt or scratches to make a
surface look old. Choose the feature scales that matter at the reference view,
then use closeups to diagnose unresolved detail.

For a suspected transfer failure, compare the editable-source render and target
render using corresponding geometry, cameras and controlled lighting. Inspect
the actual exported normal/roughness/color connections and texture dependencies.
If a procedural detail disappears, preserve it through a supported target graph
or bake it to UV textures in the editable source and recheck. A library export
success or a normal-transfer receipt establishes neither matching roughness nor
reference fidelity. Improving light or increasing samples cannot restore a
discarded material input.

For a lost procedural normal input, read
[portable surface detail](portable-surface-detail.md) for the native Blender
baking route and the limits of the export check.

For radial wedges or starbursts on an end cap, inspect its UV coordinates
separately from cylindrical side mapping; a singularity at the cap center can
distort a sound texture. Check the repaired top/detail view in both renderers.
For floating trim on a tapered body, compare its attachment to the body profile
at the trim's actual height. A radius copied from the mouth need not fit lower
down. When contents distort only through a transparent shell, compare a diagnostic
view with that shell hidden while holding camera and contents geometry fixed;
localize optics/normal errors before reshaping the contents. Restore the shell
for source-matched delivery checks.

- Match visible material classes with physically meaningful Principled BSDF parameters: base color, roughness, metallic response, transmission and IOR. Add subtle, physically scaled procedural variation only when the reference supports it.
- Respect the requested appearance provenance. When a task requires agent-generated textures, author them in code or the permitted generation path; external photographs may guide observation without becoming the surface image. Do not silently substitute a downloaded material library or describe borrowed appearance as generated. Distinguish style references, the user's observed input and generated completion.
- The host adds default lights only when `build_scene()` creates none; source-authored lights survive both semantic and final rendering. You may author reference-supported key/fill lighting in source when a flat default setup hides relief. Keep the independent local clay/grazing views for geometry diagnosis instead of accepting only flattering presentation light.
- For portable image textures or normal maps, create the image in the agent workspace, load it from an explicit workspace path and call `image.pack()` before saving. Submitted source is copied into an immutable attempt directory before execution: paths relative to the submitted script's `__file__` therefore do not point back to workspace assets. Use image textures with explicit UVs, appropriate color spaces and Normal Map nodes when detail must survive GLB or USD export. Procedural Blender graphs can become constant-color surfaces in the target. Inspect the exported material connections and retained texture files, then target-render a relevant view rather than inferring preservation from a successful export or a Blender preview.
- Judge grain direction and scale on the actual furniture, not just the texture image. Uniform stripes, exaggerated normal relief and identical roughness across wood, paper, fabric and paint can look synthetic even when maps export correctly. For rooms, useful material/lighting references include credible daylight gradients, contact shadows, restrained highlights and clear glazing. Increasing overall brightness alone does not restore surface identity or correct an unsupported glass shader.
- At beveled room-shell junctions, nominal edge contact between wall bounds can leave a real opening. Check exported surfaces and intentional openings; hidden solid overlap can close the shell while preserving interior dimensions. Keep decorative and backing surfaces separated where coplanarity would cause flicker.
- A bright polygon seen through glazing may be the visible area-light emitter rather than the exterior. In RTX, distinguish primary-ray visibility from reflected/refracted visibility: `visibleInPrimaryRay` alone may be insufficient, while the per-light `ghostFactor` controls those other paths. Preserve illumination and record any renderer-specific USD override with the delivered environment; verify it through the actual glass and camera.
- Model geometry for silhouette, contact, seams and interaction-critical features. For dense repeated details that occupy few pixels, prefer one optimized mesh, linked instances, or a shader/normal pattern over hundreds of separate objects; watch for aliasing and moire.
- Preserve visible labels and dial markings without inventing unseen branding. Make them large and contrasted enough to inspect, and convert text or curves to exportable meshes before the final GLB export.
- Inspect a close-up of controls or other identity-bearing details. Set `obj["real2sim_detail"] = True` on the geometry you want grouped into the final detail view; choose targets from the current reference evidence rather than a fixed object category. A clean whole-object render can conceal missing text, flat materials, floating trim, or poor glass response.
- Call `render_final` at 1024/128 first. Use 1536/256 only when the extra samples and resolution answer a visible quality question. Inspect every returned hero view and the detail close-up before acceptance.

Rendering can remove sampling noise but cannot repair wrong geometry, missing texture, implausible roughness, or unsupported inferred details. Keep hidden surfaces plausible and disclose single-view uncertainty.
