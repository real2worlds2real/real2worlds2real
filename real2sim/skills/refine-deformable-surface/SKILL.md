---
name: refine-deformable-surface
description: Improve cloth-like asset realism using directional geometric folds, a simulation proxy driving detailed surfaces, and local material/clay/grazing-light review.
---

# Refine a deforming surface

Use when cloth-like surfaces look molded, scratched, uniformly noisy or lose convincing detail in motion. Infer the material and construction from the supplied images; this is not a garment or bedding recipe.

Separate three scales: silhouette and broad drape, intermediate geometric folds, and fine material response. Intermediate folds change surface shape, self-shadowing and sometimes silhouette. A normal map alone cannot supply those effects. Let visible seams, compression, stretch direction, attachments and contact motivate wrinkle direction and density; avoid covering the entire surface with independent random lines. A smooth interpolation modifier on a coarse mesh does not create missing fold structure.

## Simulation proxy and detailed surface

Use a proxy when simulating all visual detail is inefficient. A single mesh with detail modifiers after physics is also valid when it preserves the required shape and motion. Choose the representation rather than treating proxy use as a quality score.

- Mark the simulation driver `proxy['real2sim_sim_proxy'] = True`. Give it native CLOTH/SOFT_BODY, enabled in viewport and render. The host hides the object from rendering and excludes it from GLB/statistics, but still bakes and inspects its physics. Keep it available in the viewport for dependency evaluation.
- Drive the visual mesh using a bound Surface Deform (`modifier.target = proxy`) or Mesh Deform (`modifier.object = proxy`) modifier. Do not parent the visible mesh to a proxy mesh. Target topology must be suitable for the chosen modifier; for Surface Deform, remove doubles, degenerate and concave faces, and nonmanifold edges shared by more than two faces. Triangulation can resolve concave quads.
- With the visual object active and selected, bind Surface Deform using `bpy.ops.object.surfacedeform_bind(modifier=modifier.name)`. `modifier.is_bound` must be true before returning from `build_scene()`.
- Add meaningful intermediate shape to the visual mesh: authored or force-conditioned displacement, a subdivided rest surface, or a retained detail layer. Check whether subdivision/detail belongs before or after the binding for the chosen construction. Preserve its relationship to material UVs. Avoid a visible mesh so far from the proxy that interpolation distorts it.
- The host checks bindings, samples actual evaluated visual vertices, rejects a proxy whose visible followers stay static, and verifies both proxy and visual replay in a fresh process. This is motion consistency, not contact or material calibration.

Physical shape should have plausible support. Broad pinning can freeze a sculpt and suppress drape. For closed thick shells, native internal springs or pressure/volume controls are optional approximations; uniform air pressure is not a calibrated filled material. Choose and disclose a model that fits the visible behavior, without claiming unmeasured filling mechanics.

## Inspect actual surface evidence

`inspect_surface(attempt_id, center, size_m, view_direction, frame)` uses metres in world coordinates. The three `size_m` components define a positive axis-aligned region box; direction points from its center toward the camera. Use `frame=null` for the initial saved scene or an integer covered by a successful deformation cache for a later pose.

The tool returns:

- material under the source lights;
- neutral clay under the same grazing light, removing material color and normal-map relief;
- material under grazing light, revealing relief, roughness and aliasing.

Choose regions from current evidence: the shape feature that should read as a fold, an attachment/contact and a material patch are useful when present. If the clay image stays smooth while the material looks scratched, repair geometric detail rather than increasing normal strength. Compare initial and later motion when a detail could detach, flatten or stretch. Record concrete remaining differences from the reference; a successful render is not visual acceptance.

Keep a neutral whole-object view for diagnosis, and choose presentation lighting that reveals the material. Do not accept only a flattering close-up or hide missing geometry with dark lighting. Keep packed image maps and UVs for details that must survive GLB export; the editable .blend may retain richer procedural material.

References: [Surface Deform](https://docs.blender.org/manual/en/latest/modeling/modifiers/deform/surface_deform.html), [Blender cloth sculpt brush](https://docs.blender.org/manual/en/4.5/sculpt_paint/sculpting/brushes/cloth.html), [robotic cloth data pipeline](https://github.com/tlpss/synthetic-cloth-data). The last separates physical deformation and rendering; it does not imply that every robotics system uses a two-mesh proxy.
