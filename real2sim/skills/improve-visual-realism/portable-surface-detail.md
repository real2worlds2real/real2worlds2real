# Preserve procedural normal detail in USD

Use this route when the source's required normal/bump detail is absent in target
USD. First establish that the editable-source render actually contains the
desired detail. Baking an inadequate source material only preserves that
inadequacy. Keep shape, surface construction and material-transfer findings
separate.

The normal-transfer receipt names affected source materials. The standard USD
exporter does not translate arbitrary procedural Noise/Bump graphs. Blender's
native Cycles normal bake can convert supported source normals into a tangent
normal image that the normal exporter can carry through an Image Texture and
Normal Map node.

- Retain a source revision. Select the affected visible meshes and identify all
  participating material slots. Give their surfaces adequate UV coverage without
  overlaps. Multiple objects sharing an atlas need a joint UV layout; independent
  full-square UVs would overwrite each other's bake. Preserve existing texture
  mappings and geometry/part hierarchy. Use a separate UV set when necessary,
  explicitly link that set in the normal graph, and verify target support.
- Configure the authorized Cycles GPU and choose resolution from feature scale
  and coverage. Add an Image Texture node for the destination image to every
  participating material, make it the active node, and keep the original
  normal/bump connection in place while baking. Mark normal images Non-Color.
- With the intended meshes selected and one active, use
  `bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', margin=margin_px,
  use_clear=True)`. Inspect the resulting image for coverage and meaningful
  normal variation. Do not assume a successful operator produced the right map.
- Save and pack the image. After baking, connect Image Texture Color to Normal
  Map Color, then Normal Map Normal to Principled Normal. Use the UV map that
  produced the bake. Keep the source graph/revision available for correction;
  repeatedly baking an already baked graph can accumulate errors.
- Save the editable source and re-export. Check the normal-transfer receipt,
  texture dependencies and actual target-rendered feature. Compare the same
  source/target view, and use a controlled normal-connected/disconnected target
  comparison when the effect is uncertain. Do not adjust lighting or geometry
  between that control pair.

This route transfers normals only. Roughness/color variation, coats, displacement,
glass optics and other shader behavior need their own supported representation.
The receipt verifies a connected normal texture path and its retained files; it
does not establish UV correctness, shader equivalence or reference fidelity.
