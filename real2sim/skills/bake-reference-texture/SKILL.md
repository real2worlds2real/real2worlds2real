---
name: bake-reference-texture
description: Transfer observed printed patterns, wear or surface colors from registered photographs into a generated mesh UV atlas with visibility and coverage evidence.
---

# Bake observed appearance

Use `bake_reference_texture` when actual RGB observations contain useful surface detail that procedural materials do not preserve. It is optional for both individual objects and scene meshes. Reflective or translucent surfaces may benefit more from authored PBR materials than from baking photographed lighting.

Read [the request contract](../../docs/OBSERVATION_TOOLS.md#observed-texture-baking). Supply the mesh in the same frame as actual undistorted cameras, nonoverlapping triangle-corner UVs, and posed RGB views. Masks/ignore regions or observed depth can exclude hands, clutter and other external occluders; mesh self-visibility alone cannot detect them.

If UVs are missing, `uv_mode: auto` uses the optional mature xatlas backend; `unwrap` forces a new atlas. Geometry/triangle order is retained. Existing Blender UVs can be preserved with `existing`, and the Blender bridge also offers Smart UV Project plus a matching editable source copy. For sparse coverage, optional `fill_unseen` generates a separate completed atlas; its new pixels remain inferred and do not increase observed coverage.

Call `bake_reference_texture(request_path="texture.json")`. Inspect the atlas and coverage preview, use the matching UV layer when importing it into Blender, and preserve explicit handling of transparent/unseen texels. Unseen completion is inferred appearance. The source photos are retained and hashed, so you can revise the inputs and rerun without losing provenance.

Check printed details, seams and coverage against the references. This operation transfers photographed color and lighting; it does not recover intrinsic albedo or certify materials/physics. Geometry or camera errors should be corrected rather than hidden by increasing visibility tolerance. Final simulator review remains separate.
