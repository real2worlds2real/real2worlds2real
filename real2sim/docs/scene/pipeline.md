# Scene reconstruction

Start with the [reconstruction workflow](../agent/workflow.md) and the [twin stages and gates](twin_pipeline.md).

1. Extract registered views from the capture. Estimate camera intrinsics and poses, segment the task objects and support, and recover metric scale from available calibration evidence.
2. Reconstruct the support and objects, preserving their relative poses. Compare each candidate against source views. Iterate on its structure, materials and contact geometry.
3. For a full scene, assemble the surrounding environment using the selected mesh, splat or hybrid route. Verify multi-view appearance, scene bounds and support relationships before freezing the workspace.
4. Run the native scene and task checks, inspect the evidence and export the accepted scene with its scale provenance. Use the frozen source when building digital cousins.
