---
name: segment-reference-views
description: Obtain optional SAM3 pixel masks from photographs or selected video frames when boundaries, instances or geometric fitting need numerical evidence.
---

Use native media inspection to choose views and semantic prompts. A mask model
can supply boundaries for silhouette fitting, reference baking, support classes
or removing captured operation objects from a Gaussian background. Visual
understanding and final decisions remain yours; a simple asset may need no masks.

When uncertain foreground boundaries obstruct a twin comparison or numerical
fit, obtain and inspect masks instead of guessing the boundary. Reuse valid
supplied masks or simple task-local segmentation when sufficient; SAM3 is useful
for complex or ambiguous instances, not a mandatory invocation for every image.
Preserve openings and thin parts supported by the image. An outer instance mask
does not measure internal creases, seams or material boundaries; inspect those
in the original RGB and corresponding render as well.

The operator supplies a Python environment exposing Transformers `Sam3Model` and
`Sam3Processor`, Torch, this project, and locally authorized SAM3 weights. Run:

```bash
/path/to/model-python /path/to/real2sim/scripts/segment_reference_views.py \
  --request /path/to/request.json --output /path/to/new-output \
  --weights /path/to/local-sam3 --device cuda
```

Alternatively configure operator-owned `REAL2SIM_SEGMENTATION_RUNNER` using
[the example](../../examples/segmentation-runner.json), then call the shared
`segment_reference_views(request_path="masks.json")` tool in either transport.
It snapshots inputs and restricts model execution to the task plus the operator's
explicit runtime/weight read paths. Request JSON cannot select host commands or
grant filesystem access. See optional `requirements/segmentation.txt` and the
[official SAM3 interface](https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/sam3)
for a compatible isolated model environment. No model weights are bundled.
If a local model snapshot uses symlinks, the operator also lists its specific
backing model files/directories in `read_paths`, or supplies ordinary local files.

Request: `{"views":[{"image":"frame.png","time_seconds":0}],"prompts":["mug"],"threshold":0.3}`.

When a text prompt misses or confuses an object you can identify visually, add
per-view coarse spatial hints, rather than accepting the wrong instance:
`"prompt_boxes":{"mug":{"boxes":[[x0,y0,x1,y1]],"labels":[1],"box_only":true}}`.
Coordinates are displayed-image pixels in xyxy order, within that view's raster.
Labels1/0 indicate inclusion/exclusion hints; at least one inclusion is required.
`box_only:true` uses visual-box prompting while retaining the declared prompt
name as the caller's identity label; omit it to combine text and box hints.
Hints are approximate annotations, not a finished contour or measured geometry.
Inspect candidates and preserve supported structure before fitting.

Paths are relative to the request. Select chronological video frames with native
tools or reuse camera-recovery frames; preserve their actual pixel domains and
timestamps. The command loads local weights only and encodes each image once
across its prompts. Keep weights/accounts outside task evidence and shared Git.

Read `segmentation_report.json` and inspect returned masks. Distinct instances,
scores, centroids and per-prompt union images are retained. Choose the intended
instance rather than treating a union of several mugs as one mug. Compare the
selected boundary with the original RGB: shadows, touching neighbors, holes and
occlusion can make the predicted region differ from the intended object's
visible extent. Reject or mark uncertain portions; confidence alone cannot
settle that conflict. A mask does not supply metric dimensions, depth or a 3D
cross-section. Retain the visually justified structure when using it to constrain
model dimensions and placement; hidden geometry must not shrink to copy the
visible region. For fitting,
reference baking and support recovery, register the corresponding cameras and
convert masks to the receiving tool's explicit labels. Reject unsupported edges.

Mask acquisition alone does not verify a twin. For a consequential geometry or
layout question, carry the selected masks into the current-source projection
review in [fit-multiview-geometry](../fit-multiview-geometry/SKILL.md), then use its
measured discrepancy to choose a repair. A failed or ambiguous prompt requires
identity/boundary resolution or a concrete limitation; it cannot be replaced by
an unrelated point residual and marked complete.

`track_id` is a conservative adjacent-mask IoU proposal within each prompt.
Ambiguous matches get new IDs and an ambiguity flag; missing detections or large
camera changes can break identity. This is not a learned video tracking model.
Use geometry/appearance evidence to resolve identity when needed. Mask confidence
is not calibrated geometric accuracy, and absence of a mask is not proof of free
space. The command never accepts or modifies a simulation asset.
