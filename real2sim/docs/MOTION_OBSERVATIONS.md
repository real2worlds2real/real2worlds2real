# Preparing motion evidence from video

`prepare_motion_observations` turns existing measurements into timed comparison
tracks. It does not choose an object, invoke a tracking model, infer forces, or
decide which physical parameters to estimate. The external agent selects an
appropriate perception tool or supplies measured landmarks. Existing SAM3 masks
and an operator-owned `calibrated_scene_reconstruction` replay `track.npz` are supported inputs.

Call through `real2sim-harness task call <run> prepare_motion_observations --request <call.json>`
where `call.json` contains `{"request_path":"motion/request.json"}`. The referenced
workspace file has this shape:

```json
{
  "schema_version": 1,
  "video": "source.mp4",
  "video_sha256": "actual hash of the measured original video",
  "camera_motion": {"kind":"fixed", "source":"evidence for a stationary camera"},
  "clock": {"seconds_per_video_second":1.0, "rate_source":"capture/playback evidence"},
  "tracks": [{
    "id":"target.landmark", "source":"selected tracker and verification record",
    "image_size":[1280,720], "frame":"registered-camera-A",
    "observable":"landmark", "residual_scale":2.0,
    "samples":[
      {"frame_index":0, "point_px":[500,320], "sample_state":"observed"},
      {"frame_index":1, "point_px":[null,null], "sample_state":"missing"},
      {"frame_index":2, "point_px":[510,320], "sample_state":"observed"}
    ]
  }],
  "event_frames":{"release":2, "settle":null}
}
```

Paths resolve relative to the request. The task tool retains the source video,
archives and masks with hashes. Explicit original frame ordinals map measurements
to actual decoded presentation timestamps, including variable-rate video. The
importer does not substitute `frame_index / nominal_fps`, nor silently use a
tracker archive's nominal timestamps. At least two samples are needed per track.

## Existing tracker archives

Replace `samples` with:

```json
{"archive":"track.npz", "value_key":"centroid", "valid_key":"valid",
 "frame_indices":[0,5,10,15]}
```

Keep the other track fields. This is compatible with the replay track format
containing `centroid (N,2)`, `valid (N,)` and optional `score (N,)`. Every sample
needs its original video frame index; a selected stride must be mapped explicitly.
No perception weights or additional repository installation is required to import
the resulting archive. The producer's code is not copied or launched.

Nonfinite/invalid samples become missing; optional `min_score` excludes samples
below the agent-selected threshold. Scores are tracker diagnostics, not calibrated
probabilities. A source/association declaration is not independent proof of object
identity: inspect overlays and retain the tracker's original evidence.

The importer accepts image pixels only. In particular, a planar `xy` estimate
from a geometry model is not interchangeable with pixel `centroid`, even though
both have two columns. Retain metric estimates separately with their camera/scale
assumptions. Resized, cropped, rectified or rotated tracker coordinates must be
explicitly transformed into the displayed source-video pixel domain first.

## Masks and sparse image measurements

For mask-derived observations, set `observable:"visible_mask_centroid"` and use
`{"frame_index":N,"mask":"selected-mask.png"}`. Masks must have the full displayed
video dimensions; nonzero pixels mean foreground. Empty masks and masks touching
the image border are flagged missing. Explicit `occluded:true` also excludes a
sample. This does not automatically detect every occlusion or segmentation error.
Selected mask instances must already refer to the same target.

Point samples can be labeled `observed`, `missing`, `interpolated`, or `predicted`.
Only actual observed samples enter the physics comparison. No automatic smoothing,
gap filling, object association or numerical differentiation is imposed. The agent
may do additional analysis while preserving raw evidence and uncertainty labels.

## Time, camera and observable conventions

`seconds_per_video_second` converts decoded video time to physical seconds. For a
known 4x slow-motion presentation it is 0.25. The source of that rate must be
provided. With no clock evidence, timestamps stay in video seconds and the output
explicitly warns that the physical clock is unknown. Providing a source records an
assertion, not an automatic verification of the capture device or editing history.

Camera kinds are `fixed`, `registered`, or `moving_unknown`; document the evidence.
A moving camera with matched time-dependent simulated projection can still support
image comparisons. Unknown camera motion cannot be interpreted directly as object
world displacement. Registration transforms are not solved by this importer.

The comparison tool now checks matching `observable` labels when supplied.
**A visible mask centroid is not generally the projected center of mass, body
origin, or a persistent surface point.** Rotation/occlusion can shift the mask
centroid without corresponding translation. Compare it with the centroid of the
same rendered silhouette, or choose a persistent landmark and project that same
landmark in simulation. Labels express a contract; adapters must actually compute
the stated quantity. If using the Newton observer callback, implement the required
projection or silhouette measurement there rather than merely copying a label.

## Results and review

- `observations.json`: comparison-compatible tracks/events, time and camera evidence.
- `timeline.json`: source frame ordinals, decoded PTS and display rotations.
- `quality.json`: raw points, scores, exclusions and sample provenance labels.
- Full-resolution overlays at up to three selected times for quick inspection.

Overlays are samples, not full temporal review. Use the original video or dense
frame inspection where tracking errors, contact or occlusion may matter. These
tools prepare measurements; the agent still decides whether they support an
experiment and whether additional perception is needed.
