---
name: inspect-video
description: Optional time-indexed frame inspection for reference or simulation videos when the current client needs exact frame, interval or crop evidence; native media viewing remains available independently.
---

# Inspect video

Prefer the current coding client's native media abilities when they satisfy the
task. This skill documents the optional repository frame reader used by the
managed transport or direct Python callers. The external-agent CLI returns media
paths and does not require this reader or register it as a duplicate viewing tool.

Use `list_video_sources()` to obtain stable IDs for original reference videos, registered review movies and rendered simulation candidates. Original streams are retained; initial extracted reference images are an overview, not the complete evidence. When the input is a video, actively inspect it to understand the target before committing to a model.

`inspect_video` accepts `video_id`, `mode`, `start_seconds`, `end_seconds`, `sample_fps`, `frame_indices`, `crop`, and `max_frames`. Supply null for unused fields. Modes:

- `overview`: choose up to `max_frames` distributed original frames to understand the object, changing viewpoints and stages of motion.
- `interval`: inspect `[start_seconds,end_seconds)`. With `sample_fps=null`, every original frame in the interval is returned. Above the requested frame budget, the tool asks you to split the interval rather than silently sparsifying it. An explicit positive `sample_fps` is available when dense viewing is unnecessary.
- `frames`: inspect exact original `frame_indices`. Use this for comparisons, occlusion transitions, contact failures and detail.

`crop=null` keeps the full image. A crop `[left,top,right,bottom]` uses normalized displayed coordinates. Frame/crop requests return full-resolution images in addition to ordered contact sheets. Each image records its source frame and presentation timestamp; the source hash ties evidence to the movie. Overview samples do not establish absence of short-lived defects. Use dense adjacent intervals when transient motion matters. Audio is outside the current tool.

For twin modeling, use changing views to resolve hidden surfaces, seams, part connectivity and relative proportions. Observe which parts move together, their pivot/slide direction and limits, cloth attachment locations, and deformation around contact. Separate camera movement from object movement; video alone does not give measured scale, stiffness or calibrated physics. Geometry/camera tools remain optional when they resolve a specific uncertainty.

For complex dynamic candidates, consider `render_motion_video(attempt_id, assumptions)` after simulation and appearance checks look ready. Hybrid assets and Newton cloth replay their retained target trajectories. Rigid/articulated PhysX assets use your chosen operation in workspace `motion_plan.json`; read the workspace `physics_motion_review.md` for that schema. The tool returns a final candidate video ID for this same reader and does not accept delivery. Avoid full movie rendering for every draft. Other returned/registered videos use the same reader irrespective of their renderer or simulator.

Useful checks include both sides and meaningful stages of the requested operation. Choose the checks needed for the case; if uncertainty warrants it, cover the whole movie using adjacent dense intervals. For exposed internal parts, crossing cloth, detached details or discontinuities, read `review-dynamic-contact`, retain the failing timestamps/frames and return to source/simulation correction. Final acceptance requires the selected target-simulator checks and your evidence-based judgment.

## Account for the reader's blind spots

This repository reader returns decoded ordered images. It does not provide native MP4 input or continuous playback. Do not assume equivalence to another application's video handling. Access to the original movie or all its frames does not mean you have inspected them. Use the reader's timestamps, selected indices and detail settings to know what evidence you actually received.

Before judging a reference or candidate, consider which plausible uncertainty the present view could hide. The following are investigation prompts, not a mandatory checklist or evidence that a defect exists:

- **Temporal gaps:** Sparse overview or 1–2 fps samples can miss a brief contact, frame-to-frame flicker or rapid flutter. An apparently still sequence can be an oscillation sampled at similar phases. Dense adjacent original frames can help; timestamps are necessary to judge speed and duration.
- **Spatial detail:** Small contact-sheet tiles can hide a thin mechanism, narrow gap or local penetration. Use full images or a crop when those features matter. A crop omits the surroundings, so retain enough context to understand what moves relative to what.
- **Incomplete phases:** A good endpoint or successful operation does not establish a natural trajectory. For flexible assets, settling, lifting, release and a stationary-controller hold may reveal different problems. Inspect free regions as well as the grasp. Do not explain persistent flutter as wind or deliberate shaking without configured excitation or other evidence.
- **Attention and reference ambiguity:** Looking only at the operated object can miss unstable background surfaces. Camera motion, occlusion and changing object configurations can also masquerade as object motion or inconsistent geometry. Compare useful views and intervals before deciding; distinguish what was observed from what remains hidden or assumed.

Choose follow-up viewing and analysis for the task rather than waiting for a user to name the failure. If ordered images leave motion uncertain, native task access lets you inspect saved states and write an analysis of displacement, speed, decay or control timing. Camera-space image differences need motion compensation or known geometry before attribution to a physical or shader defect. A low-stretch or finite-state report does not establish natural motion, and numerical analysis cannot replace appearance review. Read `review-dynamic-contact` and its lift-motion reference when that investigation applies.

Report the actual scope of the conclusion: sampled overview, a dense interval, local detail or numerical trajectory evidence. Keep material uncertainty open rather than labeling an inadequately inspected movie clean. Neither dense frames, a program score nor these reminders guarantee that every perceptual defect has been found.

`detail=null` automatically returns full decoded images for dense interval or exact-frame requests of up to 24 frames, in addition to overview sheets. Use `detail='full'` to request this explicitly, or `'overview'` for cheap thumbnail-only viewing. Split full-detail requests above 24 frames. Large contact sheets alone can hide thin rods, seams or brief occlusions; native-size continuous images and crops are available without reducing the retained source resolution. Captions pair each returned full image with its exact frame and presentation time.

When selected views would help VGGT, multiview geometry or camera alignment, `promote_video_frames(video_id, frame_indices, reason)` adds those uncropped frames to the run's numbered references and returns their indices. This is optional. For a rigid-scene geometry prior, choose useful camera viewpoints while the object's configuration stays consistent; different door angles or cloth deformations must not silently become one rigid reconstruction. Preserve timestamps for reasoning about function.
