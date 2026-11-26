# Source part labels

An object-level semantic query can merge different materials, such as petals
and their wooden stem. Inspect the original RGB beside the source labels before
changing geometry. Use `scene.objects.part_label_review` to acquire a clearer
query or source-box proposal, then review every connected component.

A supplemental query may repair some frames and miss others. Preserve both
queries and select one complete source proposal per frame with
`scene.objects.part_label_combine`. Its config-relative JSON contains `version:
1`, `object`, the original `prompt`, and a `frames` mapping:

```json
{
  "version": 1,
  "object": "target",
  "prompt": "target material",
  "frames": {
    "frame.jpg": {
      "proposal": "qa/source_parts/target/round_2",
      "basis": "Original RGB separates the target material from its rigid holder."
    }
  }
}
```

Include every currently used frame. Choose labels by physical identity in the
source, never by candidate geometry, silhouette overlap or fit scores. The
combiner rejects missing views, empty chosen labels, changed acquisitions and
mixed image dimensions. It retains source proposal hashes and emits new
component sheets with a pending review. Review those sheets and bind the final
manifest/review hashes through `part_review` in the physical observation as
usual. Whole-object observations and source acceptance thresholds stay unchanged.

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.objects.part_label_combine \
  sessions/<session> target "target material" agent/part_choices.json
```
