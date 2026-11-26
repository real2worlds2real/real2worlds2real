# Module contracts and nested loops

Every module receives the complete intake contract, its additive checks and the
current accepted predecessor artifacts. It returns immutable artifacts, numerical
results, semantic review, unresolved failures and implementation/input provenance.
The agent owns decisions; command execution and gate acceptance are separate.

| Module | Configure from the task | Local loop and acceptance | Failure destinations |
|---|---|---|---|
| Research | Known capabilities, new categories/mechanisms/models/tasks, transfer risks | Inspect implementation → competing hypotheses → online search of related work and primary sources → minimal controls → integrated validation → renewed research when evidence exposes a gap; record sources and limitations | Research or the implicated module |
| Capture | Source type/coverage, scale authority, robot/board route, views, motion/static exclusions, HDR | Input review → keyframes/labels → camera/metric fit → source overlays/held-out checks → revised labels/calibration; S0/S1 remain native | Capture; request only unavailable essential source information |
| Model | Required semantic parts, critical cavities/contact geometry, rigid/articulated/deformable representation, bounds/priors | Source understanding → parametric model → multi-start fit → every-view evidence → neutral structure/part/contact review → revised model; OBJ/SUP gates | Source identity/calibration, model or representation research |
| Physics | Native backend/profile, uncertain material priors, actual task interaction and controls | Passive/material/contact controls → full assembled task → nominal/low/high cases where required → unforced release → review; preserve raw failures | Model, physical assumptions, numerical representation or task controller |
| Environment | Twin/cousin role, protected task assets, background route, deployment distribution | Generate/build → protected-asset audit → assembled rendering/physics → multi-view review → refinement; no task changes to improve background | Environment, source understanding or rendering backend |
| Twin | Applicable scene gates, package scope, current source/model/physics | Recompute native S0–S6 and OBJ/SUP → package reload → freeze and verify | Exact stale/failing upstream module |
| Data | Expert, episode/coverage target, cousin mix, DR, sensor/controller model, failure policy | Small expert/record/export probe → planned collection → label/alignment/split/coverage QA → targeted recollection; actual usable count from manifest | Expert/task adapter, environment, data/labels or native asset |
| Train | Model/initialization, dataset mixture, augmentations, normalization, schedule, compute | I/O round trip → short train/rollout → full training → validation-selected checkpoint → deployment parity | Dataset/label pipeline, model/optimization or controller |
| Validation | Fixed development scenario matrix, perturbations, metric/statistic, latency | Closed-loop rollout → all-stratum accounting → causal failure analysis → transfer-oriented feedback → affected modules | Model/physics/environment/data/train/research, with explicit rationale |
| Holdout | Untouched groups/seeds, same policy/control/checkpoint and success definition | Execute once → every declared case and stratum → fixed statistic/transfer checks; preserve failures | New experiment with new holdout; no in-run tuning |
| Delivery | Self-contained policy/runtime/assets, observations/actions, residual gaps | Reproducibility and interface checks → immutable package → human deployment procedure | Packaging or unresolved module; never claim real success from sim |

## Fidelity customization without acceptance drift

Use task sensitivity to allocate effort. A cosmetic background feature and an
insertion cavity do not need the same precision. For critical concavity, require
source-supported opening/interior geometry, preserved free space, actual cooked
collider fidelity and native contact tests at the operating scale. An outer
silhouette or convex hull is insufficient. State unresolved hidden geometry as
uncertainty and test alternatives; policy reward is not a dimension measurement.

Task-specific tolerances must be tied to observability and real contact/control
requirements. `geometry_error_m` and `contact_error_m` require a documented
measurement definition, units, evidence, and uncertainty; a silhouette IoU cannot
be relabelled as a metric error. If the required precision is unobservable, retain
the gap and choose an informative experiment instead of inventing accuracy.

Existing flexible `task_and_semantics` versus `quantitative` policies remain
explicit. Approximate secondary appearance/response does not permit invalid
material, primary contact errors, missing structure or a rigid replacement for
a flexible task mechanism. Scoped relaxations retain raw/default and effective
verdicts and require their existing proofs.

## Feedback and invalidation

A feedback revision names the earliest affected modules and retains:

```json
{
  "targets": ["environment", "data"],
  "diagnosis": "Development failures cluster under partial camera occlusion.",
  "transfer_hypothesis": "The current scene distribution omits plausible deployed occluders.",
  "expected_real_benefit": "Improve recovery under deployment occlusions without privileged observations.",
  "anti_hack_review": "Preserve task geometry, physics and success criteria; add realistic difficulty.",
  "ablation_plan": "Compare fixed validation scenes with and without justified occluder variation.",
  "evidence": ["/absolute/path/to/failure-analysis.json"]
}
```

The controller invalidates targets and descendants and retains previous attempts.
It does not erase failed evidence, reuse old acceptance under new inputs, or
automatically change thresholds. New attempts may use the same local refinement
tools; acceptance still requires their own local checks and current artifacts.

A development evaluation can legitimately trigger simulator improvements. Direct
new real measurements are not mandatory for every such change: physical theory,
source evidence, deployment specifications, independently qualified controls and
uncertainty coverage can justify it. The expected benefit must be transfer, not
an easier simulator. See [the transfer criteria](transfer.md).
