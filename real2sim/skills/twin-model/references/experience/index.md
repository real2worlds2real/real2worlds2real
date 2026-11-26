# Reconstruction experience modules

Read only the modules relevant to the current object and interaction. These are
composable references for `twin-model`, not additional standalone skills or
fixed asset templates. They preserve methods learned from earlier work; their
numbers, fitted assets and old approvals are not inputs to a new independent
reconstruction. The phase's source, semantic and native gates still apply.

## Select and compose

| Stable module id | Triggers | Reusable method | Compose with |
| --- | --- | --- | --- |
| [open-vessels](open-vessels.md) | Cup, bottle, vase, hollow container | Material cross-section, wall/base/cavity, mouth and hidden-construction uncertainty | `rigid-insertion` for nesting |
| [rigid-insertion](rigid-insertion.md) | Peg/hole, shaft/bore, nesting, axial assembly | Functional axes, observed aperture, swept clearance, seating and released state | `open-vessels` or `gears` for the participating geometry |
| [gears](gears.md) | Toothed wheel, gear train, gear mounted on a post | Tooth-count hypotheses, compatible pitch, real tooth profile, bore and meshing checks | `rigid-insertion` for post mounting |
| [threaded-closures](threaded-closures.md) | Screw cap/neck, threaded engagement, tightening or release | Mating helical profiles, finite runout, contact-derived lead and bounded-wrench trials | `open-vessels` for bottle/cap shell geometry |
| [hinged-lids](hinged-lids.md) | Attached cover, opening, trapped sheet and extraction | Source joint state, loaded/unloaded seating, finite opening and released material | `open-vessels` for the container geometry |
| [liquids](liquids.md) | Included liquid, visible fill boundary or separate spill | Native particles, volume/mass mapping, material priors and visible surfaces | `open-vessels` for the physical cavity |
| [flexible-bonded-surfaces](flexible-bonded-surfaces.md) | Attached leaf, petal or sheet | Separate material rest shape, attachment frames and owner contact in a release trial | `observed-surface-depth` for the initial shape |
| [observed-surface-depth](observed-surface-depth.md) | Thin material with sparse depth or occluded tips | Check sample ownership, local depth bounds and held-view reprojection | `flexible-bonded-surfaces` for a bonded part |

For a room or task-area distractors, see the environment skill's [spatial cousin review](../../../reconstruct-environment/references/experience/spatial-cousin-review.md).

Examples:

- A new nesting cup: `open-vessels` → `rigid-insertion`. Verify the whole insertion
  path and uncertainty in the hidden walls, not just the two rim diameters. If the
  nested cup does not come to rest natively, read the self-locking taper section
  of `rigid-insertion` before changing solver settings.
- A new peg: `rigid-insertion`. Infer the opening and shaft from this source;
  unseen hole depth is a construction hypothesis.
- A gear on a post: `gears` → `rigid-insertion`; add meshing checks only when the
  task includes another gear. Mounting and meshing are separate claims.
- **Depends on: `threaded-closures`.** A screw closure combines vessel geometry with mating-contact checks; plausible shell appearance does not establish screw contact or retention.
- **Depends on: `hinged-lids`.** A lidded container with a free sheet needs source state, opening, material retention and extraction checks.
- **Depends on: `liquids`.** Liquid inside a vessel needs the actual cavity and material, native volume/mass evidence and the combined render.

Composition adds obligations; a passing submodule does not approve the assembly.
Share the current capture's scale and functional frames across modules. Resolve
contradictory assumptions before fitting. Do not let a downstream task result
silently change an upstream source-derived dimension.

## Module contract

Each module uses the [module template](module-template.md): identity/scope,
observations, representation and parameter relationships, procedure, checks and
failure diagnosis, implementation/evidence, and reuse limits. Keep criteria and
diagnostics near the failure they explain. Cite code for implemented methods;
label proposals and limitations explicitly. An isolated geometry or CPU test
must not be described as native physical success.

Current entries are **source-backed procedures** distilled from maintained code
and instructions. This organization does not establish new physical validation
or cross-capture generalization. Session-specific pointers document provenance;
do not open their fitted parameters or QA as input to an independent capture.
Prefer the distilled procedure and shared library entrypoints.

## Accumulate new experience

1. Match the mechanism/representation first. Improve an existing module when
   the new case changes a diagnostic or supported setting; create a module only
   for a distinct reusable method. Do not create one module per captured object.
   Capture/environment/review experience can use the same template beside its
   owning phase skill; link it from that phase rather than forcing it into an
   object-modeling module. This initial index covers object and mechanism work.
2. Use [real2sim-evolve](../../../real2sim-evolve/SKILL.md) for development and
   evidence. Add `experience_module` to the ledger's `context`, using the stable
   id above, alongside relevant mechanism, backend/version or observation
   settings. It is a conventional string context key, not a new CLI option.
3. Keep candidate findings and case-specific values with their experiment.
   Distill the reusable relationship, diagnostic, failed approach and bounds
   into the module after validation. Include the module and this index in the
   proposal's `implementation` when changed, and the module in adoption
   `entrypoints`; final checks must bind the actual published instruction text.
4. For composite changes, use the primary changed module in `experience_module`
   and document the combination in this index. Pin all changed references
   and required checks; a single adopted record is not blanket approval of
   every component. Separate records are useful for independently reusable claims.
5. Reuse by reading the module and retrieving matching records with `recall`.
   Check applicability, counterexamples and current evidence status, then
   re-estimate from the new capture and validate its affected scene gates.
   A fresh clone may have the recipe without local ledger artifacts.

For example, a context file containing `{"experience_module":"gears"}` selects
local gear records through the existing `recall ROOT CONTEXT.json` interface.
For older records without this key, search their existing `mechanism` or other
context tags. Never rewrite old proposals just to migrate their labels.
