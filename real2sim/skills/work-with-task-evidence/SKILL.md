---
name: work-with-task-evidence
description: Common Real2Sim working methods for understanding the task, investigating source and retained evidence, and judging completion in an external or managed coding session.
---

# Common working methods

Read these common methods at task startup or continuation. External task startup
includes them in `external/startup.md`; the optional managed entry injects them
at startup and resume. The user's objective and explicit constraints define the task. Follow the current task mode and its
required finish; choose specialized skills only after the relevant task or
uncertainty is known. Instructions and tool descriptions do not grant access
beyond the execution environment's actual permissions.

## Investigate without waiting for a new tool

Read the supplied task context (`external/context.json` or `project_access.json`). It gives
the readable framework source, this task's records, writable workspace and
analysis environment. In a native coding session you can search files, read
source/configuration/logs, inspect saved numerical states and write/run your own
analysis programs. Use those capabilities when a report is insufficient; a
dedicated symptom-specific tool or an operator-prepared explanation is not
required. Some transports expose only function tools: use the capabilities
actually available and do not claim native shell access there.

Sources and task records are available for on-demand investigation, not an
instruction to read everything. Follow a concrete uncertainty to the relevant
input, script, effective configuration, output and timestamp. Keep analysis and
editable modeling code in the workspace. Use installed dependencies and the
declared task directories; authorization stores and unrelated personal files are
outside the task. A real access error is feedback to report, not a reason to
invent missing evidence or bypass the boundary.

## Reason from actual evidence

Distinguish the user's observation, model assumptions, generated completion,
effective runtime behavior and your interpretation. Live outputs and source
code can contradict a default, an old note or a prior acceptance. Read the
relevant records and revise the conclusion. A tool returning success establishes
only the checks it actually ran. Visual quality and intended behavior still
require your judgment; sparse images do not establish continuous video review.

Use native media capabilities to review original references and returned
candidates when available. The optional repository `inspect-video` reader supplies
ordered images and timestamps; read its skill if choosing that reader. Native
clients may process media differently, so judge the evidence you actually receive.
Plausible motion or appearance defects may be invisible in sparse samples or small thumbnails. Choose additional evidence
for the uncertainty at hand; the user should not have to identify the defect
first. Suspected failures are hypotheses to investigate, not facts to invent.

When something fails, investigate the responsible layer: construction, controls,
conversion, runtime or presentation. Choose useful comparisons or write an
analysis suited to the evidence. Preserve failed attempts and repair demonstrated
causes. Changes need matching fresh evidence before reacceptance; do not alter
retained results or relax a bound merely to pass.

When a repair changes several factors together, report the combined result;
attribute individual causes only when the evidence separates their effects.

## Skills and completion

Read the current task's general workflow from its startup instructions. The catalog contains
additional skills; search/read those relevant to the chosen asset representation,
modeling stage or demonstrated uncertainty. Detailed resources are optional and
loaded only when needed. You control the specific tools, inspection coverage and
iteration strategy within the task's constraints and available resources.

The catalog exposes names and descriptions, not the specialist instructions.
Use the task workflow's links to select guidance for the current requirement or
observed mismatch, then read it before applying that capability. Follow only
dependencies whose evidence is needed or missing; reuse valid existing evidence.
A specialist's output returns to the task's visual/functional review and does
not replace it. When handing off a participant or resuming work, retain the
relevant requirement, evidence paths and unresolved finding so the next agent
can select the same guidance without loading the entire catalog.

Prefer a provided capability when it covers the identified uncertainty and is
available. Read its contract before writing another camera solver, segmentation
route, fitter or geometry reviewer. Authoring the parameterized model, preparing
observations and adapting formats remain normal agent work. Task-local analysis
can supplement these capabilities; replacing their checks needs a concrete
coverage/runtime gap or already retained equivalent evidence, with the same
current-source and observation coverage. A small residual from a new partial
solver does not automatically discharge the workflow's required verification.

For reference-based twin generation, appearance is a primary completion
requirement alongside intended behavior. Establish corresponding draft/reference
comparisons early enough to correct shape and proportions. A clear visual defect
must drive useful investigation and repair; repeating successful physics or
presentation checks cannot resolve it. Optional tool selection does not make
the necessary comparison, diagnosis or repair optional. Use the relevant
specialist tools or suitable task-local diagnostics when evidence is missing or
a consequential mismatch remains, and retain evidence of their actual effect.

For precision twins, use the information the supplied images, multiview captures
or videos actually provide. Establish quantitative checks for important
properties the observations can constrain, using suitable available tools;
plausible appearance alone does not justify skipping this baseline. For video,
select complementary views or informative time intervals, retain timestamps and
distinguish camera motion from changing object state. A few convenient frames
cannot certify omitted views or dynamics. Limited observations narrow the claim:
use visible single-view contours where useful without inventing depth or hidden
structure. Reuse adequate evidence instead of requiring every model or tool.

Allocate repair effort by its effect on the requested twin. Wrong identity,
silhouette, proportions or layout should take precedence over decorative detail
that cannot correct them. Establish an adequate coarse appearance comparison
before investing heavily in texture microdetail, individual contents or final
presentation. This is a priority judgment, not a fixed stage order: appearance
that depends on simulation may require early physical work. Repeated failed
comparisons should trigger a change in diagnosis or method, not more detail on
the same incorrect shape.

Keep consequential comparison findings open until their own evidence resolves
them. Isolating or repairing one shading/material cause does not establish that
the outline and proportions match; return to those comparisons after the
diagnostic. A cropped or occluded reference supports review of its visible
regions, not a complete hidden shape. Compare corresponding visible portions
rather than resizing an entire candidate to an incomplete photograph.

Before choosing geometry or fitting parameters, reason about reference-supported
structural priors such as rotational/mirror symmetry, straight surfaces, shared
sections and mechanism constraints. Distinguish observations from assumptions
and apply a prior to the appropriate parts rather than imposing a category
template. Use visual understanding to choose the structure and meaningful
adjustable dimensions; use measurements to constrain their scale, proportions,
pose and uncertainty. A view-dependent mask is evidence of visible occupancy,
not a 3D section or an instruction to copy its irregularities into the mesh.
Encode supported priors in the editable model so optimization preserves
them. A higher image score cannot justify unsupported deformation or loss of an
identity-bearing feature. When fit and prior disagree, investigate camera,
segmentation, occlusion and model freedom; revise the prior only when evidence
warrants it. Inspect the resulting geometry and corresponding RGB appearance in
addition to numerical scores.

A structural prior is not the draft's current parameterization. Rotational
symmetry constrains shared radial sections but does not establish a straight
wall, a particular radial profile or guessed wall thickness. Keep those choices
adjustable where the reference constrains them, and mark unsupported values as
assumptions. Holding a draft's geometry fixed for a controlled diagnostic is
useful; that control does not turn its geometry into an observed property.

The common, task-entry and specialist layers are framework-maintained official
skills. You are encouraged to distill verified, reusable experience into the
caller's personal skills, using their coding client's skill location and format.
Keep those skills separate from the framework's official catalog and load them
when relevant; creating one is optional, not a required task stage. Describe the
conditions and evidence behind a useful method rather than turning one example's
parameters into universal rules. Personal experience does not automatically
rewrite official skills or override the current user's requirements.

Generation is not delivery until the selected task's acceptance requirements
pass and you judge the evidence adequate. State remaining uncertainty and real
adapter/resource limitations honestly. An object-generation task can finish
independently of scene generation; robot training and policy design are downstream
responsibilities.

Retaining an honest partial result preserves progress; it does not satisfy an
unmet generation objective. Continue repair when feasible. If a concrete
capability, data or resource limitation prevents progress, report that limitation
and the remaining criteria rather than treating a partial asset or a completed
experience package as finished delivery.

Once the user's requirements and necessary acceptance checks are satisfied,
complete delivery. Further iteration should address a concrete issue affecting
the requested outcome; do not independently raise the standard or repeatedly
optimize diagnostic values that have no meaningful impact on that outcome.
