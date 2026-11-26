---
name: review-numerical-evidence
description: Diagnose retained rigid trajectories, commanded motion, screw travel, convex probe clearance or required scene supports with shared numerical audits.
---

Use `audit_task_evidence(request_path)` for a relevant numerical uncertainty;
read that mode in [the request contract](../../docs/EVIDENCE_AUDITS.md).
The framework retains inputs and hashes; reuse these computations instead of
rewriting them in each task.

| Question | Audit mode | Limit |
| --- | --- | --- |
| Is a named body quiet in the requested final window? | `rigid_window` | Quiet motion does not prove support or perturbation stability |
| Did actual readback follow an external translation command? | `command_tracking` | Observe the whole command interval; commands are not readback |
| Does relative screw travel match a lead, with later escape checked? | `axial_motion` | Fitted motion does not prove contact or release |
| Are spherical starts clear of every convex material piece? | `convex_sphere_starts` | Conservative unresolved cases do not prove collision |
| Is every declared participant/support relation verified? | `support_coverage` | A contact elsewhere cannot satisfy this relation |

Use recorded body names, explicit quaternion order and a common metre/frame/time
convention. Retain readback provenance and describe assumptions. Full-window and
full-command coverage are required by default; explicitly partial diagnostics
cannot certify omitted motion.

When these questions matter to delivery, resolve them with this tool or equivalent
retained native evidence. Inspect `audit_passed`, coverage, flags and scope, then
review actual motion and appearance. Computation success is not acceptance, and
cannot replace missing simulator capability.
