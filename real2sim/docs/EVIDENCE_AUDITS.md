# Numerical evidence audits

`audit_task_evidence(request_path)` is shared by object and scene tasks. Put the
request JSON and evidence in the task workspace; external inputs are copied into
the retained tool directory. Requests need `schema_version:1`, an `audit` mode
and nonempty `assumptions` describing units, frame, timestamps and provenance.
Results and input hashes are retained in `report.json`.

The `axial_motion` and `convex_sphere_starts` modes use SciPy; install the existing
`simulation` or `fitting` extra when those modes are needed. Other modes use the
normal host dependencies.

`ok:true` means the computation completed. `audit_passed` is `true`, `false` or
`null` for the stated numerical scope; `null` means diagnostics without a complete
pass criterion. `asset_accepted` is always false. A successful calculation never
overrides a failed delivery check or missing motion evidence.

## Trajectory layout

Trajectory modes read NPZ without pickle, using these defaults:

- `time_s[T]`: finite, strictly increasing seconds, at least two samples.
- `body_names[B]`: unique nonempty Unicode names. Select bodies by exact name.
- `poses_xyzw[T,B,7]`: world XYZ metres, followed by unit quaternion XYZW.

For other arrays supply `layout:{"time":"clock","bodies":"names",
"poses":"poses","quaternion_order":"WXYZ"}`. WXYZ means XYZ position followed
by quaternion WXYZ; order is never guessed. Convert retained data without changing
identity or substituting prescribed targets for actual readback. These audits do
not independently establish how an archive was produced. Irregular timestamps
are supported; sampled states cannot certify what happened between samples.

## Final rigid window

```json
{"schema_version":1,"audit":"rigid_window","trajectory":"states.npz",
 "body":"housing","last_seconds":0.5,
 "assumptions":"World/metres, runtime rigid readback, timestamps in seconds."}
```

Returns mean position, maximum deviation, sampled speed and orientation change
from the first state of the final window. `require_full_window` defaults to true.
Explicit false permits incomplete data and reports `window_complete:false`.
There is no universal quiet threshold and no support/contact certificate.

## Command versus rigid readback

Use `audit:"command_tracking"`, `trajectory`, `body`, `commands` as increasing
`[seconds,x_m,y_m,z_m]` rows, and `tolerance_m`. The command is interpolated at
recorded times; extrapolation is rejected. Default `require_full_command:true`
rejects archives missing any part of the commanded interval. Explicit false
permits partial diagnostics, but `audit_passed` remains false without complete
coverage. Returns the worst observed/commanded sample and translation errors.
It does not measure rotation, force transfer or cloth attachment behavior.

## Relative axial motion

Use `audit:"axial_motion"`, `trajectory`, `moving`, `reference`, `axis` in reference
body coordinates, and `interval_s:[start,end]`. Optional `max_excursion_m` flags
escape anywhere in the retained trajectory, including after the fit interval.
That interval must be covered and contain five samples spanning at least 0.25
radians. Reports fitted lead per revolution, residuals, full-trace axial/transverse
excursion and swing. Quaternion sign flips are handled; twist unwrapping assumes
less than pi radians per sample. Select the engaged interval from actual motion,
rather than choosing it solely for a good score. No flag does not prove engagement,
clamping or contact-driven release.

## Convex sphere starts

Use `audit:"convex_sphere_starts"`, `colliders:[{"name":"wall","vertices":[...]}]`,
`spheres:[{"name":"probe","center_m":[...],"radius_m":0.01}]` and optional
nonnegative `margin_m`. Vertices must span a 3D convex volume; supply each material
piece separately. Concave meshes are unsupported here: their convex hull can seal
openings. All geometry shares one metre frame. The whole sphere needs proven
clearance from every piece. The plane-distance bound is conservative near edges
and corners: `unresolved` does not mean collision; use precise geometry/native
evidence when necessary. No cooking, swept-motion or simulator-contact claim.

## Required scene supports

Use `audit:"support_coverage"`, `support_report` pointing to runtime JSON and
`requirements` in the [scene support schema](SCENES.md#required-support-responses).
Bounds are recomputed per named pair; a global contact flag cannot satisfy a
missing relation. This standalone diagnostic does not change scene acceptance.
Declaring `support_requirements` in the scene manifest invokes the same check
in host review and capability negotiation.
