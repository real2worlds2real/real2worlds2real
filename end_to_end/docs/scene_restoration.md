# Restoring a frozen scene into current contracts

Restoration is a provenance-preserving recovery route. It does not certify a
scene, silently update its historical reviews, or change its geometry to improve
task scores. Use a separate repository with the same session basename; relative
asset paths remain valid while the original source and freeze remain untouched.

1. Run `real2sim/tools/scene/restore_derivative.py SOURCE_REPO DESTINATION_REPO SESSION
   --max-bytes LIMIT --report NEW_PLAN.json` to inspect the file inventory,
   checksum differences and transfer budget. Add `--execute` only for the intended
   isolated destination. Copy timestamps are preserved; historical missing
   artifacts remain explicit. The inherited freeze is archived under
   `provenance/derivative_restore`, never installed as an active new freeze.
2. Inspect the actual raw source views and task, then author current semantic
   declarations. Observed shape, inferred construction and unknown material
   properties must remain distinguishable. Neutral geometry renders and an
   independent review complement source silhouettes.
3. When an accepted rigid model is unchanged but its historical input contract
   predates a new declaration, use `object_revalidate` rather than rewriting old
   `input.json` or `fit.json`. Archive the original `scene_graph.json` inside the
   derivative session and provide session-relative paths:

   ```json
   {
     "session": "real2sim/sessions/example",
     "object": "body",
     "inherited_freeze": "provenance/derivative_restore/inherited_independent_freeze.json",
     "inherited_scene_graph": "provenance/derivative_restore/inherited_scene_graph.json"
   }
   ```

   Invoke it through the ordinary JSON agent tool:

   ```sh
   (cd real2sim && PYTHONPATH=src python -m calibrated_scene_reconstruction.agent.tool object_revalidate \
     --request request.json --output outputs/recovery_source_check)
   ```

   The tool verifies frozen model/mesh/fit bytes, the original graph, exact pose
   and physical settings. Historical input files may be transitively bound by
   the frozen fit's recorded input hash; missing binding is an error. Physical
   ownership comes from the actual NPZ asset loader, with legacy visual-part
   metadata retained separately. It creates an append-only round and measures
   current source scores; failures remain failures. The tool neither activates
   the round nor approves native physics. Only rigid assemblies without joints
   or surface bindings currently support this migration.
4. After independent structural/source review, ordinary `object_export` may
   author the candidate for native tests. Final OBJ/delivery checks still require
   semantic and measured physical evidence. The explicit qualification-only
   scene builder can materialize unaccepted, source-bound candidates for
   intrinsic assays; it cannot issue policy-training or evaluation receipts.
   Rigid intrinsic assays record actual passive, contact and force/release
   quantities with setup assumptions. Successful assays do not identify true
   material properties or prove real-robot success.
5. Inspect the actual native traces, complete current semantic/OBJ/stage reviews,
   regenerate delivery artifacts, reopen the package independently, then create
   a new freeze. Only this newly qualified derivative may enter policy training.

Ordinary rigid task checks now bind current configuration, active geometry,
source calibration and producer files before and after execution, even without
a mechanism declaration. Relocating an old passing task JSON is not a fresh
native check. Script-supplied input hashes must also match. Source-only recovery
does not refresh S5. The new freeze includes verified parent candidate, look and
archived-freeze dependencies used by a revalidated round.

An inherited image/model acceptance record is evidence of that historical scope,
not a pass under current native contracts. Geometry, source poses or physical
settings that actually change require the ordinary fit and qualification loop.
