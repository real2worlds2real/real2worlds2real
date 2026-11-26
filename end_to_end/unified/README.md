# Unified (reconstruction-bound) policy path

Bound native jobs on a world binding (`astra_native_policy_binding_v1`) from an accepted reconstruction: `bound_native.py`
(worker), `bound_training.py`, `bound_native_review.py`, probes and the insertion factory, the decision relay
(`astra_unified_decide.py`, `astra_unified_relay.py`, `astra_decide.py`, `campaign_ledger.py`) and the campaign preparers.
Documentation: [end_to_end/docs/unified_harness.md](../docs/unified_harness.md). The harness service uses
this directory as its default `learning_root`.

Twins can also be served as stations to the station path (`policy_development/src/astra_station/twin_task.py`,
`policy_development/station/server/twin.py`), which brings the station methods to a reconstruction.
