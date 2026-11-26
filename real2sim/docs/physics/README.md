# Physical identification

`physics/replay/` uses video plus robot joint trajectories; `physics/video/`
uses observed free-motion segments. Shared projection, mask statistics and
association live in `physics/common/observations.py`; JSON contracts live in
`common/schema.py`. Each route retains its own plane-noise and trimming policies.

`core/perception/` owns model loading and I/O adaptation, not object selection,
alignment scores or acceptance. DROID column/pose adaptation lives in
`core/adapters/droid.py`. Assets owns mesh and pose measurements; sim builds
native stages from prepared inputs.

The default manipulated-object route requires a reviewed agent Blender model.
Missing model evidence blocks it; template/SAM 3D routes are explicit choices.
Geometry remains fixed during physical identification. A shape change requires
new model/source review. Static imagery cannot identify all material values;
report measurements, bounds and priors separately.

Policy evaluation can motivate a better physical model under a transfer hypothesis
and independent consistency checks. It cannot identify a real parameter merely
by rewarding simulator success. See [transfer rules](../../../end_to_end/docs/transfer.md).

Entry points: `apps/physics/replay.py`, `apps/physics/video.py`,
`apps/physics/soft.py` and `tools/physics/video_physics.sh`. Legacy real2sim/vidphys
imports forward to these domains. Backend code consumes prepared data rather
than importing the identification module. Existing physical thresholds remain.

Read [replay](replay.md) and [video](video.md) for their actual data and procedures.
