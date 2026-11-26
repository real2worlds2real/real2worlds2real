# Episode recording and dataset export

`apps/datagen/episode.py` composes a rigid task environment, its recorder and an optional policy runner after Isaac starts. `apps/episode.py` forwards the original command line, including recording, policy evaluation, resumed states, batch execution and camera replay.

```sh
tools/run_isaac.sh apps/datagen/episode.py sessions/<session> --robot <robot> --episodes 2
python -m calibrated_scene_reconstruction.datagen.export.lerobot outputs/<dataset> sessions/<session> --only-success
```

The environment in `tasks/rigid/runtime.py` owns robot control, planning, resets, shared state and camera handles. Each run uses an explicit `EpisodeRuntime` instance. Named services are injected by the app; domain modules do not import one another to construct dependencies.

`datagen/recording.py` consumes the `RecordingContext` protocol. It captures frames and state, writes raw metadata/arrays and completes video encoding. Recording callbacks share the environment instance, including temporary batch robots and the active output path. Validation and recording retain the same physics stepping order.

`policy/rollout.py` owns policy construction and the observe/infer/step/check loop. The rigid adapter in `tasks/rigid/evaluate.py` owns task outcomes, goal interpretation, robot actions, tracing, video and visited states for subsequent data collection. Generic policy code does not contain rigid task success rules.

Recorded episodes retain `raw/meta.json`, `raw/steps.npz`, per-camera images/video and `result.json`. The LeRobot exporter retains its feature schema, camera names, frame rate, joint ordering and action convention. `--tags` selects the raw metadata's tag; it does not select an episode directory suffix.

The separate demonstration retargeter is composed through `tools/datagen/generate.py`. Its package API accepts an explicit scene factory; this keeps scene preparation outside datagen. `python -m calibrated_scene_reconstruction.demo.generate` retains the original scene-composing CLI as a compatibility adapter.

Video composition is `python -m calibrated_scene_reconstruction.datagen.video`; `calibrated_scene_reconstruction.video` remains an identity-preserving alias for import callers. Shared encoding primitives remain in core, while video composition belongs to datagen.

## Which scene an episode is recorded in

`episode.py` builds its stage from the session's inputs (`sim.scene_stage.build_scene`): the session's own task
objects as the agent modelled them, its set pieces and static objects, inside a procedurally built room (a cousin's
room and looks when a variant names one). It does not load the single USDZ a frozen session delivers at gate S6, and
its tasks are `pick` and `place`, where a place goal is a receptacle the sampler adds (`grasping.place`), never
another object of the scene. A task such as "put the blue cup into the white cup" therefore has no episode recorder
here.

Working **in the delivered twin itself** goes through policy development: bind the twin to a station
([environment factories](../../examples/station/README.md)). A twin station's interface is `state_vector`, so its
methods are those that apply to every interface (`evaluate`, `synthesize_program`, `research_session`); the station's
demonstration recorder (`record_demos`) and the learning methods built on recorded data apply only to the `cloud_q29`
interface of the lab's hand-written tasks. Recording demonstrations in a delivered twin is not available yet.

To use `episode.py` on a frozen twin's objects, work on a copy, never on the frozen session (its freeze manifest is
hashed by every delivery record and binding made from it): copy the session directory under a new name and treat the
copy as a new, unfrozen session. Gate S5 (whole-scene physics) is not inherited by the copy; anything the copy changes,
including the room it is recorded in, is outside what the twin's acceptance covered.
