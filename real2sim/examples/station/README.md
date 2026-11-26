# From a delivered twin to a station: environment factories

A delivery record says which twin was accepted. A policy-development station additionally needs an *environment
factory*: code that builds the simulation from the delivered files, puts a robot in it, and exposes observations,
actions and a hidden judge. The factory is the one piece a station author writes per task. This directory holds a worked
example for each of real2sim's two routes.

| Example | Route | Twin | Task |
|---|---|---|---|
| [`cup_nest_factory.py`](cup_nest_factory.py) | scene pipeline (walk-around video, gates S0-S6) | a frozen two-cup reconstruction supplied by the operator | put the blue cup into the white cup |
| [`workspace_lift_factory.py`](workspace_lift_factory.py) with [`workspace_lift.setup.json`](workspace_lift.setup.json) | task workspace (a composed, accepted scene) | the scene of [`workspace_lift.scene.json`](workspace_lift.scene.json) | lift the blue cup off the desk |

## The protocol

The station names the factory in its task (`--factory <file.py>:<function>`) and calls it once at start:

```python
def build(source, task, num_envs, run_dir) -> list:   # one environment per station env
```

`source` is the frozen session (scene pipeline) or the accepted run (task workspace); `task` is the station task spec,
whose `twin` block carries the delivery record's path, the route and the scene file. Each environment provides:

| Method | Meaning |
|---|---|
| `reset(case)` | `None` for a fresh training start, or a sealed start (an integer seed or a dict) |
| `observe() -> dict` | This environment's observation: numpy arrays without the env axis |
| `step(action)` | One control step |
| `measure() -> dict` | The task state. `"success"` is the hidden judge; an optional `"failed"` string ends the episode early |
| `info() -> dict` | Optional: `action_dim`, `interface`, `observation_layout`, `kinematics`, ... |

The station keeps the episode semantics (success held for `--twin-hold` steps, a reported failure, or the step limit)
and shows a client only success, the termination reason and the step count. A factory imports nothing from the station:
it follows this protocol and nothing else, so it lives with the twin. It runs in the simulator's interpreter
(`tools/run_isaac.sh`).

Both examples expose privileged simulator state (object poses) as the observation. That is enough to develop and compare
methods in simulation; it is not an observation a real robot has.

## What each route gives the factory

**Scene pipeline.** The session states the metre, the support and where the robot stood (`photo_robot.base` in the
scene graph, calibrated at gate S1), and carries the task with its target and goal. `cup_nest_factory.py` therefore
takes everything from the session: it builds the scene as the session's own task check does and places the robot where
it was filmed. Set the station setup's `robot` field to the robot specification that matches the capture. The example defaults to `vega_1p_sharpa_arm`. Its judge is the task definition the session's agent wrote.

**Task workspace.** The record pins a composed `scene.usda`: participants with their own rigid bodies and colliders,
static support boxes, visual backgrounds. It has no physics scene and no robot, and it states neither metric scale nor a
robot placement (`statements.metric_scale` and `statements.robot_placed` are `null`). The robot interaction setup is
therefore the station author's declaration, not a property of the twin. It is written to a small file and travels in
the task spec as `twin.setup`, marked as the author's:

```bash
real2sim-harness deliver --run /path/to/run --output /path/to/lift.delivery.json
python -m astra_station new-twin-task --delivery /path/to/lift.delivery.json --task-id cup_lift \
    --factory real2sim/examples/station/workspace_lift_factory.py:build --setup real2sim/examples/station/workspace_lift.setup.json \
    --num-envs 1 --max-steps 200 --out /path/to/tasks/cup_lift \
    --policy-io '{"observation": ["TCP minus cup centre (3, m)", "cup bottom above the desk (1, m)", "cup displacement (2, m)", "hand closure (1)", "cup tilt / 90 deg (1)"], "action_dim": 4, "action": "TCP delta (3, x0.01 m per step) + hand closure delta (1, x0.05 per step), each in [-1, 1]", "interface": "state_vector"}'
```

`workspace_lift_factory.py` references the delivered file into a fresh stage (it never opens it for writing), finds the
target's rigid body and the support's top from the scene itself, and adds the robot at the declared base.

## The example scene of the task-workspace route

[`workspace_lift.scene.json`](workspace_lift.scene.json) describes a cup on a 1.20 m × 0.80 m worktop, 1.00 m above
the floor. [`desk_visual.usda`](desk_visual.usda) supplies the visible tabletop and legs. The manifest supplies the
static collision box and marks this authored layout as `scale_verified: false`. To run the example, supply your own cup's accepted PhysX simulator bundle.

Activate the [Real2Sim environment with scene support](../../README.md#start-here). From the checkout root,
initialize a scene task with your scene reference and a Blender executable:

```sh
real2sim-harness task init --scene --reference /path/to/scene-photo.jpg \
    --blender /path/to/blender --objective 'Compose a cup on the example desk.'
```

Set `WORKSPACE` to the `workspace` path printed by that command. Copy both example files there, then edit
`objects[0].bundle` in the copied JSON to the absolute path of the cup bundle directory containing `asset_bundle.json`.

```sh
cp real2sim/examples/station/workspace_lift.scene.json real2sim/examples/station/desk_visual.usda "$WORKSPACE/"
```

After setting the bundle path, assemble the scene:

```sh
printf '%s\n' '{"manifest_path":"workspace_lift.scene.json"}' > "$WORKSPACE/assemble.json"
real2sim-harness task call "$(dirname "$WORKSPACE")" assemble_scene --request "$WORKSPACE/assemble.json"
```

The response includes the composed scene directory. Continue with `run_scene_simulation` and `accept_scene`
using the [scene workflow](../../docs/SCENES.md) before handing the scene to the station.

## Each factory checks its own judge once

At start, outside any episode, each factory runs its judge on cases whose answer is known and writes the outcome to
`twin_factory.json` in the station's run directory. `workspace_lift_factory.py` refuses to serve when a case comes
out wrong. `cup_nest_factory.py`: the untouched start must fail, and the blue
cup released over the white cup under gravity must pass. `workspace_lift_factory.py`: the untouched start must fail,
and the cup raised above the lift height with the hand open and away must still fail (`lifted` true, `held` false),
because being high is not being carried. A positive lift needs a grasp, which is the policy's work, so the factory does
not fabricate one.

## Limits

- One environment per station (one PhysX scene); start the station with `--num-envs 1`.
- Neither example claims that the task is solved: they define the station on which methods are tried.
- The task-workspace example inherits its scene's limits: estimated scale, assumed friction and mass, no calibration
  against real motion.
