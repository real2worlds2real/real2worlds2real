# ASTRA: real2sim2real with an agentic harness

Code for **Real2Worlds2Real: Scaling Self-Improving Agentic Real2Sim2Real for Robot Manipulation**
(<a href="https://real2worlds2real.github.io" target="_blank" rel="noopener noreferrer">project page</a> · <a href="https://arxiv.org/abs/2611.31098" target="_blank" rel="noopener noreferrer">arXiv:2611.31098</a>).

Jinzhong Gu<sup>1,&#42;</sup>, <a href="https://jiaka1chen.github.io/" target="_blank" rel="noopener noreferrer">Jiakai Chen</a><sup>3,&#42;</sup>, Xinyi Cao<sup>4,&#42;</sup>, <a href="https://sxy7147.github.io/" target="_blank" rel="noopener noreferrer">Yan Shen</a><sup>1,&#42;</sup>,
<a href="https://suihangke.github.io/" target="_blank" rel="noopener noreferrer">Hangke Sui</a><sup>3,&#42;</sup>, Shutian Zheng<sup>1</sup>, Xiang Xiao<sup>2</sup>, Haoran Lu<sup>1</sup>, Fan Xu<sup>1</sup>,
Lu Shi<sup>2</sup>, Daochen Shi<sup>2</sup>, Ruihua Zhang<sup>2</sup>, <a href="https://zhanwei.site/" target="_blank" rel="noopener noreferrer">Wei Zhan</a><sup>1</sup>,
<a href="https://me.berkeley.edu/people/masayoshi-tomizuka/" target="_blank" rel="noopener noreferrer">Masayoshi Tomizuka</a><sup>1</sup>, <a href="https://warshallrho.github.io/" target="_blank" rel="noopener noreferrer">Ruihai Wu</a><sup>1</sup>

<sup>1</sup>University of California, Berkeley · <sup>2</sup>NVIDIA · <sup>3</sup>University of Illinois Urbana-Champaign · <sup>4</sup>Carnegie Mellon University · <strong>&#42; Equal contribution</strong>

[Citation and source history](CONTRIBUTORS.md)

From observations of a real scene to a policy prepared for the real robot. An agent (ASTRA) does the work through
layered skills and tools it selects as needed, inspects the evidence each step produces and revises its own results.
Simulation success is intermediate evidence; real validation is run and reported by people.

Start a reconstruction with the [agent module workflow](end_to_end/docs/reconstruction.md). One `astra-e2e` run can use calibrated cameras and poses, individual object workers, native physics checks and scene assembly. Choose an object, scene or policy handover as the goal.

## Public interfaces

| Component | Python package | Command |
|---|---|---|
| [Real2Sim Harness](real2sim/README.md) builds and reviews editable assets in task workspaces. | `real2sim_harness` | `real2sim-harness` |
| [Calibrated Scene Reconstruction](real2sim/docs/scene/twin_pipeline.md) reconstructs metric scenes from capture, places the robot, and checks physics and delivery. | `calibrated_scene_reconstruction` | `calibrated-scene-reconstruction` |
| [Policy station](policy_development/README.md) develops, freezes and evaluates a policy under a task budget. | `astra_station` | `astra-station` |
| [End-to-end workflow](end_to_end/README.md) binds the accepted scene to a station task and prepares the handover. | `astra_e2e` | `astra-e2e` |

Use the [configuration templates](configs/README.md) to start a run. The [reconstructed examples](examples/README.md) include interactive USDZ scenes for screw closure, gear meshing, insertion, cup nesting and pushing.

```text
real-world observations          real2sim/                    policy_development/              real robot
 photos, walk-around video, ──>  twin · cousins · physics ──>  station: methods chosen, run ──>  hand-over package,
 motion video, task in words     delivery record               and compared; one frozen          validation by people
                                                               deliverable, sealed evaluation    (pending until reported)
                                 └──────────────── end_to_end/: the chain checks each hand-over ────────────────┘
```

The two parts do not import each other. Part 1 hands over a small JSON record of an accepted twin
([delivery record](real2sim/docs/DELIVERY.md)); Part 2 starts from it
(`astra-station new-twin-task --delivery <record>`); the chain orders the stages
([how it runs](end_to_end/docs/chain.md)).

## Start

From the checkout root on Linux, create a virtual environment with Python 3.11 or 3.12 and install the three local packages together:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install ./real2sim ./policy_development ./end_to_end
source env.sh
real2sim-harness --help
calibrated-scene-reconstruction --help
astra-station --help
astra-e2e schemas
```

Use `python3.12` in the first command if that is your installed version. In a new shell, activate the same virtual environment and source `env.sh` before running the [examples](examples/README.md). The [offline installation guide](end_to_end/README.md#install-the-commands) describes how to prepare a wheel directory.

The default Vega 1P with Sharpa hands and the standard Franka Panda are included. Optional grasp synthesis uses a separate hand descriptor and solver. See the [robot inventory and expected layout](real2sim/assets/robots/README.md).

- An agent working here starts from [AGENTS.md](AGENTS.md).
- Each part documents its own environments. Part 1's reconstruction needs a GPU workstation, Blender 4.5 and Isaac
  Sim 5.1; Part 2's controller is standard library only and its stations run where the site file says.
## Scope and limits

- The agent does not operate the real robot. Every record keeps real validation pending until a person has run it.
- A twin is evidence about one capture. Scale comes from legal evidence in the capture, never from the task's true
  dimensions; a simulation score never justifies changing geometry, friction or success definitions.
- Task-specific policy-development code (task specifications, simulator backends of hand-authored tasks, real-robot
  workers) is not part of this release. Supply it as your own task directory in the
  [station harness format](policy_development/docs/station_harness.md#what-is-generic-and-what-a-task-supplies). This repository holds the
  task-independent parts and the path from a twin to a station.

## License

The source code and documentation written for this project are released under the [MIT License](LICENSE).

The licence does not cover what this repository carries from others, which keeps its own terms:

- For bundled robot models, Blender Cell Fracture and patches to upstream code, see the [repository notices](THIRD_PARTY.md). Optional tabletop grasp synthesis requires a compatible solver supplied separately under its own terms. See the [grasp setup guide](real2sim/docs/robots/grasping.md).
