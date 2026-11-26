# End to end

The layer that joins the two parts of this repository: a video of a real scene becomes a simulation
([real2sim](../real2sim/README.md)), a policy is developed in that simulation
([policy development](../policy_development/README.md)), and the result is prepared for validation on the real robot.
Neither part imports this layer; it imports both.

| Entry | What it is |
|---|---|
| [The chain](docs/chain.md): `python -m astra_e2e.chain` | Observations → twin (delivery record) → station task and binding → campaign and frozen deliverable → one sealed evaluation → hand-over for human-operated validation. No model inside, no robot operated. |
| [Oversight](docs/README.md): `python -m astra_e2e.oversight` | An agent-neutral controller for whole projects: intake contract, module obligations, command packets, evidence, budgets, feedback, procedural knowledge. |
| [Unified agent tools](docs/unified_harness.md): `python -m astra_e2e.harness` | One JSON tool interface over oversight, reconstruction-bound policy jobs, experience lookup and backend releases. |
| [`unified/`](unified/README.md) | The reconstruction-bound policy path: bound native jobs, training, review, campaign preparation. |
| [Skills](skills/) | [end-to-end](skills/end-to-end/SKILL.md), [data-policy-loop](skills/data-policy-loop/SKILL.md), [research-transfer](skills/research-transfer/SKILL.md) |

```sh
source env.sh                                   # repository root
python -m astra_e2e.chain schemas
python -m astra_e2e.oversight schemas
python -m astra_e2e.harness schemas
```

## Install the commands

From the repository root, use a Linux virtual environment with Python 3.11 or 3.12. Install the three local projects together so pip resolves the companion packages from this checkout.

The core dependencies are declared in the project metadata for [Real2Sim](../real2sim/pyproject.toml), the [policy station](../policy_development/pyproject.toml), and the [end-to-end workflow](pyproject.toml). Additional runtime requirements depend on the simulator and task. See the [reconstruction environments](../real2sim/docs/ENVIRONMENTS.md).

```sh
python -m pip install ./real2sim ./policy_development ./end_to_end
astra-e2e schemas
```

To prepare an offline installation for the same Python version and platform, build a wheel directory on a machine with access to the dependencies.

```sh
python -m pip wheel --wheel-dir wheelhouse ./real2sim ./policy_development ./end_to_end
```

Copy the complete `wheelhouse/` to the target machine. Keep dependency wheels and their bundled license notices intact. Consult the [repository notices](../THIRD_PARTY.md) before distributing a wheel directory or container. In a fresh virtual environment, install the workflow by name and check the entry point.

```sh
python -m pip install --no-index --find-links wheelhouse astra-e2e
astra-e2e schemas
```

The wheels provide the Python commands. For station runs and reconstruction jobs, also copy or clone the **same revision** of this repository to the target, activate the installed environment, and run `source env.sh` from the checkout root. The jobs use the scripts, skills, schemas and configuration templates in the checkout. Install the simulator runtimes needed for your task as described in the [chain guide](docs/chain.md) and the [reconstruction installation guide](../real2sim/docs/INSTALLATION.md).

A simulation result is intermediate evidence. Every record this layer writes keeps real validation pending until a
person has run it on the robot and reported it; see [transfer rules](docs/transfer.md).
