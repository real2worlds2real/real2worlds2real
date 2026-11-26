# Native PhysX material capabilities

These optional Python APIs add passive material representations to a caller-owned
Isaac Sim 5.1 scene. Importing them does not start Kit, change the world's clock,
run a model session, instantiate a robot or prescribe a task policy. The caller
owns startup, stepping, controls, source assets and evidence review.

They complement the existing Newton rod/MPM and PhysX impact-fracture routes.
They do not change the harness orchestration or automatically replace the
`run_strands`, `run_fluid` or `run_fracture` backend selection/receipts. Record
native checks with the task's current sources; a standard static rigid-body
check alone does not establish cutting or particle operability.

## Solid grains

```python
from real2sim_harness.physx_granular import (
    GranularSettings, author_granular, read_granular_positions, sample_lattice,
)

points = sample_lattice([-.12, -.12, .015], [.12, .12, .04], .01)
asset = author_granular(
    stage, "/World/Grains", points,
    GranularSettings(radius_m=.004, mass_kg=4e-5),
    simulation_owner="/physicsScene",
)
```

The example uses coarse grains for an inexpensive runtime check. Choose finer
sampling from the reference, contact scale and available resources; do not
treat these numbers as measured sand. Explicit Nx3 points also allow non-box
seed shapes sampled from editable volumes. `mass_kg` is per grain. Friction,
cohesion, damping, adhesion and particle contact/rest offsets remain editable.
The representation is a solid PhysX PBD particle set (`fluid=False`), not a
liquid or a calibrated continuum constitutive model.

The simulation owner must already enable GPU dynamics. Containers and tools
need actual supported colliders. The helper checks metre units and prevents
overwriting an existing asset. `read_granular_positions` requires USD state
writeback; supply a native tensor reader for a different pipeline.

`audit_grains` checks counts, finite samples, particle-radius floor clearance
and optional XY leakage. Also inspect native renders/trajectories for pile
shape, tool redistribution and stability. Center/plane sampling does not prove
continuous-time clearance against arbitrary moving or curved colliders.

## Capsule-chain rope

```python
from real2sim_harness.physx_rope import author_rope, read_rope_state

asset = author_rope(stage, "/World/Rope", centerline_points_m,
                    radius_m=.003, density_kg_m3=1000)
```

An ordered open centerline defines the geometry; no object category or letter
template is required. Dynamic capsules connect at centerline segment endpoints
through D6 joints. Adjacent collision is disabled, while nonadjacent contact
remains available. Choose segment lengths greater than twice the radius.

Total mass is `density*pi*radius**2*centerline_length`. Overlapping capsule
endcaps do not add duplicate material mass. Physical proxies extend one radius
beyond the outer centerline endpoints. The default visible surface is the
capsule chain; a detailed authored braid or jacket needs a separately reviewed
binding to these body transforms.

Translation and twist are locked; the two bending axes use configurable limits
and optional native angular drives. `bend_stiffness` and `bend_damping` are USD
angular force-drive coefficients (N m/degree and N m s/degree). They are not
Newton rod rigidity coefficients. This is useful for a shared PhysX world but
does not replace the existing rod model's stretch/shear/bend/twist mechanics.

The delivered chain has no fixed ends, scripted grasps or trajectories. Add
external fixtures/contacts or robot controls in the task scene. Review native
joint gaps, contact, support and motion, with the current timestep/solver
settings. `read_rope_state` returns centers, endpoints and joint gaps from the
CPU USD-writeback path. A low planar shape error does not certify these checks.

## Native checks

With the chosen Isaac Sim Python and this checkout's `src` on `PYTHONPATH`:

```bash
python scripts/check_physx_materials.py --case granular --output /path/to/new-grain-check
python scripts/check_physx_materials.py --case rope --output /path/to/new-rope-check
```

Each call starts Kit and requires an available compatible graphics device.
Granular dynamics additionally require CUDA. `--dt` and `--seconds` expose the
physical clock and duration. Native stepping is separate from sampled capture.
Outputs retain the USD stage, implementation hashes, trajectory, check report
and initial/interaction/final RTX images. The checks exercise passive drop and tool contact. They are neither robot
policy evaluation nor real-video material calibration. Review these artifacts;
unit tests and a successful application exit alone do not establish delivery.
