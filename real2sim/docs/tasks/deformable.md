# Deformable manipulation: garment folding and rope letters

Two tasks: garment folding, by the official Diffusion Policy (third_party/diffusion_policy, unmodified) trained on
simulated demonstrations and evaluated closed loop in the same simulator; rope letters, by model predictive control:

| | garment folding | rope letters |
|---|---|---|
| simulator | the LeHome Challenge's Isaac Lab scene (Isaac Sim 5.1, PhysX particle cloth) | Newton (VBD rope, MuJoCo arm) |
| robot | two SO-101 arms side by side, 6 joints each | one Franka FR3 |
| observed | top and the two wrist cameras, the joints | the rope's segment positions (the simulator's); head and front cameras film it |
| method | the official Diffusion Policy on the challenge's 1000 teleoperated folds, replayed in a realistic look | model predictive control of pick-and-places |
| success | the challenge's: distances between six marked cloth points | every rope segment within 2 cm of its place on the letter |

The earlier routes (cloth and rope in PhysX held by attachments; towel and T-shirt folds in Newton by a tilted
friction pinch whose outer fingertip went below the table, the arms not colliding with it; LeRobot's Diffusion
Policy; the rope letters by a Diffusion Policy on the oracle's demonstrations) are in the git history.

## Garment folding (LeHome Challenge)

The challenge (lehome-official/lehome-challenge, Apache-2.0) folds long- and short-sleeved tops and long and short
trousers, 12 garments a category (Seen 0-9, Unseen 0-1). Its grasp is contact only: the PhysX particle cloth's
friction and adhesion against the SO-101's jaws, no attachment.

**Setup** (`tools/tasks/setup_lehome.sh`): the challenge and its Isaac Lab fork, pinned, under `envs/lehome/`
(git-ignored), with the packages our Isaac Sim 5.1 image lacks in a venv made from the image's python
(`--system-site-packages`, every install constrained to the image's versions), and the challenge's assets and
demonstrations from HuggingFace. `apps/tasks/lehome_run.py` runs the challenge's modules with them in the image
(`tools/run_isaac.sh`), from the challenge's checkout:

    cd envs/lehome/lehome-challenge
    CUDA_VISIBLE_DEVICES=<g> MEM=48G ../../../tools/run_isaac.sh ../../../apps/tasks/lehome_run.py -m scripts.dataset_sim replay ...

`--merged-garments` (before `-m`) fixes the challenge's replay of its merged datasets, which loads the first garment
for every episode: a run then covers one garment's 25 episodes. Isaac runs started together race at start-up (a null
prim, or a hang after `app ready`): start them 90 s apart.

**Look** (`src/calibrated_scene_reconstruction/tasks/lehome/appearance.py`, task `R2S2R-LeHome-Garment-Look-v0`): the challenge's env with its
physics untouched and, at every reset, a real fabric on the garment (ambientCG PBR sets: colour, roughness and normal
maps; knits and plain weaves for tops, denims and twills for trousers, shirtings; a grey or white fabric dyed a colour
clothes come in, a coloured one kept), one of the challenge's table textures that reads as a table top, and the dome
light's brightness and colour temperature. The draws go to `$SCENE_RECONSTRUCTION_LOOK_LOG`.

**Data.** Use the challenge's replay command above to collect successful demonstrations.
`tools/policy/export_dp_official_deformable.py` writes the official replay buffer at 10 Hz
(every third frame), with three cameras at 240×320, 12 joints and 12 joint targets.

**Policy**: `configs/dp_official/lehome_garment_hybrid.yaml`, the official real-robot hybrid recipe (robomimic's
ResNet18 from scratch with GroupNorm and a spatial softmax per camera, 90 % random crop, 2 observations, 16 predicted
and 8 executed actions, DDIM) on the cameras alone; `_sd05` adds the joints as the low-dim observation, half of the
training samples showing their dataset mean (the state dropout that keeps a policy from following its own joints
instead of the image).

When resuming a W&B run in the earlier project, pass `logging.project=r2s2r_dp_official` with that run's ID.

**Evaluation.** Serve the trained checkpoint with `tools/policy/dp_official_server.py`.
The challenge's `scripts.eval` module uses the `dp_official` policy registered by
`apps/tasks/lehome_run.py`. Record the category, garment, seen/unseen split and
outcome for each rollout using the challenge's success criterion.

## Rope letters (Newton)

`apps/newton_rope_letter.py` (on `apps/newton_rope.py`'s scene, `tools/run_newton.sh`): a capital letter (C J L M N S
U V W Z, `src/calibrated_scene_reconstruction/deformable/letters.py`) drawn in dashes on the table, a rope as long as its stroke in a random
curve, and an FR3 that lays it along the letter by pick-and-places, holding it by friction alone. `--randomize` draws
the letter's height and the rope's thickness, bending stiffness and friction; `--record` films it
(`src/calibrated_scene_reconstruction/deformable/newton_record.py`, `--look random` for the scene's look, `appearance.py`).

**The rope** weighs what a cotton rope does (`--density 800`, about 60 g/m at 10 mm): at Isaac Lab's Lift-Cable 100
the table's friction does not hold the rope's bends, and an inner segment moved 2-6 cm sideways sprang back to 0.3 of
the move once let go. **The arm**: the IK keeps joints 1 and 3 near their middle (`IK_YAW_RANGE`); warm-started from
its last solution, it otherwise drifted along the arm's redundancy into far configurations that swung the arm.

**Planner** (`--planner`): DeformableRavens' oracle (the segment farthest from its place, set down there), or model
predictive control (`src/calibrated_scene_reconstruction/deformable/rope_mpc.py`), both from the rope's segment positions. Before each
pick-and-place the MPC rolls candidates out in a model of the rope (the segment centres dragged as a chain on the
table: the taken one follows the hand, the others only when pulled taut) and executes the one whose rope ends closest
to the letter, looking a greedy pick-and-place further; then it sees the rope again. A candidate sets one segment down
at, beyond or near its place, or traces it: carries it to the place of the segment 3, 6 or 10 along and drags it back
along the letter low over the table, the rope behind it laid along the stroke (a stroke lying a few cm off its line
cannot be moved one point at a time). What the model got wrong (a pinch that did not hold at a tight bend, a
pick-and-place that left the rope farther off) is not chosen again.

**Evaluation.** From `real2sim/`, run a rollout with an explicit seed and planner:

```sh
tools/run_newton.sh apps/newton_rope_letter.py --seed 0 --letters C --planner mpc --randomize --viewer null
```

Record the seed, letter and outcome for each rollout.
