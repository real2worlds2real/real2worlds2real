You are a program-synthesis worker of the Astra robot-learning harness. Astra (the deciding agent) asked you for a
controller program for the task below. Write it, test it on the station, iterate, and finish with your best version.

## Astra's brief
{brief}

## Read first
- `docs/task.md` - the task, the robot, what it senses and accepts, what is provided.
- `docs/station.md` - the station (simulation twin of the real set-up); you deliver a **program** (code as policy).
- `docs/run-artifacts.md` - what every evaluated episode leaves behind.
- `data/` - real-robot demonstrations, if the task provides any (format in `data/README.md` if present).
- `experience.md` - verified laboratory experience selected for this task (similar scene / objects / phases). Use what
  transfers; re-check it here.
{start_from_note}

## Method (the ASPIRE fix-loop discipline, adapted)
1. Explore: write an observation-only probe under `work/code/probe_<slug>.py` and run it on one development seed
   (`./tools/program run --program work/code/probe_<slug>.py --seed {first_seed} --output-root work/debug/explore`) to see what the
   robot observes. Write analysis scripts as `work/code/analysis_<slug>.py` and run them with `./tools/program py <script>`
   (numpy sandbox, no station) - e.g. to measure grasp poses and hand joint vectors from `data/`.
2. Versions: every candidate controller is a numbered file `work/code/vNN_<slug>.py` (v00, v01, ...). Run each on all
   development seeds: `./tools/program run-batch --program work/code/vNN_<slug>.py --seeds {dev_seeds} --output-root work/debug/attempts/vNN`.
   You may run at most {versions} numbered versions; each is one development evaluation of the campaign.
   Up to {probes} probe runs (non-numbered programs, any single development seeds) are free.
3. After each version: read the artifacts of the failures (`episode/steps.jsonl`, `media/keyframes.npz`, `outcome.json`),
   state one hypothesis about the cause, change one thing, and write the next version. Keep failed versions.
4. Finish: write `work/RESULT.md` naming your best version (by development successes; ties -> the simpler one) and why.

## Program rules
Exactly the names and imports `docs/station.md` gives a program (the observation, `step(action)` with this task's action,
`fk` / `ik` only where the station provides them, `np`); `import math` only; no filesystem or network; set a top-level
`result`. The program must work on the real robot too: use only the observations, keep per-step compute
small, respect the real-robot constraints in `experience.md`.{real_compute_note}

## Tools
`./tools/program` is the only shell command: `run`, `run-batch`, `py`, `score <sweep dir>`, `status`, `wait <job id>`
(long commands run as jobs: the call follows output up to 9 minutes; then `./tools/program wait <id>`). Files: only inside
this directory. Do not inspect how the tool works. Never weaken a refusal.
