"""Real-robot station worker: runs ONE frozen station deliverable (a `Policy` class or a CaP program, exactly as the
simulation station's evaluation runner runs it) on the robot, behind a task's real deployer.

Spawned by the task's deployer (JSON lines on stdin / stdout); no robot connection and no motor API here.  The deployer
applies the station's safety contract (arm target within measured +-0.08 rad, hand step limit, joint bounds, force /
collision stops).  What is task-specific - building the station observation from the robot's sensors, the reset of a
real episode, the arm's fk / ik on the robot - comes from a task observer (ARENA_OBSERVER=<file.py>:<Class>, in the task
repository):

    class Observer:
        def __init__(self, info): ...
        def reset(self, req) -> dict                       # facts of the real start (logged by the deployer)
        def observe(self, req) -> (obs, info)              # obs: station arrays with a leading env axis of 1
        def fk(self, q7) / ik(self, q7_seed, pos, quat, iters=10)      (programs, when the station contract has them)
        held_phases: tuple                                 # optional; phases the deployer reports while an object is held

Protocol:
  reset {...task fields...}                      -> {...observer facts..., deliverable, kind, deliverable_sha256}
  frame {tick, phase, arm14, hand22, act, warmup?, ...}  -> {target29 (if act), info}
env: ARENA_DELIVERABLE (policy.py | program.py), ARENA_KIND (policy | program), ARENA_INFO (station info json),
     ARENA_OBSERVER, ARENA_PROGRAM_STEP_S (seconds a program may take for one step before the tick holds; default 0.15)
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '2')
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_file(path, name):
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha_tree(d):
    h = hashlib.sha256()
    for f in sorted(p for p in Path(d).rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        h.update(str(f.relative_to(d)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def send(obj):
    sys.stdout.write(json.dumps(obj, default=lambda v: v.tolist() if hasattr(v, 'tolist') else float(v)) + '\n'); sys.stdout.flush()


class Worker:
    def __init__(self, env=os.environ):
        self.deliv = Path(env['ARENA_DELIVERABLE']).resolve()
        self.kind = env.get('ARENA_KIND', 'policy')
        self.step_timeout = float(env.get('ARENA_PROGRAM_STEP_S', '0.15'))   # a program that does not call step() in time: hold
        self.info = dict(json.loads(Path(env['ARENA_INFO']).read_text()), num_envs=1)
        path, _, cls = env['ARENA_OBSERVER'].partition(':')
        # the observer first: it may load native libraries that must precede torch (e.g. a bundled libstdc++)
        self.observer = getattr(load_file(path, 'arena_observer'), cls)(self.info)
        import numpy as np
        self.np = np
        runtime = env.get('ARENA_PROGRAM_RUNTIME') or next(p for p in (HERE / 'program_runtime.py',          # installed next to it
                                                                        HERE.parent / 'station/client/program_runtime.py')    # repository
                                                         if p.exists())
        self.ProgramEnv = load_file(runtime, 'arena_program_runtime').ProgramEnv
        self.sha = sha_tree(self.deliv.parent)
        self.policy = None; self.prog = None; self.tick0 = 0; self.last_tgt = None
        if self.kind == 'policy':
            self.policy = load_file(self.deliv, 'delivered_policy').Policy(self.info)
        else:
            self.src = self.deliv.read_text()

    def reset(self, req):
        self.tick0 = None
        facts = self.observer.reset(req)
        if self.prog is not None:
            self.prog.stop(); self.prog = None
        self.last_tgt = None
        if self.policy is not None and hasattr(self.policy, 'reset'):
            self.policy.reset(0)
        return dict(facts, deliverable=str(self.deliv), kind=self.kind, deliverable_sha256=self.sha)

    def frame(self, req):
        np = self.np
        t0 = time.perf_counter()
        tick = int(req['tick'])
        if self.tick0 is None:
            self.tick0 = tick
        obs, info = self.observer.observe(dict(req, step=tick - self.tick0))
        out = dict(info=info)
        if req.get('warmup'):                 # the deployer's warm-up before the countdown: build the observation, do not step the deliverable
            self.tick0 = None
            out['target29'] = obs['q29'][0].astype(float); info['policy_status'] = 'warmup'; info['ms'] = (time.perf_counter() - t0) * 1e3
            return out
        if req.get('act'):
            adim = int(self.info.get('action_dim', 29))
            q29 = obs['q29'][0].astype(float)
            if self.kind == 'policy':
                tgt = np.asarray(self.policy.act(obs), float).reshape(-1, adim)[0]; status = 'ok'
            else:
                if self.prog is None:
                    self.prog = self.ProgramEnv(self.src, self.info, self.observer, obs, batch_index=0, realtime=True).start()
                else:
                    self.prog.feed(obs)
                # 11-13: a late program holds the LAST COMMAND (holding the measured hand jumped the fingers back by the
                # command lead, > the .05 rad/tick bound, and the deployer stopped the episode)
                hold = q29 if self.last_tgt is None else self.last_tgt
                tgt = np.asarray(self.prog.next_action(hold, self.step_timeout), float)
                status = 'finished' if self.prog.finished else 'late' if getattr(self.prog, 'late', False) else 'ok'
                if self.prog.error:
                    info['program_error'] = self.prog.error
            if tgt.shape != (adim,) or not np.isfinite(tgt).all():
                tgt, status = q29.copy(), 'bad_target'
            info['policy_status'] = status
            out['target29'] = tgt
            self.last_tgt = tgt.copy()
        info['ms'] = (time.perf_counter() - t0) * 1e3
        return out


def main():
    sys.setswitchinterval(0.0005)
    try:
        w = Worker()
    except BaseException as exc:
        send(dict(error=f'init: {type(exc).__name__}: {exc}', trace=traceback.format_exc())); return
    send(dict(ready=True, model='ext', deliverable=str(w.deliv), kind=w.kind, act_sha256=w.sha, residual=False))
    for line in sys.stdin:
        try:
            req = json.loads(line); op = req.get('op')
            if op == 'quit':
                break
            send(w.reset(req) if op == 'reset' else w.frame(req) if op == 'frame' else dict(error=f'unknown op {op}'))
        except BaseException as exc:
            send(dict(error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()))


if __name__ == '__main__':
    main()
