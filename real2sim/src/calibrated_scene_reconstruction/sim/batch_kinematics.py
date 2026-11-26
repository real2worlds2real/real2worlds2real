"""Batched planning on the scene copies' robots: N planning jobs run as threads, each with a SlotRobot that stands in
for robots.articulation.Robot; their kinematic requests (IK, FK) meet in the main thread, which serves them all at
once on the copies' articulations -- one joint write, one kinematics update, one read of every link pose and Jacobian
per iteration for all N, where a single planner pays the same update for one robot.

    kin = BatchKinematics(world, base_robot, ["/World/Batch/robot0", ...], offsets)
    results = kin.run(units, n_slots, stop=lambda: enough)     # units: iterator of callables f(slot_robot)

The planning code itself is unchanged (grasping.trajgen.TrajectoryPlanner with rob=SlotRobot): its IK (solve_ik),
FK (teleport + sync + tcp_pose / link_pose / link_transforms) and any other call that touches the stage
(SlotRobot.call(fn, ...), e.g. the hand's hull points) run in the main thread. Poses are in the main scene's frame:
slot i's targets are moved by the copy's offset on the way in, its link poses back on the way out.
"""
import re
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.robots.kinematics import SlotRobot

# a planning of minutes runs no Kit frame; after one, the first camera read came back empty (replicator's TypeError)
# or an articulation view was gone, losing the configuration (collection logs: the planning before such a failure
# took 465 s at the median, 154 s elsewhere): the main thread renders this often while it serves the planners
RENDER_EVERY_S = 20.0




class BatchKinematics:
    def __init__(self, world, base, paths, offsets):
        from isaacsim.core.prims import Articulation

        self.world, self.base = world, base
        self.paths, self.offsets = list(paths), [np.asarray(o, float) for o in offsets]
        self.view = Articulation(self.paths)
        self.view.initialize()
        order = [str(p) for p in self.view.prim_paths]  # the articulation roots: the robot prim or one below it
        self.rows = [next(i for i, o in enumerate(order) if o == p or o.startswith(p + "/")) for p in self.paths]
        self.cv = threading.Condition()
        self.pending, self.running = {}, 0

    # --- the articulations ------------------------------------------------------------------------------------------
    def _state(self, Q):
        """Write the joint states {slot: q} and return {slot: (link transforms, Jacobians)} after one update."""
        slots = list(Q)
        rows = np.array([self.rows[s] for s in slots])
        n = len(self.paths)
        full = np.asarray(self.view.get_joint_positions(), float).reshape(n, -1)
        for s in slots:
            full[self.rows[s]] = Q[s]
        self.view.set_joint_positions(full[rows], indices=rows)
        self.view.set_joint_velocities(np.zeros_like(full[rows]), indices=rows)
        try:
            self.world.physics_sim_view.update_articulations_kinematic()
        except Exception:  # noqa: BLE001
            self.world.step(render=False)
        tf = self.view._physics_view.get_link_transforms()
        tf = tf.cpu().numpy() if hasattr(tf, "cpu") else np.asarray(tf)
        J = self.view.get_jacobians()
        J = J.cpu().numpy() if hasattr(J, "cpu") else np.asarray(J)
        out = {}
        for s in slots:
            t = tf[self.rows[s]].copy()
            t[:, :3] -= self.offsets[s]  # the copy's frame -> the main scene's
            out[s] = (t, J[self.rows[s]])
        return out

    def _fk(self, reqs):
        st = self._state({s: q for s, q in reqs.items()})
        return {s: st[s][0] for s in reqs}

    def _ik(self, reqs):
        """Robot.solve_ik for every request at once (same steps, limits, weights and stopping)."""
        b = self.base
        jobs = {}
        for s, r in reqs.items():
            arm = b.spec["arms"][r["side"]]
            cols = r["cols"] or arm["ik_joints"]
            ci = [b.idx[c] for c in cols]
            w = np.ones(len(cols))
            for rx, v in arm.get("ik_weights", {}).items():
                w[[i for i, c in enumerate(cols) if re.search(rx, c)]] = \
                    float(v) * (r["low_weight_scale"] if v < 1 else 1.0)
            lo, hi = b.lower[ci].copy(), b.upper[ci].copy()
            for j, c in enumerate(cols):
                if c in arm.get("ik_limits", {}):
                    lo[j], hi[j] = max(lo[j], arm["ik_limits"][c][0]), min(hi[j], arm["ik_limits"][c][1])
            q = r["q0"].copy()
            q[ci] = np.clip(q[ci], lo, hi)
            jobs[s] = dict(r, arm=arm, ci=ci, w=w, lo=lo, hi=hi, q=q, err=None, tf=None, k=0, done=False,
                           off=np.asarray(arm.get("tcp_offset", (0, 0, 0)), float),
                           row=b.body_names.index(arm["tcp_link"]) - (1 if b.fixed_base else 0),
                           jcols=[b._col0 + b.idx[c] for c in cols])
        while True:
            live = {s: j for s, j in jobs.items() if not j["done"] and j["k"] < j["iters"]}
            if not live:
                break
            st = self._state({s: j["q"] for s, j in live.items()})
            for s, j in live.items():
                tf, J_all = st[s]
                j["tf"], j["k"] = tf, j["k"] + 1
                t = tf[b.body_names.index(j["arm"]["tcp_link"])]
                R = Rotation.from_quat(t[3:7]).as_matrix()
                rr = R @ j["off"]
                ep = j["p"] - (t[:3] + rr)
                er = Rotation.from_matrix(j["R"] @ R.T).as_rotvec()
                j["err"] = (float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
                if j["err"][0] < j["tol"][0] and j["err"][1] < j["tol"][1]:
                    j["done"] = True
                    continue
                Jm = J_all[j["row"]][:, j["jcols"]].copy()
                if j["off"].any():  # velocity of a point rigidly attached to the link: v + w x r
                    Jm[:3] -= np.array([[0, -rr[2], rr[1]], [rr[2], 0, -rr[0]], [-rr[1], rr[0], 0]]) @ Jm[3:]
                Jm[3:] *= j["rot_weight"]
                JW = Jm * j["w"]
                dq = JW.T @ np.linalg.solve(JW @ Jm.T + j["damping"] ** 2 * np.eye(6),
                                            np.concatenate([ep, j["rot_weight"] * er]))
                j["q"][j["ci"]] = np.clip(j["q"][j["ci"]] + np.clip(dq, -0.15, 0.15), j["lo"], j["hi"])
        return {s: (j["q"], j["err"], j["tf"]) for s, j in jobs.items()}

    # --- the threads ------------------------------------------------------------------------------------------------
    def request(self, slot, kind, payload):
        """Called from a planning thread: hand a request to the main thread and wait for its answer."""
        ev, box = threading.Event(), {}
        with self.cv:
            self.pending[slot] = (kind, payload, ev, box)
            self.running -= 1
            self.cv.notify_all()
        ev.wait()
        if "exc" in box:
            raise box["exc"]
        return box["out"]

    def _serve(self):
        reqs, self.pending = self.pending, {}
        by = {"ik": {}, "fk": {}}
        for s, (kind, payload, _, _) in reqs.items():
            if kind in by:
                by[kind][s] = payload
        out = {}
        try:
            if by["ik"]:
                out.update(self._ik(by["ik"]))
            if by["fk"]:
                out.update(self._fk(by["fk"]))
        except Exception as e:  # noqa: BLE001 -- every waiting thread sees it
            for _, _, ev, box in reqs.values():
                box["exc"] = e
                ev.set()
            self.running += len(reqs)
            return
        for s, (kind, payload, ev, box) in reqs.items():
            try:
                if kind == "call":
                    fn, a, k = payload
                    box["out"] = fn(*a, **k)
                else:
                    box["out"] = out[s]
            except Exception as e:  # noqa: BLE001
                box["exc"] = e
        self.running += len(reqs)
        for _, _, ev, _ in reqs.values():
            ev.set()

    def run(self, units, n_slots=None, stop=None, on_result=None):
        """Run the callables of `units` (an iterator; each f(slot_robot) -> result) on up to n_slots copies at once,
        a new one in a slot as soon as its last returns, until `units` is exhausted or stop() is true; returns
        (results in completion order, whether `units` ran out); on_result(res) sees each as it comes. An exception in
        a unit is raised here once every running unit has ended."""
        n_slots = min(n_slots or len(self.paths), len(self.paths))
        robots = [SlotRobot(self, i) for i in range(n_slots)]
        results, errors, free = [], [], list(range(n_slots))
        threads, done = {}, []
        units = iter(units)
        exhausted = False
        rendered = time.time()

        def start(slot, unit):
            def body():
                try:
                    res = unit(robots[slot])
                    with self.cv:
                        done.append((slot, res, None))
                except Exception as e:  # noqa: BLE001
                    with self.cv:
                        done.append((slot, None, e))
                finally:
                    with self.cv:
                        self.running -= 1
                        self.cv.notify_all()

            with self.cv:
                self.running += 1
            th = threading.Thread(target=body, daemon=True)
            threads[slot] = th
            th.start()

        while True:
            while free and not exhausted and not errors and not (stop and stop()):
                unit = next(units, None)
                if unit is None:
                    exhausted = True
                    break
                start(free.pop(), unit)
            if not threads:
                break
            with self.cv:
                self.cv.wait_for(lambda: self.running == 0)
                finished, done[:] = list(done), []
            for slot, res, err in finished:
                threads.pop(slot).join()
                free.append(slot)
                if err is not None:
                    errors.append(err)
                else:
                    results.append(res)
                    if on_result:
                        on_result(res)
            with self.cv:
                if self.pending:
                    self._serve()
            if time.time() - rendered > RENDER_EVERY_S:  # the renderer kept alive (render only, no physics step)
                self.world.render()
                rendered = time.time()
        if errors:
            raise errors[0]
        return results, exhausted
