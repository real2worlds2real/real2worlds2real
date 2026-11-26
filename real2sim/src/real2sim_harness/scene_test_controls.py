"""Explicit review actions applied to actual shared-world states, never cached animation."""

from __future__ import annotations

import numpy as np


class SceneTestControls:
    def __init__(self, runtime):
        import warp as wp

        self.runtime = runtime
        self.records = []
        self.terminal_targets = {}
        self.body_masses = runtime["model"].body_mass.numpy()
        self.rest = runtime["model"].particle_q
        for r in runtime["records"].values():
            action = r.get("scene_action")
            if not action:
                continue
            if action["type"] == "joint_targets":
                if any(j["joint"] not in r["joint_ids"] for j in action["joints"]):
                    raise ValueError("Scene action names an unknown joint")
                self.records.append((r, action, [], []))
                continue
            if not r.get("native_representation"):
                raise ValueError("A native_test action requires a retained native test configuration")
            c = r["native_configuration"]
            drives = []
            for d in c.get("controls", []):
                R = np.asarray(r["transform"])[:3, :3]
                mask = R @ np.diag(d.get("position_axes", [False]*3)) @ R.T
                if np.max(np.abs(mask-np.diag(np.diag(mask)))) > 1e-6:
                    raise ValueError("This partial-axis review fixture needs an axis-compatible scene placement")
                drives.append((d, np.diag(mask), wp.zeros(2, dtype=wp.int32),
                               wp.zeros(1, dtype=float), wp.full(1, -1., dtype=float)))
            grasps = [(g, wp.array(np.asarray(g["indices"],np.int32)+r["particle_start"], dtype=wp.int32))
                      for g in c.get("grasps", [])]
            self.records.append((r, action, drives, grasps))

    def joint_targets(self, values, q_starts, time):
        from .strand_contract import translation_at

        for r, action, _, _ in self.records:
            if action["type"] != "joint_targets":
                continue
            t = max(0., time-action["start_seconds"])
            for joint in action["joints"]:
                # The existing smooth translation helper supplies the same
                # interpolation for native-coordinate joint targets.
                keys = [[k[0],k[1]-joint["keyframes"][0][1],0,0] for k in joint["keyframes"]]
                value = joint["keyframes"][0][1]+translation_at(keys,t)[0]
                values[q_starts[r["joint_ids"][joint["joint"]]]] = value

    def apply(self, a, b, time, dt):
        import warp as wp

        from .native_controls import apply_wrench, move_grasp, move_probe, track_contact_loss
        from .strand_contract import translation_at

        contacts = self.runtime["solver"].contacts
        for r, action, drives, grasps in self.records:
            if action["type"] != "native_test":
                continue
            t = time-action["start_seconds"]
            c = r["native_configuration"]
            placement = np.asarray(r["transform"])
            R, origin = placement[:3,:3], placement[:3,3]
            duration = (c["frames"]-1)/c["fps"]
            for i, (d, mask, state, timer, event) in enumerate(drives):
                active = d.get("start_s",0) <= t < d.get("end_s",duration)
                last_for_body = not any(other[0]["body"] == d["body"] for other in drives[i+1:])
                if action["hold_last_controls"] and last_for_body and t >= d.get("end_s",duration):
                    active = True
                    body = r["body_ids"][d["body"]]
                    key = (r["id"], body)
                    if key not in self.terminal_targets:
                        world = a.body_q.numpy()[body,:3]
                        self.terminal_targets[key] = R.T@(world-origin)
                    # An explicit review grip holds the reached location using
                    # bounded forces; continuing the lift velocity forever would
                    # send detached parts away during later scene operations.
                    d = dict(d, position_axes=[True]*3,
                             position_target_m=self.terminal_targets[key].tolist(),
                             position_stiffness=max(d.get("position_stiffness",0),200.),
                             linear_velocity_m_s=[0,0,0], angular_velocity_rad_s=[0,0,0],
                             force_n=(R.T@np.array([0,0,9.81*self.body_masses[body]])).tolist())
                    mask = np.ones(3)
                if not active:
                    continue
                stop = d.get("stop_rotation_after_contact_loss")
                if stop:
                    wp.launch(track_contact_loss, dim=1, inputs=[contacts.rigid_contact_count,
                        contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                        r["native_shape_ids"][d["body"]],r["native_shape_ids"][stop["other_body"]],
                        dt,stop.get("dwell_seconds",.05),t,state,timer,event])
                def vec(key, default=(0,0,0), R=R, d=d):
                    return wp.vec3(*(R@np.asarray(d.get(key,default),float)))
                limits = np.abs(R)@np.asarray(d.get("force_limit_n",[0,0,0]))
                torque_limits = np.abs(R)@np.asarray(d.get("torque_limit_nm",[0,0,0]))
                target = R@np.asarray(d.get("position_target_m",[0,0,0]))+origin
                wp.launch(apply_wrench, dim=1, inputs=[a.body_q,a.body_qd,a.body_f,r["body_ids"][d["body"]],state,
                    wp.vec3(*target),wp.vec3(*mask),vec("linear_velocity_m_s"),vec("force_n"),
                    vec("angular_velocity_rad_s"),vec("torque_nm"),d.get("position_stiffness",0.),
                    d.get("linear_damping",0.),d.get("angular_gain",0.),d.get("alignment_stiffness",0.),
                    vec("alignment_axis_world",(0,0,1)),wp.vec3(*limits),wp.vec3(*torque_limits)])
            for g, ids in grasps:
                offset = R@translation_at(g["keyframes"],max(t,0))
                wp.launch(move_grasp,dim=len(g["indices"]),inputs=[ids,self.rest,wp.vec3(*offset),dt,
                    t>=g.get("release_seconds",np.inf) or (t>duration and not action["hold_last_controls"]),self.runtime["model"].particle_flags,
                    a.particle_q,b.particle_q,a.particle_qd,b.particle_qd])
            for p in c.get("probes",[]):
                if p["id"] not in r.get("native_probe_ids",{}):
                    continue
                if not np.any(np.asarray(p["keyframes"], dtype=float)[:, 1:]):
                    # The builder already placed this stationary kinematic fixture.
                    continue
                target = R@(np.asarray(p["position_m"])+translation_at(p["keyframes"],max(t,0)))+origin
                wp.launch(move_probe,dim=1,inputs=[r["native_probe_ids"][p["id"]],wp.vec3(*target),dt,
                    a.body_q,b.body_q,a.body_qd,b.body_qd])
