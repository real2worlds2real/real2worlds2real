"""Episodes of the Newton rope scenes (apps/newton_rope_letter.py --record) written in apps/episode.py's raw layout,
for calibrated_scene_reconstruction.datagen.export.lerobot:

    <session>/episodes/<tag>/raw/frames/<cam>.mp4   the cameras, rendered on the GPU by Newton's SensorTiledCamera
    <session>/episodes/<tag>/raw/steps.npz          per frame: t, q, qd, action (the joint targets set over the next
                                                    frame: the arm's PD trails them), target, tcp_pose[_1],
                                                    object_pose, segment, cloth_points (the rope's segments),
                                                    cam_<cam>_T_world
    <session>/episodes/<tag>/raw/meta.json          robot, arms, fps, joint_names, cameras {K, size}, task, success, ...

Lengths in metres, quaternions (w, x, y, z), cameras' T_world with OpenCV axes, the fingers' joints in metres: the
scenes' centimetres are converted here. Only for the Newton container (warp, newton).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import newton
import numpy as np
import warp as wp

from calibrated_scene_reconstruction.sim.newton.video import Recorder


from calibrated_scene_reconstruction.core.geometry import look_at_matrix as look_at


def mat_to_quat_xyzw(R):
    """A rotation matrix's quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return np.asarray(q, float)


class EpisodeRecorder:
    """Cameras {name: (eye, target, vertical fov deg)} in the scene's units; `unit` m per scene unit. Every call of
    capture() renders the cameras (every world, then per pixel the nearest hit: the robots are separate worlds) and
    logs the state. `look`: calibrated_scene_reconstruction.tasks.rope.appearance.scene_look's draw (its moved cameras replace `cams`); None: the
    fixed look (a grey table, the default light). The look may also carry "shape_colours" ({shape index: rgb}, e.g. a
    rope's) and "decal": a goal drawn on the table top ({"lines": the polylines (the scene's units, x, y), "z": the
    table top's height, "width", "colour"}), painted where the table shows (an arm or the rope over it hides it),
    times the table's shading. The points logged are ex.object_points() (metres)."""

    def __init__(self, ex, session, tag, cams, size=(384, 240), fps=30, unit=0.01, cloth_stride=4, look=None):
        from newton.sensors import SensorTiledCamera

        self.ex, self.unit, self.fps, self.cloth_stride = ex, unit, fps, cloth_stride
        self.W, self.H = size
        self.raw = Path(session) / "episodes" / tag / "raw"
        if self.raw.exists():
            shutil.rmtree(self.raw)
        (self.raw / "frames").mkdir(parents=True)
        self.tag, self.look = tag, look
        if look is not None:
            cams = look["cameras"]
        m = ex.model
        self.sensor = SensorTiledCamera(model=m)
        shadows = True if look is None else look["shadows"]
        self.sensor.default_render_config.enable_shadows = shadows
        self.sensor.utils.create_default_light(enable_shadows=shadows, direction=None if look is None else
                                               wp.vec3f(*map(float, look["light_direction"])))
        self.names = list(cams)
        fovs = [float(np.radians(c[2])) for c in cams.values()]
        self.rays = self.sensor.utils.compute_camera_rays_pinhole(self.W, self.H, camera_fovs=fovs)
        n_cam = len(self.names)
        self.color = self.sensor.utils.create_color_image_output(self.W, self.H, n_cam)
        self.depth = self.sensor.utils.create_depth_image_output(self.W, self.H, n_cam)
        self.worlds = max(1, int(getattr(m, "world_count", 1)))
        self.T_world = {}  # per camera: world from camera, OpenCV axes, metres
        self.specs, self.K = cams, {}
        for k, c in cams.items():
            f = self.H / 2 / np.tan(np.radians(c[2]) / 2)
            self.K[k] = [[f, 0.0, self.W / 2], [0.0, f, self.H / 2], [0.0, 0.0, 1.0]]
        self.xf = wp.zeros((n_cam, self.worlds), dtype=wp.transformf)  # (cameras, worlds)
        self.place_cameras()
        clear = 0xFFC8C8C8
        if look is not None:  # packed as the image's bytes: r, g, b, a
            r, g, b = (int(round(255 * float(v))) for v in look["background"])
            clear = 0xFF000000 | b << 16 | g << 8 | r
        self.clear = type(SensorTiledCamera.GRAY_CLEAR_DATA)(clear_color=clear, clear_albedo=0xFF000000)
        # a grey table (or the look's colours)
        if m.shape_color is not None and getattr(ex, "table_shape_idx", None) is not None:
            sc = m.shape_color.numpy()
            sc[ex.table_shape_idx] = (0.55, 0.57, 0.6) if look is None else look["table"]
            if look is not None:
                planes = m.shape_type.numpy() == int(newton.GeoType.PLANE)
                planes[ex.table_shape_idx] = False  # a scene whose table top is a plane keeps the table's colour
                sc[planes] = look["floor"]
                for i, c in (look.get("shape_colours") or {}).items():
                    sc[int(i)] = c
            self.table_rgb = np.asarray(sc[ex.table_shape_idx], float)
            m.shape_color.assign(sc)
        self.gain = None if look is None else np.asarray(look["gain"], np.float32)
        self.decal = None if look is None else look.get("decal")
        if self.decal is not None:
            self.shape_index = self.sensor.utils.create_shape_index_image_output(self.W, self.H, n_cam)
            self.decal_mask = self.decal_pixels()
        self.writers = {k: Recorder(self.raw / "frames" / f"{k}.mp4", self.W, self.H, fps=fps) for k in self.names}
        self.log = {k: [] for k in ("q", "qd", "target", "tcp_pose", "object_pose", "segment", "cloth_points")}
        self.log.update({f"cam_{k}_T_world": [] for k in self.names})
        for i in range(1, len(ex.arms)):
            self.log[f"tcp_pose_{i}"] = []

    def place_cameras(self):
        """Every camera's pose in the world (self.poses: (eye, rotation), the scene's units; self.xf, self.T_world)."""
        self.poses, xf = [], []
        for k, c in self.specs.items():
            eye, target = np.asarray(c[0], float), np.asarray(c[1], float)
            R = look_at(eye, target)
            self.poses.append((eye, R))
            xf.append([*eye, *mat_to_quat_xyzw(R)])
            T = np.eye(4)
            T[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
            T[:3, 3] = eye * self.unit
            self.T_world[k] = T
        self.xf.assign(np.repeat(np.asarray(xf, np.float32)[:, None], self.worlds, axis=1))

    def decal_pixels(self):
        """(cameras, H, W): the pixels whose ray meets the table top's plane on the decal's lines."""
        from calibrated_scene_reconstruction.assets.polyline import distance_to

        rays = self.rays.numpy().reshape(len(self.poses), self.H, self.W, 2, 3)
        out = np.zeros((len(self.poses), self.H, self.W), bool)
        lines = [np.asarray(ln, float) for ln in self.decal["lines"]]
        for c, (eye, R) in enumerate(self.poses):
            o = rays[c, :, :, 0] @ R.T + eye
            d = rays[c, :, :, 1] @ R.T
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (float(self.decal["z"]) - o[..., 2]) / d[..., 2]
            ok = np.isfinite(t) & (t > 0)
            p = (o + t[..., None] * d)[ok][:, :2]
            hit = np.zeros(ok.sum(), bool)
            box = np.concatenate(lines)
            near = (p >= box.min(0) - self.decal["width"]).all(1) & (p <= box.max(0) + self.decal["width"]).all(1)
            hit[near] = distance_to(p[near], lines) < self.decal["width"] / 2
            out[c][ok] = hit
        return out

    def images(self):
        st = self.ex.state_0
        self.ex.model.bvh_refit_shapes(st)
        if self.ex.model.particle_count:
            self.ex.model.bvh_refit_particles(st)
        self.sensor.update(st, self.xf, self.rays, color_image=self.color, depth_image=self.depth,
                           shape_index_image=self.shape_index if self.decal is not None else None, clear_data=self.clear)
        rgba = self.color.numpy().view(np.uint8).reshape(self.worlds, len(self.names), self.H, self.W, 4)  # (world, cam, ...)
        d = self.depth.numpy()
        d = np.where(d > 0, d, np.inf)
        near = np.argmin(d, axis=0)  # (cams, H, W): the world whose hit is nearest
        out = np.take_along_axis(rgba[..., :3], near[None, ..., None], axis=0)[0]
        if self.decal is not None:  # the goal on the table, where the table shows, times its shading
            si = np.take_along_axis(self.shape_index.numpy(), near[None], axis=0)[0]
            m = self.decal_mask & (si == self.ex.table_shape_idx)
            if m.any():
                lin = (out[m] / 255.0) ** 2.2
                shade = lin.mean(1, keepdims=True) / max(float((self.table_rgb ** 2.2).mean()), 1e-3)
                out = out.copy()
                out[m] = np.clip(255.0 * (np.asarray(self.decal["colour"]) ** 2.2 * shade) ** (1 / 2.2), 0, 255).astype(np.uint8)
        if self.gain is not None:  # the look's exposure and white balance
            out = np.clip(out * self.gain, 0, 255).astype(np.uint8)
        return {k: out[i] for i, k in enumerate(self.names)}

    def joints(self, arr):
        """Each arm's joints (radians; the fingers, scene units, in metres), concatenated."""
        out = []
        for arm in self.ex.arms:
            q = np.asarray(arr[arm.dofs], float).copy()
            q[-2:] *= self.unit
            out.append(q)
        return np.concatenate(out)

    def capture(self, phase_index):
        self.last = self.images()  # what a policy sees this frame
        for k, im in self.last.items():
            self.writers[k].write(im)
        ex = self.ex
        self.log["q"].append(self.joints(ex.state_0.joint_q.numpy()))
        self.log["qd"].append(self.joints(ex.state_0.joint_qd.numpy()))
        self.log["target"].append(self.joints(ex.control.joint_target_q.numpy()))
        body_q = ex.state_0.body_q.numpy()
        for i, arm in enumerate(ex.arms):
            p, q = ex.tcp(body_q, arm)
            self.log["tcp_pose" if i == 0 else f"tcp_pose_{i}"].append(np.concatenate([p * self.unit, [q[3], q[0], q[1], q[2]]]))
        P = ex.object_points()
        self.log["object_pose"].append(np.concatenate([P.mean(0), [1.0, 0.0, 0.0, 0.0]]))
        self.log["cloth_points"].append(P[:: self.cloth_stride].astype(np.float32))
        self.log["segment"].append(int(phase_index))
        for k in self.names:
            self.log[f"cam_{k}_T_world"].append(self.T_world[k])

    def close(self, meta):
        for w in self.writers.values():
            w.close()
        n = len(self.log["q"])
        q = np.asarray(self.log["q"], np.float32)
        a = np.asarray(self.log["target"], np.float32)  # the targets set over the next frame (logged at its end)
        action = np.concatenate([a[1:], a[-1:]])
        arrays = {k: np.asarray(v, np.float32 if k != "segment" else np.int64) for k, v in self.log.items()}
        np.savez_compressed(self.raw / "steps.npz", t=np.arange(n) / self.fps, action=action, **arrays)
        cams = {k: {"K": self.K[k], "size": [self.W, self.H]} for k in self.names}
        if self.look is not None:
            meta = {**meta, "look": {k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                                     for k, v in self.look.items() if k not in ("cameras", "shape_colours")}}
        (self.raw / "meta.json").write_text(json.dumps({"fps": self.fps, "cameras": cams, "tag": self.tag, **meta},
                                                       indent=1, default=float))
        print(f"recorded {n} frames to {self.raw}", flush=True)
