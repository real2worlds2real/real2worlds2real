"""Presentation cameras (views.json) and a default light rig (lights.json) derived from the scene graph and task frame.

    python -m calibrated_scene_reconstruction.recon.autoviews sessions/<name> [--force]

Writes configs/<name>/{views.json, lights.json}; one edited by hand or by the agent (its `_note` changed) is kept
unless --force. Cameras orbit the target close
to the photo camera's own direction (a single photo only textures what it saw; the far side of a room is filled in, so
a view swung round to it looks synthetic): third_person (the recording camera) turned from the photo direction toward
the robot's side by at most 40 deg, overview (higher, behind the photo camera), oblique, novel views (+-20 / 35 deg,
low, close-up of the target), and a fly-through that starts and ends at the photo camera. All are clamped inside the
room."""
import argparse
import json

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame


def target_center(s, sg):
    """Target position: the aligned twin mesh if reconstructed, else the target instance's points."""
    t = sg["task"]["target"]
    fix = sg['dynamic_objects'][t].get('fix') or {}
    if fix.get('mesh') and fix.get('xy') is not None and 'support_z' in fix:
        with np.load(resolve(fix['mesh'])) as data:
            V = np.asarray(data['V'])
        a = np.radians(fix.get('yaw_deg',0.))
        R = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        V = V@R.T + [*fix['xy'],fix['support_z']]
        return (V.min(0)+V.max(0))/2
    p = s / "objects" / t / "mesh_cam.glb"
    if p.exists():
        import trimesh

        V = np.asarray(trimesh.load(p, force="mesh").vertices)
        T = np.asarray(s.world_frame()[2])
        V = V @ T[:3, :3].T + T[:3, 3]
        return (V.min(0) + V.max(0)) / 2
    geo, masks = s.geometry(), s.masks()
    m = np.zeros(masks.shape[1:], bool)
    for i in sg["dynamic_objects"][t]["ids"]:
        m |= masks[i]
    R_wc, o, _ = s.world_frame()
    P = (geo["points"][m & geo["mask"]] - o) @ R_wc.T
    return np.median(P, 0)


def photo_transform(s):
    cameras=s/'parsing/posed_frames/cameras.npz'
    keyframes=s/'capture/keyframes.json'
    if cameras.exists() and keyframes.exists():
        photo=f"0_{json.loads(keyframes.read_text())['photo']:05d}.jpg"
        with np.load(cameras) as data:
            names=list(map(str,data['names']))
            if photo not in names:
                raise ValueError('Presentation photo is missing from registered metric cameras')
            i=names.index(photo);T=np.eye(4)
            T[:3,:3]=data['R_world_cam'][i];T[:3,3]=data['centre'][i]
        return T
    return np.asarray(s.world_frame()[2])


def observation_frames(s,name,names):
    """Review fitted objects in their actual silhouette observations, excluding rejected identities."""
    spec=s.scene_graph.get('dynamic_objects',{}).get(name,{})
    path=s/'splat/object_fit'/f'{name}.npz'
    if not path.exists():
        if (spec.get('fit') or {}).get('model') or (spec.get('fix') or {}).get('model'):
            raise ValueError('Blender object closeups need its reviewed observation cache: '+name)
        return list(range(len(names)))
    with np.load(path,allow_pickle=True) as cache:
        frames=np.asarray(cache['frames'])
    if (frames.ndim!=1 or frames.dtype.kind not in 'iu' or (frames<0).any() or (frames>=len(names)).any()):
        raise ValueError('Invalid observation camera indices for '+name)
    rejected=(spec.get('fit') or {}).get('observation_rejections',{})
    selected=[int(i) for i in frames if str(names[int(i)]) not in rejected]
    if not selected:raise ValueError('No reviewed object observations for '+name)
    return list(dict.fromkeys(selected))


def build(s: Session):
    sg = s.scene_graph
    tc = target_center(s, sg)
    tf = TaskFrame(sg, tc)
    floor = np.asarray(sg["structure"]["floor"]["range"], float)
    ceil = float(next((v["at"] for k, v in sg["structure"].items() if v.get("role", k) == "ceiling"), 2.5))
    lo, hi = floor[:, 0] + 0.3, floor[:, 1] - 0.3

    T = photo_transform(s)
    cam = T[:3, 3]
    d0 = (cam - tc)[:2]
    dist0 = float(np.linalg.norm(d0))
    d0 = d0 / max(dist0, 1e-6)
    robot = -tf.forward[:2] / max(np.linalg.norm(tf.forward[:2]), 1e-6)  # the side the robot stands on
    ang = float(np.degrees(np.arctan2(d0[0] * robot[1] - d0[1] * robot[0], d0 @ robot)))
    toward = float(np.sign(ang) * min(40.0, abs(ang)))  # third person: toward the robot, at most 40 deg off

    def orbit(az_deg, dist, z, look_dz=0.0):
        a = np.radians(az_deg)
        d = np.array([np.cos(a) * d0[0] - np.sin(a) * d0[1], np.sin(a) * d0[0] + np.cos(a) * d0[1]])
        p = np.array([*(tc[:2] + dist * d), min(z, ceil - 0.1)])
        p[:2] = np.clip(p[:2], lo, hi)
        return [[round(float(v), 3) for v in p], [round(float(v), 3) for v in (tc + [0, 0, look_dz])]]

    near = float(np.clip(dist0 * 0.6, 1.3, 2.2))
    views = {
        "_note": "Generated by calibrated_scene_reconstruction.recon.autoviews around the photo camera's direction (eye, target in world metres).",
        "third_person": orbit(toward, near, 1.7),
        "overview": orbit(0.0, min(dist0 * 0.9, near + 1.0), 2.45, -0.2),
        "oblique": orbit(-np.sign(toward or 1.0) * 25.0, near + 0.2, 1.75),
        "novel": {
            "front_low": orbit(0.0, near - 0.1, 1.25),
            "left_side": orbit(35.0, near, 1.6),
            "right_side": orbit(-35.0, near, 1.6),
            "target_close": orbit(toward * 0.5, 0.55, tc[2] + 0.3),
        },
        "flythrough": [["photo"], orbit(0.0, near, 1.5), orbit(25.0, near, 1.65), orbit(toward, near - 0.2, 1.6),
                       orbit(-25.0, near, 1.65), ["photo"]],
    }
    nx = int(np.clip(round((floor[0, 1] - floor[0, 0]) / 2.2), 1, 3))
    ny = int(np.clip(round((floor[1, 1] - floor[1, 0]) / 2.2), 1, 2))
    disks = [[round(float(x), 2), round(float(y), 2)]
             for x in np.linspace(floor[0, 0], floor[0, 1], nx + 2)[1:-1]
             for y in np.linspace(floor[1, 0], floor[1, 1], ny + 2)[1:-1]]
    lights = {"_note": 'Default lighting rig with ceiling disks on a grid.',
              "lights": {"bounce": 1500, "front": 3000, "disk": 30000, "window": 9000, "ceiling_disks": disks}}
    return views, lights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["views.json", "lights.json"], help="write just this one")
    a = ap.parse_args()
    s = Session(a.session)
    views, lights = build(s)
    for name, d in (("views.json", views), ("lights.json", lights)):
        if a.only and name != a.only:
            continue
        p = s.config_dir / name
        if p.exists() and not a.force and json.loads(p.read_text()).get("_note") != d["_note"]:
            print(f"keep {p} (edited)")
            continue
        p.write_text(json.dumps(d, indent=1) + "\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
