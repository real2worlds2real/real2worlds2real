"""Semantic QA of a built scene, from the placed geometry itself (assets.shape on world points), so it also catches
a library model whose authored orientation differs from what the placement rule assumed:
  seat     seat height suits the nearest support (assets.shape.seat_range: 0.40 m below its top.. its slab underside
           - KNEE_ROOM); a library seat's backrest points away from it (within 35 deg) -- a reconstructed seat stands
           as the photo shows it;
  frame    (a library swap with an easel) its front faces the photo camera (within 60 deg);
  long     (an elongated library swap) its long axis follows the twin instance's (within 25 deg);
  handle   (a target with a handle) it points away from the hand: the lateral grasp comes from the robot's right of the
           object, so the handle points to the task frame's left, within 45 deg."""
import numpy as np
from pxr import UsdGeom

from ..assets.library import model_profile
from ..assets.shape import angle, footprint_axes, nearest_counter, profile, seat_range, toward_box, wrap
from .clearance import world_points
from .sweep import children
from calibrated_scene_reconstruction.robots.taskframe import TaskFrame


def deg(a):
    return round(float(np.degrees(abs(a))), 1)


def check(ts, twin_swaps=None):
    boxes = ts.manifest["collision_boxes"]
    out = []
    for prim in children(ts.stage, "/World/Support"):
        P = world_points(prim, 20000)
        p = profile(P)
        c = (P.min(0) + P.max(0))[:2] / 2
        name, lo, hi = nearest_counter(c, boxes)
        s0, s1 = seat_range(boxes, name, lo, hi)
        item = prim.GetName()
        out.append({"item": item, "check": "seat height", "value_m": round(p["seat_h"], 3),
                    "ok": s0 - 0.02 <= p["seat_h"] <= s1 + 0.02})
        if p["back"] and not prim.GetCustomDataByKey("calibrated_scene_reconstruction:reconstructed"):
            err = wrap(angle(p["back"]) - angle(-toward_box(c, lo, hi)))
            out.append({"item": item, "check": f"backrest away from {name}", "err_deg": deg(err), "ok": deg(err) <= 35})
    cam = ts.T_wc[:2, 3]
    for prim in children(ts.stage, "/World/Static"):
        if prim.IsA(UsdGeom.Mesh):  # reconstruction / template: the photo's own layout
            continue
        item = prim.GetName()
        if prim.GetCustomDataByKey("calibrated_scene_reconstruction:kit") and not (ts.s / "objects" / item / "mesh_cam.glb").exists():
            continue  # a set piece the agent added (scene graph set_pieces): no photo object to compare it with
        P = world_points(prim, 20000)
        p = profile(P)
        c = (P.min(0) + P.max(0))[:2] / 2
        if p["stand"]:
            err = wrap(angle(-np.asarray(p["stand"])) - angle(cam - c))
            out.append({"item": item, "check": "front faces the camera", "err_deg": deg(err), "ok": deg(err) <= 60})
        elif p["long"] is not None:
            base = (twin_swaps or {}).get(item)
            if base and model_profile(base["model"])["long"] is not None:  # the twin's own library model, as placed
                a, ratio = base["yaw"] + model_profile(base["model"])["long"], 2.0
            else:
                a, ratio = footprint_axes(ts.object_mesh_world(item)[1])
            if ratio > 1.3:
                err = wrap(p["long"] - a, np.pi)
                out.append({"item": item, "check": "long axis as in the photo", "err_deg": deg(err), "ok": deg(err) <= 25})
    target = ts.s.scene_graph["task"]["target"]
    prim = ts.stage.GetPrimAtPath(f"/World/Objects/{target}")
    if prim.IsValid():
        P = world_points(prim, 20000)
        h = profile(P)["handle"]
        if h:
            left = TaskFrame(ts.s.scene_graph, (P.min(0) + P.max(0)) / 2).left
            err = wrap(angle(h) - angle(left[:2]))
            out.append({"item": target, "check": "handle away from the hand", "err_deg": deg(err), "ok": deg(err) <= 45})
    return out


def summarize(rep):
    bad = [r for r in rep if not r["ok"]]
    return f"semantics: {len(rep) - len(bad)}/{len(rep)} ok" + (
        " | " + "; ".join(f"{r['item']} {r['check']} " + (f"{r['err_deg']}deg" if "err_deg" in r else f"{r['value_m']}m")
                          for r in bad) if bad else "")
