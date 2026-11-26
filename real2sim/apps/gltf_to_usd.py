"""Convert downloaded library models (Poly Haven glTF, ABO GLB) to USD and record up axis / size per model.

    tools/run_isaac.sh apps/gltf_to_usd.py [--force]
Only models without a USD yet are converted. Updates "usd" and "meta" ({up, mpu, size (z-up, metres)}) in
assets/polyhaven/index.json and assets/abo/usd_index.json.
"""
import argparse
import asyncio
import json



ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
FORCE = args.force

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
app.update()
import omni.kit.asset_converter as conv  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.core.paths import ABO_DIR, POLYHAVEN_DIR, rel, resolve  # noqa: E402
from calibrated_scene_reconstruction.sim.clearance import world_points  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import set_pose  # noqa: E402


async def convert(src, dst):
    ctx = conv.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.use_meter_as_world_unit = True
    task = conv.get_instance().create_converter_task(str(src), str(dst), None, ctx)
    ok = await task.wait_until_finished()
    return ok, "" if ok else task.get_error_message()


stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def convert_all(models, index, usd_dir):
    """models {name: glTF/GLB}; fills index["usd"] / index["meta"] in place."""
    index.setdefault("usd", {})
    index.setdefault("meta", {})
    for name, src in models.items():
        if name in index["usd"] and not FORCE and resolve(index["usd"][name]).exists():
            continue
        dst = usd_dir / name / f"{name}.usd"
        dst.parent.mkdir(parents=True, exist_ok=True)
        fut = asyncio.ensure_future(convert(resolve(src), dst))
        while not fut.done():
            app.update()
        ok, err = fut.result()
        if not ok:
            print(name, "FAILED", err)
            continue
        layer = Usd.Stage.Open(str(dst))
        up, mpu = UsdGeom.GetStageUpAxis(layer), UsdGeom.GetStageMetersPerUnit(layer)
        p = stage.DefinePrim(f"/Probe/{name}", "Xform")
        set_pose(p, [0, 0, 0], (0.7071068, 0.7071068, 0, 0) if up == "Y" else (1, 0, 0, 0), scale=mpu)
        stage.DefinePrim(f"/Probe/{name}/asset", "Xform").GetReferences().AddReference(str(dst))  # keep its root xform
        app.update()
        P = world_points(p, max_pts=20000)  # authored extents of converted assets can be stale
        size = (P.max(0) - P.min(0)).round(4).tolist()
        index["usd"][name] = rel(dst)
        index["meta"][name] = {"up": up, "mpu": mpu, "size": size}
        print(name, up, size, flush=True)


ph = json.loads((POLYHAVEN_DIR / "index.json").read_text())
convert_all(ph["models"], ph, POLYHAVEN_DIR / "usd")
(POLYHAVEN_DIR / "index.json").write_text(json.dumps(ph, indent=1))
if (ABO_DIR / "catalog.json").exists():
    cat = json.loads((ABO_DIR / "catalog.json").read_text())
    abo_idx = json.loads((ABO_DIR / "usd_index.json").read_text()) if (ABO_DIR / "usd_index.json").exists() else {}
    convert_all({n: c["file"] for n, c in cat.items() if resolve(c["file"]).exists()}, abo_idx, ABO_DIR / "usd")
    (ABO_DIR / "usd_index.json").write_text(json.dumps(abo_idx, indent=1))
app.close()
