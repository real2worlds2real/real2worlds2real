"""One file for a USD export: apps/scene/export_usd.py's folder (the scene, textures/, the splat's background.usdz) packed into
a single USDZ.

    envs/splat/3dgrut/.venv/bin/python apps/scene/pack_usdz.py sessions/<name>/export/usd/<tag>     (any Python with pxr)

The splat's NuRec package is unpacked into the new one (splat/default.usda, gauss.usda, background.nurec) and the
scene's reference to it pointed there: a package inside a package is not something every reader follows, files inside
one are. UsdUtils.CreateNewUsdzPackage then gathers every layer and asset the scene names (textures, the NuRec data)
into <name>_<tag>.usdz beside the folder; the stage is opened from it once to check that nothing is left outside.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("folder")
args = ap.parse_args()

from pxr import Sdf, Usd, UsdUtils  # noqa: E402

folder = Path(args.folder).resolve()
usd = next(p for p in folder.glob("*.usd") if not p.name.startswith("."))
out = folder.parent / (usd.stem + ".usdz")
with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
    tmp = Path(tmp)
    for p in folder.iterdir():
        if p.is_dir():
            shutil.copytree(p, tmp / p.name)
        elif p.suffix in (".usd", ".usda", ".usdc"):
            shutil.copy2(p, tmp / p.name)
    root = tmp / usd.name
    layer = Sdf.Layer.FindOrOpen(str(root))
    bg = folder / "background.usdz"
    if bg.exists():
        with zipfile.ZipFile(bg) as z:
            z.extractall(tmp / "splat")
        n = 0

        def repoint(path):
            global n
            spec = layer.GetPrimAtPath(path)
            if spec is None:
                return
            items = list(spec.referenceList.prependedItems)
            if any(r.assetPath == "./background.usdz" for r in items):
                spec.referenceList.prependedItems = [Sdf.Reference("./splat/default.usda", r.primPath, r.layerOffset)
                                                     if r.assetPath == "./background.usdz" else r for r in items]
                n += 1
        layer.Traverse(Sdf.Path.absoluteRootPath, repoint)
        layer.Save()
        print(f"[pack] the splat unpacked into splat/, {n} reference(s) pointed at it", flush=True)
    # every file the export names is already beside it under a relative path (textures/, the splat's files); Kit's own
    # MDL modules (OmniPBR.mdl, gltf/pbr.mdl) are search-path assets a Kit app resolves itself, which made
    # UsdUtils.CreateNewUsdzPackage give up: the package is written file by file (the root layer first, aligned)
    out.unlink(missing_ok=True)
    w = Sdf.ZipFileWriter.CreateNew(str(out))
    w.AddFile(str(root), root.name)
    for f in sorted(tmp.rglob("*")):
        if f.is_file() and f != root:
            w.AddFile(str(f), str(f.relative_to(tmp)))
    w.Save()
with zipfile.ZipFile(out) as z:
    names = z.namelist()
st = Usd.Stage.Open(str(out))
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(out)))
unresolved = [u for u in unresolved if not u.endswith(".mdl")]  # Kit's own MDL library
print(f"done: {out} ({out.stat().st_size / 1e6:.0f} MB, {len(names)} files inside: "
      f"{sum(n.endswith('.png') or n.endswith('.jpg') for n in names)} textures, "
      f"{sum(n.endswith('.nurec') for n in names)} NuRec), {sum(1 for _ in st.TraverseAll())} prims, "
      f"unresolved {len(unresolved)}", flush=True)
for u in unresolved[:10]:
    print(f"[warn] unresolved: {u}")
