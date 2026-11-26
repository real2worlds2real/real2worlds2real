"""Preserve authored Blender transmission in native Isaac environment materials.

Blender's USD Preview Surface export can drop Principled transmission while
leaving opacity at one. Recover only explicit, constant material properties
recorded by the builder. Do not infer glass from colour, names or source pixels.
"""
import json
from pathlib import Path

import numpy as np
from pxr import Tf, Usd, UsdShade


def apply(native, manifest):
    from calibrated_scene_reconstruction.sim.object_materials import surface_material
    stage = Usd.Stage.Open(str(native))
    records = json.loads(Path(manifest).read_text())
    materials = [UsdShade.Material(p) for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    report = []
    for name, props in records.items():
        matches = [m for m in materials if m.GetPrim().GetName() == Tf.MakeValidIdentifier(name)]
        if len(matches) != 1:
            raise ValueError('Optical source must identify one exported material: '+name)
        mat = matches[0]
        shader = mat.ComputeSurfaceSource()[0]
        if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            raise ValueError('Expected a source-bound Blender Preview Surface before optical conversion')
        # Verify source correspondence instead of converting a similarly named
        # but unrelated material. Mesh bindings and material paths stay intact.
        for key, expected in [('diffuseColor', props['color']), ('roughness', props['roughness']),
                              ('metallic', props['metallic']), ('ior', props['ior']), ('opacity', props['opacity'])]:
            inp = shader.GetInput(key)
            if not inp or inp.HasConnectedSource() or not np.allclose(inp.Get(), expected, rtol=1e-5, atol=1e-6):
                raise ValueError('Exported optical properties differ from the explicit Blender source: '+name+'/'+key)
        path = str(mat.GetPath())
        converted = surface_material(stage, path, props)
        mdl = converted.ComputeSurfaceSource('mdl')[0]
        if not mdl or not np.isclose(mdl.GetInput('specular_transmission_weight').Get(), props['transmission']):
            raise ValueError('Native material lost explicit transmission')
        report.append(dict(source_material=name, native_material=path,
                           transmission=props['transmission'], opacity=props['opacity'],
                           ior=props['ior'], roughness=props['roughness'],
                           mdl_shader=str(mdl.GetPath()), preview_is_approximate=True))
    stage.GetRootLayer().Save()
    return dict(materials=report, photorealism_approved=False,
                scope=__doc__.strip())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native');parser.add_argument('manifest');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(apply(args.native,args.manifest),indent=2)+'\n')


if __name__ == '__main__':
    main()
