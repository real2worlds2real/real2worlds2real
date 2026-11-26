"""Independent diagnostic layers; never mutate the cached candidate USD layer."""
from pathlib import Path

from pxr import Sdf, Usd


def copy_stage(source, output, name):
    """Copy an export, anchoring assets before changing any diagnostic setting.

    Stage.Open returns shared cached layers. Editing its root can contaminate
    later variants even without saving. Each diagnostic gets an anonymous deep
    layer copy. Preserve references, notably NuRec packages, instead of flattening
    their internal assets out of the package.
    """
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists() or source==output:raise ValueError('Use a new diagnostic output')
    if not isinstance(name,str) or not name.strip():raise ValueError('Explicit diagnostic label required')
    original=Sdf.Layer.FindOrOpen(str(source))
    if not original:raise ValueError('Cannot read source USD')
    layer=Sdf.Layer.CreateAnonymous('ablation.usda');layer.TransferContent(original)

    def anchor(path):
        if not path:return path
        # MDL modules resolved on Kit's search path are not filesystem assets.
        if path.endswith('.mdl') and '/' not in path:return path
        return Sdf.ComputeAssetPathRelativeToLayer(original,path)

    def visit(path):
        spec=layer.GetObjectAtPath(path)
        if isinstance(spec,Sdf.AttributeSpec):
            value=spec.default
            if isinstance(value,Sdf.AssetPath):spec.default=Sdf.AssetPath(anchor(value.path))
            elif isinstance(value,Sdf.AssetPathArray):spec.default=Sdf.AssetPathArray([Sdf.AssetPath(anchor(v.path)) for v in value])
        if isinstance(spec,Sdf.PrimSpec):
            refs=spec.referenceList.GetAppliedItems()
            if refs:spec.referenceList.explicitItems=[Sdf.Reference(anchor(r.assetPath),r.primPath,r.layerOffset,r.customData) for r in refs]
            payloads=spec.payloadList.GetAppliedItems()
            if payloads:spec.payloadList.explicitItems=[Sdf.Payload(anchor(p.assetPath),p.primPath,p.layerOffset) for p in payloads]

    layer.Traverse(Sdf.Path.absoluteRootPath,visit)
    layer.subLayerPaths=[anchor(p) for p in layer.subLayerPaths]
    metadata=dict(layer.customLayerData);metadata['r2s2r_diagnostic']=name
    metadata['r2s2r_ablation_source']=str(source);layer.customLayerData=metadata
    output.parent.mkdir(parents=True,exist_ok=True)
    layer.Export(str(output))
    return Usd.Stage.Open(str(output))
