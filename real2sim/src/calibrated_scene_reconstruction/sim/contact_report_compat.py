"""Audit the known Isaac 5.1 unknown-face contact-report metadata limitation.

GPU mesh contacts may have no triangle index. The old report callback asks the
ordinary mesh for that unknown triangle's material and emits a warning; this
does not establish that the solver's single material is missing. Native force,
separation, motion and all other errors still require independent validation.
"""
import importlib.metadata

import numpy as np


SOURCE = 'https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/ovruntime/source/omni.physx/plugins/PhysXScene.cpp'


def validate(report):
    if (report.get('schema') != 'astra_contact_report_metadata_compat_v1'
            or report.get('isaac_version') != '5.1.0.0'
            or report.get('physx_version') != '107.3.26'
            or report.get('upstream_source') != SOURCE
            or not report.get('colliders')):
        raise ValueError('Only the inspected native report-metadata implementation is supported')
    for row in report['colliders']:
        if row.get('geometry_subsets') or not row.get('path'):
            raise ValueError('Per-face material assignments cannot use unknown-face compatibility')
        material = row.get('physics_material')
        if material:
            values = np.asarray(row.get('coefficients'), float)
            if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or values[2] > 1:
                raise ValueError('Valid authored single-material coefficients required')
        elif row.get('native_default_coefficients') != 'unknown_not_changed':
            raise ValueError('Unbound/default material uncertainty must be explicit')
    return report


def audit(stage):
    """Read source schema/materials without assigning or changing any material."""
    import omni.kit.app
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    manager = omni.kit.app.get_app().get_extension_manager()
    extension = manager.get_enabled_extension_id('omni.physx')
    metadata = manager.get_extension_dict(extension)
    report = {'schema': 'astra_contact_report_metadata_compat_v1',
              'isaac_version': importlib.metadata.version('isaacsim'),
              'physx_version': metadata['package']['version'], 'upstream_source': SOURCE,
              'colliders': [], 'changes_materials_or_collision': False,
              'limitation': 'Per-contact face-index/material metadata unavailable; force/separation and solver validity are checked independently.'}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        physical = bool(mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
        row = {'path': str(prim.GetPath()), 'geometry_type': prim.GetTypeName(),
               'geometry_subsets': [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Subset)],
               'physics_material': str(mat.GetPath()) if physical else None,
               'resolved_binding': str(relation.GetPath()) if relation else None}
        if physical:
            api = UsdPhysics.MaterialAPI(mat.GetPrim())
            row['coefficients'] = [api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get()]
        else:
            row['native_default_coefficients'] = 'unknown_not_changed'
        report['colliders'].append(row)
    return validate(report)
