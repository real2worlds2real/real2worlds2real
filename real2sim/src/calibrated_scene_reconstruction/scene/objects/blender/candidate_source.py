"""Conservative source labels for native candidate diagnostics, never approval."""
from .raster_contract import verify


def round_status(result, manifest, manifest_sha256):
    raw = bool(result.get('ok') and result.get('metrics', {}).get('ok'))
    errors = []
    try:
        verify((manifest or {}).get('source_raster_contract'))
        raster_current = True
    except ValueError as error:
        raster_current = False
        errors.append(str(error))
    manifest_current = bool(manifest_sha256 and result.get('input_sha256') == manifest_sha256)
    if not manifest_current:
        errors.append('Input manifest is missing or differs from the measured round')
    return {'raw_source_pass': raw, 'raster_contract_current': raster_current,
            'input_manifest_current': manifest_current,
            'source_geometry_pass': raw and raster_current and manifest_current,
            'object_approved': False, 'reasons': errors,
            'scope': 'Raw source metrics and current sampler contract only; complete OBJ/scene approval remains separate.'}
