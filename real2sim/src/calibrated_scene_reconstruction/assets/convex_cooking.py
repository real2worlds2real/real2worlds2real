"""Pure validation of authored convex-collision settings."""
import math
_FIELDS={"hull_vertex_limit":("HullVertexLimit",8,64),"max_convex_hulls":("MaxConvexHulls",1,1024),"voxel_resolution":("VoxelResolution",10000,10000000)}

def validate_hull(settings):
    if (set(settings) != {'hull_vertex_limit', 'minimum_thickness_m'}
            or type(settings['hull_vertex_limit']) is not int
            or not 8 <= settings['hull_vertex_limit'] <= 64
            or type(settings['minimum_thickness_m']) not in (int, float)
            or not math.isfinite(settings['minimum_thickness_m'])
            or not 1e-7 <= settings['minimum_thickness_m'] <= 1.):
        raise ValueError('Single hull cooking needs explicit vertex and metric minimum-thickness limits')
    return dict(settings)


def validate(settings):
    required=set(_FIELDS)|{'minimum_thickness_m','error_percentage','shrink_wrap'}
    if set(settings)!=required:raise ValueError('Complete explicit convex cooking settings required')
    for key,(_,lo,hi) in _FIELDS.items():
        if type(settings[key]) is not int or not lo<=settings[key]<=hi:
            raise ValueError('Invalid convex cooking integer: '+key)
    for key,lo,hi in [('minimum_thickness_m',1e-7,1.),('error_percentage',.02,25.)]:
        if type(settings[key]) not in (float,int) or not math.isfinite(settings[key]) or not lo<=settings[key]<=hi:
            raise ValueError('Invalid convex cooking scalar: '+key)
    if type(settings['shrink_wrap']) is not bool:raise ValueError('Explicit shrink-wrap boolean required')
    return dict(settings)
