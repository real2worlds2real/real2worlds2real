"""Explicit native contact-control profiles, independent of scene identity."""
import numpy as np

from calibrated_scene_reconstruction.assets.surface_contact_profile import sample


def validate(profile):
    if (not isinstance(profile,dict) or set(profile)!={'version','physics_hz','spacing_m','settings'}
            or profile['version']!=1 or type(profile['physics_hz']) is not int
            or not 240<=profile['physics_hz']<=30720 or profile['physics_hz']%40):
        raise ValueError('Explicit version1 native contact profile and supported recording clock required')
    spacing=np.asarray(profile['spacing_m'],float)
    if (spacing.shape!=(2,) or not np.isfinite(spacing).all()
            or not (spacing>=1e-5).all() or not (spacing<=.01).all()):
        raise ValueError('Two finite material-grid spacings in metres required')
    if np.prod(np.ceil(.08/spacing)+1)>250000:
        raise ValueError('Contact coupon exceeds its declared250000-node construction budget')
    settings=profile['settings']
    required={'youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density',
              'elasticity_damping','linear_damping','friction','contact_offset','rest_offset',
              'self_collision_filter_distance','solver_position_iterations','collision_pair_updates',
              'collision_iteration_multiplier','max_depenetration_velocity','max_linear_velocity'}
    if not isinstance(settings,dict) or set(settings)!=required:
        raise ValueError('Complete explicit surface-FEM material and contact settings required')
    if not np.isfinite(list(settings.values())).all():
        raise ValueError('Finite native material and numerical settings required')
    if (min(settings[k] for k in ['youngs_modulus','thickness','flexural_rigidity','areal_density'])<=0
            or not -1<settings['poissons_ratio']<.5
            or min(settings[k] for k in ['elasticity_damping','linear_damping','friction'])<0
            or settings['friction']>1.5 or not 0<settings['max_depenetration_velocity']<=10
            or not (settings['max_linear_velocity']==-1 or settings['max_linear_velocity']>0)):
        raise ValueError('Physical positive material parameters and bounded numerical settings required')
    iterations=settings['solver_position_iterations']
    if (type(iterations) is not int or not 4<=iterations<=255
            or type(settings['collision_pair_updates']) is not int
            or not 1<=settings['collision_pair_updates']<=iterations
            or type(settings['collision_iteration_multiplier']) is not int
            or not 1<=settings['collision_iteration_multiplier']<=iterations//2):
        raise ValueError('Native contact iteration settings must be explicit bounded integers')
    sample({'thickness':settings['thickness'],'contact_offset':settings['contact_offset'],
            'rest_offset':settings['rest_offset'],'self_filter_distance':settings['self_collision_filter_distance']},
           settings['thickness'])
    return profile


def edge_range(points,faces):
    p=np.asarray(points,float);f=np.asarray(faces)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu'
            or f.min()<0 or f.max()>=len(p)):
        raise ValueError('Finite material points and indexed triangles required')
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    length=np.linalg.norm(p[edges[:,1]]-p[edges[:,0]],axis=1)
    if length.min()<=0:raise ValueError('Nonzero material edges required')
    return [float(length.min()),float(length.max())]


def resolution_matches(control,actual):
    """Bind both fine and coarse edges; a scalar spacing cannot cover anisotropy."""
    control,actual=np.asarray(control,float),np.asarray(actual,float)
    if (control.shape!=(2,) or actual.shape!=(2,) or not np.isfinite([control,actual]).all()
            or min(control.min(),actual.min())<=0 or control[1]<control[0] or actual[1]<actual[0]):
        raise ValueError('Positive ordered control and actual material-edge ranges required')
    ratio=actual/control
    return bool(((ratio>=.9)&(ratio<=1.1)).all())
