"""Explicit scene simulation inputs shared by checking and delivery."""
import json
from calibrated_scene_reconstruction.core.session_code import source_path


def default_solver(session):
    """Native particle-cloth impulse/contact controls select PGS; rigid stays TGS.

    This selects a numerical backend, not task approval or a material prior.
    Full mechanism validation remains the responsibility of its declaration.
    """
    path = source_path(session.config_dir, 'mechanisms.json')
    mechanism = json.loads(path.read_text()) if path.is_file() else {}
    cloth = any(link.get('kind') == 'cloth' for obj in mechanism.get('objects', {}).values()
                for link in obj.get('links', {}).values())
    return 'PGS' if cloth else 'TGS'

def settings(session):
    if not hasattr(session, 'config_dir'):
        return None
    missing = object()
    value = session.config('simulation.json', missing)
    if value is missing or value is None:
        return None
    if (not isinstance(value, dict) or set(value)-{'physics_hz', 'friction_type', 'basis', 'solver', 'rigid_iterations', 'gpu_dynamics', 'initialization_hz', 'gpu_max_deformable_surface_contacts','gpu_found_lost_aggregate_pairs_capacity', 'external_forces_every_iteration'}
            or not isinstance(value.get('basis'), str) or not value['basis'].strip()):
        raise ValueError('simulation.json needs declared physics_hz, friction_type and basis')
    hz = value.get('physics_hz')
    if type(hz) is not int or not 60 <= hz <= 7680:
        raise ValueError('Simulation physics_hz must be an integer60..7680')
    initialization=value.get('initialization_hz')
    if initialization is not None and (type(initialization) is not int or not hz<=initialization<=20000):
        raise ValueError('Optional initialization_hz must be an integer between physics_hz and20000')
    if value.get('friction_type') not in {'patch', 'oneDirectional', 'twoDirectional'}:
        raise ValueError('Simulation friction_type must be a native PhysX friction model')
    solver = value.get('solver', default_solver(session))
    if solver not in {'PGS', 'TGS'}:
        raise ValueError('Simulation solver must be PGS or TGS')
    iterations=value.get('rigid_iterations')
    if iterations is not None and (not isinstance(iterations,list) or len(iterations)!=2
            or any(type(n) is not int or not 1<=n<=256 for n in iterations)):
        raise ValueError('Rigid position/velocity iteration counts must be two integers1..256')
    if 'gpu_dynamics' in value and type(value['gpu_dynamics']) is not bool:
        raise ValueError('gpu_dynamics must be an explicit boolean')
    if 'external_forces_every_iteration' in value:
        if type(value['external_forces_every_iteration']) is not bool:
            raise ValueError('external_forces_every_iteration must be an explicit boolean')
        if value['external_forces_every_iteration'] and solver != 'TGS':
            raise ValueError('external_forces_every_iteration=true requires TGS')
    capacity=value.get('gpu_max_deformable_surface_contacts')
    if capacity is not None and (type(capacity) is not int or not 1024<=capacity<=16777216):
        raise ValueError('gpu_max_deformable_surface_contacts must be an integer1024..16777216')
    capacity=value.get('gpu_found_lost_aggregate_pairs_capacity')
    if capacity is not None and (type(capacity) is not int or not 1024<=capacity<=16777216):
        raise ValueError('gpu_found_lost_aggregate_pairs_capacity must be an integer1024..16777216')
    return {**value, 'solver': solver}
