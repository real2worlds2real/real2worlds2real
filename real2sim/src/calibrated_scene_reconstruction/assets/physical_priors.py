"""Agent physical estimates with explicit provenance and uncertainty.

Task success is a feasibility observation, not evidence identifying real mass,
friction or stiffness. Static RGB alone generally cannot identify those values.
"""
import copy

import numpy as np

UNITS = {'mass': 'kg', 'density': 'kg/m3', 'areal_density': 'kg/m2', 'static_friction': '1', 'dynamic_friction': '1',
         'restitution': '1', 'linear_damping': '1/s', 'angular_damping': '1/s',
         'joint_friction': 'N*m', 'joint_damping': 'N*m*s/rad',
         'youngs_modulus': 'Pa', 'poissons_ratio': '1', 'damping_scale': '1',
         'stretch_stiffness': 'N/m', 'bend_stiffness': 'N/m', 'shear_stiffness': 'N/m',
         'spring_damping': 'kg/s','thickness':'m','flexural_rigidity':'N*m','elasticity_damping':'1/s','bend_damping':'1/s',
         }
POSITIVE = {'mass', 'density', 'areal_density', 'youngs_modulus', 'stretch_stiffness', 'bend_stiffness', 'shear_stiffness','thickness','flexural_rigidity'}


def parameter(name, value):
    from calibrated_scene_reconstruction.core.capabilities import entries
    supported = {**UNITS, **entries('physical_units')}
    units = {supported[name]} if name in supported else set()
    if name == 'joint_damping':
        units.add('N*s/m')
    if name == 'joint_friction':
        units.add('N')
    if name == 'elasticity_damping':
        # Legacy plans mislabeled this native rate as dimensionless. Preserve
        # their numeric behavior and audit the label; new authoring uses1/s.
        units.add('1')
    if name not in supported or not isinstance(value, dict) or value.get('unit') not in units:
        raise ValueError(f'{name}: supported physical parameter and explicit SI unit required')
    bounds = np.asarray(value.get('bounds', []), float)
    nominal = value.get('nominal')
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or not isinstance(nominal, (int, float))
            or isinstance(nominal, bool) or not np.isfinite(nominal)
            or not bounds[0] <= nominal <= bounds[1] or bounds[0] >= bounds[1]):
        raise ValueError(f'{name}: finite prior interval containing nominal required')
    if name == 'poissons_ratio':
        if not -1 < bounds[0] < bounds[1] < .5:
            raise ValueError('Poisson ratio prior must lie inside (-1,.5)')
    elif bounds[0] < 0 or (name in POSITIVE and bounds[0] <= 0):
        raise ValueError(f'{name}: invalid nonpositive physical prior')
    if name == 'restitution' and bounds[1] > 1:
        raise ValueError('Restitution cannot exceed one')
    if value.get('basis') not in {'material_class', 'observed_motion', 'datasheet', 'assumption'} or not value.get('source'):
        raise ValueError(f'{name}: prior basis and evidence source required')
    if value.get('identification') not in {'prior_only', 'motion_constrained'}:
        raise ValueError(f'{name}: distinguish a prior estimate from motion identification')
    if value['identification'] == 'motion_constrained' and value['basis'] != 'observed_motion':
        raise ValueError('Task pass or static images cannot identify material parameters')
    if value['identification'] == 'motion_constrained' and not value.get('observation_evidence'):
        raise ValueError('Motion-constrained estimates require immutable observational evidence')
    return value


def expand_families(plan):
    """Share an explicitly uncertain material across repeated physical parts."""
    result=copy.deepcopy(plan)
    families=result.get('families',{})
    if not isinstance(families,dict):raise ValueError('Physical material families must be a mapping')
    for name,body in result.get('bodies',{}).items():
        if 'family' not in body:continue
        family=families.get(body['family'])
        if not isinstance(family,dict) or 'family' in family:
            raise ValueError('Unknown or nested physical material family: '+str(body['family']))
        merged={**family,**body}
        merged['parameters']={**family.get('parameters',{}),**body.get('parameters',{})}
        result['bodies'][name]=copy.deepcopy(merged)
    return result


def validate(plan, mechanisms):
    if not isinstance(plan, dict) or plan.get('version') != 1:
        raise ValueError('Physical prior version must be 1')
    _supported_groups(plan)
    plan=expand_families(plan)
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    for name in entries('prior_validators'):
        invoke('prior_validators', name, plan, mechanisms)
    expected = {f'{obj}/{link}': spec for obj, body in mechanisms['objects'].items()
                for link, spec in body['links'].items()}
    bodies = plan.get('bodies', {})
    if set(bodies) != set(expected):
        raise ValueError('Agent physical priors must cover every task body, including ordinary rigid bodies')
    for ref, body in bodies.items():
        if not body.get('material_hypothesis'):
            raise ValueError(f'{ref}: material/construction hypothesis required')
        params = body.get('parameters', {})
        for name, value in params.items():
            parameter(name, value)
        required = {'dynamic_friction'}
        if 'areal_density' in params and expected[ref]['kind'] != 'cloth':
            raise ValueError('Areal density is a sheet property, not rigid/FEM volumetric density')
        if expected[ref]['kind'] == 'rigid':
            if not {'mass', 'density'}.intersection(params):
                raise ValueError(f'{ref}: mass or density prior required')
            if {'mass', 'density'}.issubset(params):
                raise ValueError('Choose mass or density as the prior; derive the other from generated geometry')
            required |= {'linear_damping', 'angular_damping', 'static_friction', 'restitution'}
        elif expected[ref]['kind'] in {'deformable', 'rope'}:
            required |= {'density', 'youngs_modulus', 'poissons_ratio'}
            if expected[ref].get('backend','legacy')=='volume_fem':
                required.add('linear_damping')
                if {'damping_scale','elasticity_damping'}&params.keys():
                    raise ValueError('Volume FEM cannot reinterpret legacy damping or unqualified material rates')
            else:required.add('damping_scale')
        else:
            if len({'mass', 'areal_density'}.intersection(params)) != 1 or 'density' in params:
                raise ValueError(f'{ref}: choose cloth mass or areal density, not volumetric density or both')
            if mechanisms.get('material_state_contract',0):
                validity=body.get('validity_hypothesis',{})
                limits=[validity.get(k) for k in ('max_edge_strain','max_area_change')] if isinstance(validity,dict) else []
                if (not isinstance(validity,dict) or not isinstance(validity.get('basis'),str)
                        or not validity['basis'].strip() or any(type(v) not in (int,float)
                            or not np.isfinite(v) or v<=0 for v in limits)):
                    raise ValueError(f'{ref}: material-state contract requires finite positive strain/area validity hypotheses with a basis')
            spring={'stretch_stiffness','bend_stiffness','shear_stiffness','spring_damping'}
            continuum={'youngs_modulus','poissons_ratio','thickness','flexural_rigidity','elasticity_damping','linear_damping'}
            if expected[ref].get('backend','particle')=='surface_fem':
                required|=continuum
                if spring&params.keys():raise ValueError('Surface FEM cannot reinterpret particle spring constants')
            else:
                required|=spring
                # Physical sheet thickness also governs geometry/contact
                # validity for particles. It is not a continuum stiffness.
                if ((continuum-{'thickness'})|{'bend_damping'})&params.keys():raise ValueError('Particle cloth cannot reinterpret continuum surface constants')
        if not required.issubset(params):
            raise ValueError(f'{ref}: missing physical priors {sorted(required-set(params))}')
        if 'static_friction' in params and params['dynamic_friction']['nominal'] > params['static_friction']['nominal']:
            raise ValueError('Nominal dynamic friction must not exceed static friction')
    native = {m['id']: m for m in mechanisms['mechanisms'] if m['type'] in {'revolute', 'prismatic'}}
    for name, mechanism in native.items():
        if mechanism['role'] == 'secondary' and name not in plan.get('mechanisms', {}):
            continue
        params = plan.get('mechanisms', {}).get(name, {}).get('parameters', {})
        if not {'joint_friction', 'joint_damping'}.issubset(params):
            raise ValueError(f'{name}: joint friction and damping priors required')
        for key, value in params.items():
            parameter(key, value)
        expected_unit = 'N*m*s/rad' if mechanism['type'] == 'revolute' else 'N*s/m'
        if params['joint_damping']['unit'] != expected_unit:
            raise ValueError('Joint damping unit must match revolute/prismatic motion')
        expected_effort = 'N*m' if mechanism['type'] == 'revolute' else 'N'
        if params['joint_friction']['unit'] != expected_effort:
            raise ValueError('Joint dry-friction effort unit must match revolute/prismatic motion')
    for mechanism in mechanisms['mechanisms']:
        if mechanism['type'] == 'contact' and mechanism['role'] != 'secondary':
            contact = plan.get('mechanisms', {}).get(mechanism['id'], {})
            rigid = all(expected[r]['kind'] == 'rigid' for r in mechanism['links'])
            if contact.get('friction_combine') != ('average' if rigid else 'native') or not contact.get('contact_hypothesis'):
                raise ValueError('Contact mechanisms need an explicit friction-pair/construction hypothesis')
    for name, value in plan.get('supports', {}).items():
        if not {'static_friction', 'dynamic_friction', 'restitution'}.issubset(value.get('parameters', {})):
            raise ValueError(f'{name}: support contact friction and restitution priors required')
        for key, p in value.get('parameters', {}).items():
            parameter(key, p)
    return plan


def sample(plan, case='nominal'):
    if case not in {'nominal', 'low', 'high'}:
        raise ValueError('Select nominal/low/high prior sample')
    _supported_groups(plan)
    result = expand_families(plan);corrections=[]
    for group in ('bodies', 'mechanisms', 'supports', *prior_groups()):
        for ref,body in result.get(group, {}).items():
            for name, value in body.get('parameters', {}).items():
                parameter(name, value)
                if name=='elasticity_damping' and value['unit']=='1':
                    corrections.append({'parameter':group+'/'+ref+'/'+name,'authored_unit':'1',
                        'native_unit':'1/s','numeric_value_changed':False})
                body['parameters'][name] = value['nominal'] if case == 'nominal' else value['bounds'][1 if case == 'high' else 0]
            p = body.get('parameters', {})
            if p.get('dynamic_friction', 0) > p.get('static_friction', float('inf')):
                raise ValueError('Sampled dynamic friction exceeds static friction; specify coherent priors')
    if corrections:result['legacy_unit_corrections']=corrections
    return result


def probe_scope(plan, mechanisms, objects, *, supports=()):
    """Select an isolated probe without weakening the full-scene contract.

    Validate the complete declaration first. Keep every link of a selected
    object and every internal mechanism; explicitly report omitted cross-object
    interactions. The result is a probe input, never a scene acceptance result.
    """
    full = validate(plan, mechanisms)
    names = set(objects)
    if not names or names - mechanisms['objects'].keys():
        raise ValueError('Probe scope must name existing objects')
    support_names = set(supports)
    if support_names - full.get('supports', {}).keys():
        raise ValueError('Probe scope names an undeclared support')
    declaration = copy.deepcopy(mechanisms)
    declaration['objects'] = {k: v for k, v in declaration['objects'].items() if k in names}
    refs = {f'{obj}/{link}' for obj, body in declaration['objects'].items() for link in body['links']}
    declaration['mechanisms'] = [m for m in declaration['mechanisms'] if set(m['links']) <= refs]
    ids = {m['id'] for m in declaration['mechanisms']}
    scoped = copy.deepcopy(full)
    scoped['bodies'] = {k: v for k, v in scoped['bodies'].items() if k in refs}
    scoped['mechanisms'] = {k: v for k, v in scoped.get('mechanisms', {}).items() if k in ids}
    scoped['supports'] = {k: v for k, v in scoped.get('supports', {}).items() if k in support_names}
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    for name in entries('prior_scope'):
        invoke('prior_scope', name, declaration, scoped, refs)
    validate(scoped, declaration)
    receipt = {'diagnostic_only': True, 'task_approved': False, 'objects': sorted(names),
               'bodies': sorted(refs), 'supports': sorted(support_names),
               'omitted_mechanisms': [m['id'] for m in mechanisms['mechanisms'] if m['id'] not in ids]}
    for name in entries('prior_receipts'):
        receipt.update(invoke('prior_receipts', name, full, declaration))
    return scoped, declaration, receipt


def prior_groups():
    from calibrated_scene_reconstruction.core.capabilities import entries
    return tuple(entries('prior_groups'))


def _supported_groups(plan):
    from calibrated_scene_reconstruction.core.capabilities import require
    base = {'version', 'bodies', 'mechanisms', 'supports', 'families', 'legacy_unit_corrections'}
    for key in plan.keys() - base:
        require('prior_groups', key)
