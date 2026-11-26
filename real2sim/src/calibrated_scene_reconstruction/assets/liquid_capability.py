"""Liquid contributions to declarations, priors and isolated probes."""
from .continuum_components import validate, task_objects, result_failures


def extra_objects(contract):
    return task_objects(contract) - set(contract['objects'])


def priors(plan, mechanisms):
    from .physical_priors import parameter
    from .continuum_components import validate as continuum_components
    fluids=continuum_components(mechanisms)
    declared=plan.get('continuum_components',{})
    if not isinstance(declared,dict) or set(declared)!=set(fluids):
        raise ValueError('Physical priors must cover every declared native fluid component')
    for name,body in declared.items():
        params=body.get('parameters',{})
        if not body.get('material_hypothesis') or not {'density','dynamic_viscosity','surface_tension'}<=set(params):
            raise ValueError(name+': liquid needs density, viscosity and surface-tension priors')
        for key,value in params.items():parameter(key,value)


def probe(declaration, scoped, refs):
    # Only keep fluids whose complete physical dependency set is present.
    fluids = {}
    for name, fluid in declaration.get('continuum_components', {}).items():
        dependencies = {fluid['container_link']} if fluid.get('container_link') else set()
        if fluid.get('absorption'):
            dependencies.add(fluid['absorption']['absorber_link'])
        if dependencies and dependencies <= refs and not fluid.get('task_object'):
            fluids[name] = fluid
    declaration['continuum_components'] = fluids
    scoped['continuum_components'] = {k: v for k, v in scoped.get('continuum_components', {}).items() if k in fluids}
    return fluids


def receipt(full, declaration):
    return {'omitted_continuum_components': sorted(set(full.get('continuum_components', {})) - declaration.get('continuum_components', {}).keys())}


def disclose(scope):
    if scope['omitted_continuum_components']:
        scope['continuum_omission_disclosed'] = True
