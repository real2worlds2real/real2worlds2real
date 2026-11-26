"""Report which declared cloth descriptions each native authoring path uses.

Authoring an attribute is not evidence that its physical response is correct.
"""


def usage(rest,backend):
    if backend not in {'particle','surface_fem'}:
        raise ValueError('Explicit supported native cloth backend required')
    rest=rest or {};features={}
    for name,owner in [('bend_memory','particle'),('bend_reference','surface_fem'),('bend_angles','surface_fem')]:
        if name not in rest:continue
        features[name]={'authored_to_native':backend==owner,'backend':owner,
                        'source':rest[name].get('source'),
                        'representation':('independent signed rest dihedrals' if name=='bend_angles' else
                                          'distance bending rest lengths' if owner=='particle' else 'isometric angular rest reference')}
        if name=='bend_angles':
            features[name].update(authored_to_native=False,delivery_supported=False,
                reason='Matched native positive/reverse angle controls were unresponsive; only explicit diagnostic authoring is allowed')
    return {'backend':backend,'features':features,
            'unapplied_descriptions':[name for name,value in features.items() if not value['authored_to_native']],
            'cross_backend_equivalence_established':False,'runtime_response_validated':False,
            'scope':'Authoring-path audit only. Alternative bending descriptions are not interchangeable; native feature controls, source response, material/contact and task checks remain required.'}
