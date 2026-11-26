"""Keep collision rest thickness coherent across surface-material prior cases."""
import numpy as np


def sample(reference,thickness):
    h0=reference['thickness'];contact=reference['contact_offset'];filtering=reference['self_filter_distance']
    rest0=reference.get('rest_offset',h0/2)
    if (not np.isfinite([h0,contact,filtering,thickness,rest0]).all() or min(h0,thickness)<=0
            or rest0<h0/2*(1-1e-6) or contact<=rest0 or filtering<2*rest0*(1-1e-6)):
        raise ValueError('Finite coherent surface thickness and numerical contact reference required')
    # Preserve the authored numerical margin independently of uncertain
    # material thickness. Never derive it from a previously sampled case.
    padding=max(0.,rest0-h0/2);rest=thickness/2+padding
    return {'contact_offset':float(max(contact,1.5*rest)),'rest_offset':float(rest),
            'self_filter_distance':float(max(2*rest,filtering*thickness/h0))}


def audit(thickness,expected,material_thickness,actual):
    """Verify material thickness and declared numerical margins separately."""
    sample({'thickness':thickness,**expected},thickness)
    keys=('contact_offset','rest_offset','self_filter_distance')
    if (not np.isfinite([material_thickness,*[actual[k] for k in keys]]).all()
            or not np.isclose(material_thickness,thickness,rtol=1e-5,atol=1e-12)
            or any(not np.isclose(actual[k],expected[k],rtol=1e-5,atol=1e-12) for k in keys)):
        raise ValueError('Runtime surface material or numerical contacts differ from the sampled profile')
    return {'physical_material_thickness_m':float(material_thickness),
            'numerical_rest_padding_m':float(actual['rest_offset']-thickness/2),
            'contacts':{k:float(actual[k]) for k in keys},
            'scope':'Native material and contact attribute readback; numerical padding is not material thickness.'}


def padding_proposal(reference,clearance,*,padding_budget_m,witness_rest_gap_m,safety_factor=2.):
    """Propose bounded numerical padding from a failed native layer-gap screen.

    No material coefficient, thickness, gate or geometry is changed here.
    Rest-distance visibility, prepared geometry and fresh matching native
    controls are still required. Crossing/zero-gap states need geometry work.
    """
    h=reference['thickness'];sample(reference,h)
    gap=clearance.get('minimum_gap_within_search_m');required=clearance.get('required_clearance_m')
    tolerance=clearance.get('numerical_tolerance_m',1e-6)
    if (gap is None or required is None or not np.isfinite([gap,required,tolerance,padding_budget_m,witness_rest_gap_m,safety_factor]).all()
            or not np.isclose(required,h,rtol=1e-6,atol=1e-12) or padding_budget_m<0
            or tolerance<=0 or safety_factor<1 or witness_rest_gap_m<=0):
        raise ValueError('Matching physical thickness, measured gap, explicit padding budget and rest-pair visibility required')
    if gap<=tolerance or clearance.get('pass') is not False or clearance.get('violating_pairs',0)<=0 or gap>=h:
        return {'proposal_available':False,'reason':'Requires a separated failed native layer-gap witness; no crossing or gate relaxation', 'task_approved':False}
    old_rest=reference.get('rest_offset',h/2)
    rest=old_rest+safety_factor*(h-gap)/2+2*tolerance
    filtering=max(reference['self_filter_distance'],2*rest)
    if rest-h/2>padding_budget_m or filtering+tolerance>=witness_rest_gap_m:
        return {'proposal_available':False,'reason':'Padding exceeds the declared budget or hides witnessed self-contact pairs', 'task_approved':False}
    proposed={**reference,'rest_offset':float(rest),'contact_offset':float(max(reference['contact_offset'],1.5*rest)),
              'self_filter_distance':float(filtering)}
    return {'proposal_available':True,'reference':proposed,'physical_thickness_unchanged':True,
            'measured_gap_deficit_m':float(h-gap),'numerical_padding_m':float(rest-h/2),
            'padding_budget_m':float(padding_budget_m),'task_approved':False,
            'requires_geometry_envelope_check':True,'requires_matching_native_qualification':True,
            'scope':'Numerical candidate only; preserve material, initial support and source geometry, rebuild if needed, then rerun passive/contact/task/export checks.'}
