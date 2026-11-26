"""Propose a bounded contact correction from measured native surface error.

This never approves a collider or edits a scene. The measured field, contact
separations and complete task dynamics must all remain source-bound.
"""
import math


def sdf_contact_proposal(report,*,maximum_surface_error_m,maximum_separation_m,contact_offset_m):
    if (not isinstance(report,dict) or not report.get('native_errors',{}).get('pass')
            or not report.get('precision',{}).get('pass') or report.get('samples',0)<=0):
        raise ValueError('Successful native SDF measurement and precision audit required')
    low=report['minimum_signed_distance_m'];high=report['maximum_signed_distance_m']
    values=[low,high,maximum_surface_error_m,maximum_separation_m,contact_offset_m]
    if (not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
            or low>high or min(values[2:])<=0):
        raise ValueError('Finite measured errors and explicit positive budgets required')
    if max(abs(low),abs(high))>maximum_surface_error_m:
        raise ValueError('Native field exceeds its declared shape error budget; improve the representation')
    # Positive native distance on a true source point means the cooked field
    # shrank inside the source. Correct only that measured amount, rounded up.
    rest=math.ceil(max(0.,high)*1e6)/1e6
    separation=rest+max(0.,-low)
    if rest>=contact_offset_m or separation>maximum_separation_m:
        raise ValueError('Correction would exceed the declared separation/contact budget')
    return {'proposal_only':True,'task_approved':False,'rest_offset_m':rest,
            'sampled_single_body_separation_bound_m':separation,
            'maximum_surface_error_m':maximum_surface_error_m,
            'maximum_separation_m':maximum_separation_m,
            'requires':['source-bound native field','actual all-body surface contacts',
                        'actual contact separations','free-release stability','task feasibility'],
            'scope':'Finite measured source samples only. Pair errors may add; actual interactions must be checked.'}
