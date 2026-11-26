"""Explicit coordinate/material conversion for native simulation proposals.

Source assets and priors remain SI. This constructs a separate native-unit
representation; it does not authorize a backend or change any physical value.
"""
import copy
import numpy as np


def factor(unit_length_m):
    if isinstance(unit_length_m,bool) or not np.isfinite(unit_length_m) or not .001<=unit_length_m<=1:
        raise ValueError('Native unit length must be explicitly bounded in metres')
    return 1/float(unit_length_m)


def kit(source,unit_length_m):
    scale=factor(unit_length_m);result=copy.deepcopy(source)
    for part in result['parts']:
        part['V']=np.asarray(part['V'])*scale
        rest=part.get('cloth_rest')
        if rest:
            rest['points']=(np.asarray(rest['points'])*scale).tolist()
            if rest.get('bend_reference'):rest['bend_reference']['points']=(np.asarray(rest['bend_reference']['points'])*scale).tolist()
            if rest.get('bend_memory'):rest['bend_memory']['lengths']=(np.asarray(rest['bend_memory']['lengths'])*scale).tolist()
    for joint in result.get('joints',[]):
        for key in ('anchor0','anchor1'):
            if key in joint:joint[key]=(np.asarray(joint[key])*scale).tolist()
        if joint['type']=='prismatic':joint['limits']=(np.asarray(joint['limits'])*scale).tolist()
    for key in ('height','half'):
        if key in result:result[key]=np.asarray(result[key])*scale
    return result


def settings(source,unit_length_m):
    scale=factor(unit_length_m);result=copy.deepcopy(source)
    lengths={'thickness','contact_offset','rest_offset','solid_rest_offset','particle_contact_offset',
             'self_collision_filter_distance','max_depenetration_velocity','max_linear_velocity'}
    for name,value in result.items():
        if not isinstance(value,dict):continue
        for key in lengths:
            if key in value:value[key]*=scale
        for key,power in [('density',-3),('areal_density',-2),('youngs_modulus',-1),('flexural_rigidity',2),('pressure',-1)]:
            if key in value:value[key]*=scale**power
        if isinstance(value.get('sdf'),dict):
            for key in ('contact_offset','rest_offset'):
                if key in value['sdf']:value['sdf'][key]*=scale
        if isinstance(value.get('collision_slices'),dict):
            value['collision_slices']['max_span_m']*=scale
    return result
